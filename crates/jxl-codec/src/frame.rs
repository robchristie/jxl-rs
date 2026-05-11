use std::fmt;

use crate::bitstream::{BitReader, U32Distribution, bits_offset, val};
use crate::error::{Error, Result};
use crate::frame_data::compute_group_layout;
use crate::metadata::{ImageMetadata, read_extensions, unpack_signed};

const MAX_NUM_PASSES: usize = 11;
const FLAG_USE_DC_FRAME: u64 = 32;

#[derive(Debug, Clone, PartialEq)]
pub struct FrameHeader {
    pub encoding: FrameEncoding,
    pub frame_type: FrameType,
    pub flags: u64,
    pub color_transform: ColorTransform,
    pub chroma_subsampling: YCbCrChromaSubsampling,
    pub group_size_shift: u32,
    pub x_qm_scale: u32,
    pub b_qm_scale: u32,
    pub passes: Passes,
    pub dc_level: u32,
    pub custom_size_or_origin: bool,
    pub frame_origin: FrameOrigin,
    pub frame_size: FrameSize,
    pub upsampling: u32,
    pub extra_channel_upsampling: Vec<u32>,
    pub blending_info: BlendingInfo,
    pub extra_channel_blending_info: Vec<BlendingInfo>,
    pub animation_frame: AnimationFrame,
    pub is_last: bool,
    pub save_as_reference: u32,
    pub save_before_color_transform: bool,
    pub name: String,
    pub loop_filter: LoopFilter,
    pub extensions: u64,
    pub group_layout: FrameGroupLayout,
}

impl FrameHeader {
    pub fn is_modular(&self) -> bool {
        self.encoding == FrameEncoding::Modular
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameEncoding {
    VarDct,
    Modular,
}

impl fmt::Display for FrameEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::VarDct => "VarDCT",
            Self::Modular => "Modular",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum FrameType {
    Regular = 0,
    Dc = 1,
    ReferenceOnly = 2,
    SkipProgressive = 3,
}

impl TryFrom<u32> for FrameType {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::Regular),
            1 => Ok(Self::Dc),
            2 => Ok(Self::ReferenceOnly),
            3 => Ok(Self::SkipProgressive),
            _ => Err(Error::InvalidCodestream("invalid frame type")),
        }
    }
}

impl fmt::Display for FrameType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Regular => "Regular",
            Self::Dc => "DC",
            Self::ReferenceOnly => "ReferenceOnly",
            Self::SkipProgressive => "SkipProgressive",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ColorTransform {
    Xyb = 0,
    None = 1,
    YCbCr = 2,
}

impl fmt::Display for ColorTransform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Xyb => "XYB",
            Self::None => "None",
            Self::YCbCr => "YCbCr",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameOrigin {
    pub x0: i32,
    pub y0: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameSize {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameGroupLayout {
    pub group_dim: u32,
    pub groups_x: u32,
    pub groups_y: u32,
    pub num_groups: u32,
    pub dc_group_dim: u32,
    pub dc_groups_x: u32,
    pub dc_groups_y: u32,
    pub num_dc_groups: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct YCbCrChromaSubsampling {
    pub channel_mode: [u32; 3],
    pub max_h_shift: u8,
    pub max_v_shift: u8,
}

impl Default for YCbCrChromaSubsampling {
    fn default() -> Self {
        Self::from_modes([0, 0, 0])
    }
}

impl YCbCrChromaSubsampling {
    fn from_modes(channel_mode: [u32; 3]) -> Self {
        const H_SHIFT: [u8; 4] = [0, 1, 1, 0];
        const V_SHIFT: [u8; 4] = [0, 1, 0, 1];
        let mut max_h_shift = 0;
        let mut max_v_shift = 0;
        for mode in channel_mode {
            max_h_shift = max_h_shift.max(H_SHIFT[mode as usize]);
            max_v_shift = max_v_shift.max(V_SHIFT[mode as usize]);
        }
        Self {
            channel_mode,
            max_h_shift,
            max_v_shift,
        }
    }

    pub fn h_shift(&self, channel: usize) -> Option<u8> {
        const H_SHIFT: [u8; 4] = [0, 1, 1, 0];
        let mode = *self.channel_mode.get(channel)? as usize;
        Some(self.max_h_shift.checked_sub(*H_SHIFT.get(mode)?)?)
    }

    pub fn v_shift(&self, channel: usize) -> Option<u8> {
        const V_SHIFT: [u8; 4] = [0, 1, 0, 1];
        let mode = *self.channel_mode.get(channel)? as usize;
        Some(self.max_v_shift.checked_sub(*V_SHIFT.get(mode)?)?)
    }

    pub fn is_444(&self) -> bool {
        (0..3).all(|channel| self.h_shift(channel) == Some(0) && self.v_shift(channel) == Some(0))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Passes {
    pub num_passes: u32,
    pub num_downsample: u32,
    pub downsample: Vec<u32>,
    pub last_pass: Vec<u32>,
    pub shift: Vec<u32>,
}

impl Default for Passes {
    fn default() -> Self {
        Self {
            num_passes: 1,
            num_downsample: 0,
            downsample: Vec::new(),
            last_pass: Vec::new(),
            shift: vec![0],
        }
    }
}

impl Passes {
    pub fn downsampling_bracket(&self, pass: usize) -> Result<(i32, i32)> {
        if pass >= self.num_passes as usize {
            return Err(Error::InvalidCodestream("pass index exceeds pass count"));
        }

        let mut max_shift = 2;
        let mut min_shift = 3;
        for index in 0..=pass {
            for (&last_pass, &downsample) in self.last_pass.iter().zip(&self.downsample) {
                if index as u32 == last_pass {
                    min_shift = match downsample {
                        1 => 0,
                        2 => 1,
                        4 => 2,
                        8 => 3,
                        _ => return Err(Error::InvalidCodestream("invalid pass downsample")),
                    };
                }
            }
            if index as u32 == self.num_passes - 1 {
                min_shift = 0;
            }
            if index == pass {
                return Ok((min_shift, max_shift));
            }
            max_shift = min_shift - 1;
        }

        Err(Error::InvalidCodestream("pass index exceeds pass count"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum BlendMode {
    Replace = 0,
    Add = 1,
    Blend = 2,
    AlphaWeightedAdd = 3,
    Mul = 4,
}

impl TryFrom<u32> for BlendMode {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::Replace),
            1 => Ok(Self::Add),
            2 => Ok(Self::Blend),
            3 => Ok(Self::AlphaWeightedAdd),
            4 => Ok(Self::Mul),
            _ => Err(Error::InvalidCodestream("invalid blend mode")),
        }
    }
}

impl fmt::Display for BlendMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Replace => "Replace",
            Self::Add => "Add",
            Self::Blend => "Blend",
            Self::AlphaWeightedAdd => "AlphaWeightedAdd",
            Self::Mul => "Mul",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlendingInfo {
    pub mode: BlendMode,
    pub alpha_channel: u32,
    pub clamp: bool,
    pub source: u32,
}

impl Default for BlendingInfo {
    fn default() -> Self {
        Self {
            mode: BlendMode::Replace,
            alpha_channel: 0,
            clamp: false,
            source: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AnimationFrame {
    pub duration: u32,
    pub timecode: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoopFilter {
    pub gab: bool,
    pub gab_custom: bool,
    pub gab_weights: Option<[f32; 6]>,
    pub epf_iters: u32,
    pub epf_sharp_custom: bool,
    pub epf_sharp_lut: Option<[f32; 8]>,
    pub epf_weight_custom: bool,
    pub epf_channel_scale: Option<[f32; 3]>,
    pub epf_pass1_zeroflush: Option<f32>,
    pub epf_pass2_zeroflush: Option<f32>,
    pub epf_sigma_custom: bool,
    pub epf_quant_mul: Option<f32>,
    pub epf_pass0_sigma_scale: Option<f32>,
    pub epf_pass2_sigma_scale: Option<f32>,
    pub epf_border_sad_mul: Option<f32>,
    pub epf_sigma_for_modular: Option<f32>,
    pub extensions: u64,
}

impl Default for LoopFilter {
    fn default() -> Self {
        Self {
            gab: true,
            gab_custom: false,
            gab_weights: None,
            epf_iters: 2,
            epf_sharp_custom: false,
            epf_sharp_lut: None,
            epf_weight_custom: false,
            epf_channel_scale: None,
            epf_pass1_zeroflush: None,
            epf_pass2_zeroflush: None,
            epf_sigma_custom: false,
            epf_quant_mul: None,
            epf_pass0_sigma_scale: None,
            epf_pass2_sigma_scale: None,
            epf_border_sad_mul: None,
            epf_sigma_for_modular: None,
            extensions: 0,
        }
    }
}

pub fn read_frame_header(
    reader: &mut BitReader<'_>,
    image_width: u32,
    image_height: u32,
    metadata: &ImageMetadata,
) -> Result<FrameHeader> {
    if reader.read_bool()? {
        return Ok(default_frame_header(image_width, image_height, metadata));
    }

    let frame_type =
        FrameType::try_from(reader.read_u32_selector(val(0), val(1), val(2), val(3))?)?;
    let is_modular = reader.read_bool()?;
    let encoding = if is_modular {
        FrameEncoding::Modular
    } else {
        FrameEncoding::VarDct
    };
    let flags = reader.read_u64()?;

    let color_transform = if metadata.xyb_encoded {
        ColorTransform::Xyb
    } else if reader.read_bool()? {
        ColorTransform::YCbCr
    } else {
        ColorTransform::None
    };

    let chroma_subsampling =
        if color_transform == ColorTransform::YCbCr && flags & FLAG_USE_DC_FRAME == 0 {
            read_chroma_subsampling(reader)?
        } else {
            YCbCrChromaSubsampling::default()
        };

    let num_extra_channels = metadata.extra_channels.len();
    let mut upsampling = 1;
    let mut extra_channel_upsampling = Vec::new();
    if flags & FLAG_USE_DC_FRAME == 0 {
        upsampling = reader.read_u32_selector(val(1), val(2), val(4), val(8))?;
        extra_channel_upsampling.reserve(num_extra_channels);
        for extra_channel in &metadata.extra_channels {
            let encoded = reader.read_u32_selector(val(1), val(2), val(4), val(8))?;
            let ec_upsampling =
                encoded
                    .checked_shl(extra_channel.dim_shift)
                    .ok_or(Error::InvalidCodestream(
                        "extra channel upsampling overflow",
                    ))?;
            if ec_upsampling < upsampling {
                return Err(Error::InvalidCodestream(
                    "extra channel upsampling is smaller than color upsampling",
                ));
            }
            if ec_upsampling > 8 {
                return Err(Error::InvalidCodestream(
                    "extra channel upsampling exceeds 8",
                ));
            }
            extra_channel_upsampling.push(ec_upsampling);
        }
    }

    let group_size_shift = if encoding == FrameEncoding::Modular {
        reader.read_bits(2)? as u32
    } else {
        1
    };

    let (x_qm_scale, b_qm_scale) =
        if encoding == FrameEncoding::VarDct && color_transform == ColorTransform::Xyb {
            (reader.read_bits(3)? as u32, reader.read_bits(3)? as u32)
        } else {
            (2, 2)
        };

    let passes = if frame_type != FrameType::ReferenceOnly {
        read_passes(reader)?
    } else {
        Passes::default()
    };

    let dc_level = if frame_type == FrameType::Dc {
        reader.read_u32_selector(val(1), val(2), val(3), val(4))?
    } else {
        0
    };

    let mut custom_size_or_origin = false;
    let mut frame_origin = FrameOrigin { x0: 0, y0: 0 };
    let mut frame_size = FrameSize {
        width: image_width,
        height: image_height,
    };
    let mut is_partial_frame = false;
    if frame_type != FrameType::Dc {
        custom_size_or_origin = reader.read_bool()?;
        if custom_size_or_origin {
            if frame_type == FrameType::Regular || frame_type == FrameType::SkipProgressive {
                frame_origin.x0 = read_frame_i32(reader)?;
                frame_origin.y0 = read_frame_i32(reader)?;
            }
            frame_size.width = read_frame_u32(reader)?;
            frame_size.height = read_frame_u32(reader)?;
            if frame_size.width == 0 || frame_size.height == 0 {
                return Err(Error::InvalidCodestream("zero-sized custom frame"));
            }
            if frame_type == FrameType::Regular || frame_type == FrameType::SkipProgressive {
                is_partial_frame = frame_origin.x0 > 0
                    || frame_origin.y0 > 0
                    || (frame_size.width as i32 + frame_origin.x0) < image_width as i32
                    || (frame_size.height as i32 + frame_origin.y0) < image_height as i32;
            }
        }
    }

    let mut blending_info = BlendingInfo::default();
    let mut extra_channel_blending_info = Vec::new();
    let mut animation_frame = AnimationFrame::default();
    let is_last;
    if frame_type == FrameType::Regular || frame_type == FrameType::SkipProgressive {
        blending_info = read_blending_info(reader, num_extra_channels, is_partial_frame)?;
        extra_channel_blending_info.reserve(num_extra_channels);
        for _ in 0..num_extra_channels {
            extra_channel_blending_info.push(read_blending_info(
                reader,
                num_extra_channels,
                is_partial_frame,
            )?);
        }
        if metadata.animation.is_some() {
            animation_frame = read_animation_frame(reader, metadata)?;
        }
        is_last = reader.read_bool()?;
    } else {
        is_last = false;
    }

    let save_as_reference = if frame_type != FrameType::Dc && !is_last {
        reader.read_u32_selector(val(0), val(1), val(2), val(3))?
    } else {
        0
    };

    let can_be_referenced = !is_last
        && frame_type != FrameType::Dc
        && (animation_frame.duration == 0 || save_as_reference != 0);
    let save_before_color_transform = if frame_type == FrameType::Dc {
        true
    } else if frame_type == FrameType::ReferenceOnly
        || (can_be_referenced
            && blending_info.mode == BlendMode::Replace
            && !is_partial_frame
            && (frame_type == FrameType::Regular || frame_type == FrameType::SkipProgressive))
    {
        reader.read_bool()?
    } else {
        false
    };

    let name = reader.read_name()?;
    let loop_filter = read_loop_filter(reader, is_modular)?;
    let extensions = read_extensions(reader)?;
    let group_layout = compute_group_layout(
        frame_size,
        dc_level,
        group_size_shift,
        upsampling,
        chroma_subsampling.max_h_shift,
        chroma_subsampling.max_v_shift,
        encoding == FrameEncoding::Modular,
    );

    Ok(FrameHeader {
        encoding,
        frame_type,
        flags,
        color_transform,
        chroma_subsampling,
        group_size_shift,
        x_qm_scale,
        b_qm_scale,
        passes,
        dc_level,
        custom_size_or_origin,
        frame_origin,
        frame_size,
        upsampling,
        extra_channel_upsampling,
        blending_info,
        extra_channel_blending_info,
        animation_frame,
        is_last,
        save_as_reference,
        save_before_color_transform,
        name,
        loop_filter,
        extensions,
        group_layout,
    })
}

fn default_frame_header(
    image_width: u32,
    image_height: u32,
    metadata: &ImageMetadata,
) -> FrameHeader {
    let frame_size = FrameSize {
        width: image_width,
        height: image_height,
    };
    FrameHeader {
        encoding: FrameEncoding::VarDct,
        frame_type: FrameType::Regular,
        flags: 0,
        color_transform: if metadata.xyb_encoded {
            ColorTransform::Xyb
        } else {
            ColorTransform::None
        },
        chroma_subsampling: YCbCrChromaSubsampling::default(),
        group_size_shift: 1,
        x_qm_scale: if metadata.xyb_encoded { 3 } else { 2 },
        b_qm_scale: 2,
        passes: Passes::default(),
        dc_level: 0,
        custom_size_or_origin: false,
        frame_origin: FrameOrigin { x0: 0, y0: 0 },
        frame_size,
        upsampling: 1,
        extra_channel_upsampling: metadata
            .extra_channels
            .iter()
            .map(|channel| 1u32 << channel.dim_shift)
            .collect(),
        blending_info: BlendingInfo::default(),
        extra_channel_blending_info: vec![BlendingInfo::default(); metadata.extra_channels.len()],
        animation_frame: AnimationFrame::default(),
        is_last: true,
        save_as_reference: 0,
        save_before_color_transform: false,
        name: String::new(),
        loop_filter: LoopFilter::default(),
        extensions: 0,
        group_layout: compute_group_layout(frame_size, 0, 1, 1, 0, 0, false),
    }
}

fn read_chroma_subsampling(reader: &mut BitReader<'_>) -> Result<YCbCrChromaSubsampling> {
    let mut modes = [0; 3];
    for mode in &mut modes {
        *mode = reader.read_bits(2)? as u32;
    }
    Ok(YCbCrChromaSubsampling::from_modes(modes))
}

fn read_passes(reader: &mut BitReader<'_>) -> Result<Passes> {
    let num_passes = reader.read_u32_selector(val(1), val(2), val(3), bits_offset(3, 4))?;
    if num_passes as usize > MAX_NUM_PASSES {
        return Err(Error::InvalidCodestream("too many frame passes"));
    }

    let mut passes = Passes {
        num_passes,
        num_downsample: 0,
        downsample: Vec::new(),
        last_pass: Vec::new(),
        shift: vec![0; num_passes as usize],
    };

    if num_passes != 1 {
        passes.num_downsample =
            reader.read_u32_selector(val(0), val(1), val(2), bits_offset(1, 3))?;
        if passes.num_downsample > num_passes {
            return Err(Error::InvalidCodestream(
                "more downsample entries than passes",
            ));
        }

        for index in 0..num_passes - 1 {
            passes.shift[index as usize] = reader.read_bits(2)? as u32;
        }

        for _ in 0..passes.num_downsample {
            let downsample = reader.read_u32_selector(val(1), val(2), val(4), val(8))?;
            if let Some(previous) = passes.downsample.last()
                && downsample >= *previous
            {
                return Err(Error::InvalidCodestream(
                    "downsample sequence is not decreasing",
                ));
            }
            passes.downsample.push(downsample);
        }

        for _ in 0..passes.num_downsample {
            let last_pass = reader.read_u32_selector(
                val(0),
                val(1),
                val(2),
                U32Distribution::BitsOffset { bits: 3, offset: 0 },
            )?;
            if let Some(previous) = passes.last_pass.last()
                && last_pass <= *previous
            {
                return Err(Error::InvalidCodestream(
                    "last_pass sequence is not increasing",
                ));
            }
            if last_pass >= num_passes {
                return Err(Error::InvalidCodestream("last_pass exceeds pass count"));
            }
            passes.last_pass.push(last_pass);
        }
    }

    Ok(passes)
}

fn read_blending_info(
    reader: &mut BitReader<'_>,
    num_extra_channels: usize,
    is_partial_frame: bool,
) -> Result<BlendingInfo> {
    let mode = BlendMode::try_from(reader.read_u32_selector(
        val(0),
        val(1),
        val(2),
        bits_offset(2, 3),
    )?)?;

    let uses_alpha =
        num_extra_channels > 0 && (mode == BlendMode::Blend || mode == BlendMode::AlphaWeightedAdd);
    let alpha_channel = if uses_alpha {
        let alpha_channel = reader.read_u32_selector(val(0), val(1), val(2), bits_offset(3, 3))?;
        if alpha_channel as usize >= num_extra_channels {
            return Err(Error::InvalidCodestream(
                "invalid alpha channel for blending",
            ));
        }
        alpha_channel
    } else {
        0
    };

    let clamp = if uses_alpha || mode == BlendMode::Mul {
        reader.read_bool()?
    } else {
        false
    };
    let source = if mode != BlendMode::Replace || is_partial_frame {
        reader.read_u32_selector(val(0), val(1), val(2), val(3))?
    } else {
        0
    };

    Ok(BlendingInfo {
        mode,
        alpha_channel,
        clamp,
        source,
    })
}

fn read_animation_frame(
    reader: &mut BitReader<'_>,
    metadata: &ImageMetadata,
) -> Result<AnimationFrame> {
    let duration = reader.read_u32_selector(
        val(0),
        val(1),
        U32Distribution::BitsOffset { bits: 8, offset: 0 },
        U32Distribution::BitsOffset {
            bits: 32,
            offset: 0,
        },
    )?;
    let timecode = if metadata
        .animation
        .map(|animation| animation.have_timecodes)
        .unwrap_or(false)
    {
        reader.read_bits(32)? as u32
    } else {
        0
    };
    Ok(AnimationFrame { duration, timecode })
}

fn read_loop_filter(reader: &mut BitReader<'_>, is_modular: bool) -> Result<LoopFilter> {
    if reader.read_bool()? {
        return Ok(LoopFilter::default());
    }

    let mut loop_filter = LoopFilter {
        gab: reader.read_bool()?,
        ..LoopFilter::default()
    };
    if loop_filter.gab {
        loop_filter.gab_custom = reader.read_bool()?;
        if loop_filter.gab_custom {
            let mut weights = [0.0; 6];
            for value in &mut weights {
                *value = reader.read_f16()?;
            }
            if (1.0 + (weights[0] + weights[1]) * 4.0).abs() < 1e-8
                || (1.0 + (weights[2] + weights[3]) * 4.0).abs() < 1e-8
                || (1.0 + (weights[4] + weights[5]) * 4.0).abs() < 1e-8
            {
                return Err(Error::InvalidCodestream("invalid gaborish weights"));
            }
            loop_filter.gab_weights = Some(weights);
        }
    }

    loop_filter.epf_iters = reader.read_bits(2)? as u32;
    if loop_filter.epf_iters > 0 {
        if !is_modular {
            loop_filter.epf_sharp_custom = reader.read_bool()?;
            if loop_filter.epf_sharp_custom {
                let mut lut = [0.0; 8];
                for value in &mut lut {
                    *value = reader.read_f16()?;
                }
                loop_filter.epf_sharp_lut = Some(lut);
            }
        }

        loop_filter.epf_weight_custom = reader.read_bool()?;
        if loop_filter.epf_weight_custom {
            let mut channel_scale = [0.0; 3];
            for value in &mut channel_scale {
                *value = reader.read_f16()?;
            }
            loop_filter.epf_channel_scale = Some(channel_scale);
            loop_filter.epf_pass1_zeroflush = Some(reader.read_f16()?);
            loop_filter.epf_pass2_zeroflush = Some(reader.read_f16()?);
        }

        loop_filter.epf_sigma_custom = reader.read_bool()?;
        if loop_filter.epf_sigma_custom {
            if !is_modular {
                loop_filter.epf_quant_mul = Some(reader.read_f16()?);
            }
            loop_filter.epf_pass0_sigma_scale = Some(reader.read_f16()?);
            loop_filter.epf_pass2_sigma_scale = Some(reader.read_f16()?);
            loop_filter.epf_border_sad_mul = Some(reader.read_f16()?);
        }

        if is_modular {
            let sigma = reader.read_f16()?;
            if sigma < 1e-8 {
                return Err(Error::InvalidCodestream("modular EPF sigma is too small"));
            }
            loop_filter.epf_sigma_for_modular = Some(sigma);
        }
    }

    loop_filter.extensions = read_extensions(reader)?;
    Ok(loop_filter)
}

fn read_frame_i32(reader: &mut BitReader<'_>) -> Result<i32> {
    Ok(unpack_signed(read_frame_u32(reader)?))
}

fn read_frame_u32(reader: &mut BitReader<'_>) -> Result<u32> {
    reader.read_u32_selector(
        U32Distribution::BitsOffset { bits: 8, offset: 0 },
        bits_offset(11, 256),
        bits_offset(14, 2304),
        bits_offset(30, 18_688),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::ImageMetadata;

    #[test]
    fn reads_all_default_frame_header() {
        let metadata = ImageMetadata::default();
        let mut reader = BitReader::new(&[1]);
        let frame = read_frame_header(&mut reader, 320, 240, &metadata).unwrap();

        assert_eq!(frame.encoding, FrameEncoding::VarDct);
        assert_eq!(frame.frame_type, FrameType::Regular);
        assert_eq!(frame.frame_size.width, 320);
        assert_eq!(frame.frame_size.height, 240);
        assert!(frame.is_last);
        assert_eq!(frame.group_layout.group_dim, 256);
        assert_eq!(frame.group_layout.num_groups, 2);
    }

    #[test]
    fn pass_downsampling_brackets_match_reference_progression() {
        let default = Passes::default();
        assert_eq!(default.downsampling_bracket(0).unwrap(), (0, 2));
        assert_eq!(
            default.downsampling_bracket(1),
            Err(Error::InvalidCodestream("pass index exceeds pass count"))
        );

        let progressive = Passes {
            num_passes: 4,
            num_downsample: 2,
            downsample: vec![8, 2],
            last_pass: vec![0, 2],
            shift: vec![0, 0, 0, 0],
        };
        assert_eq!(progressive.downsampling_bracket(0).unwrap(), (3, 2));
        assert_eq!(progressive.downsampling_bracket(1).unwrap(), (3, 2));
        assert_eq!(progressive.downsampling_bracket(2).unwrap(), (1, 2));
        assert_eq!(progressive.downsampling_bracket(3).unwrap(), (0, 0));
    }
}
