use std::fmt;

use crate::bitstream::{BitReader, bits_offset, val};
use crate::codestream::{SizeHeader, read_size_header};
use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct ImageMetadata {
    pub orientation: Orientation,
    pub intrinsic_size: Option<SizeHeader>,
    pub preview_size: Option<PreviewHeader>,
    pub animation: Option<AnimationHeader>,
    pub bit_depth: BitDepth,
    pub modular_16_bit_buffer_sufficient: bool,
    pub extra_channels: Vec<ExtraChannelInfo>,
    pub xyb_encoded: bool,
    pub color_encoding: ColorEncoding,
    pub tone_mapping: ToneMapping,
    pub extensions: u64,
}

impl Default for ImageMetadata {
    fn default() -> Self {
        Self {
            orientation: Orientation::Identity,
            intrinsic_size: None,
            preview_size: None,
            animation: None,
            bit_depth: BitDepth::default(),
            modular_16_bit_buffer_sufficient: true,
            extra_channels: Vec::new(),
            xyb_encoded: true,
            color_encoding: ColorEncoding::default(),
            tone_mapping: ToneMapping::default(),
            extensions: 0,
        }
    }
}

impl ImageMetadata {
    pub fn alpha_channel(&self) -> Option<&ExtraChannelInfo> {
        self.extra_channels
            .iter()
            .find(|channel| channel.channel_type == ExtraChannelType::Alpha)
    }

    pub fn alpha_bits(&self) -> u32 {
        self.alpha_channel()
            .map(|channel| channel.bit_depth.bits_per_sample)
            .unwrap_or(0)
    }

    pub fn alpha_exponent_bits(&self) -> u32 {
        self.alpha_channel()
            .map(|channel| channel.bit_depth.exponent_bits_per_sample)
            .unwrap_or(0)
    }

    pub fn alpha_premultiplied(&self) -> bool {
        self.alpha_channel()
            .map(|channel| channel.alpha_associated)
            .unwrap_or(false)
    }

    pub fn num_color_channels(&self) -> u32 {
        if self.color_encoding.color_space == ColorSpace::Gray {
            1
        } else {
            3
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Orientation {
    Identity = 1,
    FlipHorizontal = 2,
    Rotate180 = 3,
    FlipVertical = 4,
    Transpose = 5,
    Rotate90Cw = 6,
    AntiTranspose = 7,
    Rotate90Ccw = 8,
}

impl TryFrom<u32> for Orientation {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            1 => Ok(Self::Identity),
            2 => Ok(Self::FlipHorizontal),
            3 => Ok(Self::Rotate180),
            4 => Ok(Self::FlipVertical),
            5 => Ok(Self::Transpose),
            6 => Ok(Self::Rotate90Cw),
            7 => Ok(Self::AntiTranspose),
            8 => Ok(Self::Rotate90Ccw),
            _ => Err(Error::InvalidCodestream("invalid orientation")),
        }
    }
}

impl fmt::Display for Orientation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Identity => "Identity",
            Self::FlipHorizontal => "Flip horizontal",
            Self::Rotate180 => "Rotate 180",
            Self::FlipVertical => "Flip vertical",
            Self::Transpose => "Transpose",
            Self::Rotate90Cw => "Rotate 90 CW",
            Self::AntiTranspose => "Anti-transpose",
            Self::Rotate90Ccw => "Rotate 90 CCW",
        };
        f.write_str(name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreviewHeader {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnimationHeader {
    pub tps_numerator: u32,
    pub tps_denominator: u32,
    pub num_loops: u32,
    pub have_timecodes: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitDepth {
    pub floating_point_sample: bool,
    pub bits_per_sample: u32,
    pub exponent_bits_per_sample: u32,
}

impl Default for BitDepth {
    fn default() -> Self {
        Self {
            floating_point_sample: false,
            bits_per_sample: 8,
            exponent_bits_per_sample: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExtraChannelInfo {
    pub channel_type: ExtraChannelType,
    pub bit_depth: BitDepth,
    pub dim_shift: u32,
    pub name: String,
    pub alpha_associated: bool,
    pub spot_color: Option<[f32; 4]>,
    pub cfa_channel: Option<u32>,
}

impl Default for ExtraChannelInfo {
    fn default() -> Self {
        Self {
            channel_type: ExtraChannelType::Alpha,
            bit_depth: BitDepth::default(),
            dim_shift: 0,
            name: String::new(),
            alpha_associated: false,
            spot_color: None,
            cfa_channel: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ExtraChannelType {
    Alpha = 0,
    Depth = 1,
    SpotColor = 2,
    SelectionMask = 3,
    Black = 4,
    Cfa = 5,
    Thermal = 6,
    Unknown = 15,
    Optional = 16,
}

impl TryFrom<u32> for ExtraChannelType {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::Alpha),
            1 => Ok(Self::Depth),
            2 => Ok(Self::SpotColor),
            3 => Ok(Self::SelectionMask),
            4 => Ok(Self::Black),
            5 => Ok(Self::Cfa),
            6 => Ok(Self::Thermal),
            15 => Ok(Self::Unknown),
            16 => Ok(Self::Optional),
            _ => Err(Error::InvalidCodestream("invalid extra channel type")),
        }
    }
}

impl fmt::Display for ExtraChannelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Alpha => "Alpha",
            Self::Depth => "Depth",
            Self::SpotColor => "Spot color",
            Self::SelectionMask => "Selection mask",
            Self::Black => "Black",
            Self::Cfa => "CFA",
            Self::Thermal => "Thermal",
            Self::Unknown => "Unknown",
            Self::Optional => "Optional",
        };
        f.write_str(name)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColorEncoding {
    pub want_icc: bool,
    pub color_space: ColorSpace,
    pub white_point: WhitePoint,
    pub custom_white_point: Option<Customxy>,
    pub primaries: Primaries,
    pub custom_primaries: Option<CustomPrimaries>,
    pub transfer_function: TransferFunction,
    pub gamma: Option<u32>,
    pub rendering_intent: RenderingIntent,
}

impl Default for ColorEncoding {
    fn default() -> Self {
        Self {
            want_icc: false,
            color_space: ColorSpace::Rgb,
            white_point: WhitePoint::D65,
            custom_white_point: None,
            primaries: Primaries::Srgb,
            custom_primaries: None,
            transfer_function: TransferFunction::Srgb,
            gamma: None,
            rendering_intent: RenderingIntent::Relative,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ColorSpace {
    Rgb = 0,
    Gray = 1,
    Xyb = 2,
    Unknown = 3,
}

impl TryFrom<u32> for ColorSpace {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::Rgb),
            1 => Ok(Self::Gray),
            2 => Ok(Self::Xyb),
            3 => Ok(Self::Unknown),
            _ => Err(Error::InvalidCodestream("invalid color space")),
        }
    }
}

impl fmt::Display for ColorSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Rgb => "RGB",
            Self::Gray => "Grayscale",
            Self::Xyb => "XYB",
            Self::Unknown => "Unknown",
        };
        f.write_str(name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum WhitePoint {
    D65 = 1,
    Custom = 2,
    E = 10,
    Dci = 11,
}

impl TryFrom<u32> for WhitePoint {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            1 => Ok(Self::D65),
            2 => Ok(Self::Custom),
            10 => Ok(Self::E),
            11 => Ok(Self::Dci),
            _ => Err(Error::InvalidCodestream("invalid white point")),
        }
    }
}

impl fmt::Display for WhitePoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::D65 => "D65",
            Self::Custom => "Custom",
            Self::E => "E",
            Self::Dci => "DCI",
        };
        f.write_str(name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Primaries {
    Srgb = 1,
    Custom = 2,
    Rec2100 = 9,
    P3 = 11,
}

impl TryFrom<u32> for Primaries {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            1 => Ok(Self::Srgb),
            2 => Ok(Self::Custom),
            9 => Ok(Self::Rec2100),
            11 => Ok(Self::P3),
            _ => Err(Error::InvalidCodestream("invalid primaries")),
        }
    }
}

impl fmt::Display for Primaries {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Srgb => "sRGB",
            Self::Custom => "Custom",
            Self::Rec2100 => "Rec.2100",
            Self::P3 => "P3",
        };
        f.write_str(name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum TransferFunction {
    Bt709 = 1,
    Unknown = 2,
    Linear = 8,
    Srgb = 13,
    Pq = 16,
    Dci = 17,
    Hlg = 18,
}

impl TryFrom<u32> for TransferFunction {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            1 => Ok(Self::Bt709),
            2 => Ok(Self::Unknown),
            8 => Ok(Self::Linear),
            13 => Ok(Self::Srgb),
            16 => Ok(Self::Pq),
            17 => Ok(Self::Dci),
            18 => Ok(Self::Hlg),
            _ => Err(Error::InvalidCodestream("invalid transfer function")),
        }
    }
}

impl fmt::Display for TransferFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Bt709 => "BT.709",
            Self::Unknown => "Unknown",
            Self::Linear => "Linear",
            Self::Srgb => "sRGB",
            Self::Pq => "PQ",
            Self::Dci => "DCI",
            Self::Hlg => "HLG",
        };
        f.write_str(name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum RenderingIntent {
    Perceptual = 0,
    Relative = 1,
    Saturation = 2,
    Absolute = 3,
}

impl TryFrom<u32> for RenderingIntent {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::Perceptual),
            1 => Ok(Self::Relative),
            2 => Ok(Self::Saturation),
            3 => Ok(Self::Absolute),
            _ => Err(Error::InvalidCodestream("invalid rendering intent")),
        }
    }
}

impl fmt::Display for RenderingIntent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Perceptual => "Perceptual",
            Self::Relative => "Relative",
            Self::Saturation => "Saturation",
            Self::Absolute => "Absolute",
        };
        f.write_str(name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Customxy {
    pub x: i32,
    pub y: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CustomPrimaries {
    pub red: Customxy,
    pub green: Customxy,
    pub blue: Customxy,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToneMapping {
    pub intensity_target: f32,
    pub min_nits: f32,
    pub relative_to_max_display: bool,
    pub linear_below: f32,
}

impl Default for ToneMapping {
    fn default() -> Self {
        Self {
            intensity_target: 255.0,
            min_nits: 0.0,
            relative_to_max_display: false,
            linear_below: 0.0,
        }
    }
}

pub fn read_image_metadata(reader: &mut BitReader<'_>) -> Result<ImageMetadata> {
    if reader.read_bool()? {
        return Ok(ImageMetadata::default());
    }

    let mut metadata = ImageMetadata::default();

    let extra_fields = reader.read_bool()?;
    if extra_fields {
        let orientation_minus_one = reader.read_bits(3)? as u32;
        metadata.orientation = Orientation::try_from(orientation_minus_one + 1)?;

        if reader.read_bool()? {
            metadata.intrinsic_size = Some(read_size_header(reader)?);
        }

        if reader.read_bool()? {
            metadata.preview_size = Some(read_preview_header(reader)?);
        }

        if reader.read_bool()? {
            metadata.animation = Some(read_animation_header(reader)?);
        }
    }

    metadata.bit_depth = read_bit_depth(reader)?;
    metadata.modular_16_bit_buffer_sufficient = reader.read_bool()?;

    let num_extra_channels =
        reader.read_u32_selector(val(0), val(1), bits_offset(4, 2), bits_offset(12, 1))?;
    let mut extra_channels = Vec::with_capacity(num_extra_channels as usize);
    for _ in 0..num_extra_channels {
        extra_channels.push(read_extra_channel_info(reader)?);
    }
    metadata.extra_channels = extra_channels;

    metadata.xyb_encoded = reader.read_bool()?;
    metadata.color_encoding = read_color_encoding(reader)?;

    if extra_fields {
        metadata.tone_mapping = read_tone_mapping(reader)?;
    }

    metadata.extensions = read_extensions(reader)?;

    Ok(metadata)
}

fn read_bit_depth(reader: &mut BitReader<'_>) -> Result<BitDepth> {
    let floating_point_sample = reader.read_bool()?;
    if !floating_point_sample {
        let bits_per_sample =
            reader.read_u32_selector(val(8), val(10), val(12), bits_offset(6, 1))?;
        if bits_per_sample > 31 {
            return Err(Error::InvalidCodestream("integer bit depth exceeds 31"));
        }
        Ok(BitDepth {
            floating_point_sample,
            bits_per_sample,
            exponent_bits_per_sample: 0,
        })
    } else {
        let bits_per_sample =
            reader.read_u32_selector(val(32), val(16), val(24), bits_offset(6, 1))?;
        let exponent_bits_per_sample = reader.read_bits(4)? as u32 + 1;
        if !(2..=8).contains(&exponent_bits_per_sample) {
            return Err(Error::InvalidCodestream("invalid float exponent bits"));
        }
        let mantissa_bits = bits_per_sample as i32 - exponent_bits_per_sample as i32 - 1;
        if !(2..=23).contains(&mantissa_bits) {
            return Err(Error::InvalidCodestream("invalid floating point bit depth"));
        }

        Ok(BitDepth {
            floating_point_sample,
            bits_per_sample,
            exponent_bits_per_sample,
        })
    }
}

fn read_extra_channel_info(reader: &mut BitReader<'_>) -> Result<ExtraChannelInfo> {
    if reader.read_bool()? {
        return Ok(ExtraChannelInfo::default());
    }

    let channel_type =
        ExtraChannelType::try_from(reader.read_enum(&[0, 1, 2, 3, 4, 5, 6, 15, 16])?)?;
    if channel_type == ExtraChannelType::Unknown {
        return Err(Error::Unsupported("unknown required extra channel"));
    }

    let bit_depth = read_bit_depth(reader)?;
    let dim_shift = reader.read_u32_selector(val(0), val(3), val(4), bits_offset(3, 1))?;
    if dim_shift > 3 {
        return Err(Error::InvalidCodestream(
            "extra channel dim_shift exceeds 3",
        ));
    }

    let name = reader.read_name()?;
    let alpha_associated = if channel_type == ExtraChannelType::Alpha {
        reader.read_bool()?
    } else {
        false
    };
    let spot_color = if channel_type == ExtraChannelType::SpotColor {
        Some([
            reader.read_f16()?,
            reader.read_f16()?,
            reader.read_f16()?,
            reader.read_f16()?,
        ])
    } else {
        None
    };
    let cfa_channel = if channel_type == ExtraChannelType::Cfa {
        Some(reader.read_u32_selector(
            val(1),
            crate::bitstream::U32Distribution::BitsOffset { bits: 2, offset: 0 },
            bits_offset(4, 3),
            bits_offset(8, 19),
        )?)
    } else {
        None
    };

    Ok(ExtraChannelInfo {
        channel_type,
        bit_depth,
        dim_shift,
        name,
        alpha_associated,
        spot_color,
        cfa_channel,
    })
}

fn read_color_encoding(reader: &mut BitReader<'_>) -> Result<ColorEncoding> {
    if reader.read_bool()? {
        return Ok(ColorEncoding::default());
    }

    let mut color = ColorEncoding {
        want_icc: reader.read_bool()?,
        color_space: ColorSpace::try_from(reader.read_enum(&[0, 1, 2, 3])?)?,
        ..ColorEncoding::default()
    };

    if !color.want_icc {
        if color.color_space != ColorSpace::Xyb {
            color.white_point = WhitePoint::try_from(reader.read_enum(&[1, 2, 10, 11])?)?;
            if color.white_point == WhitePoint::Custom {
                color.custom_white_point = Some(read_custom_xy(reader)?);
            }
        }

        if has_primaries(color.color_space) {
            color.primaries = Primaries::try_from(reader.read_enum(&[1, 2, 9, 11])?)?;
            if color.primaries == Primaries::Custom {
                color.custom_primaries = Some(CustomPrimaries {
                    red: read_custom_xy(reader)?,
                    green: read_custom_xy(reader)?,
                    blue: read_custom_xy(reader)?,
                });
            }
        }

        if color.color_space == ColorSpace::Xyb {
            color.gamma = Some(3_333_333);
        } else if reader.read_bool()? {
            let gamma = reader.read_bits(24)? as u32;
            validate_gamma(gamma)?;
            color.gamma = Some(gamma);
        } else {
            color.transfer_function =
                TransferFunction::try_from(reader.read_enum(&[1, 2, 8, 13, 16, 17, 18])?)?;
        }

        color.rendering_intent = RenderingIntent::try_from(reader.read_enum(&[0, 1, 2, 3])?)?;

        if color.color_space == ColorSpace::Unknown
            || (color.gamma.is_none() && color.transfer_function == TransferFunction::Unknown)
        {
            return Err(Error::InvalidCodestream(
                "unknown color space or transfer function requires ICC",
            ));
        }
    }

    Ok(color)
}

fn read_tone_mapping(reader: &mut BitReader<'_>) -> Result<ToneMapping> {
    if reader.read_bool()? {
        return Ok(ToneMapping::default());
    }

    let intensity_target = reader.read_f16()?;
    if intensity_target <= 0.0 {
        return Err(Error::InvalidCodestream("invalid intensity target"));
    }

    let min_nits = reader.read_f16()?;
    if min_nits < 0.0 || min_nits > intensity_target {
        return Err(Error::InvalidCodestream("invalid min_nits"));
    }

    let relative_to_max_display = reader.read_bool()?;
    let linear_below = reader.read_f16()?;
    if linear_below < 0.0 || (relative_to_max_display && linear_below > 1.0) {
        return Err(Error::InvalidCodestream("invalid linear_below"));
    }

    Ok(ToneMapping {
        intensity_target,
        min_nits,
        relative_to_max_display,
        linear_below,
    })
}

fn read_preview_header(reader: &mut BitReader<'_>) -> Result<PreviewHeader> {
    let div8 = reader.read_bool()?;
    let height = if div8 {
        reader.read_u32_selector(val(16), val(32), bits_offset(5, 1), bits_offset(9, 33))? * 8
    } else {
        reader.read_u32_selector(
            bits_offset(6, 1),
            bits_offset(8, 65),
            bits_offset(10, 321),
            bits_offset(12, 1345),
        )?
    };

    let ratio = reader.read_bits(3)? as u32;
    let width = if ratio == 0 {
        if div8 {
            reader.read_u32_selector(val(16), val(32), bits_offset(5, 1), bits_offset(9, 33))? * 8
        } else {
            reader.read_u32_selector(
                bits_offset(6, 1),
                bits_offset(8, 65),
                bits_offset(10, 321),
                bits_offset(12, 1345),
            )?
        }
    } else {
        fixed_aspect_width(ratio, height)?
    };

    Ok(PreviewHeader { width, height })
}

fn read_animation_header(reader: &mut BitReader<'_>) -> Result<AnimationHeader> {
    Ok(AnimationHeader {
        tps_numerator: reader.read_u32_selector(
            val(100),
            val(1000),
            bits_offset(10, 1),
            bits_offset(30, 1),
        )?,
        tps_denominator: reader.read_u32_selector(
            val(1),
            val(1001),
            bits_offset(8, 1),
            bits_offset(10, 1),
        )?,
        num_loops: reader.read_u32_selector(
            val(0),
            crate::bitstream::U32Distribution::BitsOffset { bits: 3, offset: 0 },
            crate::bitstream::U32Distribution::BitsOffset {
                bits: 16,
                offset: 0,
            },
            crate::bitstream::U32Distribution::BitsOffset {
                bits: 32,
                offset: 0,
            },
        )?,
        have_timecodes: reader.read_bool()?,
    })
}

fn read_custom_xy(reader: &mut BitReader<'_>) -> Result<Customxy> {
    let x = read_packed_signed_xy(reader)?;
    let y = read_packed_signed_xy(reader)?;
    Ok(Customxy { x, y })
}

fn read_packed_signed_xy(reader: &mut BitReader<'_>) -> Result<i32> {
    let packed = reader.read_u32_selector(
        crate::bitstream::U32Distribution::BitsOffset {
            bits: 19,
            offset: 0,
        },
        bits_offset(19, 524_288),
        bits_offset(20, 1_048_576),
        bits_offset(21, 2_097_152),
    )?;
    Ok(unpack_signed(packed))
}

fn unpack_signed(value: u32) -> i32 {
    if value & 1 == 0 {
        (value >> 1) as i32
    } else {
        -((value >> 1) as i32) - 1
    }
}

fn read_extensions(reader: &mut BitReader<'_>) -> Result<u64> {
    let extensions = reader.read_u64()?;
    let mut total_extension_bits = 0u64;
    let mut remaining = extensions;
    while remaining != 0 {
        remaining &= remaining - 1;
        let extension_bits = reader.read_u64()?;
        total_extension_bits = total_extension_bits
            .checked_add(extension_bits)
            .ok_or(Error::InvalidCodestream("extension bit count overflow"))?;
    }
    reader.skip_bits(
        total_extension_bits
            .try_into()
            .map_err(|_| Error::InvalidCodestream("extension bit count overflow"))?,
    )?;
    Ok(extensions)
}

fn fixed_aspect_width(ratio: u32, height: u32) -> Result<u32> {
    let (num, den) = match ratio {
        1 => (1, 1),
        2 => (12, 10),
        3 => (4, 3),
        4 => (3, 2),
        5 => (16, 9),
        6 => (5, 4),
        7 => (2, 1),
        _ => return Err(Error::InvalidCodestream("invalid size aspect ratio")),
    };

    Ok(((height as u64 * num) / den) as u32)
}

fn has_primaries(color_space: ColorSpace) -> bool {
    color_space != ColorSpace::Gray && color_space != ColorSpace::Xyb
}

fn validate_gamma(gamma: u32) -> Result<()> {
    const GAMMA_MUL: u64 = 10_000_000;
    const MAX_GAMMA: u64 = 1_000_000_000;
    if u64::from(gamma) > GAMMA_MUL || u64::from(gamma) * MAX_GAMMA < GAMMA_MUL {
        return Err(Error::InvalidCodestream("invalid gamma"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_all_default_image_metadata() {
        let mut reader = BitReader::new(&[1]);
        let metadata = read_image_metadata(&mut reader).unwrap();

        assert_eq!(metadata.orientation, Orientation::Identity);
        assert_eq!(metadata.bit_depth.bits_per_sample, 8);
        assert!(metadata.xyb_encoded);
        assert_eq!(metadata.color_encoding.color_space, ColorSpace::Rgb);
        assert_eq!(
            metadata.color_encoding.transfer_function,
            TransferFunction::Srgb
        );
    }
}
