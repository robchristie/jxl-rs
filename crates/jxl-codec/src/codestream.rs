use crate::bitstream::{BitReader, bits_offset};
use crate::decode::DecodeConfig;
use crate::error::{Error, Result};
use crate::frame::{FrameHeader, read_frame_header};
use crate::frame_data::{FrameData, read_frame_data};
use crate::icc::read_icc_profile;
use crate::metadata::{ImageMetadata, read_image_metadata};
use crate::modular::{ModularFrameMetadata, read_modular_frame_metadata};
use crate::transform::{CustomTransformData, read_custom_transform_data};
use crate::vardct::{VarDctDecodePlan, VarDctFrameMetadata, read_vardct_decode_plan};

pub const CODESTREAM_SIGNATURE: [u8; 2] = [0xff, 0x0a];

#[derive(Debug, Clone, PartialEq)]
pub struct Codestream {
    pub basic_info: BasicInfo,
    pub metadata: ImageMetadata,
    pub transform_data: CustomTransformData,
    pub icc_profile: Option<Vec<u8>>,
    pub frames: Vec<FrameHeader>,
    pub frame_data: Vec<FrameData>,
    pub modular_frames: Vec<Option<ModularFrameMetadata>>,
    pub vardct_plans: Vec<Option<VarDctDecodePlan>>,
    pub vardct_frames: Vec<Option<VarDctFrameMetadata>>,
    pub first_frame: Option<FrameHeader>,
    pub first_frame_data: Option<FrameData>,
    pub first_frame_modular: Option<ModularFrameMetadata>,
    pub first_frame_vardct: Option<VarDctFrameMetadata>,
    pub first_frame_vardct_plan: Option<VarDctDecodePlan>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BasicInfo {
    pub width: u32,
    pub height: u32,
    pub bits_per_sample: u32,
    pub exponent_bits_per_sample: u32,
    pub intensity_target: f32,
    pub min_nits: f32,
    pub relative_to_max_display: bool,
    pub linear_below: f32,
    pub uses_original_profile: bool,
    pub have_preview: bool,
    pub have_animation: bool,
    pub orientation: u32,
    pub num_color_channels: u32,
    pub num_extra_channels: u32,
    pub alpha_bits: u32,
    pub alpha_exponent_bits: u32,
    pub alpha_premultiplied: bool,
    pub intrinsic_width: u32,
    pub intrinsic_height: u32,
    pub header_bits_consumed: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SizeHeader {
    pub width: u32,
    pub height: u32,
}

pub fn parse_codestream(input: &[u8]) -> Result<Codestream> {
    parse_codestream_with_config(input, DecodeConfig::default())
}

pub fn parse_codestream_with_config(input: &[u8], config: DecodeConfig) -> Result<Codestream> {
    let config = config.validate()?;
    let payload = input
        .strip_prefix(&CODESTREAM_SIGNATURE)
        .ok_or(Error::InvalidCodestream("missing codestream signature"))?;
    let mut reader = BitReader::new(payload);
    let size = read_size_header(&mut reader)?;
    let metadata = read_image_metadata(&mut reader)?;
    let transform_data = read_custom_transform_data(&mut reader, metadata.xyb_encoded)?;
    let icc_profile = if metadata.color_encoding.want_icc {
        Some(read_icc_profile(&mut reader)?)
    } else {
        None
    };
    reader.jump_to_byte_boundary()?;

    let mut frames = Vec::new();
    let mut frame_data = Vec::new();
    let mut header_bits_consumed = None;
    loop {
        let frame = read_frame_header(&mut reader, size.width, size.height, &metadata)?;
        if header_bits_consumed.is_none() {
            header_bits_consumed = Some(CODESTREAM_SIGNATURE.len() * 8 + reader.bits_consumed());
        }
        let is_last = frame.is_last;
        let data = read_frame_data(&mut reader, &frame, CODESTREAM_SIGNATURE.len())?;
        frames.push(frame);
        frame_data.push(data);
        if is_last {
            break;
        }
    }

    let header_bits_consumed = header_bits_consumed.unwrap_or(CODESTREAM_SIGNATURE.len() * 8);
    let first_frame = frames.first().cloned();
    let first_frame_data = frame_data.first().cloned();

    let modular_frames = frames
        .iter()
        .zip(&frame_data)
        .map(|(frame, frame_data)| {
            read_modular_frame_metadata(input, &metadata, frame, frame_data, config)
        })
        .collect::<Result<Vec<_>>>()?;
    let vardct_plans = frames
        .iter()
        .zip(&frame_data)
        .map(|(frame, frame_data)| {
            read_vardct_decode_plan(input, &metadata, &transform_data, frame, frame_data)
        })
        .collect::<Result<Vec<_>>>()?;
    let vardct_frames = vardct_plans
        .iter()
        .map(|plan| plan.as_ref().map(|plan| plan.frame.clone()))
        .collect::<Vec<_>>();

    let first_frame_modular = modular_frames.first().cloned().flatten();
    let first_frame_vardct_plan = vardct_plans.first().cloned().flatten();
    let first_frame_vardct = first_frame_vardct_plan
        .as_ref()
        .map(|plan| plan.frame.clone());

    Ok(Codestream {
        basic_info: BasicInfo {
            width: size.width,
            height: size.height,
            bits_per_sample: metadata.bit_depth.bits_per_sample,
            exponent_bits_per_sample: metadata.bit_depth.exponent_bits_per_sample,
            intensity_target: metadata.tone_mapping.intensity_target,
            min_nits: metadata.tone_mapping.min_nits,
            relative_to_max_display: metadata.tone_mapping.relative_to_max_display,
            linear_below: metadata.tone_mapping.linear_below,
            uses_original_profile: !metadata.xyb_encoded,
            have_preview: metadata.preview_size.is_some(),
            have_animation: metadata.animation.is_some(),
            orientation: metadata.orientation as u32,
            num_color_channels: metadata.num_color_channels(),
            num_extra_channels: metadata.extra_channels.len() as u32,
            alpha_bits: metadata.alpha_bits(),
            alpha_exponent_bits: metadata.alpha_exponent_bits(),
            alpha_premultiplied: metadata.alpha_premultiplied(),
            intrinsic_width: metadata
                .intrinsic_size
                .map(|size| size.width)
                .unwrap_or(size.width),
            intrinsic_height: metadata
                .intrinsic_size
                .map(|size| size.height)
                .unwrap_or(size.height),
            header_bits_consumed,
        },
        metadata,
        transform_data,
        icc_profile,
        frames,
        frame_data,
        modular_frames,
        vardct_plans,
        vardct_frames,
        first_frame,
        first_frame_data,
        first_frame_modular,
        first_frame_vardct,
        first_frame_vardct_plan,
    })
}

pub fn read_size_header(reader: &mut BitReader<'_>) -> Result<SizeHeader> {
    let small = reader.read_bool()?;

    let ysize = if small {
        let ysize_div8_minus_1 = reader.read_bits(5)? as u32;
        (ysize_div8_minus_1 + 1) * 8
    } else {
        reader.read_u32_selector(
            bits_offset(9, 1),
            bits_offset(13, 1),
            bits_offset(18, 1),
            bits_offset(30, 1),
        )?
    };

    let ratio = reader.read_bits(3)? as u32;
    if ratio > 7 {
        return Err(Error::InvalidCodestream("invalid size aspect ratio"));
    }

    let width = if ratio == 0 {
        if small {
            let xsize_div8_minus_1 = reader.read_bits(5)? as u32;
            (xsize_div8_minus_1 + 1) * 8
        } else {
            reader.read_u32_selector(
                bits_offset(9, 1),
                bits_offset(13, 1),
                bits_offset(18, 1),
                bits_offset(30, 1),
            )?
        }
    } else {
        fixed_aspect_width(ratio, ysize)?
    };

    if width == 0 || ysize == 0 {
        return Err(Error::InvalidCodestream("empty image"));
    }

    Ok(SizeHeader {
        width,
        height: ysize,
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_small_square_size_header() {
        // small = true, ysize_div8_minus_1 = 0, ratio = 1 => 8x8.
        let mut reader = BitReader::new(&[0b0100_0001, 0]);

        assert_eq!(
            read_size_header(&mut reader).unwrap(),
            SizeHeader {
                width: 8,
                height: 8
            }
        );
    }

    #[test]
    fn parses_small_explicit_size_header() {
        // small = true, ysize_div8_minus_1 = 1, ratio = 0,
        // xsize_div8_minus_1 = 2 => 24x16.
        let mut reader = BitReader::new(&[0b0000_0011, 0b0000_0100]);

        assert_eq!(
            read_size_header(&mut reader).unwrap(),
            SizeHeader {
                width: 24,
                height: 16
            }
        );
    }
}
