//! Public Rust-native JPEG XL API.
//!
//! The API is intentionally small while the decoder is being built out. It
//! exposes stable metadata inspection now and leaves room for future streaming
//! decode, region decode, and pixel-output builders without committing to a
//! C-style event API.

pub use jxl_codec::{
    BasicInfo, BitDepth, BlendMode, BlendingInfo, BoxRecord, ColorEncoding, ColorSpace, Container,
    CustomTransformData, Error, ExtraChannelInfo, ExtraChannelType, FileFormat, FrameData,
    FrameEncoding, FrameGroupLayout, FrameHeader, FrameSection, FrameSectionKind, FrameToc,
    FrameType, ImageMetadata, MaTree, MaTreeNode, ModularChannel, ModularChannelPlan,
    ModularDecodedChannel, ModularDecodedGroup, ModularFrameMetadata, ModularGlobalSection,
    ModularGroupChannelPlan, ModularGroupHeader, ModularImage, ModularImageChannel,
    ModularPredictor, ModularResiduals, ModularSectionMetadata, ModularTransform,
    ModularTreeMetadata, OpsinInverseMatrix, Orientation, Primaries, RenderingIntent, Result,
    SqueezeParams, TocEntry, ToneMapping, TransferFunction, TransformId, WeightedPredictorHeader,
    WhitePoint,
};

#[derive(Debug, Clone, PartialEq)]
pub struct ImageInfo {
    pub format: FileFormat,
    pub width: u32,
    pub height: u32,
    pub basic_info: BasicInfo,
    pub metadata: ImageMetadata,
    pub transform_data: CustomTransformData,
    pub icc_profile: Option<Vec<u8>>,
    pub first_frame: Option<FrameHeader>,
    pub first_frame_data: Option<FrameData>,
    pub first_frame_modular: Option<ModularFrameMetadata>,
    pub boxes: Vec<BoxRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    pub color_channels: usize,
    pub bit_depth: u32,
    pub pixels: PixelData,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PixelData {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

pub fn inspect(input: &[u8]) -> Result<ImageInfo> {
    let (extracted, codestream) = jxl_codec::parse_file(input)?;
    Ok(ImageInfo {
        format: extracted.format,
        width: codestream.basic_info.width,
        height: codestream.basic_info.height,
        basic_info: codestream.basic_info,
        metadata: codestream.metadata,
        transform_data: codestream.transform_data,
        icc_profile: codestream.icc_profile,
        first_frame: codestream.first_frame,
        first_frame_data: codestream.first_frame_data,
        first_frame_modular: codestream.first_frame_modular,
        boxes: extracted
            .container
            .map(|container| container.boxes)
            .unwrap_or_default(),
    })
}

pub fn decode(input: &[u8]) -> Result<DecodedImage> {
    let (_, codestream) = jxl_codec::parse_file(input)?;
    if codestream.basic_info.have_animation {
        return Err(Error::Unsupported("animated image decode"));
    }
    if !codestream.metadata.extra_channels.is_empty() {
        return Err(Error::Unsupported("extra-channel image decode"));
    }
    let frame = codestream
        .first_frame
        .as_ref()
        .ok_or(Error::Unsupported("image has no decoded frame"))?;
    if frame.encoding != FrameEncoding::Modular {
        return Err(Error::Unsupported("VarDCT image decode"));
    }
    let modular = codestream
        .first_frame_modular
        .as_ref()
        .ok_or(Error::Unsupported("modular image metadata"))?;
    let image = modular
        .image
        .as_ref()
        .ok_or(Error::Unsupported("modular pixel reconstruction"))?;
    let color_channels = codestream.metadata.num_color_channels() as usize;
    if image.channels.len() != color_channels {
        return Err(Error::Unsupported("non-color modular channel output"));
    }
    if image
        .channels
        .iter()
        .any(|channel| channel.width != image.width || channel.height != image.height)
    {
        return Err(Error::Unsupported("subsampled raw channel output"));
    }

    let bit_depth = codestream.metadata.bit_depth.bits_per_sample;
    if codestream.metadata.bit_depth.floating_point_sample {
        return Err(Error::Unsupported("floating-point sample output"));
    }
    if bit_depth <= 8 {
        Ok(DecodedImage {
            width: image.width,
            height: image.height,
            color_channels,
            bit_depth,
            pixels: PixelData::U8(interleave_u8(image, color_channels, bit_depth)?),
        })
    } else if bit_depth <= 16 {
        Ok(DecodedImage {
            width: image.width,
            height: image.height,
            color_channels,
            bit_depth,
            pixels: PixelData::U16(interleave_u16(image, color_channels, bit_depth)?),
        })
    } else {
        Err(Error::Unsupported("integer sample depths above 16 bits"))
    }
}

fn interleave_u8(image: &ModularImage, color_channels: usize, bit_depth: u32) -> Result<Vec<u8>> {
    let max_sample = max_sample_value(bit_depth)?;
    let pixels = image_sample_count(image)?
        .checked_mul(color_channels)
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    let mut output = Vec::with_capacity(pixels);
    for index in 0..image_sample_count(image)? {
        for channel in &image.channels {
            let sample = checked_sample(channel.samples[index], max_sample)?;
            output.push(sample as u8);
        }
    }
    Ok(output)
}

fn interleave_u16(image: &ModularImage, color_channels: usize, bit_depth: u32) -> Result<Vec<u16>> {
    let max_sample = max_sample_value(bit_depth)?;
    let pixels = image_sample_count(image)?
        .checked_mul(color_channels)
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    let mut output = Vec::with_capacity(pixels);
    for index in 0..image_sample_count(image)? {
        for channel in &image.channels {
            let sample = checked_sample(channel.samples[index], max_sample)?;
            output.push(sample as u16);
        }
    }
    Ok(output)
}

fn image_sample_count(image: &ModularImage) -> Result<usize> {
    (image.width as usize)
        .checked_mul(image.height as usize)
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))
}

fn max_sample_value(bit_depth: u32) -> Result<u32> {
    if bit_depth == 0 || bit_depth > 16 {
        return Err(Error::Unsupported("unsupported integer sample depth"));
    }
    Ok((1u32 << bit_depth) - 1)
}

fn checked_sample(sample: i32, max_sample: u32) -> Result<u32> {
    if sample < 0 || sample as u32 > max_sample {
        return Err(Error::InvalidCodestream(
            "decoded sample outside bit-depth range",
        ));
    }
    Ok(sample as u32)
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::*;

    #[test]
    fn decodes_generated_rgb_modular_fixture() {
        let bytes = std::fs::read(workspace_path(
            "crates/jxl-codec/tests/generated/icc_rec2020_lossless.jxl",
        ))
        .unwrap();
        let image = decode(&bytes).unwrap();

        assert_eq!(image.width, 64);
        assert_eq!(image.height, 64);
        assert_eq!(image.color_channels, 3);
        assert_eq!(image.bit_depth, 16);
        let PixelData::U16(pixels) = image.pixels else {
            panic!("expected 16-bit pixels");
        };
        assert_eq!(pixels.len(), 64 * 64 * 3);
        assert_eq!(*pixels.iter().min().unwrap(), 0);
        assert_eq!(*pixels.iter().max().unwrap(), 14482);
    }

    #[test]
    fn decodes_gray_palette_modular_fixture() {
        let bytes = std::fs::read(workspace_path(
            "reference/libjxl/testdata/jxl/pq_gradient.jxl",
        ))
        .unwrap();
        let image = decode(&bytes).unwrap();

        assert_eq!(image.width, 1088);
        assert_eq!(image.height, 64);
        assert_eq!(image.color_channels, 1);
        assert_eq!(image.bit_depth, 16);
        let PixelData::U16(pixels) = image.pixels else {
            panic!("expected 16-bit pixels");
        };
        assert_eq!(pixels.len(), 1088 * 64);
        assert_eq!(*pixels.iter().min().unwrap(), 6682);
        assert_eq!(*pixels.iter().max().unwrap(), 58853);
    }

    #[test]
    fn rejects_var_dct_for_now() {
        let bytes = std::fs::read(workspace_path(
            "reference/libjxl/testdata/jxl/boxes/square-extended-size-container.jxl",
        ))
        .unwrap();

        assert_eq!(
            decode(&bytes),
            Err(Error::Unsupported("VarDCT image decode"))
        );
    }

    fn workspace_path(relative: impl AsRef<Path>) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(relative)
    }
}
