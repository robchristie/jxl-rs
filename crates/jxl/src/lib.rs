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
    pub alpha: Option<AlphaInfo>,
    pub bit_depth: u32,
    pub pixels: PixelData,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedChannels {
    pub width: u32,
    pub height: u32,
    pub color_channels: usize,
    pub alpha: Option<AlphaInfo>,
    pub bit_depth: u32,
    pub channels: Vec<DecodedChannel>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedChannel {
    pub width: u32,
    pub height: u32,
    pub hshift: i32,
    pub vshift: i32,
    pub samples: ChannelData,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlphaInfo {
    pub bit_depth: u32,
    pub premultiplied: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbaImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rgba16Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PixelData {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChannelData {
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

pub fn decode_channels(input: &[u8]) -> Result<DecodedChannels> {
    let (_, codestream) = jxl_codec::parse_file(input)?;
    if codestream.basic_info.have_animation {
        return Err(Error::Unsupported("animated image decode"));
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
    let bit_depth = codestream.metadata.bit_depth.bits_per_sample;
    if codestream.metadata.bit_depth.floating_point_sample {
        return Err(Error::Unsupported("floating-point sample output"));
    }
    if bit_depth > 16 {
        return Err(Error::Unsupported("integer sample depths above 16 bits"));
    }
    let max_sample = max_sample_value(bit_depth)?;
    let channels = image
        .channels
        .iter()
        .map(|channel| decode_channel(image.width, image.height, channel, bit_depth, max_sample))
        .collect::<Result<Vec<_>>>()?;
    Ok(DecodedChannels {
        width: image.width,
        height: image.height,
        color_channels,
        alpha: raw_alpha_info(&codestream.metadata)?,
        bit_depth,
        channels,
    })
}

pub fn decode(input: &[u8]) -> Result<DecodedImage> {
    let channels = decode_channels(input)?;
    let alpha = decode_interleaved_alpha(&channels)?;
    let output_channels = channels.color_channels + usize::from(alpha.is_some());
    if channels.channels.len() != output_channels {
        return Err(Error::Unsupported("non-color modular channel output"));
    }
    if channels
        .channels
        .iter()
        .any(|channel| channel.width != channels.width || channel.height != channels.height)
    {
        return Err(Error::Unsupported("subsampled raw channel output"));
    }

    if channels.bit_depth <= 8 {
        Ok(DecodedImage {
            width: channels.width,
            height: channels.height,
            color_channels: channels.color_channels,
            alpha,
            bit_depth: channels.bit_depth,
            pixels: PixelData::U8(interleave_channel_u8(&channels)?),
        })
    } else {
        Ok(DecodedImage {
            width: channels.width,
            height: channels.height,
            color_channels: channels.color_channels,
            alpha,
            bit_depth: channels.bit_depth,
            pixels: PixelData::U16(interleave_channel_u16(&channels)?),
        })
    }
}

pub fn decode_rgba8(input: &[u8]) -> Result<RgbaImage> {
    let decoded = decode(input)?;
    let pixels = match &decoded.pixels {
        PixelData::U8(samples) => rgba8_from_u8(&decoded, samples)?,
        PixelData::U16(samples) => rgba8_from_u16(&decoded, samples)?,
    };
    Ok(RgbaImage {
        width: decoded.width,
        height: decoded.height,
        pixels,
    })
}

pub fn decode_rgba16(input: &[u8]) -> Result<Rgba16Image> {
    let decoded = decode(input)?;
    let pixels = match &decoded.pixels {
        PixelData::U8(samples) => rgba16_from_u8(&decoded, samples)?,
        PixelData::U16(samples) => rgba16_from_u16(&decoded, samples)?,
    };
    Ok(Rgba16Image {
        width: decoded.width,
        height: decoded.height,
        pixels,
    })
}

fn raw_alpha_info(metadata: &ImageMetadata) -> Result<Option<AlphaInfo>> {
    let Some(alpha) = metadata
        .extra_channels
        .iter()
        .find(|channel| channel.channel_type == ExtraChannelType::Alpha)
    else {
        return Ok(None);
    };
    if alpha.bit_depth.floating_point_sample {
        return Err(Error::Unsupported("floating-point alpha output"));
    }
    Ok(Some(AlphaInfo {
        bit_depth: alpha.bit_depth.bits_per_sample,
        premultiplied: alpha.alpha_associated,
    }))
}

fn decode_interleaved_alpha(channels: &DecodedChannels) -> Result<Option<AlphaInfo>> {
    let alpha = channels.alpha;
    if let Some(alpha) = alpha {
        if alpha.bit_depth != channels.bit_depth {
            return Err(Error::Unsupported("mixed bit-depth alpha output"));
        }
        if channels.channels.len() <= channels.color_channels {
            return Err(Error::Unsupported("missing alpha channel output"));
        }
        let alpha_channel = &channels.channels[channels.color_channels];
        if alpha_channel.hshift != 0 || alpha_channel.vshift != 0 {
            return Err(Error::Unsupported("subsampled alpha image decode"));
        }
    }
    Ok(alpha)
}

fn decode_channel(
    image_width: u32,
    image_height: u32,
    channel: &ModularImageChannel,
    bit_depth: u32,
    max_sample: u32,
) -> Result<DecodedChannel> {
    let expected = (channel.width as usize)
        .checked_mul(channel.height as usize)
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    if channel.samples.len() != expected {
        return Err(Error::InvalidCodestream("decoded pixel count mismatch"));
    }
    let (hshift, vshift) =
        infer_channel_shifts(image_width, image_height, channel.width, channel.height)?;
    let samples = if bit_depth <= 8 {
        ChannelData::U8(
            channel
                .samples
                .iter()
                .copied()
                .map(|sample| checked_sample(sample, max_sample).map(|sample| sample as u8))
                .collect::<Result<Vec<_>>>()?,
        )
    } else {
        ChannelData::U16(
            channel
                .samples
                .iter()
                .copied()
                .map(|sample| checked_sample(sample, max_sample).map(|sample| sample as u16))
                .collect::<Result<Vec<_>>>()?,
        )
    };
    Ok(DecodedChannel {
        width: channel.width,
        height: channel.height,
        hshift,
        vshift,
        samples,
    })
}

fn infer_channel_shifts(
    image_width: u32,
    image_height: u32,
    channel_width: u32,
    channel_height: u32,
) -> Result<(i32, i32)> {
    let hshift = infer_shift(image_width, channel_width)?;
    let vshift = infer_shift(image_height, channel_height)?;
    Ok((hshift, vshift))
}

fn infer_shift(full: u32, shifted: u32) -> Result<i32> {
    for shift in 0..=30 {
        let divisor = 1u32 << shift;
        if full.div_ceil(divisor) == shifted {
            return Ok(shift);
        }
    }
    Err(Error::Unsupported("non power-of-two channel geometry"))
}

fn interleave_channel_u8(image: &DecodedChannels) -> Result<Vec<u8>> {
    let output_channels = channel_output_channels(image);
    let sample_count = decoded_channel_sample_count(image)?;
    let pixels = sample_count
        .checked_mul(output_channels)
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    let mut output = Vec::with_capacity(pixels);
    for index in 0..sample_count {
        for channel in &image.channels {
            let ChannelData::U8(samples) = &channel.samples else {
                return Err(Error::InvalidCodestream(
                    "decoded channel bit-depth mismatch",
                ));
            };
            output.push(samples[index]);
        }
    }
    Ok(output)
}

fn interleave_channel_u16(image: &DecodedChannels) -> Result<Vec<u16>> {
    let output_channels = channel_output_channels(image);
    let sample_count = decoded_channel_sample_count(image)?;
    let pixels = sample_count
        .checked_mul(output_channels)
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    let mut output = Vec::with_capacity(pixels);
    for index in 0..sample_count {
        for channel in &image.channels {
            let ChannelData::U16(samples) = &channel.samples else {
                return Err(Error::InvalidCodestream(
                    "decoded channel bit-depth mismatch",
                ));
            };
            output.push(samples[index]);
        }
    }
    Ok(output)
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

fn rgba8_from_u8(image: &DecodedImage, samples: &[u8]) -> Result<Vec<u8>> {
    let input_channels = decoded_image_output_channels(image);
    let sample_count = decoded_image_sample_count(image)?;
    if samples.len() != sample_count * input_channels {
        return Err(Error::InvalidCodestream("decoded pixel count mismatch"));
    }
    let mut rgba = Vec::with_capacity(sample_count * 4);
    for pixel in samples.chunks_exact(input_channels) {
        append_rgba8_pixel(
            &mut rgba,
            image.color_channels,
            image.alpha.is_some(),
            |index| scale_sample_to_u8(u32::from(pixel[index]), image.bit_depth),
        )?;
    }
    Ok(rgba)
}

fn rgba8_from_u16(image: &DecodedImage, samples: &[u16]) -> Result<Vec<u8>> {
    let input_channels = decoded_image_output_channels(image);
    let sample_count = decoded_image_sample_count(image)?;
    if samples.len() != sample_count * input_channels {
        return Err(Error::InvalidCodestream("decoded pixel count mismatch"));
    }
    let mut rgba = Vec::with_capacity(sample_count * 4);
    for pixel in samples.chunks_exact(input_channels) {
        append_rgba8_pixel(
            &mut rgba,
            image.color_channels,
            image.alpha.is_some(),
            |index| scale_sample_to_u8(u32::from(pixel[index]), image.bit_depth),
        )?;
    }
    Ok(rgba)
}

fn rgba16_from_u8(image: &DecodedImage, samples: &[u8]) -> Result<Vec<u16>> {
    let input_channels = decoded_image_output_channels(image);
    let sample_count = decoded_image_sample_count(image)?;
    if samples.len() != sample_count * input_channels {
        return Err(Error::InvalidCodestream("decoded pixel count mismatch"));
    }
    let mut rgba = Vec::with_capacity(sample_count * 4);
    for pixel in samples.chunks_exact(input_channels) {
        append_rgba16_pixel(
            &mut rgba,
            image.color_channels,
            image.alpha.is_some(),
            |index| scale_sample_to_u16(u32::from(pixel[index]), image.bit_depth),
        )?;
    }
    Ok(rgba)
}

fn rgba16_from_u16(image: &DecodedImage, samples: &[u16]) -> Result<Vec<u16>> {
    let input_channels = decoded_image_output_channels(image);
    let sample_count = decoded_image_sample_count(image)?;
    if samples.len() != sample_count * input_channels {
        return Err(Error::InvalidCodestream("decoded pixel count mismatch"));
    }
    let mut rgba = Vec::with_capacity(sample_count * 4);
    for pixel in samples.chunks_exact(input_channels) {
        append_rgba16_pixel(
            &mut rgba,
            image.color_channels,
            image.alpha.is_some(),
            |index| scale_sample_to_u16(u32::from(pixel[index]), image.bit_depth),
        )?;
    }
    Ok(rgba)
}

fn append_rgba8_pixel(
    rgba: &mut Vec<u8>,
    color_channels: usize,
    has_alpha: bool,
    sample: impl Fn(usize) -> u8,
) -> Result<()> {
    match color_channels {
        1 => {
            let gray = sample(0);
            rgba.extend_from_slice(&[gray, gray, gray]);
        }
        3 => {
            rgba.extend_from_slice(&[sample(0), sample(1), sample(2)]);
        }
        _ => return Err(Error::Unsupported("unsupported color channel count")),
    }
    rgba.push(if has_alpha {
        sample(color_channels)
    } else {
        255
    });
    Ok(())
}

fn append_rgba16_pixel(
    rgba: &mut Vec<u16>,
    color_channels: usize,
    has_alpha: bool,
    sample: impl Fn(usize) -> u16,
) -> Result<()> {
    match color_channels {
        1 => {
            let gray = sample(0);
            rgba.extend_from_slice(&[gray, gray, gray]);
        }
        3 => {
            rgba.extend_from_slice(&[sample(0), sample(1), sample(2)]);
        }
        _ => return Err(Error::Unsupported("unsupported color channel count")),
    }
    rgba.push(if has_alpha {
        sample(color_channels)
    } else {
        u16::MAX
    });
    Ok(())
}

fn channel_output_channels(image: &DecodedChannels) -> usize {
    image.color_channels + usize::from(image.alpha.is_some())
}

fn decoded_channel_sample_count(image: &DecodedChannels) -> Result<usize> {
    (image.width as usize)
        .checked_mul(image.height as usize)
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))
}

fn decoded_image_output_channels(image: &DecodedImage) -> usize {
    image.color_channels + usize::from(image.alpha.is_some())
}

fn decoded_image_sample_count(image: &DecodedImage) -> Result<usize> {
    (image.width as usize)
        .checked_mul(image.height as usize)
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))
}

fn scale_sample_to_u8(sample: u32, bit_depth: u32) -> u8 {
    scale_sample_to(sample, bit_depth, u8::MAX as u32) as u8
}

fn scale_sample_to_u16(sample: u32, bit_depth: u32) -> u16 {
    scale_sample_to(sample, bit_depth, u16::MAX as u32) as u16
}

fn scale_sample_to(sample: u32, bit_depth: u32, output_max: u32) -> u32 {
    let max = (1u32 << bit_depth) - 1;
    ((sample * output_max + max / 2) / max).min(output_max)
}

#[cfg(test)]
mod tests {
    use std::{
        path::{Path, PathBuf},
        process::Command,
        time::{SystemTime, UNIX_EPOCH},
    };

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
        assert_eq!(image.alpha, None);
        assert_eq!(image.bit_depth, 16);
        let PixelData::U16(pixels) = image.pixels else {
            panic!("expected 16-bit pixels");
        };
        assert_eq!(pixels.len(), 64 * 64 * 3);
        assert_eq!(*pixels.iter().min().unwrap(), 0);
        assert_eq!(*pixels.iter().max().unwrap(), 14482);
    }

    #[test]
    fn decodes_generated_rgb_modular_channels() {
        let bytes = std::fs::read(workspace_path(
            "crates/jxl-codec/tests/generated/icc_rec2020_lossless.jxl",
        ))
        .unwrap();
        let image = decode_channels(&bytes).unwrap();

        assert_eq!(image.width, 64);
        assert_eq!(image.height, 64);
        assert_eq!(image.color_channels, 3);
        assert_eq!(image.alpha, None);
        assert_eq!(image.bit_depth, 16);
        assert_eq!(image.channels.len(), 3);
        for channel in &image.channels {
            assert_eq!(channel.width, 64);
            assert_eq!(channel.height, 64);
            assert_eq!(channel.hshift, 0);
            assert_eq!(channel.vshift, 0);
            let ChannelData::U16(samples) = &channel.samples else {
                panic!("expected 16-bit channel samples");
            };
            assert_eq!(samples.len(), 64 * 64);
        }
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
        assert_eq!(image.alpha, None);
        assert_eq!(image.bit_depth, 16);
        let PixelData::U16(pixels) = image.pixels else {
            panic!("expected 16-bit pixels");
        };
        assert_eq!(pixels.len(), 1088 * 64);
        assert_eq!(*pixels.iter().min().unwrap(), 6682);
        assert_eq!(*pixels.iter().max().unwrap(), 58853);
    }

    #[test]
    fn decodes_gray_palette_modular_channels() {
        let bytes = std::fs::read(workspace_path(
            "reference/libjxl/testdata/jxl/pq_gradient.jxl",
        ))
        .unwrap();
        let image = decode_channels(&bytes).unwrap();

        assert_eq!(image.width, 1088);
        assert_eq!(image.height, 64);
        assert_eq!(image.color_channels, 1);
        assert_eq!(image.alpha, None);
        assert_eq!(image.bit_depth, 16);
        assert_eq!(image.channels.len(), 1);
        let channel = &image.channels[0];
        assert_eq!(channel.width, 1088);
        assert_eq!(channel.height, 64);
        assert_eq!(channel.hshift, 0);
        assert_eq!(channel.vshift, 0);
        let ChannelData::U16(samples) = &channel.samples else {
            panic!("expected 16-bit channel samples");
        };
        assert_eq!(samples.len(), 1088 * 64);
        assert_eq!(*samples.iter().min().unwrap(), 6682);
        assert_eq!(*samples.iter().max().unwrap(), 58853);
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

    #[test]
    fn decode_rgb_pixels_match_reference_djxl_when_available() {
        let Some(djxl) = reference_djxl() else {
            eprintln!("skipping public decode djxl comparison; tool is not built");
            return;
        };

        let fixture = workspace_path("crates/jxl-codec/tests/generated/icc_rec2020_lossless.jxl");
        let output = unique_temp_path("jxl-decode-reference", "ppm");
        let djxl_output = Command::new(&djxl)
            .arg(&fixture)
            .arg(&output)
            .arg("--quiet")
            .output()
            .unwrap();
        assert!(
            djxl_output.status.success(),
            "reference djxl failed for {}: {}",
            fixture.display(),
            String::from_utf8_lossy(&djxl_output.stderr)
        );

        let reference = std::fs::read(&output).unwrap();
        let _ = std::fs::remove_file(&output);
        let reference = parse_ppm_rgb(&reference);
        let bytes = std::fs::read(&fixture).unwrap();
        let decoded = decode(&bytes).unwrap();

        assert_eq!(decoded.width, reference.width);
        assert_eq!(decoded.height, reference.height);
        assert_eq!(decoded.color_channels, 3);
        assert_eq!(decoded.bit_depth, 16);
        assert_eq!(decoded_samples_u16(&decoded), reference.samples);
    }

    #[test]
    fn decode_generated_squeeze_pixels_match_reference_djxl_when_available() {
        let (Some(cjxl), Some(djxl)) = (reference_cjxl(), reference_djxl()) else {
            eprintln!(
                "skipping public decode generated squeeze comparison; reference tools are not built"
            );
            return;
        };

        let input = unique_temp_path("jxl-decode-squeeze-source", "ppm");
        let encoded = unique_temp_path("jxl-decode-squeeze", "jxl");
        let reference_output = unique_temp_path("jxl-decode-squeeze-reference", "ppm");
        write_progressive_squeeze_source_ppm(&input);

        let cjxl_output = Command::new(&cjxl)
            .arg(&input)
            .arg(&encoded)
            .args(["-d", "0", "-m", "1", "-p", "--container=0", "--quiet"])
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&input);
        assert!(
            cjxl_output.status.success(),
            "reference cjxl failed: {}",
            String::from_utf8_lossy(&cjxl_output.stderr)
        );

        let djxl_output = Command::new(&djxl)
            .arg(&encoded)
            .arg(&reference_output)
            .arg("--quiet")
            .output()
            .unwrap();
        assert!(
            djxl_output.status.success(),
            "reference djxl failed: {}",
            String::from_utf8_lossy(&djxl_output.stderr)
        );

        let reference = std::fs::read(&reference_output).unwrap();
        let reference = parse_ppm_rgb(&reference);
        let encoded_bytes = std::fs::read(&encoded).unwrap();
        let _ = std::fs::remove_file(&encoded);
        let _ = std::fs::remove_file(&reference_output);
        let decoded = decode(&encoded_bytes).unwrap();

        assert_eq!(decoded.width, reference.width);
        assert_eq!(decoded.height, reference.height);
        assert_eq!(decoded.color_channels, 3);
        assert_eq!(decoded.alpha, None);
        assert_eq!(decoded.bit_depth, 8);
        assert_eq!(decoded_samples_u16(&decoded), reference.samples);
    }

    #[test]
    fn decode_generated_alpha_pixels_match_reference_djxl_when_available() {
        let (Some(cjxl), Some(djxl)) = (reference_cjxl(), reference_djxl()) else {
            eprintln!(
                "skipping public decode generated alpha comparison; reference tools are not built"
            );
            return;
        };

        let input = unique_temp_path("jxl-decode-alpha-source", "pam");
        let encoded = unique_temp_path("jxl-decode-alpha", "jxl");
        let reference_output = unique_temp_path("jxl-decode-alpha-reference", "pam");
        write_alpha_source_pam(&input);

        let cjxl_output = Command::new(&cjxl)
            .arg(&input)
            .arg(&encoded)
            .args(["-d", "0", "-m", "1", "--container=0", "--quiet"])
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&input);
        assert!(
            cjxl_output.status.success(),
            "reference cjxl failed: {}",
            String::from_utf8_lossy(&cjxl_output.stderr)
        );

        let djxl_output = Command::new(&djxl)
            .arg(&encoded)
            .arg(&reference_output)
            .arg("--quiet")
            .output()
            .unwrap();
        assert!(
            djxl_output.status.success(),
            "reference djxl failed: {}",
            String::from_utf8_lossy(&djxl_output.stderr)
        );

        let reference = std::fs::read(&reference_output).unwrap();
        let reference = parse_pam_rgba(&reference);
        let encoded_bytes = std::fs::read(&encoded).unwrap();
        let _ = std::fs::remove_file(&encoded);
        let _ = std::fs::remove_file(&reference_output);
        let decoded = decode(&encoded_bytes).unwrap();

        assert_eq!(decoded.width, reference.width);
        assert_eq!(decoded.height, reference.height);
        assert_eq!(decoded.color_channels, 3);
        assert_eq!(
            decoded.alpha,
            Some(AlphaInfo {
                bit_depth: 8,
                premultiplied: false,
            })
        );
        assert_eq!(decoded.bit_depth, 8);
        assert_eq!(decoded_samples_u16(&decoded), reference.samples);
    }

    #[test]
    fn decode_rgba8_expands_gray_fixture() {
        let bytes = std::fs::read(workspace_path(
            "reference/libjxl/testdata/jxl/pq_gradient.jxl",
        ))
        .unwrap();
        let raw = decode(&bytes).unwrap();
        let rgba = decode_rgba8(&bytes).unwrap();

        assert_eq!(rgba.width, 1088);
        assert_eq!(rgba.height, 64);
        assert_eq!(rgba.pixels.len(), 1088 * 64 * 4);
        let PixelData::U16(samples) = raw.pixels else {
            panic!("expected 16-bit raw samples");
        };
        let gray = scale_sample_to_u8(u32::from(samples[0]), raw.bit_depth);
        assert_eq!(&rgba.pixels[..4], &[gray, gray, gray, 255]);
    }

    #[test]
    fn decode_rgba8_converts_rgb_fixture() {
        let bytes = std::fs::read(workspace_path(
            "crates/jxl-codec/tests/generated/icc_rec2020_lossless.jxl",
        ))
        .unwrap();
        let raw = decode(&bytes).unwrap();
        let rgba = decode_rgba8(&bytes).unwrap();

        assert_eq!(rgba.width, 64);
        assert_eq!(rgba.height, 64);
        assert_eq!(rgba.pixels.len(), 64 * 64 * 4);
        let PixelData::U16(samples) = raw.pixels else {
            panic!("expected 16-bit raw samples");
        };
        assert_eq!(
            &rgba.pixels[..4],
            &[
                scale_sample_to_u8(u32::from(samples[0]), raw.bit_depth),
                scale_sample_to_u8(u32::from(samples[1]), raw.bit_depth),
                scale_sample_to_u8(u32::from(samples[2]), raw.bit_depth),
                255,
            ]
        );
    }

    #[test]
    fn decode_rgba16_expands_gray_fixture() {
        let bytes = std::fs::read(workspace_path(
            "reference/libjxl/testdata/jxl/pq_gradient.jxl",
        ))
        .unwrap();
        let raw = decode(&bytes).unwrap();
        let rgba = decode_rgba16(&bytes).unwrap();

        assert_eq!(rgba.width, 1088);
        assert_eq!(rgba.height, 64);
        assert_eq!(rgba.pixels.len(), 1088 * 64 * 4);
        let PixelData::U16(samples) = raw.pixels else {
            panic!("expected 16-bit raw samples");
        };
        let gray = scale_sample_to_u16(u32::from(samples[0]), raw.bit_depth);
        assert_eq!(&rgba.pixels[..4], &[gray, gray, gray, u16::MAX]);
    }

    #[test]
    fn decode_rgba16_converts_rgb_fixture() {
        let bytes = std::fs::read(workspace_path(
            "crates/jxl-codec/tests/generated/icc_rec2020_lossless.jxl",
        ))
        .unwrap();
        let raw = decode(&bytes).unwrap();
        let rgba = decode_rgba16(&bytes).unwrap();

        assert_eq!(rgba.width, 64);
        assert_eq!(rgba.height, 64);
        assert_eq!(rgba.pixels.len(), 64 * 64 * 4);
        let PixelData::U16(samples) = raw.pixels else {
            panic!("expected 16-bit raw samples");
        };
        assert_eq!(
            &rgba.pixels[..4],
            &[
                scale_sample_to_u16(u32::from(samples[0]), raw.bit_depth),
                scale_sample_to_u16(u32::from(samples[1]), raw.bit_depth),
                scale_sample_to_u16(u32::from(samples[2]), raw.bit_depth),
                u16::MAX,
            ]
        );
    }

    #[test]
    fn decode_rgba8_preserves_alpha_when_available() {
        let (Some(cjxl), Some(djxl)) = (reference_cjxl(), reference_djxl()) else {
            eprintln!("skipping rgba8 alpha comparison; reference tools are not built");
            return;
        };

        let input = unique_temp_path("jxl-rgba8-alpha-source", "pam");
        let encoded = unique_temp_path("jxl-rgba8-alpha", "jxl");
        let reference_output = unique_temp_path("jxl-rgba8-alpha-reference", "pam");
        write_alpha_source_pam(&input);

        let cjxl_output = Command::new(&cjxl)
            .arg(&input)
            .arg(&encoded)
            .args(["-d", "0", "-m", "1", "--container=0", "--quiet"])
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&input);
        assert!(
            cjxl_output.status.success(),
            "reference cjxl failed: {}",
            String::from_utf8_lossy(&cjxl_output.stderr)
        );

        let djxl_output = Command::new(&djxl)
            .arg(&encoded)
            .arg(&reference_output)
            .arg("--quiet")
            .output()
            .unwrap();
        assert!(
            djxl_output.status.success(),
            "reference djxl failed: {}",
            String::from_utf8_lossy(&djxl_output.stderr)
        );

        let reference = std::fs::read(&reference_output).unwrap();
        let reference = parse_pam_rgba(&reference);
        let encoded_bytes = std::fs::read(&encoded).unwrap();
        let _ = std::fs::remove_file(&encoded);
        let _ = std::fs::remove_file(&reference_output);
        let rgba = decode_rgba8(&encoded_bytes).unwrap();

        assert_eq!(rgba.width, reference.width);
        assert_eq!(rgba.height, reference.height);
        assert_eq!(
            rgba.pixels,
            reference
                .samples
                .iter()
                .copied()
                .map(|sample| sample as u8)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn scales_samples_to_u8_with_rounding() {
        assert_eq!(scale_sample_to_u8(0, 16), 0);
        assert_eq!(scale_sample_to_u8(65_535, 16), 255);
        assert_eq!(scale_sample_to_u8(32_768, 16), 128);
        assert_eq!(scale_sample_to_u8(128, 8), 128);
        assert_eq!(scale_sample_to_u8(1, 1), 255);
        assert_eq!(scale_sample_to_u16(0, 16), 0);
        assert_eq!(scale_sample_to_u16(65_535, 16), 65_535);
        assert_eq!(scale_sample_to_u16(128, 8), 32_896);
        assert_eq!(scale_sample_to_u16(1, 1), 65_535);
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct PpmRgb {
        width: u32,
        height: u32,
        samples: Vec<u16>,
    }

    fn parse_ppm_rgb(bytes: &[u8]) -> PpmRgb {
        let (magic, offset) = netpbm_token(bytes, 0);
        assert_eq!(magic, b"P6");
        let (width, offset) = netpbm_token(bytes, offset);
        let (height, offset) = netpbm_token(bytes, offset);
        let (maxval, mut offset) = netpbm_token(bytes, offset);
        let maxval = parse_ascii_u32(maxval);
        assert!(matches!(maxval, 255 | 65535));
        assert!(
            offset < bytes.len() && bytes[offset].is_ascii_whitespace(),
            "PPM header was not followed by binary sample data"
        );
        offset += 1;

        let width = parse_ascii_u32(width);
        let height = parse_ascii_u32(height);
        let bytes_per_sample = if maxval > 255 { 2 } else { 1 };
        let expected_bytes = width as usize * height as usize * 3 * bytes_per_sample;
        let data = &bytes[offset..];
        assert_eq!(data.len(), expected_bytes);
        let samples = if bytes_per_sample == 2 {
            data.chunks_exact(2)
                .map(|chunk| u16::from_be_bytes([chunk[0], chunk[1]]))
                .collect()
        } else {
            data.iter().copied().map(u16::from).collect()
        };
        PpmRgb {
            width,
            height,
            samples,
        }
    }

    fn parse_pam_rgba(bytes: &[u8]) -> PpmRgb {
        let header_end = bytes
            .windows(7)
            .position(|window| window == b"ENDHDR\n")
            .map(|index| index + 7)
            .expect("PAM header did not contain ENDHDR");
        let header = std::str::from_utf8(&bytes[..header_end]).unwrap();
        assert!(header.starts_with("P7\n"));
        let mut width = None;
        let mut height = None;
        let mut depth = None;
        let mut maxval = None;
        let mut tupltype = None;
        for line in header.lines() {
            let mut fields = line.splitn(2, ' ');
            match (fields.next(), fields.next()) {
                (Some("WIDTH"), Some(value)) => width = Some(value.parse::<u32>().unwrap()),
                (Some("HEIGHT"), Some(value)) => height = Some(value.parse::<u32>().unwrap()),
                (Some("DEPTH"), Some(value)) => depth = Some(value.parse::<u32>().unwrap()),
                (Some("MAXVAL"), Some(value)) => maxval = Some(value.parse::<u32>().unwrap()),
                (Some("TUPLTYPE"), Some(value)) => tupltype = Some(value),
                _ => {}
            }
        }
        assert_eq!(depth, Some(4));
        assert_eq!(maxval, Some(255));
        assert_eq!(tupltype, Some("RGB_ALPHA"));
        let width = width.unwrap();
        let height = height.unwrap();
        let data = &bytes[header_end..];
        assert_eq!(data.len(), width as usize * height as usize * 4);
        PpmRgb {
            width,
            height,
            samples: data.iter().copied().map(u16::from).collect(),
        }
    }

    fn netpbm_token(bytes: &[u8], mut offset: usize) -> (&[u8], usize) {
        loop {
            while offset < bytes.len() && bytes[offset].is_ascii_whitespace() {
                offset += 1;
            }
            if offset < bytes.len() && bytes[offset] == b'#' {
                while offset < bytes.len() && bytes[offset] != b'\n' {
                    offset += 1;
                }
                continue;
            }
            break;
        }
        let start = offset;
        while offset < bytes.len() && !bytes[offset].is_ascii_whitespace() {
            offset += 1;
        }
        (&bytes[start..offset], offset)
    }

    fn parse_ascii_u32(bytes: &[u8]) -> u32 {
        std::str::from_utf8(bytes).unwrap().parse().unwrap()
    }

    fn decoded_samples_u16(image: &DecodedImage) -> Vec<u16> {
        match &image.pixels {
            PixelData::U8(samples) => samples.iter().copied().map(u16::from).collect(),
            PixelData::U16(samples) => samples.clone(),
        }
    }

    fn write_progressive_squeeze_source_ppm(path: &Path) {
        let width = 128u32;
        let height = 128u32;
        let mut state = 2u32;
        let mut bytes = format!("P6\n{width} {height}\n255\n").into_bytes();
        for _ in 0..width * height * 3 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            bytes.push((state >> 24) as u8);
        }
        std::fs::write(path, bytes).unwrap();
    }

    fn write_alpha_source_pam(path: &Path) {
        let width = 64u32;
        let height = 64u32;
        let mut state = 3u32;
        let mut bytes = format!(
            "P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n"
        )
        .into_bytes();
        for _ in 0..width * height * 4 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            bytes.push((state >> 24) as u8);
        }
        std::fs::write(path, bytes).unwrap();
    }

    fn workspace_path(relative: impl AsRef<Path>) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(relative)
    }

    fn reference_cjxl() -> Option<PathBuf> {
        if let Ok(path) = std::env::var("JXL_RS_REFERENCE_CJXL") {
            let path = PathBuf::from(path);
            if path.is_file() {
                return Some(path);
            }
        }

        let default = workspace_path("reference/libjxl/build-rs-oracle/tools/cjxl");
        default.is_file().then_some(default)
    }

    fn reference_djxl() -> Option<PathBuf> {
        if let Ok(path) = std::env::var("JXL_RS_REFERENCE_DJXL") {
            let path = PathBuf::from(path);
            if path.is_file() {
                return Some(path);
            }
        }

        let default = workspace_path("reference/libjxl/build-rs-oracle/tools/djxl");
        default.is_file().then_some(default)
    }

    fn unique_temp_path(prefix: &str, extension: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "{prefix}-{}-{nanos}.{extension}",
            std::process::id()
        ))
    }
}
