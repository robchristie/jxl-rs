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
    SqueezeParams, TocEntry, ToneMapping, TransferFunction, TransformId,
    VarDctBlockContextMapMetadata, VarDctColorCorrelationMetadata, VarDctDcDequantMetadata,
    VarDctDcGroupCursorMetadata, VarDctDcGroupMetadata, VarDctDcGroupPayloadMetadata,
    VarDctDecodePlan, VarDctFrameMetadata, VarDctGlobalCursorMetadata, VarDctGlobalMetadata,
    VarDctGroupMetadata, VarDctGroupPayloadMetadata, VarDctGroupSectionMetadata,
    VarDctPassGroupPayloadMetadata, VarDctPassGroupSectionMetadata, VarDctQuantizerMetadata,
    VarDctSectionMetadata, VarDctSectionPayloadMetadata, WeightedPredictorHeader, WhitePoint,
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
    pub first_frame_vardct: Option<VarDctFrameMetadata>,
    pub first_frame_vardct_plan: Option<VarDctDecodePlan>,
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
    /// Integer sample bit depth for this decoded channel.
    ///
    /// This usually equals [`DecodedChannels::bit_depth`] for color channels,
    /// but JPEG XL extra channels may use their own bit depth.
    pub bit_depth: u32,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decoder {
    options: DecodeOptions,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DecodeOptions {
    pub output: DecodeOutput,
    pub roi: Option<Rect>,
    /// Selects exactly one VarDCT AC pass for public RGB/RGBA output.
    ///
    /// `None` uses final VarDCT reconstruction. `Some(pass)` is intended for
    /// progressive preview-style output and does not merge earlier or later AC
    /// passes. Modular decode rejects this option.
    pub vardct_pass: Option<usize>,
    pub threads: ThreadingMode,
    pub memory_limit: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodeOutput {
    #[default]
    Channels,
    Interleaved,
    Rgba8,
    Rgba16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThreadingMode {
    #[default]
    Auto,
    Single,
    Threads(usize),
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder {
    pub fn new() -> Self {
        Self {
            options: DecodeOptions::default(),
        }
    }

    pub fn with_options(options: DecodeOptions) -> Self {
        Self { options }
    }

    pub fn options(&self) -> &DecodeOptions {
        &self.options
    }

    pub fn options_mut(&mut self) -> &mut DecodeOptions {
        &mut self.options
    }

    pub fn output(mut self, output: DecodeOutput) -> Self {
        self.options.output = output;
        self
    }

    pub fn roi(mut self, roi: Rect) -> Self {
        self.options.roi = Some(roi);
        self
    }

    pub fn threads(mut self, threads: ThreadingMode) -> Self {
        self.options.threads = threads;
        self
    }

    pub fn memory_limit(mut self, bytes: usize) -> Self {
        self.options.memory_limit = Some(bytes);
        self
    }

    pub fn vardct_pass(mut self, pass: usize) -> Self {
        self.options.vardct_pass = Some(pass);
        self
    }

    /// Decodes raw image channels.
    ///
    /// If [`Decoder::roi`] is set, the returned [`DecodedChannels::width`] and
    /// [`DecodedChannels::height`] are the requested region dimensions. Channel
    /// samples are ROI-local: sample `(0, 0)` corresponds to the requested
    /// image-space coordinate `(roi.x, roi.y)`.
    ///
    /// [`DecodedChannels::bit_depth`] is the main image bit depth. Individual
    /// channels also expose [`DecodedChannel::bit_depth`] because JPEG XL extra
    /// channels, including alpha, may use a different sample depth.
    ///
    /// Modular still images return decoded integer channels. Supported VarDCT
    /// still images return reconstructed 8-bit sRGB RGB channels, not original
    /// codestream channels. VarDCT extra channels are exposed when their
    /// modular AC side streams are decoded.
    pub fn decode_channels(&self, input: &[u8]) -> Result<DecodedChannels> {
        self.validate_shared_options()?;
        decode_channels_buffered(input, self.codec_config(), self.options.vardct_pass)
    }

    /// Decodes an interleaved image.
    ///
    /// Modular still images return their decoded integer samples, preserving
    /// the decoded sample bit depth. The interleaved output includes color
    /// channels plus the first alpha channel when present; other extra channels
    /// remain available through [`Decoder::decode_channels`]. Supported VarDCT
    /// still images return 8-bit sRGB RGB samples plus decoded alpha when
    /// present. Pixel output applies JPEG XL orientation metadata.
    ///
    /// VarDCT output is currently a reconstruction convenience path: it does
    /// not yet apply full JPEG XL color management.
    /// VarDCT ROI is implemented as post-reconstruction cropping and may decode
    /// the full frame internally. Unreconstructed VarDCT layouts return
    /// [`Error::Unsupported`].
    pub fn decode(&self, input: &[u8]) -> Result<DecodedImage> {
        self.validate_shared_options()?;
        decode_buffered(input, self.codec_config(), self.options.vardct_pass)
    }

    /// Decodes to interleaved straight-alpha RGBA8.
    ///
    /// Modular still images are decoded through the raw-channel path and then
    /// scaled or expanded to RGBA8. If the codestream marks alpha as
    /// associated/premultiplied, color samples are unpremultiplied for this
    /// presentation output. Non-alpha extra channels are ignored. Supported
    /// VarDCT still images return sRGB RGBA8 with decoded alpha when present.
    /// Pixel output applies JPEG XL orientation metadata.
    ///
    /// VarDCT output is currently a reconstruction convenience path: it does
    /// not yet apply full JPEG XL color management.
    /// VarDCT ROI is implemented as post-reconstruction cropping and may decode
    /// the full frame internally. Unreconstructed VarDCT layouts return
    /// [`Error::Unsupported`].
    pub fn decode_rgba8(&self, input: &[u8]) -> Result<RgbaImage> {
        self.validate_shared_options()?;
        decode_rgba8_buffered(input, self.codec_config(), self.options.vardct_pass)
    }

    /// Decodes to interleaved straight-alpha RGBA16.
    ///
    /// Modular still images are decoded through the raw-channel path and then
    /// scaled or expanded to RGBA16. If the codestream marks alpha as
    /// associated/premultiplied, color samples are unpremultiplied for this
    /// presentation output. Non-alpha extra channels are ignored. Supported
    /// VarDCT still images return sRGB RGBA16 with decoded alpha when present.
    /// Pixel output applies JPEG XL orientation metadata.
    ///
    /// VarDCT output is currently a reconstruction convenience path: it does
    /// not yet apply full JPEG XL color management.
    /// VarDCT ROI is implemented as post-reconstruction cropping and may decode
    /// the full frame internally. Unreconstructed VarDCT layouts return
    /// [`Error::Unsupported`].
    pub fn decode_rgba16(&self, input: &[u8]) -> Result<Rgba16Image> {
        self.validate_shared_options()?;
        decode_rgba16_buffered(input, self.codec_config(), self.options.vardct_pass)
    }

    fn validate_shared_options(&self) -> Result<()> {
        if self.options.memory_limit.is_some() {
            return Err(Error::Unsupported("memory-limited decode"));
        }
        if self.options.threads == ThreadingMode::Threads(0) {
            return Err(Error::Unsupported("zero decoder threads"));
        }
        Ok(())
    }

    fn codec_config(&self) -> jxl_codec::DecodeConfig {
        jxl_codec::DecodeConfig {
            modular_group_execution: match self.options.threads {
                ThreadingMode::Auto | ThreadingMode::Single => {
                    jxl_codec::ModularGroupExecution::Serial
                }
                ThreadingMode::Threads(threads) => {
                    jxl_codec::ModularGroupExecution::RequestedThreads(threads)
                }
            },
            region: self.options.roi.map(|roi| jxl_codec::ImageRegion {
                x: roi.x,
                y: roi.y,
                width: roi.width,
                height: roi.height,
            }),
        }
    }
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
        first_frame_vardct: codestream.first_frame_vardct,
        first_frame_vardct_plan: codestream.first_frame_vardct_plan,
        boxes: extracted
            .container
            .map(|container| container.boxes)
            .unwrap_or_default(),
    })
}

pub fn decode_channels(input: &[u8]) -> Result<DecodedChannels> {
    Decoder::new().decode_channels(input)
}

pub fn decode(input: &[u8]) -> Result<DecodedImage> {
    Decoder::new().decode(input)
}

pub fn decode_rgba8(input: &[u8]) -> Result<RgbaImage> {
    Decoder::new().decode_rgba8(input)
}

pub fn decode_rgba16(input: &[u8]) -> Result<Rgba16Image> {
    Decoder::new().decode_rgba16(input)
}

fn decode_channels_buffered(
    input: &[u8],
    config: jxl_codec::DecodeConfig,
    vardct_pass: Option<usize>,
) -> Result<DecodedChannels> {
    let (_, codestream) = parse_file_for_public_pixel_decode(input, config)?;
    decode_channels_codestream(codestream, config.region, vardct_pass)
}

fn parse_file_for_public_pixel_decode(
    input: &[u8],
    config: jxl_codec::DecodeConfig,
) -> Result<(jxl_codec::ExtractedCodestream, jxl_codec::Codestream)> {
    if config.region.is_some() {
        let parsed = jxl_codec::parse_file(input)?;
        if first_frame_encoding(&parsed.1)? == FrameEncoding::VarDct {
            return Ok(parsed);
        }
    }
    jxl_codec::parse_file_with_config(input, config)
}

fn first_frame_encoding(codestream: &jxl_codec::Codestream) -> Result<FrameEncoding> {
    if codestream.basic_info.have_animation {
        return Err(Error::Unsupported("animated image decode"));
    }
    Ok(codestream
        .first_frame
        .as_ref()
        .ok_or(Error::Unsupported("image has no decoded frame"))?
        .encoding)
}

fn first_frame_vardct_plan(codestream: &jxl_codec::Codestream) -> Result<&VarDctDecodePlan> {
    codestream
        .first_frame_vardct_plan
        .as_ref()
        .ok_or(Error::Unsupported("VarDCT image reconstruction"))
}

fn decode_channels_codestream(
    codestream: jxl_codec::Codestream,
    region: Option<jxl_codec::ImageRegion>,
    vardct_pass: Option<usize>,
) -> Result<DecodedChannels> {
    if first_frame_encoding(&codestream)? == FrameEncoding::VarDct {
        return decode_vardct_channels_codestream(&codestream, region, vardct_pass);
    }
    reject_vardct_pass_for_non_vardct(vardct_pass)?;
    let modular = codestream
        .first_frame_modular
        .as_ref()
        .ok_or(Error::Unsupported("modular image metadata"))?;
    let image = match modular.image.as_ref() {
        Some(image) => image,
        None => {
            return Err(modular.image_error.clone().unwrap_or(if region.is_some() {
                Error::Unsupported("region-of-interest raw channel decode")
            } else {
                Error::Unsupported("modular pixel reconstruction")
            }));
        }
    };
    let color_channels = codestream.metadata.num_color_channels() as usize;
    let bit_depth = codestream.metadata.bit_depth.bits_per_sample;
    if codestream.metadata.bit_depth.floating_point_sample {
        return Err(Error::Unsupported("floating-point sample output"));
    }
    if bit_depth > 16 {
        return Err(Error::Unsupported("integer sample depths above 16 bits"));
    }
    let channel_bit_depths = decoded_channel_bit_depths(&codestream.metadata, color_channels)?;
    let channels = image
        .channels
        .iter()
        .enumerate()
        .map(|(index, channel)| {
            let channel_bit_depth =
                channel_bit_depths
                    .get(index)
                    .copied()
                    .ok_or(Error::InvalidCodestream(
                        "decoded channel missing bit-depth metadata",
                    ))?;
            let max_sample = max_sample_value(channel_bit_depth)?;
            decode_channel(
                image.width,
                image.height,
                channel,
                channel_bit_depth,
                max_sample,
            )
        })
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

fn decode_buffered(
    input: &[u8],
    config: jxl_codec::DecodeConfig,
    vardct_pass: Option<usize>,
) -> Result<DecodedImage> {
    let (_, codestream) = parse_file_for_public_pixel_decode(input, config)?;
    if first_frame_encoding(&codestream)? == FrameEncoding::VarDct {
        let alpha_channel_index = raw_alpha_channel_index(&codestream.metadata)?;
        if alpha_channel_index.is_some() {
            let channels =
                decode_vardct_channels_codestream(&codestream, config.region, vardct_pass)?;
            return decode_buffered_channels(channels, alpha_channel_index);
        }
        reject_vardct_alpha_output(&codestream.metadata)?;
        let orientation = codestream.metadata.orientation;
        let image = decoded_image_from_vardct_srgb8(vardct_srgb8_image_from_codestream(
            &codestream,
            config.region,
            vardct_pass,
        )?)?;
        return orient_decoded_image(image, orientation);
    }
    reject_vardct_pass_for_non_vardct(vardct_pass)?;

    decode_buffered_codestream(codestream)
}

fn decode_buffered_codestream(codestream: jxl_codec::Codestream) -> Result<DecodedImage> {
    let orientation = codestream.metadata.orientation;
    let alpha_channel_index = raw_alpha_channel_index(&codestream.metadata)?;
    let channels = decode_channels_codestream(codestream, None, None)?;
    orient_decoded_image(
        decode_buffered_channels(channels, alpha_channel_index)?,
        orientation,
    )
}

fn decode_buffered_channels(
    channels: DecodedChannels,
    alpha_channel_index: Option<usize>,
) -> Result<DecodedImage> {
    let alpha = decode_interleaved_alpha(&channels, alpha_channel_index)?;
    let output_channel_indices = interleaved_channel_indices(&channels, alpha_channel_index)?;
    if output_channel_indices.iter().any(|&index| {
        channels.channels[index].width != channels.width
            || channels.channels[index].height != channels.height
    }) {
        return Err(Error::Unsupported("subsampled raw channel output"));
    }
    if output_channel_indices
        .iter()
        .any(|&index| channels.channels[index].bit_depth != channels.bit_depth)
    {
        return Err(Error::Unsupported("mixed bit-depth interleaved output"));
    }

    if channels.bit_depth <= 8 {
        Ok(DecodedImage {
            width: channels.width,
            height: channels.height,
            color_channels: channels.color_channels,
            alpha,
            bit_depth: channels.bit_depth,
            pixels: PixelData::U8(interleave_channel_u8(&channels, &output_channel_indices)?),
        })
    } else {
        Ok(DecodedImage {
            width: channels.width,
            height: channels.height,
            color_channels: channels.color_channels,
            alpha,
            bit_depth: channels.bit_depth,
            pixels: PixelData::U16(interleave_channel_u16(&channels, &output_channel_indices)?),
        })
    }
}

fn decode_rgba8_buffered(
    input: &[u8],
    config: jxl_codec::DecodeConfig,
    vardct_pass: Option<usize>,
) -> Result<RgbaImage> {
    let (_, codestream) = parse_file_for_public_pixel_decode(input, config)?;
    if first_frame_encoding(&codestream)? == FrameEncoding::VarDct {
        let alpha_channel_index = raw_alpha_channel_index(&codestream.metadata)?;
        if alpha_channel_index.is_some() {
            let channels =
                decode_vardct_channels_codestream(&codestream, config.region, vardct_pass)?;
            return rgba8_from_decoded_channels(&channels, alpha_channel_index);
        }
        reject_vardct_alpha_output(&codestream.metadata)?;
        let orientation = codestream.metadata.orientation;
        let image = rgba8_from_vardct_srgb8(vardct_srgb8_image_from_codestream(
            &codestream,
            config.region,
            vardct_pass,
        )?)?;
        return orient_rgba8(image, orientation);
    }
    reject_vardct_pass_for_non_vardct(vardct_pass)?;

    rgba8_from_modular_codestream(codestream)
}

fn rgba8_from_modular_codestream(codestream: jxl_codec::Codestream) -> Result<RgbaImage> {
    let orientation = codestream.metadata.orientation;
    let alpha_channel_index = raw_alpha_channel_index(&codestream.metadata)?;
    let channels = decode_channels_codestream(codestream, None, None)?;
    orient_rgba8(
        rgba8_from_decoded_channels(&channels, alpha_channel_index)?,
        orientation,
    )
}

fn rgba8_from_decoded_channels(
    channels: &DecodedChannels,
    alpha_channel_index: Option<usize>,
) -> Result<RgbaImage> {
    let output_channel_indices = rgba_channel_indices(channels, alpha_channel_index)?;
    let pixels = rgba8_from_channel_indices(channels, &output_channel_indices)?;
    Ok(RgbaImage {
        width: channels.width,
        height: channels.height,
        pixels,
    })
}

fn rgba8_from_vardct_srgb8(image: jxl_codec::VarDctSrgb8Image) -> Result<RgbaImage> {
    let sample_count = vardct_srgb_sample_count(image.width, image.height)?;
    if image.pixels.len() != sample_count * 3 {
        return Err(Error::InvalidCodestream("decoded pixel count mismatch"));
    }

    let mut pixels = Vec::with_capacity(sample_count * 4);
    for rgb in image.pixels.chunks_exact(3) {
        pixels.extend_from_slice(rgb);
        pixels.push(255);
    }
    Ok(RgbaImage {
        width: image.width,
        height: image.height,
        pixels,
    })
}

fn decoded_image_from_vardct_srgb8(image: jxl_codec::VarDctSrgb8Image) -> Result<DecodedImage> {
    let sample_count = vardct_srgb_sample_count(image.width, image.height)?;
    if image.pixels.len() != sample_count * 3 {
        return Err(Error::InvalidCodestream("decoded pixel count mismatch"));
    }

    Ok(DecodedImage {
        width: image.width,
        height: image.height,
        color_channels: 3,
        alpha: None,
        bit_depth: 8,
        pixels: PixelData::U8(image.pixels),
    })
}

fn decoded_channels_from_vardct_srgb8(
    image: jxl_codec::VarDctSrgb8Image,
) -> Result<DecodedChannels> {
    let sample_count = vardct_srgb_sample_count(image.width, image.height)?;
    let mut channels = [
        Vec::with_capacity(sample_count),
        Vec::with_capacity(sample_count),
        Vec::with_capacity(sample_count),
    ];
    for pixel in image.pixels.chunks_exact(3) {
        channels[0].push(pixel[0]);
        channels[1].push(pixel[1]);
        channels[2].push(pixel[2]);
    }

    Ok(DecodedChannels {
        width: image.width,
        height: image.height,
        color_channels: 3,
        alpha: None,
        bit_depth: 8,
        channels: channels
            .into_iter()
            .map(|samples| DecodedChannel {
                width: image.width,
                height: image.height,
                hshift: 0,
                vshift: 0,
                bit_depth: 8,
                samples: ChannelData::U8(samples),
            })
            .collect(),
    })
}

fn decoded_channels_from_vardct_srgb16(
    image: jxl_codec::VarDctSrgb16Image,
) -> Result<DecodedChannels> {
    let sample_count = vardct_srgb_sample_count(image.width, image.height)?;
    let mut channels = [
        Vec::with_capacity(sample_count),
        Vec::with_capacity(sample_count),
        Vec::with_capacity(sample_count),
    ];
    for pixel in image.pixels.chunks_exact(3) {
        channels[0].push(pixel[0]);
        channels[1].push(pixel[1]);
        channels[2].push(pixel[2]);
    }

    Ok(DecodedChannels {
        width: image.width,
        height: image.height,
        color_channels: 3,
        alpha: None,
        bit_depth: 16,
        channels: channels
            .into_iter()
            .map(|samples| DecodedChannel {
                width: image.width,
                height: image.height,
                hshift: 0,
                vshift: 0,
                bit_depth: 16,
                samples: ChannelData::U16(samples),
            })
            .collect(),
    })
}

fn decode_vardct_channels_codestream(
    codestream: &jxl_codec::Codestream,
    region: Option<jxl_codec::ImageRegion>,
    vardct_pass: Option<usize>,
) -> Result<DecodedChannels> {
    let orientation = codestream.metadata.orientation;
    let image = vardct_srgb8_image_from_codestream(codestream, region, vardct_pass)?;
    let mut channels = decoded_channels_from_vardct_srgb8(image)?;
    append_vardct_extra_channels(&mut channels, codestream, region, vardct_pass)?;
    orient_decoded_channels(channels, orientation)
}

fn decode_vardct_channels_codestream_rgb16(
    codestream: &jxl_codec::Codestream,
    region: Option<jxl_codec::ImageRegion>,
    vardct_pass: Option<usize>,
) -> Result<DecodedChannels> {
    let orientation = codestream.metadata.orientation;
    let image = vardct_srgb16_image_from_codestream(codestream, region, vardct_pass)?;
    let mut channels = decoded_channels_from_vardct_srgb16(image)?;
    append_vardct_extra_channels(&mut channels, codestream, region, vardct_pass)?;
    orient_decoded_channels(channels, orientation)
}

fn append_vardct_extra_channels(
    channels: &mut DecodedChannels,
    codestream: &jxl_codec::Codestream,
    region: Option<jxl_codec::ImageRegion>,
    vardct_pass: Option<usize>,
) -> Result<()> {
    if codestream.metadata.extra_channels.is_empty() {
        return Ok(());
    }
    if vardct_pass.is_some() {
        return Err(Error::Unsupported(
            "VarDCT extra-channel progressive pass output",
        ));
    }
    if channels.color_channels != codestream.metadata.num_color_channels() as usize {
        return Err(Error::Unsupported("VarDCT non-RGB raw channel output"));
    }

    let mut extra_channels = vardct_extra_channels_from_ac(codestream)?;
    if let Some(region) = region {
        extra_channels = extra_channels
            .into_iter()
            .map(|channel| crop_decoded_channel(channel, region))
            .collect::<Result<Vec<_>>>()?;
    }
    channels.alpha = raw_alpha_info(&codestream.metadata)?;
    channels.channels.extend(extra_channels);
    Ok(())
}

struct VarDctExtraChannelPlane {
    width: u32,
    height: u32,
    bit_depth: u32,
    samples: Vec<i32>,
    filled: Vec<bool>,
}

fn vardct_extra_channels_from_ac(
    codestream: &jxl_codec::Codestream,
) -> Result<Vec<DecodedChannel>> {
    let plan = first_frame_vardct_plan(codestream)?;
    let color_channels = codestream.metadata.num_color_channels() as usize;
    let channel_bit_depths = decoded_channel_bit_depths(&codestream.metadata, color_channels)?;
    let mut planes = codestream
        .metadata
        .extra_channels
        .iter()
        .enumerate()
        .map(|(extra_index, extra)| {
            let bit_depth = *channel_bit_depths.get(color_channels + extra_index).ok_or(
                Error::InvalidCodestream("decoded channel missing bit-depth metadata"),
            )?;
            let shift = extra.dim_shift;
            let width = codestream.basic_info.width.div_ceil(1u32 << shift);
            let height = codestream.basic_info.height.div_ceil(1u32 << shift);
            let sample_count = (width as usize)
                .checked_mul(height as usize)
                .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
            Ok(VarDctExtraChannelPlane {
                width,
                height,
                bit_depth,
                samples: vec![0; sample_count],
                filled: vec![false; sample_count],
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut saw_extra_channel = vec![false; planes.len()];
    for group_metadata in &plan.ac_group_metadata {
        if !group_metadata.payload.modular_ac_channels.is_empty()
            && group_metadata.modular_ac.is_none()
        {
            if let Some(error) = &group_metadata.modular_ac_error {
                return Err(error.clone());
            }
            return Err(Error::Unsupported("VarDCT extra-channel output"));
        }

        let Some(group) = &group_metadata.modular_ac else {
            continue;
        };
        for channel in &group.channels {
            if channel.channel_index < color_channels {
                continue;
            }
            let extra_index = channel.channel_index - color_channels;
            let Some(plane) = planes.get_mut(extra_index) else {
                return Err(Error::InvalidCodestream(
                    "decoded VarDCT extra channel index is out of range",
                ));
            };
            copy_vardct_extra_channel_chunk(plane, channel)?;
            saw_extra_channel[extra_index] = true;
        }
    }

    for saw in &saw_extra_channel {
        if !saw {
            return Err(Error::Unsupported("VarDCT extra-channel output"));
        }
    }

    planes
        .into_iter()
        .map(|plane| {
            if plane.filled.iter().any(|filled| !filled) {
                return Err(Error::Unsupported("VarDCT extra-channel output"));
            }
            let max_sample = max_sample_value(plane.bit_depth)?;
            decode_channel(
                codestream.basic_info.width,
                codestream.basic_info.height,
                &ModularImageChannel {
                    width: plane.width,
                    height: plane.height,
                    samples: plane.samples,
                },
                plane.bit_depth,
                max_sample,
            )
        })
        .collect()
}

fn copy_vardct_extra_channel_chunk(
    plane: &mut VarDctExtraChannelPlane,
    channel: &ModularDecodedChannel,
) -> Result<()> {
    let sample_count = (channel.width as usize)
        .checked_mul(channel.height as usize)
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    if channel.samples.len() != sample_count {
        return Err(Error::InvalidCodestream("decoded pixel count mismatch"));
    }
    let end_x = channel
        .x0
        .checked_add(channel.width)
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    let end_y = channel
        .y0
        .checked_add(channel.height)
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    if end_x > plane.width || end_y > plane.height {
        return Err(Error::InvalidCodestream(
            "decoded VarDCT extra channel exceeds image bounds",
        ));
    }

    for row in 0..channel.height as usize {
        let source_start = row
            .checked_mul(channel.width as usize)
            .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
        let source_end = source_start
            .checked_add(channel.width as usize)
            .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
        let target_start = ((channel.y0 as usize + row)
            .checked_mul(plane.width as usize)
            .and_then(|index| index.checked_add(channel.x0 as usize)))
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
        let target_end = target_start
            .checked_add(channel.width as usize)
            .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;

        if plane.filled[target_start..target_end]
            .iter()
            .any(|filled| *filled)
        {
            return Err(Error::InvalidCodestream(
                "overlapping VarDCT extra channel chunks",
            ));
        }
        plane.samples[target_start..target_end]
            .copy_from_slice(&channel.samples[source_start..source_end]);
        plane.filled[target_start..target_end].fill(true);
    }
    Ok(())
}

fn vardct_srgb8_image_from_codestream(
    codestream: &jxl_codec::Codestream,
    region: Option<jxl_codec::ImageRegion>,
    pass: Option<usize>,
) -> Result<jxl_codec::VarDctSrgb8Image> {
    let plan = first_frame_vardct_plan(codestream)?;
    let mut image = match pass {
        Some(pass) => jxl_codec::assemble_vardct_srgb8_image_for_pass(plan, pass)?,
        None => jxl_codec::assemble_vardct_srgb8_image(plan)?,
    }
    .ok_or(Error::Unsupported("VarDCT image reconstruction"))?;
    if let Some(region) = region {
        image = crop_vardct_srgb8(image, region)?;
    }
    Ok(image)
}

fn decode_rgba16_buffered(
    input: &[u8],
    config: jxl_codec::DecodeConfig,
    vardct_pass: Option<usize>,
) -> Result<Rgba16Image> {
    let (_, codestream) = parse_file_for_public_pixel_decode(input, config)?;
    if first_frame_encoding(&codestream)? == FrameEncoding::VarDct {
        let alpha_channel_index = raw_alpha_channel_index(&codestream.metadata)?;
        if alpha_channel_index.is_some() {
            let channels =
                decode_vardct_channels_codestream_rgb16(&codestream, config.region, vardct_pass)?;
            return rgba16_from_decoded_channels(&channels, alpha_channel_index);
        }
        reject_vardct_alpha_output(&codestream.metadata)?;
        let orientation = codestream.metadata.orientation;
        let image = rgba16_from_vardct_srgb16(vardct_srgb16_image_from_codestream(
            &codestream,
            config.region,
            vardct_pass,
        )?)?;
        return orient_rgba16(image, orientation);
    }
    reject_vardct_pass_for_non_vardct(vardct_pass)?;

    rgba16_from_modular_codestream(codestream)
}

fn rgba16_from_modular_codestream(codestream: jxl_codec::Codestream) -> Result<Rgba16Image> {
    let orientation = codestream.metadata.orientation;
    let alpha_channel_index = raw_alpha_channel_index(&codestream.metadata)?;
    let channels = decode_channels_codestream(codestream, None, None)?;
    orient_rgba16(
        rgba16_from_decoded_channels(&channels, alpha_channel_index)?,
        orientation,
    )
}

fn rgba16_from_decoded_channels(
    channels: &DecodedChannels,
    alpha_channel_index: Option<usize>,
) -> Result<Rgba16Image> {
    let output_channel_indices = rgba_channel_indices(channels, alpha_channel_index)?;
    let pixels = rgba16_from_channel_indices(channels, &output_channel_indices)?;
    Ok(Rgba16Image {
        width: channels.width,
        height: channels.height,
        pixels,
    })
}

fn rgba16_from_vardct_srgb16(image: jxl_codec::VarDctSrgb16Image) -> Result<Rgba16Image> {
    let sample_count = vardct_srgb_sample_count(image.width, image.height)?;
    if image.pixels.len() != sample_count * 3 {
        return Err(Error::InvalidCodestream("decoded pixel count mismatch"));
    }

    let mut pixels = Vec::with_capacity(sample_count * 4);
    for rgb in image.pixels.chunks_exact(3) {
        pixels.extend_from_slice(rgb);
        pixels.push(u16::MAX);
    }
    Ok(Rgba16Image {
        width: image.width,
        height: image.height,
        pixels,
    })
}

fn vardct_srgb16_image_from_codestream(
    codestream: &jxl_codec::Codestream,
    region: Option<jxl_codec::ImageRegion>,
    pass: Option<usize>,
) -> Result<jxl_codec::VarDctSrgb16Image> {
    let plan = first_frame_vardct_plan(codestream)?;
    let mut image = match pass {
        Some(pass) => jxl_codec::assemble_vardct_srgb16_image_for_pass(plan, pass)?,
        None => jxl_codec::assemble_vardct_srgb16_image(plan)?,
    }
    .ok_or(Error::Unsupported("VarDCT image reconstruction"))?;
    if let Some(region) = region {
        image = crop_vardct_srgb16(image, region)?;
    }
    Ok(image)
}

fn reject_vardct_pass_for_non_vardct(pass: Option<usize>) -> Result<()> {
    if pass.is_some() {
        return Err(Error::Unsupported("VarDCT progressive pass decode"));
    }
    Ok(())
}

fn reject_vardct_alpha_output(metadata: &ImageMetadata) -> Result<()> {
    if raw_alpha_channel_index(metadata)?.is_some() {
        return Err(Error::Unsupported("VarDCT alpha output"));
    }
    Ok(())
}

fn crop_vardct_srgb8(
    image: jxl_codec::VarDctSrgb8Image,
    region: jxl_codec::ImageRegion,
) -> Result<jxl_codec::VarDctSrgb8Image> {
    validate_decode_region(image.width, image.height, region)?;
    Ok(jxl_codec::VarDctSrgb8Image {
        width: region.width,
        height: region.height,
        pixels: crop_interleaved_u8(&image.pixels, image.width, 3, region)?,
    })
}

fn crop_vardct_srgb16(
    image: jxl_codec::VarDctSrgb16Image,
    region: jxl_codec::ImageRegion,
) -> Result<jxl_codec::VarDctSrgb16Image> {
    validate_decode_region(image.width, image.height, region)?;
    Ok(jxl_codec::VarDctSrgb16Image {
        width: region.width,
        height: region.height,
        pixels: crop_interleaved_u16(&image.pixels, image.width, 3, region)?,
    })
}

fn validate_decode_region(width: u32, height: u32, region: jxl_codec::ImageRegion) -> Result<()> {
    if region.width == 0 || region.height == 0 {
        return Err(Error::InvalidCodestream("empty decode region"));
    }
    let end_x = region
        .x
        .checked_add(region.width)
        .ok_or(Error::InvalidCodestream("decode region is outside image"))?;
    let end_y = region
        .y
        .checked_add(region.height)
        .ok_or(Error::InvalidCodestream("decode region is outside image"))?;
    if end_x > width || end_y > height {
        return Err(Error::InvalidCodestream("decode region is outside image"));
    }
    Ok(())
}

fn crop_interleaved_u8(
    samples: &[u8],
    width: u32,
    channels: usize,
    region: jxl_codec::ImageRegion,
) -> Result<Vec<u8>> {
    let output_len = (region.width as usize)
        .checked_mul(region.height as usize)
        .and_then(|samples| samples.checked_mul(channels))
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    let row_stride = (width as usize)
        .checked_mul(channels)
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    let x = (region.x as usize)
        .checked_mul(channels)
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    let copy_width = (region.width as usize)
        .checked_mul(channels)
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    let mut output = Vec::with_capacity(output_len);
    for y in region.y as usize..(region.y + region.height) as usize {
        let start = y
            .checked_mul(row_stride)
            .and_then(|start| start.checked_add(x))
            .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
        let end = start
            .checked_add(copy_width)
            .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
        let row = samples
            .get(start..end)
            .ok_or(Error::InvalidCodestream("decoded pixel count mismatch"))?;
        output.extend_from_slice(row);
    }
    Ok(output)
}

fn crop_interleaved_u16(
    samples: &[u16],
    width: u32,
    channels: usize,
    region: jxl_codec::ImageRegion,
) -> Result<Vec<u16>> {
    let output_len = (region.width as usize)
        .checked_mul(region.height as usize)
        .and_then(|samples| samples.checked_mul(channels))
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    let row_stride = (width as usize)
        .checked_mul(channels)
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    let x = (region.x as usize)
        .checked_mul(channels)
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    let copy_width = (region.width as usize)
        .checked_mul(channels)
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    let mut output = Vec::with_capacity(output_len);
    for y in region.y as usize..(region.y + region.height) as usize {
        let start = y
            .checked_mul(row_stride)
            .and_then(|start| start.checked_add(x))
            .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
        let end = start
            .checked_add(copy_width)
            .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
        let row = samples
            .get(start..end)
            .ok_or(Error::InvalidCodestream("decoded pixel count mismatch"))?;
        output.extend_from_slice(row);
    }
    Ok(output)
}

fn crop_decoded_channel(
    channel: DecodedChannel,
    region: jxl_codec::ImageRegion,
) -> Result<DecodedChannel> {
    if channel.hshift != 0 || channel.vshift != 0 {
        return Err(Error::Unsupported(
            "subsampled VarDCT extra-channel ROI output",
        ));
    }
    validate_decode_region(channel.width, channel.height, region)?;
    match channel.samples {
        ChannelData::U8(samples) => Ok(DecodedChannel {
            width: region.width,
            height: region.height,
            hshift: 0,
            vshift: 0,
            bit_depth: channel.bit_depth,
            samples: ChannelData::U8(crop_interleaved_u8(&samples, channel.width, 1, region)?),
        }),
        ChannelData::U16(samples) => Ok(DecodedChannel {
            width: region.width,
            height: region.height,
            hshift: 0,
            vshift: 0,
            bit_depth: channel.bit_depth,
            samples: ChannelData::U16(crop_interleaved_u16(&samples, channel.width, 1, region)?),
        }),
    }
}

fn orient_decoded_channels(
    channels: DecodedChannels,
    orientation: Orientation,
) -> Result<DecodedChannels> {
    if orientation == Orientation::Identity {
        return Ok(channels);
    }
    let (width, height) = oriented_dimensions(channels.width, channels.height, orientation);
    let oriented_channels = channels
        .channels
        .into_iter()
        .map(|channel| orient_decoded_channel(channel, width, height, orientation))
        .collect::<Result<Vec<_>>>()?;
    Ok(DecodedChannels {
        width,
        height,
        color_channels: channels.color_channels,
        alpha: channels.alpha,
        bit_depth: channels.bit_depth,
        channels: oriented_channels,
    })
}

fn orient_decoded_channel(
    channel: DecodedChannel,
    image_width: u32,
    image_height: u32,
    orientation: Orientation,
) -> Result<DecodedChannel> {
    match channel.samples {
        ChannelData::U8(samples) => {
            let (width, height, samples) =
                orient_interleaved(samples, channel.width, channel.height, 1, orientation)?;
            let (hshift, vshift) = infer_channel_shifts(image_width, image_height, width, height)?;
            Ok(DecodedChannel {
                width,
                height,
                hshift,
                vshift,
                bit_depth: channel.bit_depth,
                samples: ChannelData::U8(samples),
            })
        }
        ChannelData::U16(samples) => {
            let (width, height, samples) =
                orient_interleaved(samples, channel.width, channel.height, 1, orientation)?;
            let (hshift, vshift) = infer_channel_shifts(image_width, image_height, width, height)?;
            Ok(DecodedChannel {
                width,
                height,
                hshift,
                vshift,
                bit_depth: channel.bit_depth,
                samples: ChannelData::U16(samples),
            })
        }
    }
}

fn orient_decoded_image(image: DecodedImage, orientation: Orientation) -> Result<DecodedImage> {
    let channels = decoded_image_output_channels(&image);
    match image.pixels {
        PixelData::U8(samples) => {
            let (width, height, pixels) =
                orient_interleaved(samples, image.width, image.height, channels, orientation)?;
            Ok(DecodedImage {
                width,
                height,
                color_channels: image.color_channels,
                alpha: image.alpha,
                bit_depth: image.bit_depth,
                pixels: PixelData::U8(pixels),
            })
        }
        PixelData::U16(samples) => {
            let (width, height, pixels) =
                orient_interleaved(samples, image.width, image.height, channels, orientation)?;
            Ok(DecodedImage {
                width,
                height,
                color_channels: image.color_channels,
                alpha: image.alpha,
                bit_depth: image.bit_depth,
                pixels: PixelData::U16(pixels),
            })
        }
    }
}

fn orient_rgba8(image: RgbaImage, orientation: Orientation) -> Result<RgbaImage> {
    let (width, height, pixels) =
        orient_interleaved(image.pixels, image.width, image.height, 4, orientation)?;
    Ok(RgbaImage {
        width,
        height,
        pixels,
    })
}

fn orient_rgba16(image: Rgba16Image, orientation: Orientation) -> Result<Rgba16Image> {
    let (width, height, pixels) =
        orient_interleaved(image.pixels, image.width, image.height, 4, orientation)?;
    Ok(Rgba16Image {
        width,
        height,
        pixels,
    })
}

fn orient_interleaved<T: Copy>(
    samples: Vec<T>,
    width: u32,
    height: u32,
    channels: usize,
    orientation: Orientation,
) -> Result<(u32, u32, Vec<T>)> {
    let sample_count = (width as usize)
        .checked_mul(height as usize)
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    let expected_len = sample_count
        .checked_mul(channels)
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    if samples.len() != expected_len {
        return Err(Error::InvalidCodestream("decoded pixel count mismatch"));
    }
    if orientation == Orientation::Identity {
        return Ok((width, height, samples));
    }

    let (output_width, output_height) = oriented_dimensions(width, height, orientation);
    let mut output = Vec::with_capacity(expected_len);
    for y in 0..output_height {
        for x in 0..output_width {
            let (source_x, source_y) = oriented_source_position(width, height, x, y, orientation);
            let source_index = ((source_y as usize)
                .checked_mul(width as usize)
                .and_then(|index| index.checked_add(source_x as usize))
                .and_then(|index| index.checked_mul(channels)))
            .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
            let pixel = samples
                .get(source_index..source_index + channels)
                .ok_or(Error::InvalidCodestream("decoded pixel count mismatch"))?;
            output.extend_from_slice(pixel);
        }
    }

    Ok((output_width, output_height, output))
}

fn oriented_dimensions(width: u32, height: u32, orientation: Orientation) -> (u32, u32) {
    match orientation {
        Orientation::Transpose
        | Orientation::Rotate90Cw
        | Orientation::AntiTranspose
        | Orientation::Rotate90Ccw => (height, width),
        Orientation::Identity
        | Orientation::FlipHorizontal
        | Orientation::Rotate180
        | Orientation::FlipVertical => (width, height),
    }
}

fn oriented_source_position(
    width: u32,
    height: u32,
    x: u32,
    y: u32,
    orientation: Orientation,
) -> (u32, u32) {
    match orientation {
        Orientation::Identity => (x, y),
        Orientation::FlipHorizontal => (width - 1 - x, y),
        Orientation::Rotate180 => (width - 1 - x, height - 1 - y),
        Orientation::FlipVertical => (x, height - 1 - y),
        Orientation::Transpose => (y, x),
        Orientation::Rotate90Cw => (y, height - 1 - x),
        Orientation::AntiTranspose => (width - 1 - y, height - 1 - x),
        Orientation::Rotate90Ccw => (width - 1 - y, x),
    }
}

fn vardct_srgb_sample_count(width: u32, height: u32) -> Result<usize> {
    (width as usize)
        .checked_mul(height as usize)
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))
}

fn raw_alpha_info(metadata: &ImageMetadata) -> Result<Option<AlphaInfo>> {
    raw_alpha_channel_index_and_info(metadata).map(|alpha| alpha.map(|(_, alpha)| alpha))
}

fn raw_alpha_channel_index(metadata: &ImageMetadata) -> Result<Option<usize>> {
    raw_alpha_channel_index_and_info(metadata).map(|alpha| alpha.map(|(index, _)| index))
}

fn raw_alpha_channel_index_and_info(
    metadata: &ImageMetadata,
) -> Result<Option<(usize, AlphaInfo)>> {
    let Some((extra_index, alpha)) = metadata
        .extra_channels
        .iter()
        .enumerate()
        .find(|(_, channel)| channel.channel_type == ExtraChannelType::Alpha)
    else {
        return Ok(None);
    };
    if alpha.bit_depth.floating_point_sample {
        return Err(Error::Unsupported("floating-point alpha output"));
    }
    Ok(Some((
        metadata.num_color_channels() as usize + extra_index,
        AlphaInfo {
            bit_depth: alpha.bit_depth.bits_per_sample,
            premultiplied: alpha.alpha_associated,
        },
    )))
}

fn decoded_channel_bit_depths(metadata: &ImageMetadata, color_channels: usize) -> Result<Vec<u32>> {
    let mut bit_depths = Vec::with_capacity(color_channels + metadata.extra_channels.len());
    bit_depths.resize(color_channels, metadata.bit_depth.bits_per_sample);
    for extra in &metadata.extra_channels {
        if extra.bit_depth.floating_point_sample {
            return Err(Error::Unsupported("floating-point extra-channel output"));
        }
        if extra.bit_depth.bits_per_sample > 16 {
            return Err(Error::Unsupported(
                "integer extra-channel sample depths above 16 bits",
            ));
        }
        bit_depths.push(extra.bit_depth.bits_per_sample);
    }
    Ok(bit_depths)
}

fn decode_interleaved_alpha(
    channels: &DecodedChannels,
    alpha_channel_index: Option<usize>,
) -> Result<Option<AlphaInfo>> {
    let alpha = channels.alpha;
    if let Some(alpha) = alpha {
        let alpha_channel_index = alpha_channel_index.ok_or(Error::InvalidCodestream(
            "decoded alpha metadata missing channel index",
        ))?;
        if channels.channels.len() <= alpha_channel_index {
            return Err(Error::Unsupported("missing alpha channel output"));
        }
        let alpha_channel = &channels.channels[alpha_channel_index];
        if alpha_channel.bit_depth != alpha.bit_depth {
            return Err(Error::InvalidCodestream(
                "decoded alpha channel bit-depth mismatch",
            ));
        }
        if alpha_channel.hshift != 0 || alpha_channel.vshift != 0 {
            return Err(Error::Unsupported("subsampled alpha image decode"));
        }
    } else if alpha_channel_index.is_some() {
        return Err(Error::InvalidCodestream(
            "decoded alpha channel index without alpha metadata",
        ));
    }
    Ok(alpha)
}

fn interleaved_channel_indices(
    channels: &DecodedChannels,
    alpha_channel_index: Option<usize>,
) -> Result<Vec<usize>> {
    if channels.channels.len() < channels.color_channels {
        return Err(Error::Unsupported("missing color channel output"));
    }
    let output_channels = channels.color_channels + usize::from(channels.alpha.is_some());
    let mut indices = Vec::with_capacity(output_channels);
    indices.extend(0..channels.color_channels);
    if let Some(alpha_channel_index) = alpha_channel_index {
        indices.push(alpha_channel_index);
    }
    Ok(indices)
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
        bit_depth,
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

fn interleave_channel_u8(image: &DecodedChannels, channel_indices: &[usize]) -> Result<Vec<u8>> {
    let output_channels = channel_indices.len();
    let sample_count = decoded_channel_sample_count(image)?;
    let pixels = sample_count
        .checked_mul(output_channels)
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    let mut output = Vec::with_capacity(pixels);
    for index in 0..sample_count {
        for &channel_index in channel_indices {
            let channel = &image.channels[channel_index];
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

fn interleave_channel_u16(image: &DecodedChannels, channel_indices: &[usize]) -> Result<Vec<u16>> {
    let output_channels = channel_indices.len();
    let sample_count = decoded_channel_sample_count(image)?;
    let pixels = sample_count
        .checked_mul(output_channels)
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    let mut output = Vec::with_capacity(pixels);
    for index in 0..sample_count {
        for &channel_index in channel_indices {
            let channel = &image.channels[channel_index];
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

fn rgba_channel_indices(
    channels: &DecodedChannels,
    alpha_channel_index: Option<usize>,
) -> Result<Vec<usize>> {
    let alpha = decode_interleaved_alpha(channels, alpha_channel_index)?;
    let output_channel_indices = interleaved_channel_indices(channels, alpha_channel_index)?;
    if output_channel_indices.iter().any(|&index| {
        channels.channels[index].width != channels.width
            || channels.channels[index].height != channels.height
    }) {
        return Err(Error::Unsupported("subsampled raw channel output"));
    }
    if alpha != channels.alpha {
        return Err(Error::InvalidCodestream("decoded alpha metadata mismatch"));
    }
    Ok(output_channel_indices)
}

fn rgba8_from_channel_indices(
    channels: &DecodedChannels,
    channel_indices: &[usize],
) -> Result<Vec<u8>> {
    let sample_count = decoded_channel_sample_count(channels)?;
    let mut rgba = Vec::with_capacity(sample_count * 4);
    for index in 0..sample_count {
        append_rgba8_pixel(
            &mut rgba,
            channels.color_channels,
            channels.alpha,
            |channel| channel_sample(&channels.channels[channel_indices[channel]], index),
        )?;
    }
    Ok(rgba)
}

fn rgba16_from_channel_indices(
    channels: &DecodedChannels,
    channel_indices: &[usize],
) -> Result<Vec<u16>> {
    let sample_count = decoded_channel_sample_count(channels)?;
    let mut rgba = Vec::with_capacity(sample_count * 4);
    for index in 0..sample_count {
        append_rgba16_pixel(
            &mut rgba,
            channels.color_channels,
            channels.alpha,
            |channel| channel_sample(&channels.channels[channel_indices[channel]], index),
        )?;
    }
    Ok(rgba)
}

fn channel_sample(channel: &DecodedChannel, index: usize) -> Result<(u32, u32)> {
    let sample = match &channel.samples {
        ChannelData::U8(samples) => u32::from(
            *samples
                .get(index)
                .ok_or(Error::InvalidCodestream("decoded pixel count mismatch"))?,
        ),
        ChannelData::U16(samples) => u32::from(
            *samples
                .get(index)
                .ok_or(Error::InvalidCodestream("decoded pixel count mismatch"))?,
        ),
    };
    Ok((sample, channel.bit_depth))
}

fn append_rgba8_pixel(
    rgba: &mut Vec<u8>,
    color_channels: usize,
    alpha: Option<AlphaInfo>,
    sample: impl Fn(usize) -> Result<(u32, u32)>,
) -> Result<()> {
    let alpha_sample = if alpha.is_some() {
        Some(sample(color_channels)?)
    } else {
        None
    };
    let color_sample = |index| -> Result<u8> {
        let (value, bit_depth) = sample(index)?;
        Ok(
            scale_or_unpremultiply_sample_to(value, bit_depth, alpha, alpha_sample, u8::MAX as u32)
                as u8,
        )
    };
    match color_channels {
        1 => {
            let gray = color_sample(0)?;
            rgba.extend_from_slice(&[gray, gray, gray]);
        }
        3 => {
            rgba.extend_from_slice(&[color_sample(0)?, color_sample(1)?, color_sample(2)?]);
        }
        _ => return Err(Error::Unsupported("unsupported color channel count")),
    }
    rgba.push(
        if let Some((alpha_sample, alpha_bit_depth)) = alpha_sample {
            scale_sample_to_u8(alpha_sample, alpha_bit_depth)
        } else {
            255
        },
    );
    Ok(())
}

fn append_rgba16_pixel(
    rgba: &mut Vec<u16>,
    color_channels: usize,
    alpha: Option<AlphaInfo>,
    sample: impl Fn(usize) -> Result<(u32, u32)>,
) -> Result<()> {
    let alpha_sample = if alpha.is_some() {
        Some(sample(color_channels)?)
    } else {
        None
    };
    let color_sample = |index| -> Result<u16> {
        let (value, bit_depth) = sample(index)?;
        Ok(
            scale_or_unpremultiply_sample_to(value, bit_depth, alpha, alpha_sample, u16::MAX as u32)
                as u16,
        )
    };
    match color_channels {
        1 => {
            let gray = color_sample(0)?;
            rgba.extend_from_slice(&[gray, gray, gray]);
        }
        3 => {
            rgba.extend_from_slice(&[color_sample(0)?, color_sample(1)?, color_sample(2)?]);
        }
        _ => return Err(Error::Unsupported("unsupported color channel count")),
    }
    rgba.push(
        if let Some((alpha_sample, alpha_bit_depth)) = alpha_sample {
            scale_sample_to_u16(alpha_sample, alpha_bit_depth)
        } else {
            u16::MAX
        },
    );
    Ok(())
}

fn decoded_channel_sample_count(image: &DecodedChannels) -> Result<usize> {
    (image.width as usize)
        .checked_mul(image.height as usize)
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))
}

fn decoded_image_output_channels(image: &DecodedImage) -> usize {
    image.color_channels + usize::from(image.alpha.is_some())
}

#[cfg(test)]
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

fn scale_or_unpremultiply_sample_to(
    sample: u32,
    bit_depth: u32,
    alpha: Option<AlphaInfo>,
    alpha_sample: Option<(u32, u32)>,
    output_max: u32,
) -> u32 {
    if alpha.is_some_and(|alpha| alpha.premultiplied) {
        let (alpha_sample, alpha_bit_depth) = alpha_sample.unwrap_or((0, bit_depth));
        unpremultiply_sample_to(sample, bit_depth, alpha_sample, alpha_bit_depth, output_max)
    } else {
        scale_sample_to(sample, bit_depth, output_max)
    }
}

fn unpremultiply_sample_to(
    sample: u32,
    bit_depth: u32,
    alpha: u32,
    alpha_bit_depth: u32,
    output_max: u32,
) -> u32 {
    if alpha == 0 {
        return if sample == 0 { 0 } else { output_max };
    }
    let sample_max = (1u64 << bit_depth) - 1;
    let alpha_max = (1u64 << alpha_bit_depth) - 1;
    let numerator = u64::from(sample) * alpha_max * u64::from(output_max);
    let denominator = sample_max * u64::from(alpha);
    ((numerator + denominator / 2) / denominator).min(u64::from(output_max)) as u32
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
    fn decoder_defaults_are_buffered_channel_decode() {
        let decoder = Decoder::new();

        assert_eq!(decoder.options().output, DecodeOutput::Channels);
        assert_eq!(decoder.options().roi, None);
        assert_eq!(decoder.options().vardct_pass, None);
        assert_eq!(decoder.options().threads, ThreadingMode::Auto);
        assert_eq!(decoder.options().memory_limit, None);
    }

    #[test]
    fn decoder_methods_match_convenience_functions() {
        let bytes = std::fs::read(workspace_path(
            "crates/jxl-codec/tests/generated/icc_rec2020_lossless.jxl",
        ))
        .unwrap();
        let decoder = Decoder::new().threads(ThreadingMode::Threads(2));

        assert_eq!(decoder.decode_channels(&bytes), decode_channels(&bytes));
        assert_eq!(decoder.decode(&bytes), decode(&bytes));
        assert_eq!(decoder.decode_rgba8(&bytes), decode_rgba8(&bytes));
        assert_eq!(decoder.decode_rgba16(&bytes), decode_rgba16(&bytes));
    }

    #[test]
    fn decoder_rejects_unsupported_options() {
        let bytes = std::fs::read(workspace_path(
            "crates/jxl-codec/tests/generated/icc_rec2020_lossless.jxl",
        ))
        .unwrap();

        let roi_decoder = Decoder::new().roi(Rect {
            x: 0,
            y: 0,
            width: 8,
            height: 8,
        });
        assert_eq!(roi_decoder.decode(&bytes).unwrap().width, 8);
        assert_eq!(roi_decoder.decode_rgba8(&bytes).unwrap().height, 8);
        assert_eq!(roi_decoder.decode_rgba16(&bytes).unwrap().width, 8);

        let memory_decoder = Decoder::new().memory_limit(1024);
        assert_eq!(
            memory_decoder.decode(&bytes),
            Err(Error::Unsupported("memory-limited decode"))
        );

        let zero_threads_decoder = Decoder::new().threads(ThreadingMode::Threads(0));
        assert_eq!(
            zero_threads_decoder.decode_rgba8(&bytes),
            Err(Error::Unsupported("zero decoder threads"))
        );

        let vardct_pass_decoder = Decoder::new().vardct_pass(0);
        assert_eq!(
            vardct_pass_decoder.decode_channels(&bytes),
            Err(Error::Unsupported("VarDCT progressive pass decode"))
        );
        assert_eq!(
            vardct_pass_decoder.decode(&bytes),
            Err(Error::Unsupported("VarDCT progressive pass decode"))
        );
        assert_eq!(
            vardct_pass_decoder.decode_rgba8(&bytes),
            Err(Error::Unsupported("VarDCT progressive pass decode"))
        );
        assert_eq!(
            vardct_pass_decoder.decode_rgba16(&bytes),
            Err(Error::Unsupported("VarDCT progressive pass decode"))
        );

        let out_of_bounds_roi_decoder = Decoder::new().roi(Rect {
            x: 64,
            y: 0,
            width: 1,
            height: 1,
        });
        assert_eq!(
            out_of_bounds_roi_decoder.decode_channels(&bytes),
            Err(Error::InvalidCodestream("modular region is outside image"))
        );
    }

    #[test]
    fn interleaved_orientation_matches_jpeg_xl_codes() {
        let samples = vec![1u8, 2, 3, 4, 5, 6];
        let cases = [
            (Orientation::Identity, 2, 3, vec![1, 2, 3, 4, 5, 6]),
            (Orientation::FlipHorizontal, 2, 3, vec![2, 1, 4, 3, 6, 5]),
            (Orientation::Rotate180, 2, 3, vec![6, 5, 4, 3, 2, 1]),
            (Orientation::FlipVertical, 2, 3, vec![5, 6, 3, 4, 1, 2]),
            (Orientation::Transpose, 3, 2, vec![1, 3, 5, 2, 4, 6]),
            (Orientation::Rotate90Cw, 3, 2, vec![5, 3, 1, 6, 4, 2]),
            (Orientation::AntiTranspose, 3, 2, vec![6, 4, 2, 5, 3, 1]),
            (Orientation::Rotate90Ccw, 3, 2, vec![2, 4, 6, 1, 3, 5]),
        ];

        for (orientation, width, height, expected) in cases {
            let (actual_width, actual_height, actual) =
                orient_interleaved(samples.clone(), 2, 3, 1, orientation).unwrap();
            assert_eq!(
                (actual_width, actual_height, actual),
                (width, height, expected),
                "orientation {orientation:?}"
            );
        }
    }

    #[test]
    fn interleaved_orientation_preserves_pixel_components() {
        let samples = vec![1u8, 10, 2, 20, 3, 30, 4, 40];
        let (width, height, oriented) =
            orient_interleaved(samples, 2, 2, 2, Orientation::Rotate90Cw).unwrap();

        assert_eq!(width, 2);
        assert_eq!(height, 2);
        assert_eq!(oriented, vec![3, 30, 1, 10, 4, 40, 2, 20]);
    }

    #[test]
    fn decode_channels_roi_supports_palette_modular_fixture() {
        let bytes = std::fs::read(workspace_path(
            "reference/libjxl/testdata/jxl/pq_gradient.jxl",
        ))
        .unwrap();
        let roi = Rect {
            x: 600,
            y: 0,
            width: 32,
            height: 32,
        };
        let full = decode_channels(&bytes).unwrap();
        let roi_image = Decoder::new().roi(roi).decode_channels(&bytes).unwrap();

        assert_roi_matches_full_channels(&roi_image, &full, roi);
    }

    #[test]
    fn decode_roi_supports_rct_modular_pixels_and_rgba() {
        let bytes = std::fs::read(workspace_path(
            "crates/jxl-codec/tests/generated/icc_rec2020_lossless.jxl",
        ))
        .unwrap();
        let roi = Rect {
            x: 7,
            y: 11,
            width: 19,
            height: 17,
        };
        let full = decode(&bytes).unwrap();
        let roi_image = Decoder::new().roi(roi).decode(&bytes).unwrap();
        assert_roi_matches_full_image(&roi_image, &full, roi);

        let full_rgba8 = decode_rgba8(&bytes).unwrap();
        let roi_rgba8 = Decoder::new().roi(roi).decode_rgba8(&bytes).unwrap();
        assert_roi_matches_full_rgba8(&roi_rgba8, &full_rgba8, roi);

        let full_rgba16 = decode_rgba16(&bytes).unwrap();
        let roi_rgba16 = Decoder::new().roi(roi).decode_rgba16(&bytes).unwrap();
        assert_roi_matches_full_rgba16(&roi_rgba16, &full_rgba16, roi);
    }

    #[test]
    fn decode_roi_supports_palette_modular_pixels_and_rgba() {
        let bytes = std::fs::read(workspace_path(
            "reference/libjxl/testdata/jxl/pq_gradient.jxl",
        ))
        .unwrap();
        let roi = Rect {
            x: 600,
            y: 0,
            width: 32,
            height: 32,
        };
        let full = decode(&bytes).unwrap();
        let roi_image = Decoder::new().roi(roi).decode(&bytes).unwrap();
        assert_roi_matches_full_image(&roi_image, &full, roi);

        let full_rgba8 = decode_rgba8(&bytes).unwrap();
        let roi_rgba8 = Decoder::new().roi(roi).decode_rgba8(&bytes).unwrap();
        assert_roi_matches_full_rgba8(&roi_rgba8, &full_rgba8, roi);
    }

    #[test]
    fn decode_channels_roi_supports_rct_modular_fixture() {
        let bytes = std::fs::read(workspace_path(
            "crates/jxl-codec/tests/generated/icc_rec2020_lossless.jxl",
        ))
        .unwrap();
        let roi = Rect {
            x: 5,
            y: 7,
            width: 11,
            height: 9,
        };
        let full = decode_channels(&bytes).unwrap();
        let roi_image = Decoder::new().roi(roi).decode_channels(&bytes).unwrap();

        assert_roi_matches_full_channels(&roi_image, &full, roi);
    }

    #[test]
    fn decode_channels_roi_supports_rct_modular_fixture_with_threads() {
        let bytes = std::fs::read(workspace_path(
            "crates/jxl-codec/tests/generated/icc_rec2020_lossless.jxl",
        ))
        .unwrap();
        let roi = Rect {
            x: 17,
            y: 3,
            width: 13,
            height: 15,
        };
        let full = decode_channels(&bytes).unwrap();
        let roi_image = Decoder::new()
            .roi(roi)
            .threads(ThreadingMode::Threads(2))
            .decode_channels(&bytes)
            .unwrap();

        assert_roi_matches_full_channels(&roi_image, &full, roi);
    }

    #[test]
    fn decode_channels_roi_crops_transform_free_modular_image_when_available() {
        let Some(cjxl) = reference_cjxl() else {
            eprintln!("skipping rgba channel ROI comparison; reference tools are not built");
            return;
        };

        let input = unique_temp_path("jxl-roi-source", "pgm");
        let encoded = unique_temp_path("jxl-roi", "jxl");
        write_roi_source_pgm(&input);

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

        let encoded_bytes = std::fs::read(&encoded).unwrap();
        let _ = std::fs::remove_file(&encoded);
        let image = Decoder::new()
            .roi(Rect {
                x: 2,
                y: 1,
                width: 3,
                height: 2,
            })
            .decode_channels(&encoded_bytes)
            .unwrap();

        assert_eq!(image.width, 3);
        assert_eq!(image.height, 2);
        assert_eq!(image.color_channels, 1);
        assert_eq!(image.alpha, None);
        assert_eq!(image.bit_depth, 8);
        assert_eq!(image.channels.len(), 1);
        assert_eq!(image.channels[0].width, 3);
        assert_eq!(image.channels[0].height, 2);
        assert_eq!(image.channels[0].hshift, 0);
        assert_eq!(image.channels[0].vshift, 0);
        let ChannelData::U8(samples) = &image.channels[0].samples else {
            panic!("expected 8-bit ROI samples");
        };
        assert_eq!(samples, &[206, 213, 220, 142, 149, 156]);
    }

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
    fn rejects_unreconstructed_var_dct_fixture() {
        let bytes = std::fs::read(workspace_path(
            "reference/libjxl/testdata/jxl/boxes/square-extended-size-container.jxl",
        ))
        .unwrap();
        let info = inspect(&bytes).unwrap();
        assert!(info.first_frame_modular.is_none());
        assert_eq!(
            info.first_frame_vardct.as_ref().unwrap().sections[0].section_kind,
            FrameSectionKind::Combined
        );

        let roi_decoder = Decoder::new().roi(Rect {
            x: 0,
            y: 0,
            width: 4,
            height: 4,
        });
        assert_eq!(
            decode(&bytes),
            Err(Error::Unsupported("VarDCT image reconstruction"))
        );
        assert_eq!(
            decode_channels(&bytes),
            Err(Error::Unsupported("VarDCT image reconstruction"))
        );
        assert_eq!(
            decode_rgba8(&bytes),
            Err(Error::Unsupported("VarDCT image reconstruction"))
        );
        assert_eq!(
            roi_decoder.decode_channels(&bytes),
            Err(Error::Unsupported("VarDCT image reconstruction"))
        );
        assert_eq!(
            roi_decoder.decode(&bytes),
            Err(Error::Unsupported("VarDCT image reconstruction"))
        );
        assert_eq!(
            roi_decoder.decode_rgba8(&bytes),
            Err(Error::Unsupported("VarDCT image reconstruction"))
        );
        assert_eq!(
            roi_decoder.decode_rgba16(&bytes),
            Err(Error::Unsupported("VarDCT image reconstruction"))
        );
        assert_eq!(
            decode_rgba16(&bytes),
            Err(Error::Unsupported("VarDCT image reconstruction"))
        );
    }

    #[test]
    fn decode_rgba_supports_generated_var_dct_when_available() {
        let Some(cjxl) = reference_cjxl() else {
            eprintln!("skipping public VarDCT rgba decode; reference cjxl is not built");
            return;
        };

        let input = unique_temp_path("jxl-rgba-vardct-source", "ppm");
        let encoded = unique_temp_path("jxl-rgba-vardct", "jxl");
        write_split_vardct_source_ppm(&input);

        let cjxl_output = Command::new(&cjxl)
            .arg(&input)
            .arg(&encoded)
            .args([
                "-d",
                "1.0",
                "-m",
                "0",
                "--container=0",
                "--progressive_ac",
                "--quiet",
            ])
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&input);
        assert!(
            cjxl_output.status.success(),
            "reference cjxl failed: {}",
            String::from_utf8_lossy(&cjxl_output.stderr)
        );

        let reference = reference_djxl().map(|djxl| {
            let output = unique_temp_path("jxl-rgba-vardct-reference", "ppm");
            let djxl_output = Command::new(&djxl)
                .arg(&encoded)
                .arg(&output)
                .arg("--quiet")
                .output()
                .unwrap();
            assert!(
                djxl_output.status.success(),
                "reference djxl failed: {}",
                String::from_utf8_lossy(&djxl_output.stderr)
            );

            let reference = std::fs::read(&output).unwrap();
            let _ = std::fs::remove_file(&output);
            parse_ppm_rgb(&reference)
        });

        let encoded_bytes = std::fs::read(&encoded).unwrap();
        let _ = std::fs::remove_file(&encoded);
        let roi = Rect {
            x: 17,
            y: 19,
            width: 41,
            height: 29,
        };
        let roi_decoder = Decoder::new().roi(Rect {
            x: roi.x,
            y: roi.y,
            width: roi.width,
            height: roi.height,
        });
        let out_of_bounds_roi_decoder = Decoder::new().roi(Rect {
            x: 319,
            y: 0,
            width: 2,
            height: 1,
        });

        let decoded_channels = decode_channels(&encoded_bytes).unwrap();
        let roi_channels = roi_decoder.decode_channels(&encoded_bytes).unwrap();
        let decoded = decode(&encoded_bytes).unwrap();
        let rgba = decode_rgba8(&encoded_bytes).unwrap();
        let rgba16 = decode_rgba16(&encoded_bytes).unwrap();
        let roi_decoded = roi_decoder.decode(&encoded_bytes).unwrap();
        let roi_rgba = roi_decoder.decode_rgba8(&encoded_bytes).unwrap();
        let roi_rgba16 = roi_decoder.decode_rgba16(&encoded_bytes).unwrap();
        let pass0_decoded = Decoder::new()
            .vardct_pass(0)
            .decode(&encoded_bytes)
            .unwrap();
        let pass0_rgba = Decoder::new()
            .vardct_pass(0)
            .decode_rgba8(&encoded_bytes)
            .unwrap();
        let pass0_rgba16 = Decoder::new()
            .vardct_pass(0)
            .decode_rgba16(&encoded_bytes)
            .unwrap();
        let pass0_roi = Decoder::new()
            .vardct_pass(0)
            .roi(roi)
            .decode(&encoded_bytes)
            .unwrap();
        let missing_pass_decoder = Decoder::new().vardct_pass(99);

        assert_eq!(decoded.width, 320);
        assert_eq!(decoded.height, 192);
        assert_eq!(decoded.color_channels, 3);
        assert_eq!(decoded.alpha, None);
        assert_eq!(decoded.bit_depth, 8);
        let PixelData::U8(decoded_pixels) = &decoded.pixels else {
            panic!("expected VarDCT decode to return 8-bit RGB");
        };
        assert_eq!(decoded_pixels.len(), 320 * 192 * 3);
        assert!(
            decoded_pixels
                .chunks_exact(3)
                .any(|pixel| pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0)
        );
        assert_decoded_channels_match_image(&decoded_channels, &decoded);
        assert_roi_matches_full_channels(&roi_channels, &decoded_channels, roi);
        if let Some(reference) = &reference {
            assert_eq!(decoded.width, reference.width);
            assert_eq!(decoded.height, reference.height);
            let metrics = srgb8_oracle_metrics(
                &decoded,
                reference,
                &[0, decoded_pixels.len() / 2, decoded_pixels.len() - 1],
            );
            assert_eq!(
                metrics,
                Srgb8OracleMetrics {
                    max_abs_error: 255,
                    sum_abs_error: 13_423_127,
                    checksum: 15_223_620_237_915_187_279,
                    anchors: vec![3, 40, 0],
                    reference_anchors: vec![0, 21, 255],
                }
            );
        } else {
            eprintln!("skipping public VarDCT djxl comparison; tool is not built");
        }
        assert_roi_matches_full_image(&roi_decoded, &decoded, roi);
        assert_eq!(pass0_decoded.width, 320);
        assert_eq!(pass0_decoded.height, 192);
        assert_eq!(pass0_decoded.color_channels, 3);
        assert_eq!(pass0_decoded.alpha, None);
        assert_eq!(pass0_decoded.bit_depth, 8);
        let pass0_channels = Decoder::new()
            .vardct_pass(0)
            .decode_channels(&encoded_bytes)
            .unwrap();
        let pass0_roi_channels = Decoder::new()
            .vardct_pass(0)
            .roi(roi)
            .decode_channels(&encoded_bytes)
            .unwrap();
        assert_decoded_channels_match_image(&pass0_channels, &pass0_decoded);
        assert_roi_matches_full_image(&pass0_roi, &pass0_decoded, roi);
        assert_roi_matches_full_channels(&pass0_roi_channels, &pass0_channels, roi);

        assert_eq!(rgba.width, 320);
        assert_eq!(rgba.height, 192);
        assert_eq!(rgba.pixels.len(), 320 * 192 * 4);
        assert!(rgba.pixels.chunks_exact(4).all(|pixel| pixel[3] == 255));
        assert!(
            rgba.pixels
                .chunks_exact(4)
                .any(|pixel| pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0)
        );
        assert_roi_matches_full_rgba8(&roi_rgba, &rgba, roi);
        assert_eq!(pass0_rgba.width, 320);
        assert_eq!(pass0_rgba.height, 192);
        assert_eq!(pass0_rgba.pixels.len(), 320 * 192 * 4);
        assert!(
            pass0_rgba
                .pixels
                .chunks_exact(4)
                .all(|pixel| pixel[3] == 255)
        );

        assert_eq!(rgba16.width, 320);
        assert_eq!(rgba16.height, 192);
        assert_eq!(rgba16.pixels.len(), 320 * 192 * 4);
        assert!(
            rgba16
                .pixels
                .chunks_exact(4)
                .all(|pixel| pixel[3] == u16::MAX)
        );
        assert!(
            rgba16
                .pixels
                .chunks_exact(4)
                .any(|pixel| pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0)
        );
        assert_roi_matches_full_rgba16(&roi_rgba16, &rgba16, roi);
        assert_eq!(pass0_rgba16.width, 320);
        assert_eq!(pass0_rgba16.height, 192);
        assert_eq!(pass0_rgba16.pixels.len(), 320 * 192 * 4);
        assert!(
            pass0_rgba16
                .pixels
                .chunks_exact(4)
                .all(|pixel| pixel[3] == u16::MAX)
        );
        assert_eq!(
            out_of_bounds_roi_decoder.decode(&encoded_bytes),
            Err(Error::InvalidCodestream("decode region is outside image"))
        );
        assert_eq!(
            out_of_bounds_roi_decoder.decode_rgba8(&encoded_bytes),
            Err(Error::InvalidCodestream("decode region is outside image"))
        );
        assert_eq!(
            out_of_bounds_roi_decoder.decode_rgba16(&encoded_bytes),
            Err(Error::InvalidCodestream("decode region is outside image"))
        );
        assert_eq!(
            missing_pass_decoder.decode(&encoded_bytes),
            Err(Error::Unsupported("VarDCT image reconstruction"))
        );
        assert_eq!(
            missing_pass_decoder.decode_channels(&encoded_bytes),
            Err(Error::Unsupported("VarDCT image reconstruction"))
        );
        assert_eq!(
            missing_pass_decoder.decode_rgba8(&encoded_bytes),
            Err(Error::Unsupported("VarDCT image reconstruction"))
        );
        assert_eq!(
            missing_pass_decoder.decode_rgba16(&encoded_bytes),
            Err(Error::Unsupported("VarDCT image reconstruction"))
        );
    }

    #[test]
    fn decode_channels_exposes_generated_var_dct_alpha_when_available() {
        let Some(cjxl) = reference_cjxl() else {
            eprintln!("skipping VarDCT alpha raw-channel decode; reference cjxl is not built");
            return;
        };

        let input = unique_temp_path("jxl-vardct-alpha-source", "pam");
        let encoded = unique_temp_path("jxl-vardct-alpha", "jxl");
        let expected_alpha = write_vardct_alpha_source_pam(&input);

        let cjxl_output = Command::new(&cjxl)
            .arg(&input)
            .arg(&encoded)
            .args(["-d", "1.0", "-m", "0", "--container=0", "--quiet"])
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&input);
        assert!(
            cjxl_output.status.success(),
            "reference cjxl failed: {}",
            String::from_utf8_lossy(&cjxl_output.stderr)
        );

        let encoded_bytes = std::fs::read(&encoded).unwrap();
        let _ = std::fs::remove_file(&encoded);
        let info = inspect(&encoded_bytes).unwrap();
        assert!(info.first_frame_vardct.is_some());
        assert_eq!(
            raw_alpha_info(&info.metadata).unwrap(),
            Some(AlphaInfo {
                bit_depth: 8,
                premultiplied: false,
            })
        );

        let channels = decode_channels(&encoded_bytes).unwrap();
        assert_eq!(channels.width, 320);
        assert_eq!(channels.height, 192);
        assert_eq!(channels.color_channels, 3);
        assert_eq!(
            channels.alpha,
            Some(AlphaInfo {
                bit_depth: 8,
                premultiplied: false,
            })
        );
        assert_eq!(channels.bit_depth, 8);
        assert_eq!(channels.channels.len(), 4);
        let alpha_channel = &channels.channels[3];
        assert_eq!(alpha_channel.width, 320);
        assert_eq!(alpha_channel.height, 192);
        assert_eq!(alpha_channel.hshift, 0);
        assert_eq!(alpha_channel.vshift, 0);
        assert_eq!(alpha_channel.bit_depth, 8);
        let ChannelData::U8(alpha) = &alpha_channel.samples else {
            panic!("expected 8-bit VarDCT alpha channel");
        };
        assert_eq!(alpha, &expected_alpha);

        let roi = Rect {
            x: 17,
            y: 19,
            width: 37,
            height: 29,
        };
        let roi_channels = Decoder::new()
            .roi(roi)
            .decode_channels(&encoded_bytes)
            .unwrap();
        assert_eq!(roi_channels.width, roi.width);
        assert_eq!(roi_channels.height, roi.height);
        assert_eq!(roi_channels.alpha, channels.alpha);
        assert_eq!(roi_channels.channels.len(), channels.channels.len());
        let ChannelData::U8(roi_alpha) = &roi_channels.channels[3].samples else {
            panic!("expected 8-bit VarDCT ROI alpha channel");
        };
        assert_eq!(roi_alpha, &window_u8(&expected_alpha, 320, roi));

        let decoded = decode(&encoded_bytes).unwrap();
        assert_eq!(decoded.width, 320);
        assert_eq!(decoded.height, 192);
        assert_eq!(decoded.color_channels, 3);
        assert_eq!(decoded.alpha, channels.alpha);
        let PixelData::U8(decoded_pixels) = &decoded.pixels else {
            panic!("expected 8-bit VarDCT decoded image");
        };
        assert_eq!(decoded_pixels.len(), expected_alpha.len() * 4);
        assert_eq!(
            decoded_pixels
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>(),
            expected_alpha
        );

        let rgba8 = decode_rgba8(&encoded_bytes).unwrap();
        assert_eq!(rgba8.width, 320);
        assert_eq!(rgba8.height, 192);
        assert_eq!(rgba8.pixels.len(), expected_alpha.len() * 4);
        assert_eq!(
            rgba8
                .pixels
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>(),
            expected_alpha
        );

        let rgba16 = decode_rgba16(&encoded_bytes).unwrap();
        assert_eq!(rgba16.width, 320);
        assert_eq!(rgba16.height, 192);
        assert_eq!(rgba16.pixels.len(), expected_alpha.len() * 4);
        assert_eq!(
            rgba16
                .pixels
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>(),
            expected_alpha
                .iter()
                .map(|&alpha| u16::from(alpha) * 257)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn decode_channels_exposes_generated_var_dct_depth_when_available() {
        let Some(cjxl) = reference_cjxl() else {
            eprintln!("skipping VarDCT depth raw-channel decode; reference cjxl is not built");
            return;
        };

        let input = unique_temp_path("jxl-vardct-depth-source", "pam");
        let encoded = unique_temp_path("jxl-vardct-depth", "jxl");
        let expected_depth = write_vardct_rgb_depth_source_pam(&input);

        let cjxl_output = Command::new(&cjxl)
            .arg(&input)
            .arg(&encoded)
            .args(["-d", "1.0", "-m", "0", "--container=0", "--quiet"])
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&input);
        assert!(
            cjxl_output.status.success(),
            "reference cjxl failed: {}",
            String::from_utf8_lossy(&cjxl_output.stderr)
        );

        let encoded_bytes = std::fs::read(&encoded).unwrap();
        let _ = std::fs::remove_file(&encoded);
        let info = inspect(&encoded_bytes).unwrap();
        assert!(info.first_frame_vardct.is_some());
        assert_eq!(info.metadata.extra_channels.len(), 1);
        assert_eq!(
            info.metadata.extra_channels[0].channel_type,
            ExtraChannelType::Depth
        );

        let channels = decode_channels(&encoded_bytes).unwrap();
        assert_eq!(channels.width, 320);
        assert_eq!(channels.height, 192);
        assert_eq!(channels.color_channels, 3);
        assert_eq!(channels.alpha, None);
        assert_eq!(channels.bit_depth, 8);
        assert_eq!(channels.channels.len(), 4);
        let depth_channel = &channels.channels[3];
        assert_eq!(depth_channel.width, 320);
        assert_eq!(depth_channel.height, 192);
        assert_eq!(depth_channel.hshift, 0);
        assert_eq!(depth_channel.vshift, 0);
        assert_eq!(depth_channel.bit_depth, 8);
        let ChannelData::U8(depth) = &depth_channel.samples else {
            panic!("expected 8-bit VarDCT depth channel");
        };
        assert_eq!(depth, &expected_depth);

        let roi = Rect {
            x: 31,
            y: 13,
            width: 43,
            height: 27,
        };
        let roi_channels = Decoder::new()
            .roi(roi)
            .decode_channels(&encoded_bytes)
            .unwrap();
        assert_eq!(roi_channels.width, roi.width);
        assert_eq!(roi_channels.height, roi.height);
        assert_eq!(roi_channels.alpha, None);
        assert_eq!(roi_channels.channels.len(), channels.channels.len());
        let ChannelData::U8(roi_depth) = &roi_channels.channels[3].samples else {
            panic!("expected 8-bit VarDCT ROI depth channel");
        };
        assert_eq!(roi_depth, &window_u8(&expected_depth, 320, roi));

        let rgba8 = decode_rgba8(&encoded_bytes).unwrap();
        assert_eq!(rgba8.width, 320);
        assert_eq!(rgba8.height, 192);
        assert!(rgba8.pixels.chunks_exact(4).all(|pixel| pixel[3] == 255));

        let rgba16 = decode_rgba16(&encoded_bytes).unwrap();
        assert_eq!(rgba16.width, 320);
        assert_eq!(rgba16.height, 192);
        assert!(
            rgba16
                .pixels
                .chunks_exact(4)
                .all(|pixel| pixel[3] == u16::MAX)
        );
    }

    #[test]
    fn decode_channels_rejects_unreconstructed_generated_var_dct_extra_channels() {
        let Some(cjxl) = reference_cjxl() else {
            eprintln!("skipping VarDCT extra-channel rejection; reference cjxl is not built");
            return;
        };

        let input = unique_temp_path("jxl-vardct-depth-source", "pam");
        let encoded = unique_temp_path("jxl-vardct-depth", "jxl");
        write_rgb_depth_source_pam(&input);

        let cjxl_output = Command::new(&cjxl)
            .arg(&input)
            .arg(&encoded)
            .args(["-d", "1.0", "-m", "0", "--container=0", "--quiet"])
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&input);
        assert!(
            cjxl_output.status.success(),
            "reference cjxl failed: {}",
            String::from_utf8_lossy(&cjxl_output.stderr)
        );

        let encoded_bytes = std::fs::read(&encoded).unwrap();
        let _ = std::fs::remove_file(&encoded);
        let info = inspect(&encoded_bytes).unwrap();
        assert!(info.first_frame_vardct.is_some());
        assert_eq!(info.metadata.extra_channels.len(), 1);
        assert_eq!(
            info.metadata.extra_channels[0].channel_type,
            ExtraChannelType::Depth
        );

        assert_eq!(
            decode_channels(&encoded_bytes),
            Err(Error::Unsupported("VarDCT image reconstruction"))
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

        let roi = Rect {
            x: 19,
            y: 23,
            width: 37,
            height: 29,
        };
        let full_channels = decode_channels(&encoded_bytes).unwrap();
        let roi_channels = Decoder::new()
            .roi(roi)
            .decode_channels(&encoded_bytes)
            .unwrap();
        assert_roi_matches_full_channels(&roi_channels, &full_channels, roi);

        let roi_decoded = Decoder::new().roi(roi).decode(&encoded_bytes).unwrap();
        assert_roi_matches_full_image(&roi_decoded, &decoded, roi);

        let full_rgba8 = decode_rgba8(&encoded_bytes).unwrap();
        let roi_rgba8 = Decoder::new()
            .roi(roi)
            .decode_rgba8(&encoded_bytes)
            .unwrap();
        assert_roi_matches_full_rgba8(&roi_rgba8, &full_rgba8, roi);

        let top_roi = Rect {
            x: 19,
            y: 0,
            width: 37,
            height: 29,
        };
        let top_roi_channels = Decoder::new()
            .roi(top_roi)
            .decode_channels(&encoded_bytes)
            .unwrap();
        assert_roi_matches_full_channels(&top_roi_channels, &full_channels, top_roi);
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
    fn decode_rgba8_expands_gray_alpha_when_available() {
        let Some(cjxl) = reference_cjxl() else {
            eprintln!("skipping grayscale-alpha RGBA8 comparison; reference cjxl is not built");
            return;
        };

        let input = unique_temp_path("jxl-rgba8-gray-alpha-source", "pam");
        let encoded = unique_temp_path("jxl-rgba8-gray-alpha", "jxl");
        let source = write_gray_alpha_source_pam(&input);

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

        let encoded_bytes = std::fs::read(&encoded).unwrap();
        let _ = std::fs::remove_file(&encoded);

        let decoded = decode(&encoded_bytes).unwrap();
        assert_eq!(decoded.width, source.width);
        assert_eq!(decoded.height, source.height);
        assert_eq!(decoded.color_channels, 1);
        assert_eq!(
            decoded.alpha,
            Some(AlphaInfo {
                bit_depth: 8,
                premultiplied: false,
            })
        );
        assert_eq!(decoded.bit_depth, 8);
        let PixelData::U8(decoded_samples) = decoded.pixels else {
            panic!("expected 8-bit grayscale-alpha samples");
        };
        assert_eq!(decoded_samples, source.gray_alpha);

        let channels = decode_channels(&encoded_bytes).unwrap();
        assert_eq!(channels.width, source.width);
        assert_eq!(channels.height, source.height);
        assert_eq!(channels.color_channels, 1);
        assert_eq!(channels.alpha, decoded.alpha);
        assert_eq!(
            channels
                .channels
                .iter()
                .map(|channel| channel.bit_depth)
                .collect::<Vec<_>>(),
            vec![8, 8]
        );
        let ChannelData::U8(gray) = &channels.channels[0].samples else {
            panic!("expected 8-bit gray channel");
        };
        assert_eq!(gray, &source.gray);
        let ChannelData::U8(alpha) = &channels.channels[1].samples else {
            panic!("expected 8-bit alpha channel");
        };
        assert_eq!(alpha, &source.alpha);

        let rgba = decode_rgba8(&encoded_bytes).unwrap();
        assert_eq!(rgba.width, source.width);
        assert_eq!(rgba.height, source.height);
        assert_eq!(rgba.pixels, source.rgba);
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
    fn decode_rgba16_expands_gray_alpha_when_available() {
        let Some(cjxl) = reference_cjxl() else {
            eprintln!("skipping grayscale-alpha RGBA16 comparison; reference cjxl is not built");
            return;
        };

        let input = unique_temp_path("jxl-rgba16-gray-alpha-source", "pam");
        let encoded = unique_temp_path("jxl-rgba16-gray-alpha", "jxl");
        let source = write_gray_alpha_source_pam16(&input);

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

        let encoded_bytes = std::fs::read(&encoded).unwrap();
        let _ = std::fs::remove_file(&encoded);

        let decoded = decode(&encoded_bytes).unwrap();
        assert_eq!(decoded.width, source.width);
        assert_eq!(decoded.height, source.height);
        assert_eq!(decoded.color_channels, 1);
        assert_eq!(
            decoded.alpha,
            Some(AlphaInfo {
                bit_depth: 16,
                premultiplied: false,
            })
        );
        assert_eq!(decoded.bit_depth, 16);
        let PixelData::U16(decoded_samples) = decoded.pixels else {
            panic!("expected 16-bit grayscale-alpha samples");
        };
        assert_eq!(decoded_samples, source.gray_alpha);

        let channels = decode_channels(&encoded_bytes).unwrap();
        assert_eq!(channels.width, source.width);
        assert_eq!(channels.height, source.height);
        assert_eq!(channels.color_channels, 1);
        assert_eq!(channels.alpha, decoded.alpha);
        assert_eq!(
            channels
                .channels
                .iter()
                .map(|channel| channel.bit_depth)
                .collect::<Vec<_>>(),
            vec![16, 16]
        );
        let ChannelData::U16(gray) = &channels.channels[0].samples else {
            panic!("expected 16-bit gray channel");
        };
        assert_eq!(gray, &source.gray);
        let ChannelData::U16(alpha) = &channels.channels[1].samples else {
            panic!("expected 16-bit alpha channel");
        };
        assert_eq!(alpha, &source.alpha);

        let rgba = decode_rgba16(&encoded_bytes).unwrap();
        assert_eq!(rgba.width, source.width);
        assert_eq!(rgba.height, source.height);
        assert_eq!(rgba.pixels, source.rgba);
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
    fn decode_rgba8_ignores_non_alpha_extra_channels_when_available() {
        let Some(cjxl) = reference_cjxl() else {
            eprintln!("skipping extra-channel RGBA comparison; reference cjxl is not built");
            return;
        };

        let input = unique_temp_path("jxl-rgba8-alpha-depth-source", "pam");
        let encoded = unique_temp_path("jxl-rgba8-alpha-depth", "jxl");
        let source = write_alpha_depth_source_pam(&input);

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

        let encoded_bytes = std::fs::read(&encoded).unwrap();
        let _ = std::fs::remove_file(&encoded);

        let decoded = decode(&encoded_bytes).unwrap();
        assert_eq!(decoded.width, source.width);
        assert_eq!(decoded.height, source.height);
        assert_eq!(decoded.color_channels, 3);
        assert_eq!(
            decoded.alpha,
            Some(AlphaInfo {
                bit_depth: 8,
                premultiplied: false,
            })
        );
        assert_eq!(
            decoded_samples_u16(&decoded),
            source
                .rgba
                .iter()
                .copied()
                .map(u16::from)
                .collect::<Vec<_>>()
        );

        let channels = decode_channels(&encoded_bytes).unwrap();
        assert_eq!(channels.channels.len(), 5);
        assert_eq!(channels.alpha, decoded.alpha);
        assert_eq!(
            channels
                .channels
                .iter()
                .map(|channel| channel.bit_depth)
                .collect::<Vec<_>>(),
            vec![8, 8, 8, 8, 8]
        );
        let ChannelData::U8(depth) = &channels.channels[3].samples else {
            panic!("expected 8-bit depth extra channel");
        };
        assert_eq!(depth, &source.depth);

        let rgba = decode_rgba8(&encoded_bytes).unwrap();
        assert_eq!(rgba.width, source.width);
        assert_eq!(rgba.height, source.height);
        assert_eq!(rgba.pixels, source.rgba);
    }

    #[test]
    fn decode_rgba16_ignores_non_alpha_extra_channels_when_available() {
        let Some(cjxl) = reference_cjxl() else {
            eprintln!("skipping 16-bit extra-channel RGBA comparison; reference cjxl is not built");
            return;
        };

        let input = unique_temp_path("jxl-rgba16-alpha-depth-source", "pam");
        let encoded = unique_temp_path("jxl-rgba16-alpha-depth", "jxl");
        let source = write_alpha_depth_source_pam16(&input);

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

        let encoded_bytes = std::fs::read(&encoded).unwrap();
        let _ = std::fs::remove_file(&encoded);

        let decoded = decode(&encoded_bytes).unwrap();
        assert_eq!(decoded.width, source.width);
        assert_eq!(decoded.height, source.height);
        assert_eq!(decoded.color_channels, 3);
        assert_eq!(
            decoded.alpha,
            Some(AlphaInfo {
                bit_depth: 16,
                premultiplied: false,
            })
        );
        assert_eq!(decoded_samples_u16(&decoded), source.rgba);

        let channels = decode_channels(&encoded_bytes).unwrap();
        assert_eq!(channels.channels.len(), 5);
        assert_eq!(channels.alpha, decoded.alpha);
        assert_eq!(
            channels
                .channels
                .iter()
                .map(|channel| channel.bit_depth)
                .collect::<Vec<_>>(),
            vec![16, 16, 16, 16, 16]
        );
        let ChannelData::U16(depth) = &channels.channels[3].samples else {
            panic!("expected 16-bit depth extra channel");
        };
        assert_eq!(depth, &source.depth);

        let rgba = decode_rgba16(&encoded_bytes).unwrap();
        assert_eq!(rgba.width, source.width);
        assert_eq!(rgba.height, source.height);
        assert_eq!(rgba.pixels, source.rgba);
    }

    #[test]
    fn decode_rgba8_unpremultiplies_associated_alpha_when_available() {
        let Some(cjxl) = reference_cjxl() else {
            eprintln!("skipping premultiplied alpha comparison; reference cjxl is not built");
            return;
        };

        let input = unique_temp_path("jxl-rgba8-premul-alpha-source", "pam");
        let encoded = unique_temp_path("jxl-rgba8-premul-alpha", "jxl");
        let expected_rgba = write_premultiplied_alpha_source_pam(&input);

        let cjxl_output = Command::new(&cjxl)
            .arg(&input)
            .arg(&encoded)
            .args([
                "-d",
                "0",
                "-m",
                "1",
                "--container=0",
                "--premultiply=1",
                "--quiet",
            ])
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&input);
        assert!(
            cjxl_output.status.success(),
            "reference cjxl failed: {}",
            String::from_utf8_lossy(&cjxl_output.stderr)
        );

        let encoded_bytes = std::fs::read(&encoded).unwrap();
        let _ = std::fs::remove_file(&encoded);
        let decoded = decode(&encoded_bytes).unwrap();

        assert_eq!(
            decoded.alpha,
            Some(AlphaInfo {
                bit_depth: 8,
                premultiplied: true,
            })
        );
        let rgba = decode_rgba8(&encoded_bytes).unwrap();
        assert_eq!(rgba.width, decoded.width);
        assert_eq!(rgba.height, decoded.height);
        assert_eq!(rgba.pixels, expected_rgba);
    }

    #[test]
    fn decode_rgba16_unpremultiplies_associated_alpha_when_available() {
        let Some(cjxl) = reference_cjxl() else {
            eprintln!(
                "skipping 16-bit premultiplied alpha comparison; reference cjxl is not built"
            );
            return;
        };

        let input = unique_temp_path("jxl-rgba16-premul-alpha-source", "pam");
        let encoded = unique_temp_path("jxl-rgba16-premul-alpha", "jxl");
        let expected_rgba = write_premultiplied_alpha_source_pam16(&input);

        let cjxl_output = Command::new(&cjxl)
            .arg(&input)
            .arg(&encoded)
            .args([
                "-d",
                "0",
                "-m",
                "1",
                "--container=0",
                "--premultiply=1",
                "--quiet",
            ])
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&input);
        assert!(
            cjxl_output.status.success(),
            "reference cjxl failed: {}",
            String::from_utf8_lossy(&cjxl_output.stderr)
        );

        let encoded_bytes = std::fs::read(&encoded).unwrap();
        let _ = std::fs::remove_file(&encoded);
        let decoded = decode(&encoded_bytes).unwrap();

        assert_eq!(
            decoded.alpha,
            Some(AlphaInfo {
                bit_depth: 16,
                premultiplied: true,
            })
        );
        let rgba = decode_rgba16(&encoded_bytes).unwrap();
        assert_eq!(rgba.width, decoded.width);
        assert_eq!(rgba.height, decoded.height);
        assert_eq!(rgba.pixels, expected_rgba);
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

    #[test]
    fn unpremultiplies_associated_alpha_with_rounding_and_clamping() {
        assert_eq!(unpremultiply_sample_to(0, 8, 0, 8, 255), 0);
        assert_eq!(unpremultiply_sample_to(7, 8, 0, 8, 255), 255);
        assert_eq!(unpremultiply_sample_to(64, 8, 128, 8, 255), 128);
        assert_eq!(unpremultiply_sample_to(200, 8, 128, 8, 255), 255);
        assert_eq!(unpremultiply_sample_to(128, 8, 255, 8, 65_535), 32_896);
        assert_eq!(unpremultiply_sample_to(32_896, 16, 128, 8, 255), 255);
    }

    #[test]
    fn rgba_output_scales_mixed_bit_depth_alpha_channels() {
        let channels = DecodedChannels {
            width: 2,
            height: 1,
            color_channels: 3,
            alpha: Some(AlphaInfo {
                bit_depth: 8,
                premultiplied: true,
            }),
            bit_depth: 16,
            channels: vec![
                decoded_u16_channel(2, 1, &[16_448, 0]),
                decoded_u16_channel(2, 1, &[32_896, 0]),
                decoded_u16_channel(2, 1, &[65_535, 257]),
                DecodedChannel {
                    width: 2,
                    height: 1,
                    hshift: 0,
                    vshift: 0,
                    bit_depth: 8,
                    samples: ChannelData::U8(vec![128, 0]),
                },
            ],
        };
        assert_eq!(
            channels
                .channels
                .iter()
                .map(|channel| channel.bit_depth)
                .collect::<Vec<_>>(),
            vec![16, 16, 16, 8]
        );

        let rgba8 = rgba8_from_decoded_channels(&channels, Some(3)).unwrap();
        assert_eq!(rgba8.pixels, vec![128, 255, 255, 128, 0, 0, 255, 0]);

        let rgba16 = rgba16_from_decoded_channels(&channels, Some(3)).unwrap();
        assert_eq!(
            rgba16.pixels,
            vec![32_768, 65_535, 65_535, 32_896, 0, 0, 65_535, 0]
        );

        assert_eq!(
            decode_buffered_channels(channels, Some(3)),
            Err(Error::Unsupported("mixed bit-depth interleaved output"))
        );
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct PpmRgb {
        width: u32,
        height: u32,
        samples: Vec<u16>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AlphaDepthPam {
        width: u32,
        height: u32,
        rgba: Vec<u8>,
        depth: Vec<u8>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AlphaDepthPam16 {
        width: u32,
        height: u32,
        rgba: Vec<u16>,
        depth: Vec<u16>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct GrayAlphaPam {
        width: u32,
        height: u32,
        gray_alpha: Vec<u8>,
        gray: Vec<u8>,
        alpha: Vec<u8>,
        rgba: Vec<u8>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct GrayAlphaPam16 {
        width: u32,
        height: u32,
        gray_alpha: Vec<u16>,
        gray: Vec<u16>,
        alpha: Vec<u16>,
        rgba: Vec<u16>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Srgb8OracleMetrics {
        max_abs_error: u16,
        sum_abs_error: u64,
        checksum: u64,
        anchors: Vec<u16>,
        reference_anchors: Vec<u16>,
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

    fn decoded_u16_channel(width: u32, height: u32, samples: &[u16]) -> DecodedChannel {
        DecodedChannel {
            width,
            height,
            hshift: 0,
            vshift: 0,
            bit_depth: 16,
            samples: ChannelData::U16(samples.to_vec()),
        }
    }

    fn srgb8_oracle_metrics(
        decoded: &DecodedImage,
        reference: &PpmRgb,
        anchor_indices: &[usize],
    ) -> Srgb8OracleMetrics {
        assert_eq!(decoded.width, reference.width);
        assert_eq!(decoded.height, reference.height);
        assert_eq!(decoded.color_channels, 3);
        assert_eq!(decoded.bit_depth, 8);
        let PixelData::U8(samples) = &decoded.pixels else {
            panic!("expected public oracle comparison to use 8-bit RGB output");
        };
        assert_eq!(samples.len(), reference.samples.len());

        let mut max_abs_error = 0u16;
        let mut sum_abs_error = 0u64;
        let mut checksum = 0xcbf2_9ce4_8422_2325u64;
        for (index, (&actual, &reference)) in samples.iter().zip(&reference.samples).enumerate() {
            let actual = u16::from(actual);
            let error = actual.abs_diff(reference);
            max_abs_error = max_abs_error.max(error);
            sum_abs_error += u64::from(error);
            checksum ^= ((index as u64) << 32) ^ ((actual as u64) << 16) ^ u64::from(reference);
            checksum = checksum.wrapping_mul(0x0000_0100_0000_01b3);
        }

        Srgb8OracleMetrics {
            max_abs_error,
            sum_abs_error,
            checksum,
            anchors: anchor_indices
                .iter()
                .map(|&index| u16::from(samples[index]))
                .collect(),
            reference_anchors: anchor_indices
                .iter()
                .map(|&index| reference.samples[index])
                .collect(),
        }
    }

    fn assert_roi_matches_full_image(roi_image: &DecodedImage, full: &DecodedImage, roi: Rect) {
        assert_eq!(roi_image.width, roi.width);
        assert_eq!(roi_image.height, roi.height);
        assert_eq!(roi_image.color_channels, full.color_channels);
        assert_eq!(roi_image.alpha, full.alpha);
        assert_eq!(roi_image.bit_depth, full.bit_depth);
        let channels = decoded_image_output_channels(full);
        match (&roi_image.pixels, &full.pixels) {
            (PixelData::U8(roi_pixels), PixelData::U8(full_pixels)) => {
                assert_eq!(
                    roi_pixels,
                    &window_interleaved_u8(full_pixels, full.width, channels, roi)
                );
            }
            (PixelData::U16(roi_pixels), PixelData::U16(full_pixels)) => {
                assert_eq!(
                    roi_pixels,
                    &window_interleaved_u16(full_pixels, full.width, channels, roi)
                );
            }
            _ => panic!("ROI and full pixel bit depths differed"),
        }
    }

    fn assert_roi_matches_full_rgba8(roi_image: &RgbaImage, full: &RgbaImage, roi: Rect) {
        assert_eq!(roi_image.width, roi.width);
        assert_eq!(roi_image.height, roi.height);
        assert_eq!(
            roi_image.pixels,
            window_interleaved_u8(&full.pixels, full.width, 4, roi)
        );
    }

    fn assert_roi_matches_full_rgba16(roi_image: &Rgba16Image, full: &Rgba16Image, roi: Rect) {
        assert_eq!(roi_image.width, roi.width);
        assert_eq!(roi_image.height, roi.height);
        assert_eq!(
            roi_image.pixels,
            window_interleaved_u16(&full.pixels, full.width, 4, roi)
        );
    }

    fn assert_decoded_channels_match_image(channels: &DecodedChannels, image: &DecodedImage) {
        assert_eq!(channels.width, image.width);
        assert_eq!(channels.height, image.height);
        assert_eq!(channels.color_channels, image.color_channels);
        assert_eq!(channels.alpha, image.alpha);
        assert_eq!(channels.bit_depth, image.bit_depth);
        assert_eq!(
            channels.channels.len(),
            decoded_image_output_channels(image)
        );
        for channel in &channels.channels {
            assert_eq!(channel.width, image.width);
            assert_eq!(channel.height, image.height);
            assert_eq!(channel.hshift, 0);
            assert_eq!(channel.vshift, 0);
            assert_eq!(channel.bit_depth, image.bit_depth);
        }

        match &image.pixels {
            PixelData::U8(pixels) => {
                let mut interleaved = Vec::with_capacity(pixels.len());
                for index in 0..decoded_image_sample_count(image).unwrap() {
                    for channel in &channels.channels {
                        let ChannelData::U8(samples) = &channel.samples else {
                            panic!("channel bit depth did not match decoded image");
                        };
                        interleaved.push(samples[index]);
                    }
                }
                assert_eq!(&interleaved, pixels);
            }
            PixelData::U16(pixels) => {
                let mut interleaved = Vec::with_capacity(pixels.len());
                for index in 0..decoded_image_sample_count(image).unwrap() {
                    for channel in &channels.channels {
                        let ChannelData::U16(samples) = &channel.samples else {
                            panic!("channel bit depth did not match decoded image");
                        };
                        interleaved.push(samples[index]);
                    }
                }
                assert_eq!(&interleaved, pixels);
            }
        }
    }

    fn assert_roi_matches_full_channels(
        roi_image: &DecodedChannels,
        full: &DecodedChannels,
        roi: Rect,
    ) {
        assert_eq!(roi_image.width, roi.width);
        assert_eq!(roi_image.height, roi.height);
        assert_eq!(roi_image.color_channels, full.color_channels);
        assert_eq!(roi_image.alpha, full.alpha);
        assert_eq!(roi_image.bit_depth, full.bit_depth);
        assert_eq!(roi_image.channels.len(), full.channels.len());
        for (roi_channel, full_channel) in roi_image.channels.iter().zip(&full.channels) {
            assert_eq!(roi_channel.width, roi.width);
            assert_eq!(roi_channel.height, roi.height);
            assert_eq!(roi_channel.hshift, 0);
            assert_eq!(roi_channel.vshift, 0);
            assert_eq!(roi_channel.bit_depth, full_channel.bit_depth);
            match (&roi_channel.samples, &full_channel.samples) {
                (ChannelData::U8(roi_samples), ChannelData::U8(full_samples)) => {
                    assert_eq!(
                        roi_samples,
                        &window_u8(full_samples, full_channel.width, roi)
                    );
                }
                (ChannelData::U16(roi_samples), ChannelData::U16(full_samples)) => {
                    assert_eq!(
                        roi_samples,
                        &window_u16(full_samples, full_channel.width, roi)
                    );
                }
                _ => panic!("ROI and full channel bit depths differed"),
            }
        }
    }

    fn window_interleaved_u8(samples: &[u8], width: u32, channels: usize, roi: Rect) -> Vec<u8> {
        let mut output = Vec::with_capacity(roi.width as usize * roi.height as usize * channels);
        let row_stride = width as usize * channels;
        let x = roi.x as usize * channels;
        let copy_width = roi.width as usize * channels;
        for y in roi.y as usize..(roi.y + roi.height) as usize {
            let start = y * row_stride + x;
            output.extend_from_slice(&samples[start..start + copy_width]);
        }
        output
    }

    fn window_interleaved_u16(samples: &[u16], width: u32, channels: usize, roi: Rect) -> Vec<u16> {
        let mut output = Vec::with_capacity(roi.width as usize * roi.height as usize * channels);
        let row_stride = width as usize * channels;
        let x = roi.x as usize * channels;
        let copy_width = roi.width as usize * channels;
        for y in roi.y as usize..(roi.y + roi.height) as usize {
            let start = y * row_stride + x;
            output.extend_from_slice(&samples[start..start + copy_width]);
        }
        output
    }

    fn window_u8(samples: &[u8], width: u32, roi: Rect) -> Vec<u8> {
        let mut output = Vec::with_capacity(roi.width as usize * roi.height as usize);
        let width = width as usize;
        let x = roi.x as usize;
        let copy_width = roi.width as usize;
        for y in roi.y as usize..(roi.y + roi.height) as usize {
            let start = y * width + x;
            output.extend_from_slice(&samples[start..start + copy_width]);
        }
        output
    }

    fn window_u16(samples: &[u16], width: u32, roi: Rect) -> Vec<u16> {
        let mut output = Vec::with_capacity(roi.width as usize * roi.height as usize);
        let width = width as usize;
        let x = roi.x as usize;
        let copy_width = roi.width as usize;
        for y in roi.y as usize..(roi.y + roi.height) as usize {
            let start = y * width + x;
            output.extend_from_slice(&samples[start..start + copy_width]);
        }
        output
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

    fn write_split_vardct_source_ppm(path: &Path) {
        let width = 320u32;
        let height = 192u32;
        let mut bytes = format!("P6\n{width} {height}\n255\n").into_bytes();
        for y in 0..height {
            for x in 0..width {
                let checker = (((x / 16) ^ (y / 16)) & 1) * 48;
                bytes.push(((x * 255 / (width - 1)) ^ checker) as u8);
                bytes.push(((y * 255 / (height - 1)) ^ checker) as u8);
                bytes.push((((x + y) * 255 / (width + height - 2)) ^ checker) as u8);
            }
        }
        std::fs::write(path, bytes).unwrap();
    }

    fn write_roi_source_pgm(path: &Path) {
        let width = 64u32;
        let height = 64u32;
        let mut bytes = format!("P5\n{width} {height}\n255\n").into_bytes();
        bytes.extend((0..width * height).map(|index| (index.wrapping_mul(7) & 0xff) as u8));
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

    fn write_vardct_alpha_source_pam(path: &Path) -> Vec<u8> {
        let width = 320u32;
        let height = 192u32;
        let mut alpha = Vec::with_capacity(width as usize * height as usize);
        let mut bytes = format!(
            "P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n"
        )
        .into_bytes();
        for y in 0..height {
            for x in 0..width {
                let checker = (((x / 16) ^ (y / 16)) & 1) * 48;
                bytes.push(((x * 255 / (width - 1)) ^ checker) as u8);
                bytes.push(((y * 255 / (height - 1)) ^ checker) as u8);
                bytes.push((((x + y) * 255 / (width + height - 2)) ^ checker) as u8);
                let alpha_sample = ((x * 29 + y * 31 + 43) & 0xff) as u8;
                bytes.push(alpha_sample);
                alpha.push(alpha_sample);
            }
        }
        std::fs::write(path, bytes).unwrap();
        alpha
    }

    fn write_vardct_rgb_depth_source_pam(path: &Path) -> Vec<u8> {
        let width = 320u32;
        let height = 192u32;
        let mut depth = Vec::with_capacity(width as usize * height as usize);
        let mut bytes = format!(
            "P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB\nTUPLTYPE Depth\nENDHDR\n"
        )
        .into_bytes();
        for y in 0..height {
            for x in 0..width {
                let checker = (((x / 16) ^ (y / 16)) & 1) * 48;
                bytes.push(((x * 255 / (width - 1)) ^ checker) as u8);
                bytes.push(((y * 255 / (height - 1)) ^ checker) as u8);
                bytes.push((((x + y) * 255 / (width + height - 2)) ^ checker) as u8);
                let depth_sample = ((x * 37 + y * 41 + 73) & 0xff) as u8;
                bytes.push(depth_sample);
                depth.push(depth_sample);
            }
        }
        std::fs::write(path, bytes).unwrap();
        depth
    }

    fn write_rgb_depth_source_pam(path: &Path) {
        let width = 35u32;
        let height = 21u32;
        let mut bytes = format!(
            "P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB\nTUPLTYPE Depth\nENDHDR\n"
        )
        .into_bytes();
        for y in 0..height {
            for x in 0..width {
                bytes.push(((x * 11 + y * 3 + 17) & 0xff) as u8);
                bytes.push(((x * 7 + y * 13 + 29) & 0xff) as u8);
                bytes.push(((x * 19 + y * 5 + 43) & 0xff) as u8);
                bytes.push(((x * 37 + y * 41 + 73) & 0xff) as u8);
            }
        }
        std::fs::write(path, bytes).unwrap();
    }

    fn write_gray_alpha_source_pam(path: &Path) -> GrayAlphaPam {
        let width = 31u32;
        let height = 17u32;
        let mut bytes = format!(
            "P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH 2\nMAXVAL 255\nTUPLTYPE GRAYSCALE_ALPHA\nENDHDR\n"
        )
        .into_bytes();
        let mut gray_alpha = Vec::with_capacity(width as usize * height as usize * 2);
        let mut gray = Vec::with_capacity(width as usize * height as usize);
        let mut alpha = Vec::with_capacity(width as usize * height as usize);
        let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
        for y in 0..height {
            for x in 0..width {
                let gray_sample = ((x * 13 + y * 7 + 5) & 0xff) as u8;
                let alpha_sample = ((x * 29 + y * 31 + 43) & 0xff) as u8;
                bytes.push(gray_sample);
                bytes.push(alpha_sample);
                gray_alpha.extend_from_slice(&[gray_sample, alpha_sample]);
                gray.push(gray_sample);
                alpha.push(alpha_sample);
                rgba.extend_from_slice(&[gray_sample, gray_sample, gray_sample, alpha_sample]);
            }
        }
        std::fs::write(path, bytes).unwrap();
        GrayAlphaPam {
            width,
            height,
            gray_alpha,
            gray,
            alpha,
            rgba,
        }
    }

    fn write_gray_alpha_source_pam16(path: &Path) -> GrayAlphaPam16 {
        let width = 29u32;
        let height = 15u32;
        let mut bytes = format!(
            "P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH 2\nMAXVAL 65535\nTUPLTYPE GRAYSCALE_ALPHA\nENDHDR\n"
        )
        .into_bytes();
        let mut gray_alpha = Vec::with_capacity(width as usize * height as usize * 2);
        let mut gray = Vec::with_capacity(width as usize * height as usize);
        let mut alpha = Vec::with_capacity(width as usize * height as usize);
        let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
        for y in 0..height {
            for x in 0..width {
                let gray_sample = ((x * 3203 + y * 787 + 5) & 0xffff) as u16;
                let alpha_sample = ((x * 1733 + y * 2411 + 43) & 0xffff) as u16;
                bytes.extend_from_slice(&gray_sample.to_be_bytes());
                bytes.extend_from_slice(&alpha_sample.to_be_bytes());
                gray_alpha.extend_from_slice(&[gray_sample, alpha_sample]);
                gray.push(gray_sample);
                alpha.push(alpha_sample);
                rgba.extend_from_slice(&[gray_sample, gray_sample, gray_sample, alpha_sample]);
            }
        }
        std::fs::write(path, bytes).unwrap();
        GrayAlphaPam16 {
            width,
            height,
            gray_alpha,
            gray,
            alpha,
            rgba,
        }
    }

    fn write_alpha_depth_source_pam(path: &Path) -> AlphaDepthPam {
        let width = 23u32;
        let height = 19u32;
        let mut bytes = format!(
            "P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH 5\nMAXVAL 255\nTUPLTYPE RGB\nTUPLTYPE Depth\nTUPLTYPE Alpha\nENDHDR\n"
        )
        .into_bytes();
        let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
        let mut depth = Vec::with_capacity(width as usize * height as usize);
        for y in 0..height {
            for x in 0..width {
                let pixel = [
                    ((x * 11 + y * 3 + 17) & 0xff) as u8,
                    ((x * 7 + y * 13 + 29) & 0xff) as u8,
                    ((x * 19 + y * 5 + 43) & 0xff) as u8,
                    ((x * 23 + y * 31 + 61) & 0xff) as u8,
                ];
                let depth_sample = ((x * 37 + y * 41 + 73) & 0xff) as u8;
                bytes.extend_from_slice(&pixel[..3]);
                bytes.push(depth_sample);
                bytes.push(pixel[3]);
                rgba.extend_from_slice(&pixel);
                depth.push(depth_sample);
            }
        }
        std::fs::write(path, bytes).unwrap();
        AlphaDepthPam {
            width,
            height,
            rgba,
            depth,
        }
    }

    fn write_alpha_depth_source_pam16(path: &Path) -> AlphaDepthPam16 {
        let width = 21u32;
        let height = 13u32;
        let mut bytes = format!(
            "P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH 5\nMAXVAL 65535\nTUPLTYPE RGB\nTUPLTYPE Depth\nTUPLTYPE Alpha\nENDHDR\n"
        )
        .into_bytes();
        let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
        let mut depth = Vec::with_capacity(width as usize * height as usize);
        for y in 0..height {
            for x in 0..width {
                let pixel = [
                    ((x * 1021 + y * 137 + 17) & 0xffff) as u16,
                    ((x * 257 + y * 1879 + 29) & 0xffff) as u16,
                    ((x * 4093 + y * 383 + 43) & 0xffff) as u16,
                    ((x * 1723 + y * 3253 + 61) & 0xffff) as u16,
                ];
                let depth_sample = ((x * 2017 + y * 1543 + 73) & 0xffff) as u16;
                for sample in pixel[..3].iter().copied().chain([depth_sample, pixel[3]]) {
                    bytes.extend_from_slice(&sample.to_be_bytes());
                }
                rgba.extend_from_slice(&pixel);
                depth.push(depth_sample);
            }
        }
        std::fs::write(path, bytes).unwrap();
        AlphaDepthPam16 {
            width,
            height,
            rgba,
            depth,
        }
    }

    fn write_premultiplied_alpha_source_pam(path: &Path) -> Vec<u8> {
        let width = 17u32;
        let height = 11u32;
        let mut bytes = format!(
            "P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n"
        )
        .into_bytes();
        let mut expected_rgba = Vec::with_capacity(width as usize * height as usize * 4);
        for y in 0..height {
            for x in 0..width {
                let alpha = match (x + y * 3) % 9 {
                    0 => 0,
                    1 => 1,
                    2 => 17,
                    3 => 64,
                    4 => 128,
                    5 => 191,
                    6 => 254,
                    _ => 255,
                };
                let straight = [
                    ((x * 23 + y * 7 + 11) & 0xff) as u8,
                    ((x * 5 + y * 29 + 37) & 0xff) as u8,
                    ((x * 13 + y * 17 + 91) & 0xff) as u8,
                ];
                for sample in straight {
                    let premultiplied = ((u32::from(sample) * alpha + 127) / 255) as u8;
                    bytes.push(premultiplied);
                    expected_rgba.push(unpremultiply_sample_to(
                        u32::from(premultiplied),
                        8,
                        alpha,
                        8,
                        u8::MAX as u32,
                    ) as u8);
                }
                bytes.push(alpha as u8);
                expected_rgba.push(alpha as u8);
            }
        }
        std::fs::write(path, bytes).unwrap();
        expected_rgba
    }

    fn write_premultiplied_alpha_source_pam16(path: &Path) -> Vec<u16> {
        let width = 19u32;
        let height = 15u32;
        let mut bytes = format!(
            "P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH 4\nMAXVAL 65535\nTUPLTYPE RGB_ALPHA\nENDHDR\n"
        )
        .into_bytes();
        let mut expected_rgba = Vec::with_capacity(width as usize * height as usize * 4);
        for y in 0..height {
            for x in 0..width {
                let alpha = match (x + y * 5) % 11 {
                    0 => 0,
                    1 => 1,
                    2 => 257,
                    3 => 4096,
                    4 => 16_384,
                    5 => 32_768,
                    6 => 49_152,
                    7 => 65_534,
                    _ => 65_535,
                };
                let straight = [
                    ((x * 2971 + y * 359 + 11) & 0xffff) as u16,
                    ((x * 811 + y * 2371 + 37) & 0xffff) as u16,
                    ((x * 1237 + y * 1597 + 91) & 0xffff) as u16,
                ];
                for sample in straight {
                    let premultiplied = ((u32::from(sample) * alpha + 32_767) / 65_535) as u16;
                    bytes.extend_from_slice(&premultiplied.to_be_bytes());
                    expected_rgba.push(unpremultiply_sample_to(
                        u32::from(premultiplied),
                        16,
                        alpha,
                        16,
                        u16::MAX as u32,
                    ) as u16);
                }
                bytes.extend_from_slice(&(alpha as u16).to_be_bytes());
                expected_rgba.push(alpha as u16);
            }
        }
        std::fs::write(path, bytes).unwrap();
        expected_rgba
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
