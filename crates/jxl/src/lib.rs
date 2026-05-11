//! Public Rust-native JPEG XL API.
//!
//! The API is intentionally small while the decoder is being built out. It
//! exposes stable metadata inspection now and leaves room for future streaming
//! decode, region decode, and pixel-output builders without committing to a
//! C-style event API.

pub use jxl_codec::{
    BasicInfo, BitDepth, BlendMode, BlendingInfo, BoxRecord, ColorEncoding, ColorSpace, Container,
    CustomTransformData, DequantizedSplineMetadata, Error, ExtraChannelInfo, ExtraChannelType,
    FileFormat, FrameData, FrameEncoding, FrameFeatureMetadata, FrameGroupLayout, FrameHeader,
    FrameSection, FrameSectionKind, FrameToc, FrameType, ImageMetadata, MaTree, MaTreeNode,
    ModularChannel, ModularChannelPlan, ModularDecodedChannel, ModularDecodedGroup,
    ModularFrameMetadata, ModularGlobalSection, ModularGroupChannelPlan, ModularGroupHeader,
    ModularImage, ModularImageChannel, ModularPredictor, ModularResiduals, ModularSectionMetadata,
    ModularTransform, ModularTreeMetadata, NoiseFrameMetadata, OpsinInverseMatrix, Orientation,
    Primaries, QuantizedSplineMetadata, RenderingIntent, Result, SplineFloatPoint,
    SplineFrameMetadata, SplinePoint, SplineRenderPlan, SplineSegmentMetadata, SqueezeParams,
    TocEntry, ToneMapping, TransferFunction, TransformId, VarDctBlockContextMapMetadata,
    VarDctColorCorrelationMetadata, VarDctDcDequantMetadata, VarDctDcGroupCursorMetadata,
    VarDctDcGroupMetadata, VarDctDcGroupPayloadMetadata, VarDctDecodePlan, VarDctFrameMetadata,
    VarDctGlobalCursorMetadata, VarDctGlobalMetadata, VarDctGroupMetadata,
    VarDctGroupPayloadMetadata, VarDctGroupSectionMetadata, VarDctPassGroupPayloadMetadata,
    VarDctPassGroupSectionMetadata, VarDctQuantizerMetadata, VarDctSectionMetadata,
    VarDctSectionPayloadMetadata, WeightedPredictorHeader, WhitePoint,
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
pub struct RgbaFrame {
    pub x: i32,
    pub y: i32,
    pub duration: u32,
    pub timecode: u32,
    pub image: RgbaImage,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearRgbImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<f32>,
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

#[derive(Debug, Clone, PartialEq)]
pub enum DecodedOutput {
    Channels(DecodedChannels),
    Interleaved(DecodedImage),
    Rgba8(RgbaImage),
    Rgba16(Rgba16Image),
    LinearRgb(LinearRgbImage),
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
    /// Maximum bytes allowed for the returned decoded sample buffers.
    ///
    /// This is a final-output guard for the buffered API. It does not yet
    /// account for all parser and reconstruction working memory.
    pub memory_limit: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodeOutput {
    #[default]
    Channels,
    Interleaved,
    Rgba8,
    Rgba16,
    LinearRgb,
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

    pub fn without_roi(mut self) -> Self {
        self.options.roi = None;
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

    pub fn without_memory_limit(mut self) -> Self {
        self.options.memory_limit = None;
        self
    }

    pub fn vardct_pass(mut self, pass: usize) -> Self {
        self.options.vardct_pass = Some(pass);
        self
    }

    pub fn final_vardct_pass(mut self) -> Self {
        self.options.vardct_pass = None;
        self
    }

    /// Decodes using the configured [`DecodeOutput`] mode.
    pub fn decode_output(&self, input: &[u8]) -> Result<DecodedOutput> {
        match self.options.output {
            DecodeOutput::Channels => self.decode_channels(input).map(DecodedOutput::Channels),
            DecodeOutput::Interleaved => self.decode(input).map(DecodedOutput::Interleaved),
            DecodeOutput::Rgba8 => self.decode_rgba8(input).map(DecodedOutput::Rgba8),
            DecodeOutput::Rgba16 => self.decode_rgba16(input).map(DecodedOutput::Rgba16),
            DecodeOutput::LinearRgb => self.decode_linear_rgb(input).map(DecodedOutput::LinearRgb),
        }
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
        let channels =
            decode_channels_buffered(input, self.codec_config(), self.options.vardct_pass)?;
        enforce_decoded_channels_memory_limit(&channels, self.options.memory_limit)?;
        Ok(channels)
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
        let image = decode_buffered(input, self.codec_config(), self.options.vardct_pass)?;
        enforce_decoded_image_memory_limit(&image, self.options.memory_limit)?;
        Ok(image)
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
        let image = decode_rgba8_buffered(input, self.codec_config(), self.options.vardct_pass)?;
        enforce_memory_limit(image.pixels.len(), self.options.memory_limit)?;
        Ok(image)
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
        let image = decode_rgba16_buffered(input, self.codec_config(), self.options.vardct_pass)?;
        enforce_memory_limit(
            checked_sample_bytes(image.pixels.len(), 2)?,
            self.options.memory_limit,
        )?;
        Ok(image)
    }

    /// Decodes each frame rectangle to interleaved straight-alpha RGBA8.
    ///
    /// This is a raw frame-sequence API: frames are returned with their
    /// codestream origin, duration, and timecode, but are not composited with
    /// previous frames. Blended animation output is not implemented yet.
    ///
    /// Non-animated images return a single frame at origin `(0, 0)`.
    pub fn decode_rgba8_frames(&self, input: &[u8]) -> Result<Vec<RgbaFrame>> {
        self.validate_shared_options()?;
        let frames =
            decode_rgba8_frames_buffered(input, self.codec_config(), self.options.vardct_pass)?;
        let required = frames.iter().try_fold(0usize, |sum, frame| {
            sum.checked_add(frame.image.pixels.len())
                .ok_or(Error::InvalidCodestream("decoded image size overflow"))
        })?;
        enforce_memory_limit(required, self.options.memory_limit)?;
        Ok(frames)
    }

    /// Decodes an animation to composited full-canvas RGBA8 frames.
    ///
    /// This currently supports modular frame sequences using `Replace` and
    /// source-over `Blend` against the previous canvas. Other blend modes
    /// return [`Error::Unsupported`] until their compositing rules are
    /// implemented.
    pub fn decode_rgba8_animation(&self, input: &[u8]) -> Result<Vec<RgbaFrame>> {
        self.validate_shared_options()?;
        let frames =
            decode_rgba8_animation_buffered(input, self.codec_config(), self.options.vardct_pass)?;
        let required = frames.iter().try_fold(0usize, |sum, frame| {
            sum.checked_add(frame.image.pixels.len())
                .ok_or(Error::InvalidCodestream("decoded image size overflow"))
        })?;
        enforce_memory_limit(required, self.options.memory_limit)?;
        Ok(frames)
    }

    /// Decodes color to interleaved linear RGB `f32` samples.
    ///
    /// XYB images are converted with the JPEG XL inverse opsin path. Non-XYB
    /// modular RGB/gray images are supported when their encoded color space is
    /// the default sRGB/D65 profile or linear transfer. This path intentionally
    /// does not yet perform full JPEG XL output color management; ICC,
    /// wide-gamut, custom-primary, and non-XYB VarDCT outputs currently return
    /// [`Error::Unsupported`].
    pub fn decode_linear_rgb(&self, input: &[u8]) -> Result<LinearRgbImage> {
        self.validate_shared_options()?;
        let image =
            decode_linear_rgb_buffered(input, self.codec_config(), self.options.vardct_pass)?;
        enforce_memory_limit(
            checked_sample_bytes(image.pixels.len(), std::mem::size_of::<f32>())?,
            self.options.memory_limit,
        )?;
        Ok(image)
    }

    fn validate_shared_options(&self) -> Result<()> {
        if self.options.threads == ThreadingMode::Threads(0) {
            return Err(Error::Unsupported("zero decoder threads"));
        }
        Ok(())
    }

    fn codec_config(&self) -> jxl_codec::DecodeConfig {
        jxl_codec::DecodeConfig {
            modular_group_execution: modular_group_execution_for_threading(self.options.threads),
            region: self.options.roi.map(|roi| jxl_codec::ImageRegion {
                x: roi.x,
                y: roi.y,
                width: roi.width,
                height: roi.height,
            }),
        }
    }
}

fn modular_group_execution_for_threading(
    threading: ThreadingMode,
) -> jxl_codec::ModularGroupExecution {
    match threading {
        ThreadingMode::Single => jxl_codec::ModularGroupExecution::Serial,
        ThreadingMode::Threads(threads) => {
            jxl_codec::ModularGroupExecution::RequestedThreads(threads)
        }
        ThreadingMode::Auto => {
            let threads = std::thread::available_parallelism()
                .map(|threads| threads.get())
                .unwrap_or(1);
            if threads > 1 {
                jxl_codec::ModularGroupExecution::RequestedThreads(threads)
            } else {
                jxl_codec::ModularGroupExecution::Serial
            }
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
        frames: codestream.frames,
        frame_data: codestream.frame_data,
        modular_frames: codestream.modular_frames,
        vardct_plans: codestream.vardct_plans,
        vardct_frames: codestream.vardct_frames,
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

pub fn decode_with_options(input: &[u8], options: DecodeOptions) -> Result<DecodedOutput> {
    Decoder::with_options(options).decode_output(input)
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

pub fn decode_rgba8_frames(input: &[u8]) -> Result<Vec<RgbaFrame>> {
    Decoder::new().decode_rgba8_frames(input)
}

pub fn decode_rgba8_animation(input: &[u8]) -> Result<Vec<RgbaFrame>> {
    Decoder::new().decode_rgba8_animation(input)
}

pub fn decode_linear_rgb(input: &[u8]) -> Result<LinearRgbImage> {
    Decoder::new().decode_linear_rgb(input)
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

fn first_frame_color_transform(
    codestream: &jxl_codec::Codestream,
) -> Result<jxl_codec::ColorTransform> {
    Ok(codestream
        .first_frame
        .as_ref()
        .ok_or(Error::Unsupported("image has no decoded frame"))?
        .color_transform)
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
    decoded_channels_from_modular_frame(&codestream.metadata, modular, region)
}

fn decoded_channels_from_modular_frame(
    metadata: &ImageMetadata,
    modular: &ModularFrameMetadata,
    region: Option<jxl_codec::ImageRegion>,
) -> Result<DecodedChannels> {
    if let Some(error) = &modular.image_error {
        return Err(error.clone());
    }
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
    let color_channels = metadata.num_color_channels() as usize;
    let bit_depth = metadata.bit_depth.bits_per_sample;
    if metadata.bit_depth.floating_point_sample {
        return Err(Error::Unsupported("floating-point sample output"));
    }
    if bit_depth > 16 {
        return Err(Error::Unsupported("integer sample depths above 16 bits"));
    }
    let channel_bit_depths = decoded_channel_bit_depths(metadata, color_channels)?;
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
        alpha: raw_alpha_info(metadata)?,
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
            let needs_full_alpha = vardct_alpha_is_subsampled(&codestream)?;
            let decode_region = if needs_full_alpha {
                None
            } else {
                config.region
            };
            let channels = if raw_alpha_info(&codestream.metadata)?
                .is_some_and(|alpha| alpha.bit_depth > 8)
            {
                decode_vardct_channels_codestream_rgb16(&codestream, decode_region, vardct_pass)?
            } else {
                decode_vardct_channels_codestream(&codestream, decode_region, vardct_pass)?
            };
            let image = decode_buffered_channels_with_transform_data(
                channels,
                alpha_channel_index,
                Some(&codestream.transform_data),
            )?;
            return if let (true, Some(region)) = (needs_full_alpha, config.region) {
                crop_decoded_image(image, region)
            } else {
                Ok(image)
            };
        }
        reject_vardct_alpha_output(&codestream.metadata)?;
        let orientation = codestream.metadata.orientation;
        let color_channels = codestream.metadata.num_color_channels() as usize;
        let image = decoded_image_from_vardct_srgb8(
            vardct_srgb8_image_from_codestream(&codestream, config.region, vardct_pass)?,
            color_channels,
        )?;
        return orient_decoded_image(image, orientation);
    }
    reject_vardct_pass_for_non_vardct(vardct_pass)?;

    if let Some(region) = config.region
        && raw_alpha_channel_index(&codestream.metadata)?.is_some()
    {
        let (_, full_codestream) = jxl_codec::parse_file(input)?;
        let image = decode_buffered_codestream(full_codestream, None)?;
        return crop_decoded_image(image, region);
    }

    decode_buffered_codestream(codestream, config.region)
}

fn decode_buffered_codestream(
    codestream: jxl_codec::Codestream,
    region: Option<jxl_codec::ImageRegion>,
) -> Result<DecodedImage> {
    let orientation = codestream.metadata.orientation;
    let alpha_channel_index = raw_alpha_channel_index(&codestream.metadata)?;
    let transform_data = codestream.transform_data.clone();
    let color_transform = first_frame_color_transform(&codestream)?;
    if color_transform == jxl_codec::ColorTransform::Xyb {
        let channels = if raw_alpha_info(&codestream.metadata)?
            .is_some_and(|alpha| alpha.bit_depth > u8::BITS)
        {
            modular_xyb_decoded_channels_srgb16_from_codestream(&codestream, region)?
        } else {
            modular_xyb_decoded_channels_srgb8_from_codestream(&codestream, region)?
        };
        return orient_decoded_image(
            decode_buffered_channels_with_transform_data(channels, alpha_channel_index, None)?,
            orientation,
        );
    }
    let channels = decode_channels_codestream(codestream, None, None)?;
    orient_decoded_image(
        decode_buffered_channels_with_transform_data(
            channels,
            alpha_channel_index,
            Some(&transform_data),
        )?,
        orientation,
    )
}

#[cfg(test)]
fn decode_buffered_channels(
    channels: DecodedChannels,
    alpha_channel_index: Option<usize>,
) -> Result<DecodedImage> {
    decode_buffered_channels_with_transform_data(channels, alpha_channel_index, None)
}

fn decode_buffered_channels_with_transform_data(
    channels: DecodedChannels,
    alpha_channel_index: Option<usize>,
    transform_data: Option<&CustomTransformData>,
) -> Result<DecodedImage> {
    let alpha = decode_interleaved_alpha(&channels, alpha_channel_index)?;
    let output_channel_indices = interleaved_channel_indices(&channels, alpha_channel_index)?;
    validate_interleaved_channel_geometry(&channels, &output_channel_indices, alpha_channel_index)?;
    let output_bit_depth = interleaved_output_bit_depth(&channels, &output_channel_indices)?;

    if output_bit_depth <= 8 {
        Ok(DecodedImage {
            width: channels.width,
            height: channels.height,
            color_channels: channels.color_channels,
            alpha,
            bit_depth: output_bit_depth,
            pixels: PixelData::U8(interleave_channel_u8(
                &channels,
                &output_channel_indices,
                transform_data,
                output_bit_depth,
            )?),
        })
    } else {
        Ok(DecodedImage {
            width: channels.width,
            height: channels.height,
            color_channels: channels.color_channels,
            alpha,
            bit_depth: output_bit_depth,
            pixels: PixelData::U16(interleave_channel_u16(
                &channels,
                &output_channel_indices,
                transform_data,
                output_bit_depth,
            )?),
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
            let needs_full_alpha = vardct_alpha_is_subsampled(&codestream)?;
            let decode_region = if needs_full_alpha {
                None
            } else {
                config.region
            };
            let channels =
                decode_vardct_channels_codestream(&codestream, decode_region, vardct_pass)?;
            let image = rgba8_from_decoded_channels_with_transform_data(
                &channels,
                alpha_channel_index,
                Some(&codestream.transform_data),
            )?;
            return if let (true, Some(region)) = (needs_full_alpha, config.region) {
                crop_rgba8_image(image, region)
            } else {
                Ok(image)
            };
        }
        reject_vardct_alpha_output(&codestream.metadata)?;
        if codestream.metadata.num_color_channels() == 1 {
            let channels =
                decode_vardct_channels_codestream(&codestream, config.region, vardct_pass)?;
            return rgba8_from_decoded_channels_with_transform_data(
                &channels,
                alpha_channel_index,
                Some(&codestream.transform_data),
            );
        }
        let orientation = codestream.metadata.orientation;
        let image = rgba8_from_vardct_srgb8(vardct_srgb8_image_from_codestream(
            &codestream,
            config.region,
            vardct_pass,
        )?)?;
        return orient_rgba8(image, orientation);
    }
    reject_vardct_pass_for_non_vardct(vardct_pass)?;

    if let Some(region) = config.region
        && raw_alpha_channel_index(&codestream.metadata)?.is_some()
    {
        let (_, full_codestream) = jxl_codec::parse_file(input)?;
        let image = rgba8_from_modular_codestream(full_codestream, None)?;
        return crop_rgba8_image(image, region);
    }

    rgba8_from_modular_codestream(codestream, config.region)
}

fn decode_rgba8_frames_buffered(
    input: &[u8],
    config: jxl_codec::DecodeConfig,
    vardct_pass: Option<usize>,
) -> Result<Vec<RgbaFrame>> {
    if config.region.is_some() {
        return Err(Error::Unsupported("region-of-interest frame decode"));
    }
    reject_vardct_pass_for_non_vardct(vardct_pass)?;
    let (_, codestream) = jxl_codec::parse_file(input)?;
    if codestream.frames.is_empty() {
        return Err(Error::Unsupported("image has no decoded frame"));
    }
    if codestream.frames.len() == 1 && codestream.frames[0].encoding == FrameEncoding::VarDct {
        let image = decode_rgba8(input)?;
        return Ok(vec![RgbaFrame {
            x: 0,
            y: 0,
            duration: codestream.frames[0].animation_frame.duration,
            timecode: codestream.frames[0].animation_frame.timecode,
            image,
        }]);
    }

    let alpha_channel_index = raw_alpha_channel_index(&codestream.metadata)?;
    let transform_data = codestream.transform_data.clone();
    let mut frames = Vec::with_capacity(codestream.frames.len());
    for (frame, modular) in codestream.frames.iter().zip(&codestream.modular_frames) {
        if frame.encoding != FrameEncoding::Modular {
            return Err(Error::Unsupported("VarDCT frame sequence decode"));
        }
        if frame.color_transform == jxl_codec::ColorTransform::Xyb {
            return Err(Error::Unsupported("XYB frame sequence decode"));
        }
        let modular = modular
            .as_ref()
            .ok_or(Error::Unsupported("modular frame metadata"))?;
        let channels = decoded_channels_from_modular_frame(&codestream.metadata, modular, None)?;
        let image = rgba8_from_decoded_channels_with_transform_data(
            &channels,
            alpha_channel_index,
            Some(&transform_data),
        )?;
        frames.push(RgbaFrame {
            x: frame.frame_origin.x0,
            y: frame.frame_origin.y0,
            duration: frame.animation_frame.duration,
            timecode: frame.animation_frame.timecode,
            image,
        });
    }
    Ok(frames)
}

fn decode_rgba8_animation_buffered(
    input: &[u8],
    config: jxl_codec::DecodeConfig,
    vardct_pass: Option<usize>,
) -> Result<Vec<RgbaFrame>> {
    if config.region.is_some() {
        return Err(Error::Unsupported("region-of-interest animation decode"));
    }
    reject_vardct_pass_for_non_vardct(vardct_pass)?;
    let (_, codestream) = jxl_codec::parse_file(input)?;
    if !codestream.basic_info.have_animation {
        let image = decode_rgba8(input)?;
        return Ok(vec![RgbaFrame {
            x: 0,
            y: 0,
            duration: codestream
                .first_frame
                .as_ref()
                .map(|frame| frame.animation_frame.duration)
                .unwrap_or(0),
            timecode: codestream
                .first_frame
                .as_ref()
                .map(|frame| frame.animation_frame.timecode)
                .unwrap_or(0),
            image,
        }]);
    }

    let raw_frames = decode_rgba8_frames_buffered(input, config, vardct_pass)?;
    let canvas_width = codestream.basic_info.width;
    let canvas_height = codestream.basic_info.height;
    let canvas_len = (canvas_width as usize)
        .checked_mul(canvas_height as usize)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    let mut canvas = vec![0u8; canvas_len];
    let mut output = Vec::with_capacity(raw_frames.len());
    for (raw_frame, frame_header) in raw_frames.into_iter().zip(&codestream.frames) {
        match frame_header.blending_info.mode {
            BlendMode::Replace => {
                composite_replace_rgba8(&mut canvas, canvas_width, canvas_height, &raw_frame)?;
            }
            BlendMode::Blend => {
                if frame_header.blending_info.source != 1 {
                    return Err(Error::Unsupported("animated blend source"));
                }
                if frame_header.blending_info.alpha_channel != 0 || frame_header.blending_info.clamp
                {
                    return Err(Error::Unsupported("animated blend parameters"));
                }
                composite_blend_rgba8(&mut canvas, canvas_width, canvas_height, &raw_frame)?;
            }
            _ => return Err(Error::Unsupported("animated blend mode")),
        }
        output.push(RgbaFrame {
            x: 0,
            y: 0,
            duration: raw_frame.duration,
            timecode: raw_frame.timecode,
            image: RgbaImage {
                width: canvas_width,
                height: canvas_height,
                pixels: canvas.clone(),
            },
        });
    }
    Ok(output)
}

fn composite_replace_rgba8(
    canvas: &mut [u8],
    canvas_width: u32,
    canvas_height: u32,
    frame: &RgbaFrame,
) -> Result<()> {
    let region = compositing_region(canvas, canvas_width, canvas_height, frame)?;
    if region.dst_x0 >= region.dst_x1 || region.dst_y0 >= region.dst_y1 {
        return Ok(());
    }
    let copy_width = (region.dst_x1 - region.dst_x0) as usize * 4;
    for dst_y in region.dst_y0..region.dst_y1 {
        let src_x = (region.dst_x0 as i32 - frame.x) as u32;
        let src_y = (dst_y as i32 - frame.y) as u32;
        let src_start = (src_y as usize * frame.image.width as usize + src_x as usize) * 4;
        let dst_start = (dst_y as usize * canvas_width as usize + region.dst_x0 as usize) * 4;
        canvas[dst_start..dst_start + copy_width]
            .copy_from_slice(&frame.image.pixels[src_start..src_start + copy_width]);
    }
    Ok(())
}

fn composite_blend_rgba8(
    canvas: &mut [u8],
    canvas_width: u32,
    canvas_height: u32,
    frame: &RgbaFrame,
) -> Result<()> {
    let region = compositing_region(canvas, canvas_width, canvas_height, frame)?;
    for dst_y in region.dst_y0..region.dst_y1 {
        let src_x = (region.dst_x0 as i32 - frame.x) as u32;
        let src_y = (dst_y as i32 - frame.y) as u32;
        let src_start = (src_y as usize * frame.image.width as usize + src_x as usize) * 4;
        let dst_start = (dst_y as usize * canvas_width as usize + region.dst_x0 as usize) * 4;
        for column in 0..(region.dst_x1 - region.dst_x0) as usize {
            let src = src_start + column * 4;
            let dst = dst_start + column * 4;
            let alpha = u32::from(frame.image.pixels[src + 3]);
            for channel in 0..3 {
                canvas[dst + channel] = blend_u8(
                    frame.image.pixels[src + channel],
                    canvas[dst + channel],
                    alpha,
                );
            }
            canvas[dst + 3] = blend_alpha_u8(frame.image.pixels[src + 3], canvas[dst + 3]);
        }
    }
    Ok(())
}

fn blend_u8(source: u8, destination: u8, alpha: u32) -> u8 {
    ((u32::from(source) * alpha + u32::from(destination) * (255 - alpha) + 127) / 255) as u8
}

fn blend_alpha_u8(source: u8, destination: u8) -> u8 {
    let source = u32::from(source);
    let destination = u32::from(destination);
    (source + (destination * (255 - source) + 127) / 255).min(255) as u8
}

struct CompositingRegion {
    dst_x0: u32,
    dst_y0: u32,
    dst_x1: u32,
    dst_y1: u32,
}

fn compositing_region(
    canvas: &[u8],
    canvas_width: u32,
    canvas_height: u32,
    frame: &RgbaFrame,
) -> Result<CompositingRegion> {
    let frame_width = frame.image.width;
    let frame_height = frame.image.height;
    validate_rgba8_buffer(&frame.image)?;
    let expected_canvas_len = (canvas_width as usize)
        .checked_mul(canvas_height as usize)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    if canvas.len() != expected_canvas_len {
        return Err(Error::InvalidCodestream("decoded pixel count mismatch"));
    }

    let frame_x0 = i64::from(frame.x);
    let frame_y0 = i64::from(frame.y);
    let frame_x1 = frame_x0 + i64::from(frame_width);
    let frame_y1 = frame_y0 + i64::from(frame_height);
    let dst_x0 = frame_x0.max(0).min(i64::from(canvas_width)) as u32;
    let dst_y0 = frame_y0.max(0).min(i64::from(canvas_height)) as u32;
    let dst_x1 = frame_x1.max(0).min(i64::from(canvas_width)) as u32;
    let dst_y1 = frame_y1.max(0).min(i64::from(canvas_height)) as u32;
    Ok(CompositingRegion {
        dst_x0,
        dst_y0,
        dst_x1,
        dst_y1,
    })
}

fn validate_rgba8_buffer(image: &RgbaImage) -> Result<()> {
    let expected = (image.width as usize)
        .checked_mul(image.height as usize)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    if image.pixels.len() != expected {
        return Err(Error::InvalidCodestream("decoded pixel count mismatch"));
    }
    Ok(())
}

fn decode_linear_rgb_buffered(
    input: &[u8],
    config: jxl_codec::DecodeConfig,
    vardct_pass: Option<usize>,
) -> Result<LinearRgbImage> {
    let (_, codestream) = parse_file_for_public_pixel_decode(input, config)?;
    if first_frame_encoding(&codestream)? != FrameEncoding::VarDct {
        reject_vardct_pass_for_non_vardct(vardct_pass)?;
        if first_frame_color_transform(&codestream)? != jxl_codec::ColorTransform::Xyb {
            let orientation = codestream.metadata.orientation;
            return orient_linear_rgb(
                modular_non_xyb_linear_rgb_from_codestream(codestream, config.region)?,
                orientation,
            );
        }
        let orientation = codestream.metadata.orientation;
        return orient_linear_rgb(
            modular_xyb_linear_rgb_from_codestream(&codestream, config.region)?,
            orientation,
        );
    }
    if first_frame_color_transform(&codestream)? != jxl_codec::ColorTransform::Xyb {
        return Err(Error::Unsupported("linear RGB non-XYB output"));
    }

    let orientation = codestream.metadata.orientation;
    orient_linear_rgb(
        vardct_linear_rgb_image_from_codestream(&codestream, config.region, vardct_pass)?,
        orientation,
    )
}

fn rgba8_from_modular_codestream(
    codestream: jxl_codec::Codestream,
    region: Option<jxl_codec::ImageRegion>,
) -> Result<RgbaImage> {
    let orientation = codestream.metadata.orientation;
    let alpha_channel_index = raw_alpha_channel_index(&codestream.metadata)?;
    let transform_data = codestream.transform_data.clone();
    let color_transform = first_frame_color_transform(&codestream)?;
    let channels = if color_transform == jxl_codec::ColorTransform::Xyb {
        modular_xyb_decoded_channels_srgb8_from_codestream(&codestream, region)?
    } else {
        decode_channels_codestream(codestream, None, None)?
    };
    orient_rgba8(
        rgba8_from_decoded_channels_with_transform_data(
            &channels,
            alpha_channel_index,
            (color_transform != jxl_codec::ColorTransform::Xyb).then_some(&transform_data),
        )?,
        orientation,
    )
}

#[cfg(test)]
fn rgba8_from_decoded_channels(
    channels: &DecodedChannels,
    alpha_channel_index: Option<usize>,
) -> Result<RgbaImage> {
    rgba8_from_decoded_channels_with_transform_data(channels, alpha_channel_index, None)
}

fn rgba8_from_decoded_channels_with_transform_data(
    channels: &DecodedChannels,
    alpha_channel_index: Option<usize>,
    transform_data: Option<&CustomTransformData>,
) -> Result<RgbaImage> {
    let output_channel_indices = rgba_channel_indices(channels, alpha_channel_index)?;
    let pixels = rgba8_from_channel_indices(channels, &output_channel_indices, transform_data)?;
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

fn linear_rgb_from_vardct_rgb(image: jxl_codec::VarDctRgbImage) -> Result<LinearRgbImage> {
    let sample_count = vardct_srgb_sample_count(image.width, image.height)?;
    if image
        .channels
        .iter()
        .any(|channel| channel.len() != sample_count)
    {
        return Err(Error::InvalidCodestream("decoded pixel count mismatch"));
    }

    let mut pixels = Vec::with_capacity(sample_count * 3);
    for index in 0..sample_count {
        pixels.push(image.channels[0][index]);
        pixels.push(image.channels[1][index]);
        pixels.push(image.channels[2][index]);
    }
    Ok(LinearRgbImage {
        width: image.width,
        height: image.height,
        pixels,
    })
}

fn decoded_image_from_vardct_srgb8(
    image: jxl_codec::VarDctSrgb8Image,
    color_channels: usize,
) -> Result<DecodedImage> {
    let sample_count = vardct_srgb_sample_count(image.width, image.height)?;
    if image.pixels.len() != sample_count * 3 {
        return Err(Error::InvalidCodestream("decoded pixel count mismatch"));
    }
    let pixels = match color_channels {
        1 => image
            .pixels
            .chunks_exact(3)
            .map(|pixel| pixel[0])
            .collect::<Vec<_>>(),
        3 => image.pixels,
        _ => return Err(Error::Unsupported("unsupported color channel count")),
    };

    Ok(DecodedImage {
        width: image.width,
        height: image.height,
        color_channels,
        alpha: None,
        bit_depth: 8,
        pixels: PixelData::U8(pixels),
    })
}

fn decoded_channels_from_vardct_srgb8(
    image: jxl_codec::VarDctSrgb8Image,
    color_channels: usize,
) -> Result<DecodedChannels> {
    let sample_count = vardct_srgb_sample_count(image.width, image.height)?;
    if image.pixels.len() != sample_count * 3 {
        return Err(Error::InvalidCodestream("decoded pixel count mismatch"));
    }
    if !matches!(color_channels, 1 | 3) {
        return Err(Error::Unsupported("unsupported color channel count"));
    }
    let mut channels = (0..color_channels)
        .map(|_| Vec::with_capacity(sample_count))
        .collect::<Vec<_>>();
    for pixel in image.pixels.chunks_exact(3) {
        channels[0].push(pixel[0]);
        if color_channels == 3 {
            channels[1].push(pixel[1]);
            channels[2].push(pixel[2]);
        }
    }

    Ok(DecodedChannels {
        width: image.width,
        height: image.height,
        color_channels,
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
    color_channels: usize,
) -> Result<DecodedChannels> {
    let sample_count = vardct_srgb_sample_count(image.width, image.height)?;
    if image.pixels.len() != sample_count * 3 {
        return Err(Error::InvalidCodestream("decoded pixel count mismatch"));
    }
    if !matches!(color_channels, 1 | 3) {
        return Err(Error::Unsupported("unsupported color channel count"));
    }
    let mut channels = (0..color_channels)
        .map(|_| Vec::with_capacity(sample_count))
        .collect::<Vec<_>>();
    for pixel in image.pixels.chunks_exact(3) {
        channels[0].push(pixel[0]);
        if color_channels == 3 {
            channels[1].push(pixel[1]);
            channels[2].push(pixel[2]);
        }
    }

    Ok(DecodedChannels {
        width: image.width,
        height: image.height,
        color_channels,
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

fn modular_xyb_decoded_channels_srgb8_from_codestream(
    codestream: &jxl_codec::Codestream,
    region: Option<jxl_codec::ImageRegion>,
) -> Result<DecodedChannels> {
    let opsin = jxl_codec::xyb_opsin_params(&codestream.metadata, &codestream.transform_data);
    let srgb = jxl_codec::xyb_image_to_srgb8_with_variant(
        &modular_codestream_to_xyb_image(codestream, region)?,
        &opsin,
        jxl_codec::VarDctXybInverseVariant::BMinusBias,
    );
    let mut channels = decoded_channels_from_vardct_srgb8(srgb, 3)?;
    append_modular_extra_channels(&mut channels, codestream)?;
    Ok(channels)
}

fn modular_xyb_decoded_channels_srgb16_from_codestream(
    codestream: &jxl_codec::Codestream,
    region: Option<jxl_codec::ImageRegion>,
) -> Result<DecodedChannels> {
    let opsin = jxl_codec::xyb_opsin_params(&codestream.metadata, &codestream.transform_data);
    let srgb = jxl_codec::xyb_image_to_srgb16_with_variant(
        &modular_codestream_to_xyb_image(codestream, region)?,
        &opsin,
        jxl_codec::VarDctXybInverseVariant::BMinusBias,
    );
    let mut channels = decoded_channels_from_vardct_srgb16(srgb, 3)?;
    append_modular_extra_channels(&mut channels, codestream)?;
    Ok(channels)
}

fn modular_xyb_linear_rgb_from_codestream(
    codestream: &jxl_codec::Codestream,
    region: Option<jxl_codec::ImageRegion>,
) -> Result<LinearRgbImage> {
    let opsin = jxl_codec::xyb_opsin_params(&codestream.metadata, &codestream.transform_data);
    let rgb = jxl_codec::xyb_image_to_linear_rgb_with_variant(
        &modular_codestream_to_xyb_image(codestream, region)?,
        &opsin,
        jxl_codec::VarDctXybInverseVariant::BMinusBias,
    );
    linear_rgb_from_vardct_rgb(rgb)
}

fn modular_non_xyb_linear_rgb_from_codestream(
    codestream: jxl_codec::Codestream,
    region: Option<jxl_codec::ImageRegion>,
) -> Result<LinearRgbImage> {
    let color_encoding = codestream.metadata.color_encoding.clone();
    let color_channels = codestream.metadata.num_color_channels() as usize;
    validate_linear_rgb_color_encoding(&color_encoding)?;
    let channels = decode_channels_codestream(codestream, region, None)?;
    linear_rgb_from_decoded_channels(&channels, &color_encoding, color_channels)
}

fn validate_linear_rgb_color_encoding(color: &ColorEncoding) -> Result<()> {
    match color.color_space {
        ColorSpace::Rgb => {
            if color.want_icc
                || color.white_point != WhitePoint::D65
                || color.custom_white_point.is_some()
                || color.primaries != Primaries::Srgb
                || color.custom_primaries.is_some()
            {
                return Err(Error::Unsupported("linear RGB color management"));
            }
        }
        ColorSpace::Gray => {
            if color.want_icc {
                return Err(Error::Unsupported("linear RGB color management"));
            }
        }
        _ => return Err(Error::Unsupported("linear RGB modular output")),
    }
    if !matches!(
        color.transfer_function,
        TransferFunction::Srgb | TransferFunction::Linear
    ) || color.gamma.is_some()
    {
        return Err(Error::Unsupported("linear RGB transfer function"));
    }
    Ok(())
}

fn linear_rgb_from_decoded_channels(
    channels: &DecodedChannels,
    color_encoding: &ColorEncoding,
    color_channels: usize,
) -> Result<LinearRgbImage> {
    if !matches!(color_channels, 1 | 3) {
        return Err(Error::Unsupported("unsupported color channel count"));
    }
    if channels.color_channels != color_channels || channels.channels.len() < color_channels {
        return Err(Error::Unsupported("missing color channel output"));
    }
    let channel_indices = (0..color_channels).collect::<Vec<_>>();
    validate_interleaved_channel_geometry(channels, &channel_indices, None)?;
    let sample_count = decoded_channel_sample_count(channels)?;
    let mut pixels = Vec::with_capacity(sample_count * 3);
    for index in 0..sample_count {
        if color_channels == 1 {
            let (sample, bit_depth) = channel_sample(&channels.channels[0], index)?;
            let linear = linear_sample_from_encoded_sample(sample, bit_depth, color_encoding)?;
            pixels.extend_from_slice(&[linear, linear, linear]);
        } else {
            for channel in 0..3 {
                let (sample, bit_depth) = channel_sample(&channels.channels[channel], index)?;
                pixels.push(linear_sample_from_encoded_sample(
                    sample,
                    bit_depth,
                    color_encoding,
                )?);
            }
        }
    }
    Ok(LinearRgbImage {
        width: channels.width,
        height: channels.height,
        pixels,
    })
}

fn linear_sample_from_encoded_sample(
    sample: u32,
    bit_depth: u32,
    color_encoding: &ColorEncoding,
) -> Result<f32> {
    let max = max_sample_value(bit_depth)? as f32;
    let encoded = sample as f32 / max;
    Ok(match color_encoding.transfer_function {
        TransferFunction::Linear => encoded,
        TransferFunction::Srgb => srgb_sample_to_linear(encoded),
        _ => return Err(Error::Unsupported("linear RGB transfer function")),
    })
}

fn srgb_sample_to_linear(sample: f32) -> f32 {
    let sample = sample.clamp(0.0, 1.0);
    if sample <= 0.040_45 {
        sample / 12.92
    } else {
        ((sample + 0.055) / 1.055).powf(2.4)
    }
}

fn modular_codestream_to_xyb_image(
    codestream: &jxl_codec::Codestream,
    region: Option<jxl_codec::ImageRegion>,
) -> Result<jxl_codec::VarDctXybImage> {
    let modular = codestream
        .first_frame_modular
        .as_ref()
        .ok_or(Error::Unsupported("modular image metadata"))?;
    let image = modular.image.as_ref().ok_or_else(|| {
        modular
            .image_error
            .clone()
            .unwrap_or(Error::Unsupported("modular pixel reconstruction"))
    })?;
    if codestream.metadata.num_color_channels() != 3 || image.channels.len() < 3 {
        return Err(Error::Unsupported("modular XYB color output"));
    }
    for channel in image.channels.iter().take(3) {
        if channel.width != image.width || channel.height != image.height {
            return Err(Error::Unsupported("modular XYB shifted color output"));
        }
    }

    let sample_count = (image.width as usize)
        .checked_mul(image.height as usize)
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    let dc_quant = modular.global.dc_quant();
    let mut xyb = jxl_codec::VarDctXybImage {
        width: image.width,
        height: image.height,
        groups_assembled: 0,
        groups_missing: 0,
        channels: [
            Vec::with_capacity(sample_count),
            Vec::with_capacity(sample_count),
            Vec::with_capacity(sample_count),
        ],
    };
    for index in 0..sample_count {
        let raw_y = image.channels[0].samples[index] as f32;
        let raw_x = image.channels[1].samples[index] as f32;
        let raw_b_minus_y = image.channels[2].samples[index] as f32;
        xyb.channels[0].push(raw_x * dc_quant[0]);
        xyb.channels[1].push(raw_y * dc_quant[1]);
        xyb.channels[2].push((raw_b_minus_y + raw_y) * dc_quant[2]);
    }
    let frame = codestream
        .first_frame
        .as_ref()
        .ok_or(Error::Unsupported("image has no decoded frame"))?;
    let full_width = frame.frame_size.width.div_ceil(frame.upsampling);
    let full_height = frame.frame_size.height.div_ceil(frame.upsampling);
    let origin = region.map(|region| (region.x, region.y)).unwrap_or((0, 0));
    if let Some(splines) = &modular.global.features.splines {
        jxl_codec::render_splines_into_xyb_image(
            &mut xyb,
            splines,
            full_width,
            full_height,
            origin,
        )?;
    }
    if let Some(noise) = &modular.global.features.noise {
        if frame.upsampling != 1 {
            return Err(Error::Unsupported("noise rendering with frame upsampling"));
        }
        jxl_codec::render_noise_into_xyb_image(
            &mut xyb,
            noise,
            full_width,
            full_height,
            frame.group_layout.group_dim,
            origin,
        )?;
    }
    Ok(xyb)
}

fn append_modular_extra_channels(
    channels: &mut DecodedChannels,
    codestream: &jxl_codec::Codestream,
) -> Result<()> {
    channels.alpha = raw_alpha_info(&codestream.metadata)?;
    if codestream.metadata.extra_channels.is_empty() {
        return Ok(());
    }
    let image = codestream
        .first_frame_modular
        .as_ref()
        .and_then(|modular| modular.image.as_ref())
        .ok_or(Error::Unsupported("modular pixel reconstruction"))?;
    let color_channels = codestream.metadata.num_color_channels() as usize;
    let channel_bit_depths = decoded_channel_bit_depths(&codestream.metadata, color_channels)?;
    for (index, channel) in image.channels.iter().enumerate().skip(color_channels) {
        let channel_bit_depth =
            channel_bit_depths
                .get(index)
                .copied()
                .ok_or(Error::InvalidCodestream(
                    "decoded channel missing bit-depth metadata",
                ))?;
        let max_sample = max_sample_value(channel_bit_depth)?;
        channels.channels.push(decode_channel(
            image.width,
            image.height,
            channel,
            channel_bit_depth,
            max_sample,
        )?);
    }
    Ok(())
}

fn decode_vardct_channels_codestream(
    codestream: &jxl_codec::Codestream,
    region: Option<jxl_codec::ImageRegion>,
    vardct_pass: Option<usize>,
) -> Result<DecodedChannels> {
    let orientation = codestream.metadata.orientation;
    let color_channels = codestream.metadata.num_color_channels() as usize;
    let image = vardct_srgb8_image_from_codestream(codestream, region, vardct_pass)?;
    let mut channels = decoded_channels_from_vardct_srgb8(image, color_channels)?;
    append_vardct_extra_channels(&mut channels, codestream, region, vardct_pass)?;
    orient_decoded_channels(channels, orientation)
}

fn decode_vardct_channels_codestream_rgb16(
    codestream: &jxl_codec::Codestream,
    region: Option<jxl_codec::ImageRegion>,
    vardct_pass: Option<usize>,
) -> Result<DecodedChannels> {
    let orientation = codestream.metadata.orientation;
    let color_channels = codestream.metadata.num_color_channels() as usize;
    let image = vardct_srgb16_image_from_codestream(codestream, region, vardct_pass)?;
    let mut channels = decoded_channels_from_vardct_srgb16(image, color_channels)?;
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
    if channels.color_channels != codestream.metadata.num_color_channels() as usize {
        return Err(Error::Unsupported("VarDCT non-RGB raw channel output"));
    }

    let mut extra_channels = vardct_extra_channels_from_ac(codestream, vardct_pass)?;
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
    vardct_pass: Option<usize>,
) -> Result<Vec<DecodedChannel>> {
    let plan = first_frame_vardct_plan(codestream)?;
    let frame = codestream
        .first_frame
        .as_ref()
        .ok_or(Error::Unsupported("image has no decoded frame"))?;
    let color_channels = codestream.metadata.num_color_channels() as usize;
    // VarDCT modular side streams are indexed after the three internal color planes,
    // even when the public image color space is grayscale.
    let vardct_modular_extra_base = 3usize;
    let channel_bit_depths = decoded_channel_bit_depths(&codestream.metadata, color_channels)?;
    let mut planes = codestream
        .metadata
        .extra_channels
        .iter()
        .enumerate()
        .map(|(extra_index, _)| {
            let bit_depth = *channel_bit_depths.get(color_channels + extra_index).ok_or(
                Error::InvalidCodestream("decoded channel missing bit-depth metadata"),
            )?;
            let upsampling = *frame.extra_channel_upsampling.get(extra_index).ok_or(
                Error::InvalidCodestream("missing extra-channel upsampling factor"),
            )?;
            if upsampling == 0 {
                return Err(Error::InvalidCodestream("zero extra-channel upsampling"));
            }
            let width = frame.frame_size.width.div_ceil(upsampling);
            let height = frame.frame_size.height.div_ceil(upsampling);
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
    if let Some(group) = &plan.modular_global {
        for channel in &group.channels {
            if channel.channel_index < vardct_modular_extra_base {
                continue;
            }
            let extra_index = channel.channel_index - vardct_modular_extra_base;
            let Some(plane) = planes.get_mut(extra_index) else {
                return Err(Error::InvalidCodestream(
                    "decoded VarDCT extra channel index is out of range",
                ));
            };
            copy_vardct_extra_channel_chunk(plane, channel)?;
            saw_extra_channel[extra_index] = true;
        }
    } else if let Some(error) = &plan.modular_global_error {
        return Err(error.clone());
    }

    for group_metadata in plan.ac_group_metadata.iter().filter(|group| {
        vardct_pass
            .map(|pass| group.payload.pass == pass)
            .unwrap_or(true)
    }) {
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
            if channel.channel_index < vardct_modular_extra_base {
                continue;
            }
            let extra_index = channel.channel_index - vardct_modular_extra_base;
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
    let color_transform = first_frame_color_transform(codestream)?;
    let plan = first_frame_vardct_plan(codestream)?;
    let mut image = match color_transform {
        jxl_codec::ColorTransform::Xyb => match pass {
            Some(pass) => jxl_codec::assemble_vardct_srgb8_image_for_pass(plan, pass)?,
            None => jxl_codec::assemble_vardct_srgb8_image(plan)?,
        },
        jxl_codec::ColorTransform::YCbCr => match pass {
            Some(pass) => jxl_codec::assemble_vardct_ycbcr_srgb8_image_for_pass(plan, pass)?,
            None => jxl_codec::assemble_vardct_ycbcr_srgb8_image(plan)?,
        },
        jxl_codec::ColorTransform::None => match pass {
            Some(pass) => jxl_codec::assemble_vardct_rgb_srgb8_image_for_pass(plan, pass)?,
            None => jxl_codec::assemble_vardct_rgb_srgb8_image(plan)?,
        },
    }
    .ok_or(Error::Unsupported("VarDCT image reconstruction"))?;
    if let Some(region) = region {
        image = crop_vardct_srgb8(image, region)?;
    }
    Ok(image)
}

fn vardct_linear_rgb_image_from_codestream(
    codestream: &jxl_codec::Codestream,
    region: Option<jxl_codec::ImageRegion>,
    pass: Option<usize>,
) -> Result<LinearRgbImage> {
    let plan = first_frame_vardct_plan(codestream)?;
    let xyb = match pass {
        Some(pass) => jxl_codec::assemble_vardct_xyb_image_for_pass(plan, pass)?,
        None => jxl_codec::assemble_vardct_xyb_image(plan)?,
    }
    .ok_or(Error::Unsupported("VarDCT image reconstruction"))?;
    let rgb = jxl_codec::xyb_image_to_linear_rgb(&xyb, &plan.opsin_params);
    let mut image = linear_rgb_from_vardct_rgb(rgb)?;
    if let Some(region) = region {
        image = crop_linear_rgb_image(image, region)?;
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
            let needs_full_alpha = vardct_alpha_is_subsampled(&codestream)?;
            let decode_region = if needs_full_alpha {
                None
            } else {
                config.region
            };
            let channels =
                decode_vardct_channels_codestream_rgb16(&codestream, decode_region, vardct_pass)?;
            let image = rgba16_from_decoded_channels_with_transform_data(
                &channels,
                alpha_channel_index,
                Some(&codestream.transform_data),
            )?;
            return if let (true, Some(region)) = (needs_full_alpha, config.region) {
                crop_rgba16_image(image, region)
            } else {
                Ok(image)
            };
        }
        reject_vardct_alpha_output(&codestream.metadata)?;
        if codestream.metadata.num_color_channels() == 1 {
            let channels =
                decode_vardct_channels_codestream_rgb16(&codestream, config.region, vardct_pass)?;
            return rgba16_from_decoded_channels_with_transform_data(
                &channels,
                alpha_channel_index,
                Some(&codestream.transform_data),
            );
        }
        let orientation = codestream.metadata.orientation;
        let image = rgba16_from_vardct_srgb16(vardct_srgb16_image_from_codestream(
            &codestream,
            config.region,
            vardct_pass,
        )?)?;
        return orient_rgba16(image, orientation);
    }
    reject_vardct_pass_for_non_vardct(vardct_pass)?;

    if let Some(region) = config.region
        && raw_alpha_channel_index(&codestream.metadata)?.is_some()
    {
        let (_, full_codestream) = jxl_codec::parse_file(input)?;
        let image = rgba16_from_modular_codestream(full_codestream, None)?;
        return crop_rgba16_image(image, region);
    }

    rgba16_from_modular_codestream(codestream, config.region)
}

fn rgba16_from_modular_codestream(
    codestream: jxl_codec::Codestream,
    region: Option<jxl_codec::ImageRegion>,
) -> Result<Rgba16Image> {
    let orientation = codestream.metadata.orientation;
    let alpha_channel_index = raw_alpha_channel_index(&codestream.metadata)?;
    let transform_data = codestream.transform_data.clone();
    let color_transform = first_frame_color_transform(&codestream)?;
    let channels = if color_transform == jxl_codec::ColorTransform::Xyb {
        modular_xyb_decoded_channels_srgb16_from_codestream(&codestream, region)?
    } else {
        decode_channels_codestream(codestream, None, None)?
    };
    orient_rgba16(
        rgba16_from_decoded_channels_with_transform_data(
            &channels,
            alpha_channel_index,
            (color_transform != jxl_codec::ColorTransform::Xyb).then_some(&transform_data),
        )?,
        orientation,
    )
}

#[cfg(test)]
fn rgba16_from_decoded_channels(
    channels: &DecodedChannels,
    alpha_channel_index: Option<usize>,
) -> Result<Rgba16Image> {
    rgba16_from_decoded_channels_with_transform_data(channels, alpha_channel_index, None)
}

fn rgba16_from_decoded_channels_with_transform_data(
    channels: &DecodedChannels,
    alpha_channel_index: Option<usize>,
    transform_data: Option<&CustomTransformData>,
) -> Result<Rgba16Image> {
    let output_channel_indices = rgba_channel_indices(channels, alpha_channel_index)?;
    let pixels = rgba16_from_channel_indices(channels, &output_channel_indices, transform_data)?;
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
    let color_transform = first_frame_color_transform(codestream)?;
    let plan = first_frame_vardct_plan(codestream)?;
    let mut image = match color_transform {
        jxl_codec::ColorTransform::Xyb => match pass {
            Some(pass) => jxl_codec::assemble_vardct_srgb16_image_for_pass(plan, pass)?,
            None => jxl_codec::assemble_vardct_srgb16_image(plan)?,
        },
        jxl_codec::ColorTransform::YCbCr => match pass {
            Some(pass) => jxl_codec::assemble_vardct_ycbcr_srgb16_image_for_pass(plan, pass)?,
            None => jxl_codec::assemble_vardct_ycbcr_srgb16_image(plan)?,
        },
        jxl_codec::ColorTransform::None => match pass {
            Some(pass) => jxl_codec::assemble_vardct_rgb_srgb16_image_for_pass(plan, pass)?,
            None => jxl_codec::assemble_vardct_rgb_srgb16_image(plan)?,
        },
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

fn vardct_alpha_is_subsampled(codestream: &jxl_codec::Codestream) -> Result<bool> {
    let Some(alpha_channel_index) = raw_alpha_channel_index(&codestream.metadata)? else {
        return Ok(false);
    };
    let color_channels = codestream.metadata.num_color_channels() as usize;
    if alpha_channel_index < color_channels {
        return Err(Error::InvalidCodestream(
            "decoded alpha channel index is in color channels",
        ));
    }
    let extra_index = alpha_channel_index - color_channels;
    let frame = codestream
        .first_frame
        .as_ref()
        .ok_or(Error::Unsupported("image has no decoded frame"))?;
    let upsampling =
        *frame
            .extra_channel_upsampling
            .get(extra_index)
            .ok_or(Error::InvalidCodestream(
                "missing extra-channel upsampling factor",
            ))?;
    Ok(upsampling > 1)
}

fn crop_decoded_image(image: DecodedImage, region: jxl_codec::ImageRegion) -> Result<DecodedImage> {
    validate_decode_region(image.width, image.height, region)?;
    let output_channels = decoded_image_output_channels(&image);
    let pixels = match image.pixels {
        PixelData::U8(samples) => PixelData::U8(crop_interleaved_u8(
            &samples,
            image.width,
            output_channels,
            region,
        )?),
        PixelData::U16(samples) => PixelData::U16(crop_interleaved_u16(
            &samples,
            image.width,
            output_channels,
            region,
        )?),
    };
    Ok(DecodedImage {
        width: region.width,
        height: region.height,
        color_channels: image.color_channels,
        alpha: image.alpha,
        bit_depth: image.bit_depth,
        pixels,
    })
}

fn crop_rgba8_image(image: RgbaImage, region: jxl_codec::ImageRegion) -> Result<RgbaImage> {
    validate_decode_region(image.width, image.height, region)?;
    Ok(RgbaImage {
        width: region.width,
        height: region.height,
        pixels: crop_interleaved_u8(&image.pixels, image.width, 4, region)?,
    })
}

fn crop_rgba16_image(image: Rgba16Image, region: jxl_codec::ImageRegion) -> Result<Rgba16Image> {
    validate_decode_region(image.width, image.height, region)?;
    Ok(Rgba16Image {
        width: region.width,
        height: region.height,
        pixels: crop_interleaved_u16(&image.pixels, image.width, 4, region)?,
    })
}

fn crop_linear_rgb_image(
    image: LinearRgbImage,
    region: jxl_codec::ImageRegion,
) -> Result<LinearRgbImage> {
    validate_decode_region(image.width, image.height, region)?;
    Ok(LinearRgbImage {
        width: region.width,
        height: region.height,
        pixels: crop_interleaved_f32(&image.pixels, image.width, 3, region)?,
    })
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

fn crop_interleaved_f32(
    samples: &[f32],
    width: u32,
    channels: usize,
    region: jxl_codec::ImageRegion,
) -> Result<Vec<f32>> {
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
    if channel.hshift < 0 || channel.vshift < 0 {
        return Err(Error::Unsupported(
            "subsampled VarDCT extra-channel ROI output",
        ));
    }
    let channel_region = shifted_decode_region(region, channel.hshift, channel.vshift)?;
    validate_decode_region(channel.width, channel.height, channel_region)?;
    match channel.samples {
        ChannelData::U8(samples) => Ok(DecodedChannel {
            width: channel_region.width,
            height: channel_region.height,
            hshift: channel.hshift,
            vshift: channel.vshift,
            bit_depth: channel.bit_depth,
            samples: ChannelData::U8(crop_interleaved_u8(
                &samples,
                channel.width,
                1,
                channel_region,
            )?),
        }),
        ChannelData::U16(samples) => Ok(DecodedChannel {
            width: channel_region.width,
            height: channel_region.height,
            hshift: channel.hshift,
            vshift: channel.vshift,
            bit_depth: channel.bit_depth,
            samples: ChannelData::U16(crop_interleaved_u16(
                &samples,
                channel.width,
                1,
                channel_region,
            )?),
        }),
    }
}

fn shifted_decode_region(
    region: jxl_codec::ImageRegion,
    hshift: i32,
    vshift: i32,
) -> Result<jxl_codec::ImageRegion> {
    if hshift < 0 || vshift < 0 {
        return Err(Error::Unsupported("upshifted raw channel ROI output"));
    }
    Ok(jxl_codec::ImageRegion {
        x: shifted_region_start(region.x, hshift as u32)?,
        y: shifted_region_start(region.y, vshift as u32)?,
        width: shifted_region_len(region.x, region.width, hshift as u32)?,
        height: shifted_region_len(region.y, region.height, vshift as u32)?,
    })
}

fn shifted_region_start(start: u32, shift: u32) -> Result<u32> {
    if shift >= u32::BITS {
        return Err(Error::InvalidCodestream("decode region is outside image"));
    }
    Ok(start >> shift)
}

fn shifted_region_len(start: u32, len: u32, shift: u32) -> Result<u32> {
    if shift >= u32::BITS {
        return Err(Error::InvalidCodestream("decode region is outside image"));
    }
    let end = start
        .checked_add(len)
        .ok_or(Error::InvalidCodestream("decode region is outside image"))?;
    Ok(end.div_ceil(1u32 << shift) - (start >> shift))
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

fn orient_linear_rgb(image: LinearRgbImage, orientation: Orientation) -> Result<LinearRgbImage> {
    let (width, height, pixels) =
        orient_interleaved(image.pixels, image.width, image.height, 3, orientation)?;
    Ok(LinearRgbImage {
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
        validate_upsampled_channel_geometry(channels, alpha_channel)?;
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

fn validate_interleaved_channel_geometry(
    channels: &DecodedChannels,
    output_channel_indices: &[usize],
    alpha_channel_index: Option<usize>,
) -> Result<()> {
    for &index in output_channel_indices {
        let channel = channels
            .channels
            .get(index)
            .ok_or(Error::Unsupported("missing color channel output"))?;
        if Some(index) == alpha_channel_index {
            validate_upsampled_channel_geometry(channels, channel)?;
        } else if channel.width != channels.width || channel.height != channels.height {
            return Err(Error::Unsupported("subsampled raw channel output"));
        }
    }
    Ok(())
}

fn interleaved_output_bit_depth(
    channels: &DecodedChannels,
    output_channel_indices: &[usize],
) -> Result<u32> {
    let mut output_bit_depth = 0;
    for &index in output_channel_indices {
        let bit_depth = channels
            .channels
            .get(index)
            .ok_or(Error::Unsupported("missing color channel output"))?
            .bit_depth;
        if bit_depth > 16 {
            return Err(Error::Unsupported("integer sample depths above 16 bits"));
        }
        output_bit_depth = output_bit_depth.max(bit_depth);
    }
    if output_bit_depth == 0 {
        return Err(Error::Unsupported("missing color channel output"));
    }
    Ok(output_bit_depth)
}

fn validate_upsampled_channel_geometry(
    channels: &DecodedChannels,
    channel: &DecodedChannel,
) -> Result<()> {
    if channel.hshift < 0 || channel.vshift < 0 {
        return Err(Error::Unsupported("upshifted alpha image decode"));
    }
    let expected_width = shifted_len(channels.width, channel.hshift as u32)?;
    let expected_height = shifted_len(channels.height, channel.vshift as u32)?;
    if channel.width != expected_width || channel.height != expected_height {
        return Err(Error::Unsupported("non power-of-two channel geometry"));
    }
    if (channel.hshift == 0) != (channel.vshift == 0) {
        return Err(Error::Unsupported("asymmetric alpha image decode"));
    }
    if channel.hshift > 3 || channel.vshift > 3 {
        return Err(Error::Unsupported("subsampled alpha image decode"));
    }
    Ok(())
}

fn shifted_len(len: u32, shift: u32) -> Result<u32> {
    if shift >= u32::BITS {
        return Err(Error::InvalidCodestream("decoded image size overflow"));
    }
    Ok(len.div_ceil(1u32 << shift))
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

fn interleave_channel_u8(
    image: &DecodedChannels,
    channel_indices: &[usize],
    transform_data: Option<&CustomTransformData>,
    output_bit_depth: u32,
) -> Result<Vec<u8>> {
    let output_max = max_sample_value(output_bit_depth)?;
    let output_channels = channel_indices.len();
    let sample_count = decoded_channel_sample_count(image)?;
    let pixels = sample_count
        .checked_mul(output_channels)
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    let mut output = Vec::with_capacity(pixels);
    for y in 0..image.height {
        for x in 0..image.width {
            let index = (y as usize)
                .checked_mul(image.width as usize)
                .and_then(|index| index.checked_add(x as usize))
                .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
            for &channel_index in channel_indices {
                let channel = &image.channels[channel_index];
                let (sample, bit_depth) =
                    channel_sample_at(channel, image.width, x, y, index, transform_data)?;
                if bit_depth > output_bit_depth || output_bit_depth > 8 {
                    return Err(Error::InvalidCodestream(
                        "decoded channel bit-depth mismatch",
                    ));
                }
                output.push(scale_sample_to(sample, bit_depth, output_max) as u8);
            }
        }
    }
    Ok(output)
}

fn interleave_channel_u16(
    image: &DecodedChannels,
    channel_indices: &[usize],
    transform_data: Option<&CustomTransformData>,
    output_bit_depth: u32,
) -> Result<Vec<u16>> {
    let output_max = max_sample_value(output_bit_depth)?;
    let output_channels = channel_indices.len();
    let sample_count = decoded_channel_sample_count(image)?;
    let pixels = sample_count
        .checked_mul(output_channels)
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    let mut output = Vec::with_capacity(pixels);
    for y in 0..image.height {
        for x in 0..image.width {
            let index = (y as usize)
                .checked_mul(image.width as usize)
                .and_then(|index| index.checked_add(x as usize))
                .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
            for &channel_index in channel_indices {
                let channel = &image.channels[channel_index];
                let (sample, bit_depth) =
                    channel_sample_at(channel, image.width, x, y, index, transform_data)?;
                if bit_depth > output_bit_depth || output_bit_depth > 16 {
                    return Err(Error::InvalidCodestream(
                        "decoded channel bit-depth mismatch",
                    ));
                }
                output.push(scale_sample_to(sample, bit_depth, output_max) as u16);
            }
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
    validate_interleaved_channel_geometry(channels, &output_channel_indices, alpha_channel_index)?;
    if alpha != channels.alpha {
        return Err(Error::InvalidCodestream("decoded alpha metadata mismatch"));
    }
    Ok(output_channel_indices)
}

fn rgba8_from_channel_indices(
    channels: &DecodedChannels,
    channel_indices: &[usize],
    transform_data: Option<&CustomTransformData>,
) -> Result<Vec<u8>> {
    let sample_count = decoded_channel_sample_count(channels)?;
    let mut rgba = Vec::with_capacity(sample_count * 4);
    for y in 0..channels.height {
        for x in 0..channels.width {
            let index = (y as usize)
                .checked_mul(channels.width as usize)
                .and_then(|index| index.checked_add(x as usize))
                .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
            append_rgba8_pixel(
                &mut rgba,
                channels.color_channels,
                channels.alpha,
                |channel| {
                    channel_sample_at(
                        &channels.channels[channel_indices[channel]],
                        channels.width,
                        x,
                        y,
                        index,
                        transform_data,
                    )
                },
            )?;
        }
    }
    Ok(rgba)
}

fn rgba16_from_channel_indices(
    channels: &DecodedChannels,
    channel_indices: &[usize],
    transform_data: Option<&CustomTransformData>,
) -> Result<Vec<u16>> {
    let sample_count = decoded_channel_sample_count(channels)?;
    let mut rgba = Vec::with_capacity(sample_count * 4);
    for y in 0..channels.height {
        for x in 0..channels.width {
            let index = (y as usize)
                .checked_mul(channels.width as usize)
                .and_then(|index| index.checked_add(x as usize))
                .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
            append_rgba16_pixel(
                &mut rgba,
                channels.color_channels,
                channels.alpha,
                |channel| {
                    channel_sample_at(
                        &channels.channels[channel_indices[channel]],
                        channels.width,
                        x,
                        y,
                        index,
                        transform_data,
                    )
                },
            )?;
        }
    }
    Ok(rgba)
}

fn channel_sample_at(
    channel: &DecodedChannel,
    image_width: u32,
    x: u32,
    y: u32,
    unshifted_index: usize,
    transform_data: Option<&CustomTransformData>,
) -> Result<(u32, u32)> {
    if channel.hshift == 0 && channel.vshift == 0 {
        return channel_sample(channel, unshifted_index);
    }
    if channel.hshift == channel.vshift && matches!(channel.hshift, 1 | 2 | 3) {
        return upsample_channel_sample(
            channel,
            image_width,
            x,
            y,
            channel.hshift as u32,
            transform_data,
        );
    }
    Err(Error::Unsupported("subsampled alpha image decode"))
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

const DEFAULT_UPSAMPLING2_WEIGHTS: [f32; 15] = [
    -0.017166138,
    -0.03451538,
    -0.040222168,
    -0.029205322,
    -0.0062446594,
    0.14111328,
    0.2890625,
    0.0027866364,
    -0.016098022,
    0.56640625,
    0.03778076,
    -0.019866943,
    -0.031433105,
    -0.01184845,
    -0.0021362305,
];

const DEFAULT_UPSAMPLING4_WEIGHTS: [f32; 55] = [
    -0.024185181,
    -0.034912109,
    -0.03692627,
    -0.030944824,
    -0.0052986145,
    -0.01663208,
    -0.035583496,
    -0.038879395,
    -0.03515625,
    -0.0098953247,
    0.23657227,
    0.33398438,
    -0.010734558,
    -0.013130188,
    -0.035552979,
    0.13049316,
    0.40112305,
    0.039520264,
    -0.020782471,
    0.46923828,
    -0.0020923615,
    -0.014846802,
    -0.040649414,
    0.18945312,
    0.56298828,
    0.066772461,
    -0.023361206,
    -0.035522461,
    -0.0075492859,
    -0.022674561,
    -0.023635864,
    0.0031585693,
    -0.033996582,
    -0.013595581,
    -0.00091648102,
    -0.0033550262,
    -0.011634827,
    -0.016098022,
    -0.0097427368,
    -0.0019159317,
    -0.010955811,
    -0.031982422,
    -0.044555664,
    -0.027999878,
    -0.0064582825,
    0.063903809,
    0.22961426,
    0.0063095093,
    -0.018966675,
    0.67529297,
    0.084838867,
    -0.025344849,
    -0.02204895,
    -0.016677856,
    -0.0038452148,
];

const DEFAULT_UPSAMPLING8_WEIGHTS: [f32; 210] = [
    -0.029281616,
    -0.03704834,
    -0.037841797,
    -0.033233643,
    -0.0044746399,
    -0.025192261,
    -0.037536621,
    -0.039001465,
    -0.036621094,
    -0.0064659119,
    -0.0206604,
    -0.038391113,
    -0.040008545,
    -0.039001465,
    -0.0090179443,
    -0.016265869,
    -0.039550781,
    -0.040466309,
    -0.039794922,
    -0.012245178,
    0.29907227,
    0.35766602,
    -0.024475098,
    -0.010818481,
    -0.043151855,
    0.23901367,
    0.41113281,
    -0.0057296753,
    -0.014503479,
    -0.042480469,
    0.17565918,
    0.45214844,
    0.022872925,
    -0.019363403,
    -0.035827637,
    0.11572266,
    0.47412109,
    0.062866211,
    -0.026855469,
    0.42724609,
    -0.022491455,
    -0.011550903,
    -0.045623779,
    0.28686523,
    0.4909668,
    -7.891655e-05,
    -0.015457153,
    -0.045623779,
    0.21240234,
    0.54003906,
    0.033691406,
    -0.020706177,
    -0.038665771,
    0.14233398,
    0.56591797,
    0.080444336,
    -0.028884888,
    -0.036804199,
    -0.0054206848,
    -0.029205322,
    -0.027893066,
    -0.021179199,
    -0.039428711,
    -0.0077552795,
    -0.024337769,
    -0.031951904,
    -0.020309448,
    -0.040435791,
    -0.010742188,
    -0.019302368,
    -0.036193848,
    -0.019744873,
    -0.03918457,
    -0.014564514,
    -0.00045061111,
    -0.0036010742,
    -0.0102005,
    -0.012321472,
    -0.0063896179,
    -0.00071573257,
    -0.002790451,
    -0.0095748901,
    -0.012886047,
    -0.00730896,
    -0.001077652,
    -0.0021018982,
    -0.0089035034,
    -0.013175964,
    -0.008140564,
    -0.001534462,
    -0.021286011,
    -0.041717529,
    -0.048309326,
    -0.032928467,
    -0.0052528381,
    -0.017196655,
    -0.040527344,
    -0.050445557,
    -0.036071777,
    -0.0073814392,
    -0.013420105,
    -0.039642334,
    -0.051513672,
    -0.038146973,
    -0.010055542,
    0.18969727,
    0.33056641,
    -0.013000488,
    -0.01373291,
    -0.040161133,
    0.1373291,
    0.36401367,
    0.010276794,
    -0.018325806,
    -0.033660889,
    0.087341309,
    0.38183594,
    0.043395996,
    -0.025253296,
    0.56396484,
    0.0045852661,
    -0.016479492,
    -0.04888916,
    0.24584961,
    0.62011719,
    0.043151855,
    -0.022140503,
    -0.041564941,
    0.16638184,
    0.65039062,
    0.096191406,
    -0.031021118,
    -0.04083252,
    -0.0090484619,
    -0.027908325,
    -0.021179199,
    0.0079879761,
    -0.03994751,
    -0.012435913,
    -0.022323608,
    -0.029464722,
    0.0099182129,
    -0.036010742,
    -0.016845703,
    -0.0011167526,
    -0.0041122437,
    -0.012969971,
    -0.017242432,
    -0.010223389,
    -0.0016527176,
    -0.0031318665,
    -0.012176514,
    -0.01763916,
    -0.011253357,
    -0.0023174286,
    -0.01374054,
    -0.037963867,
    -0.051422119,
    -0.031173706,
    -0.0058174133,
    -0.010643005,
    -0.036071777,
    -0.052734375,
    -0.033752441,
    -0.0079574585,
    0.096252441,
    0.27124023,
    -0.0035381317,
    -0.017333984,
    -0.031524658,
    0.056854248,
    0.28491211,
    0.02230835,
    -0.023742676,
    0.68212891,
    0.050170898,
    -0.023208618,
    -0.043823242,
    0.18457031,
    0.71533203,
    0.10803223,
    -0.032623291,
    -0.036376953,
    -0.013946533,
    -0.025115967,
    -0.017288208,
    0.054077148,
    -0.028671265,
    -0.018936157,
    -0.0024089813,
    -0.0044670105,
    -0.016357422,
    -0.023773193,
    -0.015228271,
    -0.0033340454,
    -0.0082015991,
    -0.029647827,
    -0.04498291,
    -0.027450562,
    -0.0061225891,
    0.027267456,
    0.19445801,
    0.0015983582,
    -0.022323608,
    0.75,
    0.11450195,
    -0.033477783,
    -0.016052246,
    -0.020706177,
    -0.0045814514,
];

fn upsample_channel_sample(
    channel: &DecodedChannel,
    image_width: u32,
    x: u32,
    y: u32,
    shift: u32,
    transform_data: Option<&CustomTransformData>,
) -> Result<(u32, u32)> {
    let factor = 1u32 << shift;
    let source_x = (x >> shift) as isize;
    let source_y = (y >> shift) as isize;
    let ox = (x & (factor - 1)) as usize;
    let oy = (y & (factor - 1)) as usize;
    let mut min_sample = f32::INFINITY;
    let mut max_sample = f32::NEG_INFINITY;
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;

    for i in 0..25 {
        let px = i % 5;
        let py = i / 5;
        let sample = channel_sample_f32(
            channel,
            source_x + px as isize - 2,
            source_y + py as isize - 2,
        )?;
        min_sample = min_sample.min(sample);
        max_sample = max_sample.max(sample);
        let weight = upsampling_kernel(shift, ox, oy, px, py, transform_data)?;
        match i % 3 {
            0 => acc0 = sample.mul_add(weight, acc0),
            1 => acc1 = sample.mul_add(weight, acc1),
            _ => acc2 = sample.mul_add(weight, acc2),
        }
    }

    let output = (acc1 + acc2) + acc0;
    let output = output.clamp(min_sample, max_sample).round();
    let max = max_sample_value(channel.bit_depth)? as f32;
    let sample = output.clamp(0.0, max) as u32;
    let expected_width = image_width.div_ceil(factor);
    if expected_width != channel.width {
        return Err(Error::Unsupported("non power-of-two channel geometry"));
    }
    Ok((sample, channel.bit_depth))
}

fn upsampling_kernel(
    shift: u32,
    ox: usize,
    oy: usize,
    px: usize,
    py: usize,
    transform_data: Option<&CustomTransformData>,
) -> Result<f32> {
    let factor = 1usize << shift;
    let half = factor / 2;
    let kernel_x = if ox < half { ox } else { factor - 1 - ox };
    let kernel_y = if oy < half { oy } else { factor - 1 - oy };
    let px = if ox < half { px } else { 4 - px };
    let py = if oy < half { py } else { 4 - py };
    let i = 5 * kernel_x + px;
    let j = 5 * kernel_y + py;
    let min = i.min(j);
    let max = i.max(j);
    let index = 5 * half * min - min * min.saturating_sub(1) / 2 + max - min;
    match upsampling_weights(shift, transform_data)? {
        Some(weights) => weights
            .get(index)
            .copied()
            .ok_or(Error::InvalidCodestream("invalid upsampling kernel index")),
        _ => Err(Error::Unsupported("subsampled alpha image decode")),
    }
}

fn upsampling_weights<'a>(
    shift: u32,
    transform_data: Option<&'a CustomTransformData>,
) -> Result<Option<&'a [f32]>> {
    let weights = match shift {
        1 => transform_data
            .and_then(|transform_data| transform_data.upsampling2_weights.as_deref())
            .unwrap_or(&DEFAULT_UPSAMPLING2_WEIGHTS),
        2 => transform_data
            .and_then(|transform_data| transform_data.upsampling4_weights.as_deref())
            .unwrap_or(&DEFAULT_UPSAMPLING4_WEIGHTS),
        3 => transform_data
            .and_then(|transform_data| transform_data.upsampling8_weights.as_deref())
            .unwrap_or(&DEFAULT_UPSAMPLING8_WEIGHTS),
        _ => return Ok(None),
    };
    let expected_len = match shift {
        1 => 15,
        2 => 55,
        3 => 210,
        _ => unreachable!(),
    };
    if weights.len() != expected_len {
        return Err(Error::InvalidCodestream(
            "invalid custom upsampling weight count",
        ));
    }
    Ok(Some(weights))
}

fn channel_sample_f32(channel: &DecodedChannel, x: isize, y: isize) -> Result<f32> {
    let x = mirror_coordinate(x, channel.width as usize);
    let y = mirror_coordinate(y, channel.height as usize);
    let index = y
        .checked_mul(channel.width as usize)
        .and_then(|index| index.checked_add(x))
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    let (sample, _) = channel_sample(channel, index)?;
    Ok(sample as f32)
}

fn mirror_coordinate(mut coordinate: isize, size: usize) -> usize {
    let size = size as isize;
    while coordinate < 0 || coordinate >= size {
        if coordinate < 0 {
            coordinate = -coordinate - 1;
        } else {
            coordinate = 2 * size - 1 - coordinate;
        }
    }
    coordinate as usize
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

fn enforce_decoded_channels_memory_limit(
    channels: &DecodedChannels,
    limit: Option<usize>,
) -> Result<()> {
    let mut bytes = 0usize;
    for channel in &channels.channels {
        bytes = bytes
            .checked_add(channel_data_bytes(&channel.samples)?)
            .ok_or(Error::InvalidCodestream("decoded image size overflow"))?;
    }
    enforce_memory_limit(bytes, limit)
}

fn enforce_decoded_image_memory_limit(image: &DecodedImage, limit: Option<usize>) -> Result<()> {
    let bytes = match &image.pixels {
        PixelData::U8(samples) => samples.len(),
        PixelData::U16(samples) => checked_sample_bytes(samples.len(), 2)?,
    };
    enforce_memory_limit(bytes, limit)
}

fn channel_data_bytes(samples: &ChannelData) -> Result<usize> {
    match samples {
        ChannelData::U8(samples) => Ok(samples.len()),
        ChannelData::U16(samples) => checked_sample_bytes(samples.len(), 2),
    }
}

fn checked_sample_bytes(samples: usize, bytes_per_sample: usize) -> Result<usize> {
    samples
        .checked_mul(bytes_per_sample)
        .ok_or(Error::InvalidCodestream("decoded image size overflow"))
}

fn enforce_memory_limit(required: usize, limit: Option<usize>) -> Result<()> {
    if let Some(limit) = limit
        && required > limit
    {
        return Err(Error::Unsupported("memory limit exceeded"));
    }
    Ok(())
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
    fn decoder_memory_limit_can_be_cleared() {
        let decoder = Decoder::new().memory_limit(1).without_memory_limit();

        assert_eq!(decoder.options().memory_limit, None);
    }

    #[test]
    fn decoder_region_and_pass_can_be_cleared() {
        let decoder = Decoder::new()
            .roi(Rect {
                x: 1,
                y: 2,
                width: 3,
                height: 4,
            })
            .vardct_pass(0)
            .without_roi()
            .final_vardct_pass();

        assert_eq!(decoder.options().roi, None);
        assert_eq!(decoder.options().vardct_pass, None);
    }

    #[test]
    fn threading_modes_map_to_modular_group_execution() {
        assert_eq!(
            modular_group_execution_for_threading(ThreadingMode::Single),
            jxl_codec::ModularGroupExecution::Serial
        );
        assert_eq!(
            modular_group_execution_for_threading(ThreadingMode::Threads(3)),
            jxl_codec::ModularGroupExecution::RequestedThreads(3)
        );
        let expected_auto_threads = std::thread::available_parallelism()
            .map(|threads| threads.get())
            .unwrap_or(1);
        let expected_auto = if expected_auto_threads > 1 {
            jxl_codec::ModularGroupExecution::RequestedThreads(expected_auto_threads)
        } else {
            jxl_codec::ModularGroupExecution::Serial
        };
        assert_eq!(
            modular_group_execution_for_threading(ThreadingMode::Auto),
            expected_auto
        );
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
    fn decoder_output_option_selects_decode_method() {
        let bytes = std::fs::read(workspace_path(
            "crates/jxl-codec/tests/generated/icc_rec2020_lossless.jxl",
        ))
        .unwrap();

        assert_eq!(
            Decoder::new().decode_output(&bytes),
            decode_channels(&bytes).map(DecodedOutput::Channels)
        );
        assert_eq!(
            Decoder::new()
                .output(DecodeOutput::Interleaved)
                .decode_output(&bytes),
            decode(&bytes).map(DecodedOutput::Interleaved)
        );
        assert_eq!(
            Decoder::new()
                .output(DecodeOutput::Rgba8)
                .decode_output(&bytes),
            decode_rgba8(&bytes).map(DecodedOutput::Rgba8)
        );
        assert_eq!(
            Decoder::new()
                .output(DecodeOutput::Rgba16)
                .decode_output(&bytes),
            decode_rgba16(&bytes).map(DecodedOutput::Rgba16)
        );
        assert_eq!(
            Decoder::new()
                .output(DecodeOutput::LinearRgb)
                .decode_output(&bytes),
            Err(Error::Unsupported("linear RGB color management"))
        );
    }

    #[test]
    fn decode_with_options_uses_configured_output() {
        let bytes = std::fs::read(workspace_path(
            "crates/jxl-codec/tests/generated/icc_rec2020_lossless.jxl",
        ))
        .unwrap();
        let options = DecodeOptions {
            output: DecodeOutput::Rgba8,
            ..DecodeOptions::default()
        };

        assert_eq!(
            decode_with_options(&bytes, options),
            decode_rgba8(&bytes).map(DecodedOutput::Rgba8)
        );
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

        let memory_decoder = Decoder::new().memory_limit(usize::MAX);
        assert_eq!(memory_decoder.decode(&bytes), decode(&bytes));
        assert_eq!(
            memory_decoder.decode_channels(&bytes),
            decode_channels(&bytes)
        );
        assert_eq!(memory_decoder.decode_rgba8(&bytes), decode_rgba8(&bytes));
        assert_eq!(memory_decoder.decode_rgba16(&bytes), decode_rgba16(&bytes));

        let tight_memory_decoder = Decoder::new().memory_limit(1);
        assert_eq!(
            tight_memory_decoder.decode(&bytes),
            Err(Error::Unsupported("memory limit exceeded"))
        );
        assert_eq!(
            tight_memory_decoder.decode_channels(&bytes),
            Err(Error::Unsupported("memory limit exceeded"))
        );
        assert_eq!(
            tight_memory_decoder.decode_rgba8(&bytes),
            Err(Error::Unsupported("memory limit exceeded"))
        );
        assert_eq!(
            tight_memory_decoder.decode_rgba16(&bytes),
            Err(Error::Unsupported("memory limit exceeded"))
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
    fn decoded_channel_orientation_preserves_shifted_extra_channel_geometry() {
        let channels = DecodedChannels {
            width: 5,
            height: 3,
            color_channels: 1,
            alpha: None,
            bit_depth: 8,
            channels: vec![
                decoded_u8_channel(5, 3, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
                DecodedChannel {
                    width: 3,
                    height: 2,
                    hshift: 1,
                    vshift: 1,
                    bit_depth: 8,
                    samples: ChannelData::U8(vec![10, 11, 12, 13, 14, 15]),
                },
            ],
        };

        let oriented = orient_decoded_channels(channels, Orientation::Rotate90Cw).unwrap();
        assert_eq!(oriented.width, 3);
        assert_eq!(oriented.height, 5);
        assert_eq!(oriented.color_channels, 1);
        assert_eq!(oriented.alpha, None);
        assert_eq!(oriented.bit_depth, 8);
        assert_eq!(oriented.channels.len(), 2);

        let color = &oriented.channels[0];
        assert_eq!(color.width, 3);
        assert_eq!(color.height, 5);
        assert_eq!(color.hshift, 0);
        assert_eq!(color.vshift, 0);
        let ChannelData::U8(color_samples) = &color.samples else {
            panic!("expected oriented color channel to stay 8-bit");
        };
        assert_eq!(
            color_samples,
            &[10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4]
        );

        let extra = &oriented.channels[1];
        assert_eq!(extra.width, 2);
        assert_eq!(extra.height, 3);
        assert_eq!(extra.hshift, 1);
        assert_eq!(extra.vshift, 1);
        assert_eq!(extra.bit_depth, 8);
        let ChannelData::U8(extra_samples) = &extra.samples else {
            panic!("expected oriented extra channel to stay 8-bit");
        };
        assert_eq!(extra_samples, &[13, 10, 14, 11, 15, 12]);
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
    fn inspect_exposes_animation_frame_sequence() {
        let bytes = std::fs::read(workspace_path(
            "reference/libjxl/testdata/jxl/blending/cropped_traffic_light.jxl",
        ))
        .unwrap();
        let info = inspect(&bytes).unwrap();

        assert!(info.basic_info.have_animation);
        assert_eq!(info.frames.len(), 4);
        assert_eq!(info.frame_data.len(), 4);
        assert_eq!(info.modular_frames.len(), 4);
        assert_eq!(info.vardct_plans.len(), 4);
        assert_eq!(info.vardct_frames.len(), 4);
        assert!(info.modular_frames.iter().all(Option::is_some));
        assert!(info.vardct_plans.iter().all(Option::is_none));
        assert!(info.vardct_frames.iter().all(Option::is_none));
        assert_eq!(info.first_frame.as_ref(), info.frames.first());
        assert_eq!(info.first_frame_data.as_ref(), info.frame_data.first());
        assert_eq!(
            info.first_frame_modular.as_ref(),
            info.modular_frames.first().and_then(Option::as_ref)
        );
        assert_eq!(
            info.frames
                .iter()
                .map(|frame| frame.animation_frame.duration)
                .collect::<Vec<_>>(),
            vec![300, 100, 300, 100]
        );
        assert_eq!(
            info.frames
                .iter()
                .map(|frame| frame.is_last)
                .collect::<Vec<_>>(),
            vec![false, false, false, true]
        );
    }

    #[test]
    fn decode_rgba8_frames_exposes_raw_animation_rectangles() {
        let bytes = std::fs::read(workspace_path(
            "reference/libjxl/testdata/jxl/blending/cropped_traffic_light.jxl",
        ))
        .unwrap();
        let info = inspect(&bytes).unwrap();
        let frames = decode_rgba8_frames(&bytes).unwrap();

        assert_eq!(frames.len(), 4);
        for (index, (decoded, header)) in frames.iter().zip(&info.frames).enumerate() {
            assert_eq!(decoded.x, header.frame_origin.x0, "frame {index} x");
            assert_eq!(decoded.y, header.frame_origin.y0, "frame {index} y");
            assert_eq!(
                decoded.duration, header.animation_frame.duration,
                "frame {index} duration"
            );
            assert_eq!(
                decoded.timecode, header.animation_frame.timecode,
                "frame {index} timecode"
            );
            assert_eq!(
                decoded.image.width, header.frame_size.width,
                "frame {index} width"
            );
            assert_eq!(
                decoded.image.height, header.frame_size.height,
                "frame {index} height"
            );
            assert_eq!(
                decoded.image.pixels.len(),
                header.frame_size.width as usize * header.frame_size.height as usize * 4,
                "frame {index} pixels"
            );
        }
        assert!(
            frames
                .iter()
                .flat_map(|frame| frame.image.pixels.chunks_exact(4))
                .any(|pixel| pixel[3] != 0)
        );
        assert_eq!(
            Decoder::new().memory_limit(1).decode_rgba8_frames(&bytes),
            Err(Error::Unsupported("memory limit exceeded"))
        );
        assert_eq!(
            Decoder::new()
                .roi(Rect {
                    x: 0,
                    y: 0,
                    width: 8,
                    height: 8,
                })
                .decode_rgba8_frames(&bytes),
            Err(Error::Unsupported("region-of-interest frame decode"))
        );
    }

    #[test]
    fn decode_rgba8_animation_composites_replace_frames() {
        let bytes = std::fs::read(workspace_path(
            "reference/libjxl/testdata/jxl/blending/cropped_traffic_light.jxl",
        ))
        .unwrap();
        let info = inspect(&bytes).unwrap();
        assert_eq!(info.frames[0].blending_info.mode, BlendMode::Replace);
        assert!(
            info.frames[1..]
                .iter()
                .all(|frame| frame.blending_info.mode == BlendMode::Blend)
        );

        let raw_frames = decode_rgba8_frames(&bytes).unwrap();
        let animation = decode_rgba8_animation(&bytes).unwrap();
        assert_eq!(animation.len(), raw_frames.len());
        assert_eq!(animation.len(), 4);
        for (index, (composited, raw)) in animation.iter().zip(&raw_frames).enumerate() {
            assert_eq!(composited.x, 0, "frame {index} x");
            assert_eq!(composited.y, 0, "frame {index} y");
            assert_eq!(composited.duration, raw.duration, "frame {index} duration");
            assert_eq!(composited.timecode, raw.timecode, "frame {index} timecode");
            assert_eq!(composited.image.width, info.width, "frame {index} width");
            assert_eq!(composited.image.height, info.height, "frame {index} height");
            assert!(
                composited
                    .image
                    .pixels
                    .chunks_exact(4)
                    .all(|pixel| pixel[3] == 255),
                "frame {index} alpha"
            );
        }
        assert_eq!(
            animation[0].image.pixels,
            window_interleaved_u8(
                &raw_frames[0].image.pixels,
                raw_frames[0].image.width,
                4,
                Rect {
                    x: 0,
                    y: 0,
                    width: info.width,
                    height: info.height,
                },
            )
        );
        assert_ne!(
            animation[1].image.pixels, animation[0].image.pixels,
            "blend frame should change the composited canvas"
        );
        assert_eq!(
            animation
                .iter()
                .map(|frame| rgba8_checksum(&frame.image.pixels))
                .collect::<Vec<_>>(),
            vec![
                2_552_782_184_964_619_063,
                3_342_112_483_607_925_487,
                2_450_278_185_916_923_380,
                2_999_354_724_818_656_045,
            ]
        );
        assert_eq!(
            Decoder::new()
                .memory_limit(1)
                .decode_rgba8_animation(&bytes),
            Err(Error::Unsupported("memory limit exceeded"))
        );
        assert_eq!(
            Decoder::new()
                .roi(Rect {
                    x: 0,
                    y: 0,
                    width: 8,
                    height: 8,
                })
                .decode_rgba8_animation(&bytes),
            Err(Error::Unsupported("region-of-interest animation decode"))
        );
    }

    #[test]
    fn decodes_small_combined_var_dct_fixture() {
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

        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
        assert_eq!(decoded.color_channels, 3);
        assert_eq!(decoded.alpha, None);
        assert_eq!(decoded.bit_depth, 8);
        assert_eq!(decoded.pixels, PixelData::U8(vec![0; 8 * 8 * 3]));

        let channels = decode_channels(&bytes).unwrap();
        assert_eq!(channels.width, 8);
        assert_eq!(channels.height, 8);
        assert_eq!(channels.color_channels, 3);
        assert_eq!(channels.alpha, None);
        assert_eq!(channels.bit_depth, 8);
        assert_eq!(channels.channels.len(), 3);
        for channel in &channels.channels {
            assert_eq!(channel.width, 8);
            assert_eq!(channel.height, 8);
            assert_eq!(channel.hshift, 0);
            assert_eq!(channel.vshift, 0);
            assert_eq!(channel.bit_depth, 8);
            assert_eq!(channel.samples, ChannelData::U8(vec![0; 8 * 8]));
        }

        let rgba8 = decode_rgba8(&bytes).unwrap();
        assert_eq!(rgba8.width, 8);
        assert_eq!(rgba8.height, 8);
        assert_eq!(rgba8.pixels.len(), 8 * 8 * 4);
        assert!(
            rgba8
                .pixels
                .chunks_exact(4)
                .all(|pixel| pixel == [0, 0, 0, 255])
        );

        let rgba16 = decode_rgba16(&bytes).unwrap();
        assert_eq!(rgba16.width, 8);
        assert_eq!(rgba16.height, 8);
        assert_eq!(rgba16.pixels.len(), 8 * 8 * 4);
        assert!(
            rgba16
                .pixels
                .chunks_exact(4)
                .all(|pixel| pixel == [0, 0, 0, u16::MAX])
        );

        let roi_channels = roi_decoder.decode_channels(&bytes).unwrap();
        assert_eq!(roi_channels.width, 4);
        assert_eq!(roi_channels.height, 4);
        for channel in &roi_channels.channels {
            assert_eq!(channel.samples, ChannelData::U8(vec![0; 4 * 4]));
        }
        assert_eq!(
            roi_decoder.decode(&bytes).unwrap().pixels,
            PixelData::U8(vec![0; 4 * 4 * 3])
        );
        assert_eq!(
            roi_decoder.decode_rgba8(&bytes).unwrap().pixels.len(),
            4 * 4 * 4
        );
        assert_eq!(
            roi_decoder.decode_rgba16(&bytes).unwrap().pixels.len(),
            4 * 4 * 4
        );
    }

    #[test]
    fn decodes_public_ycbcr_var_dct_reconstruction() {
        let bytes = std::fs::read(workspace_path(
            "reference/libjxl/testdata/jxl/jpeg_reconstruction/1x1_exif_xmp.jxl",
        ))
        .unwrap();
        let info = inspect(&bytes).unwrap();
        assert_eq!(
            info.first_frame.as_ref().unwrap().color_transform,
            jxl_codec::ColorTransform::YCbCr
        );
        let plan = info.first_frame_vardct_plan.as_ref().unwrap();
        assert_eq!(
            plan.ac_global_metadata
                .as_ref()
                .unwrap()
                .all_default_quant_matrices,
            Some(false)
        );
        assert_eq!(plan.ac_global_metadata.as_ref().unwrap().parse_error, None);

        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded.width, 1);
        assert_eq!(decoded.height, 1);
        assert_eq!(decoded.color_channels, 3);
        assert_eq!(decoded.bit_depth, 8);
        assert_eq!(decoded.pixels, PixelData::U8(vec![255, 255, 255]));

        let channels = decode_channels(&bytes).unwrap();
        assert_eq!(channels.width, 1);
        assert_eq!(channels.height, 1);
        assert_eq!(channels.color_channels, 3);
        assert_eq!(channels.bit_depth, 8);
        assert!(channels.channels.iter().take(3).all(|channel| {
            channel.width == 1
                && channel.height == 1
                && channel.bit_depth == 8
                && channel.samples == ChannelData::U8(vec![255])
        }));

        assert_eq!(
            decode_rgba8(&bytes).unwrap(),
            RgbaImage {
                width: 1,
                height: 1,
                pixels: vec![255, 255, 255, 255],
            }
        );
        assert_eq!(
            decode_rgba16(&bytes).unwrap(),
            Rgba16Image {
                width: 1,
                height: 1,
                pixels: vec![u16::MAX, u16::MAX, u16::MAX, u16::MAX],
            }
        );
        assert_eq!(
            decode_linear_rgb(&bytes),
            Err(Error::Unsupported("linear RGB non-XYB output"))
        );
    }

    #[test]
    fn decode_supports_generated_ycbcr_420_var_dct_when_available() {
        let (Some(cjxl), Some(djxl)) = (reference_cjxl(), reference_djxl()) else {
            eprintln!("skipping public YCbCr 4:2:0 VarDCT decode; reference tools are not built");
            return;
        };

        let input =
            workspace_path("reference/libjxl/testdata/jxl/flower/flower.png.im_q85_420.jpg");
        let encoded = unique_temp_path("jxl-ycbcr-420-vardct", "jxl");
        let reference_output = unique_temp_path("jxl-ycbcr-420-vardct-reference", "ppm");
        let cjxl_output = Command::new(&cjxl)
            .arg(&input)
            .arg(&encoded)
            .args(["--allow_jpeg_reconstruction=0", "--container=0", "--quiet"])
            .output()
            .unwrap();
        assert!(
            cjxl_output.status.success(),
            "reference cjxl failed for public YCbCr 4:2:0 VarDCT: {}",
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
            "reference djxl failed for public YCbCr 4:2:0 VarDCT: {}",
            String::from_utf8_lossy(&djxl_output.stderr)
        );
        let reference = std::fs::read(&reference_output).unwrap();
        let _ = std::fs::remove_file(&reference_output);
        let reference = parse_ppm_rgb(&reference);

        let encoded_bytes = std::fs::read(&encoded).unwrap();
        let _ = std::fs::remove_file(&encoded);
        let info = inspect(&encoded_bytes).unwrap();
        let frame = info.first_frame.as_ref().unwrap();
        assert_eq!(frame.color_transform, jxl_codec::ColorTransform::YCbCr);
        assert!(!frame.chroma_subsampling.is_444());
        assert_eq!(frame.chroma_subsampling.h_shift(0), Some(1));
        assert_eq!(frame.chroma_subsampling.v_shift(0), Some(1));
        assert_eq!(frame.chroma_subsampling.h_shift(1), Some(0));
        assert_eq!(frame.chroma_subsampling.v_shift(1), Some(0));
        assert_eq!(frame.chroma_subsampling.h_shift(2), Some(1));
        assert_eq!(frame.chroma_subsampling.v_shift(2), Some(1));

        let decoded = decode(&encoded_bytes).unwrap();
        assert_eq!(decoded.width, reference.width);
        assert_eq!(decoded.height, reference.height);
        assert_eq!(decoded.color_channels, 3);
        assert_eq!(decoded.bit_depth, 8);
        let channels = decode_channels(&encoded_bytes).unwrap();
        assert_decoded_channels_match_image(&channels, &decoded);
        let roi = Rect {
            x: 128,
            y: 96,
            width: 64,
            height: 48,
        };
        let roi_channels = Decoder::new()
            .roi(roi)
            .decode_channels(&encoded_bytes)
            .unwrap();
        assert_roi_matches_full_channels(&roi_channels, &channels, roi);
        let roi_decoded = Decoder::new().roi(roi).decode(&encoded_bytes).unwrap();
        assert_roi_matches_full_image(&roi_decoded, &decoded, roi);
        let rgba = decode_rgba8(&encoded_bytes).unwrap();
        assert_eq!(rgba.width, decoded.width);
        assert_eq!(rgba.height, decoded.height);
        assert!(rgba.pixels.chunks_exact(4).all(|pixel| pixel[3] == 255));
        let roi_rgba = Decoder::new()
            .roi(roi)
            .decode_rgba8(&encoded_bytes)
            .unwrap();
        assert_roi_matches_full_rgba8(&roi_rgba, &rgba, roi);
        let rgba16 = decode_rgba16(&encoded_bytes).unwrap();
        assert_eq!(rgba16.width, decoded.width);
        assert_eq!(rgba16.height, decoded.height);
        assert!(
            rgba16
                .pixels
                .chunks_exact(4)
                .all(|pixel| pixel[3] == u16::MAX)
        );
        let roi_rgba16 = Decoder::new()
            .roi(roi)
            .decode_rgba16(&encoded_bytes)
            .unwrap();
        assert_roi_matches_full_rgba16(&roi_rgba16, &rgba16, roi);

        let width = decoded.width as usize;
        let height = decoded.height as usize;
        let center = (height / 2 * width + width / 2) * 3;
        let metrics = srgb8_oracle_metrics(
            &decoded,
            &reference,
            &[
                0,
                center,
                center + 1,
                ((height - 1) * width + width - 1) * 3 + 2,
            ],
        );
        assert_eq!(
            metrics,
            Srgb8OracleMetrics {
                max_abs_error: 163,
                sum_abs_error: 83_008_163,
                checksum: 11583904892958779392,
                anchors: vec![115, 113, 72, 64],
                reference_anchors: vec![111, 126, 51, 67],
            }
        );
    }

    #[test]
    fn decode_supports_generated_rgb_jpeg_var_dct_when_available() {
        let (Some(cjxl), Some(djxl)) = (reference_cjxl(), reference_djxl()) else {
            eprintln!("skipping public RGB JPEG VarDCT decode; reference tools are not built");
            return;
        };

        let input =
            workspace_path("reference/libjxl/testdata/jxl/flower/flower.png.im_q85_rgb.jpg");
        let encoded = unique_temp_path("jxl-rgb-jpeg-vardct", "jxl");
        let reference_output = unique_temp_path("jxl-rgb-jpeg-vardct-reference", "ppm");
        let cjxl_output = Command::new(&cjxl)
            .arg(&input)
            .arg(&encoded)
            .args(["--allow_jpeg_reconstruction=0", "--container=0", "--quiet"])
            .output()
            .unwrap();
        assert!(
            cjxl_output.status.success(),
            "reference cjxl failed for public RGB JPEG VarDCT: {}",
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
            "reference djxl failed for public RGB JPEG VarDCT: {}",
            String::from_utf8_lossy(&djxl_output.stderr)
        );
        let reference = std::fs::read(&reference_output).unwrap();
        let _ = std::fs::remove_file(&reference_output);
        let reference = parse_ppm_rgb(&reference);

        let encoded_bytes = std::fs::read(&encoded).unwrap();
        let _ = std::fs::remove_file(&encoded);
        let info = inspect(&encoded_bytes).unwrap();
        let frame = info.first_frame.as_ref().unwrap();
        assert_eq!(frame.color_transform, jxl_codec::ColorTransform::None);
        assert!(frame.chroma_subsampling.is_444());

        let decoded = decode(&encoded_bytes).unwrap();
        assert_eq!(decoded.width, reference.width);
        assert_eq!(decoded.height, reference.height);
        assert_eq!(decoded.color_channels, 3);
        assert_eq!(decoded.bit_depth, 8);
        let rgba = decode_rgba8(&encoded_bytes).unwrap();
        assert_eq!(rgba.width, decoded.width);
        assert_eq!(rgba.height, decoded.height);
        assert!(rgba.pixels.chunks_exact(4).all(|pixel| pixel[3] == 255));

        let width = decoded.width as usize;
        let height = decoded.height as usize;
        let center = (height / 2 * width + width / 2) * 3;
        let metrics = srgb8_oracle_metrics(
            &decoded,
            &reference,
            &[
                0,
                center,
                center + 1,
                ((height - 1) * width + width - 1) * 3 + 2,
            ],
        );
        assert_eq!(
            metrics,
            Srgb8OracleMetrics {
                max_abs_error: 172,
                sum_abs_error: 73_181_470,
                checksum: 10211171078667196464,
                anchors: vec![115, 132, 63, 63],
                reference_anchors: vec![108, 127, 50, 67],
            }
        );
    }

    #[test]
    fn decode_supports_generated_gray_jpeg_var_dct_when_available() {
        let (Some(cjxl), Some(djxl)) = (reference_cjxl(), reference_djxl()) else {
            eprintln!(
                "skipping public grayscale JPEG VarDCT decode; reference tools are not built"
            );
            return;
        };

        let input =
            workspace_path("reference/libjxl/testdata/jxl/flower/flower.png.im_q85_gray.jpg");
        let encoded = unique_temp_path("jxl-gray-jpeg-vardct", "jxl");
        let reference_output = unique_temp_path("jxl-gray-jpeg-vardct-reference", "pgm");
        let cjxl_output = Command::new(&cjxl)
            .arg(&input)
            .arg(&encoded)
            .args(["--allow_jpeg_reconstruction=0", "--container=0", "--quiet"])
            .output()
            .unwrap();
        assert!(
            cjxl_output.status.success(),
            "reference cjxl failed for public grayscale JPEG VarDCT: {}",
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
            "reference djxl failed for public grayscale JPEG VarDCT: {}",
            String::from_utf8_lossy(&djxl_output.stderr)
        );
        let reference = std::fs::read(&reference_output).unwrap();
        let _ = std::fs::remove_file(&reference_output);
        let reference = parse_pgm_gray(&reference);

        let encoded_bytes = std::fs::read(&encoded).unwrap();
        let _ = std::fs::remove_file(&encoded);
        let info = inspect(&encoded_bytes).unwrap();
        assert_eq!(info.metadata.color_encoding.color_space, ColorSpace::Gray);
        assert_eq!(info.metadata.num_color_channels(), 1);

        let decoded = decode(&encoded_bytes).unwrap();
        assert_eq!(decoded.width, reference.width);
        assert_eq!(decoded.height, reference.height);
        assert_eq!(decoded.color_channels, 1);
        assert_eq!(decoded.bit_depth, 8);
        let metrics = gray8_oracle_metrics(
            &decoded,
            &reference,
            &[
                0,
                (decoded.height as usize / 2) * decoded.width as usize + decoded.width as usize / 2,
                decoded.width as usize * decoded.height as usize - 1,
            ],
        );
        assert_eq!(
            metrics,
            Srgb8OracleMetrics {
                max_abs_error: 148,
                sum_abs_error: 25_258_774,
                checksum: 8569466704890236855,
                anchors: vec![112, 83, 94],
                reference_anchors: vec![106, 71, 99],
            }
        );

        let channels = decode_channels(&encoded_bytes).unwrap();
        assert_decoded_channels_match_image(&channels, &decoded);
        let roi = Rect {
            x: 128,
            y: 96,
            width: 64,
            height: 48,
        };
        let roi_decoded = Decoder::new().roi(roi).decode(&encoded_bytes).unwrap();
        assert_roi_matches_full_image(&roi_decoded, &decoded, roi);
        let roi_channels = Decoder::new()
            .roi(roi)
            .decode_channels(&encoded_bytes)
            .unwrap();
        assert_roi_matches_full_channels(&roi_channels, &channels, roi);

        let PixelData::U8(gray) = &decoded.pixels else {
            panic!("expected grayscale VarDCT decode to return 8-bit samples");
        };
        let rgba = decode_rgba8(&encoded_bytes).unwrap();
        for (pixel, &gray) in rgba.pixels.chunks_exact(4).zip(gray) {
            assert_eq!(pixel, &[gray, gray, gray, 255]);
        }
        let rgba16 = decode_rgba16(&encoded_bytes).unwrap();
        for pixel in rgba16.pixels.chunks_exact(4) {
            assert_eq!(pixel[0], pixel[1]);
            assert_eq!(pixel[1], pixel[2]);
            assert_eq!(pixel[3], u16::MAX);
        }
    }

    #[test]
    fn decodes_public_spline_rendering() {
        let bytes =
            std::fs::read(workspace_path("reference/libjxl/testdata/jxl/splines.jxl")).unwrap();

        let channels = decode_channels(&bytes).unwrap();
        assert_eq!(channels.width, 2048);
        assert_eq!(channels.height, 2048);
        assert_eq!(channels.channels.len(), 3);
        let channel_max = channels
            .channels
            .iter()
            .filter_map(|channel| match &channel.samples {
                ChannelData::U8(samples) => samples.iter().copied().max(),
                ChannelData::U16(_) => None,
            })
            .max();
        assert_eq!(channel_max, Some(230));
        let roi = Rect {
            x: 512,
            y: 256,
            width: 64,
            height: 48,
        };
        let roi_channels = Decoder::new().roi(roi).decode_channels(&bytes).unwrap();
        assert_roi_matches_full_channels(&roi_channels, &channels, roi);

        let rgba8 = decode_rgba8(&bytes).unwrap();
        assert_eq!(rgba8.width, 2048);
        assert_eq!(rgba8.height, 2048);
        assert_eq!(rgba8.pixels.len(), 2048 * 2048 * 4);
    }

    #[test]
    fn decode_linear_rgb_supports_generated_modular_srgb_when_available() {
        let Some(cjxl) = reference_cjxl() else {
            eprintln!("skipping public modular sRGB linear decode; reference cjxl is not built");
            return;
        };

        let input = unique_temp_path("jxl-modular-srgb-source", "ppm");
        let encoded = unique_temp_path("jxl-modular-srgb", "jxl");
        write_split_vardct_source_ppm(&input);

        let cjxl_output = Command::new(&cjxl)
            .arg(&input)
            .arg(&encoded)
            .args(["-d", "0", "-m", "1", "--container=0", "--quiet"])
            .output()
            .unwrap();
        let source = std::fs::read(&input).unwrap();
        let _ = std::fs::remove_file(&input);
        assert!(
            cjxl_output.status.success(),
            "reference cjxl failed for modular sRGB: {}",
            String::from_utf8_lossy(&cjxl_output.stderr)
        );
        let source = parse_ppm_rgb(&source);

        let encoded_bytes = std::fs::read(&encoded).unwrap();
        let _ = std::fs::remove_file(&encoded);
        let info = inspect(&encoded_bytes).unwrap();
        assert_eq!(info.metadata.color_encoding.color_space, ColorSpace::Rgb);
        assert_eq!(
            info.metadata.color_encoding.transfer_function,
            TransferFunction::Srgb
        );
        assert_eq!(
            info.first_frame.as_ref().unwrap().encoding,
            FrameEncoding::Modular
        );

        let decoded = decode(&encoded_bytes).unwrap();
        assert_eq!(decoded.width, source.width);
        assert_eq!(decoded.height, source.height);
        assert_eq!(decoded.color_channels, 3);
        assert_eq!(decoded.bit_depth, 8);
        assert_eq!(
            Decoder::new()
                .output(DecodeOutput::LinearRgb)
                .decode_output(&encoded_bytes),
            decode_linear_rgb(&encoded_bytes).map(DecodedOutput::LinearRgb)
        );

        let linear = decode_linear_rgb(&encoded_bytes).unwrap();
        assert_eq!(linear.width, source.width);
        assert_eq!(linear.height, source.height);
        assert_eq!(linear.pixels.len(), source.samples.len());
        for (index, (&actual, &sample)) in linear.pixels.iter().zip(&source.samples).enumerate() {
            let expected = srgb_sample_to_linear(sample as f32 / 255.0);
            assert!(
                (actual - expected).abs() <= f32::EPSILON,
                "linear sample mismatch at {index}: actual={actual}, expected={expected}"
            );
        }

        let roi = Rect {
            x: 17,
            y: 19,
            width: 41,
            height: 29,
        };
        let roi_linear = Decoder::new()
            .roi(roi)
            .decode_linear_rgb(&encoded_bytes)
            .unwrap();
        assert_roi_matches_full_linear_rgb(&roi_linear, &linear, roi);
        assert_eq!(
            decode_with_options(
                &encoded_bytes,
                DecodeOptions {
                    output: DecodeOutput::LinearRgb,
                    roi: Some(roi),
                    ..DecodeOptions::default()
                }
            ),
            Ok(DecodedOutput::LinearRgb(roi_linear))
        );
        assert_eq!(
            Decoder::new()
                .memory_limit(1)
                .decode_linear_rgb(&encoded_bytes),
            Err(Error::Unsupported("memory limit exceeded"))
        );
    }

    #[test]
    fn decode_rgba_supports_generated_modular_xyb_when_available() {
        let (Some(cjxl), Some(djxl)) = (reference_cjxl(), reference_djxl()) else {
            eprintln!("skipping public modular XYB decode; reference tools are not built");
            return;
        };

        for (name, extra_args) in [
            ("jxl-modular-xyb", Vec::<&str>::new()),
            ("jxl-modular-xyb-noise", vec!["--photon_noise_iso=3200"]),
        ] {
            let input = unique_temp_path(&format!("{name}-source"), "ppm");
            let encoded = unique_temp_path(name, "jxl");
            let reference_output = unique_temp_path(&format!("{name}-reference"), "ppm");
            write_split_vardct_source_ppm(&input);

            let mut cjxl_command = Command::new(&cjxl);
            cjxl_command
                .arg(&input)
                .arg(&encoded)
                .args(["-d", "1", "-m", "1", "--container=0", "--quiet"])
                .args(extra_args);
            let cjxl_output = cjxl_command.output().unwrap();
            let _ = std::fs::remove_file(&input);
            assert!(
                cjxl_output.status.success(),
                "reference cjxl failed for {name}: {}",
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
                "reference djxl failed for {name}: {}",
                String::from_utf8_lossy(&djxl_output.stderr)
            );
            let reference = std::fs::read(&reference_output).unwrap();
            let _ = std::fs::remove_file(&reference_output);
            let reference = parse_ppm_rgb(&reference);

            let encoded_bytes = std::fs::read(&encoded).unwrap();
            let _ = std::fs::remove_file(&encoded);
            let info = inspect(&encoded_bytes).unwrap();
            assert_eq!(
                info.first_frame.as_ref().unwrap().color_transform,
                jxl_codec::ColorTransform::Xyb,
                "{name}"
            );

            let decoded = decode(&encoded_bytes).unwrap();
            let rgba = decode_rgba8(&encoded_bytes).unwrap();
            let linear = decode_linear_rgb(&encoded_bytes).unwrap();
            assert_eq!(rgba.width, 320, "{name}");
            assert_eq!(rgba.height, 192, "{name}");
            assert!(rgba.pixels.chunks_exact(4).all(|pixel| pixel[3] == 255));
            assert_eq!(linear.width, 320, "{name}");
            assert_eq!(linear.height, 192, "{name}");
            assert_eq!(linear.pixels.len(), 320 * 192 * 3, "{name}");
            assert!(
                linear.pixels.iter().all(|sample| sample.is_finite()),
                "{name}"
            );
            assert!(
                linear
                    .pixels
                    .chunks_exact(3)
                    .any(|pixel| pixel[0] != 0.0 || pixel[1] != 0.0 || pixel[2] != 0.0),
                "{name}"
            );
            let roi = Rect {
                x: 17,
                y: 19,
                width: 41,
                height: 29,
            };
            let roi_linear = Decoder::new()
                .roi(roi)
                .decode_linear_rgb(&encoded_bytes)
                .unwrap();
            assert_roi_matches_full_linear_rgb(&roi_linear, &linear, roi);

            let metrics = srgb8_oracle_metrics(
                &decoded,
                &reference,
                &[0, 320 * 192 * 3 / 2, 320 * 192 * 3 - 1],
            );
            assert!(
                metrics.max_abs_error <= 1,
                "modular XYB max error for {name}: {}",
                metrics.max_abs_error
            );
            assert!(
                metrics.sum_abs_error <= 50_000,
                "modular XYB absolute error sum for {name}: {}",
                metrics.sum_abs_error
            );
        }
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
        let linear = decode_linear_rgb(&encoded_bytes).unwrap();
        let configured_linear = Decoder::new()
            .output(DecodeOutput::LinearRgb)
            .decode_output(&encoded_bytes)
            .unwrap();
        let roi_decoded = roi_decoder.decode(&encoded_bytes).unwrap();
        let roi_rgba = roi_decoder.decode_rgba8(&encoded_bytes).unwrap();
        let roi_rgba16 = roi_decoder.decode_rgba16(&encoded_bytes).unwrap();
        let roi_linear = roi_decoder.decode_linear_rgb(&encoded_bytes).unwrap();
        let roi_configured_linear = decode_with_options(
            &encoded_bytes,
            DecodeOptions {
                output: DecodeOutput::LinearRgb,
                roi: Some(roi),
                ..DecodeOptions::default()
            },
        )
        .unwrap();
        let pass0_decoded = Decoder::new()
            .vardct_pass(0)
            .decode(&encoded_bytes)
            .unwrap();
        let pass0_linear = Decoder::new()
            .vardct_pass(0)
            .decode_linear_rgb(&encoded_bytes)
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
        assert_eq!(linear.width, 320);
        assert_eq!(linear.height, 192);
        assert_eq!(linear.pixels.len(), 320 * 192 * 3);
        assert!(linear.pixels.iter().all(|sample| sample.is_finite()));
        assert!(
            linear
                .pixels
                .chunks_exact(3)
                .any(|pixel| pixel[0] != 0.0 || pixel[1] != 0.0 || pixel[2] != 0.0)
        );
        assert_roi_matches_full_linear_rgb(&roi_linear, &linear, roi);
        assert_eq!(configured_linear, DecodedOutput::LinearRgb(linear.clone()));
        assert_eq!(
            roi_configured_linear,
            DecodedOutput::LinearRgb(roi_linear.clone())
        );
        assert_eq!(
            Decoder::new()
                .memory_limit(1)
                .decode_linear_rgb(&encoded_bytes),
            Err(Error::Unsupported("memory limit exceeded"))
        );
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
                    sum_abs_error: 13_657_167,
                    checksum: 11_814_460_042_320_799_823,
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
        assert_eq!(pass0_linear.width, 320);
        assert_eq!(pass0_linear.height, 192);
        assert_eq!(pass0_linear.pixels.len(), 320 * 192 * 3);
        assert!(pass0_linear.pixels.iter().all(|sample| sample.is_finite()));
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
            out_of_bounds_roi_decoder.decode_linear_rgb(&encoded_bytes),
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
        assert_eq!(
            missing_pass_decoder.decode_linear_rgb(&encoded_bytes),
            Err(Error::Unsupported("VarDCT image reconstruction"))
        );
    }

    #[test]
    fn decode_supports_multigroup_var_dct_block_contexts_when_available() {
        let Some(cjxl) = reference_cjxl() else {
            eprintln!(
                "skipping multi-group VarDCT block-context decode; reference cjxl is not built"
            );
            return;
        };

        let input = unique_temp_path("jxl-rgba-vardct-multigroup-source", "ppm");
        let encoded = unique_temp_path("jxl-rgba-vardct-multigroup", "jxl");
        write_multigroup_vardct_source_ppm(&input);

        let cjxl_output = Command::new(&cjxl)
            .arg(&input)
            .arg(&encoded)
            .args([
                "-d",
                "1.0",
                "-e",
                "7",
                "-m",
                "0",
                "--container=0",
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
            let output = unique_temp_path("jxl-rgba-vardct-multigroup-reference", "ppm");
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
        let info = inspect(&encoded_bytes).unwrap();
        let vardct = info.first_frame_vardct.as_ref().unwrap();
        let plan = info.first_frame_vardct_plan.as_ref().unwrap();
        let global = plan.global.as_ref().unwrap();
        assert_eq!(vardct.width, 1024);
        assert_eq!(vardct.height, 512);
        assert_eq!(vardct.groups_x, 4);
        assert_eq!(vardct.groups_y, 2);
        assert_eq!(plan.ac_group_metadata.len(), 8);
        assert!(!global.block_context_map.all_default);
        assert!(!global.block_context_map.qf_thresholds.is_empty());
        assert_eq!(global.block_context_map.num_contexts, 6);
        assert!(
            plan.ac_group_metadata
                .iter()
                .all(|group| group.spatial_with_dc_grid.is_some())
        );

        let decoded = decode(&encoded_bytes).unwrap();
        let rgba = decode_rgba8(&encoded_bytes).unwrap();
        assert_eq!(decoded.width, 1024);
        assert_eq!(decoded.height, 512);
        assert_eq!(decoded.color_channels, 3);
        assert_eq!(decoded.alpha, None);
        let PixelData::U8(decoded_pixels) = &decoded.pixels else {
            panic!("expected multi-group VarDCT decode to return 8-bit RGB");
        };
        assert_eq!(decoded_pixels.len(), 1024 * 512 * 3);
        assert_decoded_channels_match_image(&decode_channels(&encoded_bytes).unwrap(), &decoded);
        assert_eq!(rgba.width, 1024);
        assert_eq!(rgba.height, 512);
        assert!(rgba.pixels.chunks_exact(4).all(|pixel| pixel[3] == 255));

        if let Some(reference) = &reference {
            let metrics = srgb8_oracle_metrics(
                &decoded,
                reference,
                &[0, decoded_pixels.len() / 2, decoded_pixels.len() - 1],
            );
            assert_eq!(
                metrics,
                Srgb8OracleMetrics {
                    max_abs_error: 254,
                    sum_abs_error: 164301497,
                    checksum: 16729806134982033339,
                    anchors: vec![0, 13, 0],
                    reference_anchors: vec![0, 4, 254],
                }
            );
        } else {
            eprintln!("skipping multi-group VarDCT djxl comparison; tool is not built");
        }
    }

    #[test]
    fn decode_supports_resampled_generated_var_dct_when_available() {
        let Some(cjxl) = reference_cjxl() else {
            eprintln!("skipping resampled VarDCT decode; reference cjxl is not built");
            return;
        };

        let input = unique_temp_path("jxl-rgba-vardct-resampled-source", "ppm");
        let encoded = unique_temp_path("jxl-rgba-vardct-resampled", "jxl");
        write_resampled_vardct_source_ppm(&input);

        let cjxl_output = Command::new(&cjxl)
            .arg(&input)
            .arg(&encoded)
            .args([
                "-d",
                "1.0",
                "-m",
                "0",
                "--container=0",
                "--resampling",
                "2",
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
            let output = unique_temp_path("jxl-rgba-vardct-resampled-reference", "ppm");
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
        let info = inspect(&encoded_bytes).unwrap();
        let vardct = info.first_frame_vardct.as_ref().unwrap();
        assert_eq!(vardct.width, 96);
        assert_eq!(vardct.height, 64);
        assert_eq!(vardct.coded_width, 48);
        assert_eq!(vardct.coded_height, 32);
        assert_eq!(vardct.upsampling, 2);
        assert!(vardct.is_combined);

        let roi = Rect {
            x: 13,
            y: 11,
            width: 29,
            height: 23,
        };
        let decoded_channels = decode_channels(&encoded_bytes).unwrap();
        let roi_channels = Decoder::new()
            .roi(roi)
            .decode_channels(&encoded_bytes)
            .unwrap();
        let decoded = decode(&encoded_bytes).unwrap();
        let rgba = decode_rgba8(&encoded_bytes).unwrap();
        let rgba16 = decode_rgba16(&encoded_bytes).unwrap();
        let roi_decoded = Decoder::new().roi(roi).decode(&encoded_bytes).unwrap();
        let roi_rgba = Decoder::new()
            .roi(roi)
            .decode_rgba8(&encoded_bytes)
            .unwrap();
        assert_eq!(decoded.width, 96);
        assert_eq!(decoded.height, 64);
        assert_eq!(decoded.color_channels, 3);
        assert_eq!(decoded.alpha, None);
        assert_eq!(decoded.bit_depth, 8);
        let PixelData::U8(decoded_pixels) = &decoded.pixels else {
            panic!("expected resampled VarDCT decode to return 8-bit RGB");
        };
        assert_eq!(decoded_pixels.len(), 96 * 64 * 3);
        assert!(
            decoded_pixels
                .chunks_exact(3)
                .any(|pixel| pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0)
        );
        assert_decoded_channels_match_image(&decoded_channels, &decoded);
        assert_roi_matches_full_channels(&roi_channels, &decoded_channels, roi);
        assert_roi_matches_full_image(&roi_decoded, &decoded, roi);

        assert_eq!(rgba.width, 96);
        assert_eq!(rgba.height, 64);
        assert_eq!(rgba.pixels.len(), 96 * 64 * 4);
        assert!(rgba.pixels.chunks_exact(4).all(|pixel| pixel[3] == 255));
        assert_roi_matches_full_rgba8(&roi_rgba, &rgba, roi);

        assert_eq!(rgba16.width, 96);
        assert_eq!(rgba16.height, 64);
        assert_eq!(rgba16.pixels.len(), 96 * 64 * 4);
        assert!(
            rgba16
                .pixels
                .chunks_exact(4)
                .all(|pixel| pixel[3] == u16::MAX)
        );

        if let Some(reference) = &reference {
            let metrics = srgb8_oracle_metrics(
                &decoded,
                reference,
                &[0, decoded_pixels.len() / 2, decoded_pixels.len() - 1],
            );
            assert_eq!(
                metrics,
                Srgb8OracleMetrics {
                    max_abs_error: 249,
                    sum_abs_error: 1_864_842,
                    checksum: 3_383_233_688_300_954_244,
                    anchors: vec![14, 15, 0],
                    reference_anchors: vec![6, 1, 249],
                }
            );
        } else {
            eprintln!("skipping resampled VarDCT djxl comparison; tool is not built");
        }
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

        let reference = reference_djxl().map(|djxl| {
            let output = unique_temp_path("jxl-vardct-alpha-reference", "pam");
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
            parse_pam_rgba(&reference)
        });

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
        if let Some(reference) = &reference {
            assert_eq!(reference.width, channels.width);
            assert_eq!(reference.height, channels.height);
            assert_eq!(
                reference
                    .samples
                    .chunks_exact(4)
                    .map(|pixel| pixel[3] as u8)
                    .collect::<Vec<_>>(),
                alpha.clone()
            );
        } else {
            eprintln!("skipping VarDCT alpha djxl comparison; tool is not built");
        }

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

        let pass0_channels = Decoder::new()
            .vardct_pass(0)
            .decode_channels(&encoded_bytes)
            .unwrap();
        assert_eq!(pass0_channels.alpha, channels.alpha);
        let ChannelData::U8(pass0_alpha) = &pass0_channels.channels[3].samples else {
            panic!("expected 8-bit VarDCT pass alpha channel");
        };
        assert_eq!(pass0_alpha, &expected_alpha);

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

        let linear = decode_linear_rgb(&encoded_bytes).unwrap();
        assert_eq!(linear.width, 320);
        assert_eq!(linear.height, 192);
        assert_eq!(linear.pixels.len(), 320 * 192 * 3);
        assert!(linear.pixels.iter().all(|sample| sample.is_finite()));
        assert!(
            linear
                .pixels
                .chunks_exact(3)
                .any(|pixel| pixel[0] != 0.0 || pixel[1] != 0.0 || pixel[2] != 0.0)
        );
        let roi_linear = Decoder::new()
            .roi(roi)
            .decode_linear_rgb(&encoded_bytes)
            .unwrap();
        assert_roi_matches_full_linear_rgb(&roi_linear, &linear, roi);
        let pass0_linear = Decoder::new()
            .vardct_pass(0)
            .decode_linear_rgb(&encoded_bytes)
            .unwrap();
        assert_eq!(pass0_linear.width, linear.width);
        assert_eq!(pass0_linear.height, linear.height);
        assert_eq!(pass0_linear.pixels.len(), linear.pixels.len());

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

        let pass0_rgba8 = Decoder::new()
            .vardct_pass(0)
            .decode_rgba8(&encoded_bytes)
            .unwrap();
        assert_eq!(
            pass0_rgba8
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
    fn decode_channels_exposes_generated_gray_var_dct_alpha_when_available() {
        let Some(cjxl) = reference_cjxl() else {
            eprintln!(
                "skipping grayscale VarDCT alpha raw-channel decode; reference cjxl is not built"
            );
            return;
        };

        let input = unique_temp_path("jxl-gray-vardct-alpha-source", "pam");
        let encoded = unique_temp_path("jxl-gray-vardct-alpha", "jxl");
        let source = write_gray_alpha_source_pam(&input);

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

        let reference = reference_djxl().map(|djxl| {
            let output = unique_temp_path("jxl-gray-vardct-alpha-reference", "pam");
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
            parse_pam_gray_alpha(&reference)
        });

        let encoded_bytes = std::fs::read(&encoded).unwrap();
        let _ = std::fs::remove_file(&encoded);
        let info = inspect(&encoded_bytes).unwrap();
        assert!(info.first_frame_vardct.is_some());
        assert_eq!(info.metadata.color_encoding.color_space, ColorSpace::Gray);
        assert_eq!(info.metadata.num_color_channels(), 1);
        assert_eq!(
            raw_alpha_info(&info.metadata).unwrap(),
            Some(AlphaInfo {
                bit_depth: 8,
                premultiplied: false,
            })
        );

        let channels = decode_channels(&encoded_bytes).unwrap();
        assert_eq!(channels.width, source.width);
        assert_eq!(channels.height, source.height);
        assert_eq!(channels.color_channels, 1);
        assert_eq!(
            channels.alpha,
            Some(AlphaInfo {
                bit_depth: 8,
                premultiplied: false,
            })
        );
        assert_eq!(channels.bit_depth, 8);
        assert_eq!(channels.channels.len(), 2);
        for channel in &channels.channels {
            assert_eq!(channel.width, source.width);
            assert_eq!(channel.height, source.height);
            assert_eq!(channel.hshift, 0);
            assert_eq!(channel.vshift, 0);
            assert_eq!(channel.bit_depth, 8);
        }
        let ChannelData::U8(gray) = &channels.channels[0].samples else {
            panic!("expected 8-bit grayscale VarDCT color channel");
        };
        let ChannelData::U8(alpha) = &channels.channels[1].samples else {
            panic!("expected 8-bit grayscale VarDCT alpha channel");
        };
        assert_eq!(alpha, &source.alpha);

        if let Some(reference) = &reference {
            let reference_gray = PgmGray {
                width: reference.width,
                height: reference.height,
                samples: reference.gray.iter().copied().map(u16::from).collect(),
            };
            let metrics = gray8_samples_oracle_metrics(
                gray,
                channels.width,
                channels.height,
                &reference_gray,
                &[0, gray.len() / 2, gray.len() - 1],
            );
            assert_eq!(
                metrics,
                Srgb8OracleMetrics {
                    max_abs_error: 255,
                    sum_abs_error: 42_617,
                    checksum: 16_107_006_911_524_474_586,
                    anchors: vec![100, 255, 177],
                    reference_anchors: vec![3, 0, 249],
                }
            );
            assert_eq!(alpha, &reference.alpha);
        } else {
            eprintln!("skipping grayscale VarDCT alpha djxl comparison; tool is not built");
        }

        let decoded = decode(&encoded_bytes).unwrap();
        assert_eq!(decoded.width, source.width);
        assert_eq!(decoded.height, source.height);
        assert_eq!(decoded.color_channels, 1);
        assert_eq!(decoded.alpha, channels.alpha);
        assert_decoded_channels_match_image(&channels, &decoded);
        let PixelData::U8(decoded_pixels) = &decoded.pixels else {
            panic!("expected 8-bit grayscale-alpha VarDCT image");
        };
        assert_eq!(decoded_pixels.len(), source.gray_alpha.len());
        assert_eq!(
            decoded_pixels
                .chunks_exact(2)
                .map(|pixel| pixel[1])
                .collect::<Vec<_>>(),
            source.alpha
        );

        let roi = Rect {
            x: 5,
            y: 3,
            width: 19,
            height: 11,
        };
        let roi_channels = Decoder::new()
            .roi(roi)
            .decode_channels(&encoded_bytes)
            .unwrap();
        assert_roi_matches_full_channels(&roi_channels, &channels, roi);
        let roi_decoded = Decoder::new().roi(roi).decode(&encoded_bytes).unwrap();
        assert_roi_matches_full_image(&roi_decoded, &decoded, roi);

        let rgba8 = decode_rgba8(&encoded_bytes).unwrap();
        assert_eq!(rgba8.width, source.width);
        assert_eq!(rgba8.height, source.height);
        for ((pixel, &gray), &alpha) in rgba8.pixels.chunks_exact(4).zip(gray).zip(&source.alpha) {
            assert_eq!(pixel, &[gray, gray, gray, alpha]);
        }
        let roi_rgba8 = Decoder::new()
            .roi(roi)
            .decode_rgba8(&encoded_bytes)
            .unwrap();
        assert_roi_matches_full_rgba8(&roi_rgba8, &rgba8, roi);

        let rgba16 = decode_rgba16(&encoded_bytes).unwrap();
        assert_eq!(rgba16.width, source.width);
        assert_eq!(rgba16.height, source.height);
        for (pixel, &alpha) in rgba16.pixels.chunks_exact(4).zip(&source.alpha) {
            assert_eq!(pixel[0], pixel[1]);
            assert_eq!(pixel[1], pixel[2]);
            assert_eq!(pixel[3], u16::from(alpha) * 257);
        }
        let roi_rgba16 = Decoder::new()
            .roi(roi)
            .decode_rgba16(&encoded_bytes)
            .unwrap();
        assert_roi_matches_full_rgba16(&roi_rgba16, &rgba16, roi);

        let pass0_channels = Decoder::new()
            .vardct_pass(0)
            .decode_channels(&encoded_bytes)
            .unwrap();
        assert_eq!(pass0_channels.color_channels, 1);
        assert_eq!(pass0_channels.alpha, channels.alpha);
        let ChannelData::U8(pass0_alpha) = &pass0_channels.channels[1].samples else {
            panic!("expected 8-bit grayscale VarDCT pass alpha channel");
        };
        assert_eq!(pass0_alpha, &source.alpha);
    }

    #[test]
    fn decode_outputs_generated_gray_var_dct_alpha16_when_available() {
        let Some(cjxl) = reference_cjxl() else {
            eprintln!("skipping grayscale VarDCT alpha16 decode; reference cjxl is not built");
            return;
        };

        let input = unique_temp_path("jxl-gray-vardct-alpha16-source", "pam");
        let encoded = unique_temp_path("jxl-gray-vardct-alpha16", "jxl");
        let source = write_gray_alpha_source_pam16(&input);

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

        let reference = reference_djxl().map(|djxl| {
            let output = unique_temp_path("jxl-gray-vardct-alpha16-reference", "pam");
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
            parse_pam_gray_alpha16(&reference)
        });

        let encoded_bytes = std::fs::read(&encoded).unwrap();
        let _ = std::fs::remove_file(&encoded);
        let info = inspect(&encoded_bytes).unwrap();
        assert!(info.first_frame_vardct.is_some());
        assert_eq!(info.metadata.color_encoding.color_space, ColorSpace::Gray);
        assert_eq!(info.metadata.num_color_channels(), 1);
        assert_eq!(
            raw_alpha_info(&info.metadata).unwrap(),
            Some(AlphaInfo {
                bit_depth: 16,
                premultiplied: false,
            })
        );

        let channels = decode_channels(&encoded_bytes).unwrap();
        assert_eq!(channels.width, source.width);
        assert_eq!(channels.height, source.height);
        assert_eq!(channels.color_channels, 1);
        assert_eq!(
            channels.alpha,
            Some(AlphaInfo {
                bit_depth: 16,
                premultiplied: false,
            })
        );
        assert_eq!(channels.bit_depth, 8);
        assert_eq!(channels.channels.len(), 2);
        let gray_channel = &channels.channels[0];
        assert_eq!(gray_channel.width, source.width);
        assert_eq!(gray_channel.height, source.height);
        assert_eq!(gray_channel.hshift, 0);
        assert_eq!(gray_channel.vshift, 0);
        assert_eq!(gray_channel.bit_depth, 8);
        let alpha_channel = &channels.channels[1];
        assert_eq!(alpha_channel.width, source.width);
        assert_eq!(alpha_channel.height, source.height);
        assert_eq!(alpha_channel.hshift, 0);
        assert_eq!(alpha_channel.vshift, 0);
        assert_eq!(alpha_channel.bit_depth, 16);
        let ChannelData::U8(gray) = &gray_channel.samples else {
            panic!("expected 8-bit grayscale VarDCT color channel");
        };
        let ChannelData::U16(alpha) = &alpha_channel.samples else {
            panic!("expected 16-bit grayscale VarDCT alpha channel");
        };
        assert_eq!(alpha, &source.alpha);

        if let Some(reference) = &reference {
            let scaled_reference_gray = PgmGray {
                width: reference.width,
                height: reference.height,
                samples: reference
                    .gray
                    .iter()
                    .map(|&sample| ((u32::from(sample) * 255 + 32_767) / 65_535) as u16)
                    .collect(),
            };
            let metrics = gray8_samples_oracle_metrics(
                gray,
                channels.width,
                channels.height,
                &scaled_reference_gray,
                &[0, gray.len() / 2, gray.len() - 1],
            );
            assert_eq!(
                metrics,
                Srgb8OracleMetrics {
                    max_abs_error: 255,
                    sum_abs_error: 36_224,
                    checksum: 5_450_027_092_381_118_560,
                    anchors: vec![53, 57, 131],
                    reference_anchors: vec![0, 197, 137],
                }
            );
            assert_eq!(alpha, &reference.alpha);
        } else {
            eprintln!("skipping grayscale VarDCT alpha16 djxl comparison; tool is not built");
        }

        let decoded = decode(&encoded_bytes).unwrap();
        assert_eq!(decoded.width, source.width);
        assert_eq!(decoded.height, source.height);
        assert_eq!(decoded.color_channels, 1);
        assert_eq!(decoded.alpha, channels.alpha);
        assert_eq!(decoded.bit_depth, 16);
        let PixelData::U16(decoded_pixels) = &decoded.pixels else {
            panic!("expected 16-bit grayscale-alpha VarDCT image");
        };
        assert_eq!(decoded_pixels.len(), source.gray_alpha.len());
        assert_eq!(
            decoded_pixels
                .chunks_exact(2)
                .map(|pixel| pixel[1])
                .collect::<Vec<_>>(),
            source.alpha
        );

        let roi = Rect {
            x: 4,
            y: 2,
            width: 17,
            height: 9,
        };
        let roi_channels = Decoder::new()
            .roi(roi)
            .decode_channels(&encoded_bytes)
            .unwrap();
        assert_roi_matches_full_channels(&roi_channels, &channels, roi);
        let roi_decoded = Decoder::new().roi(roi).decode(&encoded_bytes).unwrap();
        assert_roi_matches_full_image(&roi_decoded, &decoded, roi);

        let rgba8 = decode_rgba8(&encoded_bytes).unwrap();
        assert_eq!(rgba8.width, source.width);
        assert_eq!(rgba8.height, source.height);
        assert_eq!(
            rgba8
                .pixels
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>(),
            source
                .alpha
                .iter()
                .map(|&alpha| ((u32::from(alpha) * 255 + 32_767) / 65_535) as u8)
                .collect::<Vec<_>>()
        );
        for pixel in rgba8.pixels.chunks_exact(4) {
            assert_eq!(pixel[0], pixel[1]);
            assert_eq!(pixel[1], pixel[2]);
        }
        let roi_rgba8 = Decoder::new()
            .roi(roi)
            .decode_rgba8(&encoded_bytes)
            .unwrap();
        assert_roi_matches_full_rgba8(&roi_rgba8, &rgba8, roi);

        let rgba16 = decode_rgba16(&encoded_bytes).unwrap();
        assert_eq!(rgba16.width, source.width);
        assert_eq!(rgba16.height, source.height);
        for (pixel, &alpha) in rgba16.pixels.chunks_exact(4).zip(&source.alpha) {
            assert_eq!(pixel[0], pixel[1]);
            assert_eq!(pixel[1], pixel[2]);
            assert_eq!(pixel[3], alpha);
        }
        let roi_rgba16 = Decoder::new()
            .roi(roi)
            .decode_rgba16(&encoded_bytes)
            .unwrap();
        assert_roi_matches_full_rgba16(&roi_rgba16, &rgba16, roi);
    }

    #[test]
    fn decode_outputs_subsampled_gray_var_dct_alpha16_when_available() {
        let Some(cjxl) = reference_cjxl() else {
            eprintln!(
                "skipping subsampled grayscale VarDCT alpha16 decode; reference cjxl is not built"
            );
            return;
        };

        let input = unique_temp_path("jxl-gray-vardct-alpha16-subsampled-source", "pam");
        let encoded = unique_temp_path("jxl-gray-vardct-alpha16-subsampled", "jxl");
        let source = write_gray_alpha_source_pam16(&input);

        let cjxl_output = Command::new(&cjxl)
            .arg(&input)
            .arg(&encoded)
            .args([
                "-d",
                "1.0",
                "-m",
                "0",
                "--container=0",
                "--ec_resampling",
                "2",
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
            let output = unique_temp_path("jxl-gray-vardct-alpha16-subsampled-reference", "pam");
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
            parse_pam_gray_alpha16(&reference)
        });

        let encoded_bytes = std::fs::read(&encoded).unwrap();
        let _ = std::fs::remove_file(&encoded);
        let info = inspect(&encoded_bytes).unwrap();
        assert!(info.first_frame_vardct.is_some());
        assert_eq!(info.metadata.color_encoding.color_space, ColorSpace::Gray);
        assert_eq!(info.metadata.num_color_channels(), 1);
        assert_eq!(
            raw_alpha_info(&info.metadata).unwrap(),
            Some(AlphaInfo {
                bit_depth: 16,
                premultiplied: false,
            })
        );

        let channels = decode_channels(&encoded_bytes).unwrap();
        assert_eq!(channels.width, source.width);
        assert_eq!(channels.height, source.height);
        assert_eq!(channels.color_channels, 1);
        assert_eq!(
            channels.alpha,
            Some(AlphaInfo {
                bit_depth: 16,
                premultiplied: false,
            })
        );
        assert_eq!(channels.bit_depth, 8);
        assert_eq!(channels.channels.len(), 2);
        let gray_channel = &channels.channels[0];
        assert_eq!(gray_channel.width, source.width);
        assert_eq!(gray_channel.height, source.height);
        assert_eq!(gray_channel.hshift, 0);
        assert_eq!(gray_channel.vshift, 0);
        let alpha_channel = &channels.channels[1];
        assert_eq!(alpha_channel.width, source.width.div_ceil(2));
        assert_eq!(alpha_channel.height, source.height.div_ceil(2));
        assert_eq!(alpha_channel.hshift, 1);
        assert_eq!(alpha_channel.vshift, 1);
        assert_eq!(alpha_channel.bit_depth, 16);
        let ChannelData::U16(alpha) = &alpha_channel.samples else {
            panic!("expected 16-bit subsampled grayscale VarDCT alpha channel");
        };
        assert_eq!(
            alpha.len(),
            alpha_channel.width as usize * alpha_channel.height as usize
        );
        assert_eq!(alpha.iter().copied().min(), Some(2_115));
        assert_eq!(alpha.iter().copied().max(), Some(64_239));
        let alpha_checksum = alpha
            .iter()
            .enumerate()
            .fold(0u64, |checksum, (index, sample)| {
                checksum
                    .wrapping_mul(1_099_511_628_211)
                    .wrapping_add(index as u64)
                    .rotate_left(11)
                    ^ u64::from(*sample)
            });
        assert_eq!(alpha_checksum, 16_058_973_670_731_747_394);

        let roi = Rect {
            x: 4,
            y: 2,
            width: 17,
            height: 9,
        };
        let roi_channels = Decoder::new()
            .roi(roi)
            .decode_channels(&encoded_bytes)
            .unwrap();
        assert_eq!(roi_channels.width, roi.width);
        assert_eq!(roi_channels.height, roi.height);
        let roi_alpha_channel = &roi_channels.channels[1];
        assert_eq!(roi_alpha_channel.width, 9);
        assert_eq!(roi_alpha_channel.height, 5);
        assert_eq!(roi_alpha_channel.hshift, 1);
        assert_eq!(roi_alpha_channel.vshift, 1);
        let ChannelData::U16(roi_alpha) = &roi_alpha_channel.samples else {
            panic!("expected 16-bit subsampled grayscale VarDCT ROI alpha channel");
        };
        let shifted_roi = Rect {
            x: 2,
            y: 1,
            width: 9,
            height: 5,
        };
        assert_eq!(
            roi_alpha,
            &window_u16(alpha, alpha_channel.width, shifted_roi)
        );

        if let Some(reference) = &reference {
            let rgba16 = decode_rgba16(&encoded_bytes).unwrap();
            let rgba16_alpha = rgba16
                .pixels
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>();
            assert_alpha16_matches_reference(&rgba16_alpha, &reference.alpha);

            let rgba8 = decode_rgba8(&encoded_bytes).unwrap();
            let rgba8_alpha = rgba8
                .pixels
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>();
            let reference_alpha8 = reference
                .alpha
                .iter()
                .map(|&alpha| ((u32::from(alpha) * 255 + 32_767) / 65_535) as u8)
                .collect::<Vec<_>>();
            assert_alpha_matches_reference(&rgba8_alpha, &reference_alpha8);
        } else {
            eprintln!(
                "skipping subsampled grayscale VarDCT alpha16 djxl comparison; tool is not built"
            );
        }
    }

    #[test]
    fn decode_channels_exposes_subsampled_var_dct_alpha_when_available() {
        let Some(cjxl) = reference_cjxl() else {
            eprintln!(
                "skipping subsampled VarDCT alpha raw-channel decode; reference cjxl is not built"
            );
            return;
        };

        let input = unique_temp_path("jxl-vardct-alpha-subsampled-source", "pam");
        let encoded = unique_temp_path("jxl-vardct-alpha-subsampled", "jxl");
        let reference_output = unique_temp_path("jxl-vardct-alpha-subsampled-reference", "pam");
        write_vardct_alpha_source_pam(&input);

        let cjxl_output = Command::new(&cjxl)
            .arg(&input)
            .arg(&encoded)
            .args([
                "-d",
                "1.0",
                "-m",
                "0",
                "--container=0",
                "--ec_resampling",
                "2",
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
        let channels = decode_channels(&encoded_bytes).unwrap();
        assert_eq!(channels.width, 320);
        assert_eq!(channels.height, 192);
        assert_eq!(
            channels.alpha,
            Some(AlphaInfo {
                bit_depth: 8,
                premultiplied: false,
            })
        );
        assert_eq!(channels.channels.len(), 4);
        let alpha_channel = &channels.channels[3];
        assert_eq!(alpha_channel.width, 160);
        assert_eq!(alpha_channel.height, 96);
        assert_eq!(alpha_channel.hshift, 1);
        assert_eq!(alpha_channel.vshift, 1);
        assert_eq!(alpha_channel.bit_depth, 8);
        let ChannelData::U8(alpha) = &alpha_channel.samples else {
            panic!("expected 8-bit subsampled VarDCT alpha channel");
        };
        assert_eq!(alpha.len(), 160 * 96);
        assert_eq!(alpha.iter().copied().min(), Some(31));
        assert_eq!(alpha.iter().copied().max(), Some(225));
        let alpha_checksum = alpha
            .iter()
            .enumerate()
            .fold(0u64, |checksum, (index, sample)| {
                checksum
                    .wrapping_mul(1_099_511_628_211)
                    .wrapping_add(index as u64)
                    .rotate_left(11)
                    ^ u64::from(*sample)
            });
        assert_eq!(alpha_checksum, 497_137_486_042_797_447);

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
        let roi_alpha_channel = &roi_channels.channels[3];
        assert_eq!(roi_alpha_channel.width, 19);
        assert_eq!(roi_alpha_channel.height, 15);
        assert_eq!(roi_alpha_channel.hshift, 1);
        assert_eq!(roi_alpha_channel.vshift, 1);
        let ChannelData::U8(roi_alpha) = &roi_alpha_channel.samples else {
            panic!("expected 8-bit subsampled VarDCT ROI alpha channel");
        };
        let shifted_roi = Rect {
            x: 8,
            y: 9,
            width: 19,
            height: 15,
        };
        assert_eq!(roi_alpha, &window_u8(alpha, 160, shifted_roi));

        if let Some(djxl) = reference_djxl() {
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
            let reference_alpha = reference
                .samples
                .chunks_exact(4)
                .map(|pixel| pixel[3] as u8)
                .collect::<Vec<_>>();

            let decoded = decode(&encoded_bytes).unwrap();
            let PixelData::U8(decoded_pixels) = &decoded.pixels else {
                panic!("expected 8-bit subsampled-alpha VarDCT image");
            };
            let decoded_alpha = decoded_pixels
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>();
            assert_alpha_matches_reference(&decoded_alpha, &reference_alpha);

            let rgba8 = decode_rgba8(&encoded_bytes).unwrap();
            let rgba8_alpha = rgba8
                .pixels
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>();
            assert_alpha_matches_reference(&rgba8_alpha, &reference_alpha);

            let rgba16 = decode_rgba16(&encoded_bytes).unwrap();
            let rgba16_alpha = rgba16
                .pixels
                .chunks_exact(4)
                .map(|pixel| ((u32::from(pixel[3]) + 128) / 257) as u8)
                .collect::<Vec<_>>();
            assert_alpha_matches_reference(&rgba16_alpha, &reference_alpha);

            let roi_rgba8 = Decoder::new()
                .roi(roi)
                .decode_rgba8(&encoded_bytes)
                .unwrap();
            let roi_rgba8_alpha = roi_rgba8
                .pixels
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>();
            assert_alpha_matches_reference(
                &roi_rgba8_alpha,
                &window_u8(&reference_alpha, reference.width, roi),
            );
        } else {
            eprintln!("skipping subsampled VarDCT alpha djxl comparison; tool is not built");
        }

        let _ = std::fs::remove_file(&encoded);
        let _ = std::fs::remove_file(&reference_output);
    }

    #[test]
    fn decode_uses_custom_var_dct_alpha_upsampling_weights_when_available() {
        let channels = DecodedChannels {
            width: 4,
            height: 4,
            color_channels: 3,
            alpha: Some(AlphaInfo {
                bit_depth: 8,
                premultiplied: false,
            }),
            bit_depth: 8,
            channels: vec![
                decoded_u8_channel(4, 4, &[1; 16]),
                decoded_u8_channel(4, 4, &[2; 16]),
                decoded_u8_channel(4, 4, &[3; 16]),
                DecodedChannel {
                    width: 2,
                    height: 2,
                    hshift: 1,
                    vshift: 1,
                    bit_depth: 8,
                    samples: ChannelData::U8(vec![10, 20, 30, 40]),
                },
            ],
        };
        let mut nearest_neighbor_weights = vec![0.0; 15];
        nearest_neighbor_weights[9] = 1.0;
        let transform_data = CustomTransformData {
            custom_weights_mask: 0x1,
            upsampling2_weights: Some(nearest_neighbor_weights),
            ..CustomTransformData::default()
        };

        let rgba = rgba8_from_decoded_channels_with_transform_data(
            &channels,
            Some(3),
            Some(&transform_data),
        )
        .unwrap();
        let alpha = rgba
            .pixels
            .chunks_exact(4)
            .map(|pixel| pixel[3])
            .collect::<Vec<_>>();
        assert_eq!(
            alpha,
            vec![
                10, 10, 20, 20, 10, 10, 20, 20, 30, 30, 40, 40, 30, 30, 40, 40
            ]
        );
    }

    #[test]
    fn decode_outputs_quarter_resolution_var_dct_alpha_when_available() {
        let Some(cjxl) = reference_cjxl() else {
            eprintln!("skipping 4x subsampled VarDCT alpha decode; reference cjxl is not built");
            return;
        };

        let input = unique_temp_path("jxl-vardct-alpha-subsampled4-source", "pam");
        let encoded = unique_temp_path("jxl-vardct-alpha-subsampled4", "jxl");
        let reference_output = unique_temp_path("jxl-vardct-alpha-subsampled4-reference", "pam");
        write_vardct_alpha_source_pam(&input);

        let cjxl_output = Command::new(&cjxl)
            .arg(&input)
            .arg(&encoded)
            .args([
                "-d",
                "1.0",
                "-m",
                "0",
                "--container=0",
                "--ec_resampling",
                "4",
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
        let channels = decode_channels(&encoded_bytes).unwrap();
        assert_eq!(channels.width, 320);
        assert_eq!(channels.height, 192);
        assert_eq!(
            channels.alpha,
            Some(AlphaInfo {
                bit_depth: 8,
                premultiplied: false,
            })
        );
        assert_eq!(channels.channels.len(), 4);
        let alpha_channel = &channels.channels[3];
        assert_eq!(alpha_channel.width, 80);
        assert_eq!(alpha_channel.height, 48);
        assert_eq!(alpha_channel.hshift, 2);
        assert_eq!(alpha_channel.vshift, 2);
        let ChannelData::U8(alpha) = &alpha_channel.samples else {
            panic!("expected 8-bit 4x subsampled VarDCT alpha channel");
        };
        assert_eq!(alpha.len(), 80 * 48);

        let decoded = decode(&encoded_bytes).unwrap();
        assert_eq!(decoded.width, 320);
        assert_eq!(decoded.height, 192);
        assert_eq!(decoded.alpha, channels.alpha);
        let rgba8 = decode_rgba8(&encoded_bytes).unwrap();
        assert_eq!(rgba8.width, 320);
        assert_eq!(rgba8.height, 192);
        assert_eq!(rgba8.pixels.len(), 320 * 192 * 4);
        let rgba16 = decode_rgba16(&encoded_bytes).unwrap();
        assert_eq!(rgba16.width, 320);
        assert_eq!(rgba16.height, 192);
        assert_eq!(rgba16.pixels.len(), 320 * 192 * 4);

        let roi = Rect {
            x: 17,
            y: 19,
            width: 37,
            height: 29,
        };
        let roi_rgba8 = Decoder::new()
            .roi(roi)
            .decode_rgba8(&encoded_bytes)
            .unwrap();
        assert_eq!(roi_rgba8.width, roi.width);
        assert_eq!(roi_rgba8.height, roi.height);
        assert_eq!(
            roi_rgba8.pixels.len(),
            roi.width as usize * roi.height as usize * 4
        );

        if let Some(djxl) = reference_djxl() {
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
            let reference_alpha = reference
                .samples
                .chunks_exact(4)
                .map(|pixel| pixel[3] as u8)
                .collect::<Vec<_>>();

            let PixelData::U8(decoded_pixels) = &decoded.pixels else {
                panic!("expected 8-bit 4x subsampled-alpha VarDCT image");
            };
            let decoded_alpha = decoded_pixels
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>();
            assert_alpha_matches_reference(&decoded_alpha, &reference_alpha);

            let rgba8_alpha = rgba8
                .pixels
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>();
            assert_alpha_matches_reference(&rgba8_alpha, &reference_alpha);

            let roi_rgba8_alpha = roi_rgba8
                .pixels
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>();
            assert_alpha_matches_reference(
                &roi_rgba8_alpha,
                &window_u8(&reference_alpha, reference.width, roi),
            );
        } else {
            eprintln!("skipping 4x subsampled VarDCT alpha djxl comparison; tool is not built");
        }

        let _ = std::fs::remove_file(&encoded);
        let _ = std::fs::remove_file(&reference_output);
    }

    #[test]
    fn decode_outputs_eighth_resolution_var_dct_alpha_when_available() {
        let Some(cjxl) = reference_cjxl() else {
            eprintln!("skipping 8x subsampled VarDCT alpha decode; reference cjxl is not built");
            return;
        };

        let input = unique_temp_path("jxl-vardct-alpha-subsampled8-source", "pam");
        let encoded = unique_temp_path("jxl-vardct-alpha-subsampled8", "jxl");
        let reference_output = unique_temp_path("jxl-vardct-alpha-subsampled8-reference", "pam");
        write_vardct_alpha_source_pam(&input);

        let cjxl_output = Command::new(&cjxl)
            .arg(&input)
            .arg(&encoded)
            .args([
                "-d",
                "1.0",
                "-m",
                "0",
                "--container=0",
                "--ec_resampling",
                "8",
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
        let channels = decode_channels(&encoded_bytes).unwrap();
        assert_eq!(channels.width, 320);
        assert_eq!(channels.height, 192);
        assert_eq!(
            channels.alpha,
            Some(AlphaInfo {
                bit_depth: 8,
                premultiplied: false,
            })
        );
        assert_eq!(channels.channels.len(), 4);
        let alpha_channel = &channels.channels[3];
        assert_eq!(alpha_channel.width, 40);
        assert_eq!(alpha_channel.height, 24);
        assert_eq!(alpha_channel.hshift, 3);
        assert_eq!(alpha_channel.vshift, 3);
        let ChannelData::U8(alpha) = &alpha_channel.samples else {
            panic!("expected 8-bit 8x subsampled VarDCT alpha channel");
        };
        assert_eq!(alpha.len(), 40 * 24);

        let decoded = decode(&encoded_bytes).unwrap();
        assert_eq!(decoded.width, 320);
        assert_eq!(decoded.height, 192);
        assert_eq!(decoded.alpha, channels.alpha);
        let rgba8 = decode_rgba8(&encoded_bytes).unwrap();
        assert_eq!(rgba8.width, 320);
        assert_eq!(rgba8.height, 192);
        assert_eq!(rgba8.pixels.len(), 320 * 192 * 4);

        let roi = Rect {
            x: 17,
            y: 19,
            width: 37,
            height: 29,
        };
        let roi_rgba8 = Decoder::new()
            .roi(roi)
            .decode_rgba8(&encoded_bytes)
            .unwrap();
        assert_eq!(roi_rgba8.width, roi.width);
        assert_eq!(roi_rgba8.height, roi.height);
        assert_eq!(
            roi_rgba8.pixels.len(),
            roi.width as usize * roi.height as usize * 4
        );

        if let Some(djxl) = reference_djxl() {
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
            let reference_alpha = reference
                .samples
                .chunks_exact(4)
                .map(|pixel| pixel[3] as u8)
                .collect::<Vec<_>>();

            let PixelData::U8(decoded_pixels) = &decoded.pixels else {
                panic!("expected 8-bit 8x subsampled-alpha VarDCT image");
            };
            let decoded_alpha = decoded_pixels
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>();
            assert_alpha_matches_reference(&decoded_alpha, &reference_alpha);

            let rgba8_alpha = rgba8
                .pixels
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>();
            assert_alpha_matches_reference(&rgba8_alpha, &reference_alpha);

            let roi_rgba8_alpha = roi_rgba8
                .pixels
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>();
            assert_alpha_matches_reference(
                &roi_rgba8_alpha,
                &window_u8(&reference_alpha, reference.width, roi),
            );
        } else {
            eprintln!("skipping 8x subsampled VarDCT alpha djxl comparison; tool is not built");
        }

        let _ = std::fs::remove_file(&encoded);
        let _ = std::fs::remove_file(&reference_output);
    }

    #[test]
    fn decode_outputs_generated_var_dct_alpha16_when_available() {
        let Some(cjxl) = reference_cjxl() else {
            eprintln!("skipping VarDCT alpha16 decode; reference cjxl is not built");
            return;
        };

        let input = unique_temp_path("jxl-vardct-alpha16-source", "pam");
        let encoded = unique_temp_path("jxl-vardct-alpha16", "jxl");
        let expected_alpha = write_vardct_alpha_source_pam16(&input);

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
                bit_depth: 16,
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
                bit_depth: 16,
                premultiplied: false,
            })
        );
        assert_eq!(channels.bit_depth, 8);
        assert_eq!(channels.channels.len(), 4);
        let ChannelData::U16(alpha) = &channels.channels[3].samples else {
            panic!("expected 16-bit VarDCT alpha channel");
        };
        assert_eq!(alpha, &expected_alpha);

        let decoded = decode(&encoded_bytes).unwrap();
        assert_eq!(decoded.width, 320);
        assert_eq!(decoded.height, 192);
        assert_eq!(decoded.color_channels, 3);
        assert_eq!(decoded.alpha, channels.alpha);
        assert_eq!(decoded.bit_depth, 16);
        let PixelData::U16(decoded_pixels) = &decoded.pixels else {
            panic!("expected 16-bit VarDCT decoded image");
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
        assert_eq!(
            rgba8
                .pixels
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>(),
            expected_alpha
                .iter()
                .map(|&alpha| ((u32::from(alpha) * 255 + 32_767) / 65_535) as u8)
                .collect::<Vec<_>>()
        );

        let rgba16 = decode_rgba16(&encoded_bytes).unwrap();
        assert_eq!(rgba16.width, 320);
        assert_eq!(rgba16.height, 192);
        assert_eq!(
            rgba16
                .pixels
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>(),
            expected_alpha
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
    fn decode_channels_exposes_generated_gray_var_dct_depth_when_available() {
        let Some(cjxl) = reference_cjxl() else {
            eprintln!(
                "skipping grayscale VarDCT depth raw-channel decode; reference cjxl is not built"
            );
            return;
        };

        let input = unique_temp_path("jxl-gray-vardct-depth-source", "pam");
        let encoded = unique_temp_path("jxl-gray-vardct-depth", "jxl");
        let source = write_gray_depth_source_pam(&input);

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
        assert_eq!(info.metadata.color_encoding.color_space, ColorSpace::Gray);
        assert_eq!(info.metadata.num_color_channels(), 1);
        assert_eq!(info.metadata.extra_channels.len(), 1);
        assert_eq!(
            info.metadata.extra_channels[0].channel_type,
            ExtraChannelType::Depth
        );

        let channels = decode_channels(&encoded_bytes).unwrap();
        assert_eq!(channels.width, source.width);
        assert_eq!(channels.height, source.height);
        assert_eq!(channels.color_channels, 1);
        assert_eq!(channels.alpha, None);
        assert_eq!(channels.bit_depth, 8);
        assert_eq!(channels.channels.len(), 2);
        let gray_channel = &channels.channels[0];
        assert_eq!(gray_channel.width, source.width);
        assert_eq!(gray_channel.height, source.height);
        assert_eq!(gray_channel.hshift, 0);
        assert_eq!(gray_channel.vshift, 0);
        assert_eq!(gray_channel.bit_depth, 8);
        let depth_channel = &channels.channels[1];
        assert_eq!(depth_channel.width, source.width);
        assert_eq!(depth_channel.height, source.height);
        assert_eq!(depth_channel.hshift, 0);
        assert_eq!(depth_channel.vshift, 0);
        assert_eq!(depth_channel.bit_depth, 8);
        let ChannelData::U8(depth) = &depth_channel.samples else {
            panic!("expected 8-bit grayscale VarDCT depth channel");
        };
        assert_eq!(depth, &source.depth);

        let roi = Rect {
            x: 3,
            y: 4,
            width: 13,
            height: 9,
        };
        let roi_channels = Decoder::new()
            .roi(roi)
            .decode_channels(&encoded_bytes)
            .unwrap();
        assert_roi_matches_full_channels(&roi_channels, &channels, roi);

        let decoded = decode(&encoded_bytes).unwrap();
        assert_eq!(decoded.width, source.width);
        assert_eq!(decoded.height, source.height);
        assert_eq!(decoded.color_channels, 1);
        assert_eq!(decoded.alpha, None);
        assert_eq!(decoded.bit_depth, 8);
        let PixelData::U8(decoded_gray) = &decoded.pixels else {
            panic!("expected 8-bit grayscale VarDCT decode");
        };
        let ChannelData::U8(channel_gray) = &gray_channel.samples else {
            panic!("expected 8-bit grayscale VarDCT color channel");
        };
        assert_eq!(decoded_gray, channel_gray);
        let roi_decoded = Decoder::new().roi(roi).decode(&encoded_bytes).unwrap();
        assert_roi_matches_full_image(&roi_decoded, &decoded, roi);

        let rgba8 = decode_rgba8(&encoded_bytes).unwrap();
        assert_eq!(rgba8.width, source.width);
        assert_eq!(rgba8.height, source.height);
        for (pixel, &gray) in rgba8.pixels.chunks_exact(4).zip(channel_gray) {
            assert_eq!(pixel, &[gray, gray, gray, 255]);
        }
        let rgba16 = decode_rgba16(&encoded_bytes).unwrap();
        assert_eq!(rgba16.width, source.width);
        assert_eq!(rgba16.height, source.height);
        for pixel in rgba16.pixels.chunks_exact(4) {
            assert_eq!(pixel[0], pixel[1]);
            assert_eq!(pixel[1], pixel[2]);
            assert_eq!(pixel[3], u16::MAX);
        }
    }

    #[test]
    fn decode_channels_exposes_subsampled_gray_var_dct_depth_when_available() {
        let Some(cjxl) = reference_cjxl() else {
            eprintln!(
                "skipping subsampled grayscale VarDCT depth raw-channel decode; reference cjxl is not built"
            );
            return;
        };

        let input = unique_temp_path("jxl-gray-vardct-depth-subsampled-source", "pam");
        let encoded = unique_temp_path("jxl-gray-vardct-depth-subsampled", "jxl");
        let source = write_gray_depth_source_pam(&input);

        let cjxl_output = Command::new(&cjxl)
            .arg(&input)
            .arg(&encoded)
            .args([
                "-d",
                "1.0",
                "-m",
                "0",
                "--container=0",
                "--ec_resampling",
                "2",
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
        let info = inspect(&encoded_bytes).unwrap();
        assert!(info.first_frame_vardct.is_some());
        assert_eq!(info.metadata.color_encoding.color_space, ColorSpace::Gray);
        assert_eq!(info.metadata.num_color_channels(), 1);
        assert_eq!(info.metadata.extra_channels.len(), 1);
        assert_eq!(
            info.metadata.extra_channels[0].channel_type,
            ExtraChannelType::Depth
        );

        let channels = decode_channels(&encoded_bytes).unwrap();
        assert_eq!(channels.width, source.width);
        assert_eq!(channels.height, source.height);
        assert_eq!(channels.color_channels, 1);
        assert_eq!(channels.alpha, None);
        assert_eq!(channels.bit_depth, 8);
        assert_eq!(channels.channels.len(), 2);
        let gray_channel = &channels.channels[0];
        assert_eq!(gray_channel.width, source.width);
        assert_eq!(gray_channel.height, source.height);
        assert_eq!(gray_channel.hshift, 0);
        assert_eq!(gray_channel.vshift, 0);
        let depth_channel = &channels.channels[1];
        assert_eq!(depth_channel.width, source.width.div_ceil(2));
        assert_eq!(depth_channel.height, source.height.div_ceil(2));
        assert_eq!(depth_channel.hshift, 1);
        assert_eq!(depth_channel.vshift, 1);
        assert_eq!(depth_channel.bit_depth, 8);
        let ChannelData::U8(depth) = &depth_channel.samples else {
            panic!("expected 8-bit subsampled grayscale VarDCT depth channel");
        };
        assert_eq!(
            depth.len(),
            depth_channel.width as usize * depth_channel.height as usize
        );
        assert_eq!(depth.iter().copied().min(), Some(28));
        assert_eq!(depth.iter().copied().max(), Some(222));
        let depth_checksum = depth
            .iter()
            .enumerate()
            .fold(0u64, |checksum, (index, sample)| {
                checksum
                    .wrapping_mul(1_099_511_628_211)
                    .wrapping_add(index as u64)
                    .rotate_left(11)
                    ^ u64::from(*sample)
            });
        assert_eq!(depth_checksum, 6_577_634_186_647_311_850);

        let roi = Rect {
            x: 3,
            y: 4,
            width: 13,
            height: 9,
        };
        let roi_channels = Decoder::new()
            .roi(roi)
            .decode_channels(&encoded_bytes)
            .unwrap();
        assert_eq!(roi_channels.width, roi.width);
        assert_eq!(roi_channels.height, roi.height);
        assert_eq!(roi_channels.color_channels, channels.color_channels);
        assert_eq!(roi_channels.alpha, None);
        assert_eq!(roi_channels.channels.len(), channels.channels.len());
        let roi_depth_channel = &roi_channels.channels[1];
        assert_eq!(roi_depth_channel.width, 7);
        assert_eq!(roi_depth_channel.height, 5);
        assert_eq!(roi_depth_channel.hshift, 1);
        assert_eq!(roi_depth_channel.vshift, 1);
        let ChannelData::U8(roi_depth) = &roi_depth_channel.samples else {
            panic!("expected 8-bit subsampled grayscale VarDCT ROI depth channel");
        };
        let shifted_roi = Rect {
            x: 1,
            y: 2,
            width: 7,
            height: 5,
        };
        assert_eq!(
            roi_depth,
            &window_u8(depth, depth_channel.width, shifted_roi)
        );

        let decoded = decode(&encoded_bytes).unwrap();
        assert_eq!(decoded.width, source.width);
        assert_eq!(decoded.height, source.height);
        assert_eq!(decoded.color_channels, 1);
        assert_eq!(decoded.alpha, None);

        let rgba8 = decode_rgba8(&encoded_bytes).unwrap();
        assert_eq!(rgba8.width, source.width);
        assert_eq!(rgba8.height, source.height);
        assert!(rgba8.pixels.chunks_exact(4).all(|pixel| pixel[3] == 255));
    }

    #[test]
    fn decode_outputs_generated_var_dct_alpha_depth_when_available() {
        let Some(cjxl) = reference_cjxl() else {
            eprintln!("skipping VarDCT alpha+depth decode; reference cjxl is not built");
            return;
        };

        let input = unique_temp_path("jxl-vardct-alpha-depth-source", "pam");
        let encoded = unique_temp_path("jxl-vardct-alpha-depth", "jxl");
        let source = write_vardct_alpha_depth_source_pam(&input);

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
        assert_eq!(info.metadata.extra_channels.len(), 2);
        assert_eq!(
            info.metadata.extra_channels[0].channel_type,
            ExtraChannelType::Depth
        );
        assert_eq!(
            info.metadata.extra_channels[1].channel_type,
            ExtraChannelType::Alpha
        );

        let channels = decode_channels(&encoded_bytes).unwrap();
        assert_eq!(channels.width, source.width);
        assert_eq!(channels.height, source.height);
        assert_eq!(channels.color_channels, 3);
        assert_eq!(
            channels.alpha,
            Some(AlphaInfo {
                bit_depth: 8,
                premultiplied: false,
            })
        );
        assert_eq!(channels.channels.len(), 5);
        let ChannelData::U8(depth) = &channels.channels[3].samples else {
            panic!("expected 8-bit VarDCT depth channel");
        };
        assert_eq!(depth, &source.depth);
        let ChannelData::U8(alpha) = &channels.channels[4].samples else {
            panic!("expected 8-bit VarDCT alpha channel");
        };
        assert_eq!(alpha, &source.alpha);

        let decoded = decode(&encoded_bytes).unwrap();
        assert_eq!(decoded.width, source.width);
        assert_eq!(decoded.height, source.height);
        assert_eq!(decoded.alpha, channels.alpha);
        let PixelData::U8(decoded_pixels) = &decoded.pixels else {
            panic!("expected 8-bit VarDCT decoded image");
        };
        assert_eq!(
            decoded_pixels
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>(),
            source.alpha
        );

        let rgba8 = decode_rgba8(&encoded_bytes).unwrap();
        assert_eq!(rgba8.width, source.width);
        assert_eq!(rgba8.height, source.height);
        assert_eq!(
            rgba8
                .pixels
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>(),
            source.alpha
        );

        let roi = Rect {
            x: 23,
            y: 17,
            width: 41,
            height: 31,
        };
        let roi_channels = Decoder::new()
            .roi(roi)
            .decode_channels(&encoded_bytes)
            .unwrap();
        assert_eq!(roi_channels.width, roi.width);
        assert_eq!(roi_channels.height, roi.height);
        assert_eq!(roi_channels.alpha, channels.alpha);
        let ChannelData::U8(roi_depth) = &roi_channels.channels[3].samples else {
            panic!("expected 8-bit VarDCT ROI depth channel");
        };
        assert_eq!(roi_depth, &window_u8(&source.depth, source.width, roi));
        let ChannelData::U8(roi_alpha) = &roi_channels.channels[4].samples else {
            panic!("expected 8-bit VarDCT ROI alpha channel");
        };
        assert_eq!(roi_alpha, &window_u8(&source.alpha, source.width, roi));

        let pass0_channels = Decoder::new()
            .vardct_pass(0)
            .decode_channels(&encoded_bytes)
            .unwrap();
        assert_eq!(pass0_channels.alpha, channels.alpha);
        let ChannelData::U8(pass0_depth) = &pass0_channels.channels[3].samples else {
            panic!("expected 8-bit VarDCT pass depth channel");
        };
        assert_eq!(pass0_depth, &source.depth);
        let ChannelData::U8(pass0_alpha) = &pass0_channels.channels[4].samples else {
            panic!("expected 8-bit VarDCT pass alpha channel");
        };
        assert_eq!(pass0_alpha, &source.alpha);
    }

    #[test]
    fn decode_channels_exposes_combined_generated_var_dct_depth_when_available() {
        let Some(cjxl) = reference_cjxl() else {
            eprintln!("skipping combined VarDCT depth decode; reference cjxl is not built");
            return;
        };

        let input = unique_temp_path("jxl-vardct-depth-source", "pam");
        let encoded = unique_temp_path("jxl-vardct-depth", "jxl");
        let expected_depth = write_rgb_depth_source_pam(&input);

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
        assert_eq!(channels.width, 35);
        assert_eq!(channels.height, 21);
        assert_eq!(channels.color_channels, 3);
        assert_eq!(channels.alpha, None);
        assert_eq!(channels.bit_depth, 8);
        assert_eq!(channels.channels.len(), 4);
        let depth_channel = &channels.channels[3];
        assert_eq!(depth_channel.width, 35);
        assert_eq!(depth_channel.height, 21);
        assert_eq!(depth_channel.hshift, 0);
        assert_eq!(depth_channel.vshift, 0);
        assert_eq!(depth_channel.bit_depth, 8);
        let ChannelData::U8(depth) = &depth_channel.samples else {
            panic!("expected 8-bit combined VarDCT depth channel");
        };
        assert_eq!(depth, &expected_depth);
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
    fn decode_rgba8_upsamples_subsampled_modular_alpha_when_available() {
        let (Some(cjxl), Some(djxl)) = (reference_cjxl(), reference_djxl()) else {
            eprintln!(
                "skipping subsampled modular alpha comparison; reference tools are not built"
            );
            return;
        };

        let input = unique_temp_path("jxl-rgba8-alpha-subsampled-source", "pam");
        let encoded = unique_temp_path("jxl-rgba8-alpha-subsampled", "jxl");
        let reference_output = unique_temp_path("jxl-rgba8-alpha-subsampled-reference", "pam");
        write_alpha_source_pam(&input);

        let cjxl_output = Command::new(&cjxl)
            .arg(&input)
            .arg(&encoded)
            .args([
                "-d",
                "0",
                "-m",
                "1",
                "--container=0",
                "--ec_resampling",
                "2",
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
        let reference_alpha = reference
            .samples
            .chunks_exact(4)
            .map(|pixel| pixel[3] as u8)
            .collect::<Vec<_>>();
        let encoded_bytes = std::fs::read(&encoded).unwrap();
        let _ = std::fs::remove_file(&encoded);
        let _ = std::fs::remove_file(&reference_output);

        let channels = decode_channels(&encoded_bytes).unwrap();
        assert_eq!(channels.width, reference.width);
        assert_eq!(channels.height, reference.height);
        assert_eq!(channels.color_channels, 3);
        assert_eq!(
            channels.alpha,
            Some(AlphaInfo {
                bit_depth: 8,
                premultiplied: false,
            })
        );
        assert_eq!(channels.channels.len(), 4);
        let alpha_channel = &channels.channels[3];
        assert_eq!(alpha_channel.width, reference.width.div_ceil(2));
        assert_eq!(alpha_channel.height, reference.height.div_ceil(2));
        assert_eq!(alpha_channel.hshift, 1);
        assert_eq!(alpha_channel.vshift, 1);
        assert_eq!(alpha_channel.bit_depth, 8);
        let ChannelData::U8(alpha) = &alpha_channel.samples else {
            panic!("expected 8-bit subsampled modular alpha channel");
        };
        assert_eq!(alpha.iter().copied().min(), Some(32));
        assert_eq!(alpha.iter().copied().max(), Some(238));
        let alpha_checksum = alpha
            .iter()
            .enumerate()
            .fold(0u64, |checksum, (index, sample)| {
                checksum
                    .wrapping_mul(1_099_511_628_211)
                    .wrapping_add(index as u64)
                    .rotate_left(11)
                    ^ u64::from(*sample)
            });
        assert_eq!(alpha_checksum, 6_357_654_557_809_416_742);

        let decoded = decode(&encoded_bytes).unwrap();
        assert_eq!(decoded.width, reference.width);
        assert_eq!(decoded.height, reference.height);
        assert_eq!(decoded.color_channels, 3);
        assert_eq!(decoded.alpha, channels.alpha);
        let PixelData::U8(decoded_pixels) = &decoded.pixels else {
            panic!("expected 8-bit modular alpha decode");
        };
        let decoded_alpha = decoded_pixels
            .chunks_exact(4)
            .map(|pixel| pixel[3])
            .collect::<Vec<_>>();
        assert_alpha_matches_reference(&decoded_alpha, &reference_alpha);

        let rgba = decode_rgba8(&encoded_bytes).unwrap();
        let rgba_alpha = rgba
            .pixels
            .chunks_exact(4)
            .map(|pixel| pixel[3])
            .collect::<Vec<_>>();
        assert_alpha_matches_reference(&rgba_alpha, &reference_alpha);

        let roi = Rect {
            x: 17,
            y: 9,
            width: 23,
            height: 21,
        };
        let roi_rgba = Decoder::new()
            .roi(roi)
            .decode_rgba8(&encoded_bytes)
            .unwrap();
        let roi_alpha = roi_rgba
            .pixels
            .chunks_exact(4)
            .map(|pixel| pixel[3])
            .collect::<Vec<_>>();
        assert_alpha_matches_reference(
            &roi_alpha,
            &window_u8(&reference_alpha, reference.width, roi),
        );

        let roi_channels = Decoder::new()
            .roi(roi)
            .decode_channels(&encoded_bytes)
            .unwrap();
        let roi_alpha_channel = &roi_channels.channels[3];
        assert_eq!(roi_alpha_channel.width, 12);
        assert_eq!(roi_alpha_channel.height, 11);
        assert_eq!(roi_alpha_channel.hshift, 1);
        assert_eq!(roi_alpha_channel.vshift, 1);
    }

    #[test]
    fn decode_rgba8_upsamples_wider_subsampled_modular_alpha_when_available() {
        let (Some(cjxl), Some(djxl)) = (reference_cjxl(), reference_djxl()) else {
            eprintln!(
                "skipping wider subsampled modular alpha comparison; reference tools are not built"
            );
            return;
        };

        for (resampling, shift, expected_min, expected_max, expected_checksum) in [
            (4u32, 2i32, 84u8, 169u8, 5_389_941_862_787_020_957u64),
            (8u32, 3i32, 112u8, 148u8, 3_913_142_275_485_721_700u64),
        ] {
            let input = unique_temp_path("jxl-rgba8-alpha-wide-subsampled-source", "pam");
            let encoded = unique_temp_path("jxl-rgba8-alpha-wide-subsampled", "jxl");
            let reference_output =
                unique_temp_path("jxl-rgba8-alpha-wide-subsampled-reference", "pam");
            write_alpha_source_pam(&input);

            let cjxl_output = Command::new(&cjxl)
                .arg(&input)
                .arg(&encoded)
                .args([
                    "-d",
                    "0",
                    "-m",
                    "1",
                    "--container=0",
                    "--ec_resampling",
                    &resampling.to_string(),
                    "--quiet",
                ])
                .output()
                .unwrap();
            let _ = std::fs::remove_file(&input);
            assert!(
                cjxl_output.status.success(),
                "reference cjxl failed for resampling {resampling}: {}",
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
                "reference djxl failed for resampling {resampling}: {}",
                String::from_utf8_lossy(&djxl_output.stderr)
            );

            let reference = std::fs::read(&reference_output).unwrap();
            let reference = parse_pam_rgba(&reference);
            let reference_alpha = reference
                .samples
                .chunks_exact(4)
                .map(|pixel| pixel[3] as u8)
                .collect::<Vec<_>>();
            let encoded_bytes = std::fs::read(&encoded).unwrap();
            let _ = std::fs::remove_file(&encoded);
            let _ = std::fs::remove_file(&reference_output);

            let channels = decode_channels(&encoded_bytes).unwrap();
            let alpha_channel = &channels.channels[3];
            assert_eq!(alpha_channel.width, reference.width.div_ceil(resampling));
            assert_eq!(alpha_channel.height, reference.height.div_ceil(resampling));
            assert_eq!(alpha_channel.hshift, shift);
            assert_eq!(alpha_channel.vshift, shift);
            let ChannelData::U8(alpha) = &alpha_channel.samples else {
                panic!("expected 8-bit wider subsampled modular alpha channel");
            };
            assert_eq!(alpha.iter().copied().min(), Some(expected_min));
            assert_eq!(alpha.iter().copied().max(), Some(expected_max));
            let alpha_checksum =
                alpha
                    .iter()
                    .enumerate()
                    .fold(0u64, |checksum, (index, sample)| {
                        checksum
                            .wrapping_mul(1_099_511_628_211)
                            .wrapping_add(index as u64)
                            .rotate_left(11)
                            ^ u64::from(*sample)
                    });
            assert_eq!(alpha_checksum, expected_checksum);

            let rgba = decode_rgba8(&encoded_bytes).unwrap();
            let rgba_alpha = rgba
                .pixels
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>();
            assert_alpha_matches_reference(&rgba_alpha, &reference_alpha);

            let roi = Rect {
                x: 17,
                y: 9,
                width: 23,
                height: 21,
            };
            let roi_rgba = Decoder::new()
                .roi(roi)
                .decode_rgba8(&encoded_bytes)
                .unwrap();
            let roi_alpha = roi_rgba
                .pixels
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>();
            assert_alpha_matches_reference(
                &roi_alpha,
                &window_u8(&reference_alpha, reference.width, roi),
            );
        }
    }

    #[test]
    fn decode_rgba16_upsamples_subsampled_modular_alpha_when_available() {
        let (Some(cjxl), Some(djxl)) = (reference_cjxl(), reference_djxl()) else {
            eprintln!(
                "skipping 16-bit subsampled modular alpha comparison; reference tools are not built"
            );
            return;
        };

        let input = unique_temp_path("jxl-rgba16-alpha-subsampled-source", "pam");
        let encoded = unique_temp_path("jxl-rgba16-alpha-subsampled", "jxl");
        let reference_output = unique_temp_path("jxl-rgba16-alpha-subsampled-reference", "pam");
        let source = write_alpha_source_pam16(&input);

        let cjxl_output = Command::new(&cjxl)
            .arg(&input)
            .arg(&encoded)
            .args([
                "-d",
                "0",
                "-m",
                "1",
                "--container=0",
                "--ec_resampling",
                "2",
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
        let reference = parse_pam_rgba16(&reference);
        assert_eq!(reference.width, source.width);
        assert_eq!(reference.height, source.height);
        assert_eq!(
            source.alpha.len(),
            source.width as usize * source.height as usize
        );
        let reference_alpha = reference
            .samples
            .chunks_exact(4)
            .map(|pixel| pixel[3])
            .collect::<Vec<_>>();
        let encoded_bytes = std::fs::read(&encoded).unwrap();
        let _ = std::fs::remove_file(&encoded);
        let _ = std::fs::remove_file(&reference_output);

        let channels = decode_channels(&encoded_bytes).unwrap();
        assert_eq!(channels.width, reference.width);
        assert_eq!(channels.height, reference.height);
        assert_eq!(channels.color_channels, 3);
        assert_eq!(
            channels.alpha,
            Some(AlphaInfo {
                bit_depth: 16,
                premultiplied: false,
            })
        );
        assert_eq!(channels.channels.len(), 4);
        let alpha_channel = &channels.channels[3];
        assert_eq!(alpha_channel.width, reference.width.div_ceil(2));
        assert_eq!(alpha_channel.height, reference.height.div_ceil(2));
        assert_eq!(alpha_channel.hshift, 1);
        assert_eq!(alpha_channel.vshift, 1);
        assert_eq!(alpha_channel.bit_depth, 16);
        let ChannelData::U16(alpha) = &alpha_channel.samples else {
            panic!("expected 16-bit subsampled modular alpha channel");
        };
        assert_eq!(alpha.iter().copied().min(), Some(1_604));
        assert_eq!(alpha.iter().copied().max(), Some(63_694));
        let alpha_checksum = alpha
            .iter()
            .enumerate()
            .fold(0u64, |checksum, (index, sample)| {
                checksum
                    .wrapping_mul(1_099_511_628_211)
                    .wrapping_add(index as u64)
                    .rotate_left(11)
                    ^ u64::from(*sample)
            });
        assert_eq!(alpha_checksum, 8_116_986_388_688_171_143);

        let decoded = decode(&encoded_bytes).unwrap();
        assert_eq!(decoded.width, reference.width);
        assert_eq!(decoded.height, reference.height);
        assert_eq!(decoded.color_channels, 3);
        assert_eq!(decoded.alpha, channels.alpha);
        let PixelData::U16(decoded_pixels) = &decoded.pixels else {
            panic!("expected 16-bit modular alpha decode");
        };
        let decoded_alpha = decoded_pixels
            .chunks_exact(4)
            .map(|pixel| pixel[3])
            .collect::<Vec<_>>();
        assert_alpha16_matches_reference(&decoded_alpha, &reference_alpha);

        let rgba16 = decode_rgba16(&encoded_bytes).unwrap();
        let rgba16_alpha = rgba16
            .pixels
            .chunks_exact(4)
            .map(|pixel| pixel[3])
            .collect::<Vec<_>>();
        assert_alpha16_matches_reference(&rgba16_alpha, &reference_alpha);

        let rgba8 = decode_rgba8(&encoded_bytes).unwrap();
        let rgba8_alpha = rgba8
            .pixels
            .chunks_exact(4)
            .map(|pixel| pixel[3])
            .collect::<Vec<_>>();
        let reference_alpha8 = reference_alpha
            .iter()
            .map(|&alpha| ((u32::from(alpha) * 255 + 32_767) / 65_535) as u8)
            .collect::<Vec<_>>();
        assert_alpha_matches_reference(&rgba8_alpha, &reference_alpha8);

        let roi = Rect {
            x: 5,
            y: 4,
            width: 11,
            height: 9,
        };
        let roi_rgba16 = Decoder::new()
            .roi(roi)
            .decode_rgba16(&encoded_bytes)
            .unwrap();
        let roi_alpha = roi_rgba16
            .pixels
            .chunks_exact(4)
            .map(|pixel| pixel[3])
            .collect::<Vec<_>>();
        assert_alpha16_matches_reference(
            &roi_alpha,
            &window_u16(&reference_alpha, reference.width, roi),
        );
    }

    #[test]
    fn decode_rgba16_upsamples_wider_subsampled_modular_alpha_when_available() {
        let (Some(cjxl), Some(djxl)) = (reference_cjxl(), reference_djxl()) else {
            eprintln!(
                "skipping wider 16-bit subsampled modular alpha comparison; reference tools are not built"
            );
            return;
        };

        for (resampling, shift, expected_min, expected_max, expected_checksum) in [
            (
                4u32,
                2i32,
                6_105u16,
                58_719u16,
                5_635_816_994_059_519_416u64,
            ),
            (
                8u32,
                3i32,
                17_477u16,
                48_637u16,
                11_909_690_253_215_219_231u64,
            ),
        ] {
            let input = unique_temp_path("jxl-rgba16-alpha-wide-subsampled-source", "pam");
            let encoded = unique_temp_path("jxl-rgba16-alpha-wide-subsampled", "jxl");
            let reference_output =
                unique_temp_path("jxl-rgba16-alpha-wide-subsampled-reference", "pam");
            let source = write_alpha_source_pam16(&input);

            let cjxl_output = Command::new(&cjxl)
                .arg(&input)
                .arg(&encoded)
                .args([
                    "-d",
                    "0",
                    "-m",
                    "1",
                    "--container=0",
                    "--ec_resampling",
                    &resampling.to_string(),
                    "--quiet",
                ])
                .output()
                .unwrap();
            let _ = std::fs::remove_file(&input);
            assert!(
                cjxl_output.status.success(),
                "reference cjxl failed for 16-bit resampling {resampling}: {}",
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
                "reference djxl failed for 16-bit resampling {resampling}: {}",
                String::from_utf8_lossy(&djxl_output.stderr)
            );

            let reference = std::fs::read(&reference_output).unwrap();
            let reference = parse_pam_rgba16(&reference);
            assert_eq!(reference.width, source.width);
            assert_eq!(reference.height, source.height);
            let reference_alpha = reference
                .samples
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>();
            let encoded_bytes = std::fs::read(&encoded).unwrap();
            let _ = std::fs::remove_file(&encoded);
            let _ = std::fs::remove_file(&reference_output);

            let channels = decode_channels(&encoded_bytes).unwrap();
            let alpha_channel = &channels.channels[3];
            assert_eq!(alpha_channel.width, reference.width.div_ceil(resampling));
            assert_eq!(alpha_channel.height, reference.height.div_ceil(resampling));
            assert_eq!(alpha_channel.hshift, shift);
            assert_eq!(alpha_channel.vshift, shift);
            assert_eq!(alpha_channel.bit_depth, 16);
            let ChannelData::U16(alpha) = &alpha_channel.samples else {
                panic!("expected 16-bit wider subsampled modular alpha channel");
            };
            assert_eq!(alpha.iter().copied().min(), Some(expected_min));
            assert_eq!(alpha.iter().copied().max(), Some(expected_max));
            let alpha_checksum =
                alpha
                    .iter()
                    .enumerate()
                    .fold(0u64, |checksum, (index, sample)| {
                        checksum
                            .wrapping_mul(1_099_511_628_211)
                            .wrapping_add(index as u64)
                            .rotate_left(11)
                            ^ u64::from(*sample)
                    });
            assert_eq!(alpha_checksum, expected_checksum);

            let rgba16 = decode_rgba16(&encoded_bytes).unwrap();
            let rgba16_alpha = rgba16
                .pixels
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>();
            assert_alpha16_matches_reference(&rgba16_alpha, &reference_alpha);

            let rgba8 = decode_rgba8(&encoded_bytes).unwrap();
            let rgba8_alpha = rgba8
                .pixels
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>();
            let reference_alpha8 = reference_alpha
                .iter()
                .map(|&alpha| ((u32::from(alpha) * 255 + 32_767) / 65_535) as u8)
                .collect::<Vec<_>>();
            assert_alpha_matches_reference(&rgba8_alpha, &reference_alpha8);

            let roi = Rect {
                x: 5,
                y: 4,
                width: 11,
                height: 9,
            };
            let roi_rgba16 = Decoder::new()
                .roi(roi)
                .decode_rgba16(&encoded_bytes)
                .unwrap();
            let roi_alpha = roi_rgba16
                .pixels
                .chunks_exact(4)
                .map(|pixel| pixel[3])
                .collect::<Vec<_>>();
            assert_alpha16_matches_reference(
                &roi_alpha,
                &window_u16(&reference_alpha, reference.width, roi),
            );
        }
    }

    #[test]
    fn decode_channels_exposes_subsampled_modular_depth_when_available() {
        let Some(cjxl) = reference_cjxl() else {
            eprintln!(
                "skipping subsampled modular depth raw-channel decode; reference cjxl is not built"
            );
            return;
        };

        let input = unique_temp_path("jxl-gray-modular-depth-subsampled-source", "pam");
        let encoded = unique_temp_path("jxl-gray-modular-depth-subsampled", "jxl");
        let source = write_gray_depth_source_pam(&input);

        let cjxl_output = Command::new(&cjxl)
            .arg(&input)
            .arg(&encoded)
            .args([
                "-d",
                "0",
                "-m",
                "1",
                "--container=0",
                "--ec_resampling",
                "2",
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
        let info = inspect(&encoded_bytes).unwrap();
        assert_eq!(info.metadata.color_encoding.color_space, ColorSpace::Gray);
        assert_eq!(info.metadata.num_color_channels(), 1);
        assert_eq!(info.metadata.extra_channels.len(), 1);
        assert_eq!(
            info.metadata.extra_channels[0].channel_type,
            ExtraChannelType::Depth
        );

        let channels = decode_channels(&encoded_bytes).unwrap();
        assert_eq!(channels.width, source.width);
        assert_eq!(channels.height, source.height);
        assert_eq!(channels.color_channels, 1);
        assert_eq!(channels.alpha, None);
        assert_eq!(channels.bit_depth, 8);
        assert_eq!(channels.channels.len(), 2);
        let gray_channel = &channels.channels[0];
        assert_eq!(gray_channel.width, source.width);
        assert_eq!(gray_channel.height, source.height);
        assert_eq!(gray_channel.hshift, 0);
        assert_eq!(gray_channel.vshift, 0);
        let depth_channel = &channels.channels[1];
        assert_eq!(depth_channel.width, source.width.div_ceil(2));
        assert_eq!(depth_channel.height, source.height.div_ceil(2));
        assert_eq!(depth_channel.hshift, 1);
        assert_eq!(depth_channel.vshift, 1);
        assert_eq!(depth_channel.bit_depth, 8);
        let ChannelData::U8(depth) = &depth_channel.samples else {
            panic!("expected 8-bit subsampled modular depth channel");
        };
        assert_eq!(depth.iter().copied().min(), Some(28));
        assert_eq!(depth.iter().copied().max(), Some(222));
        let depth_checksum = depth
            .iter()
            .enumerate()
            .fold(0u64, |checksum, (index, sample)| {
                checksum
                    .wrapping_mul(1_099_511_628_211)
                    .wrapping_add(index as u64)
                    .rotate_left(11)
                    ^ u64::from(*sample)
            });
        assert_eq!(depth_checksum, 6_577_634_186_647_311_850);

        let roi = Rect {
            x: 3,
            y: 4,
            width: 13,
            height: 9,
        };
        let roi_channels = Decoder::new()
            .roi(roi)
            .decode_channels(&encoded_bytes)
            .unwrap();
        assert_eq!(roi_channels.width, roi.width);
        assert_eq!(roi_channels.height, roi.height);
        assert_eq!(roi_channels.color_channels, channels.color_channels);
        assert_eq!(roi_channels.alpha, None);
        assert_eq!(roi_channels.channels.len(), channels.channels.len());
        let roi_depth_channel = &roi_channels.channels[1];
        assert_eq!(roi_depth_channel.width, 7);
        assert_eq!(roi_depth_channel.height, 5);
        assert_eq!(roi_depth_channel.hshift, 1);
        assert_eq!(roi_depth_channel.vshift, 1);
        let ChannelData::U8(roi_depth) = &roi_depth_channel.samples else {
            panic!("expected 8-bit subsampled modular ROI depth channel");
        };
        let shifted_roi = Rect {
            x: 1,
            y: 2,
            width: 7,
            height: 5,
        };
        assert_eq!(
            roi_depth,
            &window_u8(depth, depth_channel.width, shifted_roi)
        );

        let decoded = decode(&encoded_bytes).unwrap();
        assert_eq!(decoded.width, source.width);
        assert_eq!(decoded.height, source.height);
        assert_eq!(decoded.color_channels, 1);
        assert_eq!(decoded.alpha, None);
        let PixelData::U8(gray) = &decoded.pixels else {
            panic!("expected 8-bit grayscale modular depth image");
        };
        let ChannelData::U8(raw_gray) = &gray_channel.samples else {
            panic!("expected 8-bit grayscale modular color channel");
        };
        assert_eq!(gray, raw_gray);

        let rgba = decode_rgba8(&encoded_bytes).unwrap();
        for (pixel, &gray) in rgba.pixels.chunks_exact(4).zip(raw_gray) {
            assert_eq!(pixel, &[gray, gray, gray, 255]);
        }
    }

    #[test]
    fn decode_channels_exposes_subsampled_modular_depth16_when_available() {
        let Some(cjxl) = reference_cjxl() else {
            eprintln!(
                "skipping 16-bit subsampled modular depth raw-channel decode; reference cjxl is not built"
            );
            return;
        };

        let input = unique_temp_path("jxl-gray-modular-depth16-subsampled-source", "pam");
        let encoded = unique_temp_path("jxl-gray-modular-depth16-subsampled", "jxl");
        let source = write_gray_depth_source_pam16(&input);
        assert_eq!(
            source.depth.len(),
            source.width as usize * source.height as usize
        );

        let cjxl_output = Command::new(&cjxl)
            .arg(&input)
            .arg(&encoded)
            .args([
                "-d",
                "0",
                "-m",
                "1",
                "--container=0",
                "--ec_resampling",
                "2",
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
        let info = inspect(&encoded_bytes).unwrap();
        assert_eq!(info.metadata.color_encoding.color_space, ColorSpace::Gray);
        assert_eq!(info.metadata.num_color_channels(), 1);
        assert_eq!(info.metadata.extra_channels.len(), 1);
        assert_eq!(
            info.metadata.extra_channels[0].channel_type,
            ExtraChannelType::Depth
        );

        let channels = decode_channels(&encoded_bytes).unwrap();
        assert_eq!(channels.width, source.width);
        assert_eq!(channels.height, source.height);
        assert_eq!(channels.color_channels, 1);
        assert_eq!(channels.alpha, None);
        assert_eq!(channels.bit_depth, 16);
        assert_eq!(channels.channels.len(), 2);
        let gray_channel = &channels.channels[0];
        assert_eq!(gray_channel.width, source.width);
        assert_eq!(gray_channel.height, source.height);
        assert_eq!(gray_channel.hshift, 0);
        assert_eq!(gray_channel.vshift, 0);
        assert_eq!(gray_channel.bit_depth, 16);
        let depth_channel = &channels.channels[1];
        assert_eq!(depth_channel.width, source.width.div_ceil(2));
        assert_eq!(depth_channel.height, source.height.div_ceil(2));
        assert_eq!(depth_channel.hshift, 1);
        assert_eq!(depth_channel.vshift, 1);
        assert_eq!(depth_channel.bit_depth, 16);
        let ChannelData::U16(depth) = &depth_channel.samples else {
            panic!("expected 16-bit subsampled modular depth channel");
        };
        assert_eq!(depth.iter().copied().min(), Some(1284));
        assert_eq!(depth.iter().copied().max(), Some(63735));
        let depth_checksum = depth
            .iter()
            .enumerate()
            .fold(0u64, |checksum, (index, sample)| {
                checksum
                    .wrapping_mul(1_099_511_628_211)
                    .wrapping_add(index as u64)
                    .rotate_left(11)
                    ^ u64::from(*sample)
            });
        assert_eq!(depth_checksum, 2_995_956_734_774_139_113);

        let roi = Rect {
            x: 3,
            y: 4,
            width: 13,
            height: 9,
        };
        let roi_channels = Decoder::new()
            .roi(roi)
            .decode_channels(&encoded_bytes)
            .unwrap();
        assert_eq!(roi_channels.width, roi.width);
        assert_eq!(roi_channels.height, roi.height);
        assert_eq!(roi_channels.color_channels, channels.color_channels);
        assert_eq!(roi_channels.alpha, None);
        assert_eq!(roi_channels.channels.len(), channels.channels.len());
        let roi_depth_channel = &roi_channels.channels[1];
        assert_eq!(roi_depth_channel.width, 7);
        assert_eq!(roi_depth_channel.height, 5);
        assert_eq!(roi_depth_channel.hshift, 1);
        assert_eq!(roi_depth_channel.vshift, 1);
        assert_eq!(roi_depth_channel.bit_depth, 16);
        let ChannelData::U16(roi_depth) = &roi_depth_channel.samples else {
            panic!("expected 16-bit subsampled modular ROI depth channel");
        };
        let shifted_roi = Rect {
            x: 1,
            y: 2,
            width: 7,
            height: 5,
        };
        assert_eq!(
            roi_depth,
            &window_u16(depth, depth_channel.width, shifted_roi)
        );

        let decoded = decode(&encoded_bytes).unwrap();
        assert_eq!(decoded.width, source.width);
        assert_eq!(decoded.height, source.height);
        assert_eq!(decoded.color_channels, 1);
        assert_eq!(decoded.alpha, None);
        assert_eq!(decoded.bit_depth, 16);
        let PixelData::U16(gray) = &decoded.pixels else {
            panic!("expected 16-bit grayscale modular depth image");
        };
        let ChannelData::U16(raw_gray) = &gray_channel.samples else {
            panic!("expected 16-bit grayscale modular color channel");
        };
        assert_eq!(gray, raw_gray);

        let rgba16 = decode_rgba16(&encoded_bytes).unwrap();
        for (pixel, &gray) in rgba16.pixels.chunks_exact(4).zip(raw_gray) {
            assert_eq!(pixel, &[gray, gray, gray, u16::MAX]);
        }
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
    fn converts_srgb_samples_to_linear() {
        assert_eq!(srgb_sample_to_linear(-1.0), 0.0);
        assert_eq!(srgb_sample_to_linear(0.0), 0.0);
        assert!((srgb_sample_to_linear(10.0 / 255.0) - 0.003_035_27).abs() < 1e-8);
        assert!((srgb_sample_to_linear(128.0 / 255.0) - 0.215_860_53).abs() < 1e-8);
        assert_eq!(srgb_sample_to_linear(1.0), 1.0);
        assert_eq!(srgb_sample_to_linear(2.0), 1.0);
    }

    #[test]
    fn linear_rgb_from_decoded_gray_expands_to_rgb() {
        let channels = DecodedChannels {
            width: 2,
            height: 1,
            color_channels: 1,
            alpha: None,
            bit_depth: 8,
            channels: vec![DecodedChannel {
                width: 2,
                height: 1,
                hshift: 0,
                vshift: 0,
                bit_depth: 8,
                samples: ChannelData::U8(vec![0, 255]),
            }],
        };
        let color_encoding = ColorEncoding {
            color_space: ColorSpace::Gray,
            ..ColorEncoding::default()
        };

        assert_eq!(
            linear_rgb_from_decoded_channels(&channels, &color_encoding, 1).unwrap(),
            LinearRgbImage {
                width: 2,
                height: 1,
                pixels: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            }
        );
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
            decode_buffered_channels(channels, Some(3)).unwrap(),
            DecodedImage {
                width: 2,
                height: 1,
                color_channels: 3,
                alpha: Some(AlphaInfo {
                    bit_depth: 8,
                    premultiplied: true,
                }),
                bit_depth: 16,
                pixels: PixelData::U16(vec![16_448, 32_896, 65_535, 32_896, 0, 0, 257, 0]),
            }
        );
    }

    #[test]
    fn shifted_raw_channel_roi_crops_in_channel_space() {
        let samples = (0u8..12).collect::<Vec<_>>();
        let channel = DecodedChannel {
            width: 4,
            height: 3,
            hshift: 1,
            vshift: 1,
            bit_depth: 8,
            samples: ChannelData::U8(samples.clone()),
        };
        let roi = jxl_codec::ImageRegion {
            x: 3,
            y: 1,
            width: 4,
            height: 4,
        };
        let shifted = shifted_decode_region(roi, 1, 1).unwrap();
        assert_eq!(
            shifted,
            jxl_codec::ImageRegion {
                x: 1,
                y: 0,
                width: 3,
                height: 3,
            }
        );

        let cropped = crop_decoded_channel(channel, roi).unwrap();
        assert_eq!(cropped.width, 3);
        assert_eq!(cropped.height, 3);
        assert_eq!(cropped.hshift, 1);
        assert_eq!(cropped.vshift, 1);
        assert_eq!(cropped.bit_depth, 8);
        assert_eq!(
            cropped.samples,
            ChannelData::U8(vec![1, 2, 3, 5, 6, 7, 9, 10, 11])
        );
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct PpmRgb {
        width: u32,
        height: u32,
        samples: Vec<u16>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct PgmGray {
        width: u32,
        height: u32,
        samples: Vec<u16>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AlphaDepthPam {
        width: u32,
        height: u32,
        rgba: Vec<u8>,
        alpha: Vec<u8>,
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
    struct GrayDepthPam {
        width: u32,
        height: u32,
        depth: Vec<u8>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct GrayDepthPam16 {
        width: u32,
        height: u32,
        depth: Vec<u16>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AlphaPam16 {
        width: u32,
        height: u32,
        alpha: Vec<u16>,
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

    fn parse_pgm_gray(bytes: &[u8]) -> PgmGray {
        let (magic, offset) = netpbm_token(bytes, 0);
        assert_eq!(magic, b"P5");
        let (width, offset) = netpbm_token(bytes, offset);
        let (height, offset) = netpbm_token(bytes, offset);
        let (maxval, mut offset) = netpbm_token(bytes, offset);
        let maxval = parse_ascii_u32(maxval);
        assert!(matches!(maxval, 255 | 65535));
        assert!(
            offset < bytes.len() && bytes[offset].is_ascii_whitespace(),
            "PGM header was not followed by binary sample data"
        );
        offset += 1;

        let width = parse_ascii_u32(width);
        let height = parse_ascii_u32(height);
        let bytes_per_sample = if maxval > 255 { 2 } else { 1 };
        let expected_bytes = width as usize * height as usize * bytes_per_sample;
        let data = &bytes[offset..];
        assert_eq!(data.len(), expected_bytes);
        let samples = if bytes_per_sample == 2 {
            data.chunks_exact(2)
                .map(|chunk| u16::from_be_bytes([chunk[0], chunk[1]]))
                .collect()
        } else {
            data.iter().copied().map(u16::from).collect()
        };
        PgmGray {
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

    fn parse_pam_rgba16(bytes: &[u8]) -> PpmRgb {
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
        assert_eq!(maxval, Some(65535));
        assert_eq!(tupltype, Some("RGB_ALPHA"));
        let width = width.unwrap();
        let height = height.unwrap();
        let data = &bytes[header_end..];
        assert_eq!(data.len(), width as usize * height as usize * 8);
        PpmRgb {
            width,
            height,
            samples: data
                .chunks_exact(2)
                .map(|sample| u16::from_be_bytes([sample[0], sample[1]]))
                .collect(),
        }
    }

    fn parse_pam_gray_alpha(bytes: &[u8]) -> GrayAlphaPam {
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
        assert_eq!(depth, Some(2));
        assert_eq!(maxval, Some(255));
        assert_eq!(tupltype, Some("GRAYSCALE_ALPHA"));
        let width = width.unwrap();
        let height = height.unwrap();
        let data = &bytes[header_end..];
        assert_eq!(data.len(), width as usize * height as usize * 2);

        let mut gray = Vec::with_capacity(width as usize * height as usize);
        let mut alpha = Vec::with_capacity(width as usize * height as usize);
        let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
        for pixel in data.chunks_exact(2) {
            gray.push(pixel[0]);
            alpha.push(pixel[1]);
            rgba.extend_from_slice(&[pixel[0], pixel[0], pixel[0], pixel[1]]);
        }

        GrayAlphaPam {
            width,
            height,
            gray_alpha: data.to_vec(),
            gray,
            alpha,
            rgba,
        }
    }

    fn parse_pam_gray_alpha16(bytes: &[u8]) -> GrayAlphaPam16 {
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
        assert_eq!(depth, Some(2));
        assert_eq!(maxval, Some(65535));
        assert_eq!(tupltype, Some("GRAYSCALE_ALPHA"));
        let width = width.unwrap();
        let height = height.unwrap();
        let data = &bytes[header_end..];
        assert_eq!(data.len(), width as usize * height as usize * 4);

        let mut gray_alpha = Vec::with_capacity(width as usize * height as usize * 2);
        let mut gray = Vec::with_capacity(width as usize * height as usize);
        let mut alpha = Vec::with_capacity(width as usize * height as usize);
        let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
        for pixel in data.chunks_exact(4) {
            let gray_sample = u16::from_be_bytes([pixel[0], pixel[1]]);
            let alpha_sample = u16::from_be_bytes([pixel[2], pixel[3]]);
            gray_alpha.extend_from_slice(&[gray_sample, alpha_sample]);
            gray.push(gray_sample);
            alpha.push(alpha_sample);
            rgba.extend_from_slice(&[gray_sample, gray_sample, gray_sample, alpha_sample]);
        }

        GrayAlphaPam16 {
            width,
            height,
            gray_alpha,
            gray,
            alpha,
            rgba,
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

    fn decoded_u8_channel(width: u32, height: u32, samples: &[u8]) -> DecodedChannel {
        DecodedChannel {
            width,
            height,
            hshift: 0,
            vshift: 0,
            bit_depth: 8,
            samples: ChannelData::U8(samples.to_vec()),
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

    fn gray8_oracle_metrics(
        decoded: &DecodedImage,
        reference: &PgmGray,
        anchor_indices: &[usize],
    ) -> Srgb8OracleMetrics {
        assert_eq!(decoded.width, reference.width);
        assert_eq!(decoded.height, reference.height);
        assert_eq!(decoded.color_channels, 1);
        assert_eq!(decoded.bit_depth, 8);
        let PixelData::U8(samples) = &decoded.pixels else {
            panic!("expected public grayscale oracle comparison to use 8-bit output");
        };
        gray8_samples_oracle_metrics(
            samples,
            decoded.width,
            decoded.height,
            reference,
            anchor_indices,
        )
    }

    fn gray8_samples_oracle_metrics(
        samples: &[u8],
        width: u32,
        height: u32,
        reference: &PgmGray,
        anchor_indices: &[usize],
    ) -> Srgb8OracleMetrics {
        assert_eq!(width, reference.width);
        assert_eq!(height, reference.height);
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

    fn assert_roi_matches_full_linear_rgb(
        roi_image: &LinearRgbImage,
        full: &LinearRgbImage,
        roi: Rect,
    ) {
        assert_eq!(roi_image.width, roi.width);
        assert_eq!(roi_image.height, roi.height);
        assert_eq!(
            roi_image.pixels,
            window_interleaved_f32(&full.pixels, full.width, 3, roi)
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

    fn assert_alpha_matches_reference(actual: &[u8], expected: &[u8]) {
        if actual == expected {
            return;
        }
        let first_mismatch = actual
            .iter()
            .zip(expected)
            .position(|(actual, expected)| actual != expected);
        let max_abs_error = actual
            .iter()
            .zip(expected)
            .map(|(&actual, &expected)| actual.abs_diff(expected))
            .max()
            .unwrap_or(0);
        if actual.len() == expected.len() && max_abs_error <= 1 {
            return;
        }
        panic!(
            "alpha mismatch: len actual={} expected={}, first_mismatch={:?}, max_abs_error={}, actual_prefix={:?}, expected_prefix={:?}",
            actual.len(),
            expected.len(),
            first_mismatch,
            max_abs_error,
            &actual[..actual.len().min(16)],
            &expected[..expected.len().min(16)]
        );
    }

    fn assert_alpha16_matches_reference(actual: &[u16], expected: &[u16]) {
        if actual == expected {
            return;
        }
        let first_mismatch = actual
            .iter()
            .zip(expected)
            .position(|(actual, expected)| actual != expected);
        let max_abs_error = actual
            .iter()
            .zip(expected)
            .map(|(&actual, &expected)| actual.abs_diff(expected))
            .max()
            .unwrap_or(0);
        if actual.len() == expected.len() && max_abs_error <= 257 {
            return;
        }
        panic!(
            "alpha16 mismatch: len actual={} expected={}, first_mismatch={:?}, max_abs_error={}, actual_prefix={:?}, expected_prefix={:?}",
            actual.len(),
            expected.len(),
            first_mismatch,
            max_abs_error,
            &actual[..actual.len().min(16)],
            &expected[..expected.len().min(16)]
        );
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

    fn rgba8_checksum(samples: &[u8]) -> u64 {
        samples
            .iter()
            .enumerate()
            .fold(0xcbf2_9ce4_8422_2325u64, |checksum, (index, sample)| {
                (checksum ^ ((index as u64) << 8) ^ u64::from(*sample))
                    .wrapping_mul(0x0000_0100_0000_01b3)
            })
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

    fn window_interleaved_f32(samples: &[f32], width: u32, channels: usize, roi: Rect) -> Vec<f32> {
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

    fn write_resampled_vardct_source_ppm(path: &Path) {
        let width = 96u32;
        let height = 64u32;
        let mut bytes = format!("P6\n{width} {height}\n255\n").into_bytes();
        for y in 0..height {
            for x in 0..width {
                bytes.push((x * 255 / (width - 1)) as u8);
                bytes.push((y * 255 / (height - 1)) as u8);
                bytes.push(((x + y) * 255 / (width + height - 2)) as u8);
            }
        }
        std::fs::write(path, bytes).unwrap();
    }

    fn write_multigroup_vardct_source_ppm(path: &Path) {
        let width = 1024u32;
        let height = 512u32;
        let mut bytes = format!("P6\n{width} {height}\n255\n").into_bytes();
        for y in 0..height {
            for x in 0..width {
                bytes.push((x * 255 / (width - 1)) as u8);
                bytes.push((y * 255 / (height - 1)) as u8);
                bytes.push(((x + y) * 255 / (width + height - 2)) as u8);
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

    fn write_alpha_source_pam16(path: &Path) -> AlphaPam16 {
        let width = 19u32;
        let height = 15u32;
        let mut bytes = format!(
            "P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH 4\nMAXVAL 65535\nTUPLTYPE RGB_ALPHA\nENDHDR\n"
        )
        .into_bytes();
        let mut alpha = Vec::with_capacity(width as usize * height as usize);
        for y in 0..height {
            for x in 0..width {
                let samples = [
                    ((x * 2971 + y * 359 + 11) & 0xffff) as u16,
                    ((x * 811 + y * 2371 + 37) & 0xffff) as u16,
                    ((x * 1237 + y * 1597 + 91) & 0xffff) as u16,
                    ((x * 1723 + y * 3253 + 61) & 0xffff) as u16,
                ];
                for sample in samples {
                    bytes.extend_from_slice(&sample.to_be_bytes());
                }
                alpha.push(samples[3]);
            }
        }
        std::fs::write(path, bytes).unwrap();
        AlphaPam16 {
            width,
            height,
            alpha,
        }
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

    fn write_vardct_alpha_source_pam16(path: &Path) -> Vec<u16> {
        let width = 320u32;
        let height = 192u32;
        let mut alpha = Vec::with_capacity(width as usize * height as usize);
        let mut bytes = format!(
            "P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH 4\nMAXVAL 65535\nTUPLTYPE RGB_ALPHA\nENDHDR\n"
        )
        .into_bytes();
        for y in 0..height {
            for x in 0..width {
                let checker = (((x / 16) ^ (y / 16)) & 1) * 12_336;
                let red = ((x * 65_535 / (width - 1)) ^ checker) as u16;
                let green = ((y * 65_535 / (height - 1)) ^ checker) as u16;
                let blue = (((x + y) * 65_535 / (width + height - 2)) ^ checker) as u16;
                let alpha_sample = ((x * 1733 + y * 2411 + 43) & 0xffff) as u16;
                bytes.extend_from_slice(&red.to_be_bytes());
                bytes.extend_from_slice(&green.to_be_bytes());
                bytes.extend_from_slice(&blue.to_be_bytes());
                bytes.extend_from_slice(&alpha_sample.to_be_bytes());
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

    fn write_vardct_alpha_depth_source_pam(path: &Path) -> AlphaDepthPam {
        let width = 320u32;
        let height = 192u32;
        let mut bytes = format!(
            "P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH 5\nMAXVAL 255\nTUPLTYPE RGB\nTUPLTYPE Depth\nTUPLTYPE Alpha\nENDHDR\n"
        )
        .into_bytes();
        let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
        let mut alpha = Vec::with_capacity(width as usize * height as usize);
        let mut depth = Vec::with_capacity(width as usize * height as usize);
        for y in 0..height {
            for x in 0..width {
                let checker = (((x / 16) ^ (y / 16)) & 1) * 48;
                let red = ((x * 255 / (width - 1)) ^ checker) as u8;
                let green = ((y * 255 / (height - 1)) ^ checker) as u8;
                let blue = (((x + y) * 255 / (width + height - 2)) ^ checker) as u8;
                let depth_sample = ((x * 37 + y * 41 + 73) & 0xff) as u8;
                let alpha_sample = ((x * 29 + y * 31 + 43) & 0xff) as u8;
                bytes.extend_from_slice(&[red, green, blue, depth_sample, alpha_sample]);
                rgba.extend_from_slice(&[red, green, blue, alpha_sample]);
                alpha.push(alpha_sample);
                depth.push(depth_sample);
            }
        }
        std::fs::write(path, bytes).unwrap();
        AlphaDepthPam {
            width,
            height,
            rgba,
            alpha,
            depth,
        }
    }

    fn write_rgb_depth_source_pam(path: &Path) -> Vec<u8> {
        let width = 35u32;
        let height = 21u32;
        let mut depth = Vec::with_capacity(width as usize * height as usize);
        let mut bytes = format!(
            "P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB\nTUPLTYPE Depth\nENDHDR\n"
        )
        .into_bytes();
        for y in 0..height {
            for x in 0..width {
                bytes.push(((x * 11 + y * 3 + 17) & 0xff) as u8);
                bytes.push(((x * 7 + y * 13 + 29) & 0xff) as u8);
                bytes.push(((x * 19 + y * 5 + 43) & 0xff) as u8);
                let depth_sample = ((x * 37 + y * 41 + 73) & 0xff) as u8;
                bytes.push(depth_sample);
                depth.push(depth_sample);
            }
        }
        std::fs::write(path, bytes).unwrap();
        depth
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

    fn write_gray_depth_source_pam(path: &Path) -> GrayDepthPam {
        let width = 23u32;
        let height = 19u32;
        let mut bytes = format!(
            "P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH 2\nMAXVAL 255\nTUPLTYPE GRAYSCALE\nTUPLTYPE Depth\nENDHDR\n"
        )
        .into_bytes();
        let mut depth = Vec::with_capacity(width as usize * height as usize);
        for y in 0..height {
            for x in 0..width {
                let gray_sample = ((x * 17 + y * 11 + 9) & 0xff) as u8;
                let depth_sample = ((x * 37 + y * 41 + 73) & 0xff) as u8;
                bytes.push(gray_sample);
                bytes.push(depth_sample);
                depth.push(depth_sample);
            }
        }
        std::fs::write(path, bytes).unwrap();
        GrayDepthPam {
            width,
            height,
            depth,
        }
    }

    fn write_gray_depth_source_pam16(path: &Path) -> GrayDepthPam16 {
        let width = 23u32;
        let height = 19u32;
        let mut bytes = format!(
            "P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH 2\nMAXVAL 65535\nTUPLTYPE GRAYSCALE\nTUPLTYPE Depth\nENDHDR\n"
        )
        .into_bytes();
        let mut depth = Vec::with_capacity(width as usize * height as usize);
        for y in 0..height {
            for x in 0..width {
                let gray_sample = ((x * 1543 + y * 811 + 9) & 0xffff) as u16;
                let depth_sample = ((x * 2017 + y * 1543 + 73) & 0xffff) as u16;
                bytes.extend_from_slice(&gray_sample.to_be_bytes());
                bytes.extend_from_slice(&depth_sample.to_be_bytes());
                depth.push(depth_sample);
            }
        }
        std::fs::write(path, bytes).unwrap();
        GrayDepthPam16 {
            width,
            height,
            depth,
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
        let alpha = rgba.chunks_exact(4).map(|pixel| pixel[3]).collect();
        AlphaDepthPam {
            width,
            height,
            rgba,
            alpha,
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
