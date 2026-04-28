//! Public Rust-native JPEG XL API.
//!
//! The API is intentionally small while the decoder is being built out. It
//! exposes stable metadata inspection now and leaves room for future streaming
//! decode, region decode, and pixel-output builders without committing to a
//! C-style event API.

pub use jxl_codec::{
    BasicInfo, BitDepth, BlendMode, BlendingInfo, BoxRecord, ColorEncoding, ColorSpace, Container,
    CustomTransformData, Error, ExtraChannelInfo, ExtraChannelType, FileFormat, FrameEncoding,
    FrameGroupLayout, FrameHeader, FrameType, ImageMetadata, OpsinInverseMatrix, Orientation,
    Primaries, RenderingIntent, Result, ToneMapping, TransferFunction, WhitePoint,
};

#[derive(Debug, Clone, PartialEq)]
pub struct ImageInfo {
    pub format: FileFormat,
    pub width: u32,
    pub height: u32,
    pub basic_info: BasicInfo,
    pub metadata: ImageMetadata,
    pub transform_data: CustomTransformData,
    pub first_frame: Option<FrameHeader>,
    pub boxes: Vec<BoxRecord>,
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
        first_frame: codestream.first_frame,
        boxes: extracted
            .container
            .map(|container| container.boxes)
            .unwrap_or_default(),
    })
}
