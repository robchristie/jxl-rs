//! Core JPEG XL codestream and container primitives.
//!
//! This crate intentionally starts with the parsing layers that every decoder
//! path shares: format detection, ISO BMFF-style JPEG XL boxes, codestream
//! extraction, and the compact size header. Pixel reconstruction is layered on
//! top of these pieces as modular and VarDCT support lands.

pub mod bitstream;
pub mod codestream;
pub mod container;
pub mod error;
pub mod metadata;

pub use codestream::{BasicInfo, Codestream, SizeHeader, parse_codestream};
pub use container::{
    BoxRecord, Container, ContainerBox, ExtractedCodestream, FileFormat, Signature, parse_file,
};
pub use error::{Error, Result};
pub use metadata::{
    AnimationHeader, BitDepth, ColorEncoding, ColorSpace, ExtraChannelInfo, ExtraChannelType,
    ImageMetadata, Orientation, PreviewHeader, Primaries, RenderingIntent, ToneMapping,
    TransferFunction, WhitePoint,
};
