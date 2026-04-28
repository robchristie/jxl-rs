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
pub mod frame;
pub mod frame_data;
pub mod metadata;
pub mod modular;
pub mod transform;

pub(crate) mod entropy;
pub(crate) mod icc;

pub use codestream::{BasicInfo, Codestream, SizeHeader, parse_codestream};
pub use container::{
    BoxRecord, Container, ContainerBox, ExtractedCodestream, FileFormat, Signature, parse_file,
};
pub use error::{Error, Result};
pub use frame::{
    AnimationFrame as FrameAnimation, BlendMode, BlendingInfo, ColorTransform, FrameEncoding,
    FrameGroupLayout, FrameHeader, FrameOrigin, FrameSize, FrameType, LoopFilter, Passes,
    YCbCrChromaSubsampling,
};
pub use frame_data::{FrameData, FrameSection, FrameSectionKind, FrameToc, TocEntry};
pub use metadata::{
    AnimationHeader, BitDepth, ColorEncoding, ColorSpace, ExtraChannelInfo, ExtraChannelType,
    ImageMetadata, Orientation, PreviewHeader, Primaries, RenderingIntent, ToneMapping,
    TransferFunction, WhitePoint,
};
pub use modular::{
    MaTree, MaTreeNode, ModularChannel, ModularChannelPlan, ModularFrameMetadata,
    ModularGlobalSection, ModularGroupChannelPlan, ModularGroupHeader, ModularPredictor,
    ModularSectionMetadata, ModularTransform, ModularTreeMetadata, SqueezeParams, TransformId,
    WeightedPredictorHeader,
};
pub use transform::{CustomTransformData, OpsinInverseMatrix};
