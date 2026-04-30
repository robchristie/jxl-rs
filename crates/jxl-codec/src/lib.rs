//! Core JPEG XL codestream and container primitives.
//!
//! This crate intentionally starts with the parsing layers that every decoder
//! path shares: format detection, ISO BMFF-style JPEG XL boxes, codestream
//! extraction, and the compact size header. Pixel reconstruction is layered on
//! top of these pieces as modular and VarDCT support lands.

pub mod bitstream;
pub mod codestream;
pub mod container;
pub mod decode;
pub mod error;
pub mod frame;
pub mod frame_data;
pub mod metadata;
pub mod modular;
pub mod transform;
pub mod vardct;

pub(crate) mod entropy;
pub(crate) mod icc;

pub use codestream::{
    BasicInfo, Codestream, SizeHeader, parse_codestream, parse_codestream_with_config,
};
pub use container::{
    BoxRecord, Container, ContainerBox, ExtractedCodestream, FileFormat, Signature, parse_file,
    parse_file_with_config,
};
pub use decode::{DecodeConfig, ImageRegion, ModularGroupExecution};
pub use error::{Error, Result};
pub use frame::{
    AnimationFrame as FrameAnimation, BlendMode, BlendingInfo, ColorTransform, FrameEncoding,
    FrameGroupLayout, FrameHeader, FrameOrigin, FrameSize, FrameType, LoopFilter, Passes,
    YCbCrChromaSubsampling,
};
pub use frame_data::{
    FrameData, FrameSection, FrameSectionKind, FrameToc, TocEntry, section_payload,
    section_payload_range,
};
pub use metadata::{
    AnimationHeader, BitDepth, ColorEncoding, ColorSpace, ExtraChannelInfo, ExtraChannelType,
    ImageMetadata, Orientation, PreviewHeader, Primaries, RenderingIntent, ToneMapping,
    TransferFunction, WhitePoint,
};
pub use modular::{
    MaTree, MaTreeNode, ModularChannel, ModularChannelPlan, ModularDecodedChannel,
    ModularDecodedGroup, ModularFrameMetadata, ModularGlobalSection, ModularGroupChannelPlan,
    ModularGroupHeader, ModularImage, ModularImageChannel, ModularPredictor, ModularResiduals,
    ModularSectionMetadata, ModularTransform, ModularTreeMetadata, SqueezeParams, TransformId,
    WeightedPredictorHeader,
};
pub use transform::{CustomTransformData, OpsinInverseMatrix};
pub use vardct::{
    VarDctAcBaseDequantizedChannelGrid, VarDctAcBaseDequantizedGrid, VarDctAcBlockSummary,
    VarDctAcChannelCoefficientGrid, VarDctAcChannelCoefficientSummary, VarDctAcChannelTrace,
    VarDctAcCoefficientEvent, VarDctAcCoefficientGrid, VarDctAcCoefficientProbe,
    VarDctAcCoefficientSummary, VarDctAcDequantizedChannelGrid, VarDctAcDequantizedGrid,
    VarDctAcGlobalMetadata, VarDctAcGlobalPassMetadata, VarDctAcGroupCursorMetadata,
    VarDctAcGroupMetadata, VarDctAcSpatialChannelGrid, VarDctAcSpatialGrid,
    VarDctAnsHistogramProbe, VarDctAnsHistogramProbeKind, VarDctAnsHistogramProbeStage,
    VarDctBlockContextMapMetadata, VarDctCoeffOrderMetadata, VarDctColorCorrelationMetadata,
    VarDctContextMapProbe, VarDctContextMapProbeKind, VarDctContextMapProbeStage,
    VarDctDcDequantMetadata, VarDctDcGroupCursorMetadata, VarDctDcGroupMetadata,
    VarDctDcGroupPayloadMetadata, VarDctDecodePlan, VarDctFrameMetadata,
    VarDctGlobalCursorMetadata, VarDctGlobalMetadata, VarDctGroupMetadata,
    VarDctGroupPayloadMetadata, VarDctGroupSectionMetadata, VarDctHistogramProbeStage,
    VarDctPassGroupPayloadMetadata, VarDctPassGroupSectionMetadata, VarDctQuantizerMetadata,
    VarDctSectionMetadata, VarDctSectionPayloadMetadata,
};
