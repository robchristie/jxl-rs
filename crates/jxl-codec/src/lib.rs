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
    DequantizedSplineMetadata, FrameFeatureMetadata, MaTree, MaTreeNode, ModularChannel,
    ModularChannelPlan, ModularDecodedChannel, ModularDecodedGroup, ModularFrameMetadata,
    ModularGlobalSection, ModularGroupChannelPlan, ModularGroupHeader, ModularImage,
    ModularImageChannel, ModularPredictor, ModularResiduals, ModularSectionMetadata,
    ModularTransform, ModularTreeMetadata, NoiseFrameMetadata, QuantizedSplineMetadata,
    SplineFloatPoint, SplineFrameMetadata, SplinePoint, SplineRenderPlan, SplineSegmentMetadata,
    SqueezeParams, TransformId, WeightedPredictorHeader, render_noise_into_xyb_image,
    render_splines_into_xyb_image,
};
pub use transform::{CustomTransformData, OpsinInverseMatrix};
pub use vardct::{
    VarDctAcBaseDequantizedChannelGrid, VarDctAcBaseDequantizedGrid, VarDctAcBlockSummary,
    VarDctAcChannelCoefficientGrid, VarDctAcChannelCoefficientSummary, VarDctAcChannelTrace,
    VarDctAcCoefficientEvent, VarDctAcCoefficientGrid, VarDctAcCoefficientProbe,
    VarDctAcCoefficientSummary, VarDctAcDequantizedChannelGrid, VarDctAcDequantizedGrid,
    VarDctAcGlobalMetadata, VarDctAcGlobalPassMetadata, VarDctAcGroupCursorMetadata,
    VarDctAcGroupMetadata, VarDctAcQuantMatrices, VarDctAcQuantMode, VarDctAcQuantTable,
    VarDctAcSpatialChannelGrid, VarDctAcSpatialGrid, VarDctAnsHistogramProbe,
    VarDctAnsHistogramProbeKind, VarDctAnsHistogramProbeStage, VarDctBlockContextMapMetadata,
    VarDctChannelRangeDiagnostics, VarDctCoeffOrderMetadata, VarDctColorCorrelationMetadata,
    VarDctContextMapProbe, VarDctContextMapProbeKind, VarDctContextMapProbeStage,
    VarDctDcCoefficientDiagnostics, VarDctDcDequantMetadata, VarDctDcGroupCursorMetadata,
    VarDctDcGroupMetadata, VarDctDcGroupPayloadMetadata, VarDctDcRawChannelDiagnostics,
    VarDctDcScaledChannelDiagnostics, VarDctDecodePlan, VarDctEpfMetadata, VarDctFrameMetadata,
    VarDctGlobalCursorMetadata, VarDctGlobalMetadata, VarDctGroupMetadata,
    VarDctGroupPayloadMetadata, VarDctGroupSectionMetadata, VarDctHistogramProbeStage,
    VarDctOpsinParams, VarDctPassGroupPayloadMetadata, VarDctPassGroupSectionMetadata,
    VarDctQuantizerMetadata, VarDctRgbImage, VarDctSectionMetadata, VarDctSectionPayloadMetadata,
    VarDctSrgb8Image, VarDctSrgb16Image, VarDctXybImage, VarDctXybInverseVariant,
    VarDctXybInverseVariantDiagnostics, VarDctXybRgbDiagnostics, assemble_vardct_dc_srgb8_image,
    assemble_vardct_dc_srgb8_image_with_multiplier, assemble_vardct_dc_xyb_image,
    assemble_vardct_linear_rgb_image, assemble_vardct_rgb_srgb8_image,
    assemble_vardct_rgb_srgb8_image_for_pass, assemble_vardct_rgb_srgb16_image,
    assemble_vardct_rgb_srgb16_image_for_pass, assemble_vardct_srgb8_image,
    assemble_vardct_srgb8_image_for_pass, assemble_vardct_srgb16_image,
    assemble_vardct_srgb16_image_for_pass, assemble_vardct_xyb_image,
    assemble_vardct_xyb_image_for_pass, assemble_vardct_ycbcr_srgb8_image,
    assemble_vardct_ycbcr_srgb8_image_for_pass, assemble_vardct_ycbcr_srgb16_image,
    assemble_vardct_ycbcr_srgb16_image_for_pass, vardct_dc_coefficient_diagnostics,
    vardct_xyb_inverse_variant_diagnostics, vardct_xyb_rgb_diagnostics, xyb_image_to_linear_rgb,
    xyb_image_to_srgb8, xyb_image_to_srgb8_with_variant, xyb_image_to_srgb16,
    xyb_image_to_srgb16_with_variant, xyb_opsin_params,
};
