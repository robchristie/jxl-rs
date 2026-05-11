use crate::bitstream::{BitReader, bits_offset, val};
use crate::decode::ImageRegion;
use crate::entropy::{
    AnsAliasTableProbe, AnsCode, AnsHistogramLogCountProbe, AnsHistogramPopulationProbe,
    AnsHistogramProbe, AnsHistogramProbeKind, AnsHistogramProbeStage, AnsSymbolReader,
    ContextMapProbe, ContextMapProbeKind, ContextMapProbeStage, ContextMapSymbolProbe,
    HistogramCodingProbeStage, decode_context_map, decode_histograms, probe_decode_context_map,
    probe_decode_histograms,
};
use crate::error::{Error, Result};
use crate::frame::{FrameEncoding, FrameHeader, LoopFilter};
use crate::frame_data::{FrameData, FrameSection, FrameSectionKind, section_payload_range};
use crate::metadata::ImageMetadata;
use crate::metadata::unpack_signed;
use crate::modular::{
    MaTreeLeafProbe, ModularDecodedGroup, ModularGroupChannelPlan, ModularGroupHeader,
    ModularPredictor, ModularTreeCoding, decode_modular_stream_from_reader,
    probe_modular_global_tree_coding, read_modular_global_tree_coding,
    read_modular_group_header_metadata,
};
use crate::transform::CustomTransformData;
use std::ops::Range;

const NUM_QUANT_TABLES: usize = 17;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctFrameMetadata {
    pub width: u32,
    pub height: u32,
    pub group_dim: u32,
    pub groups_x: u32,
    pub groups_y: u32,
    pub dc_groups_x: u32,
    pub dc_groups_y: u32,
    pub is_combined: bool,
    pub global_section: Option<VarDctSectionMetadata>,
    pub ac_global_section: Option<VarDctSectionMetadata>,
    pub sections: Vec<VarDctSectionMetadata>,
    pub ac_groups: Vec<VarDctGroupMetadata>,
    pub dc_groups: Vec<VarDctGroupMetadata>,
    pub ac_group_sections: Vec<VarDctPassGroupSectionMetadata>,
    pub dc_group_sections: Vec<VarDctGroupSectionMetadata>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctSectionMetadata {
    pub section_logical_id: usize,
    pub section_physical_index: usize,
    pub section_kind: FrameSectionKind,
    pub codestream_offset: usize,
    pub payload_size: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VarDctGroupMetadata {
    pub group: usize,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctGroupSectionMetadata {
    pub section: VarDctSectionMetadata,
    pub group: VarDctGroupMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctPassGroupSectionMetadata {
    pub section: VarDctSectionMetadata,
    pub pass: usize,
    pub group: VarDctGroupMetadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarDctHistogramProbeStage {
    Lz77Params,
    Lz77UintConfig,
    ContextMap,
    EntropyMode,
    LogAlphabetSize,
    UintConfig,
    PrefixAlphabetSize,
    PrefixCode,
    AnsHistogram,
    AnsAliasTable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarDctAnsHistogramProbeKind {
    Simple,
    Flat,
    Custom,
}

impl From<AnsHistogramProbeKind> for VarDctAnsHistogramProbeKind {
    fn from(kind: AnsHistogramProbeKind) -> Self {
        match kind {
            AnsHistogramProbeKind::Simple => Self::Simple,
            AnsHistogramProbeKind::Flat => Self::Flat,
            AnsHistogramProbeKind::Custom => Self::Custom,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarDctAnsHistogramProbeStage {
    Form,
    SimpleSymbolCount,
    SimpleSymbol,
    SimpleCount,
    FlatAlphabetSize,
    CustomShift,
    CustomLength,
    CustomLogCount,
    CustomRleLength,
    CustomOmit,
    CustomPopulationBits,
    CustomCount,
}

impl From<AnsHistogramProbeStage> for VarDctAnsHistogramProbeStage {
    fn from(stage: AnsHistogramProbeStage) -> Self {
        match stage {
            AnsHistogramProbeStage::Form => Self::Form,
            AnsHistogramProbeStage::SimpleSymbolCount => Self::SimpleSymbolCount,
            AnsHistogramProbeStage::SimpleSymbol => Self::SimpleSymbol,
            AnsHistogramProbeStage::SimpleCount => Self::SimpleCount,
            AnsHistogramProbeStage::FlatAlphabetSize => Self::FlatAlphabetSize,
            AnsHistogramProbeStage::CustomShift => Self::CustomShift,
            AnsHistogramProbeStage::CustomLength => Self::CustomLength,
            AnsHistogramProbeStage::CustomLogCount => Self::CustomLogCount,
            AnsHistogramProbeStage::CustomRleLength => Self::CustomRleLength,
            AnsHistogramProbeStage::CustomOmit => Self::CustomOmit,
            AnsHistogramProbeStage::CustomPopulationBits => Self::CustomPopulationBits,
            AnsHistogramProbeStage::CustomCount => Self::CustomCount,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarDctContextMapProbeKind {
    Simple,
    EntropyCoded,
}

impl From<ContextMapProbeKind> for VarDctContextMapProbeKind {
    fn from(kind: ContextMapProbeKind) -> Self {
        match kind {
            ContextMapProbeKind::Simple => Self::Simple,
            ContextMapProbeKind::EntropyCoded => Self::EntropyCoded,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarDctContextMapProbeStage {
    Kind,
    SimpleBitsPerEntry,
    SimpleEntry,
    Mtf,
    NestedHistogram,
    AnsState,
    Symbol,
    FinalState,
    Verify,
}

impl From<ContextMapProbeStage> for VarDctContextMapProbeStage {
    fn from(stage: ContextMapProbeStage) -> Self {
        match stage {
            ContextMapProbeStage::Kind => Self::Kind,
            ContextMapProbeStage::SimpleBitsPerEntry => Self::SimpleBitsPerEntry,
            ContextMapProbeStage::SimpleEntry => Self::SimpleEntry,
            ContextMapProbeStage::Mtf => Self::Mtf,
            ContextMapProbeStage::NestedHistogram => Self::NestedHistogram,
            ContextMapProbeStage::AnsState => Self::AnsState,
            ContextMapProbeStage::Symbol => Self::Symbol,
            ContextMapProbeStage::FinalState => Self::FinalState,
            ContextMapProbeStage::Verify => Self::Verify,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctContextMapProbe {
    pub start_bits: usize,
    pub end_bits: Option<usize>,
    pub len: usize,
    pub kind: Option<VarDctContextMapProbeKind>,
    pub bits_per_entry: Option<usize>,
    pub use_mtf: Option<bool>,
    pub nested_lz77_end_bits: Option<usize>,
    pub nested_context_map_end_bits: Option<usize>,
    pub nested_entropy_mode_end_bits: Option<usize>,
    pub nested_uint_config_end_bits: Option<usize>,
    pub nested_histogram_end_bits: Option<usize>,
    pub nested_histogram_count: Option<usize>,
    pub nested_use_prefix_code: Option<bool>,
    pub nested_log_alpha_size: Option<usize>,
    pub ans_start_bits: Option<usize>,
    pub ans_end_bits: Option<usize>,
    pub entries_decoded: usize,
    pub entries: Vec<u8>,
    pub raw_entries: Vec<u8>,
    pub symbol_entries: Vec<VarDctContextMapSymbolProbe>,
    pub max_symbol: Option<u32>,
    pub num_histograms: Option<usize>,
    pub final_state_valid: Option<bool>,
    pub error_stage: Option<VarDctContextMapProbeStage>,
    pub error_bits: Option<usize>,
    pub error: Option<Error>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctMaTreeLeafProbe {
    pub leaf_index: usize,
    pub node_index: usize,
    pub residual_context: usize,
    pub predictor: ModularPredictor,
    pub predictor_offset: i64,
    pub multiplier: u32,
}

impl From<&MaTreeLeafProbe> for VarDctMaTreeLeafProbe {
    fn from(probe: &MaTreeLeafProbe) -> Self {
        Self {
            leaf_index: probe.leaf_index,
            node_index: probe.node_index,
            residual_context: probe.residual_context,
            predictor: probe.predictor,
            predictor_offset: probe.predictor_offset,
            multiplier: probe.multiplier,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctContextMapSymbolProbe {
    pub index: usize,
    pub start_bits: usize,
    pub token_end_bits: usize,
    pub end_bits: usize,
    pub clustered_context: usize,
    pub token: usize,
    pub value: u32,
    pub ans_state_before: u32,
    pub ans_state_after_symbol: u32,
    pub ans_state_after: u32,
}

impl From<&ContextMapSymbolProbe> for VarDctContextMapSymbolProbe {
    fn from(probe: &ContextMapSymbolProbe) -> Self {
        Self {
            index: probe.index,
            start_bits: probe.start_bits,
            token_end_bits: probe.token_end_bits,
            end_bits: probe.end_bits,
            clustered_context: probe.clustered_context,
            token: probe.token,
            value: probe.value,
            ans_state_before: probe.ans_state_before,
            ans_state_after_symbol: probe.ans_state_after_symbol,
            ans_state_after: probe.ans_state_after,
        }
    }
}

impl From<&ContextMapProbe> for VarDctContextMapProbe {
    fn from(probe: &ContextMapProbe) -> Self {
        let nested = probe.nested_histogram.as_ref();
        Self {
            start_bits: probe.start_bits,
            end_bits: probe.end_bits,
            len: probe.len,
            kind: probe.kind.map(VarDctContextMapProbeKind::from),
            bits_per_entry: probe.bits_per_entry,
            use_mtf: probe.use_mtf,
            nested_lz77_end_bits: nested.and_then(|probe| probe.lz77_end_bits),
            nested_context_map_end_bits: nested.and_then(|probe| probe.context_map_end_bits),
            nested_entropy_mode_end_bits: nested.and_then(|probe| probe.entropy_mode_end_bits),
            nested_uint_config_end_bits: nested.and_then(|probe| probe.uint_config_end_bits),
            nested_histogram_end_bits: nested.and_then(|probe| probe.histogram_end_bits),
            nested_histogram_count: nested.and_then(|probe| probe.num_histograms),
            nested_use_prefix_code: nested.and_then(|probe| probe.use_prefix_code),
            nested_log_alpha_size: nested.and_then(|probe| probe.log_alpha_size),
            ans_start_bits: probe.ans_start_bits,
            ans_end_bits: probe.ans_end_bits,
            entries_decoded: probe.entries_decoded,
            entries: probe.entries.clone(),
            raw_entries: probe.raw_entries.clone(),
            symbol_entries: probe
                .symbol_entries
                .iter()
                .map(VarDctContextMapSymbolProbe::from)
                .collect(),
            max_symbol: probe.max_symbol,
            num_histograms: probe.num_histograms,
            final_state_valid: probe.final_state_valid,
            error_stage: probe.error_stage.map(VarDctContextMapProbeStage::from),
            error_bits: probe.error_bits,
            error: probe.error.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctAnsHistogramProbe {
    pub start_bits: usize,
    pub end_bits: Option<usize>,
    pub kind: Option<VarDctAnsHistogramProbeKind>,
    pub precision_bits: usize,
    pub simple_symbol_count: Option<usize>,
    pub alphabet_size: Option<usize>,
    pub length: Option<usize>,
    pub shift: Option<u32>,
    pub omit_pos: Option<usize>,
    pub error_stage: Option<VarDctAnsHistogramProbeStage>,
    pub error_bits: Option<usize>,
    pub error: Option<Error>,
    pub log_count_entries: Vec<VarDctAnsHistogramLogCountProbe>,
    pub log_count_error_index: Option<usize>,
    pub population_entries: Vec<VarDctAnsHistogramPopulationProbe>,
    pub population_error_index: Option<usize>,
    pub total_count_before_omit: Option<i32>,
    pub omit_count: Option<i32>,
    pub final_counts: Option<Vec<i32>>,
    pub alias_table: Option<VarDctAnsAliasTableProbe>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctAnsHistogramLogCountProbe {
    pub index: usize,
    pub start_bits: usize,
    pub end_bits: usize,
    pub huffman_bits: u8,
    pub huffman_value: u8,
    pub logcount: i32,
    pub rle_length: Option<usize>,
    pub rle_end_bits: Option<usize>,
    pub next_index: usize,
}

impl From<&AnsHistogramLogCountProbe> for VarDctAnsHistogramLogCountProbe {
    fn from(probe: &AnsHistogramLogCountProbe) -> Self {
        Self {
            index: probe.index,
            start_bits: probe.start_bits,
            end_bits: probe.end_bits,
            huffman_bits: probe.huffman_bits,
            huffman_value: probe.huffman_value,
            logcount: probe.logcount,
            rle_length: probe.rle_length,
            rle_end_bits: probe.rle_end_bits,
            next_index: probe.next_index,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctAnsHistogramPopulationProbe {
    pub index: usize,
    pub start_bits: usize,
    pub end_bits: usize,
    pub code: i32,
    pub bitcount: usize,
    pub extra_bits: Option<u64>,
    pub count: i32,
    pub copied: bool,
    pub omitted: bool,
}

impl From<&AnsHistogramPopulationProbe> for VarDctAnsHistogramPopulationProbe {
    fn from(probe: &AnsHistogramPopulationProbe) -> Self {
        Self {
            index: probe.index,
            start_bits: probe.start_bits,
            end_bits: probe.end_bits,
            code: probe.code,
            bitcount: probe.bitcount,
            extra_bits: probe.extra_bits,
            count: probe.count,
            copied: probe.copied,
            omitted: probe.omitted,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctAnsAliasTableProbe {
    pub table_size: usize,
    pub entry_size: u32,
    pub distribution_len: usize,
    pub nonzero_symbols: usize,
    pub count_sum: i32,
    pub first_nonzero_symbol: Option<usize>,
    pub last_nonzero_symbol: Option<usize>,
    pub table_checksum: u64,
}

impl From<&AnsAliasTableProbe> for VarDctAnsAliasTableProbe {
    fn from(probe: &AnsAliasTableProbe) -> Self {
        Self {
            table_size: probe.table_size,
            entry_size: probe.entry_size,
            distribution_len: probe.distribution_len,
            nonzero_symbols: probe.nonzero_symbols,
            count_sum: probe.count_sum,
            first_nonzero_symbol: probe.first_nonzero_symbol,
            last_nonzero_symbol: probe.last_nonzero_symbol,
            table_checksum: probe.table_checksum,
        }
    }
}

impl From<&AnsHistogramProbe> for VarDctAnsHistogramProbe {
    fn from(probe: &AnsHistogramProbe) -> Self {
        Self {
            start_bits: probe.start_bits,
            end_bits: probe.end_bits,
            kind: probe.kind.map(VarDctAnsHistogramProbeKind::from),
            precision_bits: probe.precision_bits,
            simple_symbol_count: probe.simple_symbol_count,
            alphabet_size: probe.alphabet_size,
            length: probe.length,
            shift: probe.shift,
            omit_pos: probe.omit_pos,
            error_stage: probe.error_stage.map(VarDctAnsHistogramProbeStage::from),
            error_bits: probe.error_bits,
            error: probe.error.clone(),
            log_count_entries: probe
                .log_count_entries
                .iter()
                .map(VarDctAnsHistogramLogCountProbe::from)
                .collect(),
            log_count_error_index: probe.log_count_error_index,
            population_entries: probe
                .population_entries
                .iter()
                .map(VarDctAnsHistogramPopulationProbe::from)
                .collect(),
            population_error_index: probe.population_error_index,
            total_count_before_omit: probe.total_count_before_omit,
            omit_count: probe.omit_count,
            final_counts: probe.final_counts.clone(),
            alias_table: probe
                .alias_table
                .as_ref()
                .map(VarDctAnsAliasTableProbe::from),
        }
    }
}

impl From<HistogramCodingProbeStage> for VarDctHistogramProbeStage {
    fn from(stage: HistogramCodingProbeStage) -> Self {
        match stage {
            HistogramCodingProbeStage::Lz77Params => Self::Lz77Params,
            HistogramCodingProbeStage::Lz77UintConfig => Self::Lz77UintConfig,
            HistogramCodingProbeStage::ContextMap => Self::ContextMap,
            HistogramCodingProbeStage::EntropyMode => Self::EntropyMode,
            HistogramCodingProbeStage::LogAlphabetSize => Self::LogAlphabetSize,
            HistogramCodingProbeStage::UintConfig => Self::UintConfig,
            HistogramCodingProbeStage::PrefixAlphabetSize => Self::PrefixAlphabetSize,
            HistogramCodingProbeStage::PrefixCode => Self::PrefixCode,
            HistogramCodingProbeStage::AnsHistogram => Self::AnsHistogram,
            HistogramCodingProbeStage::AnsAliasTable => Self::AnsAliasTable,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VarDctDecodePlan {
    pub frame: VarDctFrameMetadata,
    pub loop_filter: LoopFilter,
    pub opsin_params: VarDctOpsinParams,
    pub epf_metadata: Option<VarDctEpfMetadata>,
    pub global: Option<VarDctGlobalMetadata>,
    pub modular_global_tree_payload_start_bits: Option<usize>,
    pub modular_global_tree_payload_end_bits: Option<usize>,
    pub modular_global_tree_payload_len_bits: Option<usize>,
    pub modular_global_tree_direct_start_bits: Option<usize>,
    pub modular_global_tree_direct_start_absolute_bits: Option<usize>,
    pub modular_global_tree_direct_start_remaining_bits: Option<usize>,
    pub modular_global_tree_direct_tree_end_bits: Option<usize>,
    pub modular_global_tree_direct_tree_end_absolute_bits: Option<usize>,
    pub modular_global_tree_direct_tree_end_remaining_bits: Option<usize>,
    pub modular_global_tree_direct_tree_node_count: Option<usize>,
    pub modular_global_tree_direct_tree_leaf_count: Option<usize>,
    pub modular_global_tree_direct_tree_leaves: Vec<VarDctMaTreeLeafProbe>,
    pub modular_global_tree_direct_error_bits: Option<usize>,
    pub modular_global_tree_direct_error_absolute_bits: Option<usize>,
    pub modular_global_tree_direct_error_remaining_bits: Option<usize>,
    pub modular_global_tree_direct_residual_context_count: Option<usize>,
    pub modular_global_tree_direct_residual_histogram_count: Option<usize>,
    pub modular_global_tree_direct_residual_context_map_entries: Vec<u8>,
    pub modular_global_tree_direct_residual_context_map_raw_entries: Vec<u8>,
    pub modular_global_tree_direct_residual_context_map_distinct_entries: Vec<u8>,
    pub modular_global_tree_direct_residual_context_map_histogram_usage_counts: Vec<usize>,
    pub modular_global_tree_direct_residual_context_map_max_entry: Option<u8>,
    pub modular_global_tree_direct_residual_context_map_symbol_entries:
        Vec<VarDctContextMapSymbolProbe>,
    pub modular_global_tree_direct_residual_lz77_end_bits: Option<usize>,
    pub modular_global_tree_direct_residual_context_map_end_bits: Option<usize>,
    pub modular_global_tree_direct_residual_entropy_mode_end_bits: Option<usize>,
    pub modular_global_tree_direct_residual_log_alpha_size_end_bits: Option<usize>,
    pub modular_global_tree_direct_residual_uint_config_end_bits_by_histogram: Vec<usize>,
    pub modular_global_tree_direct_residual_uint_config_end_bits: Option<usize>,
    pub modular_global_tree_direct_residual_use_prefix_code: Option<bool>,
    pub modular_global_tree_direct_residual_log_alpha_size: Option<usize>,
    pub modular_global_tree_direct_residual_failed_histogram_index: Option<usize>,
    pub modular_global_tree_direct_residual_error_stage: Option<VarDctHistogramProbeStage>,
    pub modular_global_tree_direct_residual_ans_histograms: Vec<VarDctAnsHistogramProbe>,
    pub modular_global_tree_start_bits: Option<usize>,
    pub modular_global_tree_start_absolute_bits: Option<usize>,
    pub modular_global_tree_start_remaining_bits: Option<usize>,
    pub modular_global_tree_direct_error: Option<Error>,
    pub modular_global_tree_error: Option<Error>,
    pub global_payload: Option<VarDctSectionPayloadMetadata>,
    pub ac_global_payload: Option<VarDctSectionPayloadMetadata>,
    pub ac_global_metadata: Option<VarDctAcGlobalMetadata>,
    pub ac_group_payloads: Vec<VarDctPassGroupPayloadMetadata>,
    pub ac_group_metadata: Vec<VarDctAcGroupMetadata>,
    pub dc_group_payloads: Vec<VarDctDcGroupPayloadMetadata>,
    pub dc_group_metadata: Vec<VarDctDcGroupMetadata>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VarDctOpsinParams {
    pub inverse_matrix: [[f32; 3]; 3],
    pub opsin_biases: [f32; 3],
    pub opsin_biases_cbrt: [f32; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct VarDctEpfMetadata {
    pub width_blocks: usize,
    pub height_blocks: usize,
    pub raw_quant_field: Vec<i32>,
    pub epf_sharpness: Vec<u8>,
    /// Per-image-block inverse sigma as `f32::to_bits()`.
    pub inv_sigma: Vec<u32>,
    pub first_block_count: usize,
    pub raw_quant_checksum: u64,
    pub epf_sharpness_checksum: u64,
    pub inv_sigma_checksum: u64,
    pub parse_error: Option<Error>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VarDctXybImage {
    pub width: u32,
    pub height: u32,
    /// Number of pass-0 AC groups copied into this image.
    pub groups_assembled: usize,
    /// Number of pass-0 AC groups that did not yet have spatial+DC samples.
    ///
    /// Missing groups are left as zeroes in `channels`. This keeps the
    /// assembly step useful while VarDCT group reconstruction is still being
    /// expanded to every AC group.
    pub groups_missing: usize,
    pub channels: [Vec<f32>; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct VarDctRgbImage {
    pub width: u32,
    pub height: u32,
    pub channels: [Vec<f32>; 3],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctChannelRangeDiagnostics {
    pub nonzero_samples: usize,
    pub negative_samples: usize,
    pub above_one_samples: usize,
    pub min_bits: u32,
    pub max_bits: u32,
    pub sum_bits: u32,
    pub checksum: u64,
    pub anchors_bits: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctXybRgbDiagnostics {
    pub xyb_channels: [VarDctChannelRangeDiagnostics; 3],
    pub rgb_channels: [VarDctChannelRangeDiagnostics; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarDctXybInverseVariant {
    BMinusBias,
    BPlusBias,
    NegBMinusBias,
    NegBPlusBias,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctXybInverseVariantDiagnostics {
    pub variant: VarDctXybInverseVariant,
    pub rgb_channels: [VarDctChannelRangeDiagnostics; 3],
    pub srgb8: VarDctSrgb8Image,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctSrgb8Image {
    pub width: u32,
    pub height: u32,
    /// Interleaved RGB pixels after clamping linear samples to [0, 1] and
    /// applying the sRGB electro-optical transfer function.
    pub pixels: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctSrgb16Image {
    pub width: u32,
    pub height: u32,
    /// Interleaved RGB pixels after clamping linear samples to [0, 1] and
    /// applying the sRGB electro-optical transfer function.
    pub pixels: Vec<u16>,
}

impl VarDctXybImage {
    pub fn sample(&self, channel: usize, x: u32, y: u32) -> Option<f32> {
        if channel >= self.channels.len() || x >= self.width || y >= self.height {
            return None;
        }
        self.channels[channel]
            .get((y as usize) * self.width as usize + x as usize)
            .copied()
    }
}

/// Assembles available final VarDCT spatial+DC group grids into full-frame XYB channels.
///
/// For progressive AC frames this merges all available AC passes up to the
/// latest pass for each group. Returns `Ok(None)` when no selected group has
/// spatial+DC samples. Groups without spatial+DC samples are counted in
/// `VarDctXybImage::groups_missing` and left as zeroes in the output buffers.
pub fn assemble_vardct_xyb_image(plan: &VarDctDecodePlan) -> Result<Option<VarDctXybImage>> {
    let mut image = assemble_vardct_xyb_image_final(plan)?;
    if let Some(image) = image.as_mut() {
        apply_vardct_gaborish(image, &plan.loop_filter);
        if plan.loop_filter.epf_iters >= 1 {
            if let Some(epf) = plan.epf_metadata.as_ref() {
                apply_vardct_epf(image, &plan.loop_filter, epf);
            }
        }
    }
    Ok(image)
}

/// Assembles available VarDCT XYB data for exactly one AC pass.
///
/// This is a low-level progressive reconstruction helper. It selects only group
/// metadata whose AC pass equals `pass`; it does not merge data from earlier or
/// later passes and is therefore not yet a complete progressive preview
/// compositor. Returns `Ok(None)` when no group for the requested pass has a
/// spatial+DC grid.
pub fn assemble_vardct_xyb_image_for_pass(
    plan: &VarDctDecodePlan,
    pass: usize,
) -> Result<Option<VarDctXybImage>> {
    let mut image = assemble_vardct_xyb_image_from_groups_with_mode(
        &plan.frame,
        &plan.ac_group_metadata,
        VarDctAssemblyMode::Pass { pass },
    )?;
    if let Some(image) = image.as_mut() {
        apply_vardct_gaborish(image, &plan.loop_filter);
        if plan.loop_filter.epf_iters >= 1 {
            if let Some(epf) = plan.epf_metadata.as_ref() {
                apply_vardct_epf(image, &plan.loop_filter, epf);
            }
        }
    }
    Ok(image)
}

/// Assembles a DC-only VarDCT XYB image.
///
/// This is a reconstruction diagnostic helper. It spatializes each final AC
/// group using the parsed DC coefficients and zero AC coefficients, then applies
/// the same loop filters as `assemble_vardct_xyb_image`.
pub fn assemble_vardct_dc_xyb_image(plan: &VarDctDecodePlan) -> Result<Option<VarDctXybImage>> {
    let mut image = assemble_vardct_xyb_image_dc_only(plan, 8.0)?;
    if let Some(image) = image.as_mut() {
        apply_vardct_gaborish(image, &plan.loop_filter);
        if plan.loop_filter.epf_iters >= 1 {
            if let Some(epf) = plan.epf_metadata.as_ref() {
                apply_vardct_epf(image, &plan.loop_filter, epf);
            }
        }
    }
    Ok(image)
}

/// Assembles available VarDCT XYB data and converts it to linear RGB.
///
/// This is intentionally still an internal reconstruction stage: it applies the
/// inverse opsin transform, but not output color management, transfer functions,
/// orientation, or conversion to integer samples.
pub fn assemble_vardct_linear_rgb_image(plan: &VarDctDecodePlan) -> Result<Option<VarDctRgbImage>> {
    assemble_vardct_xyb_image(plan)
        .map(|image| image.map(|image| vardct_xyb_to_linear_rgb(&image, &plan.opsin_params)))
}

/// Assembles available VarDCT XYB data and converts it to interleaved sRGB8.
///
/// This is a debugging and fixture-oracle convenience layer over
/// `assemble_vardct_linear_rgb_image`: it does not yet perform full JPEG XL
/// color management, orientation handling, or post-filtering.
pub fn assemble_vardct_srgb8_image(plan: &VarDctDecodePlan) -> Result<Option<VarDctSrgb8Image>> {
    assemble_vardct_linear_rgb_image(plan)
        .map(|image| image.map(|image| vardct_linear_rgb_to_srgb8(&image)))
}

/// Assembles one VarDCT AC pass and converts it to interleaved sRGB8.
///
/// This selects exactly one AC pass and does not merge data from earlier or
/// later passes. Returns `Ok(None)` when the requested pass has no spatial+DC
/// grid.
pub fn assemble_vardct_srgb8_image_for_pass(
    plan: &VarDctDecodePlan,
    pass: usize,
) -> Result<Option<VarDctSrgb8Image>> {
    assemble_vardct_xyb_image_for_pass(plan, pass).map(|image| {
        image.map(|image| {
            vardct_linear_rgb_to_srgb8(&vardct_xyb_to_linear_rgb(&image, &plan.opsin_params))
        })
    })
}

/// Assembles DC-only VarDCT XYB data and converts it to interleaved sRGB8.
pub fn assemble_vardct_dc_srgb8_image(plan: &VarDctDecodePlan) -> Result<Option<VarDctSrgb8Image>> {
    assemble_vardct_dc_xyb_image(plan).map(|image| {
        image.map(|image| {
            vardct_linear_rgb_to_srgb8(&vardct_xyb_to_linear_rgb(&image, &plan.opsin_params))
        })
    })
}

/// Evaluates DC-only reconstruction with an alternate DC coefficient multiplier.
///
/// This is a diagnostic helper for checking the normalization boundary between
/// parsed VarDCT DC coefficients and inverse DCT spatialization. A multiplier
/// of `8.0` is equivalent to `assemble_vardct_dc_srgb8_image`.
pub fn assemble_vardct_dc_srgb8_image_with_multiplier(
    plan: &VarDctDecodePlan,
    dc_multiplier: f32,
) -> Result<Option<VarDctSrgb8Image>> {
    assemble_vardct_xyb_image_dc_only(plan, dc_multiplier).map(|image| {
        image.map(|mut image| {
            apply_vardct_gaborish(&mut image, &plan.loop_filter);
            if plan.loop_filter.epf_iters >= 1 {
                if let Some(epf) = plan.epf_metadata.as_ref() {
                    apply_vardct_epf(&mut image, &plan.loop_filter, epf);
                }
            }
            vardct_linear_rgb_to_srgb8(&vardct_xyb_to_linear_rgb(&image, &plan.opsin_params))
        })
    })
}

/// Summarizes raw and scaled VarDCT DC coefficients for each final AC group.
pub fn vardct_dc_coefficient_diagnostics(
    plan: &VarDctDecodePlan,
) -> Result<Vec<VarDctDcCoefficientDiagnostics>> {
    final_vardct_ac_passes_by_group(&plan.ac_group_metadata)
        .into_iter()
        .map(|metadata| vardct_dc_coefficient_diagnostics_for_group(plan, metadata))
        .collect()
}

/// Summarizes XYB and linear RGB ranges for final VarDCT reconstruction.
pub fn vardct_xyb_rgb_diagnostics(
    plan: &VarDctDecodePlan,
) -> Result<Option<VarDctXybRgbDiagnostics>> {
    let Some(xyb) = assemble_vardct_xyb_image(plan)? else {
        return Ok(None);
    };
    let rgb = vardct_xyb_to_linear_rgb(&xyb, &plan.opsin_params);
    Ok(Some(VarDctXybRgbDiagnostics {
        xyb_channels: std::array::from_fn(|channel| {
            channel_range_diagnostics(&xyb.channels[channel])
        }),
        rgb_channels: std::array::from_fn(|channel| {
            channel_range_diagnostics(&rgb.channels[channel])
        }),
    }))
}

/// Evaluates alternate XYB inverse formulas against final VarDCT reconstruction.
///
/// This diagnostic makes sign and bias hypotheses measurable against fixture
/// oracles. Production output currently uses
/// `VarDctXybInverseVariant::NegBMinusBias`.
pub fn vardct_xyb_inverse_variant_diagnostics(
    plan: &VarDctDecodePlan,
) -> Result<Option<Vec<VarDctXybInverseVariantDiagnostics>>> {
    let Some(xyb) = assemble_vardct_xyb_image(plan)? else {
        return Ok(None);
    };
    let variants = [
        VarDctXybInverseVariant::BMinusBias,
        VarDctXybInverseVariant::BPlusBias,
        VarDctXybInverseVariant::NegBMinusBias,
        VarDctXybInverseVariant::NegBPlusBias,
    ];

    Ok(Some(
        variants
            .into_iter()
            .map(|variant| {
                let rgb = vardct_xyb_to_linear_rgb_with_variant(&xyb, &plan.opsin_params, variant);
                VarDctXybInverseVariantDiagnostics {
                    variant,
                    rgb_channels: std::array::from_fn(|channel| {
                        channel_range_diagnostics(&rgb.channels[channel])
                    }),
                    srgb8: vardct_linear_rgb_to_srgb8(&rgb),
                }
            })
            .collect(),
    ))
}

/// Assembles available VarDCT XYB data and converts it to interleaved sRGB16.
///
/// Like `assemble_vardct_srgb8_image`, this is a debugging and fixture-oracle
/// convenience layer rather than full JPEG XL output color management.
pub fn assemble_vardct_srgb16_image(plan: &VarDctDecodePlan) -> Result<Option<VarDctSrgb16Image>> {
    assemble_vardct_linear_rgb_image(plan)
        .map(|image| image.map(|image| vardct_linear_rgb_to_srgb16(&image)))
}

/// Assembles one VarDCT AC pass and converts it to interleaved sRGB16.
///
/// This selects exactly one AC pass and does not merge data from earlier or
/// later passes. Returns `Ok(None)` when the requested pass has no spatial+DC
/// grid.
pub fn assemble_vardct_srgb16_image_for_pass(
    plan: &VarDctDecodePlan,
    pass: usize,
) -> Result<Option<VarDctSrgb16Image>> {
    assemble_vardct_xyb_image_for_pass(plan, pass).map(|image| {
        image.map(|image| {
            vardct_linear_rgb_to_srgb16(&vardct_xyb_to_linear_rgb(&image, &plan.opsin_params))
        })
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctAcGlobalMetadata {
    pub section: VarDctSectionPayloadMetadata,
    pub all_default_quant_matrices: Option<bool>,
    pub quant_matrices_end_bits: Option<usize>,
    pub num_histograms: Option<usize>,
    pub num_histograms_end_bits: Option<usize>,
    pub used_acs: Option<u32>,
    pub passes: Vec<VarDctAcGlobalPassMetadata>,
    pub bits_consumed: Option<usize>,
    pub parse_error: Option<Error>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctAcGlobalPassMetadata {
    pub pass: usize,
    pub used_orders: Option<u32>,
    pub used_orders_end_bits: Option<usize>,
    pub coeff_orders: Vec<VarDctCoeffOrderMetadata>,
    pub coeff_order_end_bits: Option<usize>,
    pub histogram_contexts: Option<usize>,
    pub histogram_count: Option<usize>,
    pub histogram_end_bits: Option<usize>,
    pub use_prefix_code: Option<bool>,
    pub log_alpha_size: Option<usize>,
    pub error_stage: Option<VarDctHistogramProbeStage>,
    pub error_bits: Option<usize>,
    pub error: Option<Error>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctCoeffOrderMetadata {
    pub order: usize,
    pub channel: usize,
    pub skip: usize,
    pub size: usize,
    pub permutation_end: usize,
    pub checksum: u64,
    pub positions: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctSectionPayloadMetadata {
    pub section: VarDctSectionMetadata,
    pub payload_range: Range<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctGroupPayloadMetadata {
    pub section: VarDctSectionPayloadMetadata,
    pub group: VarDctGroupMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctDcGroupPayloadMetadata {
    pub section: VarDctSectionPayloadMetadata,
    pub group: VarDctGroupMetadata,
    pub var_dct_dc_stream_id: usize,
    pub modular_dc_stream_id: usize,
    pub ac_metadata_stream_id: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctDcGroupMetadata {
    pub payload: VarDctDcGroupPayloadMetadata,
    pub cursor: VarDctDcGroupCursorMetadata,
    pub extra_precision_bits: Option<u8>,
    pub var_dct_dc_header: Option<ModularGroupHeader>,
    pub var_dct_dc: Option<ModularDecodedGroup>,
    pub modular_dc: Option<ModularDecodedGroup>,
    pub modular_dc_error: Option<Error>,
    pub ac_metadata_count: Option<usize>,
    pub ac_metadata: Option<ModularDecodedGroup>,
    pub ac_metadata_error: Option<Error>,
    pub parse_error: Option<Error>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VarDctDcGroupCursorMetadata {
    pub extra_precision_start_bits: usize,
    pub extra_precision_end_bits: Option<usize>,
    pub var_dct_dc_start_bits: Option<usize>,
    pub var_dct_dc_header_end_bits: Option<usize>,
    pub var_dct_dc_end_bits: Option<usize>,
    pub modular_dc_start_bits: Option<usize>,
    pub modular_dc_end_bits: Option<usize>,
    pub ac_metadata_start_bits: Option<usize>,
    pub ac_metadata_end_bits: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctDcCoefficientDiagnostics {
    pub ac_group: usize,
    pub dc_group: usize,
    pub width_blocks: usize,
    pub height_blocks: usize,
    pub inv_quant_dc_bits: u32,
    pub dc_dequant_bits: [u32; 3],
    pub raw_channels: [VarDctDcRawChannelDiagnostics; 3],
    pub scaled_channels: [VarDctDcScaledChannelDiagnostics; 3],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctDcRawChannelDiagnostics {
    pub output_channel: usize,
    pub modular_channel: usize,
    pub width: u32,
    pub height: u32,
    pub nonzero_samples: usize,
    pub sample_min: i32,
    pub sample_max: i32,
    pub sample_sum: i64,
    pub sample_checksum: u64,
    pub anchors: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctDcScaledChannelDiagnostics {
    pub output_channel: usize,
    pub scale_bits: u32,
    pub nonzero_coefficients: usize,
    pub coefficient_checksum: u64,
    pub anchors_bits: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctPassGroupPayloadMetadata {
    pub section: VarDctSectionPayloadMetadata,
    pub pass: usize,
    pub group: VarDctGroupMetadata,
    pub modular_ac_stream_id: usize,
    pub modular_ac_min_shift: i32,
    pub modular_ac_max_shift: i32,
    pub modular_ac_channels: Vec<ModularGroupChannelPlan>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctAcGroupMetadata {
    pub payload: VarDctPassGroupPayloadMetadata,
    pub cursor: VarDctAcGroupCursorMetadata,
    pub histogram_selector_bits: usize,
    pub histogram_selector: Option<usize>,
    pub entropy_uses_prefix_code: Option<bool>,
    pub coefficient_probe: Option<VarDctAcCoefficientProbe>,
    pub channel_trace: Option<VarDctAcChannelTrace>,
    pub coefficient_summary: Option<VarDctAcCoefficientSummary>,
    pub coefficient_grid: Option<VarDctAcCoefficientGrid>,
    pub base_dequantized_grid: Option<VarDctAcBaseDequantizedGrid>,
    pub dequantized_grid: Option<VarDctAcDequantizedGrid>,
    pub spatial_grid: Option<VarDctAcSpatialGrid>,
    pub spatial_with_dc_grid: Option<VarDctAcSpatialGrid>,
    pub parse_error: Option<Error>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VarDctAcGroupCursorMetadata {
    pub payload_start_bits: usize,
    pub payload_end_bits: usize,
    pub histogram_selector_start_bits: usize,
    pub histogram_selector_end_bits: Option<usize>,
    pub ans_state_start_bits: Option<usize>,
    pub ans_state_end_bits: Option<usize>,
    pub coefficient_stream_start_bits: Option<usize>,
    pub modular_ac_start_bits: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctAcCoefficientProbe {
    pub block_x: usize,
    pub block_y: usize,
    pub channel: usize,
    pub raw_strategy: usize,
    pub order: usize,
    pub covered_blocks: usize,
    pub block_size: usize,
    pub block_context: usize,
    pub nonzero_context: usize,
    pub clustered_context: usize,
    pub start_bits: usize,
    pub nzeros_end_bits: usize,
    pub nzeros: u32,
    pub block_end_bits: Option<usize>,
    pub remaining_nzeros: Option<usize>,
    pub coefficient_event_count: usize,
    pub coefficient_events: Vec<VarDctAcCoefficientEvent>,
    pub coefficient_event_checksum: u64,
    pub placed_nonzero_coefficients: usize,
    pub placed_coefficient_checksum: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctAcCoefficientEvent {
    pub k: usize,
    pub zero_density_context: usize,
    pub context: usize,
    pub clustered_context: usize,
    pub start_bits: usize,
    pub end_bits: usize,
    pub u_coeff: u32,
    pub coeff: i32,
    pub prev_after: usize,
    pub remaining_nzeros: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctAcChannelTrace {
    pub channel: usize,
    pub blocks_decoded: usize,
    pub coefficient_events_decoded: usize,
    pub final_bits: usize,
    pub row_nzeros_checksum: u64,
    pub coefficient_event_checksum: u64,
    pub block_summaries: Vec<VarDctAcBlockSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctAcCoefficientSummary {
    pub group: usize,
    pub pass: usize,
    pub blocks_decoded: usize,
    pub final_bits: usize,
    pub per_channel: [VarDctAcChannelCoefficientSummary; 3],
    pub first_block_checksum: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VarDctAcChannelCoefficientSummary {
    pub blocks_decoded: usize,
    pub coefficients_written: usize,
    pub nonzero_coefficients: usize,
    pub coefficient_checksum: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctAcCoefficientGrid {
    pub group: usize,
    pub pass: usize,
    pub width_blocks: usize,
    pub height_blocks: usize,
    pub per_channel: [VarDctAcChannelCoefficientGrid; 3],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctAcChannelCoefficientGrid {
    pub coefficients: Vec<i32>,
    pub nonzero_coefficients: usize,
    pub coefficient_checksum: u64,
}

impl VarDctAcChannelCoefficientGrid {
    fn new(len: usize) -> Self {
        Self {
            coefficients: vec![0; len],
            nonzero_coefficients: 0,
            coefficient_checksum: 0,
        }
    }
}

impl VarDctAcCoefficientGrid {
    pub fn coefficient(
        &self,
        channel: usize,
        block_x: usize,
        block_y: usize,
        coeff: usize,
    ) -> Option<i32> {
        if channel >= self.per_channel.len()
            || block_x >= self.width_blocks
            || block_y >= self.height_blocks
            || coeff >= DCT_BLOCK_SIZE
        {
            return None;
        }
        let index = ((block_y * self.width_blocks + block_x) * DCT_BLOCK_SIZE) + coeff;
        self.per_channel[channel].coefficients.get(index).copied()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctAcBaseDequantizedGrid {
    pub group: usize,
    pub pass: usize,
    pub width_blocks: usize,
    pub height_blocks: usize,
    /// Raw `f32::to_bits()` for the global inverse scale used by this base pass.
    pub inv_global_scale_bits: u32,
    pub per_channel: [VarDctAcBaseDequantizedChannelGrid; 3],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctAcBaseDequantizedChannelGrid {
    /// Base-dequantized coefficients as `f32::to_bits()` for deterministic metadata equality.
    pub coefficients: Vec<u32>,
    pub nonzero_coefficients: usize,
    pub coefficient_checksum: u64,
}

impl VarDctAcBaseDequantizedChannelGrid {
    fn new(len: usize) -> Self {
        Self {
            coefficients: vec![0; len],
            nonzero_coefficients: 0,
            coefficient_checksum: 0,
        }
    }
}

impl VarDctAcBaseDequantizedGrid {
    pub fn coefficient(
        &self,
        channel: usize,
        block_x: usize,
        block_y: usize,
        coeff: usize,
    ) -> Option<f32> {
        if channel >= self.per_channel.len()
            || block_x >= self.width_blocks
            || block_y >= self.height_blocks
            || coeff >= DCT_BLOCK_SIZE
        {
            return None;
        }
        let index = ((block_y * self.width_blocks + block_x) * DCT_BLOCK_SIZE) + coeff;
        self.per_channel[channel]
            .coefficients
            .get(index)
            .copied()
            .map(f32::from_bits)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctAcDequantizedGrid {
    pub group: usize,
    pub pass: usize,
    pub width_blocks: usize,
    pub height_blocks: usize,
    pub per_channel: [VarDctAcDequantizedChannelGrid; 3],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctAcDequantizedChannelGrid {
    /// Dequantized coefficients as `f32::to_bits()` for deterministic metadata equality.
    pub coefficients: Vec<u32>,
    pub nonzero_coefficients: usize,
    pub coefficient_checksum: u64,
}

impl VarDctAcDequantizedChannelGrid {
    fn new(len: usize) -> Self {
        Self {
            coefficients: vec![0; len],
            nonzero_coefficients: 0,
            coefficient_checksum: 0,
        }
    }
}

impl VarDctAcDequantizedGrid {
    pub fn coefficient(
        &self,
        channel: usize,
        block_x: usize,
        block_y: usize,
        coeff: usize,
    ) -> Option<f32> {
        if channel >= self.per_channel.len()
            || block_x >= self.width_blocks
            || block_y >= self.height_blocks
            || coeff >= DCT_BLOCK_SIZE
        {
            return None;
        }
        let index = ((block_y * self.width_blocks + block_x) * DCT_BLOCK_SIZE) + coeff;
        self.per_channel[channel]
            .coefficients
            .get(index)
            .copied()
            .map(f32::from_bits)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctAcSpatialGrid {
    pub group: usize,
    pub pass: usize,
    pub width_blocks: usize,
    pub height_blocks: usize,
    pub blocks_attempted: usize,
    pub blocks_transformed: usize,
    pub blocks_skipped: usize,
    pub per_channel: [VarDctAcSpatialChannelGrid; 3],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctAcSpatialChannelGrid {
    /// Spatial-domain DCT8 samples as `f32::to_bits()` for deterministic metadata equality.
    pub samples: Vec<u32>,
    pub nonzero_samples: usize,
    pub sample_checksum: u64,
}

impl VarDctAcSpatialChannelGrid {
    fn new(len: usize) -> Self {
        Self {
            samples: vec![0; len],
            nonzero_samples: 0,
            sample_checksum: 0,
        }
    }
}

impl VarDctAcSpatialGrid {
    pub fn sample(
        &self,
        channel: usize,
        block_x: usize,
        block_y: usize,
        sample: usize,
    ) -> Option<f32> {
        if channel >= self.per_channel.len()
            || block_x >= self.width_blocks
            || block_y >= self.height_blocks
            || sample >= DCT_BLOCK_SIZE
        {
            return None;
        }
        let index = ((block_y * self.width_blocks + block_x) * DCT_BLOCK_SIZE) + sample;
        self.per_channel[channel]
            .samples
            .get(index)
            .copied()
            .map(f32::from_bits)
    }
}

#[cfg(test)]
fn assemble_vardct_xyb_image_from_groups(
    frame: &VarDctFrameMetadata,
    groups: &[VarDctAcGroupMetadata],
) -> Result<Option<VarDctXybImage>> {
    assemble_vardct_xyb_image_from_groups_with_mode(frame, groups, VarDctAssemblyMode::Final)
}

fn assemble_vardct_xyb_image_from_groups_with_mode(
    frame: &VarDctFrameMetadata,
    groups: &[VarDctAcGroupMetadata],
    mode: VarDctAssemblyMode,
) -> Result<Option<VarDctXybImage>> {
    let sample_len = (frame.width as usize)
        .checked_mul(frame.height as usize)
        .ok_or(Error::InvalidCodestream("VarDCT image is too large"))?;
    let mut image = VarDctXybImage {
        width: frame.width,
        height: frame.height,
        groups_assembled: 0,
        groups_missing: 0,
        channels: [
            vec![0.0; sample_len],
            vec![0.0; sample_len],
            vec![0.0; sample_len],
        ],
    };

    for metadata in vardct_ac_groups_for_assembly(groups, mode) {
        let Some(grid) = metadata.spatial_with_dc_grid.as_ref() else {
            image.groups_missing += 1;
            continue;
        };
        if grid.group != metadata.payload.group.group
            || grid.width_blocks != metadata.payload.group.width.div_ceil(8) as usize
            || grid.height_blocks != metadata.payload.group.height.div_ceil(8) as usize
        {
            return Err(Error::InvalidCodestream("invalid VarDCT spatial grid"));
        }
        copy_vardct_spatial_group_to_image(grid, metadata.payload.group, &mut image)?;
        image.groups_assembled += 1;
    }

    Ok((image.groups_assembled > 0).then_some(image))
}

fn assemble_vardct_xyb_image_final(plan: &VarDctDecodePlan) -> Result<Option<VarDctXybImage>> {
    let sample_len = (plan.frame.width as usize)
        .checked_mul(plan.frame.height as usize)
        .ok_or(Error::InvalidCodestream("VarDCT image is too large"))?;
    let mut image = VarDctXybImage {
        width: plan.frame.width,
        height: plan.frame.height,
        groups_assembled: 0,
        groups_missing: 0,
        channels: [
            vec![0.0; sample_len],
            vec![0.0; sample_len],
            vec![0.0; sample_len],
        ],
    };

    for metadata in final_vardct_ac_passes_by_group(&plan.ac_group_metadata) {
        let spatial = final_vardct_spatial_grid_for_group(plan, metadata)?;
        let Some(grid) = spatial.as_ref().or(metadata.spatial_with_dc_grid.as_ref()) else {
            image.groups_missing += 1;
            continue;
        };
        if grid.group != metadata.payload.group.group
            || grid.width_blocks != metadata.payload.group.width.div_ceil(8) as usize
            || grid.height_blocks != metadata.payload.group.height.div_ceil(8) as usize
        {
            return Err(Error::InvalidCodestream("invalid VarDCT spatial grid"));
        }
        copy_vardct_spatial_group_to_image(grid, metadata.payload.group, &mut image)?;
        image.groups_assembled += 1;
    }

    Ok((image.groups_assembled > 0).then_some(image))
}

fn assemble_vardct_xyb_image_dc_only(
    plan: &VarDctDecodePlan,
    dc_multiplier: f32,
) -> Result<Option<VarDctXybImage>> {
    let sample_len = (plan.frame.width as usize)
        .checked_mul(plan.frame.height as usize)
        .ok_or(Error::InvalidCodestream("VarDCT image is too large"))?;
    let mut image = VarDctXybImage {
        width: plan.frame.width,
        height: plan.frame.height,
        groups_assembled: 0,
        groups_missing: 0,
        channels: [
            vec![0.0; sample_len],
            vec![0.0; sample_len],
            vec![0.0; sample_len],
        ],
    };

    for metadata in final_vardct_ac_passes_by_group(&plan.ac_group_metadata) {
        let Some(grid) = dc_only_spatial_grid_for_group(plan, metadata, dc_multiplier)? else {
            image.groups_missing += 1;
            continue;
        };
        if grid.group != metadata.payload.group.group
            || grid.width_blocks != metadata.payload.group.width.div_ceil(8) as usize
            || grid.height_blocks != metadata.payload.group.height.div_ceil(8) as usize
        {
            return Err(Error::InvalidCodestream("invalid VarDCT spatial grid"));
        }
        copy_vardct_spatial_group_to_image(&grid, metadata.payload.group, &mut image)?;
        image.groups_assembled += 1;
    }

    Ok((image.groups_assembled > 0).then_some(image))
}

fn dc_only_spatial_grid_for_group(
    plan: &VarDctDecodePlan,
    metadata: &VarDctAcGroupMetadata,
    dc_multiplier: f32,
) -> Result<Option<VarDctAcSpatialGrid>> {
    let Some(global) = plan.global.as_ref() else {
        return Ok(None);
    };
    let width_blocks = metadata.payload.group.width.div_ceil(8) as usize;
    let height_blocks = metadata.payload.group.height.div_ceil(8) as usize;
    let coefficient_len = width_blocks
        .checked_mul(height_blocks)
        .and_then(|blocks| blocks.checked_mul(DCT_BLOCK_SIZE))
        .ok_or(Error::InvalidCodestream(
            "AC group coefficient grid is too large",
        ))?;
    let zero_ac = VarDctAcDequantizedGrid {
        group: metadata.payload.group.group,
        pass: metadata.payload.pass,
        width_blocks,
        height_blocks,
        per_channel: [
            VarDctAcDequantizedChannelGrid::new(coefficient_len),
            VarDctAcDequantizedChannelGrid::new(coefficient_len),
            VarDctAcDequantizedChannelGrid::new(coefficient_len),
        ],
    };
    spatialize_vardct_ac_grid_with_dc_multiplier(
        &zero_ac,
        Some(global),
        metadata,
        &plan.dc_group_metadata,
        dc_multiplier,
    )
    .map(Some)
}

fn final_vardct_spatial_grid_for_group(
    plan: &VarDctDecodePlan,
    final_metadata: &VarDctAcGroupMetadata,
) -> Result<Option<VarDctAcSpatialGrid>> {
    let mut passes = plan
        .ac_group_metadata
        .iter()
        .filter(|metadata| {
            metadata.payload.group.group == final_metadata.payload.group.group
                && metadata.payload.pass <= final_metadata.payload.pass
        })
        .collect::<Vec<_>>();
    passes.sort_by_key(|metadata| metadata.payload.pass);
    if passes.len() <= 1 {
        return Ok(None);
    }

    let Some(mut merged) = passes
        .first()
        .and_then(|metadata| metadata.dequantized_grid.clone())
    else {
        return Ok(None);
    };
    for metadata in passes.iter().skip(1) {
        let Some(grid) = metadata.dequantized_grid.as_ref() else {
            return Ok(None);
        };
        merge_vardct_dequantized_grid(&mut merged, grid)?;
    }
    merged.pass = final_metadata.payload.pass;

    spatialize_vardct_ac_grid(
        &merged,
        plan.global.as_ref(),
        final_metadata,
        &plan.dc_group_metadata,
    )
    .map(Some)
}

fn merge_vardct_dequantized_grid(
    merged: &mut VarDctAcDequantizedGrid,
    grid: &VarDctAcDequantizedGrid,
) -> Result<()> {
    if merged.group != grid.group
        || merged.width_blocks != grid.width_blocks
        || merged.height_blocks != grid.height_blocks
    {
        return Err(Error::InvalidCodestream(
            "incompatible progressive VarDCT AC grids",
        ));
    }

    for channel in 0..3 {
        let merged_channel = &mut merged.per_channel[channel];
        let grid_channel = &grid.per_channel[channel];
        if merged_channel.coefficients.len() != grid_channel.coefficients.len() {
            return Err(Error::InvalidCodestream(
                "incompatible progressive VarDCT AC grids",
            ));
        }
        merged_channel.nonzero_coefficients = 0;
        merged_channel.coefficient_checksum = 0;
        for (index, (merged_coeff, coeff)) in merged_channel
            .coefficients
            .iter_mut()
            .zip(&grid_channel.coefficients)
            .enumerate()
        {
            let value = f32::from_bits(*merged_coeff) + f32::from_bits(*coeff);
            *merged_coeff = value.to_bits();
            if value != 0.0 {
                merged_channel.nonzero_coefficients += 1;
                merged_channel.coefficient_checksum = checksum_dequantized_coefficient(
                    merged_channel.coefficient_checksum,
                    index,
                    value,
                );
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum VarDctAssemblyMode {
    #[cfg(test)]
    Final,
    Pass {
        pass: usize,
    },
}

fn vardct_ac_groups_for_assembly(
    groups: &[VarDctAcGroupMetadata],
    mode: VarDctAssemblyMode,
) -> Vec<&VarDctAcGroupMetadata> {
    match mode {
        #[cfg(test)]
        VarDctAssemblyMode::Final => final_vardct_ac_passes_by_group(groups),
        VarDctAssemblyMode::Pass { pass } => vardct_ac_passes_by_group(groups, pass),
    }
}

fn final_vardct_ac_passes_by_group(
    groups: &[VarDctAcGroupMetadata],
) -> Vec<&VarDctAcGroupMetadata> {
    let mut selected: Vec<&VarDctAcGroupMetadata> = Vec::new();
    for metadata in groups {
        match selected
            .iter_mut()
            .find(|existing| existing.payload.group.group == metadata.payload.group.group)
        {
            Some(existing) if metadata.payload.pass >= existing.payload.pass => {
                *existing = metadata;
            }
            Some(_) => {}
            None => selected.push(metadata),
        }
    }
    selected.sort_by_key(|metadata| metadata.payload.group.group);
    selected
}

fn vardct_ac_passes_by_group(
    groups: &[VarDctAcGroupMetadata],
    pass: usize,
) -> Vec<&VarDctAcGroupMetadata> {
    let mut selected = groups
        .iter()
        .filter(|metadata| metadata.payload.pass == pass)
        .collect::<Vec<_>>();
    selected.sort_by_key(|metadata| metadata.payload.group.group);
    selected
}

fn copy_vardct_spatial_group_to_image(
    grid: &VarDctAcSpatialGrid,
    group: VarDctGroupMetadata,
    image: &mut VarDctXybImage,
) -> Result<()> {
    let image_width = image.width as usize;
    let image_height = image.height as usize;
    for block_y in 0..grid.height_blocks {
        for block_x in 0..grid.width_blocks {
            for sample_y in 0..8 {
                for sample_x in 0..8 {
                    let local_x = block_x * 8 + sample_x;
                    let local_y = block_y * 8 + sample_y;
                    if local_x >= group.width as usize || local_y >= group.height as usize {
                        continue;
                    }
                    let image_x = group.x as usize + local_x;
                    let image_y = group.y as usize + local_y;
                    if image_x >= image_width || image_y >= image_height {
                        continue;
                    }
                    let output_index = image_y * image_width + image_x;
                    let sample = sample_y * 8 + sample_x;
                    for channel in 0..3 {
                        image.channels[channel][output_index] = grid
                            .sample(channel, block_x, block_y, sample)
                            .ok_or(Error::InvalidCodestream("invalid VarDCT spatial sample"))?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn apply_vardct_gaborish(image: &mut VarDctXybImage, loop_filter: &LoopFilter) {
    if !loop_filter.gab || image.width == 0 || image.height == 0 {
        return;
    }

    let weights = vardct_gaborish_weights(loop_filter);
    let width = image.width as usize;
    let height = image.height as usize;
    for (channel, [center_weight, cardinal_weight, diagonal_weight]) in
        weights.into_iter().enumerate()
    {
        let input = image.channels[channel].clone();
        for y in 0..height {
            let row = y * width;
            let top = mirror_coordinate(y as isize - 1, height) * width;
            let bottom = mirror_coordinate(y as isize + 1, height) * width;
            for x in 0..width {
                let left = mirror_coordinate(x as isize - 1, width);
                let right = mirror_coordinate(x as isize + 1, width);
                let sum0 = input[row + x];
                let sum1 =
                    input[row + left] + input[row + right] + input[top + x] + input[bottom + x];
                let sum2 = input[top + left]
                    + input[top + right]
                    + input[bottom + left]
                    + input[bottom + right];
                image.channels[channel][row + x] =
                    center_weight * sum0 + cardinal_weight * sum1 + diagonal_weight * sum2;
            }
        }
    }
}

fn vardct_gaborish_weights(loop_filter: &LoopFilter) -> [[f32; 3]; 3] {
    let weights = loop_filter.gab_weights.unwrap_or(DEFAULT_GABORISH_WEIGHTS);
    let mut per_channel = [
        [1.0, weights[0], weights[1]],
        [1.0, weights[2], weights[3]],
        [1.0, weights[4], weights[5]],
    ];
    for weights in &mut per_channel {
        let div = weights[0] + 4.0 * (weights[1] + weights[2]);
        let normalize = 1.0 / div;
        weights[0] *= normalize;
        weights[1] *= normalize;
        weights[2] *= normalize;
    }
    per_channel
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

fn apply_vardct_epf(image: &mut VarDctXybImage, loop_filter: &LoopFilter, epf: &VarDctEpfMetadata) {
    if image.width == 0
        || image.height == 0
        || epf.width_blocks == 0
        || epf.height_blocks == 0
        || epf.inv_sigma.len() < epf.width_blocks * epf.height_blocks
    {
        return;
    }

    let width = image.width as usize;
    let height = image.height as usize;
    let pixels = width * height;
    let mut input = image.channels.clone();
    let mut output = [vec![0.0; pixels], vec![0.0; pixels], vec![0.0; pixels]];
    let pipeline = EpfPipeline {
        width,
        height,
        loop_filter,
        epf,
        channel_scale: effective_epf_channel_scale(loop_filter),
        border_sad_mul: loop_filter.epf_border_sad_mul.unwrap_or(2.0 / 3.0),
    };

    if loop_filter.epf_iters >= 3 {
        pipeline.apply_pass(&input, &mut output, EpfPass::Zero);
        std::mem::swap(&mut input, &mut output);
    }
    if loop_filter.epf_iters >= 1 {
        pipeline.apply_pass(&input, &mut output, EpfPass::One);
        std::mem::swap(&mut input, &mut output);
    }
    if loop_filter.epf_iters >= 2 {
        pipeline.apply_pass(&input, &mut output, EpfPass::Two);
        std::mem::swap(&mut input, &mut output);
    }

    image.channels = input;
}

#[derive(Clone, Copy)]
enum EpfPass {
    Zero,
    One,
    Two,
}

impl EpfPass {
    fn neighbor_offsets(self) -> &'static [(isize, isize)] {
        match self {
            Self::Zero => &[
                (-2, 0),
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -2),
                (0, -1),
                (0, 1),
                (0, 2),
                (1, -1),
                (1, 0),
                (1, 1),
                (2, 0),
            ],
            Self::One | Self::Two => &[(0, -1), (-1, 0), (1, 0), (0, 1)],
        }
    }

    fn sigma_scale(self, loop_filter: &LoopFilter) -> f32 {
        match self {
            Self::Zero => loop_filter.epf_pass0_sigma_scale.unwrap_or(0.9),
            Self::One => 1.0,
            Self::Two => loop_filter.epf_pass2_sigma_scale.unwrap_or(6.5),
        }
    }

    fn sad(self, input: &[Vec<f32>; 3], ctx: EpfSampleContext, dx: isize, dy: isize) -> f32 {
        match self {
            Self::Zero | Self::One => epf_plus_sad(input, ctx, dx, dy),
            Self::Two => epf_pixel_sad(input, ctx, dx, dy),
        }
    }
}

struct EpfPipeline<'a> {
    width: usize,
    height: usize,
    loop_filter: &'a LoopFilter,
    epf: &'a VarDctEpfMetadata,
    channel_scale: [f32; 3],
    border_sad_mul: f32,
}

#[derive(Clone, Copy)]
struct EpfSampleContext {
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    channel_scale: [f32; 3],
}

impl EpfPipeline<'_> {
    fn apply_pass(&self, input: &[Vec<f32>; 3], output: &mut [Vec<f32>; 3], pass: EpfPass) {
        let pass_sigma_scale = pass.sigma_scale(self.loop_filter);
        let offsets = pass.neighbor_offsets();
        for y in 0..self.height {
            let block_y = (y / 8).min(self.epf.height_blocks - 1);
            for x in 0..self.width {
                let output_index = y * self.width + x;
                let block_x = (x / 8).min(self.epf.width_blocks - 1);
                let inv_sigma =
                    f32::from_bits(self.epf.inv_sigma[block_y * self.epf.width_blocks + block_x]);
                if inv_sigma < EPF_MIN_SIGMA {
                    for channel in 0..3 {
                        output[channel][output_index] = input[channel][output_index];
                    }
                    continue;
                }

                let inv_sigma = inv_sigma * self.sad_multiplier(x, y, pass_sigma_scale);
                let ctx = EpfSampleContext {
                    width: self.width,
                    height: self.height,
                    x,
                    y,
                    channel_scale: self.channel_scale,
                };
                let mut weights = [0.0f32; 12];
                let mut neighbor_weight_sum = 0.0;
                match pass {
                    EpfPass::Zero => {
                        for (index, sad) in epf_stage0_directional_sads(input, ctx)
                            .into_iter()
                            .enumerate()
                        {
                            let weight = epf_weight(sad, inv_sigma);
                            weights[index] = weight;
                            neighbor_weight_sum += weight;
                        }
                    }
                    EpfPass::One => {
                        for (index, sad) in epf_stage1_directional_sads(input, ctx)
                            .into_iter()
                            .enumerate()
                        {
                            let weight = epf_weight(sad, inv_sigma);
                            weights[index] = weight;
                            neighbor_weight_sum += weight;
                        }
                    }
                    EpfPass::Two => {
                        for (index, &(dx, dy)) in offsets.iter().enumerate() {
                            let weight = epf_weight(pass.sad(input, ctx, dx, dy), inv_sigma);
                            weights[index] = weight;
                            neighbor_weight_sum += weight;
                        }
                    }
                }
                let weight_sum = 1.0 + neighbor_weight_sum;
                for channel in 0..3 {
                    let mut sample_sum = input[channel][output_index];
                    for (weight, &(dx, dy)) in weights.iter().zip(offsets) {
                        let nx = mirror_coordinate(x as isize + dx, self.width);
                        let ny = mirror_coordinate(y as isize + dy, self.height);
                        sample_sum += *weight * input[channel][ny * self.width + nx];
                    }
                    output[channel][output_index] = sample_sum / weight_sum;
                }
            }
        }
    }

    fn sad_multiplier(&self, x: usize, y: usize, pass_sigma_scale: f32) -> f32 {
        let base = 1.65 * pass_sigma_scale;
        if x % 8 == 0 || x % 8 == 7 || y % 8 == 0 || y % 8 == 7 {
            base * self.border_sad_mul
        } else {
            base
        }
    }
}

fn epf_stage0_directional_sads(channels: &[Vec<f32>; 3], ctx: EpfSampleContext) -> [f32; 12] {
    let mut sads = [0.0; 12];
    for (index, &(dx, dy)) in EpfPass::Zero.neighbor_offsets().iter().enumerate() {
        for channel in 0..3 {
            let mut channel_sad = 0.0;
            for (px, py) in EPF_PLUS_OFFSETS {
                let center = epf_sample(channels, channel, ctx, px, py);
                let neighbor = epf_sample(channels, channel, ctx, dx + px, dy + py);
                channel_sad += (center - neighbor).abs();
            }
            sads[index] += channel_sad * ctx.channel_scale[channel];
        }
    }
    sads
}

fn epf_stage1_directional_sads(channels: &[Vec<f32>; 3], ctx: EpfSampleContext) -> [f32; 4] {
    let mut sads = [0.0; 4];
    for channel in 0..3 {
        let p20 = epf_sample(channels, channel, ctx, 0, -2);
        let p21 = epf_sample(channels, channel, ctx, 0, -1);
        let p11 = epf_sample(channels, channel, ctx, -1, -1);
        let p31 = epf_sample(channels, channel, ctx, 1, -1);
        let p02 = epf_sample(channels, channel, ctx, -2, 0);
        let p12 = epf_sample(channels, channel, ctx, -1, 0);
        let p22 = epf_sample(channels, channel, ctx, 0, 0);
        let p32 = epf_sample(channels, channel, ctx, 1, 0);
        let p42 = epf_sample(channels, channel, ctx, 2, 0);
        let p13 = epf_sample(channels, channel, ctx, -1, 1);
        let p23 = epf_sample(channels, channel, ctx, 0, 1);
        let p33 = epf_sample(channels, channel, ctx, 1, 1);
        let p24 = epf_sample(channels, channel, ctx, 0, 2);
        let scale = ctx.channel_scale[channel];

        sads[0] += scale
            * ((p20 - p21).abs()
                + (p11 - p12).abs()
                + (p22 - p21).abs()
                + (p31 - p32).abs()
                + (p22 - p23).abs());
        sads[1] += scale
            * ((p11 - p21).abs()
                + (p02 - p12).abs()
                + (p12 - p22).abs()
                + (p22 - p32).abs()
                + (p13 - p23).abs());
        sads[2] += scale
            * ((p31 - p21).abs()
                + (p12 - p22).abs()
                + (p22 - p32).abs()
                + (p42 - p32).abs()
                + (p33 - p23).abs());
        sads[3] += scale
            * ((p22 - p21).abs()
                + (p13 - p12).abs()
                + (p22 - p23).abs()
                + (p33 - p32).abs()
                + (p24 - p23).abs());
    }
    sads
}

fn epf_plus_sad(channels: &[Vec<f32>; 3], ctx: EpfSampleContext, dx: isize, dy: isize) -> f32 {
    let mut sad = 0.0;
    for channel in 0..3 {
        let mut channel_sad = 0.0;
        for (px, py) in EPF_PLUS_OFFSETS {
            let ax = mirror_coordinate(ctx.x as isize + px, ctx.width);
            let ay = mirror_coordinate(ctx.y as isize + py, ctx.height);
            let bx = mirror_coordinate(ctx.x as isize + dx + px, ctx.width);
            let by = mirror_coordinate(ctx.y as isize + dy + py, ctx.height);
            channel_sad += (channels[channel][ay * ctx.width + ax]
                - channels[channel][by * ctx.width + bx])
                .abs();
        }
        sad += channel_sad * ctx.channel_scale[channel];
    }
    sad
}

fn epf_sample(
    channels: &[Vec<f32>; 3],
    channel: usize,
    ctx: EpfSampleContext,
    dx: isize,
    dy: isize,
) -> f32 {
    let x = mirror_coordinate(ctx.x as isize + dx, ctx.width);
    let y = mirror_coordinate(ctx.y as isize + dy, ctx.height);
    channels[channel][y * ctx.width + x]
}

fn epf_pixel_sad(channels: &[Vec<f32>; 3], ctx: EpfSampleContext, dx: isize, dy: isize) -> f32 {
    let center_index = ctx.y * ctx.width + ctx.x;
    let neighbor_x = mirror_coordinate(ctx.x as isize + dx, ctx.width);
    let neighbor_y = mirror_coordinate(ctx.y as isize + dy, ctx.height);
    let neighbor_index = neighbor_y * ctx.width + neighbor_x;
    let mut sad = 0.0;
    for channel in 0..3 {
        sad += (channels[channel][center_index] - channels[channel][neighbor_index]).abs()
            * ctx.channel_scale[channel];
    }
    sad
}

fn epf_weight(sad: f32, inv_sigma: f32) -> f32 {
    (sad * inv_sigma + 1.0).max(0.0)
}

fn effective_epf_channel_scale(loop_filter: &LoopFilter) -> [f32; 3] {
    loop_filter.epf_channel_scale.unwrap_or([40.0, 5.0, 3.5])
}

fn vardct_opsin_params(
    metadata: &ImageMetadata,
    transform_data: &CustomTransformData,
) -> VarDctOpsinParams {
    let matrix = transform_data
        .opsin_inverse_matrix
        .as_ref()
        .map(|opsin| opsin.inverse_matrix)
        .unwrap_or(DEFAULT_INVERSE_OPSIN_MATRIX);
    let opsin_biases = transform_data
        .opsin_inverse_matrix
        .as_ref()
        .map(|opsin| opsin.opsin_biases)
        .unwrap_or(DEFAULT_OPSIN_BIASES);
    vardct_opsin_params_from_matrix(matrix, opsin_biases, metadata.tone_mapping.intensity_target)
}

fn vardct_opsin_params_from_matrix(
    mut inverse_matrix: [[f32; 3]; 3],
    opsin_biases: [f32; 3],
    intensity_target: f32,
) -> VarDctOpsinParams {
    let intensity_scale = 255.0 / intensity_target;
    for row in &mut inverse_matrix {
        for value in row {
            *value *= intensity_scale;
        }
    }
    let opsin_biases_cbrt = opsin_biases.map(f32::cbrt);
    VarDctOpsinParams {
        inverse_matrix,
        opsin_biases,
        opsin_biases_cbrt,
    }
}

fn vardct_xyb_to_linear_rgb(xyb: &VarDctXybImage, opsin: &VarDctOpsinParams) -> VarDctRgbImage {
    vardct_xyb_to_linear_rgb_with_variant(xyb, opsin, VarDctXybInverseVariant::NegBMinusBias)
}

fn vardct_xyb_to_linear_rgb_with_variant(
    xyb: &VarDctXybImage,
    opsin: &VarDctOpsinParams,
    variant: VarDctXybInverseVariant,
) -> VarDctRgbImage {
    let mut rgb = VarDctRgbImage {
        width: xyb.width,
        height: xyb.height,
        channels: [
            vec![0.0; xyb.channels[0].len()],
            vec![0.0; xyb.channels[1].len()],
            vec![0.0; xyb.channels[2].len()],
        ],
    };

    for index in 0..xyb.channels[0].len() {
        let [r, g, b] = xyb_sample_to_linear_rgb_with_variant(
            xyb.channels[0][index],
            xyb.channels[1][index],
            xyb.channels[2][index],
            opsin,
            variant,
        );
        rgb.channels[0][index] = r;
        rgb.channels[1][index] = g;
        rgb.channels[2][index] = b;
    }

    rgb
}

fn vardct_linear_rgb_to_srgb8(rgb: &VarDctRgbImage) -> VarDctSrgb8Image {
    let sample_count = rgb.channels[0].len();
    let mut pixels = Vec::with_capacity(sample_count * 3);
    for index in 0..sample_count {
        pixels.push(linear_sample_to_srgb8(rgb.channels[0][index]));
        pixels.push(linear_sample_to_srgb8(rgb.channels[1][index]));
        pixels.push(linear_sample_to_srgb8(rgb.channels[2][index]));
    }

    VarDctSrgb8Image {
        width: rgb.width,
        height: rgb.height,
        pixels,
    }
}

fn linear_sample_to_srgb8(sample: f32) -> u8 {
    linear_sample_to_srgb(sample, u8::MAX as f32) as u8
}

fn vardct_linear_rgb_to_srgb16(rgb: &VarDctRgbImage) -> VarDctSrgb16Image {
    let sample_count = rgb.channels[0].len();
    let mut pixels = Vec::with_capacity(sample_count * 3);
    for index in 0..sample_count {
        pixels.push(linear_sample_to_srgb16(rgb.channels[0][index]));
        pixels.push(linear_sample_to_srgb16(rgb.channels[1][index]));
        pixels.push(linear_sample_to_srgb16(rgb.channels[2][index]));
    }

    VarDctSrgb16Image {
        width: rgb.width,
        height: rgb.height,
        pixels,
    }
}

fn linear_sample_to_srgb16(sample: f32) -> u16 {
    linear_sample_to_srgb(sample, u16::MAX as f32) as u16
}

fn linear_sample_to_srgb(sample: f32, max: f32) -> u32 {
    let sample = sample.clamp(0.0, 1.0);
    let encoded = if sample <= 0.003_130_8 {
        12.92 * sample
    } else {
        1.055 * sample.powf(1.0 / 2.4) - 0.055
    };
    encoded.mul_add(max, 0.0).round().clamp(0.0, max) as u32
}

#[cfg(test)]
fn xyb_sample_to_linear_rgb(x: f32, y: f32, b: f32, opsin: &VarDctOpsinParams) -> [f32; 3] {
    xyb_sample_to_linear_rgb_with_variant(x, y, b, opsin, VarDctXybInverseVariant::NegBMinusBias)
}

fn xyb_sample_to_linear_rgb_with_variant(
    x: f32,
    y: f32,
    b: f32,
    opsin: &VarDctOpsinParams,
    variant: VarDctXybInverseVariant,
) -> [f32; 3] {
    let gamma_r = y + x - opsin.opsin_biases_cbrt[0];
    let gamma_g = y - x - opsin.opsin_biases_cbrt[1];
    let gamma_b = match variant {
        VarDctXybInverseVariant::BMinusBias => b - opsin.opsin_biases_cbrt[2],
        VarDctXybInverseVariant::BPlusBias => b + opsin.opsin_biases_cbrt[2],
        VarDctXybInverseVariant::NegBMinusBias => -b - opsin.opsin_biases_cbrt[2],
        VarDctXybInverseVariant::NegBPlusBias => -b + opsin.opsin_biases_cbrt[2],
    };
    let mixed = [
        gamma_r * gamma_r * gamma_r + opsin.opsin_biases[0],
        gamma_g * gamma_g * gamma_g + opsin.opsin_biases[1],
        gamma_b * gamma_b * gamma_b + opsin.opsin_biases[2],
    ];

    [
        opsin.inverse_matrix[0][0] * mixed[0]
            + opsin.inverse_matrix[0][1] * mixed[1]
            + opsin.inverse_matrix[0][2] * mixed[2],
        opsin.inverse_matrix[1][0] * mixed[0]
            + opsin.inverse_matrix[1][1] * mixed[1]
            + opsin.inverse_matrix[1][2] * mixed[2],
        opsin.inverse_matrix[2][0] * mixed[0]
            + opsin.inverse_matrix[2][1] * mixed[1]
            + opsin.inverse_matrix[2][2] * mixed[2],
    ]
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctAcBlockSummary {
    pub block_x: usize,
    pub block_y: usize,
    pub raw_strategy: usize,
    pub order: usize,
    pub nzeros: u32,
    pub events: usize,
    pub start_bits: usize,
    pub end_bits: usize,
    pub checksum: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VarDctGlobalMetadata {
    pub section: VarDctSectionPayloadMetadata,
    pub cursor: VarDctGlobalCursorMetadata,
    pub dc_dequant: VarDctDcDequantMetadata,
    pub quantizer: VarDctQuantizerMetadata,
    pub block_context_map: VarDctBlockContextMapMetadata,
    pub color_correlation: VarDctColorCorrelationMetadata,
    pub bits_consumed: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VarDctGlobalCursorMetadata {
    pub dc_dequant_default_end_bits: usize,
    pub dc_dequant_end_bits: usize,
    pub quantizer_global_scale_end_bits: usize,
    pub quantizer_quant_dc_end_bits: usize,
    pub quantizer_end_bits: usize,
    pub block_context_default_end_bits: usize,
    pub block_context_dc_thresholds_end_bits: usize,
    pub block_context_qf_thresholds_end_bits: usize,
    pub block_context_map_start_bits: Option<usize>,
    pub block_context_map_end_bits: Option<usize>,
    pub block_context_end_bits: usize,
    pub color_correlation_default_end_bits: usize,
    pub color_correlation_color_factor_end_bits: Option<usize>,
    pub color_correlation_base_x_end_bits: Option<usize>,
    pub color_correlation_base_b_end_bits: Option<usize>,
    pub color_correlation_ytox_dc_end_bits: Option<usize>,
    pub color_correlation_ytob_dc_end_bits: Option<usize>,
    pub color_correlation_end_bits: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VarDctDcDequantMetadata {
    pub all_default: bool,
    pub coefficients: Option<[f32; 3]>,
}

#[derive(Debug, Clone, PartialEq)]
struct VarDctDcDequantRead {
    metadata: VarDctDcDequantMetadata,
    default_end_bits: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VarDctQuantizerMetadata {
    pub global_scale: u32,
    pub quant_dc: u32,
    pub scale: f32,
    pub inv_global_scale: f32,
    pub inv_quant_dc: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct VarDctQuantizerRead {
    metadata: VarDctQuantizerMetadata,
    global_scale_end_bits: usize,
    quant_dc_end_bits: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctBlockContextMapMetadata {
    pub all_default: bool,
    pub dc_thresholds: [Vec<i32>; 3],
    pub qf_thresholds: Vec<u32>,
    pub context_map_size: usize,
    pub num_contexts: usize,
    pub num_dc_contexts: usize,
    pub context_map_probe: Option<VarDctContextMapProbe>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VarDctBlockContextMapRead {
    metadata: VarDctBlockContextMapMetadata,
    default_end_bits: usize,
    dc_thresholds_end_bits: usize,
    qf_thresholds_end_bits: usize,
    context_map_start_bits: Option<usize>,
    context_map_end_bits: Option<usize>,
    context_map_probe: Option<VarDctContextMapProbe>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VarDctColorCorrelationMetadata {
    pub all_default: bool,
    pub color_factor: u32,
    pub base_correlation_x: f32,
    pub base_correlation_b: f32,
    pub ytox_dc: i32,
    pub ytob_dc: i32,
}

#[derive(Debug, Clone, PartialEq)]
struct VarDctColorCorrelationRead {
    metadata: VarDctColorCorrelationMetadata,
    default_end_bits: usize,
    color_factor_end_bits: Option<usize>,
    base_x_end_bits: Option<usize>,
    base_b_end_bits: Option<usize>,
    ytox_dc_end_bits: Option<usize>,
    ytob_dc_end_bits: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VarDctSectionBuckets {
    is_combined: bool,
    global_section: Option<VarDctSectionMetadata>,
    ac_global_section: Option<VarDctSectionMetadata>,
    ac_group_sections: Vec<VarDctPassGroupSectionMetadata>,
    dc_group_sections: Vec<VarDctGroupSectionMetadata>,
}

impl VarDctFrameMetadata {
    pub fn ac_groups_intersecting_region(&self, region: ImageRegion) -> Vec<usize> {
        self.ac_groups
            .iter()
            .filter(|group| group_intersects_region(group, region))
            .map(|group| group.group)
            .collect()
    }

    pub fn ac_sections_for_region(
        &self,
        region: ImageRegion,
    ) -> Vec<&VarDctPassGroupSectionMetadata> {
        if self.is_combined {
            return Vec::new();
        }
        self.ac_group_sections
            .iter()
            .filter(|section| group_intersects_region(&section.group, region))
            .collect()
    }

    pub fn dc_sections_for_region(&self, region: ImageRegion) -> Vec<&VarDctGroupSectionMetadata> {
        if self.is_combined {
            return Vec::new();
        }
        self.dc_group_sections
            .iter()
            .filter(|section| group_intersects_region(&section.group, region))
            .collect()
    }
}

pub fn read_vardct_frame_metadata(
    frame_header: &FrameHeader,
    frame_data: &FrameData,
) -> Option<VarDctFrameMetadata> {
    if frame_header.encoding != FrameEncoding::VarDct {
        return None;
    }

    let sections = frame_data
        .sections
        .iter()
        .map(|section| VarDctSectionMetadata {
            section_logical_id: section.logical_id,
            section_physical_index: section.physical_index,
            section_kind: section.kind,
            codestream_offset: section.codestream_offset,
            payload_size: section.size,
        })
        .collect::<Vec<_>>();
    let ac_groups = group_metadata(
        frame_header.group_layout.groups_x,
        frame_header.group_layout.groups_y,
        frame_header.group_layout.group_dim,
        frame_header.frame_size.width,
        frame_header.frame_size.height,
    );
    let dc_groups = group_metadata(
        frame_header.group_layout.dc_groups_x,
        frame_header.group_layout.dc_groups_y,
        frame_header.group_layout.dc_group_dim,
        frame_header.frame_size.width,
        frame_header.frame_size.height,
    );
    let buckets = classify_vardct_sections(&sections, &ac_groups, &dc_groups);

    Some(VarDctFrameMetadata {
        width: frame_header.frame_size.width,
        height: frame_header.frame_size.height,
        group_dim: frame_header.group_layout.group_dim,
        groups_x: frame_header.group_layout.groups_x,
        groups_y: frame_header.group_layout.groups_y,
        dc_groups_x: frame_header.group_layout.dc_groups_x,
        dc_groups_y: frame_header.group_layout.dc_groups_y,
        is_combined: buckets.is_combined,
        global_section: buckets.global_section,
        ac_global_section: buckets.ac_global_section,
        sections,
        ac_groups,
        dc_groups,
        ac_group_sections: buckets.ac_group_sections,
        dc_group_sections: buckets.dc_group_sections,
    })
}

pub fn read_vardct_decode_plan(
    codestream: &[u8],
    metadata: &ImageMetadata,
    transform_data: &CustomTransformData,
    frame_header: &FrameHeader,
    frame_data: &FrameData,
) -> Result<Option<VarDctDecodePlan>> {
    let Some(frame) = read_vardct_frame_metadata(frame_header, frame_data) else {
        return Ok(None);
    };

    let global_payload = frame
        .global_section
        .as_ref()
        .map(|section| section_payload_metadata(codestream, frame_data, section))
        .transpose()?;
    let global = global_payload
        .as_ref()
        .map(|section| read_vardct_global_metadata(codestream, section))
        .transpose()?;
    let (
        global_tree,
        modular_global_tree_direct_start_bits,
        modular_global_tree_direct_tree_end_bits,
        modular_global_tree_direct_tree_node_count,
        modular_global_tree_direct_tree_leaf_count,
        modular_global_tree_direct_tree_leaves,
        modular_global_tree_direct_error_bits,
        modular_global_tree_direct_residual_context_count,
        modular_global_tree_direct_residual_histogram_count,
        modular_global_tree_direct_residual_context_map_entries,
        modular_global_tree_direct_residual_context_map_raw_entries,
        modular_global_tree_direct_residual_context_map_distinct_entries,
        modular_global_tree_direct_residual_context_map_histogram_usage_counts,
        modular_global_tree_direct_residual_context_map_max_entry,
        modular_global_tree_direct_residual_context_map_symbol_entries,
        modular_global_tree_direct_residual_lz77_end_bits,
        modular_global_tree_direct_residual_context_map_end_bits,
        modular_global_tree_direct_residual_entropy_mode_end_bits,
        modular_global_tree_direct_residual_log_alpha_size_end_bits,
        modular_global_tree_direct_residual_uint_config_end_bits_by_histogram,
        modular_global_tree_direct_residual_uint_config_end_bits,
        modular_global_tree_direct_residual_use_prefix_code,
        modular_global_tree_direct_residual_log_alpha_size,
        modular_global_tree_direct_residual_failed_histogram_index,
        modular_global_tree_direct_residual_error_stage,
        modular_global_tree_direct_residual_ans_histograms,
        modular_global_tree_start_bits,
        modular_global_tree_direct_error,
        modular_global_tree_error,
    ) = match (&global_payload, &global) {
        (Some(payload), Some(global)) => match read_vardct_modular_global_tree(
            codestream,
            metadata,
            frame_header,
            payload,
            global,
        ) {
            Ok(result) => (
                Some(result.tree),
                Some(result.direct_start_bits),
                result.direct_tree_end_bits,
                result.direct_tree_node_count,
                result.direct_tree_leaf_count,
                result.direct_tree_leaves,
                result.direct_error_bits,
                result.direct_residual_context_count,
                result.direct_residual_histogram_count,
                result.direct_residual_context_map_entries,
                result.direct_residual_context_map_raw_entries,
                result.direct_residual_context_map_distinct_entries,
                result.direct_residual_context_map_histogram_usage_counts,
                result.direct_residual_context_map_max_entry,
                result.direct_residual_context_map_symbol_entries,
                result.direct_residual_lz77_end_bits,
                result.direct_residual_context_map_end_bits,
                result.direct_residual_entropy_mode_end_bits,
                result.direct_residual_log_alpha_size_end_bits,
                result.direct_residual_uint_config_end_bits_by_histogram,
                result.direct_residual_uint_config_end_bits,
                result.direct_residual_use_prefix_code,
                result.direct_residual_log_alpha_size,
                result.direct_residual_failed_histogram_index,
                result.direct_residual_error_stage,
                result.direct_residual_ans_histograms,
                Some(result.tree_start_bits),
                result.direct_error,
                None,
            ),
            Err(error) => (
                None,
                Some(global.bits_consumed),
                None,
                None,
                None,
                Vec::new(),
                None,
                None,
                None,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                None,
                Vec::new(),
                None,
                None,
                None,
                None,
                Vec::new(),
                None,
                None,
                None,
                None,
                None,
                Vec::new(),
                None,
                None,
                Some(error),
            ),
        },
        _ => (
            None,
            None,
            None,
            None,
            None,
            Vec::new(),
            None,
            None,
            None,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            None,
            Vec::new(),
            None,
            None,
            None,
            None,
            Vec::new(),
            None,
            None,
            None,
            None,
            None,
            Vec::new(),
            None,
            None,
            None,
        ),
    };
    let dc_group_payloads = frame
        .dc_group_sections
        .iter()
        .map(|section| {
            Ok(VarDctDcGroupPayloadMetadata {
                section: section_payload_metadata(codestream, frame_data, &section.section)?,
                group: section.group,
                var_dct_dc_stream_id: 1 + section.group.group,
                modular_dc_stream_id: 1 + frame.dc_groups.len() + section.group.group,
                ac_metadata_stream_id: 1 + 2 * frame.dc_groups.len() + section.group.group,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let dc_group_metadata = dc_group_payloads
        .iter()
        .cloned()
        .map(|payload| {
            read_vardct_dc_group_metadata(
                codestream,
                frame_header,
                payload,
                global_tree.as_ref(),
                modular_global_tree_error.as_ref(),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let ac_global_payload = frame
        .ac_global_section
        .as_ref()
        .map(|section| section_payload_metadata(codestream, frame_data, section))
        .transpose()?;
    let used_acs = used_acs_from_dc_group_metadata(&dc_group_metadata);
    let ac_global_metadata = ac_global_payload
        .as_ref()
        .zip(global.as_ref())
        .map(|(payload, global)| {
            read_vardct_ac_global_metadata(codestream, frame_header, payload, global, used_acs)
        })
        .transpose()?;
    let ac_global_entropy = ac_global_payload
        .as_ref()
        .zip(global.as_ref())
        .and_then(|(payload, global)| {
            ac_global_metadata
                .as_ref()
                .and_then(|metadata| metadata.parse_error.is_none().then_some((payload, global)))
        })
        .map(|(payload, global)| {
            read_vardct_ac_global_entropy_tables(codestream, frame_header, payload, global)
        })
        .transpose()?;
    let ac_group_payloads = frame
        .ac_group_sections
        .iter()
        .map(|section| {
            let (modular_ac_min_shift, modular_ac_max_shift) =
                frame_header.passes.downsampling_bracket(section.pass)?;
            Ok(VarDctPassGroupPayloadMetadata {
                section: section_payload_metadata(codestream, frame_data, &section.section)?,
                pass: section.pass,
                group: section.group,
                modular_ac_stream_id: 1
                    + 3 * frame.dc_groups.len()
                    + NUM_QUANT_TABLES
                    + frame.ac_groups.len() * section.pass
                    + section.group.group,
                modular_ac_min_shift,
                modular_ac_max_shift,
                modular_ac_channels: vardct_modular_ac_channel_plan(
                    metadata,
                    frame_header,
                    section.group,
                    modular_ac_min_shift,
                    modular_ac_max_shift,
                )?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let ac_group_metadata = ac_group_payloads
        .iter()
        .cloned()
        .map(|payload| {
            read_vardct_ac_group_metadata(
                codestream,
                frame_header,
                payload,
                global.as_ref(),
                ac_global_metadata.as_ref(),
                ac_global_entropy.as_deref(),
                &dc_group_metadata,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let epf_metadata = (frame_header.loop_filter.epf_iters > 0)
        .then(|| vardct_epf_metadata(frame_header, &frame, global.as_ref(), &dc_group_metadata))
        .transpose()?;
    let modular_global_tree_payload_start_bits = global_payload
        .as_ref()
        .and_then(|payload| payload.payload_range.start.checked_mul(8));
    let modular_global_tree_payload_len_bits = global_payload
        .as_ref()
        .and_then(|payload| payload.payload_range.len().checked_mul(8));
    let modular_global_tree_payload_end_bits = modular_global_tree_payload_start_bits
        .zip(modular_global_tree_payload_len_bits)
        .and_then(|(start, len)| start.checked_add(len));
    let absolute_bits = |relative_bits: Option<usize>| {
        modular_global_tree_payload_start_bits
            .zip(relative_bits)
            .and_then(|(start, bits)| start.checked_add(bits))
    };
    let remaining_bits = |relative_bits: Option<usize>| {
        modular_global_tree_payload_len_bits
            .zip(relative_bits)
            .and_then(|(len, bits)| len.checked_sub(bits))
    };

    Ok(Some(VarDctDecodePlan {
        frame,
        loop_filter: frame_header.loop_filter.clone(),
        opsin_params: vardct_opsin_params(metadata, transform_data),
        epf_metadata,
        global,
        modular_global_tree_payload_start_bits,
        modular_global_tree_payload_end_bits,
        modular_global_tree_payload_len_bits,
        modular_global_tree_direct_start_bits,
        modular_global_tree_direct_start_absolute_bits: absolute_bits(
            modular_global_tree_direct_start_bits,
        ),
        modular_global_tree_direct_start_remaining_bits: remaining_bits(
            modular_global_tree_direct_start_bits,
        ),
        modular_global_tree_direct_tree_end_bits,
        modular_global_tree_direct_tree_end_absolute_bits: absolute_bits(
            modular_global_tree_direct_tree_end_bits,
        ),
        modular_global_tree_direct_tree_end_remaining_bits: remaining_bits(
            modular_global_tree_direct_tree_end_bits,
        ),
        modular_global_tree_direct_tree_node_count,
        modular_global_tree_direct_tree_leaf_count,
        modular_global_tree_direct_tree_leaves,
        modular_global_tree_direct_error_bits,
        modular_global_tree_direct_error_absolute_bits: absolute_bits(
            modular_global_tree_direct_error_bits,
        ),
        modular_global_tree_direct_error_remaining_bits: remaining_bits(
            modular_global_tree_direct_error_bits,
        ),
        modular_global_tree_direct_residual_context_count,
        modular_global_tree_direct_residual_histogram_count,
        modular_global_tree_direct_residual_context_map_entries,
        modular_global_tree_direct_residual_context_map_raw_entries,
        modular_global_tree_direct_residual_context_map_distinct_entries,
        modular_global_tree_direct_residual_context_map_histogram_usage_counts,
        modular_global_tree_direct_residual_context_map_max_entry,
        modular_global_tree_direct_residual_context_map_symbol_entries,
        modular_global_tree_direct_residual_lz77_end_bits,
        modular_global_tree_direct_residual_context_map_end_bits,
        modular_global_tree_direct_residual_entropy_mode_end_bits,
        modular_global_tree_direct_residual_log_alpha_size_end_bits,
        modular_global_tree_direct_residual_uint_config_end_bits_by_histogram,
        modular_global_tree_direct_residual_uint_config_end_bits,
        modular_global_tree_direct_residual_use_prefix_code,
        modular_global_tree_direct_residual_log_alpha_size,
        modular_global_tree_direct_residual_failed_histogram_index,
        modular_global_tree_direct_residual_error_stage,
        modular_global_tree_direct_residual_ans_histograms,
        modular_global_tree_start_bits,
        modular_global_tree_start_absolute_bits: absolute_bits(modular_global_tree_start_bits),
        modular_global_tree_start_remaining_bits: remaining_bits(modular_global_tree_start_bits),
        modular_global_tree_direct_error,
        modular_global_tree_error,
        global_payload,
        ac_global_payload,
        ac_global_metadata,
        ac_group_payloads,
        ac_group_metadata,
        dc_group_payloads,
        dc_group_metadata,
    }))
}

#[derive(Debug, Clone, Copy)]
struct VarDctEpfFirstBlock {
    x: usize,
    y: usize,
    raw_strategy: usize,
    quant: i32,
}

fn vardct_epf_metadata(
    frame_header: &FrameHeader,
    frame: &VarDctFrameMetadata,
    global: Option<&VarDctGlobalMetadata>,
    dc_groups: &[VarDctDcGroupMetadata],
) -> Result<VarDctEpfMetadata> {
    let width_blocks = frame.width.div_ceil(8) as usize;
    let height_blocks = frame.height.div_ceil(8) as usize;
    let sample_count = width_blocks
        .checked_mul(height_blocks)
        .ok_or(Error::InvalidCodestream("VarDCT EPF metadata is too large"))?;
    let mut metadata = VarDctEpfMetadata {
        width_blocks,
        height_blocks,
        raw_quant_field: vec![0; sample_count],
        epf_sharpness: vec![0; sample_count],
        inv_sigma: vec![0; sample_count],
        first_block_count: 0,
        raw_quant_checksum: 0,
        epf_sharpness_checksum: 0,
        inv_sigma_checksum: 0,
        parse_error: None,
    };

    let Some(global) = global else {
        metadata.parse_error = Some(Error::Unsupported("VarDCT EPF metadata"));
        return Ok(metadata);
    };

    match fill_vardct_epf_metadata(frame_header, global, dc_groups, &mut metadata) {
        Ok(()) => {}
        Err(error) => metadata.parse_error = Some(error),
    }
    metadata.raw_quant_checksum = checksum_i32_samples(&metadata.raw_quant_field);
    metadata.epf_sharpness_checksum = checksum_u8_samples(&metadata.epf_sharpness);
    metadata.inv_sigma_checksum = checksum_u32_samples(&metadata.inv_sigma);
    Ok(metadata)
}

fn fill_vardct_epf_metadata(
    frame_header: &FrameHeader,
    global: &VarDctGlobalMetadata,
    dc_groups: &[VarDctDcGroupMetadata],
    metadata: &mut VarDctEpfMetadata,
) -> Result<()> {
    let mut first_blocks = Vec::new();
    for dc_group in dc_groups {
        collect_vardct_epf_fields_for_dc_group(dc_group, metadata, &mut first_blocks)?;
    }
    metadata.first_block_count = first_blocks.len();

    let sharp_lut = effective_epf_sharp_lut(&frame_header.loop_filter);
    let epf_quant_mul = frame_header.loop_filter.epf_quant_mul.unwrap_or(0.46);
    let quant_scale = global.quantizer.global_scale as f32 / GLOBAL_SCALE_DENOMINATOR;
    if quant_scale <= 0.0 {
        return Err(Error::InvalidCodestream("invalid VarDCT quantizer"));
    }

    for block in first_blocks {
        let block_x = *STRATEGY_BLOCKS_X
            .get(block.raw_strategy)
            .ok_or(Error::InvalidCodestream("invalid AC strategy"))?;
        let block_y = *STRATEGY_BLOCKS_Y
            .get(block.raw_strategy)
            .ok_or(Error::InvalidCodestream("invalid AC strategy"))?;
        if block.quant <= 0 {
            return Err(Error::InvalidCodestream("invalid AC quant field"));
        }
        let sigma_quant =
            epf_quant_mul / (quant_scale * block.quant as f32 * EPF_INV_SIGMA_NUMERATOR);
        for dy in 0..block_y {
            for dx in 0..block_x {
                let x = block.x + dx;
                let y = block.y + dy;
                if x >= metadata.width_blocks || y >= metadata.height_blocks {
                    continue;
                }
                let index = y * metadata.width_blocks + x;
                let sharpness = metadata.epf_sharpness[index] as usize;
                let Some(&sharpness_scale) = sharp_lut.get(sharpness) else {
                    return Err(Error::InvalidCodestream("invalid EPF sharpness"));
                };
                let sigma = (sigma_quant * sharpness_scale).min(-1.0e-4);
                metadata.inv_sigma[index] = (1.0 / sigma).to_bits();
            }
        }
    }

    Ok(())
}

fn collect_vardct_epf_fields_for_dc_group(
    dc_group: &VarDctDcGroupMetadata,
    metadata: &mut VarDctEpfMetadata,
    first_blocks: &mut Vec<VarDctEpfFirstBlock>,
) -> Result<()> {
    let ac_metadata = dc_group
        .ac_metadata
        .as_ref()
        .ok_or(Error::Unsupported("VarDCT EPF metadata"))?;
    let strategy_channel = ac_metadata
        .channels
        .iter()
        .find(|channel| channel.channel_index == 2)
        .ok_or(Error::Unsupported("VarDCT EPF metadata"))?;
    let sharpness_channel = ac_metadata
        .channels
        .iter()
        .find(|channel| channel.channel_index == 3)
        .ok_or(Error::Unsupported("VarDCT EPF metadata"))?;
    let dc_width_blocks = dc_group.payload.group.width.div_ceil(8) as usize;
    let dc_height_blocks = dc_group.payload.group.height.div_ceil(8) as usize;
    let group_min_x = (dc_group.payload.group.x / 8) as usize;
    let group_min_y = (dc_group.payload.group.y / 8) as usize;
    let count = strategy_channel.width as usize;
    if strategy_channel.height != 2
        || strategy_channel.samples.len() < count * 2
        || sharpness_channel.width as usize != dc_width_blocks
        || sharpness_channel.height as usize != dc_height_blocks
        || sharpness_channel.samples.len() < dc_width_blocks * dc_height_blocks
    {
        return Err(Error::Unsupported("VarDCT EPF metadata"));
    }

    let mut valid = vec![false; dc_width_blocks * dc_height_blocks];
    let mut cursor = 0usize;
    for y in 0..dc_height_blocks {
        for x in 0..dc_width_blocks {
            let local_index = y * dc_width_blocks + x;
            let frame_x = group_min_x + x;
            let frame_y = group_min_y + y;
            if frame_x < metadata.width_blocks && frame_y < metadata.height_blocks {
                let output_index = frame_y * metadata.width_blocks + frame_x;
                let sharpness = *sharpness_channel
                    .samples
                    .get(local_index)
                    .ok_or(Error::InvalidCodestream("invalid EPF sharpness"))?;
                if !(0..8).contains(&sharpness) {
                    return Err(Error::InvalidCodestream("invalid EPF sharpness"));
                }
                metadata.epf_sharpness[output_index] = sharpness as u8;
            }
            if valid[local_index] {
                continue;
            }
            let raw_strategy = *strategy_channel
                .samples
                .get(cursor)
                .ok_or(Error::InvalidCodestream("invalid AC metadata stream"))?
                as usize;
            let block_x = *STRATEGY_BLOCKS_X
                .get(raw_strategy)
                .ok_or(Error::InvalidCodestream("invalid AC strategy"))?;
            let block_y = *STRATEGY_BLOCKS_Y
                .get(raw_strategy)
                .ok_or(Error::InvalidCodestream("invalid AC strategy"))?;
            let quant = 1
                + (*strategy_channel
                    .samples
                    .get(count + cursor)
                    .ok_or(Error::InvalidCodestream("invalid AC quant field"))?)
                .clamp(0, 32_767);
            if frame_x < metadata.width_blocks && frame_y < metadata.height_blocks {
                first_blocks.push(VarDctEpfFirstBlock {
                    x: frame_x,
                    y: frame_y,
                    raw_strategy,
                    quant,
                });
            }
            for dy in 0..block_y {
                for dx in 0..block_x {
                    let covered_x = x + dx;
                    let covered_y = y + dy;
                    if covered_x < dc_width_blocks && covered_y < dc_height_blocks {
                        valid[covered_y * dc_width_blocks + covered_x] = true;
                    }
                    let frame_covered_x = group_min_x + covered_x;
                    let frame_covered_y = group_min_y + covered_y;
                    if frame_covered_x < metadata.width_blocks
                        && frame_covered_y < metadata.height_blocks
                    {
                        metadata.raw_quant_field
                            [frame_covered_y * metadata.width_blocks + frame_covered_x] = quant;
                    }
                }
            }
            cursor += 1;
            if cursor > count {
                return Err(Error::InvalidCodestream("invalid AC metadata stream"));
            }
        }
    }

    Ok(())
}

fn effective_epf_sharp_lut(loop_filter: &LoopFilter) -> [f32; 8] {
    loop_filter.epf_sharp_lut.unwrap_or([
        0.0,
        1.0 / 7.0,
        2.0 / 7.0,
        3.0 / 7.0,
        4.0 / 7.0,
        5.0 / 7.0,
        6.0 / 7.0,
        1.0,
    ])
}

fn checksum_i32_samples(samples: &[i32]) -> u64 {
    samples
        .iter()
        .enumerate()
        .fold(0u64, |checksum, (index, value)| {
            checksum
                .wrapping_mul(1_099_511_628_211)
                .wrapping_add(index as u64)
                .rotate_left(11)
                ^ (*value as u32 as u64)
        })
}

fn checksum_u8_samples(samples: &[u8]) -> u64 {
    samples
        .iter()
        .enumerate()
        .fold(0u64, |checksum, (index, value)| {
            checksum
                .wrapping_mul(1_099_511_628_211)
                .wrapping_add(index as u64)
                .rotate_left(11)
                ^ u64::from(*value)
        })
}

fn checksum_u32_samples(samples: &[u32]) -> u64 {
    samples
        .iter()
        .enumerate()
        .fold(0u64, |checksum, (index, value)| {
            checksum
                .wrapping_mul(1_099_511_628_211)
                .wrapping_add(index as u64)
                .rotate_left(11)
                ^ u64::from(*value)
        })
}

fn read_vardct_ac_group_metadata(
    codestream: &[u8],
    frame_header: &FrameHeader,
    payload: VarDctPassGroupPayloadMetadata,
    global: Option<&VarDctGlobalMetadata>,
    ac_global: Option<&VarDctAcGlobalMetadata>,
    ac_global_entropy: Option<&[Option<VarDctAcPassEntropy>]>,
    dc_groups: &[VarDctDcGroupMetadata],
) -> Result<VarDctAcGroupMetadata> {
    let bytes = codestream
        .get(payload.section.payload_range.clone())
        .ok_or(Error::InvalidCodestream("frame section outside codestream"))?;
    let mut reader = BitReader::new(bytes);
    let payload_end_bits = bytes
        .len()
        .checked_mul(8)
        .ok_or(Error::InvalidCodestream("AC group payload too large"))?;
    let histogram_selector_bits = ac_global
        .and_then(|global| global.num_histograms)
        .map(ceil_log2_nonzero)
        .unwrap_or(0);
    let entropy_uses_prefix_code = ac_global
        .and_then(|global| global.passes.iter().find(|pass| pass.pass == payload.pass))
        .and_then(|pass| pass.use_prefix_code);
    let mut metadata = VarDctAcGroupMetadata {
        payload,
        cursor: VarDctAcGroupCursorMetadata {
            payload_start_bits: 0,
            payload_end_bits,
            histogram_selector_start_bits: 0,
            histogram_selector_end_bits: None,
            ans_state_start_bits: None,
            ans_state_end_bits: None,
            coefficient_stream_start_bits: None,
            modular_ac_start_bits: None,
        },
        histogram_selector_bits,
        histogram_selector: None,
        entropy_uses_prefix_code,
        coefficient_probe: None,
        channel_trace: None,
        coefficient_summary: None,
        coefficient_grid: None,
        base_dequantized_grid: None,
        dequantized_grid: None,
        spatial_grid: None,
        spatial_with_dc_grid: None,
        parse_error: None,
    };

    let histogram_selector = if histogram_selector_bits == 0 {
        0
    } else {
        match reader.read_bits(histogram_selector_bits) {
            Ok(selector) => selector as usize,
            Err(error) => {
                metadata.cursor.histogram_selector_end_bits = Some(reader.bits_consumed());
                metadata.parse_error = Some(error);
                return Ok(metadata);
            }
        }
    };
    metadata.histogram_selector = Some(histogram_selector);
    metadata.cursor.histogram_selector_end_bits = Some(reader.bits_consumed());
    if let Some(num_histograms) = ac_global.and_then(|global| global.num_histograms)
        && histogram_selector >= num_histograms
    {
        metadata.parse_error = Some(Error::InvalidCodestream("invalid histogram selector"));
        return Ok(metadata);
    }

    match entropy_uses_prefix_code {
        Some(false) => {
            metadata.cursor.ans_state_start_bits = Some(reader.bits_consumed());
            match ac_global_entropy
                .and_then(|passes| passes.get(metadata.payload.pass))
                .and_then(Option::as_ref)
            {
                Some(entropy) => match AnsSymbolReader::new(entropy.code.clone(), &mut reader, 0) {
                    Ok(mut symbol_reader) => {
                        metadata.cursor.ans_state_end_bits = Some(reader.bits_consumed());
                        metadata.cursor.coefficient_stream_start_bits =
                            Some(reader.bits_consumed());
                        let probe_result = match trace_vardct_ac_group_channel(
                            &mut reader,
                            &mut symbol_reader,
                            &entropy.context_map,
                            &metadata,
                            global,
                            ac_global,
                            dc_groups,
                        ) {
                            Ok((probe, trace, summary, grid)) => {
                                let base_dequantized_grid = base_dequantize_vardct_ac_grid(
                                    &grid, global, &metadata, dc_groups,
                                )
                                .ok();
                                let dequantized_grid = dequantize_vardct_ac_grid(
                                    &grid,
                                    global,
                                    &metadata,
                                    frame_header,
                                    dc_groups,
                                )
                                .ok();
                                let spatial_grid =
                                    dequantized_grid.as_ref().and_then(|dequantized| {
                                        spatialize_vardct_ac_grid(
                                            dequantized,
                                            None,
                                            &metadata,
                                            dc_groups,
                                        )
                                        .ok()
                                    });
                                let spatial_with_dc_grid =
                                    dequantized_grid.as_ref().and_then(|dequantized| {
                                        global.and_then(|global| {
                                            spatialize_vardct_ac_grid(
                                                dequantized,
                                                Some(global),
                                                &metadata,
                                                dc_groups,
                                            )
                                            .ok()
                                        })
                                    });
                                metadata.channel_trace = Some(trace);
                                metadata.coefficient_summary = Some(summary);
                                metadata.coefficient_grid = Some(grid);
                                metadata.base_dequantized_grid = base_dequantized_grid;
                                metadata.dequantized_grid = dequantized_grid;
                                metadata.spatial_grid = spatial_grid;
                                metadata.spatial_with_dc_grid = spatial_with_dc_grid;
                                Ok(probe)
                            }
                            Err(error) => Err(error),
                        };
                        match probe_result {
                            Ok(probe) => {
                                metadata.cursor.modular_ac_start_bits =
                                    Some(reader.bits_consumed());
                                metadata.coefficient_probe = Some(probe);
                                metadata.parse_error = Some(Error::Unsupported(
                                    "VarDCT AC coefficient stream decoding",
                                ));
                            }
                            Err(error) => {
                                metadata.parse_error = Some(error);
                            }
                        }
                    }
                    Err(error) => {
                        metadata.cursor.ans_state_end_bits = Some(reader.bits_consumed());
                        metadata.parse_error = Some(error);
                    }
                },
                None => match reader.read_bits(32) {
                    Ok(_) => {
                        metadata.cursor.ans_state_end_bits = Some(reader.bits_consumed());
                        metadata.cursor.coefficient_stream_start_bits =
                            Some(reader.bits_consumed());
                        metadata.parse_error =
                            Some(Error::Unsupported("VarDCT AC entropy metadata"));
                    }
                    Err(error) => {
                        metadata.cursor.ans_state_end_bits = Some(reader.bits_consumed());
                        metadata.parse_error = Some(error);
                    }
                },
            };
        }
        Some(true) => {
            metadata.cursor.coefficient_stream_start_bits = Some(reader.bits_consumed());
            metadata.parse_error =
                Some(Error::Unsupported("VarDCT AC coefficient stream decoding"));
        }
        None => {
            metadata.parse_error = Some(Error::Unsupported("VarDCT AC entropy metadata"));
        }
    }

    Ok(metadata)
}

#[derive(Debug, Clone)]
struct VarDctAcPassEntropy {
    code: AnsCode,
    context_map: Vec<u8>,
}

fn read_vardct_ac_global_entropy_tables(
    codestream: &[u8],
    frame_header: &FrameHeader,
    payload: &VarDctSectionPayloadMetadata,
    global: &VarDctGlobalMetadata,
) -> Result<Vec<Option<VarDctAcPassEntropy>>> {
    let bytes = codestream
        .get(payload.payload_range.clone())
        .ok_or(Error::InvalidCodestream("frame section outside codestream"))?;
    let mut reader = BitReader::new(bytes);
    let all_default_quant_matrices = reader.read_bool()?;
    if !all_default_quant_matrices {
        return Err(Error::Unsupported("custom VarDCT AC quant matrices"));
    }
    let num_histo_bits = ceil_log2_nonzero(frame_header.group_layout.num_groups as usize);
    let num_histograms = reader.read_bits(num_histo_bits)? as usize + 1;
    let mut passes = Vec::with_capacity(frame_header.passes.num_passes as usize);
    for _pass in 0..frame_header.passes.num_passes as usize {
        let used_orders =
            reader.read_u32_selector(val(0x5f), val(0x13), val(0), bits_offset(13, 0))?;
        read_vardct_coeff_orders(&mut reader, used_orders as u16).map_err(|error| error.error)?;
        let histogram_contexts =
            num_histograms * global.block_context_map.num_contexts * (37 + 458);
        let (code, context_map) = decode_histograms(&mut reader, histogram_contexts, false)?;
        passes.push(Some(VarDctAcPassEntropy { code, context_map }));
    }
    Ok(passes)
}

fn trace_vardct_ac_group_channel(
    reader: &mut BitReader<'_>,
    symbol_reader: &mut AnsSymbolReader,
    context_map: &[u8],
    metadata: &VarDctAcGroupMetadata,
    global: Option<&VarDctGlobalMetadata>,
    ac_global: Option<&VarDctAcGlobalMetadata>,
    dc_groups: &[VarDctDcGroupMetadata],
) -> Result<(
    VarDctAcCoefficientProbe,
    VarDctAcChannelTrace,
    VarDctAcCoefficientSummary,
    VarDctAcCoefficientGrid,
)> {
    const TRACE_CHANNEL: usize = 1;
    const CHANNEL_ORDER: [usize; 3] = [1, 0, 2];
    const SUMMARY_LIMIT: usize = 8;

    let global = global.ok_or(Error::Unsupported("VarDCT AC global metadata"))?;
    let coeff_orders = ac_global
        .and_then(|global_metadata| {
            global_metadata
                .passes
                .iter()
                .find(|pass| pass.pass == metadata.payload.pass)
        })
        .map(|pass| pass.coeff_orders.as_slice());
    let blocks = vardct_ac_blocks_for_group(metadata, dc_groups)?;
    let width_blocks = metadata.payload.group.width.div_ceil(8) as usize;
    let height_blocks = metadata.payload.group.height.div_ceil(8) as usize;
    let row_len = width_blocks
        .checked_mul(height_blocks)
        .ok_or(Error::InvalidCodestream("AC group is too large"))?;
    let coefficient_len = row_len
        .checked_mul(DCT_BLOCK_SIZE)
        .ok_or(Error::InvalidCodestream(
            "AC group coefficient grid is too large",
        ))?;
    let mut row_nzeros = [
        vec![0i32; row_len],
        vec![0i32; row_len],
        vec![0i32; row_len],
    ];
    let mut coefficient_grid = VarDctAcCoefficientGrid {
        group: metadata.payload.group.group,
        pass: metadata.payload.pass,
        width_blocks,
        height_blocks,
        per_channel: [
            VarDctAcChannelCoefficientGrid::new(coefficient_len),
            VarDctAcChannelCoefficientGrid::new(coefficient_len),
            VarDctAcChannelCoefficientGrid::new(coefficient_len),
        ],
    };
    let mut first_probe = None;
    let mut blocks_decoded = 0usize;
    let mut coefficient_events_decoded = 0usize;
    let mut coefficient_event_checksum = 0xcbf29ce484222325;
    let mut natural_coeff_orders = vec![None; STRATEGY_BLOCKS_X.len()];
    let mut coefficient_summary = VarDctAcCoefficientSummary {
        group: metadata.payload.group.group,
        pass: metadata.payload.pass,
        blocks_decoded: 0,
        final_bits: 0,
        per_channel: [VarDctAcChannelCoefficientSummary::default(); 3],
        first_block_checksum: 0,
    };
    let mut first_block_seen = false;
    let mut block_summaries = Vec::new();

    for block in blocks {
        for channel in CHANNEL_ORDER {
            let predicted_nzeros = predict_from_top_and_left(
                &row_nzeros[channel],
                width_blocks,
                block.block_x,
                block.block_y,
                32,
            );
            let capture_events = channel == TRACE_CHANNEL && first_probe.is_none();
            let probe = decode_vardct_ac_block_probe(
                reader,
                symbol_reader,
                context_map,
                global,
                block,
                channel,
                predicted_nzeros as usize,
                &mut row_nzeros[channel],
                None,
                width_blocks,
                &mut natural_coeff_orders,
                coeff_orders,
                Some(&mut coefficient_grid.per_channel[channel]),
                capture_events,
            )?;
            coefficient_summary.blocks_decoded += 1;
            let channel_summary = &mut coefficient_summary.per_channel[channel];
            channel_summary.blocks_decoded += 1;
            channel_summary.coefficients_written += probe.coefficient_event_count;
            channel_summary.nonzero_coefficients += probe.placed_nonzero_coefficients;
            channel_summary.coefficient_checksum = (channel_summary.coefficient_checksum
                ^ probe.placed_coefficient_checksum)
                .wrapping_mul(0x100000001b3);
            if !first_block_seen {
                coefficient_summary.first_block_checksum = probe.placed_coefficient_checksum;
                first_block_seen = true;
            }
            if channel == TRACE_CHANNEL {
                blocks_decoded += 1;
                coefficient_events_decoded += probe.coefficient_event_count;
                coefficient_event_checksum = (coefficient_event_checksum
                    ^ probe.coefficient_event_checksum)
                    .wrapping_mul(0x100000001b3);
                if block_summaries.len() < SUMMARY_LIMIT {
                    block_summaries.push(VarDctAcBlockSummary {
                        block_x: probe.block_x,
                        block_y: probe.block_y,
                        raw_strategy: probe.raw_strategy,
                        order: probe.order,
                        nzeros: probe.nzeros,
                        events: probe.coefficient_event_count,
                        start_bits: probe.start_bits,
                        end_bits: probe.block_end_bits.unwrap_or(probe.nzeros_end_bits),
                        checksum: probe.coefficient_event_checksum,
                    });
                }
                if first_probe.is_none() {
                    first_probe = Some(probe);
                }
            }
        }
    }

    let row_nzeros_checksum = checksum_i32_slice(&row_nzeros[TRACE_CHANNEL]);
    coefficient_summary.final_bits = reader.bits_consumed();
    let trace = VarDctAcChannelTrace {
        channel: TRACE_CHANNEL,
        blocks_decoded,
        coefficient_events_decoded,
        final_bits: reader.bits_consumed(),
        row_nzeros_checksum,
        coefficient_event_checksum,
        block_summaries,
    };
    let first_probe = first_probe.ok_or(Error::Unsupported("VarDCT AC metadata grid"))?;
    Ok((first_probe, trace, coefficient_summary, coefficient_grid))
}

#[allow(clippy::too_many_arguments)]
fn decode_vardct_ac_block_probe(
    reader: &mut BitReader<'_>,
    symbol_reader: &mut AnsSymbolReader,
    context_map: &[u8],
    global: &VarDctGlobalMetadata,
    block: VarDctFirstAcBlock,
    channel: usize,
    predicted_nzeros: usize,
    row_nzeros: &mut [i32],
    row_nzeros_top: Option<&[i32]>,
    nzeros_stride: usize,
    natural_coeff_orders: &mut [Option<Vec<usize>>],
    coeff_orders: Option<&[VarDctCoeffOrderMetadata]>,
    coefficient_grid: Option<&mut VarDctAcChannelCoefficientGrid>,
    capture_events: bool,
) -> Result<VarDctAcCoefficientProbe> {
    let _ = row_nzeros_top;
    let order = *STRATEGY_ORDER
        .get(block.raw_strategy)
        .ok_or(Error::InvalidCodestream("invalid AC strategy"))?;
    let covered_blocks =
        STRATEGY_BLOCKS_X[block.raw_strategy] * STRATEGY_BLOCKS_Y[block.raw_strategy];
    let block_size = covered_blocks * DCT_BLOCK_SIZE;
    if block_size > FIRST_AC_BLOCK_EVENT_LIMIT + covered_blocks {
        return Err(Error::Unsupported("large VarDCT AC coefficient probe"));
    }
    let block_context = vardct_block_context(&global.block_context_map, order, channel)?;
    let nonzero_context = vardct_nonzero_context(
        predicted_nzeros,
        block_context,
        global.block_context_map.num_contexts,
    );
    let clustered_context = usize::from(
        *context_map
            .get(nonzero_context)
            .ok_or(Error::InvalidCodestream("invalid AC entropy context"))?,
    );
    let start_bits = reader.bits_consumed();
    let nzeros = symbol_reader.read_hybrid_uint_clustered(clustered_context, reader)?;
    let nzeros_end_bits = reader.bits_consumed();
    if nzeros as usize > block_size - covered_blocks {
        return Err(Error::InvalidCodestream("invalid VarDCT AC nzeros"));
    }
    let log2_covered_blocks = covered_blocks.ilog2() as usize;
    let nzeros_for_block = ((nzeros as usize + covered_blocks - 1) >> log2_covered_blocks) as i32;
    let block_width = STRATEGY_BLOCKS_X[block.raw_strategy];
    let block_height = STRATEGY_BLOCKS_Y[block.raw_strategy];
    for y in 0..block_height {
        for x in 0..block_width {
            let index = block.block_x + x + (block.block_y + y) * nzeros_stride;
            if let Some(slot) = row_nzeros.get_mut(index) {
                *slot = nzeros_for_block;
            }
        }
    }

    let zero_density_context_offset = global.block_context_map.num_contexts * NONZERO_BUCKETS
        + ZERO_DENSITY_CONTEXT_COUNT * block_context;
    let mut remaining_nzeros = nzeros as usize;
    let mut prev = if remaining_nzeros > block_size / 16 {
        0
    } else {
        1
    };
    let mut coefficient_events = Vec::new();
    let mut coefficient_event_checksum = 0xcbf29ce484222325;
    let mut event_count = 0usize;
    let mut placed_nonzero_coefficients = 0usize;
    let mut placed_coefficient_checksum = 0xcbf29ce484222325;
    let mut coefficient_grid = coefficient_grid;
    for k in covered_blocks..block_size {
        if remaining_nzeros == 0 {
            break;
        }
        let zero_density_context = zero_density_context(
            remaining_nzeros,
            k,
            covered_blocks,
            log2_covered_blocks,
            prev,
        )?;
        let context = zero_density_context_offset + zero_density_context;
        let clustered_context = usize::from(
            *context_map
                .get(context)
                .ok_or(Error::InvalidCodestream("invalid AC entropy context"))?,
        );
        let start_bits = reader.bits_consumed();
        let u_coeff = symbol_reader.read_hybrid_uint_clustered(clustered_context, reader)?;
        let end_bits = reader.bits_consumed();
        let coeff = unpack_signed(u_coeff);
        let coefficient_position = coefficient_order_position(
            natural_coeff_orders,
            coeff_orders,
            order,
            channel,
            block.raw_strategy,
            k,
        )?;
        if coeff != 0 {
            placed_nonzero_coefficients += 1;
            placed_coefficient_checksum = checksum_placed_coefficient(
                placed_coefficient_checksum,
                coefficient_position,
                coeff,
            );
            if let Some(grid) = coefficient_grid.as_deref_mut() {
                write_vardct_ac_coefficient(
                    grid,
                    block,
                    nzeros_stride,
                    coefficient_position,
                    coeff,
                )?;
            }
        }
        prev = usize::from(u_coeff != 0);
        remaining_nzeros = remaining_nzeros.saturating_sub(prev);
        let event = VarDctAcCoefficientEvent {
            k,
            zero_density_context,
            context,
            clustered_context,
            start_bits,
            end_bits,
            u_coeff,
            coeff,
            prev_after: prev,
            remaining_nzeros,
        };
        coefficient_event_checksum = checksum_coefficient_event(coefficient_event_checksum, &event);
        event_count += 1;
        if capture_events {
            coefficient_events.push(event);
        }
    }
    if remaining_nzeros != 0 {
        return Err(Error::InvalidCodestream(
            "invalid VarDCT AC nzeros at end of block",
        ));
    }
    let block_end_bits = Some(reader.bits_consumed());

    Ok(VarDctAcCoefficientProbe {
        block_x: block.block_x,
        block_y: block.block_y,
        channel,
        raw_strategy: block.raw_strategy,
        order,
        covered_blocks,
        block_size,
        block_context,
        nonzero_context,
        clustered_context,
        start_bits,
        nzeros_end_bits,
        nzeros,
        block_end_bits,
        remaining_nzeros: Some(remaining_nzeros),
        coefficient_event_count: event_count,
        coefficient_events: if capture_events {
            coefficient_events
        } else {
            Vec::with_capacity(event_count)
        },
        coefficient_event_checksum,
        placed_nonzero_coefficients,
        placed_coefficient_checksum,
    })
}

#[derive(Debug, Clone, Copy)]
struct VarDctFirstAcBlock {
    block_x: usize,
    block_y: usize,
    raw_strategy: usize,
}

fn vardct_ac_blocks_for_group(
    metadata: &VarDctAcGroupMetadata,
    dc_groups: &[VarDctDcGroupMetadata],
) -> Result<Vec<VarDctFirstAcBlock>> {
    let dc_group = dc_groups
        .iter()
        .find(|dc_group| {
            let group = &dc_group.payload.group;
            metadata.payload.group.x >= group.x
                && metadata.payload.group.y >= group.y
                && metadata.payload.group.x < group.x + group.width
                && metadata.payload.group.y < group.y + group.height
        })
        .ok_or(Error::Unsupported("VarDCT AC metadata grid"))?;
    let ac_metadata = dc_group
        .ac_metadata
        .as_ref()
        .ok_or(Error::Unsupported("VarDCT AC metadata grid"))?;
    let strategy_channel = ac_metadata
        .channels
        .iter()
        .find(|channel| channel.channel_index == 2)
        .ok_or(Error::Unsupported("VarDCT AC metadata grid"))?;
    let count = strategy_channel.width as usize;
    if strategy_channel.height != 2 || strategy_channel.samples.len() < count * 2 {
        return Err(Error::Unsupported("VarDCT AC metadata grid"));
    }

    let grid_width = dc_group.payload.group.width.div_ceil(8) as usize;
    let grid_height = dc_group.payload.group.height.div_ceil(8) as usize;
    let group_min_x = ((metadata.payload.group.x - dc_group.payload.group.x) / 8) as usize;
    let group_min_y = ((metadata.payload.group.y - dc_group.payload.group.y) / 8) as usize;
    let group_max_x = group_min_x + metadata.payload.group.width.div_ceil(8).min(256 / 8) as usize;
    let group_max_y = group_min_y + metadata.payload.group.height.div_ceil(8).min(256 / 8) as usize;
    let mut valid = vec![false; grid_width * grid_height];
    let mut cursor = 0usize;
    let mut blocks = Vec::new();
    for y in 0..grid_height {
        for x in 0..grid_width {
            let index = y * grid_width + x;
            if valid[index] {
                continue;
            }
            let raw_strategy = *strategy_channel
                .samples
                .get(cursor)
                .ok_or(Error::InvalidCodestream("invalid AC metadata stream"))?
                as usize;
            let block_x = *STRATEGY_BLOCKS_X
                .get(raw_strategy)
                .ok_or(Error::InvalidCodestream("invalid AC strategy"))?;
            let block_y = *STRATEGY_BLOCKS_Y
                .get(raw_strategy)
                .ok_or(Error::InvalidCodestream("invalid AC strategy"))?;
            if x >= group_min_x && x < group_max_x && y >= group_min_y && y < group_max_y {
                blocks.push(VarDctFirstAcBlock {
                    block_x: x - group_min_x,
                    block_y: y - group_min_y,
                    raw_strategy,
                });
            }
            for dy in 0..block_y {
                for dx in 0..block_x {
                    let covered_x = x + dx;
                    let covered_y = y + dy;
                    if covered_x < grid_width && covered_y < grid_height {
                        valid[covered_y * grid_width + covered_x] = true;
                    }
                }
            }
            cursor += 1;
            if cursor > count {
                return Err(Error::InvalidCodestream("invalid AC metadata stream"));
            }
        }
    }
    Ok(blocks)
}

fn vardct_block_context(
    block_context_map: &VarDctBlockContextMapMetadata,
    order: usize,
    channel: usize,
) -> Result<usize> {
    const DEFAULT_CONTEXT_MAP: [u8; 39] = [
        0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14, 7,
        8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14,
    ];
    if !block_context_map.dc_thresholds.iter().all(Vec::is_empty)
        || !block_context_map.qf_thresholds.is_empty()
    {
        return Err(Error::Unsupported("non-default VarDCT AC block contexts"));
    }
    let mapped_channel = if channel < 2 { channel ^ 1 } else { 2 };
    let index = mapped_channel * STRATEGY_ORDER_BUCKETS + order;
    let context_map = block_context_map
        .context_map_probe
        .as_ref()
        .map(|probe| probe.entries.as_slice())
        .unwrap_or(&DEFAULT_CONTEXT_MAP);
    context_map
        .get(index)
        .copied()
        .map(usize::from)
        .ok_or(Error::InvalidCodestream("invalid AC block context"))
}

fn vardct_nonzero_context(
    predicted_nzeros: usize,
    block_context: usize,
    num_contexts: usize,
) -> usize {
    let clamped = predicted_nzeros.min(64);
    let bucket = if clamped < 8 {
        clamped
    } else {
        4 + clamped / 2
    };
    bucket * num_contexts + block_context
}

fn coefficient_order_position(
    natural_coeff_orders: &mut [Option<Vec<usize>>],
    coeff_orders: Option<&[VarDctCoeffOrderMetadata]>,
    order: usize,
    channel: usize,
    raw_strategy: usize,
    k: usize,
) -> Result<usize> {
    if let Some(custom_order) = coeff_orders.and_then(|orders| {
        orders
            .iter()
            .find(|candidate| candidate.order == order && candidate.channel == channel)
    }) {
        return custom_order
            .positions
            .get(k)
            .copied()
            .ok_or(Error::InvalidCodestream("invalid coefficient order index"));
    }
    let order = natural_coeff_orders
        .get_mut(raw_strategy)
        .ok_or(Error::InvalidCodestream("invalid AC strategy"))?;
    if order.is_none() {
        *order = Some(natural_coeff_order(raw_strategy)?);
    }
    order
        .as_ref()
        .ok_or(Error::InvalidCodestream("invalid AC strategy"))?
        .get(k)
        .copied()
        .ok_or(Error::InvalidCodestream("invalid coefficient order index"))
}

fn write_vardct_ac_coefficient(
    grid: &mut VarDctAcChannelCoefficientGrid,
    block: VarDctFirstAcBlock,
    width_blocks: usize,
    coefficient_position: usize,
    coefficient: i32,
) -> Result<()> {
    let strategy_width = STRATEGY_BLOCKS_X
        .get(block.raw_strategy)
        .copied()
        .ok_or(Error::InvalidCodestream("invalid AC strategy"))?;
    let strategy_height = STRATEGY_BLOCKS_Y
        .get(block.raw_strategy)
        .copied()
        .ok_or(Error::InvalidCodestream("invalid AC strategy"))?;
    let local_width = strategy_width * 8;
    let local_x = coefficient_position % local_width;
    let local_y = coefficient_position / local_width;
    if local_y >= strategy_height * 8 {
        return Err(Error::InvalidCodestream("invalid AC coefficient position"));
    }
    let block_x = block.block_x + local_x / 8;
    let block_y = block.block_y + local_y / 8;
    let coeff_x = local_x % 8;
    let coeff_y = local_y % 8;
    let coeff_index = coeff_y * 8 + coeff_x;
    let index = ((block_y * width_blocks + block_x) * DCT_BLOCK_SIZE) + coeff_index;
    let slot = grid
        .coefficients
        .get_mut(index)
        .ok_or(Error::InvalidCodestream(
            "AC coefficient outside group grid",
        ))?;
    if *slot == 0 {
        grid.nonzero_coefficients += 1;
    }
    *slot = coefficient;
    grid.coefficient_checksum =
        checksum_placed_coefficient(grid.coefficient_checksum, index, coefficient);
    Ok(())
}

fn base_dequantize_vardct_ac_grid(
    grid: &VarDctAcCoefficientGrid,
    global: Option<&VarDctGlobalMetadata>,
    metadata: &VarDctAcGroupMetadata,
    dc_groups: &[VarDctDcGroupMetadata],
) -> Result<VarDctAcBaseDequantizedGrid> {
    let global = global.ok_or(Error::Unsupported("VarDCT global metadata"))?;
    let quant_field = vardct_quant_field_for_group(metadata, dc_groups)?;
    let coefficient_len = grid
        .width_blocks
        .checked_mul(grid.height_blocks)
        .and_then(|blocks| blocks.checked_mul(DCT_BLOCK_SIZE))
        .ok_or(Error::InvalidCodestream(
            "AC group coefficient grid is too large",
        ))?;
    let mut dequantized = VarDctAcBaseDequantizedGrid {
        group: grid.group,
        pass: grid.pass,
        width_blocks: grid.width_blocks,
        height_blocks: grid.height_blocks,
        inv_global_scale_bits: global.quantizer.inv_global_scale.to_bits(),
        per_channel: [
            VarDctAcBaseDequantizedChannelGrid::new(coefficient_len),
            VarDctAcBaseDequantizedChannelGrid::new(coefficient_len),
            VarDctAcBaseDequantizedChannelGrid::new(coefficient_len),
        ],
    };

    for channel in 0..3 {
        for block_y in 0..grid.height_blocks {
            for block_x in 0..grid.width_blocks {
                let quant = *quant_field
                    .get(block_y * grid.width_blocks + block_x)
                    .ok_or(Error::InvalidCodestream("invalid AC quant field"))?;
                if quant <= 0 {
                    return Err(Error::InvalidCodestream("invalid AC quant field"));
                }
                let scale = global.quantizer.inv_global_scale / quant as f32;
                for coeff in 0..DCT_BLOCK_SIZE {
                    let index = ((block_y * grid.width_blocks + block_x) * DCT_BLOCK_SIZE) + coeff;
                    let quantized = grid.per_channel[channel].coefficients[index];
                    if quantized == 0 {
                        continue;
                    }
                    let value = quantized as f32 * scale;
                    let channel_grid = &mut dequantized.per_channel[channel];
                    channel_grid.coefficients[index] = value.to_bits();
                    channel_grid.nonzero_coefficients += 1;
                    channel_grid.coefficient_checksum = checksum_dequantized_coefficient(
                        channel_grid.coefficient_checksum,
                        index,
                        value,
                    );
                }
            }
        }
    }

    Ok(dequantized)
}

fn dequantize_vardct_ac_grid(
    grid: &VarDctAcCoefficientGrid,
    global: Option<&VarDctGlobalMetadata>,
    metadata: &VarDctAcGroupMetadata,
    frame_header: &FrameHeader,
    dc_groups: &[VarDctDcGroupMetadata],
) -> Result<VarDctAcDequantizedGrid> {
    let global = global.ok_or(Error::Unsupported("VarDCT global metadata"))?;
    let quant_field = vardct_quant_field_for_group(metadata, dc_groups)?;
    let color_correlation = vardct_color_correlation_for_group(metadata, global, dc_groups)?;
    let blocks = vardct_ac_blocks_for_group(metadata, dc_groups)?;
    let coefficient_len = grid
        .width_blocks
        .checked_mul(grid.height_blocks)
        .and_then(|blocks| blocks.checked_mul(DCT_BLOCK_SIZE))
        .ok_or(Error::InvalidCodestream(
            "AC group coefficient grid is too large",
        ))?;
    let mut dequantized = VarDctAcDequantizedGrid {
        group: grid.group,
        pass: grid.pass,
        width_blocks: grid.width_blocks,
        height_blocks: grid.height_blocks,
        per_channel: [
            VarDctAcDequantizedChannelGrid::new(coefficient_len),
            VarDctAcDequantizedChannelGrid::new(coefficient_len),
            VarDctAcDequantizedChannelGrid::new(coefficient_len),
        ],
    };
    let x_dm_multiplier = (1.0f32 / 1.25f32).powf(frame_header.x_qm_scale as f32 - 2.0);
    let b_dm_multiplier = (1.0f32 / 1.25f32).powf(frame_header.b_qm_scale as f32 - 2.0);

    for block in blocks {
        let quant = *quant_field
            .get(block.block_y * grid.width_blocks + block.block_x)
            .ok_or(Error::InvalidCodestream("invalid AC quant field"))?;
        if quant <= 0 {
            return Err(Error::InvalidCodestream("invalid AC quant field"));
        }
        let y_scale = global.quantizer.inv_global_scale / quant as f32;
        let x_scale = y_scale * x_dm_multiplier;
        let b_scale = y_scale * b_dm_multiplier;
        let x_cc_mul = color_correlation
            .x
            .get((block.block_y / 8) * color_correlation.width_tiles + (block.block_x / 8))
            .copied()
            .ok_or(Error::InvalidCodestream("invalid AC color correlation map"))?;
        let b_cc_mul = color_correlation
            .b
            .get((block.block_y / 8) * color_correlation.width_tiles + (block.block_x / 8))
            .copied()
            .ok_or(Error::InvalidCodestream("invalid AC color correlation map"))?;
        let strategy_width = STRATEGY_BLOCKS_X[block.raw_strategy];
        let strategy_height = STRATEGY_BLOCKS_Y[block.raw_strategy];
        let size = strategy_width * strategy_height * DCT_BLOCK_SIZE;
        let x_matrix = default_dequant_matrix(block.raw_strategy, 0)?;
        let y_matrix = default_dequant_matrix(block.raw_strategy, 1)?;
        let b_matrix = default_dequant_matrix(block.raw_strategy, 2)?;
        if x_matrix.len() != size || y_matrix.len() != size || b_matrix.len() != size {
            return Err(Error::InvalidCodestream("invalid dequant matrix size"));
        }
        for local_position in 0..size {
            let index = coefficient_grid_index_for_local_position(
                grid.width_blocks,
                block,
                local_position,
            )?;
            let quantized_x = grid.per_channel[0].coefficients[index];
            let quantized_y = grid.per_channel[1].coefficients[index];
            let quantized_b = grid.per_channel[2].coefficients[index];
            let dequant_y = adjust_quant_bias(1, quantized_y) * y_matrix[local_position] * y_scale;
            let dequant_x_cc =
                adjust_quant_bias(0, quantized_x) * x_matrix[local_position] * x_scale;
            let dequant_b_cc =
                adjust_quant_bias(2, quantized_b) * b_matrix[local_position] * b_scale;
            write_dequantized_coefficient(&mut dequantized.per_channel[1], index, dequant_y);
            write_dequantized_coefficient(
                &mut dequantized.per_channel[0],
                index,
                dequant_x_cc + x_cc_mul * dequant_y,
            );
            write_dequantized_coefficient(
                &mut dequantized.per_channel[2],
                index,
                dequant_b_cc + b_cc_mul * dequant_y,
            );
        }
    }
    Ok(dequantized)
}

fn write_dequantized_coefficient(
    grid: &mut VarDctAcDequantizedChannelGrid,
    index: usize,
    value: f32,
) {
    if value == 0.0 {
        return;
    }
    grid.coefficients[index] = value.to_bits();
    grid.nonzero_coefficients += 1;
    grid.coefficient_checksum =
        checksum_dequantized_coefficient(grid.coefficient_checksum, index, value);
}

fn spatialize_vardct_ac_grid(
    grid: &VarDctAcDequantizedGrid,
    global: Option<&VarDctGlobalMetadata>,
    metadata: &VarDctAcGroupMetadata,
    dc_groups: &[VarDctDcGroupMetadata],
) -> Result<VarDctAcSpatialGrid> {
    spatialize_vardct_ac_grid_with_dc_multiplier(grid, global, metadata, dc_groups, 8.0)
}

fn spatialize_vardct_ac_grid_with_dc_multiplier(
    grid: &VarDctAcDequantizedGrid,
    global: Option<&VarDctGlobalMetadata>,
    metadata: &VarDctAcGroupMetadata,
    dc_groups: &[VarDctDcGroupMetadata],
    dc_multiplier: f32,
) -> Result<VarDctAcSpatialGrid> {
    let blocks = vardct_ac_blocks_for_group(metadata, dc_groups)?;
    let dc_grid = global
        .map(|global| vardct_dc_coefficients_for_group(metadata, global, dc_groups))
        .transpose()?;
    let sample_len = grid
        .width_blocks
        .checked_mul(grid.height_blocks)
        .and_then(|blocks| blocks.checked_mul(DCT_BLOCK_SIZE))
        .ok_or(Error::InvalidCodestream(
            "AC group spatial grid is too large",
        ))?;
    let mut spatial = VarDctAcSpatialGrid {
        group: grid.group,
        pass: grid.pass,
        width_blocks: grid.width_blocks,
        height_blocks: grid.height_blocks,
        blocks_attempted: blocks.len(),
        blocks_transformed: 0,
        blocks_skipped: 0,
        per_channel: [
            VarDctAcSpatialChannelGrid::new(sample_len),
            VarDctAcSpatialChannelGrid::new(sample_len),
            VarDctAcSpatialChannelGrid::new(sample_len),
        ],
    };

    for block in blocks {
        let Some(transform) = spatial_transform_for_strategy(block.raw_strategy) else {
            spatial.blocks_skipped += 1;
            continue;
        };
        for channel in 0..3 {
            let mut coefficients = vec![0.0f32; transform.width * transform.height];
            for y in 0..transform.height {
                for x in 0..transform.width {
                    coefficients[y * transform.width + x] =
                        dequantized_coefficient_for_transform_position(grid, channel, block, x, y)?;
                }
            }
            if let Some(dc_grid) = &dc_grid {
                coefficients[0] =
                    dc_grid.coefficient(channel, block.block_x, block.block_y)? * dc_multiplier;
            }
            let samples = match transform.kind {
                SpatialTransformKind::Identity => coefficients,
                SpatialTransformKind::Afv(kind) => inverse_afv_8x8(kind, &coefficients)?,
                SpatialTransformKind::Dct => {
                    if transform.width == 8 && transform.height == 8 {
                        let mut block = [0.0f32; DCT_BLOCK_SIZE];
                        block.copy_from_slice(&coefficients);
                        inverse_dct_8x8(&block).to_vec()
                    } else {
                        inverse_dct_rect(transform.width, transform.height, &coefficients)?
                    }
                }
            };
            for y in 0..transform.height {
                for x in 0..transform.width {
                    write_spatial_sample_for_transform_position(
                        &mut spatial.per_channel[channel],
                        grid.width_blocks,
                        block,
                        x,
                        y,
                        samples[y * transform.width + x],
                    )?;
                }
            }
        }
        spatial.blocks_transformed += 1;
    }
    Ok(spatial)
}

#[derive(Debug, Clone, Copy)]
struct SpatialTransform {
    width: usize,
    height: usize,
    kind: SpatialTransformKind,
}

#[derive(Debug, Clone, Copy)]
enum SpatialTransformKind {
    Dct,
    Identity,
    Afv(usize),
}

fn spatial_transform_for_strategy(raw_strategy: usize) -> Option<SpatialTransform> {
    let (width, height, kind) = match raw_strategy {
        0 => (8, 8, SpatialTransformKind::Dct),
        1 => (8, 8, SpatialTransformKind::Identity),
        2 => (2, 2, SpatialTransformKind::Dct),
        4 => (16, 16, SpatialTransformKind::Dct),
        6 => (8, 16, SpatialTransformKind::Dct),
        7 => (16, 8, SpatialTransformKind::Dct),
        12 => (4, 8, SpatialTransformKind::Dct),
        13 => (8, 4, SpatialTransformKind::Dct),
        14..=17 => (8, 8, SpatialTransformKind::Afv(raw_strategy - 14)),
        _ => return None,
    };
    Some(SpatialTransform {
        width,
        height,
        kind,
    })
}

fn dequantized_coefficient_for_transform_position(
    grid: &VarDctAcDequantizedGrid,
    channel: usize,
    block: VarDctFirstAcBlock,
    local_x: usize,
    local_y: usize,
) -> Result<f32> {
    let local_width = STRATEGY_BLOCKS_X
        .get(block.raw_strategy)
        .copied()
        .ok_or(Error::InvalidCodestream("invalid AC strategy"))?
        * 8;
    let coefficient_position = local_y
        .checked_mul(local_width)
        .and_then(|offset| offset.checked_add(local_x))
        .ok_or(Error::InvalidCodestream("invalid AC coefficient position"))?;
    let index =
        coefficient_grid_index_for_local_position(grid.width_blocks, block, coefficient_position)?;
    grid.per_channel
        .get(channel)
        .and_then(|channel| channel.coefficients.get(index))
        .copied()
        .map(f32::from_bits)
        .ok_or(Error::InvalidCodestream(
            "invalid dequantized coefficient grid",
        ))
}

fn write_spatial_sample_for_transform_position(
    grid: &mut VarDctAcSpatialChannelGrid,
    width_blocks: usize,
    block: VarDctFirstAcBlock,
    local_x: usize,
    local_y: usize,
    value: f32,
) -> Result<()> {
    let block_x = block
        .block_x
        .checked_add(local_x / 8)
        .ok_or(Error::InvalidCodestream("invalid spatial sample position"))?;
    let block_y = block
        .block_y
        .checked_add(local_y / 8)
        .ok_or(Error::InvalidCodestream("invalid spatial sample position"))?;
    let sample = (local_y % 8) * 8 + (local_x % 8);
    let index = block_y
        .checked_mul(width_blocks)
        .and_then(|offset| offset.checked_add(block_x))
        .and_then(|block_index| block_index.checked_mul(DCT_BLOCK_SIZE))
        .and_then(|offset| offset.checked_add(sample))
        .ok_or(Error::InvalidCodestream("invalid spatial sample position"))?;
    if index >= grid.samples.len() {
        return Err(Error::InvalidCodestream("invalid spatial sample position"));
    }
    write_spatial_sample(grid, index, value);
    Ok(())
}

#[derive(Debug, Clone)]
struct VarDctDcCoefficientGrid {
    width_blocks: usize,
    height_blocks: usize,
    per_channel: [Vec<f32>; 3],
}

impl VarDctDcCoefficientGrid {
    fn coefficient(&self, channel: usize, block_x: usize, block_y: usize) -> Result<f32> {
        if channel >= self.per_channel.len()
            || block_x >= self.width_blocks
            || block_y >= self.height_blocks
        {
            return Err(Error::InvalidCodestream("invalid VarDCT DC coefficient"));
        }
        self.per_channel[channel]
            .get(block_y * self.width_blocks + block_x)
            .copied()
            .ok_or(Error::InvalidCodestream("invalid VarDCT DC coefficient"))
    }
}

fn vardct_dc_coefficients_for_group(
    metadata: &VarDctAcGroupMetadata,
    global: &VarDctGlobalMetadata,
    dc_groups: &[VarDctDcGroupMetadata],
) -> Result<VarDctDcCoefficientGrid> {
    const DEFAULT_DC_QUANT: [f32; 3] = [1.0 / 4096.0, 1.0 / 512.0, 1.0 / 256.0];
    const XYB_DC_CHANNELS: [usize; 3] = [1, 0, 2];

    let dc_group = vardct_dc_group_for_ac_group(metadata, dc_groups)?;
    let var_dct_dc = dc_group
        .var_dct_dc
        .as_ref()
        .ok_or(Error::Unsupported("VarDCT DC coefficients"))?;
    let width_blocks = metadata.payload.group.width.div_ceil(8).min(256 / 8) as usize;
    let height_blocks = metadata.payload.group.height.div_ceil(8).min(256 / 8) as usize;
    let group_min_x = ((metadata.payload.group.x - dc_group.payload.group.x) / 8) as usize;
    let group_min_y = ((metadata.payload.group.y - dc_group.payload.group.y) / 8) as usize;
    let dc_quant = global.dc_dequant.coefficients.unwrap_or(DEFAULT_DC_QUANT);
    let mut per_channel = [
        vec![0.0; width_blocks * height_blocks],
        vec![0.0; width_blocks * height_blocks],
        vec![0.0; width_blocks * height_blocks],
    ];

    for output_channel in 0..3 {
        let modular_channel_index = XYB_DC_CHANNELS[output_channel];
        let channel = var_dct_dc
            .channels
            .iter()
            .find(|channel| channel.channel_index == modular_channel_index)
            .ok_or(Error::Unsupported("VarDCT DC coefficients"))?;
        let scale = global.quantizer.inv_quant_dc * dc_quant[output_channel];
        for y in 0..height_blocks {
            for x in 0..width_blocks {
                let source_x = group_min_x + x;
                let source_y = group_min_y + y;
                if source_x >= channel.width as usize || source_y >= channel.height as usize {
                    return Err(Error::InvalidCodestream("invalid VarDCT DC coefficient"));
                }
                let sample = channel.samples[source_y * channel.width as usize + source_x];
                per_channel[output_channel][y * width_blocks + x] = sample as f32 * scale;
            }
        }
    }

    Ok(VarDctDcCoefficientGrid {
        width_blocks,
        height_blocks,
        per_channel,
    })
}

fn vardct_dc_coefficient_diagnostics_for_group(
    plan: &VarDctDecodePlan,
    metadata: &VarDctAcGroupMetadata,
) -> Result<VarDctDcCoefficientDiagnostics> {
    const DEFAULT_DC_QUANT: [f32; 3] = [1.0 / 4096.0, 1.0 / 512.0, 1.0 / 256.0];
    const XYB_DC_CHANNELS: [usize; 3] = [1, 0, 2];

    let global = plan
        .global
        .as_ref()
        .ok_or(Error::Unsupported("VarDCT global metadata"))?;
    let dc_group = vardct_dc_group_for_ac_group(metadata, &plan.dc_group_metadata)?;
    let var_dct_dc = dc_group
        .var_dct_dc
        .as_ref()
        .ok_or(Error::Unsupported("VarDCT DC coefficients"))?;
    let coefficients = vardct_dc_coefficients_for_group(metadata, global, &plan.dc_group_metadata)?;
    let dc_quant = global.dc_dequant.coefficients.unwrap_or(DEFAULT_DC_QUANT);
    let width_blocks = metadata.payload.group.width.div_ceil(8).min(256 / 8) as usize;
    let height_blocks = metadata.payload.group.height.div_ceil(8).min(256 / 8) as usize;
    let group_min_x = ((metadata.payload.group.x - dc_group.payload.group.x) / 8) as usize;
    let group_min_y = ((metadata.payload.group.y - dc_group.payload.group.y) / 8) as usize;

    let raw_channels = std::array::from_fn(|output_channel| {
        let modular_channel = XYB_DC_CHANNELS[output_channel];
        let channel = var_dct_dc
            .channels
            .iter()
            .find(|channel| channel.channel_index == modular_channel)
            .expect("validated VarDCT DC channel");
        let mut selected = Vec::with_capacity(width_blocks * height_blocks);
        for y in 0..height_blocks {
            for x in 0..width_blocks {
                let source_x = group_min_x + x;
                let source_y = group_min_y + y;
                selected.push(channel.samples[source_y * channel.width as usize + source_x]);
            }
        }
        let sample_min = selected.iter().copied().min().unwrap_or(0);
        let sample_max = selected.iter().copied().max().unwrap_or(0);
        VarDctDcRawChannelDiagnostics {
            output_channel,
            modular_channel,
            width: width_blocks as u32,
            height: height_blocks as u32,
            nonzero_samples: selected.iter().filter(|sample| **sample != 0).count(),
            sample_min,
            sample_max,
            sample_sum: selected.iter().map(|sample| i64::from(*sample)).sum(),
            sample_checksum: checksum_i32_slice(&selected),
            anchors: sample_anchors_i32(&selected),
        }
    });
    let scaled_channels = std::array::from_fn(|output_channel| {
        let channel = &coefficients.per_channel[output_channel];
        let scale = global.quantizer.inv_quant_dc * dc_quant[output_channel];
        let mut checksum = 0u64;
        let mut nonzero = 0usize;
        for (index, &value) in channel.iter().enumerate() {
            if value != 0.0 {
                nonzero += 1;
                checksum = checksum_dequantized_coefficient(checksum, index, value);
            }
        }
        VarDctDcScaledChannelDiagnostics {
            output_channel,
            scale_bits: scale.to_bits(),
            nonzero_coefficients: nonzero,
            coefficient_checksum: checksum,
            anchors_bits: sample_anchors_f32_bits(channel),
        }
    });

    Ok(VarDctDcCoefficientDiagnostics {
        ac_group: metadata.payload.group.group,
        dc_group: dc_group.payload.group.group,
        width_blocks,
        height_blocks,
        inv_quant_dc_bits: global.quantizer.inv_quant_dc.to_bits(),
        dc_dequant_bits: dc_quant.map(f32::to_bits),
        raw_channels,
        scaled_channels,
    })
}

fn sample_anchors_i32(samples: &[i32]) -> Vec<i32> {
    if samples.is_empty() {
        return Vec::new();
    }
    [0usize, samples.len() / 2, samples.len() - 1]
        .into_iter()
        .map(|index| samples[index])
        .collect()
}

fn sample_anchors_f32_bits(samples: &[f32]) -> Vec<u32> {
    if samples.is_empty() {
        return Vec::new();
    }
    [0usize, samples.len() / 2, samples.len() - 1]
        .into_iter()
        .map(|index| samples[index].to_bits())
        .collect()
}

fn channel_range_diagnostics(samples: &[f32]) -> VarDctChannelRangeDiagnostics {
    let (min, max) = samples
        .iter()
        .copied()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), sample| {
            (min.min(sample), max.max(sample))
        });
    let sum = samples.iter().copied().sum::<f32>();
    let mut checksum = 0u64;
    for (index, &sample) in samples.iter().enumerate() {
        if sample != 0.0 {
            checksum = checksum_dequantized_coefficient(checksum, index, sample);
        }
    }

    VarDctChannelRangeDiagnostics {
        nonzero_samples: samples.iter().filter(|sample| **sample != 0.0).count(),
        negative_samples: samples.iter().filter(|sample| **sample < 0.0).count(),
        above_one_samples: samples.iter().filter(|sample| **sample > 1.0).count(),
        min_bits: min.to_bits(),
        max_bits: max.to_bits(),
        sum_bits: sum.to_bits(),
        checksum,
        anchors_bits: sample_anchors_f32_bits(samples),
    }
}

fn write_spatial_sample(grid: &mut VarDctAcSpatialChannelGrid, index: usize, value: f32) {
    if value == 0.0 {
        return;
    }
    grid.samples[index] = value.to_bits();
    grid.nonzero_samples += 1;
    grid.sample_checksum = checksum_dequantized_coefficient(grid.sample_checksum, index, value);
}

fn inverse_dct_8x8(coefficients: &[f32; DCT_BLOCK_SIZE]) -> [f32; DCT_BLOCK_SIZE] {
    let samples = inverse_dct_rect(8, 8, coefficients).expect("valid DCT8 dimensions");
    let mut block = [0.0f32; DCT_BLOCK_SIZE];
    block.copy_from_slice(&samples);
    block
}

fn inverse_dct_rect(width: usize, height: usize, coefficients: &[f32]) -> Result<Vec<f32>> {
    if width == 0 || height == 0 || coefficients.len() != width * height {
        return Err(Error::InvalidCodestream("invalid DCT dimensions"));
    }
    let mut output = vec![0.0f32; width * height];
    let inv_sqrt_2 = std::f32::consts::FRAC_1_SQRT_2;
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0f32;
            for v in 0..height {
                let cv = if v == 0 { inv_sqrt_2 } else { 1.0 };
                let cos_y = (((2 * y + 1) as f32 * v as f32 * std::f32::consts::PI)
                    / (2 * height) as f32)
                    .cos();
                for u in 0..width {
                    let cu = if u == 0 { inv_sqrt_2 } else { 1.0 };
                    let cos_x = (((2 * x + 1) as f32 * u as f32 * std::f32::consts::PI)
                        / (2 * width) as f32)
                        .cos();
                    sum += cu * cv * coefficients[v * width + u] * cos_x * cos_y;
                }
            }
            output[y * width + x] = 2.0 / ((width * height) as f32).sqrt() * sum;
        }
    }
    Ok(output)
}

fn inverse_afv_8x8(kind: usize, coefficients: &[f32]) -> Result<Vec<f32>> {
    if kind >= 4 || coefficients.len() != DCT_BLOCK_SIZE {
        return Err(Error::InvalidCodestream("invalid AFV transform"));
    }
    let afv_x = kind & 1;
    let afv_y = kind / 2;
    let block00 = coefficients[0];
    let block01 = coefficients[1];
    let block10 = coefficients[8];
    let dcs = [
        (block00 + block10 + block01) * 4.0,
        block00 + block10 - block01,
        block00 - block10,
    ];
    let mut pixels = vec![0.0f32; DCT_BLOCK_SIZE];

    let mut afv_coefficients = [0.0f32; 16];
    afv_coefficients[0] = dcs[0];
    for y in 0..4 {
        for x in 0..4 {
            if x == 0 && y == 0 {
                continue;
            }
            afv_coefficients[y * 4 + x] = coefficients[(y * 2) * 8 + x * 2];
        }
    }
    let afv_block = inverse_afv_4x4(&afv_coefficients);
    for y in 0..4 {
        for x in 0..4 {
            let source_x = if afv_x == 1 { 3 - x } else { x };
            let source_y = if afv_y == 1 { 3 - y } else { y };
            pixels[(y + afv_y * 4) * 8 + afv_x * 4 + x] = afv_block[source_y * 4 + source_x];
        }
    }

    let mut dct4 = [0.0f32; 16];
    dct4[0] = dcs[1];
    for y in 0..4 {
        for x in 0..4 {
            if x == 0 && y == 0 {
                continue;
            }
            dct4[y * 4 + x] = coefficients[(y * 2) * 8 + x * 2 + 1];
        }
    }
    let dct4_samples = inverse_dct_rect(4, 4, &dct4)?;
    let dct4_origin_x = if afv_x == 1 { 0 } else { 4 };
    let dct4_origin_y = afv_y * 4;
    for y in 0..4 {
        for x in 0..4 {
            pixels[(dct4_origin_y + y) * 8 + dct4_origin_x + x] = dct4_samples[y * 4 + x];
        }
    }

    let mut dct4x8 = [0.0f32; 32];
    dct4x8[0] = dcs[2];
    for y in 0..4 {
        for x in 0..8 {
            if x == 0 && y == 0 {
                continue;
            }
            dct4x8[y * 8 + x] = coefficients[(1 + y * 2) * 8 + x];
        }
    }
    let dct4x8_samples = inverse_dct_rect(8, 4, &dct4x8)?;
    let dct4x8_origin_y = if afv_y == 1 { 0 } else { 4 };
    for y in 0..4 {
        for x in 0..8 {
            pixels[(dct4x8_origin_y + y) * 8 + x] = dct4x8_samples[y * 8 + x];
        }
    }

    Ok(pixels)
}

fn inverse_afv_4x4(coefficients: &[f32; 16]) -> [f32; 16] {
    let mut pixels = [0.0f32; 16];
    for pixel in 0..16 {
        pixels[pixel] = coefficients
            .iter()
            .zip(AFV_4X4_BASIS.iter())
            .map(|(coefficient, basis)| coefficient * basis[pixel])
            .sum();
    }
    pixels
}

fn coefficient_grid_index_for_local_position(
    width_blocks: usize,
    block: VarDctFirstAcBlock,
    coefficient_position: usize,
) -> Result<usize> {
    let strategy_width = STRATEGY_BLOCKS_X
        .get(block.raw_strategy)
        .copied()
        .ok_or(Error::InvalidCodestream("invalid AC strategy"))?;
    let strategy_height = STRATEGY_BLOCKS_Y
        .get(block.raw_strategy)
        .copied()
        .ok_or(Error::InvalidCodestream("invalid AC strategy"))?;
    let local_width = strategy_width * 8;
    let local_x = coefficient_position % local_width;
    let local_y = coefficient_position / local_width;
    if local_y >= strategy_height * 8 {
        return Err(Error::InvalidCodestream("invalid AC coefficient position"));
    }
    let block_x = block.block_x + local_x / 8;
    let block_y = block.block_y + local_y / 8;
    let coeff_x = local_x % 8;
    let coeff_y = local_y % 8;
    Ok(((block_y * width_blocks + block_x) * DCT_BLOCK_SIZE) + coeff_y * 8 + coeff_x)
}

#[derive(Debug, Clone)]
struct VarDctColorCorrelationGrid {
    width_tiles: usize,
    x: Vec<f32>,
    b: Vec<f32>,
}

fn vardct_color_correlation_for_group(
    metadata: &VarDctAcGroupMetadata,
    global: &VarDctGlobalMetadata,
    dc_groups: &[VarDctDcGroupMetadata],
) -> Result<VarDctColorCorrelationGrid> {
    let dc_group = vardct_dc_group_for_ac_group(metadata, dc_groups)?;
    let ac_metadata = dc_group
        .ac_metadata
        .as_ref()
        .ok_or(Error::Unsupported("VarDCT AC color correlation"))?;
    let x_channel = ac_metadata
        .channels
        .iter()
        .find(|channel| channel.channel_index == 0)
        .ok_or(Error::Unsupported("VarDCT AC color correlation"))?;
    let b_channel = ac_metadata
        .channels
        .iter()
        .find(|channel| channel.channel_index == 1)
        .ok_or(Error::Unsupported("VarDCT AC color correlation"))?;
    let width_tiles = (metadata.payload.group.width.div_ceil(8).min(256 / 8) as usize).div_ceil(8);
    let height_tiles =
        (metadata.payload.group.height.div_ceil(8).min(256 / 8) as usize).div_ceil(8);
    let dc_width_tiles = dc_group.payload.group.width.div_ceil(8).div_ceil(8) as usize;
    let group_min_tile_x = ((metadata.payload.group.x - dc_group.payload.group.x) / 64) as usize;
    let group_min_tile_y = ((metadata.payload.group.y - dc_group.payload.group.y) / 64) as usize;
    let mut x = Vec::with_capacity(width_tiles * height_tiles);
    let mut b = Vec::with_capacity(width_tiles * height_tiles);
    for tile_y in 0..height_tiles {
        for tile_x in 0..width_tiles {
            let source_index =
                (group_min_tile_y + tile_y) * dc_width_tiles + group_min_tile_x + tile_x;
            let x_factor = *x_channel
                .samples
                .get(source_index)
                .ok_or(Error::InvalidCodestream("invalid AC color correlation map"))?;
            let b_factor = *b_channel
                .samples
                .get(source_index)
                .ok_or(Error::InvalidCodestream("invalid AC color correlation map"))?;
            x.push(
                global.color_correlation.base_correlation_x
                    + x_factor as f32 / global.color_correlation.color_factor as f32,
            );
            b.push(
                global.color_correlation.base_correlation_b
                    + b_factor as f32 / global.color_correlation.color_factor as f32,
            );
        }
    }
    Ok(VarDctColorCorrelationGrid { width_tiles, x, b })
}

fn vardct_dc_group_for_ac_group<'a>(
    metadata: &VarDctAcGroupMetadata,
    dc_groups: &'a [VarDctDcGroupMetadata],
) -> Result<&'a VarDctDcGroupMetadata> {
    dc_groups
        .iter()
        .find(|dc_group| {
            let group = &dc_group.payload.group;
            metadata.payload.group.x >= group.x
                && metadata.payload.group.y >= group.y
                && metadata.payload.group.x < group.x + group.width
                && metadata.payload.group.y < group.y + group.height
        })
        .ok_or(Error::Unsupported("VarDCT DC group"))
}

fn vardct_quant_field_for_group(
    metadata: &VarDctAcGroupMetadata,
    dc_groups: &[VarDctDcGroupMetadata],
) -> Result<Vec<i32>> {
    let dc_group = vardct_dc_group_for_ac_group(metadata, dc_groups)?;
    let ac_metadata = dc_group
        .ac_metadata
        .as_ref()
        .ok_or(Error::Unsupported("VarDCT AC quant field"))?;
    let strategy_channel = ac_metadata
        .channels
        .iter()
        .find(|channel| channel.channel_index == 2)
        .ok_or(Error::Unsupported("VarDCT AC quant field"))?;
    let dc_width_blocks = dc_group.payload.group.width.div_ceil(8) as usize;
    let dc_height_blocks = dc_group.payload.group.height.div_ceil(8) as usize;
    let group_width_blocks = metadata.payload.group.width.div_ceil(8).min(256 / 8) as usize;
    let group_height_blocks = metadata.payload.group.height.div_ceil(8).min(256 / 8) as usize;
    let group_min_x = ((metadata.payload.group.x - dc_group.payload.group.x) / 8) as usize;
    let group_min_y = ((metadata.payload.group.y - dc_group.payload.group.y) / 8) as usize;
    let count = strategy_channel.width as usize;
    if strategy_channel.height != 2 || strategy_channel.samples.len() < count * 2 {
        return Err(Error::Unsupported("VarDCT AC quant field"));
    }
    let mut quant_field = vec![0; group_width_blocks * group_height_blocks];
    let mut valid = vec![false; dc_width_blocks * dc_height_blocks];
    let mut cursor = 0usize;
    for y in 0..dc_height_blocks {
        for x in 0..dc_width_blocks {
            let index = y * dc_width_blocks + x;
            if valid[index] {
                continue;
            }
            let raw_strategy = *strategy_channel
                .samples
                .get(cursor)
                .ok_or(Error::InvalidCodestream("invalid AC metadata stream"))?
                as usize;
            let block_x = *STRATEGY_BLOCKS_X
                .get(raw_strategy)
                .ok_or(Error::InvalidCodestream("invalid AC strategy"))?;
            let block_y = *STRATEGY_BLOCKS_Y
                .get(raw_strategy)
                .ok_or(Error::InvalidCodestream("invalid AC strategy"))?;
            let quant = 1
                + (*strategy_channel
                    .samples
                    .get(count + cursor)
                    .ok_or(Error::InvalidCodestream("invalid AC quant field"))?)
                .clamp(0, 32_767);
            for dy in 0..block_y {
                for dx in 0..block_x {
                    let covered_x = x + dx;
                    let covered_y = y + dy;
                    if covered_x < dc_width_blocks && covered_y < dc_height_blocks {
                        valid[covered_y * dc_width_blocks + covered_x] = true;
                    }
                    if covered_x >= group_min_x
                        && covered_x < group_min_x + group_width_blocks
                        && covered_y >= group_min_y
                        && covered_y < group_min_y + group_height_blocks
                    {
                        let local_x = covered_x - group_min_x;
                        let local_y = covered_y - group_min_y;
                        quant_field[local_y * group_width_blocks + local_x] = quant;
                    }
                }
            }
            cursor += 1;
            if cursor > count {
                return Err(Error::InvalidCodestream("invalid AC metadata stream"));
            }
        }
    }
    Ok(quant_field)
}

fn adjust_quant_bias(channel: usize, quantized: i32) -> f32 {
    const BIASES: [f32; 4] = [
        1.0 - 0.05465007330715401,
        1.0 - 0.07005449891748593,
        1.0 - 0.049935103337343655,
        0.145,
    ];
    match quantized {
        0 => 0.0,
        1 => BIASES[channel],
        -1 => -BIASES[channel],
        value => value as f32 - BIASES[3] / value as f32,
    }
}

fn default_dequant_matrix(raw_strategy: usize, channel: usize) -> Result<Vec<f32>> {
    let width = *STRATEGY_BLOCKS_X
        .get(raw_strategy)
        .ok_or(Error::InvalidCodestream("invalid AC strategy"))?
        * 8;
    let height = *STRATEGY_BLOCKS_Y
        .get(raw_strategy)
        .ok_or(Error::InvalidCodestream("invalid AC strategy"))?
        * 8;
    let weights = match raw_strategy {
        0 => default_dct_quant_weights(width, height, DCT8_QUANT_BANDS, 6, channel)?,
        1 => default_identity_quant_weights(channel),
        2 => default_dct2_quant_weights(channel),
        4 => default_dct_quant_weights(width, height, DCT16_QUANT_BANDS, 7, channel)?,
        6 | 7 => default_dct_quant_weights(width, height, DCT8X16_QUANT_BANDS, 7, channel)?,
        12 | 13 => default_dct4x8_quant_weights(width, height, channel)?,
        14..=17 => default_afv_quant_weights(channel)?,
        _ => {
            return Err(Error::Unsupported(
                "default dequant matrix for VarDCT AC strategy",
            ));
        }
    };
    Ok(weights.into_iter().map(|weight| 1.0 / weight).collect())
}

fn default_dct_quant_weights(
    width: usize,
    height: usize,
    bands: [[[f32; 8]; 3]; 1],
    num_bands: usize,
    channel: usize,
) -> Result<Vec<f32>> {
    let channel_bands = &bands[0][channel][..num_bands];
    quant_weights_from_bands(width, height, channel_bands)
}

fn quant_weights_from_bands(
    width: usize,
    height: usize,
    encoded_bands: &[f32],
) -> Result<Vec<f32>> {
    if width == 0 || height == 0 || encoded_bands.is_empty() {
        return Err(Error::InvalidCodestream("invalid dequant matrix size"));
    }
    let mut bands = Vec::with_capacity(encoded_bands.len());
    bands.push(encoded_bands[0]);
    if bands[0] <= 0.0 {
        return Err(Error::InvalidCodestream("invalid dequant matrix"));
    }
    for &encoded in &encoded_bands[1..] {
        let previous = *bands.last().unwrap();
        let multiplier = if encoded > 0.0 {
            1.0 + encoded
        } else {
            1.0 / (1.0 - encoded)
        };
        bands.push(previous * multiplier);
    }
    let scale = (bands.len() - 1) as f32 / (std::f32::consts::SQRT_2 + 1.0e-6);
    let rcp_col = if width > 1 {
        scale / (width - 1) as f32
    } else {
        0.0
    };
    let rcp_row = if height > 1 {
        scale / (height - 1) as f32
    } else {
        0.0
    };
    let mut weights = vec![0.0; width * height];
    for y in 0..height {
        let dy = y as f32 * rcp_row;
        for x in 0..width {
            let dx = x as f32 * rcp_col;
            let pos = (dx * dx + dy * dy).sqrt();
            weights[y * width + x] = interpolate_bands(pos, &bands)?;
        }
    }
    Ok(weights)
}

fn interpolate_bands(pos: f32, bands: &[f32]) -> Result<f32> {
    if bands.len() == 1 {
        return Ok(bands[0]);
    }
    let max = std::f32::consts::SQRT_2 + 1.0e-6;
    let scaled_pos = pos * (bands.len() - 1) as f32 / max;
    let idx = scaled_pos.floor() as usize;
    if idx + 1 >= bands.len() {
        return Ok(*bands.last().unwrap());
    }
    let a = bands[idx];
    let b = bands[idx + 1];
    Ok(a * (b / a).powf(scaled_pos - idx as f32))
}

fn default_identity_quant_weights(channel: usize) -> Vec<f32> {
    const IDENTITY: [[f32; 3]; 3] = [
        [280.0, 3160.0, 3160.0],
        [60.0, 864.0, 864.0],
        [18.0, 200.0, 200.0],
    ];
    let mut weights = vec![IDENTITY[channel][0]; DCT_BLOCK_SIZE];
    weights[1] = IDENTITY[channel][1];
    weights[8] = IDENTITY[channel][1];
    weights[9] = IDENTITY[channel][2];
    weights
}

fn default_dct2_quant_weights(channel: usize) -> Vec<f32> {
    const DCT2: [[f32; 6]; 3] = [
        [3840.0, 2560.0, 1280.0, 640.0, 480.0, 300.0],
        [960.0, 640.0, 320.0, 180.0, 140.0, 120.0],
        [640.0, 320.0, 128.0, 64.0, 32.0, 16.0],
    ];
    let d = DCT2[channel];
    let mut weights = vec![0.0; DCT_BLOCK_SIZE];
    weights[1] = d[0];
    weights[8] = d[0];
    weights[9] = d[1];
    for y in 0..2 {
        for x in 0..2 {
            weights[y * 8 + x + 2] = d[2];
            weights[(y + 2) * 8 + x] = d[2];
            weights[(y + 2) * 8 + x + 2] = d[3];
        }
    }
    for y in 0..4 {
        for x in 0..4 {
            weights[y * 8 + x + 4] = d[4];
            weights[(y + 4) * 8 + x] = d[4];
            weights[(y + 4) * 8 + x + 4] = d[5];
        }
    }
    weights[0] = d[0];
    weights
}

fn default_dct4x8_quant_weights(width: usize, height: usize, channel: usize) -> Result<Vec<f32>> {
    let base = quant_weights_from_bands(
        width.min(8),
        height.min(8),
        &DCT4X8_QUANT_BANDS[0][channel][..4],
    )?;
    let mut weights = vec![0.0; width * height];
    for y in 0..height {
        for x in 0..width {
            let source_x = x.min(width.min(8) - 1);
            let source_y = y.min(height.min(8) - 1);
            weights[y * width + x] = base[source_y * width.min(8) + source_x];
        }
    }
    Ok(weights)
}

fn default_afv_quant_weights(channel: usize) -> Result<Vec<f32>> {
    let mut weights = vec![0.0; DCT_BLOCK_SIZE];
    let weights4x8 = quant_weights_from_bands(8, 4, &DCT4X8_QUANT_BANDS[0][channel][..4])?;
    let weights4x4 = quant_weights_from_bands(4, 4, &DCT4_QUANT_BANDS[0][channel][..4])?;
    let afv = AFV_WEIGHTS[channel];
    weights[0] = 1.0;
    weights[8] = afv[0];
    weights[1] = afv[1];
    weights[16] = afv[2];
    weights[2] = afv[3];
    weights[18] = afv[4];
    let mut bands = [0.0; 4];
    bands[0] = afv[5];
    for i in 1..4 {
        let encoded = afv[i + 5];
        bands[i] = bands[i - 1]
            * if encoded > 0.0 {
                1.0 + encoded
            } else {
                1.0 / (1.0 - encoded)
            };
    }
    const FREQS: [f32; 16] = [
        0.0, 0.0, 0.8517779, 5.3777843, 0.0, 0.0, 4.734748, 5.4492455, 1.659827, 4.0, 7.275749,
        10.423228, 2.6629324, 7.6306577, 8.962389, 12.971662,
    ];
    let lo = 0.8517779;
    let hi = 12.971662 - lo + 1.0e-6;
    for y in 0..4 {
        for x in 0..4 {
            if x < 2 && y < 2 {
                continue;
            }
            let pos = FREQS[y * 4 + x] - lo;
            weights[(2 * y) * 8 + 2 * x] = interpolate_custom(pos, hi, &bands)?;
        }
    }
    for y in 0..4 {
        for x in 0..8 {
            if x == 0 && y == 0 {
                continue;
            }
            weights[(2 * y + 1) * 8 + x] = weights4x8[y * 8 + x];
        }
    }
    for y in 0..4 {
        for x in 0..4 {
            if x == 0 && y == 0 {
                continue;
            }
            weights[(2 * y) * 8 + 2 * x + 1] = weights4x4[y * 4 + x];
        }
    }
    Ok(weights)
}

fn interpolate_custom(pos: f32, max: f32, bands: &[f32]) -> Result<f32> {
    let scaled_pos = pos * (bands.len() - 1) as f32 / max;
    let idx = scaled_pos.floor().max(0.0) as usize;
    if idx + 1 >= bands.len() {
        return Ok(*bands.last().unwrap());
    }
    let a = bands[idx];
    let b = bands[idx + 1];
    Ok(a * (b / a).powf(scaled_pos - idx as f32))
}

const DCT8_QUANT_BANDS: [[[f32; 8]; 3]; 1] = [[
    [3150.0, 0.0, -0.4, -0.4, -0.4, -2.0, 0.0, 0.0],
    [560.0, 0.0, -0.3, -0.3, -0.3, -0.3, 0.0, 0.0],
    [512.0, -2.0, -1.0, 0.0, -1.0, -2.0, 0.0, 0.0],
]];
const DCT4_QUANT_BANDS: [[[f32; 8]; 3]; 1] = [[
    [2200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [392.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [112.0, -0.25, -0.25, -0.5, 0.0, 0.0, 0.0, 0.0],
]];
const DCT16_QUANT_BANDS: [[[f32; 8]; 3]; 1] = [[
    [
        8996.872,
        -1.3000777,
        -0.4942453,
        -0.43909377,
        -0.6350102,
        -0.9017726,
        -1.6162099,
        0.0,
    ],
    [
        3191.4836,
        -0.67424583,
        -0.80745816,
        -0.44925836,
        -0.3586544,
        -0.3132239,
        -0.37615025,
        0.0,
    ],
    [
        1157.504, -2.0531423, -1.4, -0.5068713, -0.4270873, -1.4856834, -4.920914, 0.0,
    ],
]];
const DCT8X16_QUANT_BANDS: [[[f32; 8]; 3]; 1] = [[
    [7240.7734, -0.7, -0.7, -0.2, -0.2, -0.2, -0.5, 0.0],
    [1448.1547, -0.5, -0.5, -0.5, -0.2, -0.2, -0.2, 0.0],
    [506.85413, -1.4, -0.2, -0.5, -0.5, -1.5, -3.6, 0.0],
]];
const DCT4X8_QUANT_BANDS: [[[f32; 8]; 3]; 1] = [[
    [
        2198.0505,
        -0.96269625,
        -0.7619425,
        -0.65511405,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        764.36554, -0.926302, -0.9675229, -0.2784529, 0.0, 0.0, 0.0, 0.0,
    ],
    [
        527.10754, -1.4594386, -1.4500821, -1.5843723, 0.0, 0.0, 0.0, 0.0,
    ],
]];
const AFV_WEIGHTS: [[f32; 9]; 3] = [
    [3072.0, 3072.0, 256.0, 256.0, 256.0, 414.0, 0.0, 0.0, 0.0],
    [1024.0, 1024.0, 50.0, 50.0, 50.0, 58.0, 0.0, 0.0, 0.0],
    [384.0, 384.0, 12.0, 12.0, 12.0, 22.0, -0.25, -0.25, -0.25],
];
const DEFAULT_GABORISH_WEIGHTS: [f32; 6] = [
    1.1 * 0.104699568,
    1.1 * 0.055680538,
    1.1 * 0.104699568,
    1.1 * 0.055680538,
    1.1 * 0.104699568,
    1.1 * 0.055680538,
];
const GLOBAL_SCALE_DENOMINATOR: f32 = 65_536.0;
const EPF_INV_SIGMA_NUMERATOR: f32 = -1.1715729;
const EPF_MIN_SIGMA: f32 = -3.905243;
const EPF_PLUS_OFFSETS: [(isize, isize); 5] = [(0, 0), (-1, 0), (0, -1), (1, 0), (0, 1)];
const DEFAULT_OPSIN_BIASES: [f32; 3] = [-0.0037930732, -0.0037930732, -0.0037930732];
const DEFAULT_INVERSE_OPSIN_MATRIX: [[f32; 3]; 3] = [
    [11.031567, -9.866944, -0.164623],
    [-3.2541473, 4.4187703, -0.164623],
    [-3.6588514, 2.712923, 1.9459282],
];
const AFV_4X4_BASIS: [[f32; 16]; 16] = [
    [0.25; 16],
    [
        0.87690294, 0.2206518, -0.1014005, -0.1014005, 0.2206518, -0.1014005, -0.1014005,
        -0.1014005, -0.1014005, -0.1014005, -0.1014005, -0.1014005, -0.1014005, -0.1014005,
        -0.1014005, -0.1014005,
    ],
    [
        0.0,
        0.0,
        0.40670076,
        0.44444817,
        0.0,
        0.0,
        0.19574399,
        0.29291,
        -0.40670076,
        -0.195744,
        0.0,
        0.11379074,
        -0.44444817,
        -0.29291,
        -0.11379074,
        0.0,
    ],
    [
        0.0,
        0.0,
        -0.21255748,
        0.3085497,
        0.0,
        0.47067022,
        -0.16212052,
        0.0,
        -0.21255748,
        -0.16212052,
        -0.47067022,
        -0.14642918,
        0.3085497,
        0.0,
        -0.14642918,
        0.42511496,
    ],
    [
        0.0,
        -std::f32::consts::FRAC_1_SQRT_2,
        0.0,
        0.0,
        std::f32::consts::FRAC_1_SQRT_2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        -0.41053775,
        0.62354857,
        -0.06435072,
        -0.06435072,
        0.62354857,
        -0.06435072,
        -0.06435072,
        -0.06435072,
        -0.06435072,
        -0.06435072,
        -0.06435072,
        -0.06435072,
        -0.06435072,
        -0.06435072,
        -0.06435072,
        -0.06435072,
    ],
    [
        0.0,
        0.0,
        -0.45175567,
        0.15854503,
        0.0,
        -0.040385153,
        0.0074182265,
        0.39351034,
        -0.45175567,
        0.0074182265,
        0.11074166,
        0.08298163,
        0.15854503,
        0.39351034,
        0.08298163,
        -0.45175567,
    ],
    [
        0.0,
        0.0,
        -0.30468476,
        0.51126164,
        0.0,
        0.0,
        -0.29048014,
        -0.06578702,
        0.30468476,
        0.29048014,
        0.0,
        -0.23889774,
        -0.51126164,
        0.06578702,
        0.23889774,
        0.0,
    ],
    [
        0.0,
        0.0,
        0.30179295,
        0.25792363,
        0.0,
        0.1627234,
        0.095200226,
        0.0,
        0.30179295,
        0.095200226,
        -0.1627234,
        -0.35312384,
        0.25792363,
        0.0,
        -0.35312384,
        -0.6035859,
    ],
    [
        0.0, 0.0, 0.4082483, 0.0, 0.0, 0.0, 0.0, -0.4082483, -0.4082483, 0.0, 0.0, -0.4082483, 0.0,
        0.4082483, 0.4082483, 0.0,
    ],
    [
        0.0,
        0.0,
        0.1747867,
        0.08126112,
        0.0,
        0.0,
        -0.3675398,
        -0.30788222,
        -0.1747867,
        0.3675398,
        0.0,
        0.4826689,
        -0.08126112,
        0.30788222,
        -0.4826689,
        0.0,
    ],
    [
        0.0,
        0.0,
        -0.21105601,
        0.1856718,
        0.0,
        0.0,
        0.4921586,
        -0.38525015,
        0.21105601,
        -0.4921586,
        0.0,
        0.17419413,
        -0.1856718,
        0.38525015,
        -0.17419413,
        0.0,
    ],
    [
        0.0,
        0.0,
        -0.14266084,
        -0.34164467,
        0.0,
        0.73674977,
        0.24627107,
        -0.08574019,
        -0.14266084,
        0.24627107,
        0.148834,
        -0.047686804,
        -0.34164467,
        -0.08574019,
        -0.047686804,
        -0.14266084,
    ],
    [
        0.0,
        0.0,
        -0.1381354,
        0.33022827,
        0.0,
        0.08755115,
        -0.079467066,
        -0.46133748,
        -0.1381354,
        -0.079467066,
        0.49724647,
        0.12538059,
        0.33022827,
        -0.46133748,
        0.12538059,
        -0.1381354,
    ],
    [
        0.0,
        0.0,
        -0.17437603,
        0.07027907,
        0.0,
        -0.29210266,
        0.36238173,
        0.0,
        -0.17437603,
        0.36238173,
        0.29210266,
        -0.4326608,
        0.07027907,
        0.0,
        -0.4326608,
        0.34875205,
    ],
    [
        0.0,
        0.0,
        0.11354987,
        -0.074175045,
        0.0,
        0.19402893,
        -0.4351905,
        0.21918684,
        0.11354987,
        -0.4351905,
        0.55504435,
        -0.25468278,
        -0.074175045,
        0.21918684,
        -0.25468278,
        0.11354987,
    ],
];

fn natural_coeff_order(raw_strategy: usize) -> Result<Vec<usize>> {
    let mut cx = *STRATEGY_BLOCKS_X
        .get(raw_strategy)
        .ok_or(Error::InvalidCodestream("invalid AC strategy"))?;
    let mut cy = *STRATEGY_BLOCKS_Y
        .get(raw_strategy)
        .ok_or(Error::InvalidCodestream("invalid AC strategy"))?;
    if cy > cx {
        std::mem::swap(&mut cy, &mut cx);
    }
    let size = cx * cy * DCT_BLOCK_SIZE;
    let mut order = vec![0usize; size];
    let xs = cx / cy;
    let xsm = xs - 1;
    let xss = ceil_log2_nonzero(xs);
    let block_dim = 8usize;
    let full = cx * block_dim;
    let mut cur = cx * cy;

    for i in 0..full {
        for j in 0..=i {
            let mut x = j;
            let mut y = i - j;
            if i % 2 == 1 {
                std::mem::swap(&mut x, &mut y);
            }
            if (y & xsm) != 0 {
                continue;
            }
            y >>= xss;
            let val = if x < cx && y < cy {
                y * cx + x
            } else {
                let val = cur;
                cur += 1;
                val
            };
            if val < size {
                order[val] = y * cx * block_dim + x;
            }
        }
    }

    for ip in (1..full).rev() {
        let i = ip - 1;
        for j in 0..=i {
            let mut x = full - 1 - (i - j);
            let mut y = full - 1 - j;
            if i % 2 == 1 {
                std::mem::swap(&mut x, &mut y);
            }
            if (y & xsm) != 0 {
                continue;
            }
            y >>= xss;
            let val = cur;
            cur += 1;
            if val < size {
                order[val] = y * cx * block_dim + x;
            }
        }
    }
    Ok(order)
}

fn zero_density_context(
    nonzeros_left: usize,
    k: usize,
    covered_blocks: usize,
    log2_covered_blocks: usize,
    prev: usize,
) -> Result<usize> {
    if covered_blocks == 0 || (1usize << log2_covered_blocks) != covered_blocks {
        return Err(Error::InvalidCodestream("invalid AC covered block count"));
    }
    let nonzeros_left = (nonzeros_left + covered_blocks - 1) >> log2_covered_blocks;
    let k = k >> log2_covered_blocks;
    if k == 0 || k >= 64 || nonzeros_left == 0 || nonzeros_left >= 64 {
        return Err(Error::InvalidCodestream("invalid AC zero-density context"));
    }
    Ok((COEFF_NUM_NONZERO_CONTEXT[nonzeros_left] + COEFF_FREQ_CONTEXT[k]) * 2 + prev)
}

fn predict_from_top_and_left(
    row_nzeros: &[i32],
    stride: usize,
    x: usize,
    y: usize,
    default_value: i32,
) -> i32 {
    let top = y
        .checked_sub(1)
        .and_then(|top_y| row_nzeros.get(top_y * stride + x))
        .copied();
    if x == 0 {
        return top.unwrap_or(default_value);
    }
    let left = row_nzeros
        .get(y * stride + x - 1)
        .copied()
        .unwrap_or(default_value);
    match top {
        Some(top) => (top + left + 1) / 2,
        None => left,
    }
}

fn checksum_i32_slice(values: &[i32]) -> u64 {
    values.iter().fold(0xcbf29ce484222325, |hash, value| {
        (hash ^ i64::from(*value) as u64).wrapping_mul(0x100000001b3)
    })
}

fn checksum_placed_coefficient(hash: u64, position: usize, coefficient: i32) -> u64 {
    [position as u64, i64::from(coefficient) as u64]
        .into_iter()
        .fold(hash, |hash, value| {
            (hash ^ value).wrapping_mul(0x100000001b3)
        })
}

fn checksum_dequantized_coefficient(hash: u64, position: usize, coefficient: f32) -> u64 {
    [position as u64, coefficient.to_bits() as u64]
        .into_iter()
        .fold(hash, |hash, value| {
            (hash ^ value).wrapping_mul(0x100000001b3)
        })
}

fn checksum_coefficient_event(hash: u64, event: &VarDctAcCoefficientEvent) -> u64 {
    [
        event.k as u64,
        event.zero_density_context as u64,
        event.context as u64,
        event.clustered_context as u64,
        event.start_bits as u64,
        event.end_bits as u64,
        u64::from(event.u_coeff),
        event.coeff as i64 as u64,
        event.prev_after as u64,
        event.remaining_nzeros as u64,
    ]
    .into_iter()
    .fold(hash, |hash, value| {
        (hash ^ value).wrapping_mul(0x100000001b3)
    })
}

fn read_vardct_dc_group_metadata(
    codestream: &[u8],
    frame_header: &FrameHeader,
    payload: VarDctDcGroupPayloadMetadata,
    global_tree: Option<&ModularTreeCoding>,
    global_tree_error: Option<&Error>,
) -> Result<VarDctDcGroupMetadata> {
    let bytes = codestream
        .get(payload.section.payload_range.clone())
        .ok_or(Error::InvalidCodestream("frame section outside codestream"))?;
    let mut reader = BitReader::new(bytes);
    let extra_precision_result = reader.read_bits(2).map(|bits| bits as u8);
    let extra_precision_end = extra_precision_result
        .as_ref()
        .ok()
        .map(|_| reader.bits_consumed());
    let mut stream_reader = reader.clone();
    let (extra_precision_bits, var_dct_dc_header, var_dct_dc, parse_error) =
        match extra_precision_result {
            Ok(extra_precision_bits) => {
                match read_modular_group_header_metadata(&mut reader) {
                    Ok(header) => {
                        let channels = vardct_dc_channel_plan(frame_header, &payload)?;
                        if header.use_global_tree && global_tree.is_none() {
                            (
                                Some(extra_precision_bits),
                                Some(header),
                                None,
                                Some(global_tree_error.cloned().unwrap_or(
                                    Error::InvalidCodestream(
                                        "modular stream references a missing global tree",
                                    ),
                                )),
                            )
                        } else {
                            match decode_modular_stream_from_reader(
                                &mut stream_reader,
                                payload.section.section.section_physical_index,
                                payload.var_dct_dc_stream_id,
                                &channels,
                                global_tree,
                            ) {
                                Ok((decoded_header, decoded)) => (
                                    Some(extra_precision_bits),
                                    Some(decoded_header),
                                    Some(decoded),
                                    None,
                                ),
                                Err(error) => {
                                    (Some(extra_precision_bits), Some(header), None, Some(error))
                                }
                            }
                        }
                    }
                    Err(error) => (Some(extra_precision_bits), None, None, Some(error)),
                }
            }
            Err(error) => (None, None, None, Some(error)),
        };
    let header_end = var_dct_dc_header.as_ref().map(|_| reader.bits_consumed());
    let var_dct_dc_end = parse_error
        .is_none()
        .then_some(stream_reader.bits_consumed())
        .filter(|_| var_dct_dc_header.is_some());
    let (modular_dc, modular_dc_error, modular_dc_end) = match var_dct_dc_end {
        Some(start_bits) => {
            let mut modular_dc_reader = BitReader::new(bytes);
            modular_dc_reader.skip_bits(start_bits)?;
            let decoded = ModularDecodedGroup {
                section_physical_index: payload.section.section.section_physical_index,
                stream_id: payload.modular_dc_stream_id,
                channels: Vec::new(),
                bits_consumed: modular_dc_reader.bits_consumed(),
            };
            (Some(decoded), None, Some(modular_dc_reader.bits_consumed()))
        }
        None => (None, None, None),
    };
    let (ac_metadata_count, ac_metadata, ac_metadata_error, ac_metadata_end) = match modular_dc_end
    {
        Some(start_bits) => {
            let mut ac_reader = BitReader::new(bytes);
            ac_reader.skip_bits(start_bits)?;
            match read_vardct_ac_metadata_count(&mut ac_reader, &payload) {
                Ok(count) => {
                    let channels = vardct_ac_metadata_channel_plan(&payload, count);
                    match decode_modular_stream_from_reader(
                        &mut ac_reader,
                        payload.section.section.section_physical_index,
                        payload.ac_metadata_stream_id,
                        &channels,
                        global_tree,
                    ) {
                        Ok((_, decoded)) => (
                            Some(count),
                            Some(decoded),
                            None,
                            Some(ac_reader.bits_consumed()),
                        ),
                        Err(error) => (Some(count), None, Some(error), None),
                    }
                }
                Err(error) => (None, None, Some(error), None),
            }
        }
        None => (None, None, None, None),
    };
    Ok(VarDctDcGroupMetadata {
        payload,
        cursor: VarDctDcGroupCursorMetadata {
            extra_precision_start_bits: 0,
            extra_precision_end_bits: extra_precision_end,
            var_dct_dc_start_bits: extra_precision_end,
            var_dct_dc_header_end_bits: header_end,
            var_dct_dc_end_bits: var_dct_dc_end,
            modular_dc_start_bits: var_dct_dc_end,
            modular_dc_end_bits: modular_dc_end,
            ac_metadata_start_bits: modular_dc_end,
            ac_metadata_end_bits: ac_metadata_end,
        },
        extra_precision_bits,
        var_dct_dc_header,
        var_dct_dc,
        modular_dc,
        modular_dc_error,
        ac_metadata_count,
        ac_metadata,
        ac_metadata_error,
        parse_error,
    })
}

fn read_vardct_ac_global_metadata(
    codestream: &[u8],
    frame_header: &FrameHeader,
    payload: &VarDctSectionPayloadMetadata,
    global: &VarDctGlobalMetadata,
    used_acs: Option<u32>,
) -> Result<VarDctAcGlobalMetadata> {
    let bytes = codestream
        .get(payload.payload_range.clone())
        .ok_or(Error::InvalidCodestream("frame section outside codestream"))?;
    let mut reader = BitReader::new(bytes);
    let all_default_quant_matrices = match reader.read_bool() {
        Ok(value) => value,
        Err(error) => {
            return Ok(VarDctAcGlobalMetadata {
                section: payload.clone(),
                all_default_quant_matrices: None,
                quant_matrices_end_bits: None,
                num_histograms: None,
                num_histograms_end_bits: None,
                used_acs,
                passes: Vec::new(),
                bits_consumed: None,
                parse_error: Some(error),
            });
        }
    };
    let quant_matrices_end_bits = Some(reader.bits_consumed());
    if !all_default_quant_matrices {
        return Ok(VarDctAcGlobalMetadata {
            section: payload.clone(),
            all_default_quant_matrices: Some(false),
            quant_matrices_end_bits,
            num_histograms: None,
            num_histograms_end_bits: None,
            used_acs,
            passes: Vec::new(),
            bits_consumed: None,
            parse_error: Some(Error::Unsupported("custom VarDCT AC quant matrices")),
        });
    }

    let num_histo_bits = ceil_log2_nonzero(frame_header.group_layout.num_groups as usize);
    let num_histograms = match reader.read_bits(num_histo_bits) {
        Ok(bits) => bits as usize + 1,
        Err(error) => {
            return Ok(VarDctAcGlobalMetadata {
                section: payload.clone(),
                all_default_quant_matrices: Some(true),
                quant_matrices_end_bits,
                num_histograms: None,
                num_histograms_end_bits: None,
                used_acs,
                passes: Vec::new(),
                bits_consumed: None,
                parse_error: Some(error),
            });
        }
    };
    let num_histograms_end_bits = Some(reader.bits_consumed());
    let mut passes = Vec::with_capacity(frame_header.passes.num_passes as usize);
    for pass in 0..frame_header.passes.num_passes as usize {
        let used_orders =
            match reader.read_u32_selector(val(0x5f), val(0x13), val(0), bits_offset(13, 0)) {
                Ok(used_orders) => used_orders,
                Err(error) => {
                    passes.push(VarDctAcGlobalPassMetadata {
                        pass,
                        used_orders: None,
                        used_orders_end_bits: None,
                        coeff_orders: Vec::new(),
                        coeff_order_end_bits: None,
                        histogram_contexts: None,
                        histogram_count: None,
                        histogram_end_bits: None,
                        use_prefix_code: None,
                        log_alpha_size: None,
                        error_stage: None,
                        error_bits: Some(reader.bits_consumed()),
                        error: Some(error.clone()),
                    });
                    return Ok(VarDctAcGlobalMetadata {
                        section: payload.clone(),
                        all_default_quant_matrices: Some(true),
                        quant_matrices_end_bits,
                        num_histograms: Some(num_histograms),
                        num_histograms_end_bits,
                        used_acs,
                        passes,
                        bits_consumed: None,
                        parse_error: Some(error),
                    });
                }
            };
        let used_orders_end_bits = Some(reader.bits_consumed());
        let coeff_order_probe = read_vardct_coeff_orders(&mut reader, used_orders as u16);
        let coeff_orders = match coeff_order_probe {
            Ok(coeff_orders) => coeff_orders,
            Err(error) => {
                passes.push(VarDctAcGlobalPassMetadata {
                    pass,
                    used_orders: Some(used_orders),
                    used_orders_end_bits,
                    coeff_orders: error.coeff_orders,
                    coeff_order_end_bits: error.end_bits,
                    histogram_contexts: None,
                    histogram_count: None,
                    histogram_end_bits: None,
                    use_prefix_code: None,
                    log_alpha_size: None,
                    error_stage: None,
                    error_bits: Some(error.error_bits),
                    error: Some(error.error.clone()),
                });
                return Ok(VarDctAcGlobalMetadata {
                    section: payload.clone(),
                    all_default_quant_matrices: Some(true),
                    quant_matrices_end_bits,
                    num_histograms: Some(num_histograms),
                    num_histograms_end_bits,
                    used_acs,
                    passes,
                    bits_consumed: None,
                    parse_error: Some(error.error),
                });
            }
        };
        let coeff_order_end_bits = Some(reader.bits_consumed());

        let histogram_contexts =
            num_histograms * global.block_context_map.num_contexts * (37 + 458);
        let histogram_probe = probe_decode_histograms(&mut reader, histogram_contexts, false);
        let pass_error = histogram_probe.error.clone();
        passes.push(VarDctAcGlobalPassMetadata {
            pass,
            used_orders: Some(used_orders),
            used_orders_end_bits,
            coeff_orders,
            coeff_order_end_bits,
            histogram_contexts: Some(histogram_contexts),
            histogram_count: histogram_probe.num_histograms,
            histogram_end_bits: histogram_probe.histogram_end_bits,
            use_prefix_code: histogram_probe.use_prefix_code,
            log_alpha_size: histogram_probe.log_alpha_size,
            error_stage: histogram_probe
                .error_stage
                .map(VarDctHistogramProbeStage::from),
            error_bits: histogram_probe.error_bits,
            error: pass_error.clone(),
        });
        if let Some(error) = pass_error {
            return Ok(VarDctAcGlobalMetadata {
                section: payload.clone(),
                all_default_quant_matrices: Some(true),
                quant_matrices_end_bits,
                num_histograms: Some(num_histograms),
                num_histograms_end_bits,
                used_acs,
                passes,
                bits_consumed: None,
                parse_error: Some(error),
            });
        }
    }

    Ok(VarDctAcGlobalMetadata {
        section: payload.clone(),
        all_default_quant_matrices: Some(true),
        quant_matrices_end_bits,
        num_histograms: Some(num_histograms),
        num_histograms_end_bits,
        used_acs,
        passes,
        bits_consumed: Some(reader.bits_consumed()),
        parse_error: None,
    })
}

#[derive(Debug, Clone)]
struct VarDctCoeffOrderError {
    coeff_orders: Vec<VarDctCoeffOrderMetadata>,
    end_bits: Option<usize>,
    error_bits: usize,
    error: Error,
}

const DCT_BLOCK_SIZE: usize = 64;
const PERMUTATION_CONTEXTS: usize = 8;
const STRATEGY_ORDER_BUCKETS: usize = 13;
const NONZERO_BUCKETS: usize = 37;
const ZERO_DENSITY_CONTEXT_COUNT: usize = 458;
const FIRST_AC_BLOCK_EVENT_LIMIT: usize = 4096;
const COEFF_FREQ_CONTEXT: [usize; 64] = [
    0xBAD, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19,
    19, 20, 20, 21, 21, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27,
    27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
];
const COEFF_NUM_NONZERO_CONTEXT: [usize; 64] = [
    0xBAD, 0, 31, 62, 62, 93, 93, 93, 93, 123, 123, 123, 123, 152, 152, 152, 152, 152, 152, 152,
    152, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206, 206, 206,
];
const STRATEGY_ORDER: [usize; 27] = [
    0, 1, 1, 1, 2, 3, 4, 4, 5, 5, 6, 6, 1, 1, 1, 1, 1, 1, 7, 8, 8, 9, 10, 10, 11, 12, 12,
];
const STRATEGY_BLOCKS_X: [usize; 27] = [
    1, 1, 1, 1, 2, 4, 1, 2, 1, 4, 2, 4, 1, 1, 1, 1, 1, 1, 8, 4, 8, 16, 8, 16, 32, 16, 32,
];
const STRATEGY_BLOCKS_Y: [usize; 27] = [
    1, 1, 1, 1, 2, 4, 2, 1, 4, 1, 4, 2, 1, 1, 1, 1, 1, 1, 8, 8, 4, 16, 16, 8, 32, 32, 16,
];

fn read_vardct_coeff_orders(
    reader: &mut BitReader<'_>,
    used_orders: u16,
) -> std::result::Result<Vec<VarDctCoeffOrderMetadata>, VarDctCoeffOrderError> {
    if used_orders == 0 {
        return Ok(Vec::new());
    }

    let (code, context_map) =
        decode_histograms(reader, PERMUTATION_CONTEXTS, false).map_err(|error| {
            VarDctCoeffOrderError {
                coeff_orders: Vec::new(),
                end_bits: None,
                error_bits: reader.bits_consumed(),
                error,
            }
        })?;
    let mut symbol_reader =
        AnsSymbolReader::new(code, reader, 0).map_err(|error| VarDctCoeffOrderError {
            coeff_orders: Vec::new(),
            end_bits: None,
            error_bits: reader.bits_consumed(),
            error,
        })?;

    let mut coeff_orders = Vec::new();
    let mut computed = 0u16;
    for raw_strategy in 0..STRATEGY_ORDER.len() {
        let order = STRATEGY_ORDER[raw_strategy];
        let order_bit = 1u16 << order;
        if computed & order_bit != 0 {
            continue;
        }
        computed |= order_bit;
        if used_orders & order_bit == 0 {
            continue;
        }

        let llf = STRATEGY_BLOCKS_X[raw_strategy] * STRATEGY_BLOCKS_Y[raw_strategy];
        let size = llf * DCT_BLOCK_SIZE;
        for channel in 0..3 {
            match read_vardct_coeff_order_permutation(
                reader,
                &mut symbol_reader,
                &context_map,
                order,
                channel,
                raw_strategy,
                llf,
                size,
            ) {
                Ok(metadata) => coeff_orders.push(metadata),
                Err(error) => {
                    return Err(VarDctCoeffOrderError {
                        coeff_orders,
                        end_bits: None,
                        error_bits: reader.bits_consumed(),
                        error,
                    });
                }
            }
        }
    }

    if !symbol_reader.check_final_state() {
        return Err(VarDctCoeffOrderError {
            coeff_orders,
            end_bits: Some(reader.bits_consumed()),
            error_bits: reader.bits_consumed(),
            error: Error::InvalidCodestream("invalid coefficient-order ANS state"),
        });
    }

    Ok(coeff_orders)
}

fn read_vardct_coeff_order_permutation(
    reader: &mut BitReader<'_>,
    symbol_reader: &mut AnsSymbolReader,
    context_map: &[u8],
    order: usize,
    channel: usize,
    raw_strategy: usize,
    skip: usize,
    size: usize,
) -> Result<VarDctCoeffOrderMetadata> {
    let end = symbol_reader.read_hybrid_uint(coeff_order_context(size), reader, context_map)?
        as usize
        + skip;
    if end > size {
        return Err(Error::InvalidCodestream("invalid coefficient-order size"));
    }

    let mut lehmer = vec![0u32; size];
    let mut last = 0usize;
    for (index, value) in lehmer.iter_mut().enumerate().take(end).skip(skip) {
        let code =
            symbol_reader.read_hybrid_uint(coeff_order_context(last), reader, context_map)?;
        if code as usize >= size - index {
            return Err(Error::InvalidCodestream(
                "invalid coefficient-order Lehmer code",
            ));
        }
        *value = code;
        last = code as usize;
    }
    let permutation = decode_lehmer_code(&lehmer)?;
    let natural_order = natural_coeff_order(raw_strategy)?;
    let positions = permutation
        .iter()
        .map(|&index| {
            natural_order
                .get(index)
                .copied()
                .ok_or(Error::InvalidCodestream("invalid coefficient-order entry"))
        })
        .collect::<Result<Vec<_>>>()?;
    let checksum = checksum_permutation(&permutation);

    Ok(VarDctCoeffOrderMetadata {
        order,
        channel,
        skip,
        size,
        permutation_end: end,
        checksum,
        positions,
    })
}

fn coeff_order_context(value: usize) -> usize {
    if value == 0 {
        0
    } else {
        (usize::BITS as usize - value.leading_zeros() as usize).min(PERMUTATION_CONTEXTS - 1)
    }
}

fn decode_lehmer_code(code: &[u32]) -> Result<Vec<usize>> {
    let size = code.len();
    if size == 0 {
        return Err(Error::InvalidCodestream("empty Lehmer code"));
    }
    let log2_size = usize::BITS as usize - (size - 1).leading_zeros() as usize;
    let padded_size = 1usize << log2_size;
    let mut tree = vec![0u32; padded_size];
    for (index, value) in tree.iter_mut().enumerate() {
        *value = value_of_lowest_one_bit(index + 1) as u32;
    }

    let mut permutation = vec![0; size];
    for (index, &lehmer) in code.iter().enumerate() {
        if lehmer as usize + index >= size {
            return Err(Error::InvalidCodestream("invalid Lehmer code"));
        }
        let mut rank = lehmer + 1;
        let mut bit = padded_size;
        let mut next = 0usize;
        for _ in 0..=log2_size {
            let candidate = next + bit;
            bit >>= 1;
            if tree[candidate - 1] < rank {
                next = candidate;
                rank -= tree[candidate - 1];
            }
        }
        permutation[index] = next;

        next += 1;
        while next <= padded_size {
            tree[next - 1] -= 1;
            next += value_of_lowest_one_bit(next);
        }
    }
    Ok(permutation)
}

fn value_of_lowest_one_bit(value: usize) -> usize {
    value & value.wrapping_neg()
}

fn checksum_permutation(permutation: &[usize]) -> u64 {
    permutation.iter().fold(0xcbf29ce484222325, |hash, value| {
        (hash ^ *value as u64).wrapping_mul(0x100000001b3)
    })
}

fn used_acs_from_dc_group_metadata(dc_groups: &[VarDctDcGroupMetadata]) -> Option<u32> {
    let mut used_acs = 0u32;
    for dc_group in dc_groups {
        let Some(ac_metadata) = &dc_group.ac_metadata else {
            continue;
        };
        let Some(strategy_channel) = ac_metadata
            .channels
            .iter()
            .find(|channel| channel.channel_index == 2)
        else {
            continue;
        };
        let width = strategy_channel.width as usize;
        for &sample in strategy_channel.samples.iter().take(width) {
            if !(0..STRATEGY_ORDER.len() as i32).contains(&sample) {
                return None;
            }
            used_acs |= 1u32 << sample;
        }
    }
    (used_acs != 0).then_some(used_acs)
}

fn read_vardct_modular_global_tree(
    codestream: &[u8],
    metadata: &ImageMetadata,
    frame_header: &FrameHeader,
    payload: &VarDctSectionPayloadMetadata,
    global: &VarDctGlobalMetadata,
) -> Result<VarDctModularGlobalTreeRead> {
    let bytes = codestream
        .get(payload.payload_range.clone())
        .ok_or(Error::InvalidCodestream("frame section outside codestream"))?;
    let mut reader = BitReader::new(bytes);
    reader.skip_bits(global.bits_consumed)?;
    match read_modular_global_tree_coding(&mut reader, metadata, frame_header) {
        Ok(tree) => {
            let mut direct_probe = BitReader::new(bytes);
            direct_probe.skip_bits(global.bits_consumed)?;
            let direct_probe =
                probe_modular_global_tree_coding(&mut direct_probe, metadata, frame_header);
            Ok(VarDctModularGlobalTreeRead {
                direct_start_bits: global.bits_consumed,
                direct_tree_end_bits: direct_probe.tree_end_bits,
                direct_tree_node_count: direct_probe.tree_node_count,
                direct_tree_leaf_count: direct_probe.tree_leaf_count,
                direct_tree_leaves: direct_probe
                    .tree_leaves
                    .iter()
                    .map(VarDctMaTreeLeafProbe::from)
                    .collect(),
                direct_error_bits: direct_probe.error_bits,
                direct_residual_context_count: direct_probe.residual_context_count,
                direct_residual_histogram_count: direct_probe.residual_histogram_count,
                direct_residual_context_map_entries: direct_probe
                    .residual_histogram_probe
                    .as_ref()
                    .map(|probe| probe.context_map_entries.clone())
                    .unwrap_or_default(),
                direct_residual_context_map_raw_entries: direct_probe
                    .residual_histogram_probe
                    .as_ref()
                    .map(|probe| probe.context_map_raw_entries.clone())
                    .unwrap_or_default(),
                direct_residual_context_map_distinct_entries: direct_probe
                    .residual_histogram_probe
                    .as_ref()
                    .map(|probe| probe.context_map_distinct_entries.clone())
                    .unwrap_or_default(),
                direct_residual_context_map_histogram_usage_counts: direct_probe
                    .residual_histogram_probe
                    .as_ref()
                    .map(|probe| probe.context_map_histogram_usage_counts.clone())
                    .unwrap_or_default(),
                direct_residual_context_map_max_entry: direct_probe
                    .residual_histogram_probe
                    .as_ref()
                    .and_then(|probe| probe.context_map_max_entry),
                direct_residual_context_map_symbol_entries: direct_probe
                    .residual_histogram_probe
                    .as_ref()
                    .map(|probe| {
                        probe
                            .context_map_symbol_entries
                            .iter()
                            .map(VarDctContextMapSymbolProbe::from)
                            .collect()
                    })
                    .unwrap_or_default(),
                direct_residual_lz77_end_bits: direct_probe
                    .residual_histogram_probe
                    .as_ref()
                    .and_then(|probe| probe.lz77_end_bits),
                direct_residual_context_map_end_bits: direct_probe
                    .residual_histogram_probe
                    .as_ref()
                    .and_then(|probe| probe.context_map_end_bits),
                direct_residual_entropy_mode_end_bits: direct_probe
                    .residual_histogram_probe
                    .as_ref()
                    .and_then(|probe| probe.entropy_mode_end_bits),
                direct_residual_log_alpha_size_end_bits: direct_probe
                    .residual_histogram_probe
                    .as_ref()
                    .and_then(|probe| probe.log_alpha_size_end_bits),
                direct_residual_uint_config_end_bits_by_histogram: direct_probe
                    .residual_histogram_probe
                    .as_ref()
                    .map(|probe| probe.uint_config_end_bits_by_histogram.clone())
                    .unwrap_or_default(),
                direct_residual_uint_config_end_bits: direct_probe
                    .residual_histogram_probe
                    .as_ref()
                    .and_then(|probe| probe.uint_config_end_bits),
                direct_residual_use_prefix_code: direct_probe
                    .residual_histogram_probe
                    .as_ref()
                    .and_then(|probe| probe.use_prefix_code),
                direct_residual_log_alpha_size: direct_probe
                    .residual_histogram_probe
                    .as_ref()
                    .and_then(|probe| probe.log_alpha_size),
                direct_residual_failed_histogram_index: direct_probe
                    .residual_histogram_probe
                    .as_ref()
                    .and_then(|probe| probe.failed_histogram_index),
                direct_residual_error_stage: direct_probe
                    .residual_histogram_probe
                    .as_ref()
                    .and_then(|probe| probe.error_stage)
                    .map(VarDctHistogramProbeStage::from),
                direct_residual_ans_histograms: direct_probe
                    .residual_histogram_probe
                    .as_ref()
                    .map(|probe| {
                        probe
                            .ans_histograms
                            .iter()
                            .map(VarDctAnsHistogramProbe::from)
                            .collect()
                    })
                    .unwrap_or_default(),
                tree_start_bits: global.bits_consumed,
                direct_error: None,
                tree,
            })
        }
        Err(error) => {
            let mut direct_probe = BitReader::new(bytes);
            direct_probe.skip_bits(global.bits_consumed)?;
            let direct_probe =
                probe_modular_global_tree_coding(&mut direct_probe, metadata, frame_header);
            let start = global.bits_consumed;
            let end = (global.bits_consumed + 64).min(bytes.len() * 8);
            for offset in start..end {
                let mut probe = BitReader::new(bytes);
                probe.skip_bits(offset)?;
                if let Ok(tree) =
                    read_modular_global_tree_coding(&mut probe, metadata, frame_header)
                {
                    return Ok(VarDctModularGlobalTreeRead {
                        direct_start_bits: global.bits_consumed,
                        direct_tree_end_bits: direct_probe.tree_end_bits,
                        direct_tree_node_count: direct_probe.tree_node_count,
                        direct_tree_leaf_count: direct_probe.tree_leaf_count,
                        direct_tree_leaves: direct_probe
                            .tree_leaves
                            .iter()
                            .map(VarDctMaTreeLeafProbe::from)
                            .collect(),
                        direct_error_bits: direct_probe.error_bits,
                        direct_residual_context_count: direct_probe.residual_context_count,
                        direct_residual_histogram_count: direct_probe.residual_histogram_count,
                        direct_residual_context_map_entries: direct_probe
                            .residual_histogram_probe
                            .as_ref()
                            .map(|probe| probe.context_map_entries.clone())
                            .unwrap_or_default(),
                        direct_residual_context_map_raw_entries: direct_probe
                            .residual_histogram_probe
                            .as_ref()
                            .map(|probe| probe.context_map_raw_entries.clone())
                            .unwrap_or_default(),
                        direct_residual_context_map_distinct_entries: direct_probe
                            .residual_histogram_probe
                            .as_ref()
                            .map(|probe| probe.context_map_distinct_entries.clone())
                            .unwrap_or_default(),
                        direct_residual_context_map_histogram_usage_counts: direct_probe
                            .residual_histogram_probe
                            .as_ref()
                            .map(|probe| probe.context_map_histogram_usage_counts.clone())
                            .unwrap_or_default(),
                        direct_residual_context_map_max_entry: direct_probe
                            .residual_histogram_probe
                            .as_ref()
                            .and_then(|probe| probe.context_map_max_entry),
                        direct_residual_context_map_symbol_entries: direct_probe
                            .residual_histogram_probe
                            .as_ref()
                            .map(|probe| {
                                probe
                                    .context_map_symbol_entries
                                    .iter()
                                    .map(VarDctContextMapSymbolProbe::from)
                                    .collect()
                            })
                            .unwrap_or_default(),
                        direct_residual_lz77_end_bits: direct_probe
                            .residual_histogram_probe
                            .as_ref()
                            .and_then(|probe| probe.lz77_end_bits),
                        direct_residual_context_map_end_bits: direct_probe
                            .residual_histogram_probe
                            .as_ref()
                            .and_then(|probe| probe.context_map_end_bits),
                        direct_residual_entropy_mode_end_bits: direct_probe
                            .residual_histogram_probe
                            .as_ref()
                            .and_then(|probe| probe.entropy_mode_end_bits),
                        direct_residual_log_alpha_size_end_bits: direct_probe
                            .residual_histogram_probe
                            .as_ref()
                            .and_then(|probe| probe.log_alpha_size_end_bits),
                        direct_residual_uint_config_end_bits_by_histogram: direct_probe
                            .residual_histogram_probe
                            .as_ref()
                            .map(|probe| probe.uint_config_end_bits_by_histogram.clone())
                            .unwrap_or_default(),
                        direct_residual_uint_config_end_bits: direct_probe
                            .residual_histogram_probe
                            .as_ref()
                            .and_then(|probe| probe.uint_config_end_bits),
                        direct_residual_use_prefix_code: direct_probe
                            .residual_histogram_probe
                            .as_ref()
                            .and_then(|probe| probe.use_prefix_code),
                        direct_residual_log_alpha_size: direct_probe
                            .residual_histogram_probe
                            .as_ref()
                            .and_then(|probe| probe.log_alpha_size),
                        direct_residual_failed_histogram_index: direct_probe
                            .residual_histogram_probe
                            .as_ref()
                            .and_then(|probe| probe.failed_histogram_index),
                        direct_residual_error_stage: direct_probe
                            .residual_histogram_probe
                            .as_ref()
                            .and_then(|probe| probe.error_stage)
                            .map(VarDctHistogramProbeStage::from),
                        direct_residual_ans_histograms: direct_probe
                            .residual_histogram_probe
                            .as_ref()
                            .map(|probe| {
                                probe
                                    .ans_histograms
                                    .iter()
                                    .map(VarDctAnsHistogramProbe::from)
                                    .collect()
                            })
                            .unwrap_or_default(),
                        tree_start_bits: offset,
                        direct_error: Some(error.clone()),
                        tree,
                    });
                }
            }
            Err(error)
        }
    }
}

struct VarDctModularGlobalTreeRead {
    direct_start_bits: usize,
    direct_tree_end_bits: Option<usize>,
    direct_tree_node_count: Option<usize>,
    direct_tree_leaf_count: Option<usize>,
    direct_tree_leaves: Vec<VarDctMaTreeLeafProbe>,
    direct_error_bits: Option<usize>,
    direct_residual_context_count: Option<usize>,
    direct_residual_histogram_count: Option<usize>,
    direct_residual_context_map_entries: Vec<u8>,
    direct_residual_context_map_raw_entries: Vec<u8>,
    direct_residual_context_map_distinct_entries: Vec<u8>,
    direct_residual_context_map_histogram_usage_counts: Vec<usize>,
    direct_residual_context_map_max_entry: Option<u8>,
    direct_residual_context_map_symbol_entries: Vec<VarDctContextMapSymbolProbe>,
    direct_residual_lz77_end_bits: Option<usize>,
    direct_residual_context_map_end_bits: Option<usize>,
    direct_residual_entropy_mode_end_bits: Option<usize>,
    direct_residual_log_alpha_size_end_bits: Option<usize>,
    direct_residual_uint_config_end_bits_by_histogram: Vec<usize>,
    direct_residual_uint_config_end_bits: Option<usize>,
    direct_residual_use_prefix_code: Option<bool>,
    direct_residual_log_alpha_size: Option<usize>,
    direct_residual_failed_histogram_index: Option<usize>,
    direct_residual_error_stage: Option<VarDctHistogramProbeStage>,
    direct_residual_ans_histograms: Vec<VarDctAnsHistogramProbe>,
    tree_start_bits: usize,
    direct_error: Option<Error>,
    tree: ModularTreeCoding,
}

fn vardct_dc_channel_plan(
    frame_header: &FrameHeader,
    payload: &VarDctDcGroupPayloadMetadata,
) -> Result<Vec<ModularGroupChannelPlan>> {
    let mut channels = Vec::with_capacity(3);
    for channel_index in 0..3 {
        let (hshift, vshift) = vardct_chroma_shift(frame_header, channel_index)?;
        if hshift < 0 || vshift < 0 {
            return Err(Error::InvalidCodestream("invalid VarDCT DC chroma shift"));
        }
        channels.push(ModularGroupChannelPlan {
            channel_index,
            width: payload.group.width.div_ceil(8) >> hshift as u32,
            height: payload.group.height.div_ceil(8) >> vshift as u32,
            x0: 0,
            y0: 0,
            hshift,
            vshift,
        });
    }
    Ok(channels)
}

fn vardct_modular_ac_channel_plan(
    metadata: &ImageMetadata,
    frame_header: &FrameHeader,
    group: VarDctGroupMetadata,
    min_shift: i32,
    max_shift: i32,
) -> Result<Vec<ModularGroupChannelPlan>> {
    let frame_upsampling_log = ceil_log2_nonzero(frame_header.upsampling as usize) as i32;
    let mut channels = Vec::new();
    for (extra_index, _) in metadata.extra_channels.iter().enumerate() {
        let upsampling = *frame_header
            .extra_channel_upsampling
            .get(extra_index)
            .ok_or(Error::InvalidCodestream(
                "missing extra-channel upsampling factor",
            ))?;
        if upsampling == 0 {
            return Err(Error::InvalidCodestream("zero extra-channel upsampling"));
        }
        let shift = ceil_log2_nonzero(upsampling as usize) as i32 - frame_upsampling_log;
        if shift < min_shift || shift > max_shift {
            continue;
        }

        let channel_width = frame_header.frame_size.width.div_ceil(upsampling);
        let channel_height = frame_header.frame_size.height.div_ceil(upsampling);
        let Some((x0, width)) =
            shifted_vardct_extra_axis(group.x, group.width, channel_width, shift)?
        else {
            continue;
        };
        let Some((y0, height)) =
            shifted_vardct_extra_axis(group.y, group.height, channel_height, shift)?
        else {
            continue;
        };
        channels.push(ModularGroupChannelPlan {
            channel_index: 3 + extra_index,
            width,
            height,
            x0,
            y0,
            hshift: shift,
            vshift: shift,
        });
    }
    Ok(channels)
}

fn shifted_vardct_extra_axis(
    start: u32,
    size: u32,
    channel_size: u32,
    shift: i32,
) -> Result<Option<(u32, u32)>> {
    let (start, size) = if shift >= 0 {
        let shift = shift as u32;
        (start >> shift, size >> shift)
    } else {
        let shift = (-shift) as u32;
        (
            start
                .checked_shl(shift)
                .ok_or(Error::InvalidCodestream("extra-channel region overflow"))?,
            size.checked_shl(shift)
                .ok_or(Error::InvalidCodestream("extra-channel region overflow"))?,
        )
    };
    if start >= channel_size {
        return Ok(None);
    }
    let size = size.min(channel_size - start);
    if size == 0 {
        return Ok(None);
    }
    Ok(Some((start, size)))
}

fn read_vardct_ac_metadata_count(
    reader: &mut BitReader<'_>,
    payload: &VarDctDcGroupPayloadMetadata,
) -> Result<usize> {
    let upper_bound = (payload.group.width.div_ceil(8) as usize)
        .checked_mul(payload.group.height.div_ceil(8) as usize)
        .ok_or(Error::InvalidCodestream("VarDCT AC metadata size overflow"))?;
    Ok(reader.read_bits(ceil_log2_nonzero(upper_bound))? as usize + 1)
}

fn vardct_ac_metadata_channel_plan(
    payload: &VarDctDcGroupPayloadMetadata,
    count: usize,
) -> Vec<ModularGroupChannelPlan> {
    let width_blocks = payload.group.width.div_ceil(8);
    let height_blocks = payload.group.height.div_ceil(8);
    let color_tiles_x = width_blocks.div_ceil(8);
    let color_tiles_y = height_blocks.div_ceil(8);
    vec![
        ModularGroupChannelPlan {
            channel_index: 0,
            width: color_tiles_x,
            height: color_tiles_y,
            x0: 0,
            y0: 0,
            hshift: 3,
            vshift: 3,
        },
        ModularGroupChannelPlan {
            channel_index: 1,
            width: color_tiles_x,
            height: color_tiles_y,
            x0: 0,
            y0: 0,
            hshift: 3,
            vshift: 3,
        },
        ModularGroupChannelPlan {
            channel_index: 2,
            width: count as u32,
            height: 2,
            x0: 0,
            y0: 0,
            hshift: 0,
            vshift: 0,
        },
        ModularGroupChannelPlan {
            channel_index: 3,
            width: width_blocks,
            height: height_blocks,
            x0: 0,
            y0: 0,
            hshift: 0,
            vshift: 0,
        },
    ]
}

fn vardct_chroma_shift(frame_header: &FrameHeader, channel: usize) -> Result<(i32, i32)> {
    const H_SHIFT: [i32; 4] = [0, 1, 1, 0];
    const V_SHIFT: [i32; 4] = [0, 1, 0, 1];
    let mode = *frame_header
        .chroma_subsampling
        .channel_mode
        .get(channel)
        .ok_or(Error::InvalidCodestream("invalid chroma channel"))? as usize;
    let hshift = i32::from(frame_header.chroma_subsampling.max_h_shift)
        - H_SHIFT
            .get(mode)
            .copied()
            .ok_or(Error::InvalidCodestream("invalid chroma mode"))?;
    let vshift = i32::from(frame_header.chroma_subsampling.max_v_shift)
        - V_SHIFT
            .get(mode)
            .copied()
            .ok_or(Error::InvalidCodestream("invalid chroma mode"))?;
    Ok((hshift, vshift))
}

fn ceil_log2_nonzero(value: usize) -> usize {
    usize::BITS as usize - (value - 1).leading_zeros() as usize
}

fn read_vardct_global_metadata(
    codestream: &[u8],
    section: &VarDctSectionPayloadMetadata,
) -> Result<VarDctGlobalMetadata> {
    let payload = codestream
        .get(section.payload_range.clone())
        .ok_or(Error::InvalidCodestream("frame section outside codestream"))?;
    let mut reader = BitReader::new(payload);
    let dc_dequant = read_vardct_dc_dequant(&mut reader)?;
    let dc_dequant_end_bits = reader.bits_consumed();
    let quantizer = read_vardct_quantizer(&mut reader)?;
    let quantizer_end_bits = reader.bits_consumed();
    let block_context_map = read_vardct_block_context_map(&mut reader)?;
    let block_context_end_bits = reader.bits_consumed();
    let color_correlation = read_vardct_color_correlation(&mut reader)?;
    let color_correlation_end_bits = reader.bits_consumed();
    let cursor = VarDctGlobalCursorMetadata {
        dc_dequant_default_end_bits: dc_dequant.default_end_bits,
        dc_dequant_end_bits,
        quantizer_global_scale_end_bits: quantizer.global_scale_end_bits,
        quantizer_quant_dc_end_bits: quantizer.quant_dc_end_bits,
        quantizer_end_bits,
        block_context_default_end_bits: block_context_map.default_end_bits,
        block_context_dc_thresholds_end_bits: block_context_map.dc_thresholds_end_bits,
        block_context_qf_thresholds_end_bits: block_context_map.qf_thresholds_end_bits,
        block_context_map_start_bits: block_context_map.context_map_start_bits,
        block_context_map_end_bits: block_context_map.context_map_end_bits,
        block_context_end_bits,
        color_correlation_default_end_bits: color_correlation.default_end_bits,
        color_correlation_color_factor_end_bits: color_correlation.color_factor_end_bits,
        color_correlation_base_x_end_bits: color_correlation.base_x_end_bits,
        color_correlation_base_b_end_bits: color_correlation.base_b_end_bits,
        color_correlation_ytox_dc_end_bits: color_correlation.ytox_dc_end_bits,
        color_correlation_ytob_dc_end_bits: color_correlation.ytob_dc_end_bits,
        color_correlation_end_bits,
    };
    Ok(VarDctGlobalMetadata {
        section: section.clone(),
        cursor,
        dc_dequant: dc_dequant.metadata,
        quantizer: quantizer.metadata,
        block_context_map: block_context_map.metadata,
        color_correlation: color_correlation.metadata,
        bits_consumed: reader.bits_consumed(),
    })
}

fn read_vardct_dc_dequant(reader: &mut BitReader<'_>) -> Result<VarDctDcDequantRead> {
    let all_default = reader.read_bool()?;
    let default_end_bits = reader.bits_consumed();
    let coefficients = if all_default {
        None
    } else {
        let mut coefficients = [0.0f32; 3];
        for coefficient in &mut coefficients {
            *coefficient = reader.read_f16()? * (1.0 / 128.0);
            if *coefficient <= 0.0 {
                return Err(Error::InvalidCodestream(
                    "invalid DC dequant matrix coefficient",
                ));
            }
        }
        Some(coefficients)
    };
    Ok(VarDctDcDequantRead {
        metadata: VarDctDcDequantMetadata {
            all_default,
            coefficients,
        },
        default_end_bits,
    })
}

fn read_vardct_quantizer(reader: &mut BitReader<'_>) -> Result<VarDctQuantizerRead> {
    const GLOBAL_SCALE_DENOM: f32 = 65_536.0;
    let global_scale = reader.read_u32_selector(
        bits_offset(11, 1),
        bits_offset(11, 2049),
        bits_offset(12, 4097),
        bits_offset(16, 8193),
    )?;
    let global_scale_end_bits = reader.bits_consumed();
    let quant_dc = reader.read_u32_selector(
        val(16),
        bits_offset(5, 1),
        bits_offset(8, 1),
        bits_offset(16, 1),
    )?;
    let quant_dc_end_bits = reader.bits_consumed();
    if global_scale == 0 || quant_dc == 0 {
        return Err(Error::InvalidCodestream("invalid VarDCT quantizer"));
    }
    let inv_global_scale = GLOBAL_SCALE_DENOM / global_scale as f32;
    Ok(VarDctQuantizerRead {
        metadata: VarDctQuantizerMetadata {
            global_scale,
            quant_dc,
            scale: global_scale as f32 / GLOBAL_SCALE_DENOM,
            inv_global_scale,
            inv_quant_dc: inv_global_scale / quant_dc as f32,
        },
        global_scale_end_bits,
        quant_dc_end_bits,
    })
}

fn read_vardct_block_context_map(reader: &mut BitReader<'_>) -> Result<VarDctBlockContextMapRead> {
    const NUM_ORDERS: usize = 13;
    const DEFAULT_CONTEXT_MAP_SIZE: usize = 3 * NUM_ORDERS;
    const DEFAULT_NUM_CONTEXTS: usize = 15;

    let all_default = reader.read_bool()?;
    let default_end_bits = reader.bits_consumed();
    if all_default {
        return Ok(VarDctBlockContextMapRead {
            metadata: VarDctBlockContextMapMetadata {
                all_default,
                dc_thresholds: [Vec::new(), Vec::new(), Vec::new()],
                qf_thresholds: Vec::new(),
                context_map_size: DEFAULT_CONTEXT_MAP_SIZE,
                num_contexts: DEFAULT_NUM_CONTEXTS,
                num_dc_contexts: 1,
                context_map_probe: None,
            },
            default_end_bits,
            dc_thresholds_end_bits: default_end_bits,
            qf_thresholds_end_bits: default_end_bits,
            context_map_start_bits: None,
            context_map_end_bits: None,
            context_map_probe: None,
        });
    }

    let mut dc_thresholds = [Vec::new(), Vec::new(), Vec::new()];
    let mut num_dc_contexts = 1usize;
    for thresholds in &mut dc_thresholds {
        let len = reader.read_bits(4)? as usize;
        num_dc_contexts = num_dc_contexts
            .checked_mul(len + 1)
            .ok_or(Error::InvalidCodestream(
                "VarDCT block context map is too large",
            ))?;
        thresholds.reserve(len);
        for _ in 0..len {
            let threshold = reader.read_u32_selector(
                bits_offset(4, 0),
                bits_offset(8, 16),
                bits_offset(16, 272),
                bits_offset(32, 65_808),
            )?;
            thresholds.push(unpack_signed(threshold));
        }
    }
    let dc_thresholds_end_bits = reader.bits_consumed();

    let qf_len = reader.read_bits(4)? as usize;
    let mut qf_thresholds = Vec::with_capacity(qf_len);
    for _ in 0..qf_len {
        let threshold = reader.read_u32_selector(
            bits_offset(2, 0),
            bits_offset(3, 4),
            bits_offset(5, 12),
            bits_offset(8, 44),
        )?;
        qf_thresholds.push(threshold + 1);
    }
    let qf_thresholds_end_bits = reader.bits_consumed();

    if num_dc_contexts * (qf_thresholds.len() + 1) > 64 {
        return Err(Error::InvalidCodestream(
            "VarDCT block context map is too large",
        ));
    }

    let context_map_size = 3 * NUM_ORDERS * num_dc_contexts * (qf_thresholds.len() + 1);
    let mut context_map = vec![0; context_map_size];
    let context_map_start_bits = reader.bits_consumed();
    let mut context_map_probe_reader = reader.clone();
    let num_contexts = decode_context_map(reader, &mut context_map)?;
    let context_map_end_bits = reader.bits_consumed();
    let context_map_probe =
        probe_decode_context_map(&mut context_map_probe_reader, context_map_size);
    if num_contexts > 16 {
        return Err(Error::InvalidCodestream(
            "VarDCT block context map has too many contexts",
        ));
    }
    Ok(VarDctBlockContextMapRead {
        metadata: VarDctBlockContextMapMetadata {
            all_default,
            dc_thresholds,
            qf_thresholds,
            context_map_size,
            num_contexts,
            num_dc_contexts,
            context_map_probe: Some(VarDctContextMapProbe::from(&context_map_probe)),
        },
        default_end_bits,
        dc_thresholds_end_bits,
        qf_thresholds_end_bits,
        context_map_start_bits: Some(context_map_start_bits),
        context_map_end_bits: Some(context_map_end_bits),
        context_map_probe: Some(VarDctContextMapProbe::from(&context_map_probe)),
    })
}

fn read_vardct_color_correlation(reader: &mut BitReader<'_>) -> Result<VarDctColorCorrelationRead> {
    const DEFAULT_COLOR_FACTOR: u32 = 84;
    const DEFAULT_BASE_CORRELATION_X: f32 = 0.0;
    const DEFAULT_BASE_CORRELATION_B: f32 = 1.0;

    let all_default = reader.read_bool()?;
    let default_end_bits = reader.bits_consumed();
    if all_default {
        return Ok(VarDctColorCorrelationRead {
            metadata: VarDctColorCorrelationMetadata {
                all_default,
                color_factor: DEFAULT_COLOR_FACTOR,
                base_correlation_x: DEFAULT_BASE_CORRELATION_X,
                base_correlation_b: DEFAULT_BASE_CORRELATION_B,
                ytox_dc: 0,
                ytob_dc: 0,
            },
            default_end_bits,
            color_factor_end_bits: None,
            base_x_end_bits: None,
            base_b_end_bits: None,
            ytox_dc_end_bits: None,
            ytob_dc_end_bits: None,
        });
    }

    let color_factor = reader.read_u32_selector(
        val(DEFAULT_COLOR_FACTOR),
        val(256),
        bits_offset(8, 2),
        bits_offset(16, 258),
    )?;
    let color_factor_end_bits = reader.bits_consumed();
    if color_factor == 0 {
        return Err(Error::InvalidCodestream("invalid VarDCT color factor"));
    }
    let base_correlation_x = reader.read_f16()?;
    let base_x_end_bits = reader.bits_consumed();
    if base_correlation_x.abs() > 4.0 {
        return Err(Error::InvalidCodestream(
            "VarDCT base X correlation is out of range",
        ));
    }
    let base_correlation_b = reader.read_f16()?;
    let base_b_end_bits = reader.bits_consumed();
    if base_correlation_b.abs() > 4.0 {
        return Err(Error::InvalidCodestream(
            "VarDCT base B correlation is out of range",
        ));
    }
    let ytox_dc = reader.read_bits(8)? as i32 - 128;
    let ytox_dc_end_bits = reader.bits_consumed();
    let ytob_dc = reader.read_bits(8)? as i32 - 128;
    let ytob_dc_end_bits = reader.bits_consumed();

    Ok(VarDctColorCorrelationRead {
        metadata: VarDctColorCorrelationMetadata {
            all_default,
            color_factor,
            base_correlation_x,
            base_correlation_b,
            ytox_dc,
            ytob_dc,
        },
        default_end_bits,
        color_factor_end_bits: Some(color_factor_end_bits),
        base_x_end_bits: Some(base_x_end_bits),
        base_b_end_bits: Some(base_b_end_bits),
        ytox_dc_end_bits: Some(ytox_dc_end_bits),
        ytob_dc_end_bits: Some(ytob_dc_end_bits),
    })
}

fn section_payload_metadata(
    codestream: &[u8],
    frame_data: &FrameData,
    section: &VarDctSectionMetadata,
) -> Result<VarDctSectionPayloadMetadata> {
    let frame_section = matching_frame_section(frame_data, section)?;
    let payload_range = validated_section_payload_range(codestream, frame_section)?;
    Ok(VarDctSectionPayloadMetadata {
        section: section.clone(),
        payload_range,
    })
}

fn matching_frame_section<'a>(
    frame_data: &'a FrameData,
    section: &VarDctSectionMetadata,
) -> Result<&'a FrameSection> {
    let frame_section = frame_data
        .sections
        .get(section.section_physical_index)
        .ok_or(Error::InvalidCodestream("missing VarDCT frame section"))?;
    if frame_section.logical_id != section.section_logical_id
        || frame_section.kind != section.section_kind
        || frame_section.codestream_offset != section.codestream_offset
        || frame_section.size != section.payload_size
    {
        return Err(Error::InvalidCodestream("mismatched VarDCT frame section"));
    }
    Ok(frame_section)
}

fn validated_section_payload_range(
    codestream: &[u8],
    section: &FrameSection,
) -> Result<Range<usize>> {
    let range = section_payload_range(section)?;
    codestream
        .get(range.clone())
        .ok_or(Error::InvalidCodestream("frame section outside codestream"))?;
    Ok(range)
}

fn classify_vardct_sections(
    sections: &[VarDctSectionMetadata],
    ac_groups: &[VarDctGroupMetadata],
    dc_groups: &[VarDctGroupMetadata],
) -> VarDctSectionBuckets {
    let global_section = sections
        .iter()
        .find(|section| {
            matches!(
                section.section_kind,
                FrameSectionKind::Combined | FrameSectionKind::DcGlobal
            )
        })
        .cloned();
    let ac_global_section = sections
        .iter()
        .find(|section| matches!(section.section_kind, FrameSectionKind::AcGlobal))
        .cloned();
    let dc_group_sections = sections
        .iter()
        .filter_map(|section| match section.section_kind {
            FrameSectionKind::DcGroup { group } => {
                dc_groups
                    .get(group)
                    .copied()
                    .map(|group| VarDctGroupSectionMetadata {
                        section: section.clone(),
                        group,
                    })
            }
            _ => None,
        })
        .collect();
    let ac_group_sections = sections
        .iter()
        .filter_map(|section| match section.section_kind {
            FrameSectionKind::AcGroup { pass, group } => {
                ac_groups
                    .get(group)
                    .copied()
                    .map(|group| VarDctPassGroupSectionMetadata {
                        section: section.clone(),
                        pass,
                        group,
                    })
            }
            _ => None,
        })
        .collect();
    VarDctSectionBuckets {
        is_combined: sections
            .iter()
            .any(|section| matches!(section.section_kind, FrameSectionKind::Combined)),
        global_section,
        ac_global_section,
        ac_group_sections,
        dc_group_sections,
    }
}

fn group_metadata(
    groups_x: u32,
    groups_y: u32,
    group_dim: u32,
    frame_width: u32,
    frame_height: u32,
) -> Vec<VarDctGroupMetadata> {
    let mut groups = Vec::with_capacity(groups_x as usize * groups_y as usize);
    for gy in 0..groups_y {
        for gx in 0..groups_x {
            let x = gx * group_dim;
            let y = gy * group_dim;
            groups.push(VarDctGroupMetadata {
                group: groups.len(),
                x,
                y,
                width: group_dim.min(frame_width.saturating_sub(x)),
                height: group_dim.min(frame_height.saturating_sub(y)),
            });
        }
    }
    groups
}

fn group_intersects_region(group: &VarDctGroupMetadata, region: ImageRegion) -> bool {
    let Some(group_right) = group.x.checked_add(group.width) else {
        return true;
    };
    let Some(group_bottom) = group.y.checked_add(group.height) else {
        return true;
    };
    let Some(region_right) = region.x.checked_add(region.width) else {
        return true;
    };
    let Some(region_bottom) = region.y.checked_add(region.height) else {
        return true;
    };
    group.x < region_right
        && region.x < group_right
        && group.y < region_bottom
        && region.y < group_bottom
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{
        AnimationFrame, BlendingInfo, ColorTransform, FrameGroupLayout, FrameOrigin, FrameSize,
        FrameType, LoopFilter, Passes, YCbCrChromaSubsampling,
    };
    use crate::frame_data::{FrameSection, FrameToc};
    use crate::transform::OpsinInverseMatrix;

    fn default_vardct_opsin_params() -> VarDctOpsinParams {
        vardct_opsin_params_from_matrix(
            DEFAULT_INVERSE_OPSIN_MATRIX,
            DEFAULT_OPSIN_BIASES,
            ImageMetadata::default().tone_mapping.intensity_target,
        )
    }

    #[test]
    fn classifies_multi_section_vardct_sections() {
        let ac_groups = vec![group(0, 0, 0, 128, 128), group(1, 128, 0, 128, 128)];
        let dc_groups = vec![group(0, 0, 0, 256, 128)];
        let sections = vec![
            section(0, 0, FrameSectionKind::DcGlobal),
            section(1, 1, FrameSectionKind::DcGroup { group: 0 }),
            section(2, 2, FrameSectionKind::AcGlobal),
            section(3, 3, FrameSectionKind::AcGroup { pass: 0, group: 0 }),
            section(4, 4, FrameSectionKind::AcGroup { pass: 0, group: 1 }),
        ];

        let buckets = classify_vardct_sections(&sections, &ac_groups, &dc_groups);

        assert!(!buckets.is_combined);
        assert_eq!(
            buckets.global_section.as_ref().unwrap().section_kind,
            FrameSectionKind::DcGlobal
        );
        assert_eq!(
            buckets.ac_global_section.as_ref().unwrap().section_kind,
            FrameSectionKind::AcGlobal
        );
        assert_eq!(buckets.dc_group_sections.len(), 1);
        assert_eq!(buckets.dc_group_sections[0].group, dc_groups[0]);
        assert_eq!(buckets.ac_group_sections.len(), 2);
        assert_eq!(buckets.ac_group_sections[0].pass, 0);
        assert_eq!(buckets.ac_group_sections[1].group, ac_groups[1]);
    }

    #[test]
    fn selects_group_sections_for_region() {
        let metadata = VarDctFrameMetadata {
            width: 256,
            height: 128,
            group_dim: 128,
            groups_x: 2,
            groups_y: 1,
            dc_groups_x: 1,
            dc_groups_y: 1,
            is_combined: false,
            global_section: Some(section(0, 0, FrameSectionKind::DcGlobal)),
            ac_global_section: Some(section(2, 2, FrameSectionKind::AcGlobal)),
            sections: Vec::new(),
            ac_groups: vec![group(0, 0, 0, 128, 128), group(1, 128, 0, 128, 128)],
            dc_groups: vec![group(0, 0, 0, 256, 128)],
            ac_group_sections: vec![
                VarDctPassGroupSectionMetadata {
                    section: section(3, 3, FrameSectionKind::AcGroup { pass: 0, group: 0 }),
                    pass: 0,
                    group: group(0, 0, 0, 128, 128),
                },
                VarDctPassGroupSectionMetadata {
                    section: section(4, 4, FrameSectionKind::AcGroup { pass: 0, group: 1 }),
                    pass: 0,
                    group: group(1, 128, 0, 128, 128),
                },
            ],
            dc_group_sections: vec![VarDctGroupSectionMetadata {
                section: section(1, 1, FrameSectionKind::DcGroup { group: 0 }),
                group: group(0, 0, 0, 256, 128),
            }],
        };

        let region = ImageRegion {
            x: 140,
            y: 8,
            width: 16,
            height: 16,
        };

        assert_eq!(
            metadata.ac_sections_for_region(region)[0]
                .section
                .section_logical_id,
            4
        );
        assert_eq!(
            metadata.dc_sections_for_region(region)[0]
                .section
                .section_logical_id,
            1
        );
    }

    #[test]
    fn builds_multi_section_vardct_decode_plan() {
        let frame_header = vardct_header(256, 128);
        let frame_data = frame_data(vec![
            frame_section(0, 0, FrameSectionKind::DcGlobal, 10, 3),
            frame_section(1, 1, FrameSectionKind::DcGroup { group: 0 }, 13, 5),
            frame_section(2, 2, FrameSectionKind::AcGlobal, 18, 7),
            frame_section(
                3,
                3,
                FrameSectionKind::AcGroup { pass: 0, group: 0 },
                25,
                11,
            ),
            frame_section(
                4,
                4,
                FrameSectionKind::AcGroup { pass: 0, group: 1 },
                36,
                13,
            ),
        ]);
        let mut codestream = vec![0; 64];
        codestream[10] = 1;
        codestream[12] = 0b0000_0011;
        codestream[13] = 0b0000_1000;

        let metadata = ImageMetadata::default();
        let transform_data = CustomTransformData::default();
        let plan = read_vardct_decode_plan(
            &codestream,
            &metadata,
            &transform_data,
            &frame_header,
            &frame_data,
        )
        .unwrap()
        .unwrap();

        assert!(!plan.frame.is_combined);
        let global = plan.global.as_ref().unwrap();
        assert!(global.dc_dequant.all_default);
        assert_eq!(global.quantizer.global_scale, 1);
        assert_eq!(global.quantizer.quant_dc, 16);
        assert!(global.block_context_map.all_default);
        assert_eq!(global.block_context_map.num_contexts, 15);
        assert_eq!(global.block_context_map.context_map_size, 39);
        assert!(global.color_correlation.all_default);
        assert_eq!(global.color_correlation.color_factor, 84);
        assert_eq!(global.cursor.dc_dequant_end_bits, 1);
        assert_eq!(global.cursor.quantizer_end_bits, 16);
        assert_eq!(global.cursor.block_context_end_bits, 17);
        assert_eq!(global.cursor.color_correlation_end_bits, 18);
        assert_eq!(global.bits_consumed, 18);
        assert_eq!(plan.global_payload.as_ref().unwrap().payload_range, 10..13);
        assert_eq!(
            plan.ac_global_payload.as_ref().unwrap().payload_range,
            18..25
        );
        assert_eq!(plan.dc_group_payloads.len(), 1);
        assert_eq!(plan.dc_group_payloads[0].section.payload_range, 13..18);
        assert_eq!(plan.dc_group_payloads[0].group.group, 0);
        assert_eq!(plan.dc_group_metadata.len(), 1);
        let dc_metadata = &plan.dc_group_metadata[0];
        assert_eq!(dc_metadata.payload, plan.dc_group_payloads[0]);
        assert_eq!(dc_metadata.extra_precision_bits, Some(0));
        assert_eq!(dc_metadata.cursor.extra_precision_start_bits, 0);
        assert_eq!(dc_metadata.cursor.extra_precision_end_bits, Some(2));
        assert_eq!(dc_metadata.cursor.var_dct_dc_start_bits, Some(2));
        assert_eq!(dc_metadata.cursor.var_dct_dc_header_end_bits, Some(6));
        assert_eq!(dc_metadata.cursor.var_dct_dc_end_bits, None);
        assert_eq!(dc_metadata.cursor.modular_dc_start_bits, None);
        let dc_header = dc_metadata.var_dct_dc_header.as_ref().unwrap();
        assert!(!dc_header.use_global_tree);
        assert!(dc_header.weighted_predictor.all_default);
        assert!(dc_header.transforms.is_empty());
        assert_eq!(dc_metadata.parse_error, Some(Error::Truncated));
        assert_eq!(plan.ac_group_payloads.len(), 2);
        assert_eq!(plan.ac_group_payloads[0].section.payload_range, 25..36);
        assert_eq!(plan.ac_group_payloads[0].group.group, 0);
        assert_eq!(plan.ac_group_payloads[1].section.payload_range, 36..49);
        assert_eq!(plan.ac_group_payloads[1].group.group, 1);
    }

    #[test]
    fn inverse_dct_8x8_zero_block_stays_zero() {
        let coefficients = [0.0f32; DCT_BLOCK_SIZE];
        let samples = inverse_dct_8x8(&coefficients);

        assert!(samples.iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn inverse_dct_8x8_dc_only_is_constant() {
        let mut coefficients = [0.0f32; DCT_BLOCK_SIZE];
        coefficients[0] = 8.0;
        let samples = inverse_dct_8x8(&coefficients);

        for sample in samples {
            assert!((sample - 1.0).abs() < 1.0e-6);
        }
    }

    #[test]
    fn inverse_dct_8x8_single_horizontal_ac_has_expected_shape() {
        let mut coefficients = [0.0f32; DCT_BLOCK_SIZE];
        coefficients[1] = 1.0;
        let samples = inverse_dct_8x8(&coefficients);

        assert!((samples[0] - 0.17337999).abs() < 1.0e-6);
        assert!((samples[7] + 0.17337997).abs() < 1.0e-6);
        assert!((samples[8] - samples[0]).abs() < 1.0e-6);
    }

    #[test]
    fn inverse_dct_rect_2x2_zero_block_stays_zero() {
        let coefficients = [0.0f32; 4];
        let samples = inverse_dct_rect(2, 2, &coefficients).unwrap();

        assert_eq!(samples.len(), 4);
        assert!(samples.iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn inverse_dct_rect_2x2_dc_only_is_constant() {
        let mut coefficients = [0.0f32; 4];
        coefficients[0] = 2.0;
        let samples = inverse_dct_rect(2, 2, &coefficients).unwrap();

        for sample in samples {
            assert!((sample - 1.0).abs() < 1.0e-6);
        }
    }

    #[test]
    fn inverse_dct_rect_4x8_dc_only_is_constant() {
        let mut coefficients = [0.0f32; 32];
        coefficients[0] = (32.0f32).sqrt();
        let samples = inverse_dct_rect(4, 8, &coefficients).unwrap();

        assert_eq!(samples.len(), 32);
        for sample in samples {
            assert!((sample - 1.0).abs() < 1.0e-6);
        }
    }

    #[test]
    fn large_transform_coefficient_lookup_crosses_block_grid() {
        let mut grid = VarDctAcDequantizedGrid {
            group: 0,
            pass: 0,
            width_blocks: 4,
            height_blocks: 3,
            per_channel: [
                VarDctAcDequantizedChannelGrid::new(4 * 3 * DCT_BLOCK_SIZE),
                VarDctAcDequantizedChannelGrid::new(4 * 3 * DCT_BLOCK_SIZE),
                VarDctAcDequantizedChannelGrid::new(4 * 3 * DCT_BLOCK_SIZE),
            ],
        };
        let block = VarDctFirstAcBlock {
            block_x: 1,
            block_y: 1,
            raw_strategy: 4,
        };
        let target_index = ((2 * grid.width_blocks + 2) * DCT_BLOCK_SIZE) + 3 * 8 + 4;
        grid.per_channel[0].coefficients[target_index] = 2.5f32.to_bits();

        let value =
            dequantized_coefficient_for_transform_position(&grid, 0, block, 12, 11).unwrap();

        assert_eq!(value, 2.5);
    }

    #[test]
    fn large_transform_spatial_write_crosses_block_grid() {
        let mut grid = VarDctAcSpatialChannelGrid::new(4 * 3 * DCT_BLOCK_SIZE);
        let block = VarDctFirstAcBlock {
            block_x: 1,
            block_y: 1,
            raw_strategy: 4,
        };

        write_spatial_sample_for_transform_position(&mut grid, 4, block, 12, 11, 3.25).unwrap();

        let target_index = ((2 * 4 + 2) * DCT_BLOCK_SIZE) + 3 * 8 + 4;
        assert_eq!(grid.samples[target_index], 3.25f32.to_bits());
        assert_eq!(grid.nonzero_samples, 1);
    }

    #[test]
    fn inverse_afv_zero_block_stays_zero() {
        let coefficients = [0.0f32; DCT_BLOCK_SIZE];
        let samples = inverse_afv_8x8(0, &coefficients).unwrap();

        assert_eq!(samples.len(), DCT_BLOCK_SIZE);
        assert!(samples.iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn inverse_afv_places_sub_transforms_by_kind() {
        let mut coefficients = [0.0f32; DCT_BLOCK_SIZE];
        coefficients[0] = 1.0;
        let samples = inverse_afv_8x8(3, &coefficients).unwrap();

        assert!((samples[0] - 0.17677669).abs() < 1.0e-6);
        assert!((samples[3 * 8 + 7] - 0.17677669).abs() < 1.0e-6);
        assert!((samples[4 * 8] - 0.25).abs() < 1.0e-6);
        assert!((samples[7 * 8 + 3] - 0.25).abs() < 1.0e-6);
        assert!((samples[4 * 8 + 4] - 1.0).abs() < 1.0e-6);
        assert!((samples[7 * 8 + 7] - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn assembles_vardct_xyb_image_with_edge_clipping() {
        let group = group(0, 0, 0, 10, 9);
        let mut spatial = VarDctAcSpatialGrid {
            group: 0,
            pass: 0,
            width_blocks: 2,
            height_blocks: 2,
            blocks_attempted: 4,
            blocks_transformed: 4,
            blocks_skipped: 0,
            per_channel: [
                VarDctAcSpatialChannelGrid::new(2 * 2 * DCT_BLOCK_SIZE),
                VarDctAcSpatialChannelGrid::new(2 * 2 * DCT_BLOCK_SIZE),
                VarDctAcSpatialChannelGrid::new(2 * 2 * DCT_BLOCK_SIZE),
            ],
        };
        for block_y in 0..2 {
            for block_x in 0..2 {
                for sample_y in 0..8 {
                    for sample_x in 0..8 {
                        let local_x = block_x * 8 + sample_x;
                        let local_y = block_y * 8 + sample_y;
                        let sample = sample_y * 8 + sample_x;
                        let index = ((block_y * 2 + block_x) * DCT_BLOCK_SIZE) + sample;
                        for channel in 0..3 {
                            spatial.per_channel[channel].samples[index] = (1.0
                                + channel as f32 * 1000.0
                                + local_y as f32 * 10.0
                                + local_x as f32)
                                .to_bits();
                        }
                    }
                }
            }
        }
        let frame = vardct_frame_metadata(10, 9);
        let metadata = ac_group_metadata(group, Some(spatial));

        let image = assemble_vardct_xyb_image_from_groups(&frame, &[metadata])
            .unwrap()
            .unwrap();

        assert_eq!(image.width, 10);
        assert_eq!(image.height, 9);
        assert_eq!(image.channels[0].len(), 90);
        assert_eq!(image.sample(0, 0, 0), Some(1.0));
        assert_eq!(image.sample(0, 9, 8), Some(90.0));
        assert_eq!(image.sample(1, 9, 8), Some(1090.0));
        assert_eq!(image.sample(2, 9, 8), Some(2090.0));
        assert_eq!(image.sample(0, 10, 8), None);
        assert_eq!(image.sample(0, 9, 9), None);
    }

    #[test]
    fn vardct_xyb_image_assembly_returns_none_without_spatial_dc_grid() {
        let frame = vardct_frame_metadata(8, 8);
        let metadata = ac_group_metadata(group(0, 0, 0, 8, 8), None);

        let image = assemble_vardct_xyb_image_from_groups(&frame, &[metadata]).unwrap();

        assert!(image.is_none());
    }

    #[test]
    fn vardct_xyb_image_assembly_uses_final_progressive_ac_pass_per_group() {
        let mut pass0_spatial = VarDctAcSpatialGrid {
            group: 0,
            pass: 0,
            width_blocks: 1,
            height_blocks: 1,
            blocks_attempted: 1,
            blocks_transformed: 1,
            blocks_skipped: 0,
            per_channel: [
                VarDctAcSpatialChannelGrid::new(DCT_BLOCK_SIZE),
                VarDctAcSpatialChannelGrid::new(DCT_BLOCK_SIZE),
                VarDctAcSpatialChannelGrid::new(DCT_BLOCK_SIZE),
            ],
        };
        let mut pass2_spatial = pass0_spatial.clone();
        pass2_spatial.pass = 2;
        for channel in 0..3 {
            pass0_spatial.per_channel[channel].samples[0] = (10.0 + channel as f32).to_bits();
            pass2_spatial.per_channel[channel].samples[0] = (20.0 + channel as f32).to_bits();
        }
        let group = group(0, 0, 0, 8, 8);
        let pass0 = ac_group_metadata_for_pass(0, group, Some(pass0_spatial));
        let pass2 = ac_group_metadata_for_pass(2, group, Some(pass2_spatial));
        let frame = vardct_frame_metadata(8, 8);

        let image = assemble_vardct_xyb_image_from_groups(&frame, &[pass2, pass0])
            .unwrap()
            .unwrap();

        assert_eq!(image.groups_assembled, 1);
        assert_eq!(image.groups_missing, 0);
        assert_eq!(image.sample(0, 0, 0), Some(20.0));
        assert_eq!(image.sample(1, 0, 0), Some(21.0));
        assert_eq!(image.sample(2, 0, 0), Some(22.0));
    }

    #[test]
    fn vardct_xyb_image_assembly_can_select_specific_progressive_ac_pass() {
        let mut pass0_spatial = VarDctAcSpatialGrid {
            group: 0,
            pass: 0,
            width_blocks: 1,
            height_blocks: 1,
            blocks_attempted: 1,
            blocks_transformed: 1,
            blocks_skipped: 0,
            per_channel: [
                VarDctAcSpatialChannelGrid::new(DCT_BLOCK_SIZE),
                VarDctAcSpatialChannelGrid::new(DCT_BLOCK_SIZE),
                VarDctAcSpatialChannelGrid::new(DCT_BLOCK_SIZE),
            ],
        };
        let mut pass2_spatial = pass0_spatial.clone();
        pass2_spatial.pass = 2;
        for channel in 0..3 {
            pass0_spatial.per_channel[channel].samples[0] = (10.0 + channel as f32).to_bits();
            pass2_spatial.per_channel[channel].samples[0] = (20.0 + channel as f32).to_bits();
        }
        let group = group(0, 0, 0, 8, 8);
        let pass0 = ac_group_metadata_for_pass(0, group, Some(pass0_spatial));
        let pass2 = ac_group_metadata_for_pass(2, group, Some(pass2_spatial));
        let frame = vardct_frame_metadata(8, 8);

        let image = assemble_vardct_xyb_image_from_groups_with_mode(
            &frame,
            &[pass2, pass0],
            VarDctAssemblyMode::Pass { pass: 0 },
        )
        .unwrap()
        .unwrap();

        assert_eq!(image.groups_assembled, 1);
        assert_eq!(image.groups_missing, 0);
        assert_eq!(image.sample(0, 0, 0), Some(10.0));
        assert_eq!(image.sample(1, 0, 0), Some(11.0));
        assert_eq!(image.sample(2, 0, 0), Some(12.0));
    }

    #[test]
    fn vardct_xyb_image_assembly_missing_specific_pass_returns_none() {
        let spatial = VarDctAcSpatialGrid {
            group: 0,
            pass: 0,
            width_blocks: 1,
            height_blocks: 1,
            blocks_attempted: 1,
            blocks_transformed: 1,
            blocks_skipped: 0,
            per_channel: [
                VarDctAcSpatialChannelGrid::new(DCT_BLOCK_SIZE),
                VarDctAcSpatialChannelGrid::new(DCT_BLOCK_SIZE),
                VarDctAcSpatialChannelGrid::new(DCT_BLOCK_SIZE),
            ],
        };
        let metadata = ac_group_metadata_for_pass(0, group(0, 0, 0, 8, 8), Some(spatial));
        let frame = vardct_frame_metadata(8, 8);

        let image = assemble_vardct_xyb_image_from_groups_with_mode(
            &frame,
            &[metadata],
            VarDctAssemblyMode::Pass { pass: 2 },
        )
        .unwrap();

        assert!(image.is_none());
    }

    #[test]
    fn vardct_xyb_image_for_pass_uses_requested_progressive_ac_pass() {
        let mut pass0_spatial = VarDctAcSpatialGrid {
            group: 0,
            pass: 0,
            width_blocks: 1,
            height_blocks: 1,
            blocks_attempted: 1,
            blocks_transformed: 1,
            blocks_skipped: 0,
            per_channel: [
                VarDctAcSpatialChannelGrid::new(DCT_BLOCK_SIZE),
                VarDctAcSpatialChannelGrid::new(DCT_BLOCK_SIZE),
                VarDctAcSpatialChannelGrid::new(DCT_BLOCK_SIZE),
            ],
        };
        let mut pass1_spatial = pass0_spatial.clone();
        pass1_spatial.pass = 1;
        for channel in 0..3 {
            pass0_spatial.per_channel[channel].samples[0] = (30.0 + channel as f32).to_bits();
            pass1_spatial.per_channel[channel].samples[0] = (40.0 + channel as f32).to_bits();
        }
        let group = group(0, 0, 0, 8, 8);
        let plan = VarDctDecodePlan {
            frame: vardct_frame_metadata(8, 8),
            loop_filter: LoopFilter {
                gab: false,
                epf_iters: 0,
                ..LoopFilter::default()
            },
            opsin_params: default_vardct_opsin_params(),
            epf_metadata: None,
            global: None,
            modular_global_tree_payload_start_bits: None,
            modular_global_tree_payload_end_bits: None,
            modular_global_tree_payload_len_bits: None,
            modular_global_tree_direct_start_bits: None,
            modular_global_tree_direct_start_absolute_bits: None,
            modular_global_tree_direct_start_remaining_bits: None,
            modular_global_tree_direct_tree_end_bits: None,
            modular_global_tree_direct_tree_end_absolute_bits: None,
            modular_global_tree_direct_tree_end_remaining_bits: None,
            modular_global_tree_direct_tree_node_count: None,
            modular_global_tree_direct_tree_leaf_count: None,
            modular_global_tree_direct_tree_leaves: Vec::new(),
            modular_global_tree_direct_error_bits: None,
            modular_global_tree_direct_error_absolute_bits: None,
            modular_global_tree_direct_error_remaining_bits: None,
            modular_global_tree_direct_residual_context_count: None,
            modular_global_tree_direct_residual_histogram_count: None,
            modular_global_tree_direct_residual_context_map_entries: Vec::new(),
            modular_global_tree_direct_residual_context_map_raw_entries: Vec::new(),
            modular_global_tree_direct_residual_context_map_distinct_entries: Vec::new(),
            modular_global_tree_direct_residual_context_map_histogram_usage_counts: Vec::new(),
            modular_global_tree_direct_residual_context_map_max_entry: None,
            modular_global_tree_direct_residual_context_map_symbol_entries: Vec::new(),
            modular_global_tree_direct_residual_lz77_end_bits: None,
            modular_global_tree_direct_residual_context_map_end_bits: None,
            modular_global_tree_direct_residual_entropy_mode_end_bits: None,
            modular_global_tree_direct_residual_log_alpha_size_end_bits: None,
            modular_global_tree_direct_residual_uint_config_end_bits_by_histogram: Vec::new(),
            modular_global_tree_direct_residual_uint_config_end_bits: None,
            modular_global_tree_direct_residual_use_prefix_code: None,
            modular_global_tree_direct_residual_log_alpha_size: None,
            modular_global_tree_direct_residual_failed_histogram_index: None,
            modular_global_tree_direct_residual_error_stage: None,
            modular_global_tree_direct_residual_ans_histograms: Vec::new(),
            modular_global_tree_start_bits: None,
            modular_global_tree_start_absolute_bits: None,
            modular_global_tree_start_remaining_bits: None,
            modular_global_tree_direct_error: None,
            modular_global_tree_error: None,
            global_payload: None,
            ac_global_payload: None,
            ac_global_metadata: None,
            ac_group_payloads: Vec::new(),
            ac_group_metadata: vec![
                ac_group_metadata_for_pass(1, group, Some(pass1_spatial)),
                ac_group_metadata_for_pass(0, group, Some(pass0_spatial)),
            ],
            dc_group_payloads: Vec::new(),
            dc_group_metadata: Vec::new(),
        };

        let image = assemble_vardct_xyb_image_for_pass(&plan, 0)
            .unwrap()
            .unwrap();
        let final_image = assemble_vardct_xyb_image(&plan).unwrap().unwrap();
        let missing = assemble_vardct_xyb_image_for_pass(&plan, 2).unwrap();

        assert_eq!(image.sample(0, 0, 0), Some(30.0));
        assert_eq!(image.sample(1, 0, 0), Some(31.0));
        assert_eq!(image.sample(2, 0, 0), Some(32.0));
        assert_eq!(final_image.sample(0, 0, 0), Some(40.0));
        assert!(missing.is_none());
    }

    #[test]
    fn converts_zero_xyb_to_zero_linear_rgb() {
        let opsin = default_vardct_opsin_params();
        let rgb = xyb_sample_to_linear_rgb(0.0, 0.0, 0.0, &opsin);

        assert!(rgb.iter().all(|sample| sample.abs() < 1.0e-7));
    }

    #[test]
    fn converts_xyb_image_to_linear_rgb() {
        let xyb = VarDctXybImage {
            width: 1,
            height: 1,
            groups_assembled: 1,
            groups_missing: 0,
            channels: [vec![0.1], vec![0.2], vec![0.3]],
        };

        let opsin = default_vardct_opsin_params();
        let rgb = vardct_xyb_to_linear_rgb(&xyb, &opsin);

        assert_eq!(rgb.width, 1);
        assert_eq!(rgb.height, 1);
        assert!((rgb.channels[0][0] - 0.87693274).abs() < 1.0e-6);
        assert!((rgb.channels[1][0] + 0.23766755).abs() < 1.0e-6);
        assert!((rgb.channels[2][0] + 0.31094164).abs() < 1.0e-6);
    }

    #[test]
    fn gaborish_leaves_flat_xyb_image_unchanged() {
        let mut image = VarDctXybImage {
            width: 3,
            height: 2,
            groups_assembled: 1,
            groups_missing: 0,
            channels: [vec![1.5; 6], vec![-2.0; 6], vec![0.25; 6]],
        };

        apply_vardct_gaborish(&mut image, &LoopFilter::default());

        assert!(
            image.channels[0]
                .iter()
                .all(|sample| (*sample - 1.5).abs() < 1.0e-6)
        );
        assert!(
            image.channels[1]
                .iter()
                .all(|sample| (*sample + 2.0).abs() < 1.0e-6)
        );
        assert!(
            image.channels[2]
                .iter()
                .all(|sample| (*sample - 0.25).abs() < 1.0e-6)
        );
    }

    #[test]
    fn gaborish_uses_custom_weights_and_mirrored_borders() {
        let mut image = VarDctXybImage {
            width: 2,
            height: 2,
            groups_assembled: 1,
            groups_missing: 0,
            channels: [vec![1.0, 0.0, 0.0, 0.0], vec![0.0; 4], vec![0.0; 4]],
        };
        let loop_filter = LoopFilter {
            gab: true,
            gab_custom: true,
            gab_weights: Some([0.5, 0.25, 0.0, 0.0, 0.0, 0.0]),
            ..LoopFilter::default()
        };

        apply_vardct_gaborish(&mut image, &loop_filter);

        assert_eq!(image.channels[0], vec![0.5625, 0.1875, 0.1875, 0.0625]);
        assert_eq!(image.channels[1], vec![0.0; 4]);
        assert_eq!(image.channels[2], vec![0.0; 4]);
    }

    #[test]
    fn epf_stage1_skips_when_sigma_is_below_minimum() {
        let original = [vec![1.0, 0.0, 0.0, 0.0], vec![0.0; 4], vec![0.0; 4]];
        let mut image = VarDctXybImage {
            width: 2,
            height: 2,
            groups_assembled: 1,
            groups_missing: 0,
            channels: original.clone(),
        };
        let epf = VarDctEpfMetadata {
            width_blocks: 1,
            height_blocks: 1,
            raw_quant_field: vec![1],
            epf_sharpness: vec![7],
            inv_sigma: vec![(-4.0f32).to_bits()],
            first_block_count: 1,
            raw_quant_checksum: 0,
            epf_sharpness_checksum: 0,
            inv_sigma_checksum: 0,
            parse_error: None,
        };

        let loop_filter = LoopFilter {
            epf_iters: 1,
            ..LoopFilter::default()
        };
        apply_vardct_epf(&mut image, &loop_filter, &epf);

        assert_eq!(image.channels, original);
    }

    #[test]
    fn epf_stage1_smooths_impulse_with_mirrored_borders() {
        let mut image = VarDctXybImage {
            width: 3,
            height: 3,
            groups_assembled: 1,
            groups_missing: 0,
            channels: [
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0; 9],
                vec![0.0; 9],
            ],
        };
        let epf = VarDctEpfMetadata {
            width_blocks: 1,
            height_blocks: 1,
            raw_quant_field: vec![1],
            epf_sharpness: vec![7],
            inv_sigma: vec![(-0.0001f32).to_bits()],
            first_block_count: 1,
            raw_quant_checksum: 0,
            epf_sharpness_checksum: 0,
            inv_sigma_checksum: 0,
            parse_error: None,
        };

        let loop_filter = LoopFilter {
            epf_iters: 1,
            ..LoopFilter::default()
        };
        apply_vardct_epf(&mut image, &loop_filter, &epf);

        assert!(image.channels[0][4] < 1.0);
        assert!(image.channels[0][1] > 0.0);
        assert!(image.channels[0][3] > 0.0);
        assert_eq!(image.channels[1], vec![0.0; 9]);
        assert_eq!(image.channels[2], vec![0.0; 9]);
    }

    #[test]
    fn epf_stage1_directional_sads_match_reference_kernel() {
        let channels = [
            (0..25).map(|sample| sample as f32).collect::<Vec<_>>(),
            vec![0.0; 25],
            vec![0.0; 25],
        ];
        let ctx = EpfSampleContext {
            width: 5,
            height: 5,
            x: 2,
            y: 2,
            channel_scale: [1.0, 0.0, 0.0],
        };

        let sads = epf_stage1_directional_sads(&channels, ctx);

        assert_eq!(sads, [25.0, 5.0, 5.0, 25.0]);
    }

    #[test]
    fn epf_stage0_directional_sads_cover_all_wide_offsets() {
        let channels = [
            (0..49).map(|sample| sample as f32).collect::<Vec<_>>(),
            vec![0.0; 49],
            vec![0.0; 49],
        ];
        let ctx = EpfSampleContext {
            width: 7,
            height: 7,
            x: 3,
            y: 3,
            channel_scale: [1.0, 0.0, 0.0],
        };

        let sads = epf_stage0_directional_sads(&channels, ctx);

        assert_eq!(
            sads,
            [
                10.0, 40.0, 5.0, 30.0, 70.0, 35.0, 35.0, 70.0, 30.0, 5.0, 40.0, 10.0
            ]
        );
    }

    #[test]
    fn epf_stage2_runs_after_stage1_when_enabled() {
        let mut one_pass = VarDctXybImage {
            width: 5,
            height: 5,
            groups_assembled: 1,
            groups_missing: 0,
            channels: [
                vec![
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.4, 1.0, 0.4, 0.0, 0.0,
                    0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                vec![0.0; 25],
                vec![0.0; 25],
            ],
        };
        let mut two_pass = one_pass.clone();
        let epf = VarDctEpfMetadata {
            width_blocks: 1,
            height_blocks: 1,
            raw_quant_field: vec![1],
            epf_sharpness: vec![7],
            inv_sigma: vec![(-0.0001f32).to_bits()],
            first_block_count: 1,
            raw_quant_checksum: 0,
            epf_sharpness_checksum: 0,
            inv_sigma_checksum: 0,
            parse_error: None,
        };
        let one_pass_filter = LoopFilter {
            epf_iters: 1,
            ..LoopFilter::default()
        };
        let two_pass_filter = LoopFilter {
            epf_iters: 2,
            ..LoopFilter::default()
        };

        apply_vardct_epf(&mut one_pass, &one_pass_filter, &epf);
        apply_vardct_epf(&mut two_pass, &two_pass_filter, &epf);

        assert_ne!(two_pass.channels[0], one_pass.channels[0]);
        assert!(two_pass.channels[0][12] < one_pass.channels[0][12]);
        assert_eq!(two_pass.channels[1], vec![0.0; 25]);
        assert_eq!(two_pass.channels[2], vec![0.0; 25]);
    }

    #[test]
    fn epf_stage0_runs_before_stage1_for_three_iterations() {
        let mut one_pass = VarDctXybImage {
            width: 5,
            height: 5,
            groups_assembled: 1,
            groups_missing: 0,
            channels: [
                vec![
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                vec![0.0; 25],
                vec![0.0; 25],
            ],
        };
        let mut three_pass = one_pass.clone();
        let epf = VarDctEpfMetadata {
            width_blocks: 1,
            height_blocks: 1,
            raw_quant_field: vec![1],
            epf_sharpness: vec![7],
            inv_sigma: vec![(-0.0001f32).to_bits()],
            first_block_count: 1,
            raw_quant_checksum: 0,
            epf_sharpness_checksum: 0,
            inv_sigma_checksum: 0,
            parse_error: None,
        };
        let one_pass_filter = LoopFilter {
            epf_iters: 1,
            ..LoopFilter::default()
        };
        let three_pass_filter = LoopFilter {
            epf_iters: 3,
            ..LoopFilter::default()
        };

        apply_vardct_epf(&mut one_pass, &one_pass_filter, &epf);
        apply_vardct_epf(&mut three_pass, &three_pass_filter, &epf);

        assert_ne!(three_pass.channels[0], one_pass.channels[0]);
        assert!(three_pass.channels[0][12] < one_pass.channels[0][12]);
        assert_eq!(three_pass.channels[1], vec![0.0; 25]);
        assert_eq!(three_pass.channels[2], vec![0.0; 25]);
    }

    #[test]
    fn scales_vardct_opsin_matrix_by_intensity_target() {
        let opsin = vardct_opsin_params_from_matrix(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [-8.0, -27.0, -64.0],
            510.0,
        );

        assert_eq!(
            opsin.inverse_matrix,
            [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0], [3.5, 4.0, 4.5]]
        );
        assert_eq!(opsin.opsin_biases, [-8.0, -27.0, -64.0]);
        assert_eq!(opsin.opsin_biases_cbrt, [-2.0, -3.0, -4.0]);
    }

    #[test]
    fn builds_vardct_opsin_params_from_custom_transform_data() {
        let mut metadata = ImageMetadata::default();
        metadata.tone_mapping.intensity_target = 127.5;
        let transform_data = CustomTransformData {
            opsin_inverse_matrix: Some(OpsinInverseMatrix {
                inverse_matrix: [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
                opsin_biases: [-1.0, -8.0, -27.0],
                quant_biases: [0.1, 0.2, 0.3, 0.4],
            }),
            ..CustomTransformData::default()
        };

        let opsin = vardct_opsin_params(&metadata, &transform_data);

        assert_eq!(
            opsin.inverse_matrix,
            [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6.0]]
        );
        assert_eq!(opsin.opsin_biases, [-1.0, -8.0, -27.0]);
        assert_eq!(opsin.opsin_biases_cbrt, [-1.0, -2.0, -3.0]);
    }

    #[test]
    fn converts_linear_sample_to_srgb8() {
        assert_eq!(linear_sample_to_srgb8(-1.0), 0);
        assert_eq!(linear_sample_to_srgb8(0.0), 0);
        assert_eq!(linear_sample_to_srgb8(0.003_130_8), 10);
        assert_eq!(linear_sample_to_srgb8(0.25), 137);
        assert_eq!(linear_sample_to_srgb8(0.5), 188);
        assert_eq!(linear_sample_to_srgb8(1.0), 255);
        assert_eq!(linear_sample_to_srgb8(2.0), 255);
    }

    #[test]
    fn converts_linear_sample_to_srgb16() {
        assert_eq!(linear_sample_to_srgb16(-1.0), 0);
        assert_eq!(linear_sample_to_srgb16(0.0), 0);
        assert_eq!(linear_sample_to_srgb16(0.003_130_8), 2651);
        assert_eq!(linear_sample_to_srgb16(0.25), 35199);
        assert_eq!(linear_sample_to_srgb16(0.5), 48192);
        assert_eq!(linear_sample_to_srgb16(1.0), 65535);
        assert_eq!(linear_sample_to_srgb16(2.0), 65535);
    }

    #[test]
    fn converts_linear_rgb_image_to_srgb8() {
        let rgb = VarDctRgbImage {
            width: 2,
            height: 1,
            channels: [vec![0.0, 1.0], vec![0.5, 0.25], vec![2.0, -1.0]],
        };

        let image = vardct_linear_rgb_to_srgb8(&rgb);

        assert_eq!(image.width, 2);
        assert_eq!(image.height, 1);
        assert_eq!(image.pixels, vec![0, 188, 255, 255, 137, 0]);
    }

    #[test]
    fn converts_linear_rgb_image_to_srgb16() {
        let rgb = VarDctRgbImage {
            width: 2,
            height: 1,
            channels: [vec![0.0, 1.0], vec![0.5, 0.25], vec![2.0, -1.0]],
        };

        let image = vardct_linear_rgb_to_srgb16(&rgb);

        assert_eq!(image.width, 2);
        assert_eq!(image.height, 1);
        assert_eq!(image.pixels, vec![0, 48192, 65535, 65535, 35199, 0]);
    }

    #[test]
    fn rejects_vardct_section_payload_outside_codestream() {
        let frame_header = vardct_header(8, 8);
        let frame_data = frame_data(vec![frame_section(0, 0, FrameSectionKind::Combined, 8, 8)]);
        let codestream = vec![0; 12];

        let metadata = ImageMetadata::default();
        let transform_data = CustomTransformData::default();
        let error = read_vardct_decode_plan(
            &codestream,
            &metadata,
            &transform_data,
            &frame_header,
            &frame_data,
        )
        .unwrap_err();

        assert_eq!(
            error,
            Error::InvalidCodestream("frame section outside codestream")
        );
    }

    #[test]
    fn rejects_truncated_vardct_global_prefix() {
        let frame_header = vardct_header(8, 8);
        let frame_data = frame_data(vec![frame_section(0, 0, FrameSectionKind::Combined, 0, 1)]);
        let codestream = vec![1];

        let metadata = ImageMetadata::default();
        let transform_data = CustomTransformData::default();
        let error = read_vardct_decode_plan(
            &codestream,
            &metadata,
            &transform_data,
            &frame_header,
            &frame_data,
        )
        .unwrap_err();

        assert_eq!(error, Error::Truncated);
    }

    fn section(
        logical_id: usize,
        physical_index: usize,
        kind: FrameSectionKind,
    ) -> VarDctSectionMetadata {
        VarDctSectionMetadata {
            section_logical_id: logical_id,
            section_physical_index: physical_index,
            section_kind: kind,
            codestream_offset: 100 + physical_index,
            payload_size: 10 + physical_index as u32,
        }
    }

    fn frame_section(
        logical_id: usize,
        physical_index: usize,
        kind: FrameSectionKind,
        codestream_offset: usize,
        size: u32,
    ) -> FrameSection {
        FrameSection {
            logical_id,
            physical_index,
            kind,
            codestream_offset,
            size,
        }
    }

    fn frame_data(sections: Vec<FrameSection>) -> FrameData {
        let payload_size = sections.iter().map(|section| section.size as usize).sum();
        FrameData {
            toc: FrameToc {
                entries: Vec::new(),
                has_permutation: false,
            },
            sections,
            payload_start_offset: 0,
            payload_size,
        }
    }

    fn group(group: usize, x: u32, y: u32, width: u32, height: u32) -> VarDctGroupMetadata {
        VarDctGroupMetadata {
            group,
            x,
            y,
            width,
            height,
        }
    }

    fn vardct_frame_metadata(width: u32, height: u32) -> VarDctFrameMetadata {
        VarDctFrameMetadata {
            width,
            height,
            group_dim: 256,
            groups_x: width.div_ceil(256),
            groups_y: height.div_ceil(256),
            dc_groups_x: width.div_ceil(2048),
            dc_groups_y: height.div_ceil(2048),
            is_combined: false,
            global_section: None,
            ac_global_section: None,
            sections: Vec::new(),
            ac_groups: Vec::new(),
            dc_groups: Vec::new(),
            ac_group_sections: Vec::new(),
            dc_group_sections: Vec::new(),
        }
    }

    fn ac_group_metadata(
        group: VarDctGroupMetadata,
        spatial_with_dc_grid: Option<VarDctAcSpatialGrid>,
    ) -> VarDctAcGroupMetadata {
        VarDctAcGroupMetadata {
            payload: VarDctPassGroupPayloadMetadata {
                section: VarDctSectionPayloadMetadata {
                    section: section(
                        0,
                        0,
                        FrameSectionKind::AcGroup {
                            pass: 0,
                            group: group.group,
                        },
                    ),
                    payload_range: 0..0,
                },
                pass: 0,
                group,
                modular_ac_stream_id: 0,
                modular_ac_min_shift: 0,
                modular_ac_max_shift: 2,
                modular_ac_channels: Vec::new(),
            },
            cursor: VarDctAcGroupCursorMetadata {
                payload_start_bits: 0,
                payload_end_bits: 0,
                histogram_selector_start_bits: 0,
                histogram_selector_end_bits: Some(0),
                ans_state_start_bits: None,
                ans_state_end_bits: None,
                coefficient_stream_start_bits: None,
                modular_ac_start_bits: None,
            },
            histogram_selector_bits: 0,
            histogram_selector: Some(0),
            entropy_uses_prefix_code: None,
            coefficient_probe: None,
            channel_trace: None,
            coefficient_summary: None,
            coefficient_grid: None,
            base_dequantized_grid: None,
            dequantized_grid: None,
            spatial_grid: None,
            spatial_with_dc_grid,
            parse_error: None,
        }
    }

    fn ac_group_metadata_for_pass(
        pass: usize,
        group: VarDctGroupMetadata,
        spatial_with_dc_grid: Option<VarDctAcSpatialGrid>,
    ) -> VarDctAcGroupMetadata {
        let mut metadata = ac_group_metadata(group, spatial_with_dc_grid);
        metadata.payload.pass = pass;
        metadata.payload.section.section.section_kind = FrameSectionKind::AcGroup {
            pass,
            group: metadata.payload.group.group,
        };
        metadata
    }

    fn vardct_header(width: u32, height: u32) -> FrameHeader {
        let group_dim = 128;
        let groups_x = width.div_ceil(group_dim);
        let groups_y = height.div_ceil(group_dim);
        let dc_group_dim = group_dim * 8;
        let dc_groups_x = width.div_ceil(dc_group_dim);
        let dc_groups_y = height.div_ceil(dc_group_dim);
        FrameHeader {
            encoding: FrameEncoding::VarDct,
            frame_type: FrameType::Regular,
            flags: 0,
            color_transform: ColorTransform::Xyb,
            chroma_subsampling: YCbCrChromaSubsampling::default(),
            group_size_shift: 0,
            x_qm_scale: 3,
            b_qm_scale: 2,
            passes: Passes::default(),
            dc_level: 0,
            custom_size_or_origin: false,
            frame_origin: FrameOrigin { x0: 0, y0: 0 },
            frame_size: FrameSize { width, height },
            upsampling: 1,
            extra_channel_upsampling: Vec::new(),
            blending_info: BlendingInfo::default(),
            extra_channel_blending_info: Vec::new(),
            animation_frame: AnimationFrame::default(),
            is_last: true,
            save_as_reference: 0,
            save_before_color_transform: false,
            name: String::new(),
            loop_filter: LoopFilter::default(),
            extensions: 0,
            group_layout: FrameGroupLayout {
                group_dim,
                groups_x,
                groups_y,
                num_groups: groups_x * groups_y,
                dc_group_dim,
                dc_groups_x,
                dc_groups_y,
                num_dc_groups: dc_groups_x * dc_groups_y,
            },
        }
    }
}
