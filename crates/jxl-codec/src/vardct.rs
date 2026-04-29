use crate::bitstream::{BitReader, bits_offset, val};
use crate::decode::ImageRegion;
use crate::entropy::{
    AnsAliasTableProbe, AnsHistogramLogCountProbe, AnsHistogramPopulationProbe, AnsHistogramProbe,
    AnsHistogramProbeKind, AnsHistogramProbeStage, ContextMapProbe, ContextMapProbeKind,
    ContextMapProbeStage, ContextMapSymbolProbe, HistogramCodingProbeStage, decode_context_map,
    probe_decode_context_map,
};
use crate::error::{Error, Result};
use crate::frame::{FrameEncoding, FrameHeader};
use crate::frame_data::{FrameData, FrameSection, FrameSectionKind, section_payload_range};
use crate::metadata::ImageMetadata;
use crate::metadata::unpack_signed;
use crate::modular::{
    MaTreeLeafProbe, ModularDecodedGroup, ModularGroupChannelPlan, ModularGroupHeader,
    ModularPredictor, ModularTreeCoding, decode_modular_stream_from_reader,
    probe_modular_global_tree_coding, read_modular_global_tree_coding,
    read_modular_group_header_metadata,
};
use std::ops::Range;

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
    pub ac_group_payloads: Vec<VarDctPassGroupPayloadMetadata>,
    pub dc_group_payloads: Vec<VarDctDcGroupPayloadMetadata>,
    pub dc_group_metadata: Vec<VarDctDcGroupMetadata>,
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
pub struct VarDctPassGroupPayloadMetadata {
    pub section: VarDctSectionPayloadMetadata,
    pub pass: usize,
    pub group: VarDctGroupMetadata,
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
    let ac_global_payload = frame
        .ac_global_section
        .as_ref()
        .map(|section| section_payload_metadata(codestream, frame_data, section))
        .transpose()?;
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
    let ac_group_payloads = frame
        .ac_group_sections
        .iter()
        .map(|section| {
            Ok(VarDctPassGroupPayloadMetadata {
                section: section_payload_metadata(codestream, frame_data, &section.section)?,
                pass: section.pass,
                group: section.group,
            })
        })
        .collect::<Result<Vec<_>>>()?;
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
        ac_group_payloads,
        dc_group_payloads,
        dc_group_metadata,
    }))
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
        let plan = read_vardct_decode_plan(&codestream, &metadata, &frame_header, &frame_data)
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
    fn rejects_vardct_section_payload_outside_codestream() {
        let frame_header = vardct_header(8, 8);
        let frame_data = frame_data(vec![frame_section(0, 0, FrameSectionKind::Combined, 8, 8)]);
        let codestream = vec![0; 12];

        let metadata = ImageMetadata::default();
        let error = read_vardct_decode_plan(&codestream, &metadata, &frame_header, &frame_data)
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
        let error = read_vardct_decode_plan(&codestream, &metadata, &frame_header, &frame_data)
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
