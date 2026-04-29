use crate::bitstream::{BitReader, bits_offset, val};
use crate::decode::{DecodeConfig, ImageRegion, ModularGroupExecution};
use crate::entropy::{AnsCode, AnsSymbolReader, decode_histograms};
use crate::error::{Error, Result};
use crate::frame::{ColorTransform, FrameEncoding, FrameHeader};
use crate::frame_data::{FrameData, FrameSection, FrameSectionKind};
use crate::metadata::{ImageMetadata, unpack_signed};
use rayon::prelude::*;

const TREE_CONTEXTS: usize = 6;
const SPLIT_VAL_CONTEXT: usize = 0;
const PROPERTY_CONTEXT: usize = 1;
const PREDICTOR_CONTEXT: usize = 2;
const OFFSET_CONTEXT: usize = 3;
const MULTIPLIER_LOG_CONTEXT: usize = 4;
const MULTIPLIER_BITS_CONTEXT: usize = 5;
const MAX_TREE_SIZE: usize = 1 << 22;
const TREE_HEIGHT_LIMIT: i32 = 2048;
const NUM_QUANT_TABLES: usize = 17;
const NUM_NONREF_PROPERTIES: usize = 16;
const WP_PROPERTY: i16 = 15;
const WP_NUM_PREDICTORS: usize = 4;
const WP_PRED_EXTRA_BITS: i64 = 3;
const WP_PREDICTION_ROUND: i64 = ((1 << WP_PRED_EXTRA_BITS) >> 1) - 1;
const WP_DIV_LOOKUP: [u32; 64] = [
    16_777_216, 8_388_608, 5_592_405, 4_194_304, 3_355_443, 2_796_202, 2_396_745, 2_097_152,
    1_864_135, 1_677_721, 1_525_201, 1_398_101, 1_290_555, 1_198_372, 1_118_481, 1_048_576,
    986_895, 932_067, 883_011, 838_860, 798_915, 762_600, 729_444, 699_050, 671_088, 645_277,
    621_378, 599_186, 578_524, 559_240, 541_200, 524_288, 508_400, 493_447, 479_349, 466_033,
    453_438, 441_505, 430_185, 419_430, 409_200, 399_457, 390_167, 381_300, 372_827, 364_722,
    356_962, 349_525, 342_392, 335_544, 328_965, 322_638, 316_551, 310_689, 305_040, 299_593,
    294_337, 289_262, 284_359, 279_620, 275_036, 270_600, 266_305, 262_144,
];
const FLAG_NOISE: u64 = 1;
const FLAG_PATCHES: u64 = 2;
const FLAG_SPLINES: u64 = 16;
const UNSUPPORTED_DC_GLOBAL_FEATURES: u64 = FLAG_NOISE | FLAG_PATCHES | FLAG_SPLINES;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModularFrameMetadata {
    pub global: ModularGlobalSection,
    pub channel_plan: ModularChannelPlan,
    pub groups: Vec<ModularSectionMetadata>,
    pub residuals: Option<ModularResiduals>,
    pub image: Option<ModularImage>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModularDecodePlan {
    global: ModularGlobalSection,
    channel_plan: ModularChannelPlan,
    groups: Vec<ModularSectionMetadata>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ImageRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl From<ImageRegion> for ImageRect {
    fn from(region: ImageRegion) -> Self {
        Self {
            x: region.x,
            y: region.y,
            width: region.width,
            height: region.height,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModularGroupExecutor {
    Serial,
    RequestedThreads(usize),
}

impl From<ModularGroupExecution> for ModularGroupExecutor {
    fn from(execution: ModularGroupExecution) -> Self {
        match execution {
            ModularGroupExecution::Serial => Self::Serial,
            ModularGroupExecution::RequestedThreads(threads) => Self::RequestedThreads(threads),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModularResiduals {
    pub global: Option<ModularDecodedGroup>,
    pub groups: Vec<ModularDecodedGroup>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModularImage {
    pub width: u32,
    pub height: u32,
    pub channels: Vec<ModularImageChannel>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModularImageChannel {
    pub width: u32,
    pub height: u32,
    pub samples: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModularDecodedGroup {
    pub section_physical_index: usize,
    pub stream_id: usize,
    pub channels: Vec<ModularDecodedChannel>,
    pub bits_consumed: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModularDecodedChannel {
    pub channel_index: usize,
    pub x0: u32,
    pub y0: u32,
    pub width: u32,
    pub height: u32,
    pub samples: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModularChannelPlan {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u32,
    pub nb_meta_channels: usize,
    pub channels: Vec<ModularChannel>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModularChannel {
    pub width: u32,
    pub height: u32,
    pub hshift: i32,
    pub vshift: i32,
    pub component: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModularGroupChannelPlan {
    pub channel_index: usize,
    pub width: u32,
    pub height: u32,
    pub x0: u32,
    pub y0: u32,
    pub hshift: i32,
    pub vshift: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModularGlobalSection {
    pub section_logical_id: usize,
    pub section_kind: FrameSectionKind,
    pub has_global_tree: bool,
    pub global_tree: Option<MaTree>,
    pub global_tree_contexts: Option<usize>,
    pub global_tree_context_map_size: Option<usize>,
    pub group_header: ModularGroupHeader,
    pub bits_consumed: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModularSectionMetadata {
    pub section_logical_id: usize,
    pub section_physical_index: usize,
    pub section_kind: FrameSectionKind,
    pub codestream_offset: usize,
    pub stream_id: usize,
    pub payload_size: u32,
    pub header: Option<ModularGroupHeader>,
    pub local_tree: Option<ModularTreeMetadata>,
    pub channels: Vec<ModularGroupChannelPlan>,
    pub bits_consumed: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModularTreeMetadata {
    pub tree: MaTree,
    pub contexts: usize,
    pub context_map_size: usize,
}

#[derive(Debug, Clone)]
struct ModularTreeCoding {
    tree: MaTree,
    code: AnsCode,
    context_map: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModularGroupHeader {
    pub use_global_tree: bool,
    pub weighted_predictor: WeightedPredictorHeader,
    pub transforms: Vec<ModularTransform>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightedPredictorHeader {
    pub all_default: bool,
    pub p1c: u32,
    pub p2c: u32,
    pub p3ca: u32,
    pub p3cb: u32,
    pub p3cc: u32,
    pub p3cd: u32,
    pub p3ce: u32,
    pub weights: [u32; 4],
}

impl Default for WeightedPredictorHeader {
    fn default() -> Self {
        Self {
            all_default: true,
            p1c: 16,
            p2c: 10,
            p3ca: 7,
            p3cb: 7,
            p3cc: 7,
            p3cd: 0,
            p3ce: 0,
            weights: [0xd, 0xc, 0xc, 0xc],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModularTransform {
    pub id: TransformId,
    pub begin_c: u32,
    pub rct_type: Option<u32>,
    pub num_c: Option<u32>,
    pub nb_colors: Option<u32>,
    pub nb_deltas: Option<u32>,
    pub predictor: Option<ModularPredictor>,
    pub squeezes: Vec<SqueezeParams>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum TransformId {
    Rct = 0,
    Palette = 1,
    Squeeze = 2,
}

impl TryFrom<u32> for TransformId {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::Rct),
            1 => Ok(Self::Palette),
            2 => Ok(Self::Squeeze),
            _ => Err(Error::InvalidCodestream("invalid modular transform id")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SqueezeParams {
    pub horizontal: bool,
    pub in_place: bool,
    pub begin_c: u32,
    pub num_c: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaTree {
    pub nodes: Vec<MaTreeNode>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaTreeNode {
    pub property: i16,
    pub splitval: i32,
    pub lchild: u32,
    pub rchild: u32,
    pub predictor: ModularPredictor,
    pub predictor_offset: i64,
    pub multiplier: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum ModularPredictor {
    #[default]
    Zero = 0,
    Left = 1,
    Top = 2,
    Average0 = 3,
    Select = 4,
    Gradient = 5,
    Weighted = 6,
    TopRight = 7,
    TopLeft = 8,
    LeftLeft = 9,
    Average1 = 10,
    Average2 = 11,
    Average3 = 12,
    Average4 = 13,
}

impl TryFrom<u32> for ModularPredictor {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::Zero),
            1 => Ok(Self::Left),
            2 => Ok(Self::Top),
            3 => Ok(Self::Average0),
            4 => Ok(Self::Select),
            5 => Ok(Self::Gradient),
            6 => Ok(Self::Weighted),
            7 => Ok(Self::TopRight),
            8 => Ok(Self::TopLeft),
            9 => Ok(Self::LeftLeft),
            10 => Ok(Self::Average1),
            11 => Ok(Self::Average2),
            12 => Ok(Self::Average3),
            13 => Ok(Self::Average4),
            _ => Err(Error::InvalidCodestream("invalid modular predictor")),
        }
    }
}

pub fn read_modular_frame_metadata(
    codestream: &[u8],
    metadata: &ImageMetadata,
    frame_header: &FrameHeader,
    frame_data: &FrameData,
    config: DecodeConfig,
) -> Result<Option<ModularFrameMetadata>> {
    if frame_header.encoding != FrameEncoding::Modular {
        return Ok(None);
    }
    if frame_header.flags & UNSUPPORTED_DC_GLOBAL_FEATURES != 0 {
        return Ok(None);
    }

    let plan = read_modular_decode_plan(codestream, metadata, frame_header, frame_data)?;
    let region = config.region.map(ImageRect::from);
    let residuals = decode_modular_residuals(
        codestream,
        frame_header,
        frame_data,
        &plan,
        ModularGroupExecutor::from(config.modular_group_execution),
        region,
    )
    .ok();
    let image = residuals.as_ref().and_then(|residuals| match region {
        Some(region) => assemble_modular_image_region(&plan, residuals, region).ok(),
        None => assemble_modular_image(&plan, residuals).ok(),
    });
    let ModularDecodePlan {
        global,
        channel_plan,
        groups,
    } = plan;
    Ok(Some(ModularFrameMetadata {
        global,
        channel_plan,
        groups,
        residuals,
        image,
    }))
}

fn read_modular_decode_plan(
    codestream: &[u8],
    metadata: &ImageMetadata,
    frame_header: &FrameHeader,
    frame_data: &FrameData,
) -> Result<ModularDecodePlan> {
    let section = frame_data
        .sections
        .iter()
        .find(|section| {
            matches!(
                section.kind,
                FrameSectionKind::Combined | FrameSectionKind::DcGlobal
            )
        })
        .ok_or(Error::InvalidCodestream(
            "modular frame is missing global section",
        ))?;
    let payload = section_payload(codestream, section)?;
    let mut reader = BitReader::new(payload);
    let mut global = read_global_section(&mut reader, metadata, frame_header, section)?;
    let channel_plan = build_channel_plan(metadata, frame_header, &mut global.group_header)?;
    let groups =
        read_modular_group_sections(codestream, frame_header, frame_data, &global, &channel_plan)?;
    Ok(ModularDecodePlan {
        global,
        channel_plan,
        groups,
    })
}

fn section_payload<'a>(codestream: &'a [u8], section: &FrameSection) -> Result<&'a [u8]> {
    let start = section.codestream_offset;
    let end = start
        .checked_add(section.size as usize)
        .ok_or(Error::InvalidCodestream("modular section range overflow"))?;
    codestream.get(start..end).ok_or(Error::InvalidCodestream(
        "modular section outside codestream",
    ))
}

fn read_global_section(
    reader: &mut BitReader<'_>,
    metadata: &ImageMetadata,
    frame_header: &FrameHeader,
    section: &FrameSection,
) -> Result<ModularGlobalSection> {
    skip_dc_dequant_matrices(reader)?;
    let has_global_tree = reader.read_bool()?;
    let global_tree_metadata = if has_global_tree {
        Some(read_tree_metadata(
            reader,
            global_tree_size_limit(metadata, frame_header)?,
        )?)
    } else {
        None
    };
    let group_header = read_group_header(reader)?;
    if group_header.use_global_tree && !has_global_tree {
        return Err(Error::InvalidCodestream(
            "modular stream references a missing global tree",
        ));
    }

    Ok(ModularGlobalSection {
        section_logical_id: section.logical_id,
        section_kind: section.kind,
        has_global_tree,
        global_tree: global_tree_metadata
            .as_ref()
            .map(|metadata| metadata.tree.clone()),
        global_tree_contexts: global_tree_metadata
            .as_ref()
            .map(|metadata| metadata.contexts),
        global_tree_context_map_size: global_tree_metadata
            .as_ref()
            .map(|metadata| metadata.context_map_size),
        group_header,
        bits_consumed: reader.bits_consumed(),
    })
}

fn read_tree_metadata(
    reader: &mut BitReader<'_>,
    tree_size_limit: usize,
) -> Result<ModularTreeMetadata> {
    let tree = decode_tree(reader, tree_size_limit)?;
    let contexts = tree.nodes.len().div_ceil(2);
    let (_, context_map) = decode_histograms(reader, contexts, false)?;
    Ok(ModularTreeMetadata {
        tree,
        contexts,
        context_map_size: context_map.len(),
    })
}

fn read_tree_coding(
    reader: &mut BitReader<'_>,
    tree_size_limit: usize,
) -> Result<ModularTreeCoding> {
    let tree = decode_tree(reader, tree_size_limit)?;
    let contexts = tree.nodes.len().div_ceil(2);
    let (code, context_map) = decode_histograms(reader, contexts, false)?;
    Ok(ModularTreeCoding {
        tree,
        code,
        context_map,
    })
}

fn read_modular_group_sections(
    codestream: &[u8],
    frame_header: &FrameHeader,
    frame_data: &FrameData,
    global: &ModularGlobalSection,
    channel_plan: &ModularChannelPlan,
) -> Result<Vec<ModularSectionMetadata>> {
    let mut groups = Vec::new();
    for section in &frame_data.sections {
        if !matches!(
            section.kind,
            FrameSectionKind::DcGroup { .. } | FrameSectionKind::AcGroup { .. }
        ) {
            continue;
        }
        groups.push(read_modular_group_section(
            codestream,
            frame_header,
            section,
            global.has_global_tree,
            channel_plan,
        )?);
    }
    Ok(groups)
}

fn read_modular_group_section(
    codestream: &[u8],
    frame_header: &FrameHeader,
    section: &FrameSection,
    has_global_tree: bool,
    channel_plan: &ModularChannelPlan,
) -> Result<ModularSectionMetadata> {
    let stream_id = modular_stream_id(section.kind, frame_header)?;
    let channels = group_channel_plan(section.kind, frame_header, channel_plan)?;
    if section.size == 0 {
        return Ok(ModularSectionMetadata {
            section_logical_id: section.logical_id,
            section_physical_index: section.physical_index,
            section_kind: section.kind,
            codestream_offset: section.codestream_offset,
            stream_id,
            payload_size: section.size,
            header: None,
            local_tree: None,
            channels,
            bits_consumed: 0,
        });
    }

    let payload = section_payload(codestream, section)?;
    let mut reader = BitReader::new(payload);
    let header = read_group_header(&mut reader)?;
    let local_tree = if header.use_global_tree {
        if !has_global_tree {
            return Err(Error::InvalidCodestream(
                "modular group references a missing global tree",
            ));
        }
        None
    } else {
        Some(read_tree_metadata(&mut reader, MAX_TREE_SIZE)?)
    };

    Ok(ModularSectionMetadata {
        section_logical_id: section.logical_id,
        section_physical_index: section.physical_index,
        section_kind: section.kind,
        codestream_offset: section.codestream_offset,
        stream_id,
        payload_size: section.size,
        header: Some(header),
        local_tree,
        channels,
        bits_consumed: reader.bits_consumed(),
    })
}

fn modular_stream_id(kind: FrameSectionKind, frame_header: &FrameHeader) -> Result<usize> {
    let layout = &frame_header.group_layout;
    match kind {
        FrameSectionKind::DcGroup { group } => {
            if group >= layout.num_dc_groups as usize {
                return Err(Error::InvalidCodestream("invalid modular DC group id"));
            }
            Ok(1 + layout.num_dc_groups as usize + group)
        }
        FrameSectionKind::AcGroup { pass, group } => {
            if pass >= frame_header.passes.num_passes as usize
                || group >= layout.num_groups as usize
            {
                return Err(Error::InvalidCodestream("invalid modular AC group id"));
            }
            Ok(1 + 3 * layout.num_dc_groups as usize
                + NUM_QUANT_TABLES
                + layout.num_groups as usize * pass
                + group)
        }
        _ => Err(Error::InvalidCodestream("section is not a modular group")),
    }
}

fn decode_modular_residuals(
    codestream: &[u8],
    frame_header: &FrameHeader,
    frame_data: &FrameData,
    plan: &ModularDecodePlan,
    executor: ModularGroupExecutor,
    region: Option<ImageRect>,
) -> Result<ModularResiduals> {
    let (global, global_tree) =
        decode_global_residuals(codestream, frame_header, frame_data, &plan.channel_plan)?;
    let full_image = ImageRect {
        x: 0,
        y: 0,
        width: plan.channel_plan.width,
        height: plan.channel_plan.height,
    };
    let groups = executor.select_groups_for_rect(&plan.groups, region.unwrap_or(full_image));
    let decoded_groups =
        executor.decode_groups(codestream, groups.iter().copied(), &global_tree)?;
    Ok(ModularResiduals {
        global,
        groups: decoded_groups,
    })
}

impl ModularGroupExecutor {
    fn select_groups_for_rect(
        self,
        groups: &[ModularSectionMetadata],
        rect: ImageRect,
    ) -> Vec<&ModularSectionMetadata> {
        match self {
            Self::Serial => select_modular_groups_for_rect(groups, rect),
            Self::RequestedThreads(threads) => {
                debug_assert!(threads > 0);
                select_modular_groups_for_rect(groups, rect)
            }
        }
    }

    fn decode_groups<'a>(
        self,
        codestream: &[u8],
        groups: impl IntoIterator<Item = &'a ModularSectionMetadata>,
        global_tree: &ModularTreeCoding,
    ) -> Result<Vec<ModularDecodedGroup>> {
        match self {
            Self::Serial => decode_modular_group_residuals_serial(codestream, groups, global_tree),
            Self::RequestedThreads(threads) => {
                debug_assert!(threads > 0);
                let groups = groups.into_iter().collect::<Vec<_>>();
                decode_modular_group_residuals_parallel(codestream, &groups, global_tree, threads)
            }
        }
    }
}

fn select_modular_groups_for_rect(
    groups: &[ModularSectionMetadata],
    rect: ImageRect,
) -> Vec<&ModularSectionMetadata> {
    groups
        .iter()
        .filter(|group| modular_group_intersects_rect(group, rect))
        .collect()
}

fn modular_group_intersects_rect(group: &ModularSectionMetadata, rect: ImageRect) -> bool {
    group.payload_size != 0
        && group
            .channels
            .iter()
            .any(|channel| modular_channel_intersects_rect(channel, rect))
}

fn modular_channel_intersects_rect(channel: &ModularGroupChannelPlan, rect: ImageRect) -> bool {
    let Some(channel_rect) = modular_channel_image_rect(channel) else {
        return true;
    };
    image_rects_intersect(channel_rect, rect)
}

fn modular_channel_image_rect(channel: &ModularGroupChannelPlan) -> Option<ImageRect> {
    if channel.hshift < 0 || channel.vshift < 0 {
        return None;
    }
    let x = channel.x0.checked_shl(channel.hshift as u32)?;
    let y = channel.y0.checked_shl(channel.vshift as u32)?;
    let width = channel.width.checked_shl(channel.hshift as u32)?;
    let height = channel.height.checked_shl(channel.vshift as u32)?;
    Some(ImageRect {
        x,
        y,
        width,
        height,
    })
}

fn image_rects_intersect(a: ImageRect, b: ImageRect) -> bool {
    image_rect_intersection(a, b).is_some()
}

fn image_rect_intersection(a: ImageRect, b: ImageRect) -> Option<ImageRect> {
    let Some(a_right) = a.x.checked_add(a.width) else {
        return Some(a);
    };
    let Some(a_bottom) = a.y.checked_add(a.height) else {
        return Some(a);
    };
    let Some(b_right) = b.x.checked_add(b.width) else {
        return Some(b);
    };
    let Some(b_bottom) = b.y.checked_add(b.height) else {
        return Some(b);
    };
    let x = a.x.max(b.x);
    let y = a.y.max(b.y);
    let right = a_right.min(b_right);
    let bottom = a_bottom.min(b_bottom);
    if x >= right || y >= bottom {
        return None;
    }
    Some(ImageRect {
        x,
        y,
        width: right - x,
        height: bottom - y,
    })
}

fn decode_modular_group_residuals_serial<'a>(
    codestream: &[u8],
    groups: impl IntoIterator<Item = &'a ModularSectionMetadata>,
    global_tree: &ModularTreeCoding,
) -> Result<Vec<ModularDecodedGroup>> {
    let mut decoded_groups = Vec::new();
    for group in groups {
        decoded_groups.push(decode_group_residuals(codestream, group, global_tree)?);
    }
    Ok(decoded_groups)
}

fn decode_modular_group_residuals_parallel(
    codestream: &[u8],
    groups: &[&ModularSectionMetadata],
    global_tree: &ModularTreeCoding,
    threads: usize,
) -> Result<Vec<ModularDecodedGroup>> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|_| Error::Unsupported("modular group thread pool"))?;
    pool.install(|| {
        groups
            .par_iter()
            .map(|group| decode_group_residuals(codestream, group, global_tree))
            .collect()
    })
}

fn decode_global_residuals(
    codestream: &[u8],
    frame_header: &FrameHeader,
    frame_data: &FrameData,
    channel_plan: &ModularChannelPlan,
) -> Result<(Option<ModularDecodedGroup>, ModularTreeCoding)> {
    let section = frame_data
        .sections
        .iter()
        .find(|section| {
            matches!(
                section.kind,
                FrameSectionKind::Combined | FrameSectionKind::DcGlobal
            )
        })
        .ok_or(Error::InvalidCodestream(
            "modular frame is missing global section",
        ))?;
    let payload = section_payload(codestream, section)?;
    let mut reader = BitReader::new(payload);
    skip_dc_dequant_matrices(&mut reader)?;
    if !reader.read_bool()? {
        return Err(Error::InvalidCodestream("modular frame has no global tree"));
    }
    let tree = read_tree_coding(&mut reader, MAX_TREE_SIZE)?;
    let header = read_group_header(&mut reader)?;
    if !header.use_global_tree {
        return Err(Error::InvalidCodestream(
            "global modular stream does not use its global tree",
        ));
    }

    let channels = global_channel_plan(frame_header, channel_plan);
    if channels.is_empty() {
        return Ok((None, tree));
    }

    let mut symbol_reader = AnsSymbolReader::new(
        tree.code.clone(),
        &mut reader,
        channel_distance_multiplier(&channels),
    )?;
    let mut decoded_channels = Vec::with_capacity(channels.len());
    for channel in &channels {
        decoded_channels.push(decode_channel_residuals(
            &mut reader,
            &mut symbol_reader,
            &tree,
            &header.weighted_predictor,
            channel,
            channel.channel_index,
            0,
        )?);
    }
    if !symbol_reader.check_final_state() {
        return Err(Error::InvalidCodestream("invalid modular global ANS state"));
    }

    Ok((
        Some(ModularDecodedGroup {
            section_physical_index: section.physical_index,
            stream_id: 0,
            channels: decoded_channels,
            bits_consumed: reader.bits_consumed(),
        }),
        tree,
    ))
}

fn global_channel_plan(
    frame_header: &FrameHeader,
    channel_plan: &ModularChannelPlan,
) -> Vec<ModularGroupChannelPlan> {
    let max_size = frame_header.group_layout.group_dim;
    let mut channels = Vec::new();
    for (index, channel) in channel_plan.channels.iter().enumerate() {
        if index >= channel_plan.nb_meta_channels
            && (channel.width > max_size || channel.height > max_size)
        {
            break;
        }
        channels.push(ModularGroupChannelPlan {
            channel_index: index,
            width: channel.width,
            height: channel.height,
            x0: 0,
            y0: 0,
            hshift: channel.hshift,
            vshift: channel.vshift,
        });
    }
    channels
}

fn assemble_modular_image(
    plan: &ModularDecodePlan,
    residuals: &ModularResiduals,
) -> Result<ModularImage> {
    let mut channels = plan
        .channel_plan
        .channels
        .iter()
        .map(|channel| {
            let samples = channel_sample_count(channel.width, channel.height)
                .map(|count| vec![0i32; count])?;
            Ok(ModularImageChannel {
                width: channel.width,
                height: channel.height,
                samples,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    if let Some(global) = &residuals.global {
        copy_decoded_group(&mut channels, global)?;
    }
    for group in &residuals.groups {
        copy_decoded_group(&mut channels, group)?;
    }

    inverse_transforms(&plan.channel_plan, &plan.global.group_header, channels)
}

#[allow(dead_code)]
fn assemble_modular_image_region(
    plan: &ModularDecodePlan,
    residuals: &ModularResiduals,
    rect: ImageRect,
) -> Result<ModularImage> {
    if !plan.global.group_header.transforms.is_empty() {
        return Err(Error::Unsupported(
            "modular region assembly with transforms",
        ));
    }
    if plan.channel_plan.nb_meta_channels != 0 {
        return Err(Error::Unsupported(
            "modular region assembly with meta channels",
        ));
    }
    if rect.width == 0 || rect.height == 0 {
        return Err(Error::InvalidCodestream("empty modular region"));
    }
    let Some(rect_right) = rect.x.checked_add(rect.width) else {
        return Err(Error::InvalidCodestream("modular region overflow"));
    };
    let Some(rect_bottom) = rect.y.checked_add(rect.height) else {
        return Err(Error::InvalidCodestream("modular region overflow"));
    };
    if rect_right > plan.channel_plan.width || rect_bottom > plan.channel_plan.height {
        return Err(Error::InvalidCodestream("modular region is outside image"));
    }
    if plan.channel_plan.channels.iter().any(|channel| {
        channel.hshift != 0
            || channel.vshift != 0
            || channel.width != plan.channel_plan.width
            || channel.height != plan.channel_plan.height
    }) {
        return Err(Error::Unsupported(
            "modular region assembly with shifted channels",
        ));
    }

    let mut channels = plan
        .channel_plan
        .channels
        .iter()
        .map(|_| {
            let samples =
                channel_sample_count(rect.width, rect.height).map(|count| vec![0; count])?;
            Ok(ModularImageChannel {
                width: rect.width,
                height: rect.height,
                samples,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    if let Some(global) = &residuals.global {
        copy_decoded_group_region(&mut channels, global, rect)?;
    }
    for group in &residuals.groups {
        copy_decoded_group_region(&mut channels, group, rect)?;
    }

    Ok(ModularImage {
        width: rect.width,
        height: rect.height,
        channels,
    })
}

fn copy_decoded_group(
    channels: &mut [ModularImageChannel],
    group: &ModularDecodedGroup,
) -> Result<()> {
    for decoded in &group.channels {
        let dst = channels
            .get_mut(decoded.channel_index)
            .ok_or(Error::InvalidCodestream("invalid modular decoded channel"))?;
        if decoded.x0 > dst.width
            || decoded.y0 > dst.height
            || decoded.width > dst.width - decoded.x0
            || decoded.height > dst.height - decoded.y0
        {
            return Err(Error::InvalidCodestream(
                "modular decoded channel is outside destination",
            ));
        }
        let expected = channel_sample_count(decoded.width, decoded.height)?;
        if decoded.samples.len() != expected {
            return Err(Error::InvalidCodestream(
                "modular decoded channel sample count mismatch",
            ));
        }
        let dst_width = dst.width as usize;
        let decoded_width = decoded.width as usize;
        let x0 = decoded.x0 as usize;
        let y0 = decoded.y0 as usize;
        for y in 0..decoded.height as usize {
            let src_start = y * decoded_width;
            let dst_start = (y0 + y) * dst_width + x0;
            dst.samples[dst_start..dst_start + decoded_width]
                .copy_from_slice(&decoded.samples[src_start..src_start + decoded_width]);
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn copy_decoded_group_region(
    channels: &mut [ModularImageChannel],
    group: &ModularDecodedGroup,
    rect: ImageRect,
) -> Result<()> {
    for decoded in &group.channels {
        let dst = channels
            .get_mut(decoded.channel_index)
            .ok_or(Error::InvalidCodestream("invalid modular decoded channel"))?;
        if dst.width != rect.width || dst.height != rect.height {
            return Err(Error::InvalidCodestream(
                "modular region destination mismatch",
            ));
        }
        let expected = channel_sample_count(decoded.width, decoded.height)?;
        if decoded.samples.len() != expected {
            return Err(Error::InvalidCodestream(
                "modular decoded channel sample count mismatch",
            ));
        }
        let group_rect = ImageRect {
            x: decoded.x0,
            y: decoded.y0,
            width: decoded.width,
            height: decoded.height,
        };
        let Some(intersection) = image_rect_intersection(group_rect, rect) else {
            continue;
        };
        let decoded_width = decoded.width as usize;
        let dst_width = dst.width as usize;
        let src_x = (intersection.x - decoded.x0) as usize;
        let src_y = (intersection.y - decoded.y0) as usize;
        let dst_x = (intersection.x - rect.x) as usize;
        let dst_y = (intersection.y - rect.y) as usize;
        let copy_width = intersection.width as usize;
        for y in 0..intersection.height as usize {
            let src_start = (src_y + y) * decoded_width + src_x;
            let dst_start = (dst_y + y) * dst_width + dst_x;
            dst.samples[dst_start..dst_start + copy_width]
                .copy_from_slice(&decoded.samples[src_start..src_start + copy_width]);
        }
    }
    Ok(())
}

fn inverse_transforms(
    channel_plan: &ModularChannelPlan,
    global_header: &ModularGroupHeader,
    mut channels: Vec<ModularImageChannel>,
) -> Result<ModularImage> {
    let mut nb_meta_channels = channel_plan.nb_meta_channels;
    for transform in global_header.transforms.iter().rev() {
        match transform.id {
            TransformId::Palette => {
                let result = inverse_palette_transform(transform, channels, nb_meta_channels)?;
                channels = result.channels;
                nb_meta_channels = result.nb_meta_channels;
            }
            TransformId::Rct => {
                inverse_rct_transform(transform, &mut channels)?;
            }
            TransformId::Squeeze => {
                nb_meta_channels =
                    inverse_squeeze_transform(transform, &mut channels, nb_meta_channels)?;
            }
        }
    }

    let output_channels = channels
        .into_iter()
        .skip(nb_meta_channels)
        .collect::<Vec<_>>();
    Ok(ModularImage {
        width: channel_plan.width,
        height: channel_plan.height,
        channels: output_channels,
    })
}

struct InverseTransformResult {
    channels: Vec<ModularImageChannel>,
    nb_meta_channels: usize,
}

fn inverse_palette_transform(
    transform: &ModularTransform,
    mut channels: Vec<ModularImageChannel>,
    nb_meta_channels: usize,
) -> Result<InverseTransformResult> {
    let begin_c = transform.begin_c as usize;
    let num_c = transform.num_c.ok_or(Error::InvalidCodestream(
        "palette transform missing channel count",
    ))? as usize;
    let nb_colors = transform.nb_colors.ok_or(Error::InvalidCodestream(
        "palette transform missing color count",
    ))? as usize;
    let nb_deltas = transform.nb_deltas.ok_or(Error::InvalidCodestream(
        "palette transform missing delta count",
    ))? as usize;
    if nb_deltas != 0 {
        return Err(Error::InvalidCodestream(
            "palette deltas are not supported yet",
        ));
    }
    if num_c == 0 {
        return Err(Error::InvalidCodestream("empty palette transform"));
    }
    let c0 = begin_c
        .checked_add(1)
        .ok_or(Error::InvalidCodestream("invalid palette channel index"))?;
    if c0 >= channels.len() {
        return Err(Error::InvalidCodestream(
            "palette transform has too few channels",
        ));
    }
    if c0 >= nb_meta_channels {
        if nb_meta_channels == 0 {
            return Err(Error::InvalidCodestream(
                "palette transform without meta channel",
            ));
        }
    } else if nb_meta_channels < 2usize.saturating_sub(num_c) || begin_c + num_c > nb_meta_channels
    {
        return Err(Error::InvalidCodestream(
            "invalid meta palette channel count",
        ));
    }

    let palette = &channels[0];
    if palette.width as usize != nb_colors || palette.height != num_c as u32 {
        return Err(Error::InvalidCodestream("invalid palette channel shape"));
    }

    let index_channel = &channels[c0];
    let width = index_channel.width;
    let height = index_channel.height;
    let mut decoded = Vec::with_capacity(num_c);
    for component in 0..num_c {
        let mut samples = Vec::with_capacity(index_channel.samples.len());
        for index in &index_channel.samples {
            let palette_index = (*index).clamp(0, nb_colors.saturating_sub(1) as i32) as usize;
            samples.push(palette.samples[component * nb_colors + palette_index]);
        }
        decoded.push(ModularImageChannel {
            width,
            height,
            samples,
        });
    }

    channels.splice(c0..c0 + 1, decoded);
    channels.remove(0);
    let nb_meta_channels = if c0 >= nb_meta_channels {
        nb_meta_channels
            .checked_sub(1)
            .ok_or(Error::InvalidCodestream(
                "invalid modular meta channel count",
            ))?
    } else {
        nb_meta_channels
            .checked_sub(2usize.saturating_sub(num_c))
            .ok_or(Error::InvalidCodestream(
                "invalid modular meta channel count",
            ))?
    };
    Ok(InverseTransformResult {
        channels,
        nb_meta_channels,
    })
}

fn inverse_rct_transform(
    transform: &ModularTransform,
    channels: &mut [ModularImageChannel],
) -> Result<()> {
    let begin_c = transform.begin_c as usize;
    let rct_type = transform.rct_type.ok_or(Error::InvalidCodestream(
        "RCT transform missing transform type",
    ))? as usize;
    if rct_type >= 42 {
        return Err(Error::InvalidCodestream("invalid RCT transform type"));
    }
    if rct_type == 0 {
        return Ok(());
    }
    let end = begin_c
        .checked_add(3)
        .ok_or(Error::InvalidCodestream("invalid RCT channel range"))?;
    if end > channels.len() {
        return Err(Error::InvalidCodestream("invalid RCT channel range"));
    }
    check_equal_image_channels(&channels[begin_c..end])?;
    let permutation = rct_type / 7;
    let custom = rct_type % 7;
    let out_indices = rct_permutation_indices(begin_c, permutation)?;

    let len = channels[begin_c].samples.len();
    let in0 = channels[begin_c].samples.clone();
    let in1 = channels[begin_c + 1].samples.clone();
    let in2 = channels[begin_c + 2].samples.clone();
    let mut out0 = vec![0; len];
    let mut out1 = vec![0; len];
    let mut out2 = vec![0; len];

    for index in 0..len {
        let (a, b, c) = inverse_rct_pixel(custom, in0[index], in1[index], in2[index])?;
        out0[index] = a;
        out1[index] = b;
        out2[index] = c;
    }

    channels[out_indices[0]].samples = out0;
    channels[out_indices[1]].samples = out1;
    channels[out_indices[2]].samples = out2;
    Ok(())
}

fn check_equal_image_channels(channels: &[ModularImageChannel]) -> Result<()> {
    if channels.len() != 3 {
        return Err(Error::InvalidCodestream("invalid RCT channel range"));
    }
    let first = &channels[0];
    if channels[1..].iter().any(|channel| {
        channel.width != first.width
            || channel.height != first.height
            || channel.samples.len() != first.samples.len()
    }) {
        return Err(Error::InvalidCodestream("RCT channel dimensions differ"));
    }
    Ok(())
}

fn rct_permutation_indices(begin_c: usize, permutation: usize) -> Result<[usize; 3]> {
    if permutation >= 6 {
        return Err(Error::InvalidCodestream("invalid RCT permutation"));
    }
    Ok([
        begin_c + permutation % 3,
        begin_c + (permutation + 1 + permutation / 3) % 3,
        begin_c + (permutation + 2 - permutation / 3) % 3,
    ])
}

fn inverse_rct_pixel(
    custom: usize,
    first: i32,
    second: i32,
    third: i32,
) -> Result<(i32, i32, i32)> {
    if custom == 6 {
        let y = first;
        let co = second;
        let cg = third;
        let tmp = y.wrapping_sub(cg >> 1);
        let green = cg.wrapping_add(tmp);
        let blue = tmp.wrapping_sub(co >> 1);
        let red = blue.wrapping_add(co);
        return Ok((red, green, blue));
    }
    if custom > 6 {
        return Err(Error::InvalidCodestream("invalid RCT transform type"));
    }
    let mut second = second;
    let mut third = third;
    if custom & 1 != 0 {
        third = third.wrapping_add(first);
    }
    match custom >> 1 {
        0 => {}
        1 => second = second.wrapping_add(first),
        2 => second = second.wrapping_add(first.wrapping_add(third) >> 1),
        _ => return Err(Error::InvalidCodestream("invalid RCT transform type")),
    }
    Ok((first, second, third))
}

fn inverse_squeeze_transform(
    transform: &ModularTransform,
    channels: &mut Vec<ModularImageChannel>,
    mut nb_meta_channels: usize,
) -> Result<usize> {
    if transform.squeezes.is_empty() {
        return Err(Error::InvalidCodestream("missing squeeze parameters"));
    }
    for squeeze in transform.squeezes.iter().rev() {
        let begin_c = squeeze.begin_c as usize;
        let num_c = squeeze.num_c as usize;
        let end_c = begin_c
            .checked_add(num_c)
            .and_then(|value| value.checked_sub(1))
            .ok_or(Error::InvalidCodestream("invalid squeeze channel range"))?;
        if end_c >= channels.len() {
            return Err(Error::InvalidCodestream("invalid squeeze channel range"));
        }
        let offset = if squeeze.in_place {
            end_c
                .checked_add(1)
                .ok_or(Error::InvalidCodestream("invalid squeeze channel range"))?
        } else {
            channels
                .len()
                .checked_sub(num_c)
                .ok_or(Error::InvalidCodestream("invalid squeeze channel range"))?
        };
        if offset
            .checked_add(num_c)
            .is_none_or(|end| end > channels.len())
        {
            return Err(Error::InvalidCodestream("invalid squeeze residual range"));
        }
        if begin_c < nb_meta_channels {
            nb_meta_channels =
                nb_meta_channels
                    .checked_sub(num_c)
                    .ok_or(Error::InvalidCodestream(
                        "invalid modular meta channel count",
                    ))?;
        }

        for channel_index in begin_c..=end_c {
            let residual_index = offset + channel_index - begin_c;
            if channels[channel_index].width < channels[residual_index].width
                || channels[channel_index].height < channels[residual_index].height
            {
                return Err(Error::InvalidCodestream("corrupted squeeze transform"));
            }
            let unsqueezed = if squeeze.horizontal {
                inverse_horizontal_squeeze(&channels[channel_index], &channels[residual_index])?
            } else {
                inverse_vertical_squeeze(&channels[channel_index], &channels[residual_index])?
            };
            channels[channel_index] = unsqueezed;
        }
        channels.drain(offset..offset + num_c);
    }
    Ok(nb_meta_channels)
}

fn inverse_horizontal_squeeze(
    low: &ModularImageChannel,
    residual: &ModularImageChannel,
) -> Result<ModularImageChannel> {
    if low.width != (low.width + residual.width).div_ceil(2) || low.height != residual.height {
        return Err(Error::InvalidCodestream("invalid horizontal squeeze shape"));
    }
    let width = low
        .width
        .checked_add(residual.width)
        .ok_or(Error::InvalidCodestream("modular channel size overflow"))?;
    let height = low.height;
    if residual.width == 0 {
        return Ok(low.clone());
    }
    validate_channel_samples(low)?;
    validate_channel_samples(residual)?;
    let mut output = ModularImageChannel {
        width,
        height,
        samples: vec![0; channel_sample_count(width, height)?],
    };
    let low_width = low.width as usize;
    let residual_width = residual.width as usize;
    let output_width = width as usize;
    for y in 0..height as usize {
        let low_row = y * low_width;
        let residual_row = y * residual_width;
        let output_row = y * output_width;
        for x in 0..residual_width {
            let diff_minus_tendency = residual.samples[residual_row + x];
            let avg = low.samples[low_row + x];
            let next_avg = if x + 1 < low_width {
                low.samples[low_row + x + 1]
            } else {
                avg
            };
            let left = if x > 0 {
                output.samples[output_row + (x << 1) - 1]
            } else {
                avg
            };
            let diff = diff_minus_tendency + smooth_tendency(left, avg, next_avg);
            let first = avg + diff / 2;
            output.samples[output_row + (x << 1)] = first;
            output.samples[output_row + (x << 1) + 1] = first - diff;
        }
        if width & 1 != 0 {
            output.samples[output_row + output_width - 1] = low.samples[low_row + low_width - 1];
        }
    }
    Ok(output)
}

fn inverse_vertical_squeeze(
    low: &ModularImageChannel,
    residual: &ModularImageChannel,
) -> Result<ModularImageChannel> {
    if low.height != (low.height + residual.height).div_ceil(2) || low.width != residual.width {
        return Err(Error::InvalidCodestream("invalid vertical squeeze shape"));
    }
    let width = low.width;
    let height = low
        .height
        .checked_add(residual.height)
        .ok_or(Error::InvalidCodestream("modular channel size overflow"))?;
    if residual.height == 0 {
        return Ok(low.clone());
    }
    validate_channel_samples(low)?;
    validate_channel_samples(residual)?;
    let mut output = ModularImageChannel {
        width,
        height,
        samples: vec![0; channel_sample_count(width, height)?],
    };
    let width = width as usize;
    for y in 0..residual.height as usize {
        let low_row = y * width;
        let residual_row = y * width;
        let next_low_row = if y + 1 < low.height as usize {
            (y + 1) * width
        } else {
            low_row
        };
        let prev_output_row = if y > 0 {
            ((y << 1) - 1) * width
        } else {
            low_row
        };
        let output_row = (y << 1) * width;
        let next_output_row = ((y << 1) + 1) * width;
        for x in 0..width {
            let avg = low.samples[low_row + x];
            let next_avg = low.samples[next_low_row + x];
            let top = if y > 0 {
                output.samples[prev_output_row + x]
            } else {
                avg
            };
            let diff = residual.samples[residual_row + x] + smooth_tendency(top, avg, next_avg);
            let first = avg + diff / 2;
            output.samples[output_row + x] = first;
            output.samples[next_output_row + x] = first - diff;
        }
    }
    if height & 1 != 0 {
        let low_row = (low.height as usize - 1) * width;
        let output_row = (height as usize - 1) * width;
        output.samples[output_row..output_row + width]
            .copy_from_slice(&low.samples[low_row..low_row + width]);
    }
    Ok(output)
}

fn smooth_tendency(previous: i32, average: i32, next_average: i32) -> i32 {
    let mut diff = 0;
    if previous >= average && average >= next_average {
        diff = (4 * previous - 3 * next_average - average + 6) / 12;
        if diff - (diff & 1) > 2 * (previous - average) {
            diff = 2 * (previous - average) + 1;
        }
        if diff + (diff & 1) > 2 * (average - next_average) {
            diff = 2 * (average - next_average);
        }
    } else if previous <= average && average <= next_average {
        diff = (4 * previous - 3 * next_average - average - 6) / 12;
        if diff + (diff & 1) < 2 * (previous - average) {
            diff = 2 * (previous - average) - 1;
        }
        if diff - (diff & 1) < 2 * (average - next_average) {
            diff = 2 * (average - next_average);
        }
    }
    diff
}

fn validate_channel_samples(channel: &ModularImageChannel) -> Result<()> {
    if channel.samples.len() != channel_sample_count(channel.width, channel.height)? {
        return Err(Error::InvalidCodestream(
            "modular channel sample count mismatch",
        ));
    }
    Ok(())
}

fn channel_sample_count(width: u32, height: u32) -> Result<usize> {
    (width as usize)
        .checked_mul(height as usize)
        .ok_or(Error::InvalidCodestream("modular channel size overflow"))
}

fn channel_distance_multiplier(channels: &[ModularGroupChannelPlan]) -> usize {
    channels
        .iter()
        .filter(|channel| channel.width != 0 && channel.height != 0)
        .map(|channel| channel.width as usize)
        .max()
        .unwrap_or(0)
}

fn decode_group_residuals(
    codestream: &[u8],
    group: &ModularSectionMetadata,
    global_tree: &ModularTreeCoding,
) -> Result<ModularDecodedGroup> {
    let payload = codestream
        .get(group_payload_range(group)?)
        .ok_or(Error::InvalidCodestream("modular group outside codestream"))?;
    let mut reader = BitReader::new(payload);
    let header = read_group_header(&mut reader)?;
    let tree = if header.use_global_tree {
        global_tree.clone()
    } else {
        read_tree_coding(&mut reader, MAX_TREE_SIZE)?
    };
    let mut symbol_reader = AnsSymbolReader::new(
        tree.code.clone(),
        &mut reader,
        channel_distance_multiplier(&group.channels),
    )?;
    let mut decoded_channels = Vec::new();
    for (local_channel, channel) in group.channels.iter().enumerate() {
        decoded_channels.push(decode_channel_residuals(
            &mut reader,
            &mut symbol_reader,
            &tree,
            &header.weighted_predictor,
            channel,
            local_channel,
            group.stream_id,
        )?);
    }
    if !symbol_reader.check_final_state() {
        return Err(Error::InvalidCodestream(
            "invalid modular residual ANS state",
        ));
    }
    Ok(ModularDecodedGroup {
        section_physical_index: group.section_physical_index,
        stream_id: group.stream_id,
        channels: decoded_channels,
        bits_consumed: reader.bits_consumed(),
    })
}

fn group_payload_range(group: &ModularSectionMetadata) -> Result<std::ops::Range<usize>> {
    let end = group
        .codestream_offset
        .checked_add(group.payload_size as usize)
        .ok_or(Error::InvalidCodestream("modular group range overflow"))?;
    Ok(group.codestream_offset..end)
}

fn decode_channel_residuals(
    reader: &mut BitReader<'_>,
    symbol_reader: &mut AnsSymbolReader,
    tree: &ModularTreeCoding,
    wp_header: &WeightedPredictorHeader,
    channel: &ModularGroupChannelPlan,
    local_channel: usize,
    stream_id: usize,
) -> Result<ModularDecodedChannel> {
    let sample_count = (channel.width as usize)
        .checked_mul(channel.height as usize)
        .ok_or(Error::InvalidCodestream("modular channel size overflow"))?;
    let mut samples = vec![0i32; sample_count];
    let mut properties = vec![0i32; NUM_NONREF_PROPERTIES];
    let mut wp_state = WeightedPredictorState::new(wp_header, channel.width as usize);
    properties[0] = local_channel as i32;
    properties[1] = stream_id as i32;
    for y in 0..channel.height as usize {
        properties[2] = y as i32;
        properties[9] = 0;
        for x in 0..channel.width as usize {
            fill_pixel_properties(&mut properties, &samples, channel.width as usize, x, y);
            let wp_pred = wp_state.predict(&samples, channel.width as usize, x, y, &mut properties);
            let leaf = lookup_tree_leaf(&tree.tree, &properties)?;
            let context = usize::from(
                *tree
                    .context_map
                    .get(leaf.lchild as usize)
                    .ok_or(Error::InvalidCodestream("invalid modular residual context"))?,
            );
            let guess = predict_one(
                leaf.predictor,
                &samples,
                channel.width as usize,
                x,
                y,
                wp_pred,
            )?;
            let residual =
                unpack_signed(symbol_reader.read_hybrid_uint_clustered(context, reader)?);
            let sample = residual
                .checked_mul(leaf.multiplier as i32)
                .and_then(|value| value.checked_add(leaf.predictor_offset as i32))
                .and_then(|value| value.checked_add(guess))
                .ok_or(Error::InvalidCodestream("modular residual overflow"))?;
            samples[y * channel.width as usize + x] = sample;
            wp_state.update_errors(sample, channel.width as usize, x, y);
        }
    }
    Ok(ModularDecodedChannel {
        channel_index: channel.channel_index,
        x0: channel.x0,
        y0: channel.y0,
        width: channel.width,
        height: channel.height,
        samples,
    })
}

fn lookup_tree_leaf(tree: &MaTree, properties: &[i32]) -> Result<MaTreeNode> {
    let mut index = 0usize;
    loop {
        let node = *tree
            .nodes
            .get(index)
            .ok_or(Error::InvalidCodestream("invalid modular tree node"))?;
        if node.property == -1 {
            return Ok(node);
        }
        let property = *properties
            .get(node.property as usize)
            .ok_or(Error::InvalidCodestream(
                "unsupported modular tree property",
            ))?;
        index = if property > node.splitval {
            node.lchild as usize
        } else {
            node.rchild as usize
        };
    }
}

#[derive(Debug, Clone)]
struct WeightedPredictorState {
    header: WeightedPredictorHeader,
    prediction: [i64; WP_NUM_PREDICTORS],
    pred: i64,
    pred_errors: [Vec<u32>; WP_NUM_PREDICTORS],
    error: Vec<i32>,
}

impl WeightedPredictorState {
    fn new(header: &WeightedPredictorHeader, width: usize) -> Self {
        let len = (width + 2) * 2;
        Self {
            header: *header,
            prediction: [0; WP_NUM_PREDICTORS],
            pred: 0,
            pred_errors: std::array::from_fn(|_| vec![0; len]),
            error: vec![0; len],
        }
    }

    fn predict(
        &mut self,
        samples: &[i32],
        width: usize,
        x: usize,
        y: usize,
        properties: &mut [i32],
    ) -> i32 {
        let left = i64::from(sample_left(samples, width, x, y));
        let top = i64::from(sample_top(samples, width, x, y, left as i32));
        let top_left = i64::from(sample_top_left(samples, width, x, y, left as i32));
        let top_right = if x + 1 < width && y > 0 {
            i64::from(samples[(y - 1) * width + x + 1])
        } else {
            top
        };
        let top_top = if y > 1 {
            i64::from(samples[(y - 2) * width + x])
        } else {
            top
        };

        let cur_row = if y & 1 == 1 { 0 } else { width + 2 };
        let prev_row = if y & 1 == 1 { width + 2 } else { 0 };
        let pos_n = prev_row + x;
        let pos_ne = if x < width - 1 { pos_n + 1 } else { pos_n };
        let pos_nw = if x > 0 { pos_n - 1 } else { pos_n };

        let mut weights = [0u32; WP_NUM_PREDICTORS];
        for (index, weight) in weights.iter_mut().enumerate() {
            let error = u64::from(self.pred_errors[index][pos_n])
                + u64::from(self.pred_errors[index][pos_ne])
                + u64::from(self.pred_errors[index][pos_nw]);
            *weight = error_weight(error, self.header.weights[index]);
        }

        let n = add_prediction_bits(top);
        let w = add_prediction_bits(left);
        let ne = add_prediction_bits(top_right);
        let nw = add_prediction_bits(top_left);
        let nn = add_prediction_bits(top_top);

        let te_w = if x == 0 {
            0
        } else {
            i64::from(self.error[cur_row + x - 1])
        };
        let te_n = i64::from(self.error[pos_n]);
        let te_nw = i64::from(self.error[pos_nw]);
        let te_ne = i64::from(self.error[pos_ne]);
        let sum_wn = te_n + te_w;

        let mut wp_property = te_w;
        if te_n.abs() > wp_property.abs() {
            wp_property = te_n;
        }
        if te_nw.abs() > wp_property.abs() {
            wp_property = te_nw;
        }
        if te_ne.abs() > wp_property.abs() {
            wp_property = te_ne;
        }
        properties[WP_PROPERTY as usize] =
            wp_property.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;

        self.prediction[0] = w + ne - n;
        self.prediction[1] = n - (((sum_wn + te_ne) * i64::from(self.header.p1c)) >> 5);
        self.prediction[2] = w - (((sum_wn + te_nw) * i64::from(self.header.p2c)) >> 5);
        self.prediction[3] = n
            - ((te_nw * i64::from(self.header.p3ca)
                + te_n * i64::from(self.header.p3cb)
                + te_ne * i64::from(self.header.p3cc)
                + (nn - n) * i64::from(self.header.p3cd)
                + (nw - w) * i64::from(self.header.p3ce))
                >> 5);

        self.pred = weighted_average(&self.prediction, weights);
        if ((te_n ^ te_w) | (te_n ^ te_nw)) > 0 {
            return ((self.pred + WP_PREDICTION_ROUND) >> WP_PRED_EXTRA_BITS) as i32;
        }

        let max = w.max(ne).max(n);
        let min = w.min(ne).min(n);
        self.pred = self.pred.clamp(min, max);
        ((self.pred + WP_PREDICTION_ROUND) >> WP_PRED_EXTRA_BITS) as i32
    }

    fn update_errors(&mut self, value: i32, width: usize, x: usize, y: usize) {
        let cur_row = if y & 1 == 1 { 0 } else { width + 2 };
        let prev_row = if y & 1 == 1 { width + 2 } else { 0 };
        let value = add_prediction_bits(i64::from(value));
        self.error[cur_row + x] =
            (self.pred - value).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
        for index in 0..WP_NUM_PREDICTORS {
            let error = ((self.prediction[index] - value).abs() + WP_PREDICTION_ROUND)
                >> WP_PRED_EXTRA_BITS;
            let error = error.min(i64::from(u32::MAX)) as u32;
            self.pred_errors[index][cur_row + x] = error;
            self.pred_errors[index][prev_row + x + 1] =
                self.pred_errors[index][prev_row + x + 1].saturating_add(error);
        }
    }
}

fn add_prediction_bits(value: i64) -> i64 {
    value << WP_PRED_EXTRA_BITS
}

fn error_weight(error: u64, max_weight: u32) -> u32 {
    let shift = (u64::BITS - (error + 1).leading_zeros()) as i32 - 1 - 5;
    let shift = shift.max(0) as u32;
    4 + ((u64::from(max_weight) * u64::from(WP_DIV_LOOKUP[(error >> shift) as usize])) >> shift)
        as u32
}

fn weighted_average(
    predictions: &[i64; WP_NUM_PREDICTORS],
    mut weights: [u32; WP_NUM_PREDICTORS],
) -> i64 {
    let weight_sum: u32 = weights.iter().sum();
    let log_weight = u32::BITS - weight_sum.leading_zeros() - 1;
    let mut adjusted_sum = 0u32;
    for weight in &mut weights {
        *weight >>= log_weight - 4;
        adjusted_sum += *weight;
    }
    let mut sum = i64::from(adjusted_sum / 2) - 1;
    for (prediction, weight) in predictions.iter().zip(weights) {
        sum += *prediction * i64::from(weight);
    }
    (sum * i64::from(WP_DIV_LOOKUP[(adjusted_sum - 1) as usize])) >> 24
}

fn fill_pixel_properties(
    properties: &mut [i32],
    samples: &[i32],
    width: usize,
    x: usize,
    y: usize,
) {
    let left = sample_left(samples, width, x, y);
    let top = sample_top(samples, width, x, y, left);
    let top_left = sample_top_left(samples, width, x, y, left);
    let top_right = if x + 1 < width && y > 0 {
        samples[(y - 1) * width + x + 1]
    } else {
        top
    };
    let left_left = if x > 1 {
        samples[y * width + x - 2]
    } else {
        left
    };
    let top_top = if y > 1 {
        samples[(y - 2) * width + x]
    } else {
        top
    };
    properties[3] = x as i32;
    properties[4] = top.abs();
    properties[5] = left.abs();
    properties[6] = top;
    properties[7] = left;
    properties[8] = left - properties[9];
    properties[9] = left + top - top_left;
    properties[10] = left - top_left;
    properties[11] = top_left - top;
    properties[12] = top - top_right;
    properties[13] = top - top_top;
    properties[14] = left - left_left;
}

fn predict_one(
    predictor: ModularPredictor,
    samples: &[i32],
    width: usize,
    x: usize,
    y: usize,
    weighted_prediction: i32,
) -> Result<i32> {
    let left = sample_left(samples, width, x, y);
    let top = sample_top(samples, width, x, y, left);
    let top_left = sample_top_left(samples, width, x, y, left);
    let top_right = if x + 1 < width && y > 0 {
        samples[(y - 1) * width + x + 1]
    } else {
        top
    };
    let left_left = if x > 1 {
        samples[y * width + x - 2]
    } else {
        left
    };
    let top_top = if y > 1 {
        samples[(y - 2) * width + x]
    } else {
        top
    };
    let top_right_right = if x + 2 < width && y > 0 {
        samples[(y - 1) * width + x + 2]
    } else {
        top_right
    };
    let prediction = match predictor {
        ModularPredictor::Zero => 0,
        ModularPredictor::Left => left,
        ModularPredictor::Top => top,
        ModularPredictor::Average0 => (left + top) / 2,
        ModularPredictor::Select => select_predict(left, top, top_left),
        ModularPredictor::Gradient => clamped_gradient(top, left, top_left),
        ModularPredictor::TopRight => top_right,
        ModularPredictor::TopLeft => top_left,
        ModularPredictor::LeftLeft => left_left,
        ModularPredictor::Average1 => (left + top_left) / 2,
        ModularPredictor::Average2 => (top_left + top) / 2,
        ModularPredictor::Average3 => (top + top_right) / 2,
        ModularPredictor::Average4 => {
            (6 * top - 2 * top_top + 7 * left + left_left + top_right_right + 3 * top_right + 8)
                / 16
        }
        ModularPredictor::Weighted => weighted_prediction,
    };
    Ok(prediction)
}

fn sample_left(samples: &[i32], width: usize, x: usize, y: usize) -> i32 {
    if x > 0 {
        samples[y * width + x - 1]
    } else if y > 0 {
        samples[(y - 1) * width + x]
    } else {
        0
    }
}

fn sample_top(samples: &[i32], width: usize, x: usize, y: usize, left: i32) -> i32 {
    if y > 0 {
        samples[(y - 1) * width + x]
    } else {
        left
    }
}

fn sample_top_left(samples: &[i32], width: usize, x: usize, y: usize, left: i32) -> i32 {
    if x > 0 && y > 0 {
        samples[(y - 1) * width + x - 1]
    } else {
        left
    }
}

fn select_predict(left: i32, top: i32, top_left: i32) -> i32 {
    let prediction = left + top - top_left;
    let left_error = (prediction - left).abs();
    let top_error = (prediction - top).abs();
    if left_error < top_error { left } else { top }
}

fn clamped_gradient(top: i32, left: i32, top_left: i32) -> i32 {
    let guess = left + top - top_left;
    guess.clamp(left.min(top), left.max(top))
}

fn build_channel_plan(
    metadata: &ImageMetadata,
    frame_header: &FrameHeader,
    global_header: &mut ModularGroupHeader,
) -> Result<ModularChannelPlan> {
    let width = frame_header
        .frame_size
        .width
        .div_ceil(frame_header.upsampling);
    let height = frame_header
        .frame_size
        .height
        .div_ceil(frame_header.upsampling);
    let mut channels = initial_channels(metadata, frame_header, width, height)?;
    let mut nb_meta_channels = 0usize;
    for transform in &mut global_header.transforms {
        apply_transform_metadata(transform, &mut channels, &mut nb_meta_channels)?;
    }
    Ok(ModularChannelPlan {
        width,
        height,
        bit_depth: metadata.bit_depth.bits_per_sample,
        nb_meta_channels,
        channels,
    })
}

fn initial_channels(
    metadata: &ImageMetadata,
    frame_header: &FrameHeader,
    width: u32,
    height: u32,
) -> Result<Vec<ModularChannel>> {
    let color_channels = if metadata.color_encoding.color_space == crate::metadata::ColorSpace::Gray
        && frame_header.color_transform == ColorTransform::None
    {
        1usize
    } else {
        3usize
    };
    let mut channels = Vec::with_capacity(color_channels + metadata.extra_channels.len());
    for component in 0..color_channels {
        let (hshift, vshift) = if frame_header.color_transform == ColorTransform::YCbCr {
            chroma_shift(&frame_header.chroma_subsampling, component)?
        } else {
            (0, 0)
        };
        channels.push(ModularChannel {
            width: shifted_size(width, hshift)?,
            height: shifted_size(height, vshift)?,
            hshift,
            vshift,
            component: Some(component),
        });
    }

    let frame_upsampling_log = ceil_log2_nonzero_u32(frame_header.upsampling)?;
    for (index, extra) in metadata.extra_channels.iter().enumerate() {
        let upsampling =
            *frame_header
                .extra_channel_upsampling
                .get(index)
                .ok_or(Error::InvalidCodestream(
                    "missing extra-channel upsampling factor",
                ))?;
        let shift = ceil_log2_nonzero_u32(upsampling)? - frame_upsampling_log;
        channels.push(ModularChannel {
            width: frame_header.frame_size.width.div_ceil(upsampling),
            height: frame_header.frame_size.height.div_ceil(upsampling),
            hshift: shift,
            vshift: shift,
            component: Some(color_channels + index),
        });
        if extra.dim_shift > 30 {
            return Err(Error::InvalidCodestream("invalid extra-channel shift"));
        }
    }
    Ok(channels)
}

fn chroma_shift(
    subsampling: &crate::frame::YCbCrChromaSubsampling,
    channel: usize,
) -> Result<(i32, i32)> {
    const H_SHIFT: [i32; 4] = [0, 1, 1, 0];
    const V_SHIFT: [i32; 4] = [0, 1, 0, 1];
    let mode = *subsampling
        .channel_mode
        .get(channel)
        .ok_or(Error::InvalidCodestream("invalid chroma channel"))? as usize;
    let hshift = i32::from(subsampling.max_h_shift) - H_SHIFT[mode];
    let vshift = i32::from(subsampling.max_v_shift) - V_SHIFT[mode];
    Ok((hshift, vshift))
}

fn apply_transform_metadata(
    transform: &mut ModularTransform,
    channels: &mut Vec<ModularChannel>,
    nb_meta_channels: &mut usize,
) -> Result<()> {
    match transform.id {
        TransformId::Rct => {
            check_equal_channels(channels, *nb_meta_channels, transform.begin_c as usize, 3)?;
        }
        TransformId::Palette => {
            let num_c = transform.num_c.ok_or(Error::InvalidCodestream(
                "palette transform missing channel count",
            ))? as usize;
            let nb_colors = transform.nb_colors.ok_or(Error::InvalidCodestream(
                "palette transform missing color count",
            ))?;
            let nb_deltas = transform.nb_deltas.ok_or(Error::InvalidCodestream(
                "palette transform missing delta count",
            ))?;
            apply_palette_metadata(
                channels,
                nb_meta_channels,
                transform.begin_c as usize,
                num_c,
                nb_colors,
                nb_deltas,
            )?;
        }
        TransformId::Squeeze => {
            if transform.squeezes.is_empty() {
                transform.squeezes = default_squeeze_parameters(channels, *nb_meta_channels)?;
            }
            for squeeze in &transform.squeezes {
                apply_squeeze_metadata(channels, nb_meta_channels, squeeze)?;
            }
        }
    }
    Ok(())
}

fn check_equal_channels(
    channels: &[ModularChannel],
    nb_meta_channels: usize,
    begin_c: usize,
    num_c: usize,
) -> Result<()> {
    let end_c = begin_c
        .checked_add(num_c)
        .and_then(|value| value.checked_sub(1))
        .ok_or(Error::InvalidCodestream("invalid modular channel range"))?;
    if begin_c > channels.len() || end_c >= channels.len() || end_c < begin_c {
        return Err(Error::InvalidCodestream("invalid modular channel range"));
    }
    if begin_c < nb_meta_channels && end_c >= nb_meta_channels {
        return Err(Error::InvalidCodestream(
            "transform mixes meta and non-meta channels",
        ));
    }
    let first = &channels[begin_c];
    if num_c > 1
        && channels[begin_c + 1..=end_c].iter().any(|channel| {
            channel.width != first.width
                || channel.height != first.height
                || channel.hshift != first.hshift
                || channel.vshift != first.vshift
        })
    {
        return Err(Error::InvalidCodestream(
            "modular channel dimensions differ",
        ));
    }
    Ok(())
}

fn apply_palette_metadata(
    channels: &mut Vec<ModularChannel>,
    nb_meta_channels: &mut usize,
    begin_c: usize,
    num_c: usize,
    nb_colors: u32,
    nb_deltas: u32,
) -> Result<()> {
    check_equal_channels(channels, *nb_meta_channels, begin_c, num_c)?;
    let end_c = begin_c + num_c - 1;
    if begin_c >= *nb_meta_channels {
        *nb_meta_channels += 1;
    } else {
        if end_c >= *nb_meta_channels {
            return Err(Error::InvalidCodestream("invalid meta palette transform"));
        }
        *nb_meta_channels = nb_meta_channels
            .checked_add(2)
            .and_then(|value| value.checked_sub(num_c))
            .ok_or(Error::InvalidCodestream(
                "invalid meta palette channel count",
            ))?;
    }
    channels.drain(begin_c + 1..end_c + 1);
    channels.insert(
        0,
        ModularChannel {
            width: nb_colors
                .checked_add(nb_deltas)
                .ok_or(Error::InvalidCodestream("palette channel width overflow"))?,
            height: num_c as u32,
            hshift: -1,
            vshift: -1,
            component: None,
        },
    );
    Ok(())
}

fn default_squeeze_parameters(
    channels: &[ModularChannel],
    nb_meta_channels: usize,
) -> Result<Vec<SqueezeParams>> {
    let nb_channels = channels
        .len()
        .checked_sub(nb_meta_channels)
        .ok_or(Error::InvalidCodestream("invalid meta channel count"))?;
    if nb_channels == 0 {
        return Ok(Vec::new());
    }
    let mut result = Vec::new();
    let mut width = channels[nb_meta_channels].width;
    let mut height = channels[nb_meta_channels].height;
    let wide = width > height;

    if nb_channels > 2
        && channels[nb_meta_channels + 1].width == width
        && channels[nb_meta_channels + 1].height == height
    {
        result.push(SqueezeParams {
            horizontal: true,
            in_place: false,
            begin_c: (nb_meta_channels + 1) as u32,
            num_c: 2,
        });
        result.push(SqueezeParams {
            horizontal: false,
            in_place: false,
            begin_c: (nb_meta_channels + 1) as u32,
            num_c: 2,
        });
    }

    let mut params = SqueezeParams {
        horizontal: false,
        in_place: true,
        begin_c: nb_meta_channels as u32,
        num_c: nb_channels as u32,
    };
    if !wide && height > 8 {
        params.horizontal = false;
        result.push(params);
        height = height.div_ceil(2);
    }
    while width > 8 || height > 8 {
        if width > 8 {
            params.horizontal = true;
            result.push(params);
            width = width.div_ceil(2);
        }
        if height > 8 {
            params.horizontal = false;
            result.push(params);
            height = height.div_ceil(2);
        }
    }
    Ok(result)
}

fn apply_squeeze_metadata(
    channels: &mut Vec<ModularChannel>,
    nb_meta_channels: &mut usize,
    squeeze: &SqueezeParams,
) -> Result<()> {
    let begin_c = squeeze.begin_c as usize;
    let num_c = squeeze.num_c as usize;
    let end_c = begin_c
        .checked_add(num_c)
        .and_then(|value| value.checked_sub(1))
        .ok_or(Error::InvalidCodestream("invalid squeeze channel range"))?;
    if end_c >= channels.len() {
        return Err(Error::InvalidCodestream("invalid squeeze channel range"));
    }
    if begin_c < *nb_meta_channels {
        if end_c >= *nb_meta_channels {
            return Err(Error::InvalidCodestream(
                "squeeze mixes meta and non-meta channels",
            ));
        }
        if !squeeze.in_place {
            return Err(Error::InvalidCodestream(
                "meta squeeze requires in-place residuals",
            ));
        }
        *nb_meta_channels = nb_meta_channels
            .checked_add(num_c)
            .ok_or(Error::InvalidCodestream("meta squeeze channel overflow"))?;
    }
    let offset = if squeeze.in_place {
        end_c + 1
    } else {
        channels.len()
    };
    for channel_index in begin_c..=end_c {
        let mut residual = channels[channel_index].clone();
        if residual.width == 0 || residual.height == 0 {
            return Err(Error::InvalidCodestream("squeezing empty channel"));
        }
        if residual.hshift > 30 || residual.vshift > 30 {
            return Err(Error::InvalidCodestream("too many squeezes"));
        }
        if squeeze.horizontal {
            let low_width = residual.width.div_ceil(2);
            channels[channel_index].width = low_width;
            channels[channel_index].hshift += 1;
            residual.width -= low_width;
            residual.hshift = channels[channel_index].hshift;
        } else {
            let low_height = residual.height.div_ceil(2);
            channels[channel_index].height = low_height;
            channels[channel_index].vshift += 1;
            residual.height -= low_height;
            residual.vshift = channels[channel_index].vshift;
        }
        channels.insert(offset + (channel_index - begin_c), residual);
    }
    Ok(())
}

fn group_channel_plan(
    kind: FrameSectionKind,
    frame_header: &FrameHeader,
    channel_plan: &ModularChannelPlan,
) -> Result<Vec<ModularGroupChannelPlan>> {
    let (rect_x, rect_y, rect_w, rect_h, min_shift, max_shift) = match kind {
        FrameSectionKind::DcGroup { group } => {
            let groups_x = frame_header.group_layout.dc_groups_x;
            let gx = group as u32 % groups_x;
            let gy = group as u32 / groups_x;
            (
                gx * frame_header.group_layout.dc_group_dim,
                gy * frame_header.group_layout.dc_group_dim,
                frame_header.group_layout.dc_group_dim,
                frame_header.group_layout.dc_group_dim,
                3,
                1000,
            )
        }
        FrameSectionKind::AcGroup { pass, group } => {
            let groups_x = frame_header.group_layout.groups_x;
            let gx = group as u32 % groups_x;
            let gy = group as u32 / groups_x;
            let (min_shift, max_shift) = pass_downsampling_bracket(&frame_header.passes, pass)?;
            (
                gx * frame_header.group_layout.group_dim,
                gy * frame_header.group_layout.group_dim,
                frame_header.group_layout.group_dim,
                frame_header.group_layout.group_dim,
                min_shift,
                max_shift,
            )
        }
        _ => return Ok(Vec::new()),
    };

    let begin_channel = channel_plan
        .channels
        .iter()
        .enumerate()
        .skip(channel_plan.nb_meta_channels)
        .find(|(_, channel)| {
            channel.width > frame_header.group_layout.group_dim
                || channel.height > frame_header.group_layout.group_dim
        })
        .map(|(index, _)| index)
        .unwrap_or(channel_plan.channels.len());

    let mut result = Vec::new();
    for (index, channel) in channel_plan.channels.iter().enumerate().skip(begin_channel) {
        let shift = channel.hshift.min(channel.vshift);
        if shift > max_shift || shift < min_shift {
            continue;
        }
        if let Some((x0, y0, width, height)) =
            shifted_group_rect(rect_x, rect_y, rect_w, rect_h, channel)?
        {
            result.push(ModularGroupChannelPlan {
                channel_index: index,
                width,
                height,
                x0,
                y0,
                hshift: channel.hshift,
                vshift: channel.vshift,
            });
        }
    }
    Ok(result)
}

fn shifted_group_rect(
    rect_x: u32,
    rect_y: u32,
    rect_w: u32,
    rect_h: u32,
    channel: &ModularChannel,
) -> Result<Option<(u32, u32, u32, u32)>> {
    if channel.hshift < 0 || channel.vshift < 0 {
        return Err(Error::InvalidCodestream(
            "negative shifts are only valid for meta channels",
        ));
    }
    let hshift = channel.hshift as u32;
    let vshift = channel.vshift as u32;
    let x0 = rect_x >> hshift;
    let y0 = rect_y >> vshift;
    if x0 >= channel.width || y0 >= channel.height {
        return Ok(None);
    }
    let width = (rect_w >> hshift).min(channel.width - x0);
    let height = (rect_h >> vshift).min(channel.height - y0);
    if width == 0 || height == 0 {
        return Ok(None);
    }
    Ok(Some((x0, y0, width, height)))
}

fn pass_downsampling_bracket(passes: &crate::frame::Passes, pass: usize) -> Result<(i32, i32)> {
    if pass >= passes.num_passes as usize {
        return Err(Error::InvalidCodestream("invalid pass index"));
    }
    let mut max_shift = 2;
    let mut min_shift = 3;
    for index in 0.. {
        for downsample_index in 0..passes.num_downsample as usize {
            if index == passes.last_pass[downsample_index] as usize {
                min_shift = match passes.downsample[downsample_index] {
                    8 => 3,
                    4 => 2,
                    2 => 1,
                    1 => 0,
                    _ => return Err(Error::InvalidCodestream("invalid pass downsample")),
                };
            }
        }
        if index == passes.num_passes as usize - 1 {
            min_shift = 0;
        }
        if index == pass {
            return Ok((min_shift, max_shift));
        }
        max_shift = min_shift - 1;
    }
    unreachable!()
}

fn shifted_size(size: u32, shift: i32) -> Result<u32> {
    if shift < 0 {
        return Err(Error::InvalidCodestream("negative non-meta channel shift"));
    }
    Ok(size.div_ceil(1u32 << shift as u32))
}

fn ceil_log2_nonzero_u32(value: u32) -> Result<i32> {
    if value == 0 {
        return Err(Error::InvalidCodestream("zero upsampling factor"));
    }
    Ok((u32::BITS - (value - 1).leading_zeros()) as i32)
}

fn skip_dc_dequant_matrices(reader: &mut BitReader<'_>) -> Result<()> {
    let all_default = reader.read_bool()?;
    if !all_default {
        for _ in 0..3 {
            let coefficient = reader.read_f16()?;
            if coefficient <= 0.0 {
                return Err(Error::InvalidCodestream(
                    "invalid DC dequant matrix coefficient",
                ));
            }
        }
    }
    Ok(())
}

fn global_tree_size_limit(metadata: &ImageMetadata, frame_header: &FrameHeader) -> Result<usize> {
    let nb_chans = if metadata.color_encoding.color_space == crate::metadata::ColorSpace::Gray
        && frame_header.color_transform == ColorTransform::None
    {
        1usize
    } else {
        3usize
    };
    let nb_extra = metadata.extra_channels.len();
    let xsize = frame_header
        .frame_size
        .width
        .div_ceil(frame_header.upsampling);
    let ysize = frame_header
        .frame_size
        .height
        .div_ceil(frame_header.upsampling);
    let samples = u64::from(xsize)
        .checked_mul(u64::from(ysize))
        .and_then(|value| value.checked_mul((nb_chans + nb_extra) as u64))
        .ok_or(Error::InvalidCodestream(
            "modular global tree size limit overflow",
        ))?;
    Ok(MAX_TREE_SIZE.min(1024 + (samples / 16) as usize))
}

fn read_group_header(reader: &mut BitReader<'_>) -> Result<ModularGroupHeader> {
    let use_global_tree = reader.read_bool()?;
    let weighted_predictor = read_weighted_predictor_header(reader)?;
    let num_transforms =
        reader.read_u32_selector(val(0), val(1), bits_offset(4, 2), bits_offset(8, 18))? as usize;
    let mut transforms = Vec::with_capacity(num_transforms);
    for _ in 0..num_transforms {
        transforms.push(read_transform(reader)?);
    }
    Ok(ModularGroupHeader {
        use_global_tree,
        weighted_predictor,
        transforms,
    })
}

fn read_weighted_predictor_header(reader: &mut BitReader<'_>) -> Result<WeightedPredictorHeader> {
    let all_default = reader.read_bool()?;
    if all_default {
        return Ok(WeightedPredictorHeader::default());
    }

    let mut header = WeightedPredictorHeader {
        all_default,
        ..WeightedPredictorHeader::default()
    };
    header.p1c = reader.read_bits(5)? as u32;
    header.p2c = reader.read_bits(5)? as u32;
    header.p3ca = reader.read_bits(5)? as u32;
    header.p3cb = reader.read_bits(5)? as u32;
    header.p3cc = reader.read_bits(5)? as u32;
    header.p3cd = reader.read_bits(5)? as u32;
    header.p3ce = reader.read_bits(5)? as u32;
    for weight in &mut header.weights {
        *weight = reader.read_bits(4)? as u32;
    }
    Ok(header)
}

fn read_transform(reader: &mut BitReader<'_>) -> Result<ModularTransform> {
    let id = TransformId::try_from(reader.read_u32_selector(val(0), val(1), val(2), val(3))?)?;
    let mut transform = ModularTransform {
        id,
        begin_c: 0,
        rct_type: None,
        num_c: None,
        nb_colors: None,
        nb_deltas: None,
        predictor: None,
        squeezes: Vec::new(),
    };

    if matches!(id, TransformId::Rct | TransformId::Palette) {
        transform.begin_c = reader.read_u32_selector(
            bits_offset(3, 0),
            bits_offset(6, 8),
            bits_offset(10, 72),
            bits_offset(13, 1096),
        )?;
    }

    match id {
        TransformId::Rct => {
            let rct_type = reader.read_u32_selector(
                val(6),
                bits_offset(2, 0),
                bits_offset(4, 2),
                bits_offset(6, 10),
            )?;
            if rct_type >= 42 {
                return Err(Error::InvalidCodestream("invalid RCT transform type"));
            }
            transform.rct_type = Some(rct_type);
        }
        TransformId::Palette => {
            transform.num_c =
                Some(reader.read_u32_selector(val(1), val(3), val(4), bits_offset(13, 1))?);
            transform.nb_colors = Some(reader.read_u32_selector(
                bits_offset(8, 0),
                bits_offset(10, 256),
                bits_offset(12, 1280),
                bits_offset(16, 5376),
            )?);
            transform.nb_deltas = Some(reader.read_u32_selector(
                val(0),
                bits_offset(8, 1),
                bits_offset(10, 257),
                bits_offset(16, 1281),
            )?);
            let predictor = reader.read_bits(4)? as u32;
            if predictor > ModularPredictor::Average4 as u32 {
                return Err(Error::InvalidCodestream("invalid palette predictor"));
            }
            transform.predictor = Some(ModularPredictor::try_from(predictor)?);
        }
        TransformId::Squeeze => {
            let num_squeezes = reader.read_u32_selector(
                val(0),
                bits_offset(4, 1),
                bits_offset(6, 9),
                bits_offset(8, 41),
            )? as usize;
            transform.squeezes.reserve(num_squeezes);
            for _ in 0..num_squeezes {
                transform.squeezes.push(read_squeeze_params(reader)?);
            }
        }
    }
    Ok(transform)
}

fn read_squeeze_params(reader: &mut BitReader<'_>) -> Result<SqueezeParams> {
    Ok(SqueezeParams {
        horizontal: reader.read_bool()?,
        in_place: reader.read_bool()?,
        begin_c: reader.read_u32_selector(
            bits_offset(3, 0),
            bits_offset(6, 8),
            bits_offset(10, 72),
            bits_offset(13, 1096),
        )?,
        num_c: reader.read_u32_selector(val(1), val(2), val(3), bits_offset(4, 4))?,
    })
}

fn decode_tree(reader: &mut BitReader<'_>, tree_size_limit: usize) -> Result<MaTree> {
    let (code, context_map) = decode_histograms(reader, TREE_CONTEXTS, false)?;
    let mut symbol_reader = AnsSymbolReader::new(code, reader, 0)?;
    let mut nodes = Vec::new();
    let mut leaf_id = 0u32;
    let mut to_decode = 1usize;
    let tree_size_limit = tree_size_limit.min(MAX_TREE_SIZE);

    while to_decode > 0 {
        if nodes.len() > tree_size_limit {
            return Err(Error::InvalidCodestream("modular MA tree is too large"));
        }
        to_decode -= 1;
        let prop1 = symbol_reader.read_hybrid_uint(PROPERTY_CONTEXT, reader, &context_map)?;
        if prop1 > 256 {
            return Err(Error::InvalidCodestream("invalid modular MA tree property"));
        }
        let property = prop1 as i32 - 1;
        if property == -1 {
            let predictor =
                symbol_reader.read_hybrid_uint(PREDICTOR_CONTEXT, reader, &context_map)?;
            let predictor = ModularPredictor::try_from(predictor)?;
            let predictor_offset = i64::from(unpack_signed(symbol_reader.read_hybrid_uint(
                OFFSET_CONTEXT,
                reader,
                &context_map,
            )?));
            let mul_log =
                symbol_reader.read_hybrid_uint(MULTIPLIER_LOG_CONTEXT, reader, &context_map)?;
            if mul_log >= 31 {
                return Err(Error::InvalidCodestream(
                    "invalid modular MA tree multiplier logarithm",
                ));
            }
            let mul_bits =
                symbol_reader.read_hybrid_uint(MULTIPLIER_BITS_CONTEXT, reader, &context_map)?;
            let max_mul_bits = (1u64 << (31 - mul_log)) - 1;
            if u64::from(mul_bits) >= max_mul_bits {
                return Err(Error::InvalidCodestream(
                    "invalid modular MA tree multiplier",
                ));
            }
            let multiplier =
                (mul_bits + 1)
                    .checked_shl(mul_log)
                    .ok_or(Error::InvalidCodestream(
                        "modular MA tree multiplier overflow",
                    ))?;
            nodes.push(MaTreeNode {
                property: -1,
                splitval: 0,
                lchild: leaf_id,
                rchild: 0,
                predictor,
                predictor_offset,
                multiplier,
            });
            leaf_id = leaf_id
                .checked_add(1)
                .ok_or(Error::InvalidCodestream("too many modular MA tree leaves"))?;
            continue;
        }

        let splitval = unpack_signed(symbol_reader.read_hybrid_uint(
            SPLIT_VAL_CONTEXT,
            reader,
            &context_map,
        )?);
        let lchild = nodes
            .len()
            .checked_add(to_decode)
            .and_then(|value| value.checked_add(1))
            .ok_or(Error::InvalidCodestream("modular MA tree child overflow"))?;
        let rchild = lchild
            .checked_add(1)
            .ok_or(Error::InvalidCodestream("modular MA tree child overflow"))?;
        nodes.push(MaTreeNode {
            property: property as i16,
            splitval,
            lchild: lchild as u32,
            rchild: rchild as u32,
            predictor: ModularPredictor::Zero,
            predictor_offset: 0,
            multiplier: 1,
        });
        to_decode = to_decode
            .checked_add(2)
            .ok_or(Error::InvalidCodestream("modular MA tree size overflow"))?;
    }

    if !symbol_reader.check_final_state() {
        return Err(Error::InvalidCodestream(
            "invalid modular MA tree ANS state",
        ));
    }
    validate_tree(&nodes)?;
    Ok(MaTree { nodes })
}

fn validate_tree(tree: &[MaTreeNode]) -> Result<()> {
    let num_properties = tree
        .iter()
        .filter(|node| node.property >= 0)
        .map(|node| node.property as usize + 1)
        .max()
        .unwrap_or(0);
    let mut height = vec![0i32; tree.len()];
    let mut property_ranges = vec![(i32::MIN, i32::MAX); num_properties * tree.len()];

    for (index, node) in tree.iter().enumerate() {
        if height[index] > TREE_HEIGHT_LIMIT {
            return Err(Error::InvalidCodestream("modular MA tree is too tall"));
        }
        if node.property == -1 {
            continue;
        }

        let lchild = node.lchild as usize;
        let rchild = node.rchild as usize;
        if lchild >= tree.len() || rchild >= tree.len() {
            return Err(Error::InvalidCodestream("invalid modular MA tree child"));
        }
        height[lchild] = height[index] + 1;
        height[rchild] = height[index] + 1;

        let split_property = node.property as usize;
        for property in 0..num_properties {
            let (lower, upper) = property_ranges[index * num_properties + property];
            if property == split_property {
                let split = node.splitval;
                if lower > split || upper <= split {
                    return Err(Error::InvalidCodestream("invalid modular MA tree split"));
                }
                property_ranges[lchild * num_properties + property] = (split + 1, upper);
                property_ranges[rchild * num_properties + property] = (lower, split);
            } else {
                property_ranges[lchild * num_properties + property] = (lower, upper);
                property_ranges[rchild * num_properties + property] = (lower, upper);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_palette_expands_multi_channel_palette() {
        let transform = ModularTransform {
            id: TransformId::Palette,
            begin_c: 0,
            rct_type: None,
            num_c: Some(3),
            nb_colors: Some(2),
            nb_deltas: Some(0),
            predictor: Some(ModularPredictor::Zero),
            squeezes: Vec::new(),
        };
        let channels = vec![
            image_channel(2, 3, &[10, 20, 30, 40, 50, 60]),
            image_channel(2, 2, &[0, 1, 1, 0]),
        ];

        let result = inverse_palette_transform(&transform, channels, 1).unwrap();

        assert_eq!(result.nb_meta_channels, 0);
        assert_eq!(result.channels.len(), 3);
        assert_eq!(result.channels[0].samples, vec![10, 20, 20, 10]);
        assert_eq!(result.channels[1].samples, vec![30, 40, 40, 30]);
        assert_eq!(result.channels[2].samples, vec![50, 60, 60, 50]);
    }

    #[test]
    fn inverse_rct_applies_custom_transform_and_permutation() {
        let transform = ModularTransform {
            id: TransformId::Rct,
            begin_c: 0,
            rct_type: Some(10),
            num_c: None,
            nb_colors: None,
            nb_deltas: None,
            predictor: None,
            squeezes: Vec::new(),
        };
        let mut channels = vec![
            image_channel(2, 1, &[10, 20]),
            image_channel(2, 1, &[1, 2]),
            image_channel(2, 1, &[5, 6]),
        ];

        inverse_rct_transform(&transform, &mut channels).unwrap();

        assert_eq!(channels[0].samples, vec![15, 26]);
        assert_eq!(channels[1].samples, vec![10, 20]);
        assert_eq!(channels[2].samples, vec![11, 22]);
    }

    #[test]
    fn inverse_rct_supports_ycocg() {
        assert_eq!(inverse_rct_pixel(6, 100, 20, 10).unwrap(), (105, 105, 85));
    }

    #[test]
    fn selects_modular_groups_for_roi_in_plan_order() {
        let groups = pq_gradient_like_groups();

        assert_eq!(
            selected_streams(&groups, image_rect(10, 0, 32, 32)),
            vec![21]
        );
        assert_eq!(
            selected_streams(&groups, image_rect(600, 0, 32, 32)),
            vec![22]
        );
        assert_eq!(
            selected_streams(&groups, image_rect(1040, 0, 32, 32)),
            vec![23]
        );
        assert_eq!(
            selected_streams(&groups, image_rect(500, 0, 40, 32)),
            vec![21, 22]
        );
        assert_eq!(
            selected_streams(&groups, image_rect(2000, 0, 16, 16)),
            vec![]
        );
    }

    #[test]
    fn shifted_modular_group_channels_are_selected_in_image_space() {
        let groups = vec![
            group_section(30, 0, &[group_channel(0, 0, 0, 32, 32, 1, 1)]),
            group_section(31, 1, &[group_channel(0, 0, 32, 32, 32, 1, 1)]),
        ];

        assert_eq!(
            selected_streams(&groups, image_rect(10, 10, 20, 20)),
            vec![30]
        );
        assert_eq!(
            selected_streams(&groups, image_rect(10, 70, 20, 20)),
            vec![31]
        );
    }

    #[test]
    fn serial_group_executor_preserves_selected_group_order() {
        let groups = pq_gradient_like_groups();
        let executor = ModularGroupExecutor::Serial;
        let selected = executor.select_groups_for_rect(&groups, image_rect(500, 0, 600, 64));

        assert_eq!(
            selected
                .iter()
                .map(|group| group.stream_id)
                .collect::<Vec<_>>(),
            vec![21, 22, 23]
        );
    }

    #[test]
    fn assembles_modular_region_inside_one_group() {
        let plan = region_test_plan(8, 4, 1);
        let residuals = ModularResiduals {
            global: None,
            groups: vec![decoded_group(10, 0, 0, 8, 4, &[0, 1, 2, 3, 4, 5, 6, 7])],
        };

        let image =
            assemble_modular_image_region(&plan, &residuals, image_rect(2, 1, 3, 2)).unwrap();

        assert_eq!(image.width, 3);
        assert_eq!(image.height, 2);
        assert_eq!(image.channels.len(), 1);
        assert_eq!(image.channels[0].samples, vec![10, 11, 12, 18, 19, 20]);
    }

    #[test]
    fn assembles_modular_region_spanning_groups() {
        let plan = region_test_plan(8, 2, 1);
        let residuals = ModularResiduals {
            global: None,
            groups: vec![
                decoded_group(10, 0, 0, 4, 2, &[0, 1, 2, 3]),
                decoded_group(11, 4, 0, 4, 2, &[100, 101, 102, 103]),
            ],
        };

        let image =
            assemble_modular_image_region(&plan, &residuals, image_rect(2, 0, 4, 2)).unwrap();

        assert_eq!(image.width, 4);
        assert_eq!(image.height, 2);
        assert_eq!(
            image.channels[0].samples,
            vec![2, 3, 100, 101, 6, 7, 104, 105]
        );
    }

    #[test]
    fn assembles_modular_region_preserving_channel_order() {
        let plan = region_test_plan(4, 2, 2);
        let residuals = ModularResiduals {
            global: None,
            groups: vec![ModularDecodedGroup {
                section_physical_index: 0,
                stream_id: 10,
                channels: vec![
                    decoded_channel(1, 0, 0, 4, 2, &[10, 11, 12, 13]),
                    decoded_channel(0, 0, 0, 4, 2, &[0, 1, 2, 3]),
                ],
                bits_consumed: 0,
            }],
        };

        let image =
            assemble_modular_image_region(&plan, &residuals, image_rect(1, 0, 2, 2)).unwrap();

        assert_eq!(image.channels[0].samples, vec![1, 2, 5, 6]);
        assert_eq!(image.channels[1].samples, vec![11, 12, 15, 16]);
    }

    #[test]
    fn rejects_modular_region_with_shifted_channels() {
        let mut plan = region_test_plan(8, 4, 1);
        plan.channel_plan.channels[0].hshift = 1;
        let residuals = ModularResiduals {
            global: None,
            groups: Vec::new(),
        };

        assert_eq!(
            assemble_modular_image_region(&plan, &residuals, image_rect(0, 0, 4, 4)),
            Err(Error::Unsupported(
                "modular region assembly with shifted channels"
            ))
        );
    }

    fn image_channel(width: u32, height: u32, samples: &[i32]) -> ModularImageChannel {
        ModularImageChannel {
            width,
            height,
            samples: samples.to_vec(),
        }
    }

    fn selected_streams(groups: &[ModularSectionMetadata], rect: ImageRect) -> Vec<usize> {
        select_modular_groups_for_rect(groups, rect)
            .iter()
            .map(|group| group.stream_id)
            .collect()
    }

    fn pq_gradient_like_groups() -> Vec<ModularSectionMetadata> {
        vec![
            group_section(2, 0, &[]),
            group_section(21, 1, &[group_channel(1, 0, 0, 512, 64, 0, 0)]),
            group_section(22, 2, &[group_channel(1, 512, 0, 512, 64, 0, 0)]),
            group_section(23, 3, &[group_channel(1, 1024, 0, 64, 64, 0, 0)]),
        ]
    }

    fn group_section(
        stream_id: usize,
        section_physical_index: usize,
        channels: &[ModularGroupChannelPlan],
    ) -> ModularSectionMetadata {
        ModularSectionMetadata {
            section_logical_id: section_physical_index,
            section_physical_index,
            section_kind: FrameSectionKind::AcGroup {
                pass: 0,
                group: section_physical_index,
            },
            codestream_offset: 0,
            stream_id,
            payload_size: 1,
            header: None,
            local_tree: None,
            channels: channels.to_vec(),
            bits_consumed: 0,
        }
    }

    fn group_channel(
        channel_index: usize,
        x0: u32,
        y0: u32,
        width: u32,
        height: u32,
        hshift: i32,
        vshift: i32,
    ) -> ModularGroupChannelPlan {
        ModularGroupChannelPlan {
            channel_index,
            width,
            height,
            x0,
            y0,
            hshift,
            vshift,
        }
    }

    fn image_rect(x: u32, y: u32, width: u32, height: u32) -> ImageRect {
        ImageRect {
            x,
            y,
            width,
            height,
        }
    }

    fn region_test_plan(width: u32, height: u32, channels: usize) -> ModularDecodePlan {
        ModularDecodePlan {
            global: ModularGlobalSection {
                section_logical_id: 0,
                section_kind: FrameSectionKind::Combined,
                has_global_tree: true,
                global_tree: None,
                global_tree_contexts: None,
                global_tree_context_map_size: None,
                group_header: ModularGroupHeader {
                    use_global_tree: true,
                    weighted_predictor: WeightedPredictorHeader::default(),
                    transforms: Vec::new(),
                },
                bits_consumed: 0,
            },
            channel_plan: ModularChannelPlan {
                width,
                height,
                bit_depth: 8,
                nb_meta_channels: 0,
                channels: (0..channels)
                    .map(|channel| ModularChannel {
                        width,
                        height,
                        hshift: 0,
                        vshift: 0,
                        component: Some(channel),
                    })
                    .collect(),
            },
            groups: Vec::new(),
        }
    }

    fn decoded_group(
        stream_id: usize,
        x0: u32,
        y0: u32,
        width: u32,
        height: u32,
        first_row: &[i32],
    ) -> ModularDecodedGroup {
        ModularDecodedGroup {
            section_physical_index: 0,
            stream_id,
            channels: vec![decoded_channel(0, x0, y0, width, height, first_row)],
            bits_consumed: 0,
        }
    }

    fn decoded_channel(
        channel_index: usize,
        x0: u32,
        y0: u32,
        width: u32,
        height: u32,
        first_row: &[i32],
    ) -> ModularDecodedChannel {
        assert_eq!(first_row.len(), width as usize);
        let mut samples = Vec::new();
        for y in 0..height as i32 {
            samples.extend(first_row.iter().map(|sample| sample + y * width as i32));
        }
        ModularDecodedChannel {
            channel_index,
            x0,
            y0,
            width,
            height,
            samples,
        }
    }
}
