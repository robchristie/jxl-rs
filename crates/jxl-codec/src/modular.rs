use crate::bitstream::{BitReader, bits_offset, val};
use crate::entropy::{AnsCode, AnsSymbolReader, decode_histograms};
use crate::error::{Error, Result};
use crate::frame::{ColorTransform, FrameEncoding, FrameHeader};
use crate::frame_data::{FrameData, FrameSection, FrameSectionKind};
use crate::metadata::{ImageMetadata, unpack_signed};

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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModularResiduals {
    pub groups: Vec<ModularDecodedGroup>,
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
) -> Result<Option<ModularFrameMetadata>> {
    if frame_header.encoding != FrameEncoding::Modular {
        return Ok(None);
    }
    if frame_header.flags & UNSUPPORTED_DC_GLOBAL_FEATURES != 0 {
        return Ok(None);
    }

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
    let global = read_global_section(&mut reader, metadata, frame_header, section)?;
    let channel_plan = build_channel_plan(metadata, frame_header, &global.group_header)?;
    let groups =
        read_modular_group_sections(codestream, frame_header, frame_data, &global, &channel_plan)?;
    let residuals = read_modular_residuals(codestream, frame_header, frame_data, &groups).ok();
    Ok(Some(ModularFrameMetadata {
        global,
        channel_plan,
        groups,
        residuals,
    }))
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

fn read_global_tree_coding(
    codestream: &[u8],
    frame_header: &FrameHeader,
    frame_data: &FrameData,
) -> Result<ModularTreeCoding> {
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
    let coding = read_tree_coding(&mut reader, MAX_TREE_SIZE)?;
    let header = read_group_header(&mut reader)?;
    if !header.use_global_tree {
        return Err(Error::InvalidCodestream(
            "global modular stream does not use its global tree",
        ));
    }
    let _ = frame_header;
    Ok(coding)
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

fn read_modular_residuals(
    codestream: &[u8],
    frame_header: &FrameHeader,
    frame_data: &FrameData,
    groups: &[ModularSectionMetadata],
) -> Result<ModularResiduals> {
    let global_tree = read_global_tree_coding(codestream, frame_header, frame_data)?;
    let mut decoded_groups = Vec::new();
    for group in groups {
        if group.payload_size == 0 || group.channels.is_empty() {
            continue;
        }
        decoded_groups.push(decode_group_residuals(codestream, group, &global_tree)?);
    }
    Ok(ModularResiduals {
        groups: decoded_groups,
    })
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
    let mut symbol_reader = AnsSymbolReader::new(tree.code.clone(), &mut reader)?;
    let mut decoded_channels = Vec::new();
    for (local_channel, channel) in group.channels.iter().enumerate() {
        decoded_channels.push(decode_channel_residuals(
            &mut reader,
            &mut symbol_reader,
            &tree,
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
    channel: &ModularGroupChannelPlan,
    local_channel: usize,
    stream_id: usize,
) -> Result<ModularDecodedChannel> {
    let sample_count = (channel.width as usize)
        .checked_mul(channel.height as usize)
        .ok_or(Error::InvalidCodestream("modular channel size overflow"))?;
    let mut samples = vec![0i32; sample_count];
    let mut properties = vec![0i32; NUM_NONREF_PROPERTIES];
    properties[0] = local_channel as i32;
    properties[1] = stream_id as i32;
    for y in 0..channel.height as usize {
        properties[2] = y as i32;
        properties[9] = 0;
        for x in 0..channel.width as usize {
            fill_pixel_properties(&mut properties, &samples, channel.width as usize, x, y);
            let leaf = lookup_tree_leaf(&tree.tree, &properties)?;
            let context = usize::from(
                *tree
                    .context_map
                    .get(leaf.lchild as usize)
                    .ok_or(Error::InvalidCodestream("invalid modular residual context"))?,
            );
            let guess = predict_one(leaf.predictor, &samples, channel.width as usize, x, y)?;
            let residual =
                unpack_signed(symbol_reader.read_hybrid_uint_clustered(context, reader)?);
            samples[y * channel.width as usize + x] = residual
                .checked_mul(leaf.multiplier as i32)
                .and_then(|value| value.checked_add(leaf.predictor_offset as i32))
                .and_then(|value| value.checked_add(guess))
                .ok_or(Error::InvalidCodestream("modular residual overflow"))?;
        }
    }
    Ok(ModularDecodedChannel {
        channel_index: channel.channel_index,
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
        if node.property == WP_PROPERTY {
            return Err(Error::InvalidCodestream(
                "modular tree requires weighted-predictor properties",
            ));
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
        ModularPredictor::Weighted => {
            return Err(Error::InvalidCodestream(
                "unsupported weighted modular predictor",
            ));
        }
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
    global_header: &ModularGroupHeader,
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
    for transform in &global_header.transforms {
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
    transform: &ModularTransform,
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
            let squeezes = if transform.squeezes.is_empty() {
                default_squeeze_parameters(channels, *nb_meta_channels)?
            } else {
                transform.squeezes.clone()
            };
            for squeeze in &squeezes {
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
    let mut symbol_reader = AnsSymbolReader::new(code, reader)?;
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
