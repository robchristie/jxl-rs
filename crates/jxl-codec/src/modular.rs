use crate::bitstream::{BitReader, bits_offset, val};
use crate::entropy::{AnsSymbolReader, decode_histograms};
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
const FLAG_NOISE: u64 = 1;
const FLAG_PATCHES: u64 = 2;
const FLAG_SPLINES: u64 = 16;
const UNSUPPORTED_DC_GLOBAL_FEATURES: u64 = FLAG_NOISE | FLAG_PATCHES | FLAG_SPLINES;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModularFrameMetadata {
    pub global: ModularGlobalSection,
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
    Ok(Some(ModularFrameMetadata { global }))
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
    let (global_tree, global_tree_contexts, global_tree_context_map_size) = if has_global_tree {
        let tree = decode_tree(reader, global_tree_size_limit(metadata, frame_header)?)?;
        let tree_contexts = tree.nodes.len().div_ceil(2);
        let (_, context_map) = decode_histograms(reader, tree_contexts, false)?;
        (Some(tree), Some(tree_contexts), Some(context_map.len()))
    } else {
        (None, None, None)
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
        global_tree,
        global_tree_contexts,
        global_tree_context_map_size,
        group_header,
        bits_consumed: reader.bits_consumed(),
    })
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
