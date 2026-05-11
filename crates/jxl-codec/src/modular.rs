use crate::bitstream::{BitReader, bits_offset, val};
use crate::decode::{DecodeConfig, ImageRegion, ModularGroupExecution};
use crate::entropy::{
    AnsCode, AnsSymbolReader, HistogramCodingProbe, decode_histograms, probe_decode_histograms,
};
use crate::error::{Error, Result};
use crate::frame::{ColorTransform, FrameEncoding, FrameHeader};
use crate::frame_data::{FrameData, FrameSection, FrameSectionKind, section_payload};
use crate::metadata::{ImageMetadata, unpack_signed};
use crate::vardct::VarDctXybImage;
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
const UNSUPPORTED_DC_GLOBAL_FEATURES: u64 = FLAG_PATCHES;
const NOISE_LUT_SIZE: usize = 8;
const NOISE_PRECISION: f32 = 1024.0;
const NOISE_XORSHIFT_LANES: usize = 8;
const NOISE_FLOATS_PER_BATCH: usize = NOISE_XORSHIFT_LANES * 2;
const NOISE_CONV_RADIUS: isize = 2;
const NOISE_NORM_CONST: f32 = 0.22;
const NOISE_RG_CORR: f32 = 127.0 / 128.0;
const NOISE_RGN_CORR: f32 = 1.0 / 128.0;
const NOISE_DEFAULT_VISIBLE_FRAME_INDEX: u32 = 1;
const NOISE_DEFAULT_NONVISIBLE_FRAME_INDEX: u32 = 0;
const DEFAULT_MODULAR_DC_QUANT: [f32; 3] = [1.0 / 4096.0, 1.0 / 512.0, 1.0 / 256.0];
const SPLINE_CONTEXTS: usize = 6;
const SPLINE_QUANTIZATION_ADJUSTMENT_CONTEXT: usize = 0;
const SPLINE_STARTING_POSITION_CONTEXT: usize = 1;
const SPLINE_NUM_SPLINES_CONTEXT: usize = 2;
const SPLINE_NUM_CONTROL_POINTS_CONTEXT: usize = 3;
const SPLINE_CONTROL_POINTS_CONTEXT: usize = 4;
const SPLINE_DCT_CONTEXT: usize = 5;
const SPLINE_DCT_SIZE: usize = 32;
const MAX_SPLINE_CONTROL_POINTS: usize = 1 << 20;
const MAX_SPLINE_CONTROL_POINTS_PER_PIXEL_RATIO: usize = 2;
const SPLINE_POS_LIMIT: i32 = 1 << 23;
const SPLINE_DELTA_LIMIT: i32 = 1 << 30;
const SPLINE_DESIRED_RENDERING_DISTANCE: f32 = 1.0;
const SPLINE_DEFAULT_Y_TO_X: f32 = 0.0;
const SPLINE_DEFAULT_Y_TO_B: f32 = 1.0;
const SPLINE_CHANNEL_WEIGHTS: [f32; 4] = [0.0042, 0.075, 0.07, 0.3333];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModularFrameMetadata {
    pub global: ModularGlobalSection,
    pub channel_plan: ModularChannelPlan,
    pub groups: Vec<ModularSectionMetadata>,
    pub residuals: Option<ModularResiduals>,
    pub image: Option<ModularImage>,
    pub image_error: Option<Error>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModularDecodePlan {
    global: ModularGlobalSection,
    channel_plan: ModularChannelPlan,
    groups: Vec<ModularSectionMetadata>,
    group_dim: u32,
    upsampling: u32,
    color_transform: ColorTransform,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ImageRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModularRegionPlan {
    requested_rect: ImageRect,
    decode_rect: ImageRect,
    channel_rects: Vec<ImageRect>,
    has_squeeze: bool,
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
    pub features: FrameFeatureMetadata,
    pub dc_quant_bits: [u32; 3],
    pub has_global_tree: bool,
    pub global_tree: Option<MaTree>,
    pub global_tree_contexts: Option<usize>,
    pub global_tree_context_map_size: Option<usize>,
    pub group_header: ModularGroupHeader,
    pub bits_consumed: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FrameFeatureMetadata {
    pub noise: Option<NoiseFrameMetadata>,
    pub splines: Option<SplineFrameMetadata>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoiseFrameMetadata {
    pub lut: [u16; NOISE_LUT_SIZE],
    pub bits_consumed: usize,
}

impl NoiseFrameMetadata {
    pub fn strength_lut(&self) -> [f32; NOISE_LUT_SIZE] {
        self.lut.map(|value| value as f32 / NOISE_PRECISION)
    }
}

impl ModularGlobalSection {
    pub fn dc_quant(&self) -> [f32; 3] {
        self.dc_quant_bits.map(f32::from_bits)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplineFrameMetadata {
    pub quantization_adjustment: i32,
    pub starting_points: Vec<SplinePoint>,
    pub splines: Vec<QuantizedSplineMetadata>,
    pub bits_consumed: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplinePoint {
    pub x: i32,
    pub y: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizedSplineMetadata {
    pub control_points: Vec<(i32, i32)>,
    pub color_dct: [[i32; SPLINE_DCT_SIZE]; 3],
    pub sigma_dct: [i32; SPLINE_DCT_SIZE],
}

#[derive(Debug, Clone, PartialEq)]
pub struct DequantizedSplineMetadata {
    pub control_points: Vec<SplineFloatPoint>,
    pub color_dct: [[f32; SPLINE_DCT_SIZE]; 3],
    pub sigma_dct: [f32; SPLINE_DCT_SIZE],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplineFloatPoint {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SplineRenderPlan {
    pub splines: Vec<DequantizedSplineMetadata>,
    pub segments: Vec<SplineSegmentMetadata>,
    pub segment_indices: Vec<usize>,
    pub segment_y_start: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplineSegmentMetadata {
    pub center_x: f32,
    pub center_y: f32,
    pub maximum_distance: f32,
    pub inv_sigma: f32,
    pub sigma_over_4_times_intensity: f32,
    pub color: [f32; 3],
}

impl SplineFrameMetadata {
    pub fn dequantize_default_color_correlation(
        &self,
        width: u32,
        height: u32,
    ) -> Result<Vec<DequantizedSplineMetadata>> {
        self.dequantize_splines(width, height, SPLINE_DEFAULT_Y_TO_X, SPLINE_DEFAULT_Y_TO_B)
    }

    pub fn render_plan_default_color_correlation(
        &self,
        width: u32,
        height: u32,
    ) -> Result<SplineRenderPlan> {
        self.render_plan(width, height, SPLINE_DEFAULT_Y_TO_X, SPLINE_DEFAULT_Y_TO_B)
    }

    fn dequantize_splines(
        &self,
        width: u32,
        height: u32,
        y_to_x: f32,
        y_to_b: f32,
    ) -> Result<Vec<DequantizedSplineMetadata>> {
        if self.starting_points.len() != self.splines.len() {
            return Err(Error::InvalidCodestream("spline metadata length mismatch"));
        }
        let image_size = (width as u64)
            .checked_mul(height as u64)
            .ok_or(Error::InvalidCodestream("spline frame is too large"))?;
        let mut total_estimated_area = 0u64;
        self.splines
            .iter()
            .zip(&self.starting_points)
            .map(|(spline, starting_point)| {
                dequantize_spline(
                    spline,
                    *starting_point,
                    self.quantization_adjustment,
                    y_to_x,
                    y_to_b,
                    image_size,
                    &mut total_estimated_area,
                )
            })
            .collect()
    }

    fn render_plan(
        &self,
        width: u32,
        height: u32,
        y_to_x: f32,
        y_to_b: f32,
    ) -> Result<SplineRenderPlan> {
        let splines = self.dequantize_splines(width, height, y_to_x, y_to_b)?;
        if splines.iter().any(|spline| {
            spline
                .control_points
                .windows(2)
                .any(|points| points[0] == points[1])
        }) {
            return Err(Error::InvalidCodestream(
                "identical successive spline control points",
            ));
        }

        let mut segments = Vec::new();
        let mut segments_by_y = Vec::new();
        for spline in &splines {
            let interpolated = centripetal_catmull_rom_points(&spline.control_points)?;
            let points_to_draw = equally_spaced_spline_points(&interpolated)?;
            let Some(last) = points_to_draw.last() else {
                continue;
            };
            let arc_length = (points_to_draw.len().saturating_sub(2)) as f32
                * SPLINE_DESIRED_RENDERING_DISTANCE
                + last.1;
            if arc_length <= 0.0 {
                continue;
            }
            spline_segments_from_points(
                spline,
                &points_to_draw,
                arc_length,
                &mut segments,
                &mut segments_by_y,
            );
        }

        segments_by_y.sort_unstable();
        let mut segment_indices = vec![0usize; segments_by_y.len()];
        let mut segment_y_start = vec![0usize; height as usize + 1];
        for (index, (y, segment_index)) in segments_by_y.into_iter().enumerate() {
            segment_indices[index] = segment_index;
            if y < height as usize {
                segment_y_start[y + 1] += 1;
            }
        }
        for y in 0..height as usize {
            segment_y_start[y + 1] += segment_y_start[y];
        }

        Ok(SplineRenderPlan {
            splines,
            segments,
            segment_indices,
            segment_y_start,
        })
    }
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
pub(crate) struct ModularTreeCoding {
    tree: MaTree,
    code: AnsCode,
    context_map: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModularTreeCodingProbe {
    pub has_global_tree_end_bits: Option<usize>,
    pub tree_histogram_end_bits: Option<usize>,
    pub tree_ans_start_bits: Option<usize>,
    pub tree_end_bits: Option<usize>,
    pub tree_node_count: Option<usize>,
    pub tree_leaf_count: Option<usize>,
    pub tree_leaves: Vec<MaTreeLeafProbe>,
    pub residual_context_count: Option<usize>,
    pub residual_histogram_count: Option<usize>,
    pub residual_histogram_probe: Option<HistogramCodingProbe>,
    pub residual_coding_end_bits: Option<usize>,
    pub error_bits: Option<usize>,
    pub error: Option<Error>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MaTreeLeafProbe {
    pub leaf_index: usize,
    pub node_index: usize,
    pub residual_context: usize,
    pub predictor: ModularPredictor,
    pub predictor_offset: i64,
    pub multiplier: u32,
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
    let region_plan_result = region.map(|region| plan_modular_region(&plan, region));
    let (region_plan, mut image_error) = match region_plan_result {
        Some(Ok(region_plan)) => (Some(region_plan), None),
        Some(Err(error)) => (None, Some(error)),
        None => (None, None),
    };
    let decode_region = region_plan
        .as_ref()
        .map(|region_plan| region_plan.decode_rect);
    let residuals_result = if image_error.is_none() {
        Some(decode_modular_residuals(
            codestream,
            frame_header,
            frame_data,
            &plan,
            ModularGroupExecutor::from(config.modular_group_execution),
            decode_region,
        ))
    } else {
        None
    };
    let residuals = match residuals_result {
        Some(Ok(residuals)) => Some(residuals),
        Some(Err(error)) => {
            image_error = Some(error);
            None
        }
        None => None,
    };
    let image = if image_error.is_none() {
        residuals.as_ref().and_then(|residuals| {
            let result = assemble_modular_frame_image(&plan, residuals, region_plan.as_ref());
            match result {
                Ok(image) => Some(image),
                Err(error) => {
                    image_error = Some(error);
                    None
                }
            }
        })
    } else {
        None
    };
    if image.is_some()
        && image_error.is_none()
        && plan.color_transform == ColorTransform::Xyb
        && (plan.global.features.splines.is_some() || plan.global.features.noise.is_some())
    {
        image_error = Some(if plan.global.features.splines.is_some() {
            Error::Unsupported("spline rendering with XYB")
        } else {
            Error::Unsupported("noise rendering with XYB")
        });
    }
    let ModularDecodePlan {
        global,
        channel_plan,
        groups,
        group_dim: _,
        upsampling: _,
        color_transform: _,
    } = plan;
    Ok(Some(ModularFrameMetadata {
        global,
        channel_plan,
        groups,
        residuals,
        image,
        image_error,
    }))
}

fn assemble_modular_frame_image(
    plan: &ModularDecodePlan,
    residuals: &ModularResiduals,
    region_plan: Option<&ModularRegionPlan>,
) -> Result<ModularImage> {
    let mut image = match region_plan {
        Some(region_plan) => assemble_modular_image_region(plan, residuals, region_plan)?,
        None => assemble_modular_image(plan, residuals)?,
    };
    if let Some(splines) = &plan.global.features.splines
        && plan.color_transform != ColorTransform::Xyb
    {
        let image_origin = region_plan
            .map(|region_plan| (region_plan.requested_rect.x, region_plan.requested_rect.y))
            .unwrap_or((0, 0));
        render_splines_into_modular_image(
            &mut image,
            splines,
            plan.channel_plan.bit_depth,
            plan.channel_plan.width,
            plan.channel_plan.height,
            image_origin,
        )?;
    }
    if let Some(noise) = &plan.global.features.noise {
        let image_origin = region_plan
            .map(|region_plan| (region_plan.requested_rect.x, region_plan.requested_rect.y))
            .unwrap_or((0, 0));
        if plan.color_transform != ColorTransform::Xyb {
            render_noise_into_modular_image(&mut image, noise, plan, image_origin)?;
        }
    }
    Ok(image)
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
        group_dim: frame_header.group_layout.group_dim,
        upsampling: frame_header.upsampling,
        color_transform: frame_header.color_transform,
    })
}

fn read_dc_global_features(
    reader: &mut BitReader<'_>,
    frame_header: &FrameHeader,
) -> Result<FrameFeatureMetadata> {
    let noise = if frame_header.flags & FLAG_NOISE != 0 {
        Some(read_noise_frame_metadata(reader)?)
    } else {
        None
    };
    let splines = if frame_header.flags & FLAG_SPLINES != 0 {
        Some(read_spline_frame_metadata(reader, frame_header)?)
    } else {
        None
    };
    Ok(FrameFeatureMetadata { noise, splines })
}

fn read_noise_frame_metadata(reader: &mut BitReader<'_>) -> Result<NoiseFrameMetadata> {
    let mut lut = [0u16; NOISE_LUT_SIZE];
    for value in &mut lut {
        *value = reader.read_bits(10)? as u16;
    }
    Ok(NoiseFrameMetadata {
        lut,
        bits_consumed: reader.bits_consumed(),
    })
}

fn read_spline_frame_metadata(
    reader: &mut BitReader<'_>,
    frame_header: &FrameHeader,
) -> Result<SplineFrameMetadata> {
    let (code, context_map) = decode_histograms(reader, SPLINE_CONTEXTS, false)?;
    let mut symbol_reader = AnsSymbolReader::new(code, reader, 0)?;
    let encoded_num_splines =
        symbol_reader.read_hybrid_uint(SPLINE_NUM_SPLINES_CONTEXT, reader, &context_map)? as usize;
    let num_splines = encoded_num_splines
        .checked_add(1)
        .ok_or(Error::InvalidCodestream("too many splines"))?;
    let max_control_points = max_spline_control_points(frame_header)?;
    if num_splines > max_control_points || num_splines + 1 > max_control_points {
        return Err(Error::InvalidCodestream("too many splines"));
    }

    let starting_points =
        read_spline_starting_points(reader, &mut symbol_reader, &context_map, num_splines)?;
    let quantization_adjustment = unpack_signed(symbol_reader.read_hybrid_uint(
        SPLINE_QUANTIZATION_ADJUSTMENT_CONTEXT,
        reader,
        &context_map,
    )?);

    let mut splines = Vec::with_capacity(num_splines);
    let mut total_control_points = num_splines;
    for _ in 0..num_splines {
        splines.push(read_quantized_spline_metadata(
            reader,
            &mut symbol_reader,
            &context_map,
            max_control_points,
            &mut total_control_points,
        )?);
    }
    if !symbol_reader.check_final_state() {
        return Err(Error::InvalidCodestream("invalid spline entropy stream"));
    }
    Ok(SplineFrameMetadata {
        quantization_adjustment,
        starting_points,
        splines,
        bits_consumed: reader.bits_consumed(),
    })
}

fn max_spline_control_points(frame_header: &FrameHeader) -> Result<usize> {
    let num_pixels = (frame_header.frame_size.width as usize)
        .checked_mul(frame_header.frame_size.height as usize)
        .ok_or(Error::InvalidCodestream("spline frame is too large"))?;
    Ok(MAX_SPLINE_CONTROL_POINTS.min(num_pixels / MAX_SPLINE_CONTROL_POINTS_PER_PIXEL_RATIO))
}

fn read_spline_starting_points(
    reader: &mut BitReader<'_>,
    symbol_reader: &mut AnsSymbolReader,
    context_map: &[u8],
    num_splines: usize,
) -> Result<Vec<SplinePoint>> {
    let mut points = Vec::with_capacity(num_splines);
    let mut last_x = 0i32;
    let mut last_y = 0i32;
    for index in 0..num_splines {
        let dx = symbol_reader.read_hybrid_uint(
            SPLINE_STARTING_POSITION_CONTEXT,
            reader,
            context_map,
        )?;
        let dy = symbol_reader.read_hybrid_uint(
            SPLINE_STARTING_POSITION_CONTEXT,
            reader,
            context_map,
        )?;
        let (x, y) = if index == 0 {
            (
                i32::try_from(dx)
                    .map_err(|_| Error::InvalidCodestream("spline coordinate out of bounds"))?,
                i32::try_from(dy)
                    .map_err(|_| Error::InvalidCodestream("spline coordinate out of bounds"))?,
            )
        } else {
            (
                checked_spline_pos_add(last_x, unpack_signed(dx))?,
                checked_spline_pos_add(last_y, unpack_signed(dy))?,
            )
        };
        validate_spline_point_pos(x, y)?;
        points.push(SplinePoint { x, y });
        last_x = x;
        last_y = y;
    }
    Ok(points)
}

fn checked_spline_pos_add(base: i32, delta: i32) -> Result<i32> {
    base.checked_add(delta)
        .ok_or(Error::InvalidCodestream("spline coordinate out of bounds"))
}

fn validate_spline_point_pos(x: i32, y: i32) -> Result<()> {
    if !(-SPLINE_POS_LIMIT + 1..SPLINE_POS_LIMIT).contains(&x)
        || !(-SPLINE_POS_LIMIT + 1..SPLINE_POS_LIMIT).contains(&y)
    {
        return Err(Error::InvalidCodestream("spline coordinate out of bounds"));
    }
    Ok(())
}

fn read_quantized_spline_metadata(
    reader: &mut BitReader<'_>,
    symbol_reader: &mut AnsSymbolReader,
    context_map: &[u8],
    max_control_points: usize,
    total_control_points: &mut usize,
) -> Result<QuantizedSplineMetadata> {
    let num_control_points =
        symbol_reader.read_hybrid_uint(SPLINE_NUM_CONTROL_POINTS_CONTEXT, reader, context_map)?
            as usize;
    if num_control_points > max_control_points {
        return Err(Error::InvalidCodestream("too many spline control points"));
    }
    *total_control_points = total_control_points
        .checked_add(num_control_points)
        .ok_or(Error::InvalidCodestream("too many spline control points"))?;
    if *total_control_points > max_control_points {
        return Err(Error::InvalidCodestream("too many spline control points"));
    }

    let mut control_points = Vec::with_capacity(num_control_points);
    for _ in 0..num_control_points {
        let x = unpack_signed(symbol_reader.read_hybrid_uint(
            SPLINE_CONTROL_POINTS_CONTEXT,
            reader,
            context_map,
        )?);
        let y = unpack_signed(symbol_reader.read_hybrid_uint(
            SPLINE_CONTROL_POINTS_CONTEXT,
            reader,
            context_map,
        )?);
        if !(-SPLINE_DELTA_LIMIT + 1..SPLINE_DELTA_LIMIT).contains(&x)
            || !(-SPLINE_DELTA_LIMIT + 1..SPLINE_DELTA_LIMIT).contains(&y)
        {
            return Err(Error::InvalidCodestream("spline delta-delta out of bounds"));
        }
        control_points.push((x, y));
    }

    let mut color_dct = [[0i32; SPLINE_DCT_SIZE]; 3];
    for channel in &mut color_dct {
        read_spline_dct(reader, symbol_reader, context_map, channel)?;
    }
    let mut sigma_dct = [0i32; SPLINE_DCT_SIZE];
    read_spline_dct(reader, symbol_reader, context_map, &mut sigma_dct)?;

    Ok(QuantizedSplineMetadata {
        control_points,
        color_dct,
        sigma_dct,
    })
}

fn read_spline_dct(
    reader: &mut BitReader<'_>,
    symbol_reader: &mut AnsSymbolReader,
    context_map: &[u8],
    dct: &mut [i32; SPLINE_DCT_SIZE],
) -> Result<()> {
    for value in dct {
        *value = unpack_signed(symbol_reader.read_hybrid_uint(
            SPLINE_DCT_CONTEXT,
            reader,
            context_map,
        )?);
        if *value == i32::MIN {
            return Err(Error::InvalidCodestream("invalid spline DCT coefficient"));
        }
    }
    Ok(())
}

fn dequantize_spline(
    quantized: &QuantizedSplineMetadata,
    starting_point: SplinePoint,
    quantization_adjustment: i32,
    y_to_x: f32,
    y_to_b: f32,
    image_size: u64,
    total_estimated_area: &mut u64,
) -> Result<DequantizedSplineMetadata> {
    let area_limit = (1024u64
        .checked_mul(image_size)
        .and_then(|area| area.checked_add(1u64 << 32)))
    .unwrap_or(u64::MAX)
    .min(1u64 << 42);

    validate_spline_point_pos(starting_point.x, starting_point.y)?;
    let mut control_points = Vec::with_capacity(quantized.control_points.len() + 1);
    let mut current_x = starting_point.x;
    let mut current_y = starting_point.y;
    control_points.push(SplineFloatPoint {
        x: current_x as f32,
        y: current_y as f32,
    });

    let mut current_delta_x = 0i32;
    let mut current_delta_y = 0i32;
    let mut manhattan_distance = 0u64;
    for (delta_delta_x, delta_delta_y) in &quantized.control_points {
        current_delta_x = current_delta_x
            .checked_add(*delta_delta_x)
            .ok_or(Error::InvalidCodestream("spline delta out of bounds"))?;
        current_delta_y = current_delta_y
            .checked_add(*delta_delta_y)
            .ok_or(Error::InvalidCodestream("spline delta out of bounds"))?;
        manhattan_distance = manhattan_distance
            .checked_add(current_delta_x.unsigned_abs() as u64)
            .and_then(|distance| distance.checked_add(current_delta_y.unsigned_abs() as u64))
            .ok_or(Error::InvalidCodestream("spline area is too large"))?;
        if manhattan_distance > area_limit {
            return Err(Error::InvalidCodestream("spline area is too large"));
        }
        validate_spline_point_pos(current_delta_x, current_delta_y)?;
        current_x = current_x
            .checked_add(current_delta_x)
            .ok_or(Error::InvalidCodestream("spline coordinate out of bounds"))?;
        current_y = current_y
            .checked_add(current_delta_y)
            .ok_or(Error::InvalidCodestream("spline coordinate out of bounds"))?;
        validate_spline_point_pos(current_x, current_y)?;
        control_points.push(SplineFloatPoint {
            x: current_x as f32,
            y: current_y as f32,
        });
    }

    let inv_quant = inv_spline_adjusted_quant(quantization_adjustment);
    let mut color_dct = [[0.0f32; SPLINE_DCT_SIZE]; 3];
    for (channel, dct) in color_dct.iter_mut().enumerate() {
        for (index, value) in dct.iter_mut().enumerate() {
            let inv_dct_factor = if index == 0 {
                std::f32::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            *value = quantized.color_dct[channel][index] as f32
                * inv_dct_factor
                * SPLINE_CHANNEL_WEIGHTS[channel]
                * inv_quant;
        }
    }
    for index in 0..SPLINE_DCT_SIZE {
        color_dct[0][index] += y_to_x * color_dct[1][index];
        color_dct[2][index] += y_to_b * color_dct[1][index];
    }

    let mut width_estimate = 0u64;
    let mut color = [0u64; 3];
    for (channel, color_sum) in color.iter_mut().enumerate() {
        for value in &quantized.color_dct[channel] {
            *color_sum = color_sum
                .checked_add((inv_quant * value.abs() as f32).ceil() as u64)
                .ok_or(Error::InvalidCodestream("spline area is too large"))?;
        }
    }
    color[0] = color[0]
        .checked_add((y_to_x.abs().ceil() as u64).saturating_mul(color[1]))
        .ok_or(Error::InvalidCodestream("spline area is too large"))?;
    color[2] = color[2]
        .checked_add((y_to_b.abs().ceil() as u64).saturating_mul(color[1]))
        .ok_or(Error::InvalidCodestream("spline area is too large"))?;
    let max_color = color.into_iter().max().unwrap_or(0);
    let log_color = ceil_log2_nonzero(1 + max_color).max(1);
    let weight_limit = ((area_limit as f32 / log_color as f32) / manhattan_distance.max(1) as f32)
        .sqrt()
        .ceil();

    let mut sigma_dct = [0.0f32; SPLINE_DCT_SIZE];
    for (index, value) in sigma_dct.iter_mut().enumerate() {
        let inv_dct_factor = if index == 0 {
            std::f32::consts::FRAC_1_SQRT_2
        } else {
            1.0
        };
        let quantized_sigma = quantized.sigma_dct[index];
        *value = quantized_sigma as f32 * inv_dct_factor * SPLINE_CHANNEL_WEIGHTS[3] * inv_quant;
        let weight_f = (inv_quant * quantized_sigma.abs() as f32).ceil();
        let weight = weight_limit.min(weight_f.max(1.0)) as u64;
        width_estimate = width_estimate
            .checked_add(weight.saturating_mul(weight).saturating_mul(log_color))
            .ok_or(Error::InvalidCodestream("spline area is too large"))?;
    }
    *total_estimated_area = total_estimated_area
        .checked_add(width_estimate.saturating_mul(manhattan_distance))
        .ok_or(Error::InvalidCodestream("spline area is too large"))?;
    if *total_estimated_area > area_limit {
        return Err(Error::InvalidCodestream("spline area is too large"));
    }

    Ok(DequantizedSplineMetadata {
        control_points,
        color_dct,
        sigma_dct,
    })
}

fn inv_spline_adjusted_quant(adjustment: i32) -> f32 {
    if adjustment >= 0 {
        1.0 / (1.0 + 0.125 * adjustment as f32)
    } else {
        1.0 - 0.125 * adjustment as f32
    }
}

fn ceil_log2_nonzero(value: u64) -> u64 {
    debug_assert!(value != 0);
    if value <= 1 {
        0
    } else {
        u64::BITS as u64 - (value - 1).leading_zeros() as u64
    }
}

fn centripetal_catmull_rom_points(points: &[SplineFloatPoint]) -> Result<Vec<SplineFloatPoint>> {
    if points.is_empty() {
        return Ok(Vec::new());
    }
    if points.len() == 1 {
        return Ok(vec![points[0]]);
    }

    let mut control = Vec::with_capacity(points.len() + 2);
    control.push(point_add(points[0], point_sub(points[0], points[1])));
    control.extend_from_slice(points);
    control.push(point_add(
        points[points.len() - 1],
        point_sub(points[points.len() - 1], points[points.len() - 2]),
    ));

    const INTERPOLATED_POINTS_PER_SEGMENT: usize = 16;
    let mut result = Vec::with_capacity((points.len() - 1) * INTERPOLATED_POINTS_PER_SEGMENT + 1);
    for start in 0..control.len() - 3 {
        let p = &control[start..start + 4];
        result.push(p[1]);
        let mut d = [0.0f32; 3];
        let mut t = [0.0f32; 4];
        for index in 0..3 {
            d[index] = point_distance(p[index + 1], p[index]).sqrt();
            if d[index] == 0.0 {
                return Err(Error::InvalidCodestream(
                    "identical successive spline control points",
                ));
            }
            t[index + 1] = t[index] + d[index];
        }
        for index in 1..INTERPOLATED_POINTS_PER_SEGMENT {
            let tt = d[0] + (index as f32 / INTERPOLATED_POINTS_PER_SEGMENT as f32) * d[1];
            let mut a = [SplineFloatPoint { x: 0.0, y: 0.0 }; 3];
            for k in 0..3 {
                a[k] = point_lerp(p[k], p[k + 1], (tt - t[k]) / d[k]);
            }
            let mut b = [SplineFloatPoint { x: 0.0, y: 0.0 }; 2];
            for k in 0..2 {
                b[k] = point_lerp(a[k], a[k + 1], (tt - t[k]) / (d[k] + d[k + 1]));
            }
            result.push(point_lerp(b[0], b[1], (tt - t[1]) / d[1]));
        }
    }
    result.push(control[control.len() - 2]);
    Ok(result)
}

fn equally_spaced_spline_points(
    points: &[SplineFloatPoint],
) -> Result<Vec<(SplineFloatPoint, f32)>> {
    if points.is_empty() {
        return Err(Error::InvalidCodestream("empty spline"));
    }
    let mut result = Vec::new();
    let mut current = points[0];
    result.push((current, SPLINE_DESIRED_RENDERING_DISTANCE));
    let mut next_index = 0usize;
    while next_index < points.len() {
        let mut previous = current;
        let mut arclength_from_previous = 0.0f32;
        loop {
            if next_index == points.len() {
                result.push((previous, arclength_from_previous));
                return Ok(result);
            }
            let arclength_to_next = point_distance(points[next_index], previous);
            if arclength_from_previous + arclength_to_next >= SPLINE_DESIRED_RENDERING_DISTANCE {
                current = point_lerp(
                    previous,
                    points[next_index],
                    (SPLINE_DESIRED_RENDERING_DISTANCE - arclength_from_previous)
                        / arclength_to_next,
                );
                result.push((current, SPLINE_DESIRED_RENDERING_DISTANCE));
                break;
            }
            arclength_from_previous += arclength_to_next;
            previous = points[next_index];
            next_index += 1;
        }
    }
    Ok(result)
}

fn spline_segments_from_points(
    spline: &DequantizedSplineMetadata,
    points_to_draw: &[(SplineFloatPoint, f32)],
    arc_length: f32,
    segments: &mut Vec<SplineSegmentMetadata>,
    segments_by_y: &mut Vec<(usize, usize)>,
) {
    let inv_arc_length = 1.0 / arc_length;
    for (index, (point, intensity)) in points_to_draw.iter().enumerate() {
        let progress_along_arc =
            (index as f32 * SPLINE_DESIRED_RENDERING_DISTANCE * inv_arc_length).min(1.0);
        let mut color = [0.0f32; 3];
        for (channel, color_sample) in color.iter_mut().enumerate() {
            *color_sample = continuous_idct(
                &spline.color_dct[channel],
                (SPLINE_DCT_SIZE - 1) as f32 * progress_along_arc,
            );
        }
        let sigma = continuous_idct(
            &spline.sigma_dct,
            (SPLINE_DCT_SIZE - 1) as f32 * progress_along_arc,
        );
        compute_spline_segment(*point, *intensity, color, sigma, segments, segments_by_y);
    }
}

fn continuous_idct(dct: &[f32; SPLINE_DCT_SIZE], t: f32) -> f32 {
    let mut result = 0.0f32;
    for (index, value) in dct.iter().enumerate() {
        let cos_arg = std::f32::consts::PI / SPLINE_DCT_SIZE as f32 * index as f32 * (t + 0.5);
        result += std::f32::consts::SQRT_2 * value * fast_cosf(cos_arg);
    }
    result
}

fn compute_spline_segment(
    center: SplineFloatPoint,
    intensity: f32,
    color: [f32; 3],
    sigma: f32,
    segments: &mut Vec<SplineSegmentMetadata>,
    segments_by_y: &mut Vec<(usize, usize)>,
) {
    if !(sigma.is_finite() && sigma != 0.0 && (1.0 / sigma).is_finite() && intensity.is_finite()) {
        return;
    }
    let mut max_color = 0.01f32;
    for sample in color {
        max_color = max_color.max((sample * intensity).abs());
    }
    let maximum_distance = (-2.0 * sigma * sigma * (0.1f32.ln() * 3.0 - max_color.ln())).sqrt();
    let segment = SplineSegmentMetadata {
        center_x: center.x,
        center_y: center.y,
        maximum_distance,
        inv_sigma: 1.0 / sigma,
        sigma_over_4_times_intensity: 0.25 * sigma * intensity,
        color,
    };
    let y0 = (center.y - maximum_distance).round() as isize;
    let y1 = (center.y + maximum_distance).round() as isize + 1;
    let segment_index = segments.len();
    for y in y0.max(0)..y1 {
        segments_by_y.push((y as usize, segment_index));
    }
    segments.push(segment);
}

fn point_sub(a: SplineFloatPoint, b: SplineFloatPoint) -> SplineFloatPoint {
    SplineFloatPoint {
        x: a.x - b.x,
        y: a.y - b.y,
    }
}

fn point_add(a: SplineFloatPoint, b: SplineFloatPoint) -> SplineFloatPoint {
    SplineFloatPoint {
        x: a.x + b.x,
        y: a.y + b.y,
    }
}

fn point_lerp(a: SplineFloatPoint, b: SplineFloatPoint, t: f32) -> SplineFloatPoint {
    SplineFloatPoint {
        x: a.x + t * (b.x - a.x),
        y: a.y + t * (b.y - a.y),
    }
}

fn point_distance(a: SplineFloatPoint, b: SplineFloatPoint) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    dx.hypot(dy)
}

fn render_splines_into_modular_image(
    image: &mut ModularImage,
    splines: &SplineFrameMetadata,
    bit_depth: u32,
    full_width: u32,
    full_height: u32,
    image_origin: (u32, u32),
) -> Result<()> {
    if image.channels.len() < 3 {
        return Err(Error::Unsupported("spline rendering"));
    }
    if bit_depth == 0 || bit_depth > 30 {
        return Err(Error::Unsupported("spline rendering"));
    }
    let width = image.width as usize;
    let height = image.height as usize;
    for channel in image.channels.iter().take(3) {
        if channel.width as usize != width || channel.height as usize != height {
            return Err(Error::Unsupported("spline rendering"));
        }
    }

    let pixel_count = channel_sample_count(image.width, image.height)?;
    let max_sample = ((1u64 << bit_depth) - 1) as f32;
    let mut planes = [
        image.channels[0]
            .samples
            .iter()
            .map(|sample| *sample as f32 / max_sample)
            .collect::<Vec<_>>(),
        image.channels[1]
            .samples
            .iter()
            .map(|sample| *sample as f32 / max_sample)
            .collect::<Vec<_>>(),
        image.channels[2]
            .samples
            .iter()
            .map(|sample| *sample as f32 / max_sample)
            .collect::<Vec<_>>(),
    ];
    if planes.iter().any(|plane| plane.len() != pixel_count) {
        return Err(Error::InvalidCodestream("decoded pixel count mismatch"));
    }

    let plan = splines.render_plan_default_color_correlation(full_width, full_height)?;
    render_spline_plan_into_planes(&plan, width, height, image_origin, &mut planes)?;

    for (channel, plane) in image.channels.iter_mut().take(3).zip(planes) {
        for (sample, value) in channel.samples.iter_mut().zip(plane) {
            *sample = (value * max_sample).round().clamp(0.0, max_sample) as i32;
        }
    }
    Ok(())
}

pub fn render_splines_into_xyb_image(
    image: &mut VarDctXybImage,
    splines: &SplineFrameMetadata,
    full_width: u32,
    full_height: u32,
    image_origin: (u32, u32),
) -> Result<()> {
    let width = image.width as usize;
    let height = image.height as usize;
    if width == 0 || height == 0 {
        return Err(Error::InvalidCodestream("empty spline render image"));
    }
    let pixel_count = width
        .checked_mul(height)
        .ok_or(Error::InvalidCodestream("spline image size overflow"))?;
    if image
        .channels
        .iter()
        .any(|plane| plane.len() != pixel_count)
    {
        return Err(Error::InvalidCodestream("decoded pixel count mismatch"));
    }

    let plan = splines.render_plan_default_color_correlation(full_width, full_height)?;
    render_spline_plan_into_planes(&plan, width, height, image_origin, &mut image.channels)
}

fn render_spline_plan_into_planes(
    plan: &SplineRenderPlan,
    width: usize,
    height: usize,
    image_origin: (u32, u32),
    planes: &mut [Vec<f32>; 3],
) -> Result<()> {
    let origin_x = image_origin.0 as usize;
    let origin_y = image_origin.1 as usize;
    let image_y_end = origin_y
        .checked_add(height)
        .ok_or(Error::InvalidCodestream("spline ROI size overflow"))?;
    if image_y_end >= plan.segment_y_start.len() {
        return Err(Error::InvalidCodestream("spline ROI outside render plan"));
    }

    for local_y in 0..height {
        let image_y = origin_y + local_y;
        let row_start = local_y * width;
        for index in plan.segment_y_start[image_y]..plan.segment_y_start[image_y + 1] {
            let segment = &plan.segments[plan.segment_indices[index]];
            render_spline_segment_row(segment, width, origin_x, image_y, row_start, planes);
        }
    }
    Ok(())
}

fn render_spline_segment_row(
    segment: &SplineSegmentMetadata,
    width: usize,
    origin_x: usize,
    y: usize,
    row_start: usize,
    planes: &mut [Vec<f32>; 3],
) {
    let start = (segment.center_x - segment.maximum_distance).round() as isize;
    let end = (segment.center_x + segment.maximum_distance).round() as isize;
    let image_x0 = origin_x as isize;
    let image_x1 = origin_x.saturating_add(width) as isize;
    if end < image_x0 || start >= image_x1 {
        return;
    }
    let x0 = start.max(image_x0) as usize;
    let x1 = (end + 1).clamp(image_x0, image_x1) as usize;
    for image_x in x0..x1 {
        let dx = image_x as f32 - segment.center_x;
        let dy = y as f32 - segment.center_y;
        let distance = dx.hypot(dy);
        let positive = (distance * 0.5 + 0.353553391) * segment.inv_sigma;
        let negative = (distance * 0.5 - 0.353553391) * segment.inv_sigma;
        let one_dimensional_factor = fast_erff(positive) - fast_erff(negative);
        let local_intensity =
            segment.sigma_over_4_times_intensity * one_dimensional_factor * one_dimensional_factor;
        let pixel = row_start + image_x - origin_x;
        for (channel, plane) in planes.iter_mut().enumerate() {
            plane[pixel] += segment.color[channel] * local_intensity;
        }
    }
}

fn render_noise_into_modular_image(
    image: &mut ModularImage,
    noise: &NoiseFrameMetadata,
    plan: &ModularDecodePlan,
    image_origin: (u32, u32),
) -> Result<()> {
    if noise.lut.iter().all(|value| *value == 0) {
        return Ok(());
    }
    if image.channels.len() < 3 {
        return Err(Error::Unsupported("noise rendering"));
    }
    if plan.upsampling != 1 {
        return Err(Error::Unsupported("noise rendering with frame upsampling"));
    }
    if plan.channel_plan.bit_depth == 0 || plan.channel_plan.bit_depth > 30 {
        return Err(Error::Unsupported("noise rendering"));
    }
    if plan.color_transform == ColorTransform::Xyb {
        return Err(Error::Unsupported("noise rendering with XYB"));
    }
    if plan.color_transform != ColorTransform::None {
        return Err(Error::Unsupported("noise rendering with YCbCr"));
    }

    let width = image.width as usize;
    let height = image.height as usize;
    if width == 0 || height == 0 {
        return Err(Error::InvalidCodestream("empty noise render image"));
    }
    for channel in image.channels.iter().take(3) {
        if channel.width as usize != width || channel.height as usize != height {
            return Err(Error::Unsupported("noise rendering with shifted channels"));
        }
    }

    let full_width = plan.channel_plan.width as usize;
    let full_height = plan.channel_plan.height as usize;
    let origin_x = image_origin.0 as usize;
    let origin_y = image_origin.1 as usize;
    if origin_x
        .checked_add(width)
        .filter(|right| *right <= full_width)
        .is_none()
        || origin_y
            .checked_add(height)
            .filter(|bottom| *bottom <= full_height)
            .is_none()
    {
        return Err(Error::InvalidCodestream("noise ROI outside image"));
    }

    let pixel_count = channel_sample_count(image.width, image.height)?;
    let max_sample = ((1u64 << plan.channel_plan.bit_depth) - 1) as f32;
    let mut planes = [
        image.channels[0]
            .samples
            .iter()
            .map(|sample| *sample as f32 / max_sample)
            .collect::<Vec<_>>(),
        image.channels[1]
            .samples
            .iter()
            .map(|sample| *sample as f32 / max_sample)
            .collect::<Vec<_>>(),
        image.channels[2]
            .samples
            .iter()
            .map(|sample| *sample as f32 / max_sample)
            .collect::<Vec<_>>(),
    ];
    if planes.iter().any(|plane| plane.len() != pixel_count) {
        return Err(Error::InvalidCodestream("decoded pixel count mismatch"));
    }

    let random = generate_group_noise_planes(full_width, full_height, plan.group_dim as usize)?;
    let convolved = random.map(|plane| convolve_noise_plane(&plane, full_width, full_height));
    add_noise_to_planes(
        &mut planes,
        &convolved,
        noise,
        width,
        height,
        full_width,
        (origin_x, origin_y),
    )?;

    for (channel, plane) in image.channels.iter_mut().take(3).zip(planes) {
        for (sample, value) in channel.samples.iter_mut().zip(plane) {
            *sample = (value * max_sample).round().clamp(0.0, max_sample) as i32;
        }
    }
    Ok(())
}

pub fn render_noise_into_xyb_image(
    image: &mut VarDctXybImage,
    noise: &NoiseFrameMetadata,
    full_width: u32,
    full_height: u32,
    group_dim: u32,
    image_origin: (u32, u32),
) -> Result<()> {
    if noise.lut.iter().all(|value| *value == 0) {
        return Ok(());
    }
    let width = image.width as usize;
    let height = image.height as usize;
    if width == 0 || height == 0 {
        return Err(Error::InvalidCodestream("empty noise render image"));
    }
    let full_width = full_width as usize;
    let full_height = full_height as usize;
    let group_dim = group_dim as usize;
    let origin_x = image_origin.0 as usize;
    let origin_y = image_origin.1 as usize;
    if origin_x
        .checked_add(width)
        .filter(|right| *right <= full_width)
        .is_none()
        || origin_y
            .checked_add(height)
            .filter(|bottom| *bottom <= full_height)
            .is_none()
    {
        return Err(Error::InvalidCodestream("noise ROI outside image"));
    }

    let random = generate_group_noise_planes(full_width, full_height, group_dim)?;
    let convolved = random.map(|plane| convolve_noise_plane(&plane, full_width, full_height));
    add_noise_to_planes(
        &mut image.channels,
        &convolved,
        noise,
        width,
        height,
        full_width,
        (origin_x, origin_y),
    )
}

fn generate_group_noise_planes(
    width: usize,
    height: usize,
    group_dim: usize,
) -> Result<[Vec<f32>; 3]> {
    if width == 0 || height == 0 || group_dim == 0 {
        return Err(Error::InvalidCodestream("invalid noise dimensions"));
    }
    let pixels = width
        .checked_mul(height)
        .ok_or(Error::InvalidCodestream("noise image size overflow"))?;
    let mut planes = [
        vec![0.0f32; pixels],
        vec![0.0f32; pixels],
        vec![0.0f32; pixels],
    ];
    let groups_x = width.div_ceil(group_dim);
    let groups_y = height.div_ceil(group_dim);
    for gy in 0..groups_y {
        for gx in 0..groups_x {
            let x0 = gx * group_dim;
            let y0 = gy * group_dim;
            let group_width = group_dim.min(width - x0);
            let group_height = group_dim.min(height - y0);
            let mut rng = Xorshift128Plus::from_four_seeds(
                NOISE_DEFAULT_VISIBLE_FRAME_INDEX,
                NOISE_DEFAULT_NONVISIBLE_FRAME_INDEX,
                x0 as u32,
                y0 as u32,
            );
            for plane in &mut planes {
                fill_noise_rect(&mut rng, plane, width, x0, y0, group_width, group_height);
            }
        }
    }
    Ok(planes)
}

fn fill_noise_rect(
    rng: &mut Xorshift128Plus,
    plane: &mut [f32],
    plane_width: usize,
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
) {
    let mut batch = [0.0f32; NOISE_FLOATS_PER_BATCH];
    for y in 0..height {
        let mut x = 0usize;
        while x + NOISE_FLOATS_PER_BATCH < width {
            rng.fill_f32_batch(&mut batch);
            let row_start = (y0 + y) * plane_width + x0 + x;
            plane[row_start..row_start + NOISE_FLOATS_PER_BATCH].copy_from_slice(&batch);
            x += NOISE_FLOATS_PER_BATCH;
        }

        rng.fill_f32_batch(&mut batch);
        let row_start = (y0 + y) * plane_width + x0 + x;
        plane[row_start..row_start + width - x].copy_from_slice(&batch[..width - x]);
    }
}

fn convolve_noise_plane(input: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    for y in 0..height {
        for x in 0..width {
            let center = input[y * width + x];
            let mut others = 0.0f32;
            for dy in -NOISE_CONV_RADIUS..=NOISE_CONV_RADIUS {
                let sample_y = mirror_index(y as isize + dy, height);
                for dx in -NOISE_CONV_RADIUS..=NOISE_CONV_RADIUS {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let sample_x = mirror_index(x as isize + dx, width);
                    others += input[sample_y * width + sample_x];
                }
            }
            output[y * width + x] = others.mul_add(0.16, center * -3.84);
        }
    }
    output
}

fn add_noise_to_planes(
    planes: &mut [Vec<f32>; 3],
    convolved_noise: &[Vec<f32>; 3],
    noise: &NoiseFrameMetadata,
    width: usize,
    height: usize,
    full_width: usize,
    image_origin: (usize, usize),
) -> Result<()> {
    let expected_local_len = width
        .checked_mul(height)
        .ok_or(Error::InvalidCodestream("noise ROI size overflow"))?;
    if planes.iter().any(|plane| plane.len() != expected_local_len) {
        return Err(Error::InvalidCodestream("noise ROI plane size mismatch"));
    }
    let full_height = convolved_noise[0]
        .len()
        .checked_div(full_width)
        .ok_or(Error::InvalidCodestream("invalid noise plane size"))?;
    let expected_len = full_width
        .checked_mul(full_height)
        .ok_or(Error::InvalidCodestream("noise plane size overflow"))?;
    if full_width == 0
        || full_height == 0
        || convolved_noise
            .iter()
            .any(|plane| plane.len() != expected_len)
    {
        return Err(Error::InvalidCodestream("noise plane size mismatch"));
    }
    let lut = noise.strength_lut();
    for y in 0..height {
        let local_row = y * width;
        for x in 0..width {
            let local_index = local_row + x;
            let image_x = image_origin.0 + x;
            let noise_index = (image_origin.1 + y) * full_width + image_x;
            let vx = planes[0][local_index];
            let vy = planes[1][local_index];
            let in_g = (vy - vx) * 0.5;
            let in_r = (vy + vx) * 0.5;
            let strength_g = noise_strength_lut(&lut, in_g);
            let strength_r = noise_strength_lut(&lut, in_r);
            let rnd_r = convolved_noise[0][noise_index] * NOISE_NORM_CONST;
            let rnd_g = convolved_noise[1][noise_index] * NOISE_NORM_CONST;
            let rnd_cor = convolved_noise[2][noise_index] * NOISE_NORM_CONST;
            let red_noise = strength_r * rnd_r.mul_add(NOISE_RGN_CORR, NOISE_RG_CORR * rnd_cor);
            let green_noise = strength_g * rnd_g.mul_add(NOISE_RGN_CORR, NOISE_RG_CORR * rnd_cor);
            let rg_noise = red_noise + green_noise;
            planes[0][local_index] += red_noise - green_noise;
            planes[1][local_index] += rg_noise;
            planes[2][local_index] += rg_noise;
        }
    }
    Ok(())
}

fn noise_strength_lut(lut: &[f32; NOISE_LUT_SIZE], x: f32) -> f32 {
    let scale = (NOISE_LUT_SIZE - 2) as f32;
    let scaled = (x * scale).max(0.0);
    let (floor_x, frac_x) = if scaled >= scale + 1.0 {
        (NOISE_LUT_SIZE - 2, 1.0)
    } else {
        let floor = scaled.floor();
        (floor as usize, scaled - floor)
    };
    let low = lut[floor_x];
    let high = lut[floor_x + 1];
    ((high - low).mul_add(frac_x, low)).clamp(0.0, 1.0)
}

fn mirror_index(mut x: isize, size: usize) -> usize {
    let size = size as isize;
    while x < 0 || x >= size {
        if x < 0 {
            x = -x - 1;
        } else {
            x = 2 * size - 1 - x;
        }
    }
    x as usize
}

#[derive(Debug, Clone)]
struct Xorshift128Plus {
    s0: [u64; NOISE_XORSHIFT_LANES],
    s1: [u64; NOISE_XORSHIFT_LANES],
}

impl Xorshift128Plus {
    fn from_four_seeds(seed1: u32, seed2: u32, seed3: u32, seed4: u32) -> Self {
        let mut s0 = [0u64; NOISE_XORSHIFT_LANES];
        let mut s1 = [0u64; NOISE_XORSHIFT_LANES];
        s0[0] = split_mix64(
            ((u64::from(seed1) << 32) + u64::from(seed2)).wrapping_add(0x9E3779B97F4A7C15),
        );
        s1[0] = split_mix64(
            ((u64::from(seed3) << 32) + u64::from(seed4)).wrapping_add(0x9E3779B97F4A7C15),
        );
        for i in 1..NOISE_XORSHIFT_LANES {
            s0[i] = split_mix64(s0[i - 1]);
            s1[i] = split_mix64(s1[i - 1]);
        }
        Self { s0, s1 }
    }

    #[cfg(test)]
    fn from_seed(seed: u64) -> Self {
        let mut s0 = [0u64; NOISE_XORSHIFT_LANES];
        let mut s1 = [0u64; NOISE_XORSHIFT_LANES];
        s0[0] = split_mix64(seed.wrapping_add(0x9E3779B97F4A7C15));
        s1[0] = split_mix64(s0[0]);
        for i in 1..NOISE_XORSHIFT_LANES {
            s0[i] = split_mix64(s1[i - 1]);
            s1[i] = split_mix64(s0[i]);
        }
        Self { s0, s1 }
    }

    fn fill_u64(&mut self) -> [u64; NOISE_XORSHIFT_LANES] {
        let mut bits = [0u64; NOISE_XORSHIFT_LANES];
        for (i, value) in bits.iter_mut().enumerate() {
            let mut s1 = self.s0[i];
            let s0 = self.s1[i];
            *value = s1.wrapping_add(s0);
            self.s0[i] = s0;
            s1 ^= s1 << 23;
            self.s1[i] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
        }
        bits
    }

    fn fill_f32_batch(&mut self, output: &mut [f32; NOISE_FLOATS_PER_BATCH]) {
        let bits = self.fill_u64();
        for (index, value) in bits.into_iter().enumerate() {
            output[index * 2] = noise_bits_to_float(value as u32);
            output[index * 2 + 1] = noise_bits_to_float((value >> 32) as u32);
        }
    }
}

fn split_mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn noise_bits_to_float(bits: u32) -> f32 {
    f32::from_bits((bits >> 9) | 0x3F800000)
}

fn fast_cosf(value: f32) -> f32 {
    let pi2 = std::f32::consts::PI * 2.0;
    let pi2_inv = 0.5 / std::f32::consts::PI;
    let npi2 = (value * pi2_inv).floor() * pi2;
    let xmodpi2 = value - npi2;
    let x_pi = xmodpi2.min(pi2 - xmodpi2);
    let above_pihalf = x_pi >= std::f32::consts::FRAC_PI_2;
    let x_pihalf = if above_pihalf {
        std::f32::consts::PI - x_pi
    } else {
        x_pi
    };
    let xs = x_pihalf * 0.25;
    let x2 = xs * xs;
    let x4 = x2 * x2;
    let cosx_prescaling = x4.mul_add(0.06960438, x2.mul_add(-0.84087373, 1.68179268));
    let cosx_scale1 = cosx_prescaling.mul_add(cosx_prescaling, -1.414213562);
    let cosx_scale2 = cosx_scale1.mul_add(cosx_scale1, -1.0);
    if above_pihalf {
        -cosx_scale2
    } else {
        cosx_scale2
    }
}

fn fast_erff(value: f32) -> f32 {
    let x = value.abs();
    let denom1 = x.mul_add(7.77394369e-02, 2.05260015e-04);
    let denom2 = denom1.mul_add(x, 2.32120216e-01);
    let denom3 = denom2.mul_add(x, 2.77820801e-01);
    let denom4 = denom3.mul_add(x, 1.0);
    let denom5 = denom4 * denom4;
    let inv_denom5 = 1.0 / denom5;
    let result = 1.0 - inv_denom5 * inv_denom5;
    if value <= 0.0 { -result } else { result }
}

fn read_global_section(
    reader: &mut BitReader<'_>,
    metadata: &ImageMetadata,
    frame_header: &FrameHeader,
    section: &FrameSection,
) -> Result<ModularGlobalSection> {
    let features = read_dc_global_features(reader, frame_header)?;
    let dc_quant = read_dc_dequant_matrices(reader)?;
    let has_global_tree = reader.read_bool()?;
    let global_tree_metadata = if has_global_tree {
        Some(read_tree_metadata(
            reader,
            global_tree_size_limit(metadata, frame_header)?,
        )?)
    } else {
        None
    };
    let group_header = read_modular_group_header_metadata(reader)?;
    if group_header.use_global_tree && !has_global_tree {
        return Err(Error::InvalidCodestream(
            "modular stream references a missing global tree",
        ));
    }

    Ok(ModularGlobalSection {
        section_logical_id: section.logical_id,
        section_kind: section.kind,
        features,
        dc_quant_bits: dc_quant.map(f32::to_bits),
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

pub(crate) fn read_modular_global_tree_coding(
    reader: &mut BitReader<'_>,
    metadata: &ImageMetadata,
    frame_header: &FrameHeader,
) -> Result<ModularTreeCoding> {
    if !reader.read_bool()? {
        return Err(Error::InvalidCodestream("modular frame has no global tree"));
    }
    read_tree_coding(reader, global_tree_size_limit(metadata, frame_header)?)
}

pub(crate) fn probe_modular_global_tree_coding(
    reader: &mut BitReader<'_>,
    metadata: &ImageMetadata,
    frame_header: &FrameHeader,
) -> ModularTreeCodingProbe {
    let has_global_tree = match reader.read_bool() {
        Ok(has_global_tree) => has_global_tree,
        Err(error) => {
            return ModularTreeCodingProbe {
                has_global_tree_end_bits: None,
                tree_histogram_end_bits: None,
                tree_ans_start_bits: None,
                tree_end_bits: None,
                tree_node_count: None,
                tree_leaf_count: None,
                tree_leaves: Vec::new(),
                residual_context_count: None,
                residual_histogram_count: None,
                residual_histogram_probe: None,
                residual_coding_end_bits: None,
                error_bits: Some(reader.bits_consumed()),
                error: Some(error),
            };
        }
    };
    let has_global_tree_end_bits = Some(reader.bits_consumed());
    if !has_global_tree {
        return ModularTreeCodingProbe {
            has_global_tree_end_bits,
            tree_histogram_end_bits: None,
            tree_ans_start_bits: None,
            tree_end_bits: None,
            tree_node_count: None,
            tree_leaf_count: None,
            tree_leaves: Vec::new(),
            residual_context_count: None,
            residual_histogram_count: None,
            residual_histogram_probe: None,
            residual_coding_end_bits: None,
            error_bits: Some(reader.bits_consumed()),
            error: Some(Error::InvalidCodestream("modular frame has no global tree")),
        };
    }

    let tree_size_limit = match global_tree_size_limit(metadata, frame_header) {
        Ok(limit) => limit,
        Err(error) => {
            return ModularTreeCodingProbe {
                has_global_tree_end_bits,
                tree_histogram_end_bits: None,
                tree_ans_start_bits: None,
                tree_end_bits: None,
                tree_node_count: None,
                tree_leaf_count: None,
                tree_leaves: Vec::new(),
                residual_context_count: None,
                residual_histogram_count: None,
                residual_histogram_probe: None,
                residual_coding_end_bits: None,
                error_bits: Some(reader.bits_consumed()),
                error: Some(error),
            };
        }
    };
    match decode_tree_probe(reader, tree_size_limit) {
        Ok((tree, tree_histogram_end_bits, tree_ans_start_bits)) => {
            let tree_end_bits = Some(reader.bits_consumed());
            let tree_node_count = Some(tree.nodes.len());
            let tree_leaf_count =
                Some(tree.nodes.iter().filter(|node| node.property == -1).count());
            let tree_leaves = probe_tree_leaves(&tree);
            let contexts = tree.nodes.len().div_ceil(2);
            let residual_probe_reader = reader.clone();
            match decode_histograms(reader, contexts, false) {
                Ok(_) => {
                    let mut residual_probe_reader = residual_probe_reader;
                    let residual_histogram_probe =
                        probe_decode_histograms(&mut residual_probe_reader, contexts, false);
                    ModularTreeCodingProbe {
                        has_global_tree_end_bits,
                        tree_histogram_end_bits: Some(tree_histogram_end_bits),
                        tree_ans_start_bits: Some(tree_ans_start_bits),
                        tree_end_bits,
                        tree_node_count,
                        tree_leaf_count,
                        tree_leaves,
                        residual_context_count: Some(contexts),
                        residual_histogram_count: residual_histogram_probe.num_histograms,
                        residual_histogram_probe: Some(residual_histogram_probe),
                        residual_coding_end_bits: Some(reader.bits_consumed()),
                        error_bits: None,
                        error: None,
                    }
                }
                Err(error) => {
                    let mut residual_probe_reader = residual_probe_reader;
                    let residual_histogram_probe =
                        probe_decode_histograms(&mut residual_probe_reader, contexts, false);
                    ModularTreeCodingProbe {
                        has_global_tree_end_bits,
                        tree_histogram_end_bits: Some(tree_histogram_end_bits),
                        tree_ans_start_bits: Some(tree_ans_start_bits),
                        tree_end_bits,
                        tree_node_count,
                        tree_leaf_count,
                        tree_leaves,
                        residual_context_count: Some(contexts),
                        residual_histogram_count: residual_histogram_probe.num_histograms,
                        residual_histogram_probe: Some(residual_histogram_probe),
                        residual_coding_end_bits: None,
                        error_bits: Some(reader.bits_consumed()),
                        error: Some(error),
                    }
                }
            }
        }
        Err((error, tree_histogram_end_bits, tree_ans_start_bits)) => ModularTreeCodingProbe {
            has_global_tree_end_bits,
            tree_histogram_end_bits,
            tree_ans_start_bits,
            tree_end_bits: None,
            tree_node_count: None,
            tree_leaf_count: None,
            tree_leaves: Vec::new(),
            residual_context_count: None,
            residual_histogram_count: None,
            residual_histogram_probe: None,
            residual_coding_end_bits: None,
            error_bits: Some(reader.bits_consumed()),
            error: Some(error),
        },
    }
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
    let header = read_modular_group_header_metadata(&mut reader)?;
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
    read_dc_global_features(&mut reader, frame_header)?;
    read_dc_dequant_matrices(&mut reader)?;
    if !reader.read_bool()? {
        return Err(Error::InvalidCodestream("modular frame has no global tree"));
    }
    let tree = read_tree_coding(&mut reader, MAX_TREE_SIZE)?;
    let header = read_modular_group_header_metadata(&mut reader)?;
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

fn assemble_modular_image_region(
    plan: &ModularDecodePlan,
    residuals: &ModularResiduals,
    region_plan: &ModularRegionPlan,
) -> Result<ModularImage> {
    let rect = region_plan.requested_rect;
    let decode_rect = region_plan.decode_rect;
    let channel_regions = &region_plan.channel_rects;

    let mut channels = plan
        .channel_plan
        .channels
        .iter()
        .enumerate()
        .zip(channel_regions)
        .map(|((index, channel), region)| {
            let (width, height) = if index < plan.channel_plan.nb_meta_channels {
                (channel.width, channel.height)
            } else {
                (region.width, region.height)
            };
            let samples = channel_sample_count(width, height).map(|count| vec![0; count])?;
            Ok(ModularImageChannel {
                width,
                height,
                samples,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    if let Some(global) = &residuals.global {
        copy_decoded_group_region(
            &mut channels,
            global,
            channel_regions,
            plan.channel_plan.nb_meta_channels,
        )?;
    }
    for group in &residuals.groups {
        copy_decoded_group_region(
            &mut channels,
            group,
            channel_regions,
            plan.channel_plan.nb_meta_channels,
        )?;
    }

    let image = inverse_transforms(
        &region_channel_plan(plan, decode_rect, channel_regions),
        &plan.global.group_header,
        channels,
    )?;
    crop_modular_image(image, decode_rect, rect)
}

fn plan_modular_region(plan: &ModularDecodePlan, rect: ImageRect) -> Result<ModularRegionPlan> {
    let has_squeeze = plan
        .global
        .group_header
        .transforms
        .iter()
        .any(|transform| transform.id == TransformId::Squeeze);
    if plan
        .global
        .group_header
        .transforms
        .iter()
        .any(|transform| !modular_region_transform_supported(transform))
    {
        return Err(Error::Unsupported(
            "modular region assembly with transforms",
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
    let decode_rect = modular_region_dependency_rect(plan, rect);
    let Some(decode_rect_right) = decode_rect.x.checked_add(decode_rect.width) else {
        return Err(Error::InvalidCodestream("modular region overflow"));
    };
    let Some(decode_rect_bottom) = decode_rect.y.checked_add(decode_rect.height) else {
        return Err(Error::InvalidCodestream("modular region overflow"));
    };
    if decode_rect.x > rect.x
        || decode_rect.y > rect.y
        || decode_rect_right < rect_right
        || decode_rect_bottom < rect_bottom
    {
        return Err(Error::InvalidCodestream(
            "modular dependency region misses requested region",
        ));
    }

    let channel_regions = modular_region_channel_rects(&plan.channel_plan, decode_rect)?;
    if plan
        .channel_plan
        .channels
        .iter()
        .zip(&channel_regions)
        .skip(plan.channel_plan.nb_meta_channels)
        .any(|(channel, region)| {
            (!has_squeeze
                && (channel.hshift != 0
                    || channel.vshift != 0
                    || channel.width != plan.channel_plan.width
                    || channel.height != plan.channel_plan.height))
                || region.width == 0
                || region.height == 0
        })
    {
        return Err(Error::Unsupported(
            "modular region assembly with shifted channels",
        ));
    }

    Ok(ModularRegionPlan {
        requested_rect: rect,
        decode_rect,
        channel_rects: channel_regions,
        has_squeeze,
    })
}

fn modular_region_transform_supported(transform: &ModularTransform) -> bool {
    match transform.id {
        TransformId::Palette | TransformId::Rct => true,
        TransformId::Squeeze => !transform.squeezes.is_empty(),
    }
}

fn modular_region_dependency_rect(plan: &ModularDecodePlan, rect: ImageRect) -> ImageRect {
    let mut has_horizontal_squeeze = false;
    let mut has_vertical_squeeze = false;
    for squeeze in plan
        .global
        .group_header
        .transforms
        .iter()
        .filter(|transform| transform.id == TransformId::Squeeze)
        .flat_map(|transform| &transform.squeezes)
    {
        has_horizontal_squeeze |= squeeze.horizontal;
        has_vertical_squeeze |= !squeeze.horizontal;
    }

    if has_horizontal_squeeze || has_vertical_squeeze {
        let y = if has_vertical_squeeze { 0 } else { rect.y };
        let mut height = if has_vertical_squeeze {
            if rect.y == 0 {
                let bottom = rect.y.saturating_add(rect.height);
                let alignment = max_non_meta_vshift(&plan.channel_plan)
                    .and_then(|shift| 1u32.checked_shl(shift as u32))
                    .unwrap_or(1);
                bottom
                    .saturating_add(alignment)
                    .div_ceil(alignment)
                    .saturating_mul(alignment)
            } else {
                plan.channel_plan.height
            }
        } else {
            rect.height
        };
        height = height.min(plan.channel_plan.height);
        return ImageRect {
            x: 0,
            y,
            width: plan.channel_plan.width,
            height,
        };
    }
    rect
}

fn max_non_meta_vshift(channel_plan: &ModularChannelPlan) -> Option<i32> {
    channel_plan
        .channels
        .iter()
        .skip(channel_plan.nb_meta_channels)
        .map(|channel| channel.vshift)
        .filter(|shift| *shift >= 0)
        .max()
}

fn modular_region_channel_rects(
    channel_plan: &ModularChannelPlan,
    rect: ImageRect,
) -> Result<Vec<ImageRect>> {
    channel_plan
        .channels
        .iter()
        .enumerate()
        .map(|(index, channel)| {
            if index < channel_plan.nb_meta_channels {
                Ok(ImageRect {
                    x: 0,
                    y: 0,
                    width: channel.width,
                    height: channel.height,
                })
            } else {
                shifted_channel_region(channel, rect)
            }
        })
        .collect()
}

fn shifted_channel_region(channel: &ModularChannel, rect: ImageRect) -> Result<ImageRect> {
    if channel.hshift < 0 || channel.vshift < 0 {
        return Err(Error::Unsupported(
            "modular region assembly with shifted channels",
        ));
    }
    let Some(right) = rect.x.checked_add(rect.width) else {
        return Err(Error::InvalidCodestream("modular region overflow"));
    };
    let Some(bottom) = rect.y.checked_add(rect.height) else {
        return Err(Error::InvalidCodestream("modular region overflow"));
    };
    let hscale = 1u32
        .checked_shl(channel.hshift as u32)
        .ok_or(Error::InvalidCodestream("invalid modular channel shift"))?;
    let vscale = 1u32
        .checked_shl(channel.vshift as u32)
        .ok_or(Error::InvalidCodestream("invalid modular channel shift"))?;
    let x = rect.x / hscale;
    let y = rect.y / vscale;
    let channel_right = right.div_ceil(hscale).min(channel.width);
    let channel_bottom = bottom.div_ceil(vscale).min(channel.height);
    Ok(ImageRect {
        x,
        y,
        width: channel_right.saturating_sub(x),
        height: channel_bottom.saturating_sub(y),
    })
}

fn region_channel_plan(
    plan: &ModularDecodePlan,
    rect: ImageRect,
    channel_regions: &[ImageRect],
) -> ModularChannelPlan {
    ModularChannelPlan {
        width: rect.width,
        height: rect.height,
        bit_depth: plan.channel_plan.bit_depth,
        nb_meta_channels: plan.channel_plan.nb_meta_channels,
        channels: plan
            .channel_plan
            .channels
            .iter()
            .enumerate()
            .zip(channel_regions)
            .map(|((index, channel), region)| {
                let (width, height) = if index < plan.channel_plan.nb_meta_channels {
                    (channel.width, channel.height)
                } else {
                    (region.width, region.height)
                };
                ModularChannel {
                    width,
                    height,
                    hshift: channel.hshift,
                    vshift: channel.vshift,
                    component: channel.component,
                }
            })
            .collect(),
    }
}

fn crop_modular_image(
    image: ModularImage,
    source_rect: ImageRect,
    target_rect: ImageRect,
) -> Result<ModularImage> {
    if source_rect == target_rect {
        return Ok(image);
    }
    let x = target_rect
        .x
        .checked_sub(source_rect.x)
        .ok_or(Error::InvalidCodestream("invalid modular crop region"))? as usize;
    let y = target_rect
        .y
        .checked_sub(source_rect.y)
        .ok_or(Error::InvalidCodestream("invalid modular crop region"))? as usize;
    let width = target_rect.width as usize;
    let height = target_rect.height as usize;
    let mut channels = Vec::with_capacity(image.channels.len());
    for channel in image.channels {
        if channel.width != source_rect.width || channel.height != source_rect.height {
            return Err(Error::Unsupported(
                "modular region assembly with shifted channels",
            ));
        }
        let source_width = channel.width as usize;
        let mut samples = Vec::with_capacity(width * height);
        for row in y..y + height {
            let start = row * source_width + x;
            samples.extend_from_slice(&channel.samples[start..start + width]);
        }
        channels.push(ModularImageChannel {
            width: target_rect.width,
            height: target_rect.height,
            samples,
        });
    }
    Ok(ModularImage {
        width: target_rect.width,
        height: target_rect.height,
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
        copy_decoded_channel(dst, decoded)?;
    }
    Ok(())
}

fn copy_decoded_channel(
    dst: &mut ModularImageChannel,
    decoded: &ModularDecodedChannel,
) -> Result<()> {
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
    Ok(())
}

fn copy_decoded_group_region(
    channels: &mut [ModularImageChannel],
    group: &ModularDecodedGroup,
    channel_regions: &[ImageRect],
    nb_meta_channels: usize,
) -> Result<()> {
    for decoded in &group.channels {
        let dst = channels
            .get_mut(decoded.channel_index)
            .ok_or(Error::InvalidCodestream("invalid modular decoded channel"))?;
        let rect = *channel_regions
            .get(decoded.channel_index)
            .ok_or(Error::InvalidCodestream("invalid modular decoded channel"))?;
        if decoded.channel_index < nb_meta_channels {
            copy_decoded_channel(dst, decoded)?;
            continue;
        }
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
    let (_, decoded) = decode_modular_stream_from_reader(
        &mut reader,
        group.section_physical_index,
        group.stream_id,
        &group.channels,
        Some(global_tree),
    )?;
    Ok(decoded)
}

pub(crate) fn decode_modular_stream_from_reader(
    reader: &mut BitReader<'_>,
    section_physical_index: usize,
    stream_id: usize,
    channels: &[ModularGroupChannelPlan],
    global_tree: Option<&ModularTreeCoding>,
) -> Result<(ModularGroupHeader, ModularDecodedGroup)> {
    let mut header = read_modular_group_header_metadata(reader)?;
    let tree = if header.use_global_tree {
        global_tree.cloned().ok_or(Error::InvalidCodestream(
            "modular stream references a missing global tree",
        ))?
    } else {
        read_tree_coding(reader, MAX_TREE_SIZE)?
    };
    let (residual_channels, transform_plan) =
        transformed_group_channel_plan(channels, &mut header)?;
    let mut symbol_reader = AnsSymbolReader::new(
        tree.code.clone(),
        reader,
        channel_distance_multiplier(&residual_channels),
    )?;
    let mut decoded_channels = Vec::new();
    for (local_channel, channel) in residual_channels.iter().enumerate() {
        decoded_channels.push(decode_channel_residuals(
            reader,
            &mut symbol_reader,
            &tree,
            &header.weighted_predictor,
            channel,
            local_channel,
            stream_id,
        )?);
    }
    if !symbol_reader.check_final_state() {
        return Err(Error::InvalidCodestream(
            "invalid modular residual ANS state",
        ));
    }
    if let Some(transform_plan) = transform_plan {
        decoded_channels =
            inverse_group_transforms(&transform_plan, &header, decoded_channels, channels)?;
    }
    Ok((
        header,
        ModularDecodedGroup {
            section_physical_index,
            stream_id,
            channels: decoded_channels,
            bits_consumed: reader.bits_consumed(),
        },
    ))
}

fn transformed_group_channel_plan(
    channels: &[ModularGroupChannelPlan],
    header: &mut ModularGroupHeader,
) -> Result<(Vec<ModularGroupChannelPlan>, Option<ModularChannelPlan>)> {
    if header.transforms.is_empty() {
        return Ok((channels.to_vec(), None));
    }

    let mut transform_channels = channels
        .iter()
        .map(|channel| ModularChannel {
            width: channel.width,
            height: channel.height,
            hshift: channel.hshift,
            vshift: channel.vshift,
            component: Some(channel.channel_index),
        })
        .collect::<Vec<_>>();
    let mut nb_meta_channels = 0usize;
    for transform in &mut header.transforms {
        apply_transform_metadata(transform, &mut transform_channels, &mut nb_meta_channels)?;
    }

    let residual_channels = transform_channels
        .iter()
        .enumerate()
        .map(|(index, channel)| ModularGroupChannelPlan {
            channel_index: index,
            width: channel.width,
            height: channel.height,
            x0: 0,
            y0: 0,
            hshift: channel.hshift,
            vshift: channel.vshift,
        })
        .collect::<Vec<_>>();
    let transform_plan = ModularChannelPlan {
        width: channels.first().map(|channel| channel.width).unwrap_or(0),
        height: channels.first().map(|channel| channel.height).unwrap_or(0),
        bit_depth: 0,
        nb_meta_channels,
        channels: transform_channels,
    };
    Ok((residual_channels, Some(transform_plan)))
}

fn inverse_group_transforms(
    transform_plan: &ModularChannelPlan,
    header: &ModularGroupHeader,
    decoded_channels: Vec<ModularDecodedChannel>,
    output_channels: &[ModularGroupChannelPlan],
) -> Result<Vec<ModularDecodedChannel>> {
    let image_channels = decoded_channels
        .into_iter()
        .map(|channel| ModularImageChannel {
            width: channel.width,
            height: channel.height,
            samples: channel.samples,
        })
        .collect::<Vec<_>>();
    let image = inverse_transforms(transform_plan, header, image_channels)?;
    if image.channels.len() != output_channels.len() {
        return Err(Error::InvalidCodestream(
            "modular transform output channel count mismatch",
        ));
    }
    image
        .channels
        .into_iter()
        .zip(output_channels)
        .map(|(channel, plan)| {
            if channel.width != plan.width || channel.height != plan.height {
                return Err(Error::InvalidCodestream(
                    "modular transform output channel size mismatch",
                ));
            }
            Ok(ModularDecodedChannel {
                channel_index: plan.channel_index,
                x0: plan.x0,
                y0: plan.y0,
                width: channel.width,
                height: channel.height,
                samples: channel.samples,
            })
        })
        .collect()
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

fn read_dc_dequant_matrices(reader: &mut BitReader<'_>) -> Result<[f32; 3]> {
    let all_default = reader.read_bool()?;
    if all_default {
        return Ok(DEFAULT_MODULAR_DC_QUANT);
    }

    let mut coefficients = [0.0f32; 3];
    for coefficient in &mut coefficients {
        *coefficient = reader.read_f16()? * (1.0 / 128.0);
        if *coefficient <= 0.0 {
            return Err(Error::InvalidCodestream(
                "invalid DC dequant matrix coefficient",
            ));
        }
    }
    Ok(coefficients)
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

pub(crate) fn read_modular_group_header_metadata(
    reader: &mut BitReader<'_>,
) -> Result<ModularGroupHeader> {
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
    decode_tree_nodes(reader, &mut symbol_reader, &context_map, tree_size_limit)
}

type DecodeTreeProbeResult =
    std::result::Result<(MaTree, usize, usize), (Error, Option<usize>, Option<usize>)>;

fn decode_tree_probe(reader: &mut BitReader<'_>, tree_size_limit: usize) -> DecodeTreeProbeResult {
    let (code, context_map) =
        decode_histograms(reader, TREE_CONTEXTS, false).map_err(|error| (error, None, None))?;
    let tree_histogram_end_bits = reader.bits_consumed();
    let mut symbol_reader = AnsSymbolReader::new(code, reader, 0)
        .map_err(|error| (error, Some(tree_histogram_end_bits), None))?;
    let tree_ans_start_bits = reader.bits_consumed();
    decode_tree_nodes(reader, &mut symbol_reader, &context_map, tree_size_limit)
        .map(|tree| (tree, tree_histogram_end_bits, tree_ans_start_bits))
        .map_err(|error| {
            (
                error,
                Some(tree_histogram_end_bits),
                Some(tree_ans_start_bits),
            )
        })
}

fn decode_tree_nodes(
    reader: &mut BitReader<'_>,
    symbol_reader: &mut AnsSymbolReader,
    context_map: &[u8],
    tree_size_limit: usize,
) -> Result<MaTree> {
    let mut nodes = Vec::new();
    let mut leaf_id = 0u32;
    let mut to_decode = 1usize;
    let tree_size_limit = tree_size_limit.min(MAX_TREE_SIZE);

    while to_decode > 0 {
        if nodes.len() > tree_size_limit {
            return Err(Error::InvalidCodestream("modular MA tree is too large"));
        }
        to_decode -= 1;
        let prop1 = symbol_reader.read_hybrid_uint(PROPERTY_CONTEXT, reader, context_map)?;
        if prop1 > 256 {
            return Err(Error::InvalidCodestream("invalid modular MA tree property"));
        }
        let property = prop1 as i32 - 1;
        if property == -1 {
            let predictor =
                symbol_reader.read_hybrid_uint(PREDICTOR_CONTEXT, reader, context_map)?;
            let predictor = ModularPredictor::try_from(predictor)?;
            let predictor_offset = i64::from(unpack_signed(symbol_reader.read_hybrid_uint(
                OFFSET_CONTEXT,
                reader,
                context_map,
            )?));
            let mul_log =
                symbol_reader.read_hybrid_uint(MULTIPLIER_LOG_CONTEXT, reader, context_map)?;
            if mul_log >= 31 {
                return Err(Error::InvalidCodestream(
                    "invalid modular MA tree multiplier logarithm",
                ));
            }
            let mul_bits =
                symbol_reader.read_hybrid_uint(MULTIPLIER_BITS_CONTEXT, reader, context_map)?;
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
            context_map,
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

fn probe_tree_leaves(tree: &MaTree) -> Vec<MaTreeLeafProbe> {
    tree.nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| node.property == -1)
        .map(|(node_index, node)| MaTreeLeafProbe {
            leaf_index: node.lchild as usize,
            node_index,
            residual_context: node.lchild as usize,
            predictor: node.predictor,
            predictor_offset: node.predictor_offset,
            multiplier: node.multiplier,
        })
        .collect()
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
    fn xorshift128plus_matches_reference_vector() {
        let mut rng = Xorshift128Plus::from_seed(12345);

        assert_eq!(
            rng.fill_u64(),
            [
                0x6E901576D477CBB1,
                0xE9E53789195DA2A2,
                0xB681F6DDA5E0AE99,
                0x8EFD18CE21FD6896,
                0xA898A80DF75CF532,
                0x50CEB2C9E2DE7E32,
                0x3CA7C2FEB25C0DD0,
                0xA4D0866B80B4D836,
            ]
        );
    }

    #[test]
    fn noise_strength_lut_interpolates_and_clamps() {
        let lut = [0.0, 0.1, 0.2, 0.4, 0.7, 1.0, 0.9, 0.8];

        assert_close(noise_strength_lut(&lut, -0.5), 0.0);
        assert_close(noise_strength_lut(&lut, 0.25), 0.15);
        assert_close(noise_strength_lut(&lut, 2.0), 0.8);
    }

    #[test]
    fn noise_roi_rendering_matches_crop_of_full_noise_rendering() {
        let mut plan = region_test_plan(20, 18, 3);
        plan.group_dim = 8;
        plan.color_transform = ColorTransform::None;
        let noise = NoiseFrameMetadata {
            lut: [8, 16, 32, 64, 96, 128, 160, 192],
            bits_consumed: 80,
        };
        let full_samples = (0..20 * 18)
            .map(|index| 80 + (index % 40) as i32)
            .collect::<Vec<_>>();
        let mut full = ModularImage {
            width: 20,
            height: 18,
            channels: vec![
                image_channel(20, 18, &full_samples),
                image_channel(20, 18, &full_samples),
                image_channel(20, 18, &full_samples),
            ],
        };

        render_noise_into_modular_image(&mut full, &noise, &plan, (0, 0)).unwrap();

        let origin = (5, 4);
        let mut roi = ModularImage {
            width: 7,
            height: 6,
            channels: (0..3)
                .map(|_| crop_channel(&full_samples, 20, origin, 7, 6))
                .collect(),
        };
        render_noise_into_modular_image(&mut roi, &noise, &plan, origin).unwrap();

        for channel in 0..3 {
            assert_eq!(
                roi.channels[channel].samples,
                crop_samples(&full.channels[channel].samples, 20, origin, 7, 6)
            );
        }
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
    fn modular_stream_transforms_are_applied_around_group_residuals() {
        let output_channels = [group_channel(3, 2, 1, 2, 1, 0, 0)];
        let mut header = ModularGroupHeader {
            use_global_tree: true,
            weighted_predictor: WeightedPredictorHeader::default(),
            transforms: vec![ModularTransform {
                id: TransformId::Squeeze,
                begin_c: 0,
                rct_type: None,
                num_c: None,
                nb_colors: None,
                nb_deltas: None,
                predictor: None,
                squeezes: vec![SqueezeParams {
                    horizontal: true,
                    in_place: true,
                    begin_c: 0,
                    num_c: 1,
                }],
            }],
        };

        let (residual_channels, transform_plan) =
            transformed_group_channel_plan(&output_channels, &mut header).unwrap();
        assert_eq!(
            residual_channels
                .iter()
                .map(|channel| (
                    channel.width,
                    channel.height,
                    channel.hshift,
                    channel.vshift
                ))
                .collect::<Vec<_>>(),
            vec![(1, 1, 1, 0), (1, 1, 1, 0)]
        );

        let decoded = vec![
            ModularDecodedChannel {
                channel_index: 0,
                x0: 0,
                y0: 0,
                width: 1,
                height: 1,
                samples: vec![10],
            },
            ModularDecodedChannel {
                channel_index: 1,
                x0: 0,
                y0: 0,
                width: 1,
                height: 1,
                samples: vec![0],
            },
        ];
        let output =
            inverse_group_transforms(&transform_plan.unwrap(), &header, decoded, &output_channels)
                .unwrap();
        assert_eq!(
            output,
            vec![ModularDecodedChannel {
                channel_index: 3,
                x0: 2,
                y0: 1,
                width: 2,
                height: 1,
                samples: vec![10, 10],
            }]
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

        let region = image_rect(2, 1, 3, 2);
        let image = assemble_test_region(&plan, &residuals, region).unwrap();

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

        let region = image_rect(2, 0, 4, 2);
        let image = assemble_test_region(&plan, &residuals, region).unwrap();

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

        let region = image_rect(1, 0, 2, 2);
        let image = assemble_test_region(&plan, &residuals, region).unwrap();

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
            assemble_test_region(&plan, &residuals, image_rect(0, 0, 4, 4)),
            Err(Error::Unsupported(
                "modular region assembly with shifted channels"
            ))
        );
    }

    #[test]
    fn classifies_shifted_modular_region_plan_failure() {
        let mut plan = region_test_plan(8, 4, 1);
        plan.channel_plan.channels[0].hshift = 1;

        assert_eq!(
            plan_modular_region(&plan, image_rect(0, 0, 4, 4)),
            Err(Error::Unsupported(
                "modular region assembly with shifted channels"
            ))
        );
    }

    #[test]
    fn assembles_modular_region_with_rct_transform() {
        let mut plan = region_test_plan(4, 2, 3);
        plan.global.group_header.transforms.push(ModularTransform {
            id: TransformId::Rct,
            begin_c: 0,
            rct_type: Some(10),
            num_c: None,
            nb_colors: None,
            nb_deltas: None,
            predictor: None,
            squeezes: Vec::new(),
        });
        let residuals = ModularResiduals {
            global: None,
            groups: vec![ModularDecodedGroup {
                section_physical_index: 0,
                stream_id: 10,
                channels: vec![
                    decoded_channel(0, 0, 0, 4, 2, &[10, 20, 30, 40]),
                    decoded_channel(1, 0, 0, 4, 2, &[1, 2, 3, 4]),
                    decoded_channel(2, 0, 0, 4, 2, &[5, 6, 7, 8]),
                ],
                bits_consumed: 0,
            }],
        };

        let region = image_rect(1, 0, 2, 1);
        let image = assemble_test_region(&plan, &residuals, region).unwrap();

        assert_eq!(image.channels.len(), 3);
        assert_eq!(image.channels[0].samples, vec![26, 37]);
        assert_eq!(image.channels[1].samples, vec![20, 30]);
        assert_eq!(image.channels[2].samples, vec![22, 33]);
    }

    #[test]
    fn assembles_modular_region_with_palette_transform() {
        let mut plan = region_test_plan(4, 2, 0);
        plan.channel_plan.nb_meta_channels = 1;
        plan.channel_plan.channels = vec![
            ModularChannel {
                width: 2,
                height: 1,
                hshift: -1,
                vshift: 0,
                component: None,
            },
            ModularChannel {
                width: 4,
                height: 2,
                hshift: 0,
                vshift: 0,
                component: Some(0),
            },
        ];
        plan.global.group_header.transforms.push(ModularTransform {
            id: TransformId::Palette,
            begin_c: 0,
            rct_type: None,
            num_c: Some(1),
            nb_colors: Some(2),
            nb_deltas: Some(0),
            predictor: Some(ModularPredictor::Zero),
            squeezes: Vec::new(),
        });
        let residuals = ModularResiduals {
            global: Some(ModularDecodedGroup {
                section_physical_index: 0,
                stream_id: 2,
                channels: vec![decoded_channel_exact(0, 0, 0, 2, 1, &[10, 20])],
                bits_consumed: 0,
            }),
            groups: vec![ModularDecodedGroup {
                section_physical_index: 1,
                stream_id: 10,
                channels: vec![decoded_channel_exact(
                    1,
                    0,
                    0,
                    4,
                    2,
                    &[0, 1, 0, 1, 1, 0, 1, 0],
                )],
                bits_consumed: 0,
            }],
        };

        let region = image_rect(1, 0, 2, 2);
        let image = assemble_test_region(&plan, &residuals, region).unwrap();

        assert_eq!(image.width, 2);
        assert_eq!(image.height, 2);
        assert_eq!(image.channels.len(), 1);
        assert_eq!(image.channels[0].samples, vec![20, 10, 10, 20]);
    }

    #[test]
    fn rejects_modular_region_with_squeeze_transform() {
        let mut plan = region_test_plan(8, 4, 1);
        plan.global.group_header.transforms.push(ModularTransform {
            id: TransformId::Squeeze,
            begin_c: 0,
            rct_type: None,
            num_c: None,
            nb_colors: None,
            nb_deltas: None,
            predictor: None,
            squeezes: Vec::new(),
        });
        let residuals = ModularResiduals {
            global: None,
            groups: Vec::new(),
        };

        assert_eq!(
            assemble_test_region(&plan, &residuals, image_rect(0, 0, 4, 4)),
            Err(Error::Unsupported(
                "modular region assembly with transforms"
            ))
        );
    }

    #[test]
    fn classifies_default_squeeze_region_plan_failure() {
        let mut plan = region_test_plan(8, 4, 1);
        plan.global.group_header.transforms.push(ModularTransform {
            id: TransformId::Squeeze,
            begin_c: 0,
            rct_type: None,
            num_c: None,
            nb_colors: None,
            nb_deltas: None,
            predictor: None,
            squeezes: Vec::new(),
        });

        assert_eq!(
            plan_modular_region(&plan, image_rect(0, 0, 4, 4)),
            Err(Error::Unsupported(
                "modular region assembly with transforms"
            ))
        );
    }

    #[test]
    fn squeezed_modular_dependency_region_crops_vertical_prefix() {
        let mut plan = region_test_plan(128, 128, 3);
        for channel in &mut plan.channel_plan.channels {
            channel.vshift = 5;
        }
        plan.global.group_header.transforms.push(ModularTransform {
            id: TransformId::Squeeze,
            begin_c: 0,
            rct_type: None,
            num_c: None,
            nb_colors: None,
            nb_deltas: None,
            predictor: None,
            squeezes: vec![
                SqueezeParams {
                    horizontal: true,
                    in_place: true,
                    begin_c: 0,
                    num_c: 3,
                },
                SqueezeParams {
                    horizontal: false,
                    in_place: true,
                    begin_c: 0,
                    num_c: 3,
                },
            ],
        });

        assert_eq!(
            modular_region_dependency_rect(&plan, image_rect(19, 0, 37, 29)),
            image_rect(0, 0, 128, 64)
        );
    }

    #[test]
    fn interior_vertical_squeeze_dependency_region_keeps_full_height() {
        let mut plan = region_test_plan(128, 128, 3);
        for channel in &mut plan.channel_plan.channels {
            channel.vshift = 5;
        }
        plan.global.group_header.transforms.push(ModularTransform {
            id: TransformId::Squeeze,
            begin_c: 0,
            rct_type: None,
            num_c: None,
            nb_colors: None,
            nb_deltas: None,
            predictor: None,
            squeezes: vec![SqueezeParams {
                horizontal: false,
                in_place: true,
                begin_c: 0,
                num_c: 3,
            }],
        });

        assert_eq!(
            modular_region_dependency_rect(&plan, image_rect(19, 23, 37, 29)),
            image_rect(0, 0, 128, 128)
        );
    }

    #[test]
    fn horizontal_squeeze_dependency_region_preserves_requested_rows() {
        let mut plan = region_test_plan(128, 128, 3);
        plan.global.group_header.transforms.push(ModularTransform {
            id: TransformId::Squeeze,
            begin_c: 0,
            rct_type: None,
            num_c: None,
            nb_colors: None,
            nb_deltas: None,
            predictor: None,
            squeezes: vec![SqueezeParams {
                horizontal: true,
                in_place: true,
                begin_c: 0,
                num_c: 3,
            }],
        });

        assert_eq!(
            modular_region_dependency_rect(&plan, image_rect(19, 23, 37, 29)),
            image_rect(0, 23, 128, 29)
        );
    }

    #[test]
    fn dequantizes_spline_metadata_with_default_color_correlation() {
        let frame = fixture_spline_frame();
        let dequantized = frame
            .dequantize_default_color_correlation(2048, 2048)
            .unwrap();

        assert_eq!(dequantized.len(), 1);
        assert_eq!(
            dequantized[0].control_points,
            vec![
                SplineFloatPoint { x: 64.0, y: 378.0 },
                SplineFloatPoint {
                    x: 826.0,
                    y: 1113.0
                },
                SplineFloatPoint { x: 679.0, y: 56.0 },
                SplineFloatPoint { x: 70.0, y: 280.0 },
                SplineFloatPoint {
                    x: 1540.0,
                    y: 125.0
                },
                SplineFloatPoint {
                    x: 1540.0,
                    y: 1920.0
                },
                SplineFloatPoint {
                    x: 420.0,
                    y: 1540.0
                },
            ]
        );
        assert_close(dequantized[0].color_dct[0][0], 0.49893454);
        assert_close(dequantized[0].color_dct[0][1], 0.4998);
        assert_close(dequantized[0].color_dct[1][0], 0.7424621);
        assert_close(dequantized[0].color_dct[1][30], 0.225);
        assert_close(dequantized[0].color_dct[2][0], 0.0);
        assert_close(dequantized[0].color_dct[2][1], 0.49);
        assert_close(dequantized[0].color_dct[2][2], 0.56);
        assert_close(dequantized[0].color_dct[2][30], 0.015);
        assert_close(dequantized[0].sigma_dct[0], 12.019613);
        assert_close(dequantized[0].sigma_dct[7], 3.9996);
        assert_close(dequantized[0].sigma_dct[31], 6.9993);
    }

    #[test]
    fn spline_catmull_rom_uses_centripetal_parameterization() {
        let points = [
            SplineFloatPoint { x: 0.0, y: 0.0 },
            SplineFloatPoint { x: 4.0, y: 0.0 },
            SplineFloatPoint { x: 13.0, y: 0.0 },
        ];
        let interpolated = centripetal_catmull_rom_points(&points).unwrap();

        assert_eq!(interpolated.len(), 33);
        assert_close(interpolated[1].x, 0.24707031);
        assert_close(interpolated[8].x, 1.9);
        assert_close(interpolated[17].x, 4.463623);
        assert_close(interpolated[24].x, 8.275);
        assert_eq!(interpolated[32], SplineFloatPoint { x: 13.0, y: 0.0 });
    }

    #[test]
    fn builds_spline_render_plan_row_index() {
        let plan = fixture_spline_frame()
            .render_plan_default_color_correlation(2048, 2048)
            .unwrap();

        assert_eq!(plan.splines.len(), 1);
        assert!(!plan.segments.is_empty());
        assert_eq!(plan.segment_y_start.len(), 2049);
        assert!(plan.segment_indices.len() >= *plan.segment_y_start.last().unwrap());
        assert!(
            plan.segment_y_start
                .windows(2)
                .all(|window| window[0] <= window[1])
        );
        assert!(plan.segments.iter().all(|segment| {
            segment.maximum_distance.is_finite()
                && segment.maximum_distance >= 0.0
                && segment.inv_sigma.is_finite()
        }));
    }

    #[test]
    fn renders_splines_into_xyb_planes_in_image_space() {
        let spline = fixture_spline_frame();
        let plan = spline
            .render_plan_default_color_correlation(2048, 2048)
            .unwrap();
        let segment = plan.segments.first().unwrap();
        let origin_x = (segment.center_x as i32 - 32).clamp(0, 2048 - 64) as u32;
        let origin_y = (segment.center_y as i32 - 24).clamp(0, 2048 - 48) as u32;
        let mut image = VarDctXybImage {
            width: 64,
            height: 48,
            groups_assembled: 0,
            groups_missing: 0,
            channels: [vec![0.0; 64 * 48], vec![0.0; 64 * 48], vec![0.0; 64 * 48]],
        };

        render_splines_into_xyb_image(&mut image, &spline, 2048, 2048, (origin_x, origin_y))
            .unwrap();

        assert!(image.channels.iter().any(|plane| {
            plane
                .iter()
                .any(|sample| sample.is_finite() && sample.abs() > 0.0)
        }));
    }

    fn assemble_test_region(
        plan: &ModularDecodePlan,
        residuals: &ModularResiduals,
        region: ImageRect,
    ) -> Result<ModularImage> {
        let region_plan = plan_modular_region(plan, region)?;
        assemble_modular_image_region(plan, residuals, &region_plan)
    }

    fn image_channel(width: u32, height: u32, samples: &[i32]) -> ModularImageChannel {
        ModularImageChannel {
            width,
            height,
            samples: samples.to_vec(),
        }
    }

    fn crop_channel(
        samples: &[i32],
        source_width: usize,
        origin: (u32, u32),
        width: u32,
        height: u32,
    ) -> ModularImageChannel {
        ModularImageChannel {
            width,
            height,
            samples: crop_samples(samples, source_width, origin, width, height),
        }
    }

    fn crop_samples(
        samples: &[i32],
        source_width: usize,
        origin: (u32, u32),
        width: u32,
        height: u32,
    ) -> Vec<i32> {
        let mut cropped = Vec::with_capacity((width * height) as usize);
        let origin_x = origin.0 as usize;
        let origin_y = origin.1 as usize;
        for y in 0..height as usize {
            let start = (origin_y + y) * source_width + origin_x;
            cropped.extend_from_slice(&samples[start..start + width as usize]);
        }
        cropped
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

    fn fixture_spline_frame() -> SplineFrameMetadata {
        let mut color_dct = [[0i32; SPLINE_DCT_SIZE]; 3];
        color_dct[0][0] = 168;
        color_dct[0][1] = 119;
        color_dct[1][0] = 14;
        color_dct[1][30] = 3;
        color_dct[2][0] = -15;
        color_dct[2][1] = 7;
        color_dct[2][2] = 8;
        color_dct[2][30] = -3;
        let mut sigma_dct = [0i32; SPLINE_DCT_SIZE];
        sigma_dct[0] = 51;
        sigma_dct[7] = 12;
        sigma_dct[31] = 21;
        SplineFrameMetadata {
            quantization_adjustment: 0,
            starting_points: vec![SplinePoint { x: 64, y: 378 }],
            splines: vec![QuantizedSplineMetadata {
                control_points: vec![
                    (762, 735),
                    (-909, -1792),
                    (-462, 1281),
                    (2079, -379),
                    (-1470, 1950),
                    (-1120, -2175),
                ],
                color_dct,
                sigma_dct,
            }],
            bits_consumed: 423,
        }
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1.0e-5, "{actual} != {expected}");
    }

    fn region_test_plan(width: u32, height: u32, channels: usize) -> ModularDecodePlan {
        ModularDecodePlan {
            global: ModularGlobalSection {
                section_logical_id: 0,
                section_kind: FrameSectionKind::Combined,
                features: FrameFeatureMetadata::default(),
                dc_quant_bits: DEFAULT_MODULAR_DC_QUANT.map(f32::to_bits),
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
            group_dim: 256,
            upsampling: 1,
            color_transform: ColorTransform::Xyb,
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

    fn decoded_channel_exact(
        channel_index: usize,
        x0: u32,
        y0: u32,
        width: u32,
        height: u32,
        samples: &[i32],
    ) -> ModularDecodedChannel {
        assert_eq!(samples.len(), width as usize * height as usize);
        ModularDecodedChannel {
            channel_index,
            x0,
            y0,
            width,
            height,
            samples: samples.to_vec(),
        }
    }
}
