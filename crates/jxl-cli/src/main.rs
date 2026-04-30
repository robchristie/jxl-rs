use std::{fs, path::PathBuf};

use clap::Parser;
use jxl::{ColorSpace, FileFormat};

#[derive(Debug, Parser)]
#[command(about = "Inspect JPEG XL container and codestream metadata")]
struct Args {
    /// Print parsed container boxes.
    #[arg(short, long)]
    verbose: bool,

    /// JPEG XL file to inspect.
    input: PathBuf,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("jxlinfo-rs: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let bytes = fs::read(&args.input)?;
    let info = jxl::inspect(&bytes)?;

    match info.format {
        FileFormat::NakedCodestream => println!("JPEG XL codestream (ISO/IEC 18181-1)"),
        FileFormat::Container => println!("JPEG XL file format container (ISO/IEC 18181-2)"),
    }
    println!("Dimensions: {}x{}", info.width, info.height);
    println!(
        "Bit depth: {}{}",
        if info.metadata.bit_depth.floating_point_sample {
            "float"
        } else {
            "uint"
        },
        info.metadata.bit_depth.bits_per_sample
    );
    println!("Color channels: {}", info.metadata.num_color_channels());
    println!("Extra channels: {}", info.metadata.extra_channels.len());
    if let Some(alpha) = info.metadata.alpha_channel() {
        println!(
            "Alpha: {}-bit{}",
            alpha.bit_depth.bits_per_sample,
            if alpha.alpha_associated {
                ", premultiplied"
            } else {
                ""
            }
        );
    }
    println!("Orientation: {}", info.metadata.orientation);
    println!(
        "Encoded color space: {}",
        if info.metadata.xyb_encoded {
            "XYB"
        } else {
            "original profile"
        }
    );
    println!("Color space: {}", info.metadata.color_encoding.color_space);
    println!("White point: {}", info.metadata.color_encoding.white_point);
    if info.metadata.color_encoding.color_space != ColorSpace::Gray {
        println!("Primaries: {}", info.metadata.color_encoding.primaries);
    }
    if let Some(gamma) = info.metadata.color_encoding.gamma {
        println!("Gamma: {:.7}", gamma as f64 / 10_000_000.0);
    } else {
        println!(
            "Transfer function: {}",
            info.metadata.color_encoding.transfer_function
        );
    }
    println!(
        "Rendering intent: {}",
        info.metadata.color_encoding.rendering_intent
    );
    if info.metadata.tone_mapping != jxl::ToneMapping::default() {
        println!(
            "Intensity target: {:.6} nits",
            info.metadata.tone_mapping.intensity_target
        );
        println!("Min nits: {:.6}", info.metadata.tone_mapping.min_nits);
        println!(
            "Relative to max display: {}",
            info.metadata.tone_mapping.relative_to_max_display
        );
        println!(
            "Linear below: {:.6}",
            info.metadata.tone_mapping.linear_below
        );
    }
    if let Some(preview) = info.metadata.preview_size {
        println!("Preview: {}x{}", preview.width, preview.height);
    }
    if let Some(animation) = info.metadata.animation {
        println!(
            "Animation: {}/{} ticks/s, loops={}, timecodes={}",
            animation.tps_numerator,
            animation.tps_denominator,
            animation.num_loops,
            animation.have_timecodes
        );
    }
    if let Some(icc_profile) = &info.icc_profile {
        println!("ICC profile: {} bytes", icc_profile.len());
    }
    if let Some(frame) = &info.first_frame {
        println!(
            "First frame: {}, {}, {}x{} at ({},{}), passes={}, groups={}",
            frame.encoding,
            frame.frame_type,
            frame.frame_size.width,
            frame.frame_size.height,
            frame.frame_origin.x0,
            frame.frame_origin.y0,
            frame.passes.num_passes,
            frame.group_layout.num_groups
        );
        if frame.animation_frame.duration != 0 {
            println!("First frame duration: {}", frame.animation_frame.duration);
        }
        if let Some(frame_data) = &info.first_frame_data {
            println!(
                "First frame data: sections={}, payload={} bytes",
                frame_data.sections.len(),
                frame_data.payload_size
            );
        }
        if let Some(modular) = &info.first_frame_modular {
            println!(
                "First frame modular global: tree={} transforms={} channels={} groups={}",
                modular.global.has_global_tree,
                modular.global.group_header.transforms.len(),
                modular.channel_plan.channels.len(),
                modular.groups.len()
            );
        }
        if let Some(vardct) = &info.first_frame_vardct {
            println!(
                "First frame VarDCT: sections={} combined={} ac_groups={} dc_groups={}",
                vardct.sections.len(),
                vardct.is_combined,
                vardct.ac_groups.len(),
                vardct.dc_groups.len()
            );
        }
    } else {
        println!("First frame: not parsed");
    }

    if args.verbose {
        for record in info.boxes {
            let size = record
                .total_size()
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unbounded".to_string());
            let contents = record
                .content_size
                .map(|value| value.to_string())
                .unwrap_or_else(|| "to EOF".to_string());
            println!(
                "Box: type=\"{}\" size={} contents={} offset={}",
                record.box_type_string(),
                size,
                contents,
                record.offset
            );
        }
        for (index, channel) in info.metadata.extra_channels.iter().enumerate() {
            println!(
                "Extra channel {}: type=\"{}\" bits={} dim_shift={} name=\"{}\"",
                index,
                channel.channel_type,
                channel.bit_depth.bits_per_sample,
                channel.dim_shift,
                channel.name
            );
        }
        println!(
            "Custom transform data: {}",
            if info.transform_data.is_default() {
                "default".to_string()
            } else {
                format!(
                    "custom_weights_mask=0x{:x}, custom_opsin_matrix={}",
                    info.transform_data.custom_weights_mask,
                    info.transform_data.opsin_inverse_matrix.is_some()
                )
            }
        );
        if let Some(frame) = &info.first_frame {
            println!("First frame flags: 0x{:x}", frame.flags);
            println!("First frame color transform: {}", frame.color_transform);
            println!(
                "First frame upsampling: color={}, extra={:?}",
                frame.upsampling, frame.extra_channel_upsampling
            );
            println!(
                "First frame blending: mode=\"{}\" source={} alpha={} clamp={}",
                frame.blending_info.mode,
                frame.blending_info.source,
                frame.blending_info.alpha_channel,
                frame.blending_info.clamp
            );
            println!(
                "First frame group layout: dim={} groups={}x{}",
                frame.group_layout.group_dim,
                frame.group_layout.groups_x,
                frame.group_layout.groups_y
            );
            println!(
                "First frame DC group layout: dim={} groups={}x{}",
                frame.group_layout.dc_group_dim,
                frame.group_layout.dc_groups_x,
                frame.group_layout.dc_groups_y
            );
            println!(
                "First frame loop filter: gaborish={} epf_iters={}",
                frame.loop_filter.gab, frame.loop_filter.epf_iters
            );
        }
        if let Some(frame_data) = &info.first_frame_data {
            println!(
                "First frame TOC: entries={} permutation={} payload_start={}",
                frame_data.toc.entries.len(),
                frame_data.toc.has_permutation,
                frame_data.payload_start_offset
            );
            for section in &frame_data.sections {
                println!(
                    "First frame section {}: logical={} kind={:?} offset={} size={}",
                    section.physical_index,
                    section.logical_id,
                    section.kind,
                    section.codestream_offset,
                    section.size
                );
            }
        }
        if let Some(modular) = &info.first_frame_modular {
            let global = &modular.global;
            println!(
                "First frame modular global section: logical={} kind={:?} bits={}",
                global.section_logical_id, global.section_kind, global.bits_consumed
            );
            if let Some(tree) = &global.global_tree {
                println!(
                    "First frame modular global tree: nodes={} contexts={} context_map={}",
                    tree.nodes.len(),
                    global.global_tree_contexts.unwrap_or_default(),
                    global.global_tree_context_map_size.unwrap_or_default()
                );
            }
            println!(
                "First frame modular group header: use_global_tree={} wp_default={} transforms={}",
                global.group_header.use_global_tree,
                global.group_header.weighted_predictor.all_default,
                global.group_header.transforms.len()
            );
            println!(
                "First frame modular channel plan: {}x{} bit_depth={} meta={} channels={}",
                modular.channel_plan.width,
                modular.channel_plan.height,
                modular.channel_plan.bit_depth,
                modular.channel_plan.nb_meta_channels,
                modular.channel_plan.channels.len()
            );
            for (index, channel) in modular.channel_plan.channels.iter().enumerate() {
                println!(
                    "First frame modular channel {}: {}x{} shift={}x{} component={:?}",
                    index,
                    channel.width,
                    channel.height,
                    channel.hshift,
                    channel.vshift,
                    channel.component
                );
            }
            if let Some(residuals) = &modular.residuals {
                println!(
                    "First frame modular residuals: global={} groups={}",
                    residuals.global.is_some(),
                    residuals.groups.len()
                );
                if let Some(global) = &residuals.global {
                    println!(
                        "First frame modular residual global: stream_id={} channels={} bits={}",
                        global.stream_id,
                        global.channels.len(),
                        global.bits_consumed
                    );
                    for channel in &global.channels {
                        let min = channel.samples.iter().min().copied().unwrap_or_default();
                        let max = channel.samples.iter().max().copied().unwrap_or_default();
                        println!(
                            "First frame modular residual global channel {}: {}x{} at ({},{}) samples={} min={} max={}",
                            channel.channel_index,
                            channel.width,
                            channel.height,
                            channel.x0,
                            channel.y0,
                            channel.samples.len(),
                            min,
                            max
                        );
                    }
                }
                for group in &residuals.groups {
                    println!(
                        "First frame modular residual group {}: stream_id={} channels={} bits={}",
                        group.section_physical_index,
                        group.stream_id,
                        group.channels.len(),
                        group.bits_consumed
                    );
                    for channel in &group.channels {
                        let min = channel.samples.iter().min().copied().unwrap_or_default();
                        let max = channel.samples.iter().max().copied().unwrap_or_default();
                        println!(
                            "First frame modular residual channel {}: {}x{} at ({},{}) samples={} min={} max={}",
                            channel.channel_index,
                            channel.width,
                            channel.height,
                            channel.x0,
                            channel.y0,
                            channel.samples.len(),
                            min,
                            max
                        );
                    }
                }
            } else {
                println!("First frame modular residuals: unsupported");
            }
            if let Some(image) = &modular.image {
                println!(
                    "First frame modular image: {}x{} channels={}",
                    image.width,
                    image.height,
                    image.channels.len()
                );
                for (index, channel) in image.channels.iter().enumerate() {
                    let min = channel.samples.iter().min().copied().unwrap_or_default();
                    let max = channel.samples.iter().max().copied().unwrap_or_default();
                    println!(
                        "First frame modular image channel {}: {}x{} samples={} min={} max={}",
                        index,
                        channel.width,
                        channel.height,
                        channel.samples.len(),
                        min,
                        max
                    );
                }
            } else {
                println!("First frame modular image: unsupported");
            }
            for (index, transform) in global.group_header.transforms.iter().enumerate() {
                println!(
                    "First frame modular transform {}: id={:?} begin_c={} rct_type={:?} num_c={:?} colors={:?} deltas={:?} squeezes={}",
                    index,
                    transform.id,
                    transform.begin_c,
                    transform.rct_type,
                    transform.num_c,
                    transform.nb_colors,
                    transform.nb_deltas,
                    transform.squeezes.len()
                );
            }
            for group in &modular.groups {
                println!(
                    "First frame modular group section {}: logical={} kind={:?} stream_id={} size={} bits={} header={} local_tree={}",
                    group.section_physical_index,
                    group.section_logical_id,
                    group.section_kind,
                    group.stream_id,
                    group.payload_size,
                    group.bits_consumed,
                    group.header.is_some(),
                    group.local_tree.is_some()
                );
                for channel in &group.channels {
                    println!(
                        "First frame modular group section {} channel {}: rect={}x{} at ({},{}) shift={}x{}",
                        group.section_physical_index,
                        channel.channel_index,
                        channel.width,
                        channel.height,
                        channel.x0,
                        channel.y0,
                        channel.hshift,
                        channel.vshift
                    );
                }
                if let Some(header) = &group.header {
                    println!(
                        "First frame modular group header {}: use_global_tree={} wp_default={} transforms={}",
                        group.section_physical_index,
                        header.use_global_tree,
                        header.weighted_predictor.all_default,
                        header.transforms.len()
                    );
                }
                if let Some(tree) = &group.local_tree {
                    println!(
                        "First frame modular group local tree {}: nodes={} contexts={} context_map={}",
                        group.section_physical_index,
                        tree.tree.nodes.len(),
                        tree.contexts,
                        tree.context_map_size
                    );
                }
            }
        }
        if let Some(vardct) = &info.first_frame_vardct {
            println!(
                "First frame VarDCT plan: {}x{} group_dim={} groups={}x{} dc_groups={}x{} combined={}",
                vardct.width,
                vardct.height,
                vardct.group_dim,
                vardct.groups_x,
                vardct.groups_y,
                vardct.dc_groups_x,
                vardct.dc_groups_y,
                vardct.is_combined
            );
            if let Some(section) = &vardct.global_section {
                println!(
                    "First frame VarDCT global section: logical={} kind={:?} offset={} size={}",
                    section.section_logical_id,
                    section.section_kind,
                    section.codestream_offset,
                    section.payload_size
                );
            }
            if let Some(global) = info
                .first_frame_vardct_plan
                .as_ref()
                .and_then(|plan| plan.global.as_ref())
            {
                println!(
                    "First frame VarDCT global metadata: bits={} dc_dequant_default={} global_scale={} quant_dc={} block_contexts={} block_ctx_map={} color_default={} color_factor={}",
                    global.bits_consumed,
                    global.dc_dequant.all_default,
                    global.quantizer.global_scale,
                    global.quantizer.quant_dc,
                    global.block_context_map.num_contexts,
                    global.block_context_map.context_map_size,
                    global.color_correlation.all_default,
                    global.color_correlation.color_factor
                );
                println!(
                    "First frame VarDCT global cursor: dc_dequant={} quantizer={} block_context={} color_correlation={}",
                    global.cursor.dc_dequant_end_bits,
                    global.cursor.quantizer_end_bits,
                    global.cursor.block_context_end_bits,
                    global.cursor.color_correlation_end_bits
                );
            }
            if let Some(section) = &vardct.ac_global_section {
                println!(
                    "First frame VarDCT AC global section: logical={} offset={} size={}",
                    section.section_logical_id, section.codestream_offset, section.payload_size
                );
            }
            if let Some(ac_global) = info
                .first_frame_vardct_plan
                .as_ref()
                .and_then(|plan| plan.ac_global_metadata.as_ref())
            {
                println!(
                    "First frame VarDCT AC global metadata: default_quant={} num_histograms={} used_acs={} bits={:?} error={:?}",
                    ac_global.all_default_quant_matrices.unwrap_or(false),
                    ac_global.num_histograms.unwrap_or_default(),
                    ac_global.used_acs.unwrap_or_default(),
                    ac_global.bits_consumed,
                    ac_global.parse_error
                );
                for pass in &ac_global.passes {
                    println!(
                        "First frame VarDCT AC global pass {}: used_orders={} coeff_order_bits={:?} coeff_orders={} histogram_contexts={} histograms={} histogram_bits={:?} error={:?}",
                        pass.pass,
                        pass.used_orders.unwrap_or_default(),
                        pass.coeff_order_end_bits,
                        pass.coeff_orders.len(),
                        pass.histogram_contexts.unwrap_or_default(),
                        pass.histogram_count.unwrap_or_default(),
                        pass.histogram_end_bits,
                        pass.error
                    );
                    for order in &pass.coeff_orders {
                        println!(
                            "First frame VarDCT AC coeff order pass {} order={} channel={} skip={} size={} end={} checksum={}",
                            pass.pass,
                            order.order,
                            order.channel,
                            order.skip,
                            order.size,
                            order.permutation_end,
                            order.checksum
                        );
                    }
                }
            }
            for section in &vardct.sections {
                println!(
                    "First frame VarDCT section {}: logical={} kind={:?} offset={} size={}",
                    section.section_physical_index,
                    section.section_logical_id,
                    section.section_kind,
                    section.codestream_offset,
                    section.payload_size
                );
            }
            for group in &vardct.ac_groups {
                println!(
                    "First frame VarDCT AC group {}: rect={}x{} at ({},{})",
                    group.group, group.width, group.height, group.x, group.y
                );
            }
            for section in &vardct.ac_group_sections {
                println!(
                    "First frame VarDCT AC section: pass={} group={} physical={} size={}",
                    section.pass,
                    section.group.group,
                    section.section.section_physical_index,
                    section.section.payload_size
                );
            }
            if let Some(plan) = info.first_frame_vardct_plan.as_ref() {
                for metadata in &plan.ac_group_metadata {
                    println!(
                        "First frame VarDCT AC group metadata: pass={} group={} payload_bits={} selector_bits={} selector={:?} selector_end={:?} ans_bits={:?}..{:?} coeff_start={:?} modular_start={:?} error={:?}",
                        metadata.payload.pass,
                        metadata.payload.group.group,
                        metadata.cursor.payload_end_bits,
                        metadata.histogram_selector_bits,
                        metadata.histogram_selector,
                        metadata.cursor.histogram_selector_end_bits,
                        metadata.cursor.ans_state_start_bits,
                        metadata.cursor.ans_state_end_bits,
                        metadata.cursor.coefficient_stream_start_bits,
                        metadata.cursor.modular_ac_start_bits,
                        metadata.parse_error
                    );
                    if let Some(probe) = &metadata.coefficient_probe {
                        println!(
                            "First frame VarDCT AC coeff probe: pass={} group={} block=({}, {}) channel={} strategy={} order={} covered={} size={} block_ctx={} nzero_ctx={} clustered_ctx={} nzero_bits={}..{} nzeros={} events={} block_bits={:?} remaining={:?} checksum={}",
                            metadata.payload.pass,
                            metadata.payload.group.group,
                            probe.block_x,
                            probe.block_y,
                            probe.channel,
                            probe.raw_strategy,
                            probe.order,
                            probe.covered_blocks,
                            probe.block_size,
                            probe.block_context,
                            probe.nonzero_context,
                            probe.clustered_context,
                            probe.start_bits,
                            probe.nzeros_end_bits,
                            probe.nzeros,
                            probe.coefficient_events.len(),
                            probe.block_end_bits,
                            probe.remaining_nzeros,
                            probe.coefficient_event_checksum
                        );
                    }
                    if let Some(trace) = &metadata.channel_trace {
                        println!(
                            "First frame VarDCT AC channel trace: pass={} group={} channel={} blocks={} events={} final_bits={} row_checksum={} coeff_checksum={} summaries={}",
                            metadata.payload.pass,
                            metadata.payload.group.group,
                            trace.channel,
                            trace.blocks_decoded,
                            trace.coefficient_events_decoded,
                            trace.final_bits,
                            trace.row_nzeros_checksum,
                            trace.coefficient_event_checksum,
                            trace.block_summaries.len()
                        );
                    }
                    if let Some(summary) = &metadata.coefficient_summary {
                        println!(
                            "First frame VarDCT AC coefficient summary: pass={} group={} blocks={} final_bits={} first_block_checksum={} ch0=({},{},{}) ch1=({},{},{}) ch2=({},{},{})",
                            summary.pass,
                            summary.group,
                            summary.blocks_decoded,
                            summary.final_bits,
                            summary.first_block_checksum,
                            summary.per_channel[0].blocks_decoded,
                            summary.per_channel[0].nonzero_coefficients,
                            summary.per_channel[0].coefficient_checksum,
                            summary.per_channel[1].blocks_decoded,
                            summary.per_channel[1].nonzero_coefficients,
                            summary.per_channel[1].coefficient_checksum,
                            summary.per_channel[2].blocks_decoded,
                            summary.per_channel[2].nonzero_coefficients,
                            summary.per_channel[2].coefficient_checksum
                        );
                    }
                    if let Some(grid) = &metadata.coefficient_grid {
                        println!(
                            "First frame VarDCT AC coefficient grid: pass={} group={} size={}x{} ch0=({},{}) ch1=({},{}) ch2=({},{})",
                            grid.pass,
                            grid.group,
                            grid.width_blocks,
                            grid.height_blocks,
                            grid.per_channel[0].nonzero_coefficients,
                            grid.per_channel[0].coefficient_checksum,
                            grid.per_channel[1].nonzero_coefficients,
                            grid.per_channel[1].coefficient_checksum,
                            grid.per_channel[2].nonzero_coefficients,
                            grid.per_channel[2].coefficient_checksum
                        );
                    }
                    if let Some(grid) = &metadata.base_dequantized_grid {
                        println!(
                            "First frame VarDCT AC base dequantized grid: pass={} group={} size={}x{} inv_global_scale_bits={} ch0=({},{}) ch1=({},{}) ch2=({},{})",
                            grid.pass,
                            grid.group,
                            grid.width_blocks,
                            grid.height_blocks,
                            grid.inv_global_scale_bits,
                            grid.per_channel[0].nonzero_coefficients,
                            grid.per_channel[0].coefficient_checksum,
                            grid.per_channel[1].nonzero_coefficients,
                            grid.per_channel[1].coefficient_checksum,
                            grid.per_channel[2].nonzero_coefficients,
                            grid.per_channel[2].coefficient_checksum
                        );
                    }
                    if let Some(grid) = &metadata.dequantized_grid {
                        println!(
                            "First frame VarDCT AC dequantized grid: pass={} group={} size={}x{} ch0=({},{}) ch1=({},{}) ch2=({},{})",
                            grid.pass,
                            grid.group,
                            grid.width_blocks,
                            grid.height_blocks,
                            grid.per_channel[0].nonzero_coefficients,
                            grid.per_channel[0].coefficient_checksum,
                            grid.per_channel[1].nonzero_coefficients,
                            grid.per_channel[1].coefficient_checksum,
                            grid.per_channel[2].nonzero_coefficients,
                            grid.per_channel[2].coefficient_checksum
                        );
                    }
                    if let Some(grid) = &metadata.spatial_grid {
                        println!(
                            "First frame VarDCT AC DCT8 spatial grid: pass={} group={} size={}x{} blocks=({},{},{}) ch0=({},{}) ch1=({},{}) ch2=({},{})",
                            grid.pass,
                            grid.group,
                            grid.width_blocks,
                            grid.height_blocks,
                            grid.blocks_attempted,
                            grid.blocks_transformed,
                            grid.blocks_skipped,
                            grid.per_channel[0].nonzero_samples,
                            grid.per_channel[0].sample_checksum,
                            grid.per_channel[1].nonzero_samples,
                            grid.per_channel[1].sample_checksum,
                            grid.per_channel[2].nonzero_samples,
                            grid.per_channel[2].sample_checksum
                        );
                    }
                }
            }
            for group in &vardct.dc_groups {
                println!(
                    "First frame VarDCT DC group {}: rect={}x{} at ({},{})",
                    group.group, group.width, group.height, group.x, group.y
                );
            }
            for section in &vardct.dc_group_sections {
                println!(
                    "First frame VarDCT DC section: group={} physical={} size={}",
                    section.group.group,
                    section.section.section_physical_index,
                    section.section.payload_size
                );
            }
            if let Some(plan) = &info.first_frame_vardct_plan {
                for group in &plan.dc_group_metadata {
                    println!(
                        "First frame VarDCT DC stream group {}: stream_id={} bits={} channels={} error={:?}",
                        group.payload.group.group,
                        group.payload.var_dct_dc_stream_id,
                        group.cursor.var_dct_dc_end_bits.unwrap_or_default(),
                        group
                            .var_dct_dc
                            .as_ref()
                            .map(|decoded| decoded.channels.len())
                            .unwrap_or_default(),
                        group.parse_error
                    );
                    if let Some(decoded) = &group.var_dct_dc {
                        for channel in &decoded.channels {
                            let min = channel.samples.iter().min().copied().unwrap_or_default();
                            let max = channel.samples.iter().max().copied().unwrap_or_default();
                            let sum = channel
                                .samples
                                .iter()
                                .fold(0i64, |sum, sample| sum + i64::from(*sample));
                            println!(
                                "First frame VarDCT DC stream channel {}: {}x{} at ({},{}) samples={} min={} max={} sum={}",
                                channel.channel_index,
                                channel.width,
                                channel.height,
                                channel.x0,
                                channel.y0,
                                channel.samples.len(),
                                min,
                                max,
                                sum
                            );
                        }
                    }
                    println!(
                        "First frame VarDCT modular DC stream group {}: stream_id={} bits={} channels={} error={:?}",
                        group.payload.group.group,
                        group.payload.modular_dc_stream_id,
                        group.cursor.modular_dc_end_bits.unwrap_or_default(),
                        group
                            .modular_dc
                            .as_ref()
                            .map(|decoded| decoded.channels.len())
                            .unwrap_or_default(),
                        group.modular_dc_error
                    );
                    println!(
                        "First frame VarDCT AC metadata stream group {}: stream_id={} count={} bits={} channels={} error={:?}",
                        group.payload.group.group,
                        group.payload.ac_metadata_stream_id,
                        group.ac_metadata_count.unwrap_or_default(),
                        group.cursor.ac_metadata_end_bits.unwrap_or_default(),
                        group
                            .ac_metadata
                            .as_ref()
                            .map(|decoded| decoded.channels.len())
                            .unwrap_or_default(),
                        group.ac_metadata_error
                    );
                    if let Some(decoded) = &group.ac_metadata {
                        for channel in &decoded.channels {
                            let min = channel.samples.iter().min().copied().unwrap_or_default();
                            let max = channel.samples.iter().max().copied().unwrap_or_default();
                            let sum = channel
                                .samples
                                .iter()
                                .fold(0i64, |sum, sample| sum + i64::from(*sample));
                            println!(
                                "First frame VarDCT AC metadata channel {}: {}x{} at ({},{}) samples={} min={} max={} sum={}",
                                channel.channel_index,
                                channel.width,
                                channel.height,
                                channel.x0,
                                channel.y0,
                                channel.samples.len(),
                                min,
                                max,
                                sum
                            );
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
