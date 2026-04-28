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
                    "First frame modular residuals: groups={}",
                    residuals.groups.len()
                );
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
                            "First frame modular residual channel {}: {}x{} samples={} min={} max={}",
                            channel.channel_index,
                            channel.width,
                            channel.height,
                            channel.samples.len(),
                            min,
                            max
                        );
                    }
                }
            } else {
                println!("First frame modular residuals: unsupported");
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
    }

    Ok(())
}
