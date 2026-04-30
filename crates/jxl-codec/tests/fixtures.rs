use std::{
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use jxl_codec::{
    BlendMode, ColorSpace, ColorTransform, DecodeConfig, ExtraChannelType, FileFormat,
    FrameEncoding, FrameSectionKind, FrameType, ImageRegion, ModularGroupExecution,
    TransferFunction, TransformId, VarDctSrgb8Image, assemble_vardct_linear_rgb_image,
    assemble_vardct_srgb8_image, assemble_vardct_xyb_image, parse_file, parse_file_with_config,
};

#[test]
fn parses_checked_in_fixture_dimensions() {
    let cases = [
        (
            "reference/libjxl/testdata/jxl/blending/cropped_traffic_light.jxl",
            FileFormat::NakedCodestream,
            50,
            80,
        ),
        (
            "reference/libjxl/testdata/jxl/boxes/square-extended-size-container.jxl",
            FileFormat::Container,
            8,
            8,
        ),
        (
            "reference/libjxl/testdata/jxl/jpeg_reconstruction/1x1_exif_xmp.jxl",
            FileFormat::Container,
            1,
            1,
        ),
        (
            "reference/libjxl/testdata/jxl/pq_gradient.jxl",
            FileFormat::NakedCodestream,
            1088,
            64,
        ),
        (
            "reference/libjxl/testdata/jxl/spline_on_first_frame.jxl",
            FileFormat::NakedCodestream,
            32,
            32,
        ),
        (
            "reference/libjxl/testdata/jxl/splines.jxl",
            FileFormat::NakedCodestream,
            2048,
            2048,
        ),
        (
            "crates/jxl-codec/tests/generated/icc_rec2020_lossless.jxl",
            FileFormat::Container,
            64,
            64,
        ),
    ];

    for (path, expected_format, expected_width, expected_height) in cases {
        let bytes = std::fs::read(workspace_path(path)).unwrap();
        let (extracted, codestream) = parse_file(&bytes).unwrap();

        assert_eq!(extracted.format, expected_format, "{path}");
        assert_eq!(codestream.basic_info.width, expected_width, "{path}");
        assert_eq!(codestream.basic_info.height, expected_height, "{path}");
        assert!(
            codestream.transform_data.is_default(),
            "fixture unexpectedly uses custom transform data: {path}"
        );
        assert!(
            codestream.first_frame.is_some(),
            "fixture first frame was not parsed: {path}"
        );
        assert!(
            codestream.first_frame_data.is_some(),
            "fixture first frame data was not parsed: {path}"
        );
    }
}

#[test]
fn configured_parse_matches_default_serial_parse() {
    let bytes = std::fs::read(workspace_path(
        "crates/jxl-codec/tests/generated/icc_rec2020_lossless.jxl",
    ))
    .unwrap();
    let default = parse_file(&bytes).unwrap();
    let configured = parse_file_with_config(
        &bytes,
        DecodeConfig {
            modular_group_execution: ModularGroupExecution::Serial,
            region: None,
        },
    )
    .unwrap();

    assert_eq!(configured, default);
}

#[test]
fn requested_threads_parse_matches_serial_for_now() {
    let bytes = std::fs::read(workspace_path(
        "reference/libjxl/testdata/jxl/pq_gradient.jxl",
    ))
    .unwrap();
    let serial = parse_file_with_config(
        &bytes,
        DecodeConfig {
            modular_group_execution: ModularGroupExecution::Serial,
            region: None,
        },
    )
    .unwrap();
    let requested_threads = parse_file_with_config(
        &bytes,
        DecodeConfig {
            modular_group_execution: ModularGroupExecution::RequestedThreads(2),
            region: None,
        },
    )
    .unwrap();

    assert_eq!(requested_threads, serial);
}

#[test]
fn rejects_zero_requested_threads() {
    let bytes = std::fs::read(workspace_path(
        "crates/jxl-codec/tests/generated/icc_rec2020_lossless.jxl",
    ))
    .unwrap();
    let err = parse_file_with_config(
        &bytes,
        DecodeConfig {
            modular_group_execution: ModularGroupExecution::RequestedThreads(0),
            region: None,
        },
    )
    .unwrap_err();

    assert_eq!(
        err,
        jxl_codec::Error::Unsupported("zero modular group threads")
    );
}

#[test]
fn region_config_selects_intersecting_modular_groups() {
    let bytes = std::fs::read(workspace_path(
        "reference/libjxl/testdata/jxl/pq_gradient.jxl",
    ))
    .unwrap();
    let (_, codestream) = parse_file_with_config(
        &bytes,
        DecodeConfig {
            modular_group_execution: ModularGroupExecution::Serial,
            region: Some(ImageRegion {
                x: 600,
                y: 0,
                width: 32,
                height: 32,
            }),
        },
    )
    .unwrap();

    let modular = codestream.first_frame_modular.as_ref().unwrap();
    let residuals = modular.residuals.as_ref().unwrap();
    assert_eq!(residuals.groups.len(), 1);
    assert_eq!(residuals.groups[0].stream_id, 22);
    let image = modular.image.as_ref().unwrap();
    assert_eq!(image.width, 32);
    assert_eq!(image.height, 32);
    assert_eq!(image.channels.len(), 1);
    assert_eq!(image.channels[0].width, 32);
    assert_eq!(image.channels[0].height, 32);

    let (_, full_codestream) = parse_file(&bytes).unwrap();
    let full = full_codestream
        .first_frame_modular
        .as_ref()
        .unwrap()
        .image
        .as_ref()
        .unwrap();
    let full_channel = &full.channels[0];
    let mut expected = Vec::with_capacity(32 * 32);
    for y in 0..32usize {
        let start = y * full_channel.width as usize + 600;
        expected.extend_from_slice(&full_channel.samples[start..start + 32]);
    }
    assert_eq!(image.channels[0].samples, expected);
}

#[test]
fn rejects_empty_decode_region() {
    let bytes = std::fs::read(workspace_path(
        "crates/jxl-codec/tests/generated/icc_rec2020_lossless.jxl",
    ))
    .unwrap();
    let err = parse_file_with_config(
        &bytes,
        DecodeConfig {
            modular_group_execution: ModularGroupExecution::Serial,
            region: Some(ImageRegion {
                x: 0,
                y: 0,
                width: 0,
                height: 1,
            }),
        },
    )
    .unwrap_err();

    assert_eq!(
        err,
        jxl_codec::Error::InvalidCodestream("empty decode region")
    );
}

#[test]
fn agrees_with_reference_jxlinfo_when_available() {
    let Some(jxlinfo) = reference_jxlinfo() else {
        eprintln!("skipping reference jxlinfo comparison; tool is not built");
        return;
    };

    let cases = [
        "reference/libjxl/testdata/jxl/blending/cropped_traffic_light.jxl",
        "reference/libjxl/testdata/jxl/boxes/square-extended-size-container.jxl",
        "reference/libjxl/testdata/jxl/jpeg_reconstruction/1x1_exif_xmp.jxl",
        "reference/libjxl/testdata/jxl/pq_gradient.jxl",
        "reference/libjxl/testdata/jxl/spline_on_first_frame.jxl",
        "reference/libjxl/testdata/jxl/splines.jxl",
        "crates/jxl-codec/tests/generated/icc_rec2020_lossless.jxl",
    ];

    for path in cases {
        let path = workspace_path(path);
        let bytes = std::fs::read(&path).unwrap();
        let (_, codestream) = parse_file(&bytes).unwrap();
        let output = Command::new(&jxlinfo).arg(&path).output().unwrap();

        assert!(
            output.status.success(),
            "reference jxlinfo failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        );

        let stdout = String::from_utf8_lossy(&output.stdout);
        let expected = format!(
            "{}x{}",
            codestream.basic_info.width, codestream.basic_info.height
        );
        assert!(
            stdout.contains(&expected),
            "reference jxlinfo output for {} did not contain {expected:?}:\n{stdout}",
            path.display()
        );
    }
}

#[test]
fn assembled_rgb_modular_pixels_match_reference_djxl_when_available() {
    let Some(djxl) = reference_djxl() else {
        eprintln!("skipping reference djxl comparison; tool is not built");
        return;
    };

    let fixture = workspace_path("crates/jxl-codec/tests/generated/icc_rec2020_lossless.jxl");
    let output = unique_temp_path("jxl-rs-reference", "ppm");
    let djxl_output = Command::new(&djxl)
        .arg(&fixture)
        .arg(&output)
        .arg("--quiet")
        .output()
        .unwrap();
    assert!(
        djxl_output.status.success(),
        "reference djxl failed for {}: {}",
        fixture.display(),
        String::from_utf8_lossy(&djxl_output.stderr)
    );

    let reference = std::fs::read(&output).unwrap();
    let _ = std::fs::remove_file(&output);
    let reference = parse_ppm_rgb(&reference);
    assert_eq!(reference.width, 64);
    assert_eq!(reference.height, 64);

    let codestream = parse_fixture("crates/jxl-codec/tests/generated/icc_rec2020_lossless.jxl");
    let modular = codestream.first_frame_modular.as_ref().unwrap();
    let image = modular.image.as_ref().unwrap();
    assert_eq!(image.channels.len(), 3);

    let mut actual = Vec::with_capacity(reference.samples.len());
    for index in 0..(image.width as usize * image.height as usize) {
        for channel in &image.channels {
            actual.push(u16::try_from(channel.samples[index]).unwrap());
        }
    }

    assert_eq!(actual, reference.samples);
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PpmRgb {
    width: u32,
    height: u32,
    maxval: u32,
    samples: Vec<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Srgb8OracleMetrics {
    max_abs_error: u8,
    sum_abs_error: u64,
    checksum: u64,
    anchors: Vec<u8>,
    reference_anchors: Vec<u8>,
}

#[test]
fn generated_progressive_squeeze_pixels_match_reference_djxl_when_available() {
    let (Some(cjxl), Some(djxl)) = (reference_cjxl(), reference_djxl()) else {
        eprintln!("skipping generated squeeze djxl comparison; reference tools are not built");
        return;
    };

    let input = unique_temp_path("jxl-rs-squeeze-source", "ppm");
    let encoded = unique_temp_path("jxl-rs-squeeze", "jxl");
    let reference_output = unique_temp_path("jxl-rs-squeeze-reference", "ppm");
    write_progressive_squeeze_source_ppm(&input);

    let cjxl_output = Command::new(&cjxl)
        .arg(&input)
        .arg(&encoded)
        .args(["-d", "0", "-m", "1", "-p", "--container=0", "--quiet"])
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&input);
    assert!(
        cjxl_output.status.success(),
        "reference cjxl failed: {}",
        String::from_utf8_lossy(&cjxl_output.stderr)
    );

    let djxl_output = Command::new(&djxl)
        .arg(&encoded)
        .arg(&reference_output)
        .arg("--quiet")
        .output()
        .unwrap();
    assert!(
        djxl_output.status.success(),
        "reference djxl failed: {}",
        String::from_utf8_lossy(&djxl_output.stderr)
    );

    let reference = std::fs::read(&reference_output).unwrap();
    let reference = parse_ppm_rgb(&reference);
    let encoded_bytes = std::fs::read(&encoded).unwrap();
    let _ = std::fs::remove_file(&encoded);
    let _ = std::fs::remove_file(&reference_output);
    let (_, codestream) = parse_file(&encoded_bytes).unwrap();
    let modular = codestream.first_frame_modular.as_ref().unwrap();
    assert!(
        modular
            .global
            .group_header
            .transforms
            .iter()
            .any(|transform| {
                transform.id == TransformId::Squeeze && !transform.squeezes.is_empty()
            })
    );
    let image = modular.image.as_ref().unwrap();
    assert_eq!(image.width, reference.width);
    assert_eq!(image.height, reference.height);
    assert_eq!(image.channels.len(), 3);

    let mut actual = Vec::with_capacity(reference.samples.len());
    for index in 0..(image.width as usize * image.height as usize) {
        for channel in &image.channels {
            actual.push(u16::try_from(channel.samples[index]).unwrap());
        }
    }

    assert_eq!(actual, reference.samples);
}

#[test]
fn generated_split_vardct_exposes_global_cursor_when_available() {
    let (Some(cjxl), Some(djxl)) = (reference_cjxl(), reference_djxl()) else {
        eprintln!("skipping generated split VarDCT fixture; reference tools are not built");
        return;
    };

    let input = unique_temp_path("jxl-rs-vardct-source", "ppm");
    let encoded = unique_temp_path("jxl-rs-vardct", "jxl");
    let reference_output = unique_temp_path("jxl-rs-vardct-reference", "ppm");
    write_split_vardct_source_ppm(&input);

    let cjxl_output = Command::new(&cjxl)
        .arg(&input)
        .arg(&encoded)
        .args(["-d", "1.0", "-m", "0", "--container=0", "--quiet"])
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&input);
    assert!(
        cjxl_output.status.success(),
        "reference cjxl failed: {}",
        String::from_utf8_lossy(&cjxl_output.stderr)
    );

    let djxl_output = Command::new(&djxl)
        .arg(&encoded)
        .arg(&reference_output)
        .arg("--quiet")
        .output()
        .unwrap();
    assert!(
        djxl_output.status.success(),
        "reference djxl failed: {}",
        String::from_utf8_lossy(&djxl_output.stderr)
    );
    let reference = std::fs::read(&reference_output).unwrap();
    let reference = parse_ppm_rgb(&reference);
    let _ = std::fs::remove_file(&reference_output);

    let encoded_bytes = std::fs::read(&encoded).unwrap();
    let (_, codestream) = parse_file(&encoded_bytes).unwrap();
    let plan = codestream.first_frame_vardct_plan.as_ref().unwrap();
    compare_reference_vardct_trace(&encoded, plan);
    let _ = std::fs::remove_file(&encoded);

    assert!(
        !plan.frame.is_combined,
        "generated VarDCT fixture unexpectedly used a combined section"
    );
    assert!(plan.global_payload.is_some());
    assert!(plan.ac_global_payload.is_some());
    let ac_global = plan.ac_global_metadata.as_ref().unwrap();
    assert_eq!(ac_global.section.payload_range, 2807..3843);
    assert_eq!(ac_global.all_default_quant_matrices, Some(true));
    assert_eq!(ac_global.quant_matrices_end_bits, Some(1));
    assert_eq!(ac_global.num_histograms, Some(1));
    assert_eq!(ac_global.num_histograms_end_bits, Some(2));
    assert_eq!(ac_global.used_acs, Some(61655));
    assert_eq!(ac_global.bits_consumed, Some(8288));
    assert_eq!(ac_global.parse_error, None);
    assert_eq!(ac_global.passes.len(), 1);
    assert_eq!(ac_global.passes[0].pass, 0);
    assert_eq!(ac_global.passes[0].used_orders, Some(7));
    assert_eq!(ac_global.passes[0].used_orders_end_bits, Some(17));
    assert_eq!(ac_global.passes[0].coeff_order_end_bits, Some(1000));
    assert_eq!(
        ac_global.passes[0]
            .coeff_orders
            .iter()
            .map(|order| (
                order.order,
                order.channel,
                order.skip,
                order.size,
                order.permutation_end,
                order.checksum,
            ))
            .collect::<Vec<_>>(),
        vec![
            (0, 0, 1, 64, 15, 17391013145426411885),
            (0, 1, 1, 64, 27, 328118841242175719),
            (0, 2, 1, 64, 19, 12222102711332199783),
            (1, 0, 1, 64, 12, 18141835571050413691),
            (1, 1, 1, 64, 44, 3939301074858597663),
            (1, 2, 1, 64, 24, 16382350197382754037),
            (2, 0, 4, 256, 13, 3878749180621382175),
            (2, 1, 4, 256, 38, 7348440053961575069),
            (2, 2, 4, 256, 8, 6842156909530809773),
        ]
    );
    assert_eq!(ac_global.passes[0].histogram_contexts, Some(7425));
    assert_eq!(ac_global.passes[0].histogram_count, Some(16));
    assert_eq!(ac_global.passes[0].histogram_end_bits, Some(8288));
    assert_eq!(ac_global.passes[0].use_prefix_code, Some(false));
    assert_eq!(ac_global.passes[0].log_alpha_size, Some(7));
    assert_eq!(ac_global.passes[0].error_bits, None);
    assert_eq!(ac_global.passes[0].error, None);
    assert!(!plan.dc_group_payloads.is_empty());
    assert!(!plan.ac_group_payloads.is_empty());
    assert_eq!(plan.ac_group_payloads.len(), 2);
    assert_eq!(plan.ac_group_payloads[0].section.payload_range, 3843..8335);
    assert_eq!(plan.ac_group_payloads[0].pass, 0);
    assert_eq!(plan.ac_group_payloads[0].group.group, 0);
    assert_eq!(plan.ac_group_payloads[1].section.payload_range, 8335..9359);
    assert_eq!(plan.ac_group_payloads[1].pass, 0);
    assert_eq!(plan.ac_group_payloads[1].group.group, 1);
    assert_eq!(plan.ac_group_metadata.len(), plan.ac_group_payloads.len());
    assert_eq!(
        plan.ac_group_metadata
            .iter()
            .map(|metadata| (
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
                metadata.entropy_uses_prefix_code,
                metadata.parse_error.clone(),
            ))
            .collect::<Vec<_>>(),
        vec![
            (
                0,
                0,
                35936,
                0,
                Some(0),
                Some(0),
                Some(0),
                Some(32),
                Some(32),
                None,
                Some(false),
                Some(jxl_codec::Error::Unsupported(
                    "VarDCT AC coefficient stream decoding"
                )),
            ),
            (
                0,
                1,
                8192,
                0,
                Some(0),
                Some(0),
                Some(0),
                Some(32),
                Some(32),
                None,
                Some(false),
                Some(jxl_codec::Error::Unsupported(
                    "VarDCT AC coefficient stream decoding"
                )),
            ),
        ]
    );
    assert_eq!(
        plan.ac_group_metadata
            .iter()
            .map(|metadata| {
                let probe = metadata.coefficient_probe.as_ref().unwrap();
                format!(
                    "{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{:?}:{:?}:{}",
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
                    probe.coefficient_event_checksum,
                )
            })
            .collect::<Vec<_>>(),
        vec![
            "0:0:0:1:4:2:4:256:2:302:11:32:34:18:33:Some(114):Some(0):1755776318511458984",
            "1:0:0:1:4:2:4:256:2:302:11:32:34:17:23:Some(98):Some(0):2461791933791546484",
        ]
    );
    let channel_trace = plan.ac_group_metadata[0].channel_trace.as_ref().unwrap();
    assert_eq!(channel_trace.channel, 1);
    assert_eq!(channel_trace.blocks_decoded, 477);
    assert_eq!(channel_trace.coefficient_events_decoded, 9983);
    assert_eq!(channel_trace.final_bits, 35936);
    assert_eq!(channel_trace.row_nzeros_checksum, 12510740321947942186);
    assert_eq!(
        channel_trace.coefficient_event_checksum,
        16906932721961906726
    );
    assert_eq!(channel_trace.block_summaries.len(), 8);
    let coefficient_summary = plan.ac_group_metadata[0]
        .coefficient_summary
        .as_ref()
        .unwrap();
    assert_eq!(coefficient_summary.group, 0);
    assert_eq!(coefficient_summary.pass, 0);
    assert_eq!(coefficient_summary.blocks_decoded, 1431);
    assert_eq!(coefficient_summary.final_bits, 35936);
    assert_eq!(
        coefficient_summary.first_block_checksum,
        11040521211606080740
    );
    assert_eq!(
        coefficient_summary
            .per_channel
            .iter()
            .map(|summary| (
                summary.blocks_decoded,
                summary.coefficients_written,
                summary.nonzero_coefficients,
                summary.coefficient_checksum,
            ))
            .collect::<Vec<_>>(),
        vec![
            (477, 3371, 1754, 4634077023953618635),
            (477, 9983, 5649, 1443869010259603971),
            (477, 2866, 1868, 3247447943926418888),
        ]
    );
    let coefficient_grid = plan.ac_group_metadata[0].coefficient_grid.as_ref().unwrap();
    let base_dequantized_grid = plan.ac_group_metadata[0]
        .base_dequantized_grid
        .as_ref()
        .unwrap();
    let dequantized_grid = plan.ac_group_metadata[0].dequantized_grid.as_ref().unwrap();
    let spatial_grid = plan.ac_group_metadata[0].spatial_grid.as_ref().unwrap();
    let spatial_with_dc_grid = plan.ac_group_metadata[0]
        .spatial_with_dc_grid
        .as_ref()
        .unwrap();
    assert_eq!(spatial_with_dc_grid.group, 0);
    assert_eq!(spatial_with_dc_grid.pass, 0);
    assert_eq!(spatial_with_dc_grid.width_blocks, 32);
    assert_eq!(spatial_with_dc_grid.height_blocks, 24);
    assert_eq!(spatial_with_dc_grid.blocks_attempted, 477);
    assert_eq!(spatial_with_dc_grid.blocks_transformed, 477);
    assert_eq!(spatial_with_dc_grid.blocks_skipped, 0);
    assert_eq!(
        spatial_with_dc_grid
            .per_channel
            .iter()
            .map(|channel| (channel.nonzero_samples, channel.sample_checksum))
            .collect::<Vec<_>>(),
        vec![
            (46514, 11353129071865437913),
            (46533, 11703203274754720331),
            (46533, 10920770966532215668),
        ]
    );
    assert_eq!(
        (0..spatial_with_dc_grid.height_blocks)
            .flat_map(|block_y| {
                (0..spatial_with_dc_grid.width_blocks).map(move |block_x| (block_x, block_y))
            })
            .find_map(|(block_x, block_y)| {
                let samples = (0..3)
                    .map(|channel| {
                        (0..64)
                            .filter_map(|sample| {
                                spatial_with_dc_grid
                                    .sample(channel, block_x, block_y, sample)
                                    .filter(|value| *value != 0.0)
                                    .map(|value| (channel, sample, value.to_bits()))
                            })
                            .take(8)
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                samples
                    .iter()
                    .any(|channel| !channel.is_empty())
                    .then_some((block_x, block_y, samples))
            }),
        Some((
            0,
            0,
            vec![
                vec![
                    (0, 0, 3082885028),
                    (0, 1, 3082885028),
                    (0, 2, 3082885028),
                    (0, 3, 3082885028),
                    (0, 4, 3082885028),
                    (0, 5, 3082885028),
                    (0, 6, 3082885028),
                    (0, 7, 3082885028),
                ],
                vec![
                    (1, 0, 979252264),
                    (1, 1, 977235548),
                    (1, 2, 985019156),
                    (1, 3, 990344945),
                    (1, 4, 989189456),
                    (1, 5, 988410608),
                    (1, 6, 990218499),
                    (1, 7, 986072042),
                ],
                vec![
                    (2, 0, 3128142621),
                    (2, 1, 3129302972),
                    (2, 2, 3121153438),
                    (2, 3, 3078314688),
                    (2, 4, 3106294038),
                    (2, 5, 3110967121),
                    (2, 6, 3088806776),
                    (2, 7, 3118586040),
                ],
            ],
        ))
    );
    assert_eq!(spatial_grid.group, 0);
    assert_eq!(spatial_grid.pass, 0);
    assert_eq!(spatial_grid.width_blocks, 32);
    assert_eq!(spatial_grid.height_blocks, 24);
    assert_eq!(spatial_grid.blocks_attempted, 477);
    assert_eq!(spatial_grid.blocks_transformed, 477);
    assert_eq!(spatial_grid.blocks_skipped, 0);
    assert_eq!(
        spatial_grid
            .per_channel
            .iter()
            .map(|channel| (channel.nonzero_samples, channel.sample_checksum))
            .collect::<Vec<_>>(),
        vec![
            (37265, 12240550904136186914),
            (46464, 4537595202176806849),
            (46532, 9484695427167268210),
        ]
    );
    assert_eq!(
        (0..spatial_grid.height_blocks)
            .flat_map(|block_y| {
                (0..spatial_grid.width_blocks).map(move |block_x| (block_x, block_y))
            })
            .find_map(|(block_x, block_y)| {
                let samples = (0..3)
                    .map(|channel| {
                        (0..64)
                            .filter_map(|sample| {
                                spatial_grid
                                    .sample(channel, block_x, block_y, sample)
                                    .filter(|value| *value != 0.0)
                                    .map(|value| (channel, sample, value.to_bits()))
                            })
                            .take(8)
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                samples
                    .iter()
                    .any(|channel| !channel.is_empty())
                    .then_some((block_x, block_y, samples))
            }),
        Some((
            0,
            0,
            vec![
                vec![],
                vec![
                    (1, 0, 3129099130),
                    (1, 1, 3130107486),
                    (1, 2, 3119294945),
                    (1, 3, 963412650),
                    (1, 4, 940505192),
                    (1, 5, 3094172584),
                    (1, 6, 961389490),
                    (1, 7, 3115083405),
                ],
                vec![
                    (2, 0, 3124978992),
                    (2, 1, 3126491532),
                    (2, 2, 3115417450),
                    (2, 3, 959537658),
                    (2, 4, 936801376),
                    (2, 5, 3090284240),
                    (2, 6, 958020294),
                    (2, 7, 3112258790),
                ],
            ],
        ))
    );
    let xyb_image = assemble_vardct_xyb_image(plan).unwrap().unwrap();
    assert_eq!(xyb_image.width, 320);
    assert_eq!(xyb_image.height, 192);
    assert_eq!(xyb_image.groups_assembled, 2);
    assert_eq!(xyb_image.groups_missing, 0);
    let xyb_summary = xyb_image
        .channels
        .iter()
        .map(|channel| {
            channel
                .iter()
                .enumerate()
                .filter(|(_, sample)| **sample != 0.0)
                .fold((0usize, 0u64), |(count, checksum), (index, sample)| {
                    let checksum = checksum
                        .wrapping_mul(1_099_511_628_211)
                        .wrapping_add(index as u64)
                        .rotate_left(11)
                        ^ sample.to_bits() as u64;
                    (count + 1, checksum)
                })
        })
        .collect::<Vec<_>>();
    let xyb_anchors = [
        xyb_image.sample(0, 0, 0).unwrap().to_bits(),
        xyb_image.sample(1, 0, 0).unwrap().to_bits(),
        xyb_image.sample(2, 0, 0).unwrap().to_bits(),
        xyb_image.sample(0, 319, 191).unwrap().to_bits(),
        xyb_image.sample(1, 319, 191).unwrap().to_bits(),
        xyb_image.sample(2, 319, 191).unwrap().to_bits(),
    ];
    assert_eq!(
        xyb_summary,
        vec![
            (59525, 3148783885712997862),
            (59547, 8179999271320941248),
            (59554, 11189150345044084047),
        ]
    );
    assert_eq!(
        xyb_anchors,
        [
            3082885028, 979076178, 3128274685, 940837884, 1037228049, 992849658
        ]
    );
    let rgb_image = assemble_vardct_linear_rgb_image(plan).unwrap().unwrap();
    assert_eq!(rgb_image.width, 320);
    assert_eq!(rgb_image.height, 192);
    let rgb_summary = rgb_image
        .channels
        .iter()
        .map(|channel| {
            channel
                .iter()
                .enumerate()
                .filter(|(_, sample)| **sample != 0.0)
                .fold((0usize, 0u64), |(count, checksum), (index, sample)| {
                    let checksum = checksum
                        .wrapping_mul(1_099_511_628_211)
                        .wrapping_add(index as u64)
                        .rotate_left(11)
                        ^ sample.to_bits() as u64;
                    (count + 1, checksum)
                })
        })
        .collect::<Vec<_>>();
    let rgb_anchors = [
        rgb_image.channels[0][0].to_bits(),
        rgb_image.channels[1][0].to_bits(),
        rgb_image.channels[2][0].to_bits(),
        rgb_image.channels[0][(191 * 320 + 319) as usize].to_bits(),
        rgb_image.channels[1][(191 * 320 + 319) as usize].to_bits(),
        rgb_image.channels[2][(191 * 320 + 319) as usize].to_bits(),
    ];
    assert_eq!(
        rgb_summary,
        vec![
            (61440, 11220725426516025707),
            (61440, 3408914844508450388),
            (61440, 3579285547816920914),
        ]
    );
    assert_eq!(
        rgb_anchors,
        [
            944126269, 952684638, 3107735916, 1015174801, 1015065925, 3159140182
        ]
    );
    let srgb8_image = assemble_vardct_srgb8_image(plan).unwrap().unwrap();
    let anchor_indices = [
        0usize,
        1,
        2,
        ((95 * 320 + 159) * 3) as usize,
        ((95 * 320 + 159) * 3 + 1) as usize,
        ((95 * 320 + 159) * 3 + 2) as usize,
        ((191 * 320 + 319) * 3) as usize,
        ((191 * 320 + 319) * 3 + 1) as usize,
        ((191 * 320 + 319) * 3 + 2) as usize,
    ];
    let metrics = srgb8_oracle_metrics(&srgb8_image, &reference, &anchor_indices);
    assert_eq!(metrics.max_abs_error, 255);
    assert_eq!(metrics.sum_abs_error, 20235071);
    assert_eq!(metrics.checksum, 3789787639564895058);
    assert_eq!(metrics.anchors, vec![0, 0, 0, 17, 14, 0, 34, 34, 0]);
    assert_eq!(
        metrics.reference_anchors,
        vec![0, 1, 1, 125, 128, 124, 253, 255, 255]
    );
    assert_eq!(dequantized_grid.group, 0);
    assert_eq!(dequantized_grid.pass, 0);
    assert_eq!(dequantized_grid.width_blocks, 32);
    assert_eq!(dequantized_grid.height_blocks, 24);
    assert_eq!(
        dequantized_grid
            .per_channel
            .iter()
            .map(|channel| (channel.nonzero_coefficients, channel.coefficient_checksum))
            .collect::<Vec<_>>(),
        vec![
            (1754, 7928736388124340981),
            (5649, 3488116285163030744),
            (6259, 12462676879008589506),
        ]
    );
    assert_eq!(
        (0..3)
            .map(|channel| {
                (0..64)
                    .filter_map(|coeff| {
                        dequantized_grid
                            .coefficient(channel, 0, 0, coeff)
                            .filter(|value| *value != 0.0)
                            .map(|value| (channel, coeff, value.to_bits()))
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
        vec![
            vec![],
            vec![
                (1, 2, 3145630282),
                (1, 4, 3154530179),
                (1, 5, 1002198626),
                (1, 6, 3149682305),
                (1, 7, 1002198657),
                (1, 16, 3154303014),
                (1, 24, 1007832808),
                (1, 32, 3159207169),
                (1, 40, 1011035911),
                (1, 48, 3158519591),
                (1, 56, 1011035943),
            ],
            vec![
                (2, 2, 3141460408),
                (2, 4, 3150542660),
                (2, 5, 999112906),
                (2, 6, 3146596577),
                (2, 7, 999112929),
                (2, 16, 3150201913),
                (2, 24, 1004238428),
                (2, 32, 3155837377),
                (2, 40, 1007838021),
                (2, 48, 3155321693),
                (2, 56, 1007838045),
            ],
        ]
    );
    assert_eq!(base_dequantized_grid.group, 0);
    assert_eq!(base_dequantized_grid.pass, 0);
    assert_eq!(base_dequantized_grid.width_blocks, 32);
    assert_eq!(base_dequantized_grid.height_blocks, 24);
    assert_eq!(base_dequantized_grid.inv_global_scale_bits, 1095575839);
    assert_eq!(
        base_dequantized_grid
            .per_channel
            .iter()
            .map(|channel| (channel.nonzero_coefficients, channel.coefficient_checksum))
            .collect::<Vec<_>>(),
        vec![
            (1754, 7421052046372908028),
            (5649, 16759939757032862422),
            (1868, 1782002988502924811),
        ]
    );
    assert_eq!(
        (0..3)
            .map(|channel| {
                (0..64)
                    .filter_map(|coeff| {
                        base_dequantized_grid
                            .coefficient(channel, 0, 0, coeff)
                            .filter(|value| *value != 0.0)
                            .map(|value| (channel, coeff, value.to_bits()))
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
        vec![
            vec![],
            vec![
                (1, 2, 3228203043),
                (1, 4, 3228203043),
                (1, 5, 1072330787),
                (1, 6, 3219814435),
                (1, 7, 1072330787),
                (1, 16, 3236591651),
                (1, 24, 1085266458),
                (1, 32, 3232750106),
                (1, 40, 1080719395),
                (1, 48, 3228203043),
                (1, 56, 1080719395),
            ],
            vec![],
        ]
    );
    assert_eq!(coefficient_grid.group, 0);
    assert_eq!(coefficient_grid.pass, 0);
    assert_eq!(coefficient_grid.width_blocks, 32);
    assert_eq!(coefficient_grid.height_blocks, 24);
    assert_eq!(
        coefficient_grid
            .per_channel
            .iter()
            .map(|channel| (channel.nonzero_coefficients, channel.coefficient_checksum))
            .collect::<Vec<_>>(),
        vec![
            (1754, 17786621051074088898),
            (5649, 16235058981752676428),
            (1868, 1498562649293644123),
        ]
    );
    assert_eq!(
        (0..3)
            .map(|channel| {
                (0..64)
                    .filter_map(|coeff| {
                        coefficient_grid
                            .coefficient(channel, 0, 0, coeff)
                            .filter(|value| *value != 0)
                            .map(|value| (channel, coeff, value))
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
        vec![
            vec![],
            vec![
                (1, 2, -2),
                (1, 4, -2),
                (1, 5, 1),
                (1, 6, -1),
                (1, 7, 1),
                (1, 16, -4),
                (1, 24, 3),
                (1, 32, -3),
                (1, 40, 2),
                (1, 48, -2),
                (1, 56, 2),
            ],
            vec![],
        ]
    );
    assert!(plan.ac_group_metadata[1].channel_trace.is_some());
    assert!(plan.ac_group_metadata[1].coefficient_summary.is_some());
    assert!(plan.ac_group_metadata[1].coefficient_grid.is_some());
    assert!(plan.ac_group_metadata[1].base_dequantized_grid.is_some());
    assert!(plan.ac_group_metadata[1].dequantized_grid.is_some());
    let group1_spatial = plan.ac_group_metadata[1]
        .spatial_with_dc_grid
        .as_ref()
        .unwrap();
    assert_eq!(group1_spatial.group, 1);
    assert_eq!(group1_spatial.width_blocks, 8);
    assert_eq!(group1_spatial.height_blocks, 24);
    assert_eq!(group1_spatial.blocks_attempted, 120);
    assert_eq!(group1_spatial.blocks_transformed, 120);
    assert_eq!(group1_spatial.blocks_skipped, 0);
    assert_eq!(plan.modular_global_tree_payload_start_bits, Some(192));
    assert_eq!(plan.modular_global_tree_payload_end_bits, Some(1232));
    assert_eq!(plan.modular_global_tree_payload_len_bits, Some(1040));

    assert_eq!(plan.dc_group_payloads.len(), plan.frame.dc_groups.len());
    assert_eq!(plan.dc_group_metadata.len(), plan.dc_group_payloads.len());
    for (dc_group, metadata) in plan.dc_group_payloads.iter().zip(&plan.dc_group_metadata) {
        assert!(dc_group.section.payload_range.start < dc_group.section.payload_range.end);
        assert_eq!(
            dc_group.section.payload_range.len(),
            dc_group.section.section.payload_size as usize
        );
        assert_eq!(dc_group.var_dct_dc_stream_id, 1 + dc_group.group.group);
        assert_eq!(
            dc_group.modular_dc_stream_id,
            1 + plan.frame.dc_groups.len() + dc_group.group.group
        );
        assert_eq!(
            dc_group.ac_metadata_stream_id,
            1 + 2 * plan.frame.dc_groups.len() + dc_group.group.group
        );
        assert_eq!(&metadata.payload, dc_group);
        assert_eq!(metadata.extra_precision_bits, Some(1));
        assert_eq!(metadata.cursor.extra_precision_start_bits, 0);
        assert_eq!(metadata.cursor.extra_precision_end_bits, Some(2));
        assert_eq!(metadata.cursor.var_dct_dc_start_bits, Some(2));
        let header_end_bits = metadata.cursor.var_dct_dc_header_end_bits.unwrap();
        assert_eq!(header_end_bits, 6);
        assert!(header_end_bits <= dc_group.section.payload_range.len() * 8);
        let header = metadata.var_dct_dc_header.as_ref().unwrap();
        assert!(header.use_global_tree);
        assert!(header.weighted_predictor.all_default);
        assert!(header.transforms.is_empty());
        let var_dct_dc = metadata.var_dct_dc.as_ref().unwrap();
        assert_eq!(metadata.parse_error, None);
        assert_eq!(metadata.cursor.var_dct_dc_end_bits, Some(18911));
        assert_eq!(metadata.cursor.modular_dc_start_bits, Some(18911));
        assert_eq!(var_dct_dc.section_physical_index, 1);
        assert_eq!(var_dct_dc.stream_id, dc_group.var_dct_dc_stream_id);
        assert_eq!(var_dct_dc.bits_consumed, 18911);
        assert_eq!(
            var_dct_dc
                .channels
                .iter()
                .map(|channel| (
                    channel.channel_index,
                    channel.x0,
                    channel.y0,
                    channel.width,
                    channel.height,
                    channel.samples.len(),
                    channel.samples.iter().min().copied().unwrap_or_default(),
                    channel.samples.iter().max().copied().unwrap_or_default(),
                    channel
                        .samples
                        .iter()
                        .fold(0i64, |sum, sample| sum + i64::from(*sample)),
                ))
                .collect::<Vec<_>>(),
            vec![
                (0, 0, 0, 40, 24, 960, 20, 1129, 618209),
                (1, 0, 0, 40, 24, 960, -162, 281, 17551),
                (2, 0, 0, 40, 24, 960, -162, 150, -6856),
            ]
        );
        let modular_dc = metadata.modular_dc.as_ref().unwrap();
        assert_eq!(modular_dc.section_physical_index, 1);
        assert_eq!(modular_dc.stream_id, dc_group.modular_dc_stream_id);
        assert!(modular_dc.channels.is_empty());
        assert_eq!(modular_dc.bits_consumed, 18911);
        assert_eq!(metadata.modular_dc_error, None);
        assert_eq!(metadata.cursor.modular_dc_end_bits, Some(18911));
        assert_eq!(metadata.cursor.ac_metadata_start_bits, Some(18911));
        assert_eq!(metadata.cursor.ac_metadata_end_bits, Some(21222));
        assert_eq!(metadata.ac_metadata_count, Some(597));
        assert_eq!(metadata.ac_metadata_error, None);
        let ac_metadata = metadata.ac_metadata.as_ref().unwrap();
        assert_eq!(ac_metadata.section_physical_index, 1);
        assert_eq!(ac_metadata.stream_id, dc_group.ac_metadata_stream_id);
        assert_eq!(ac_metadata.bits_consumed, 21222);
        assert_eq!(
            ac_metadata
                .channels
                .iter()
                .map(|channel| (
                    channel.channel_index,
                    channel.x0,
                    channel.y0,
                    channel.width,
                    channel.height,
                    channel.samples.len(),
                    channel.samples.iter().min().copied().unwrap_or_default(),
                    channel.samples.iter().max().copied().unwrap_or_default(),
                    channel
                        .samples
                        .iter()
                        .fold(0i64, |sum, sample| sum + i64::from(*sample)),
                ))
                .collect::<Vec<_>>(),
            vec![
                (0, 0, 0, 5, 3, 15, -1, 1, -1),
                (1, 0, 0, 5, 3, 15, -37, 0, -245),
                (2, 0, 0, 597, 2, 1194, 0, 15, 5720),
                (3, 0, 0, 40, 24, 960, 0, 7, 6650),
            ]
        );
    }
    let selected_dc = plan.frame.dc_sections_for_region(ImageRegion {
        x: 0,
        y: 0,
        width: 64,
        height: 64,
    });
    assert!(!selected_dc.is_empty());
    assert_eq!(selected_dc[0].group.group, 0);

    let global = plan.global.as_ref().unwrap();
    assert_eq!(global.cursor.dc_dequant_default_end_bits, 1);
    assert_eq!(global.cursor.dc_dequant_end_bits, 1);
    assert_eq!(global.cursor.quantizer_global_scale_end_bits, 15);
    assert_eq!(global.cursor.quantizer_quant_dc_end_bits, 22);
    assert_eq!(global.cursor.quantizer_end_bits, 22);
    assert_eq!(global.cursor.block_context_default_end_bits, 23);
    assert_eq!(global.cursor.block_context_dc_thresholds_end_bits, 35);
    assert_eq!(global.cursor.block_context_qf_thresholds_end_bits, 39);
    assert_eq!(global.cursor.block_context_map_start_bits, Some(39));
    assert_eq!(global.cursor.block_context_map_end_bits, Some(205));
    assert_eq!(global.cursor.block_context_end_bits, 205);
    assert_eq!(global.cursor.color_correlation_default_end_bits, 206);
    assert_eq!(global.cursor.color_correlation_color_factor_end_bits, None);
    assert_eq!(global.cursor.color_correlation_base_x_end_bits, None);
    assert_eq!(global.cursor.color_correlation_base_b_end_bits, None);
    assert_eq!(global.cursor.color_correlation_ytox_dc_end_bits, None);
    assert_eq!(global.cursor.color_correlation_ytob_dc_end_bits, None);
    assert_eq!(global.cursor.color_correlation_end_bits, 206);
    let context_map_probe = global.block_context_map.context_map_probe.as_ref().unwrap();
    assert_eq!(context_map_probe.start_bits, 39);
    assert_eq!(context_map_probe.end_bits, Some(205));
    assert_eq!(context_map_probe.len, 39);
    assert_eq!(
        context_map_probe.kind,
        Some(jxl_codec::VarDctContextMapProbeKind::EntropyCoded)
    );
    assert_eq!(context_map_probe.bits_per_entry, None);
    assert_eq!(context_map_probe.use_mtf, Some(true));
    assert_eq!(context_map_probe.nested_lz77_end_bits, Some(42));
    assert_eq!(context_map_probe.nested_context_map_end_bits, Some(42));
    assert_eq!(context_map_probe.nested_entropy_mode_end_bits, Some(43));
    assert_eq!(context_map_probe.nested_uint_config_end_bits, Some(51));
    assert_eq!(context_map_probe.nested_histogram_end_bits, Some(89));
    assert_eq!(context_map_probe.nested_histogram_count, Some(1));
    assert_eq!(context_map_probe.nested_use_prefix_code, Some(true));
    assert_eq!(context_map_probe.nested_log_alpha_size, Some(15));
    assert_eq!(context_map_probe.ans_start_bits, Some(89));
    assert_eq!(context_map_probe.ans_end_bits, Some(205));
    assert_eq!(context_map_probe.entries_decoded, 39);
    assert_eq!(context_map_probe.max_symbol, Some(14));
    assert_eq!(context_map_probe.num_histograms, Some(15));
    assert_eq!(context_map_probe.final_state_valid, Some(true));
    assert_eq!(context_map_probe.error_stage, None);
    assert_eq!(context_map_probe.error_bits, None);
    assert_eq!(context_map_probe.error, None);
    assert_eq!(plan.modular_global_tree_direct_start_bits, Some(206));
    assert_eq!(
        plan.modular_global_tree_direct_start_absolute_bits,
        Some(398)
    );
    assert_eq!(
        plan.modular_global_tree_direct_start_remaining_bits,
        Some(834)
    );
    assert_eq!(plan.modular_global_tree_direct_tree_end_bits, Some(520));
    assert_eq!(
        plan.modular_global_tree_direct_tree_end_absolute_bits,
        Some(712)
    );
    assert_eq!(
        plan.modular_global_tree_direct_tree_end_remaining_bits,
        Some(520)
    );
    assert_eq!(plan.modular_global_tree_direct_tree_node_count, Some(31));
    assert_eq!(plan.modular_global_tree_direct_tree_leaf_count, Some(16));
    assert_eq!(plan.modular_global_tree_direct_tree_leaves.len(), 16);
    assert_eq!(
        plan.modular_global_tree_direct_tree_leaves
            .iter()
            .map(|leaf| (
                leaf.leaf_index,
                leaf.node_index,
                leaf.residual_context,
                leaf.predictor as u32,
                leaf.predictor_offset,
                leaf.multiplier,
            ))
            .collect::<Vec<_>>(),
        vec![
            (0, 5, 0, 6, 0, 1),
            (1, 6, 1, 6, 0, 1),
            (2, 9, 2, 5, 0, 1),
            (3, 10, 3, 5, 0, 1),
            (4, 15, 4, 0, 0, 1),
            (5, 16, 5, 0, 0, 1),
            (6, 17, 6, 0, 0, 1),
            (7, 18, 7, 0, 0, 1),
            (8, 23, 8, 1, 0, 1),
            (9, 24, 9, 1, 0, 1),
            (10, 25, 10, 1, 0, 1),
            (11, 26, 11, 1, 0, 1),
            (12, 27, 12, 0, 0, 1),
            (13, 28, 13, 0, 0, 1),
            (14, 29, 14, 0, 0, 1),
            (15, 30, 15, 0, 0, 1),
        ]
    );
    assert_eq!(
        plan.modular_global_tree_direct_tree_leaves
            .iter()
            .map(|leaf| leaf.residual_context)
            .collect::<Vec<_>>(),
        (0..16).collect::<Vec<_>>()
    );
    assert_eq!(plan.modular_global_tree_direct_error_bits, None);
    assert_eq!(plan.modular_global_tree_direct_error_absolute_bits, None);
    assert_eq!(plan.modular_global_tree_direct_error_remaining_bits, None);
    assert_eq!(
        plan.modular_global_tree_direct_residual_context_count,
        Some(16)
    );
    assert_eq!(
        plan.modular_global_tree_direct_residual_context_count,
        plan.modular_global_tree_direct_tree_leaf_count
    );
    assert_eq!(
        plan.modular_global_tree_direct_residual_histogram_count,
        Some(4)
    );
    assert_eq!(
        plan.modular_global_tree_direct_residual_context_map_entries,
        vec![0, 0, 0, 1, 2, 2, 2, 2, 0, 1, 3, 0, 3, 3, 3, 3]
    );
    assert_eq!(
        plan.modular_global_tree_direct_residual_context_map_raw_entries,
        vec![0, 0, 0, 1, 2, 2, 2, 2, 0, 1, 3, 0, 3, 3, 3, 3]
    );
    assert_eq!(
        plan.modular_global_tree_direct_residual_context_map_symbol_entries
            .iter()
            .map(|entry| (
                entry.index,
                entry.start_bits,
                entry.token_end_bits,
                entry.end_bits,
                entry.clustered_context,
                entry.token,
                entry.value,
                entry.ans_state_before,
                entry.ans_state_after_symbol,
                entry.ans_state_after,
            ))
            .collect::<Vec<_>>(),
        vec![
            (0, 524, 526, 526, 0, 0, 0, 0, 0, 0),
            (1, 526, 528, 528, 0, 0, 0, 0, 0, 0),
            (2, 528, 530, 530, 0, 0, 0, 0, 0, 0),
            (3, 530, 532, 532, 0, 1, 1, 0, 0, 0),
            (4, 532, 534, 534, 0, 2, 2, 0, 0, 0),
            (5, 534, 536, 536, 0, 2, 2, 0, 0, 0),
            (6, 536, 538, 538, 0, 2, 2, 0, 0, 0),
            (7, 538, 540, 540, 0, 2, 2, 0, 0, 0),
            (8, 540, 542, 542, 0, 0, 0, 0, 0, 0),
            (9, 542, 544, 544, 0, 1, 1, 0, 0, 0),
            (10, 544, 546, 546, 0, 3, 3, 0, 0, 0),
            (11, 546, 548, 548, 0, 0, 0, 0, 0, 0),
            (12, 548, 550, 550, 0, 3, 3, 0, 0, 0),
            (13, 550, 552, 552, 0, 3, 3, 0, 0, 0),
            (14, 552, 554, 554, 0, 3, 3, 0, 0, 0),
            (15, 554, 556, 556, 0, 3, 3, 0, 0, 0),
        ]
    );
    assert_eq!(
        plan.modular_global_tree_direct_residual_context_map_distinct_entries,
        vec![0, 1, 2, 3]
    );
    assert_eq!(
        plan.modular_global_tree_direct_residual_context_map_histogram_usage_counts,
        vec![5, 2, 4, 5]
    );
    assert_eq!(
        plan.modular_global_tree_direct_residual_context_map_max_entry,
        Some(3)
    );
    assert_eq!(
        plan.modular_global_tree_direct_residual_lz77_end_bits,
        Some(521)
    );
    assert_eq!(
        plan.modular_global_tree_direct_residual_context_map_end_bits,
        Some(556)
    );
    assert_eq!(
        plan.modular_global_tree_direct_residual_entropy_mode_end_bits,
        Some(557)
    );
    assert_eq!(
        plan.modular_global_tree_direct_residual_log_alpha_size_end_bits,
        Some(559)
    );
    assert_eq!(
        plan.modular_global_tree_direct_residual_uint_config_end_bits_by_histogram,
        vec![567, 575, 583, 591]
    );
    assert_eq!(
        plan.modular_global_tree_direct_residual_uint_config_end_bits,
        Some(591)
    );
    assert_eq!(
        plan.modular_global_tree_direct_residual_use_prefix_code,
        Some(false)
    );
    assert_eq!(
        plan.modular_global_tree_direct_residual_log_alpha_size,
        Some(6)
    );
    assert_eq!(
        plan.modular_global_tree_direct_residual_failed_histogram_index,
        None
    );
    assert_eq!(plan.modular_global_tree_direct_residual_error_stage, None);
    let histograms = &plan.modular_global_tree_direct_residual_ans_histograms;
    let global_payload_start_bits = plan.modular_global_tree_payload_start_bits.unwrap();
    let global_payload_len_bits = plan.modular_global_tree_payload_len_bits.unwrap();
    assert_eq!(histograms.len(), 4);
    assert_eq!(histograms[0].start_bits, 591);
    assert_eq!(histograms[0].end_bits, Some(874));
    assert_eq!(global_payload_start_bits + histograms[0].start_bits, 783);
    assert_eq!(
        global_payload_start_bits + histograms[0].end_bits.unwrap(),
        1066
    );
    assert_eq!(global_payload_len_bits - histograms[0].start_bits, 449);
    assert_eq!(
        histograms[0].kind,
        Some(jxl_codec::VarDctAnsHistogramProbeKind::Custom)
    );
    assert_eq!(histograms[0].length, Some(57));
    assert_eq!(histograms[0].shift, Some(4));
    assert_eq!(histograms[0].omit_pos, Some(27));
    assert_eq!(histograms[0].log_count_entries.len(), 57);
    assert_eq!(histograms[0].log_count_entries[0].start_bits, 607);
    assert_eq!(histograms[0].log_count_entries[56].end_bits, 831);
    assert!(
        histograms[0]
            .log_count_entries
            .iter()
            .all(|entry| entry.rle_length.is_none())
    );
    assert_eq!(
        histograms[0]
            .log_count_entries
            .iter()
            .map(|entry| entry.logcount)
            .collect::<Vec<_>>(),
        vec![
            6, 5, 5, 6, 6, -1, -1, 7, 6, -1, -1, 7, 7, -1, -1, 6, 7, -1, -1, 7, 7, -1, -1, 7, 8, 1,
            0, 9, 7, 0, -1, 7, 8, 1, -1, 8, 7, -1, -1, 7, 7, -1, -1, 7, 7, -1, -1, 6, 6, -1, -1, 4,
            4, -1, -1, -1, 0,
        ]
    );
    assert_eq!(histograms[0].log_count_error_index, None);
    assert_eq!(histograms[0].population_entries.len(), 57);
    assert_eq!(histograms[0].population_entries[0].start_bits, 831);
    assert_eq!(histograms[0].population_entries[56].end_bits, 874);
    assert_eq!(
        histograms[0]
            .population_entries
            .iter()
            .filter(|entry| entry.bitcount > 0)
            .map(|entry| (
                entry.index,
                entry.start_bits,
                entry.end_bits,
                entry.bitcount,
                entry.extra_bits,
                entry.count,
            ))
            .collect::<Vec<_>>(),
        vec![
            (0, 831, 832, 1, Some(0), 64),
            (1, 832, 833, 1, Some(1), 48),
            (2, 833, 834, 1, Some(1), 48),
            (3, 834, 835, 1, Some(1), 96),
            (4, 835, 836, 1, Some(1), 96),
            (7, 836, 838, 2, Some(0), 128),
            (8, 838, 839, 1, Some(1), 96),
            (11, 839, 841, 2, Some(0), 128),
            (12, 841, 843, 2, Some(0), 128),
            (15, 843, 844, 1, Some(1), 96),
            (16, 844, 846, 2, Some(3), 224),
            (19, 846, 848, 2, Some(3), 224),
            (20, 848, 850, 2, Some(1), 160),
            (23, 850, 852, 2, Some(1), 160),
            (24, 852, 854, 2, Some(0), 256),
            (28, 854, 856, 2, Some(2), 192),
            (31, 856, 858, 2, Some(3), 224),
            (32, 858, 860, 2, Some(0), 256),
            (35, 860, 862, 2, Some(0), 256),
            (36, 862, 864, 2, Some(1), 160),
            (39, 864, 866, 2, Some(1), 160),
            (40, 866, 868, 2, Some(1), 160),
            (43, 868, 870, 2, Some(1), 160),
            (44, 870, 872, 2, Some(0), 128),
            (47, 872, 873, 1, Some(0), 64),
            (48, 873, 874, 1, Some(0), 64),
        ]
    );
    assert_eq!(histograms[0].population_error_index, None);
    assert_eq!(histograms[0].total_count_before_omit, Some(3815));
    assert_eq!(histograms[0].omit_count, Some(281));
    assert_eq!(
        histograms[0].final_counts.as_deref(),
        Some(
            &[
                64, 48, 48, 96, 96, 0, 0, 128, 96, 0, 0, 128, 128, 0, 0, 96, 224, 0, 0, 224, 160,
                0, 0, 160, 256, 2, 1, 281, 192, 1, 0, 224, 256, 2, 0, 256, 160, 0, 0, 160, 160, 0,
                0, 160, 128, 0, 0, 64, 64, 0, 0, 16, 16, 0, 0, 0, 1,
            ][..]
        )
    );
    let alias = histograms[0].alias_table.as_ref().unwrap();
    assert_eq!(alias.table_size, 64);
    assert_eq!(alias.entry_size, 64);
    assert_eq!(alias.distribution_len, 57);
    assert_eq!(alias.nonzero_symbols, 34);
    assert_eq!(alias.count_sum, 4096);
    assert_eq!(alias.first_nonzero_symbol, Some(0));
    assert_eq!(alias.last_nonzero_symbol, Some(56));
    assert_eq!(alias.table_checksum, 14675649862238370290);
    assert_eq!(histograms[0].error_stage, None);
    assert_eq!(histograms[1].start_bits, 874);
    assert_eq!(histograms[1].end_bits, Some(914));
    assert_eq!(global_payload_start_bits + histograms[1].start_bits, 1066);
    assert_eq!(
        global_payload_start_bits + histograms[1].end_bits.unwrap(),
        1106
    );
    assert_eq!(global_payload_len_bits - histograms[1].start_bits, 166);
    assert_eq!(
        histograms[1].kind,
        Some(jxl_codec::VarDctAnsHistogramProbeKind::Custom)
    );
    assert_eq!(histograms[1].length, Some(7));
    assert_eq!(histograms[1].shift, Some(0));
    assert_eq!(histograms[1].omit_pos, Some(0));
    assert_eq!(
        histograms[1]
            .log_count_entries
            .iter()
            .map(|entry| entry.logcount)
            .collect::<Vec<_>>(),
        vec![10, 10, 10, 7, 7, 4, 5]
    );
    assert_eq!(
        histograms[1]
            .log_count_entries
            .iter()
            .map(|entry| (entry.start_bits, entry.end_bits))
            .collect::<Vec<_>>(),
        vec![
            (883, 889),
            (889, 895),
            (895, 901),
            (901, 904),
            (904, 907),
            (907, 911),
            (911, 914),
        ]
    );
    assert_eq!(histograms[1].log_count_error_index, None);
    assert_eq!(histograms[1].population_entries.len(), 7);
    assert!(histograms[1].population_entries.iter().all(|entry| {
        entry.start_bits == 914 && entry.end_bits == 914 && entry.bitcount == 0 && !entry.copied
    }));
    assert_eq!(histograms[1].population_error_index, None);
    assert_eq!(histograms[1].total_count_before_omit, Some(2352));
    assert_eq!(histograms[1].omit_count, Some(1744));
    assert_eq!(
        histograms[1].final_counts.as_deref(),
        Some(&[1744, 1024, 1024, 128, 128, 16, 32][..])
    );
    let alias = histograms[1].alias_table.as_ref().unwrap();
    assert_eq!(alias.table_size, 64);
    assert_eq!(alias.entry_size, 64);
    assert_eq!(alias.distribution_len, 7);
    assert_eq!(alias.nonzero_symbols, 7);
    assert_eq!(alias.count_sum, 4096);
    assert_eq!(alias.first_nonzero_symbol, Some(0));
    assert_eq!(alias.last_nonzero_symbol, Some(6));
    assert_eq!(alias.table_checksum, 3386431582421457645);
    assert_eq!(histograms[1].error_stage, None);
    assert_eq!(histograms[2].start_bits, 914);
    assert_eq!(histograms[2].end_bits, Some(936));
    assert_eq!(global_payload_start_bits + histograms[2].start_bits, 1106);
    assert_eq!(
        global_payload_start_bits + histograms[2].end_bits.unwrap(),
        1128
    );
    assert_eq!(global_payload_len_bits - histograms[2].start_bits, 126);
    assert_eq!(
        histograms[2].kind,
        Some(jxl_codec::VarDctAnsHistogramProbeKind::Simple)
    );
    assert_eq!(histograms[2].simple_symbol_count, Some(2));
    assert_eq!(
        histograms[2].final_counts.as_deref(),
        Some(&[43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4053][..])
    );
    let alias = histograms[2].alias_table.as_ref().unwrap();
    assert_eq!(alias.table_size, 64);
    assert_eq!(alias.entry_size, 64);
    assert_eq!(alias.distribution_len, 15);
    assert_eq!(alias.nonzero_symbols, 2);
    assert_eq!(alias.count_sum, 4096);
    assert_eq!(alias.first_nonzero_symbol, Some(0));
    assert_eq!(alias.last_nonzero_symbol, Some(14));
    assert_eq!(alias.table_checksum, 5755618891534445105);
    assert_eq!(histograms[2].error_stage, None);
    assert_eq!(histograms[3].start_bits, 936);
    assert_eq!(histograms[3].end_bits, Some(1039));
    assert_eq!(global_payload_start_bits + histograms[3].start_bits, 1128);
    assert_eq!(global_payload_len_bits - histograms[3].start_bits, 104);
    assert_eq!(
        histograms[3].kind,
        Some(jxl_codec::VarDctAnsHistogramProbeKind::Custom)
    );
    assert_eq!(histograms[3].length, Some(23));
    assert_eq!(histograms[3].shift, Some(2));
    assert_eq!(histograms[3].omit_pos, Some(0));
    assert_eq!(
        histograms[3]
            .log_count_entries
            .iter()
            .map(|entry| entry.index)
            .collect::<Vec<_>>(),
        vec![
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22
        ]
    );
    assert_eq!(
        histograms[3]
            .log_count_entries
            .iter()
            .map(|entry| entry.logcount)
            .collect::<Vec<_>>(),
        vec![
            9, -1, 5, -1, 8, -1, -1, -1, 9, -1, -1, -1, 4, -1, 4, -1, 12, 5, -1, 6,
        ]
    );
    assert_eq!(
        histograms[3]
            .log_count_entries
            .iter()
            .map(|entry| (entry.start_bits, entry.end_bits))
            .collect::<Vec<_>>(),
        vec![
            (949, 952),
            (952, 957),
            (957, 960),
            (960, 965),
            (965, 968),
            (968, 973),
            (973, 978),
            (978, 983),
            (983, 986),
            (986, 991),
            (991, 996),
            (996, 1001),
            (1001, 1005),
            (1005, 1010),
            (1010, 1014),
            (1014, 1019),
            (1019, 1026),
            (1027, 1030),
            (1030, 1035),
            (1035, 1038),
        ]
    );
    assert_eq!(histograms[3].log_count_entries[16].rle_length, Some(0));
    assert_eq!(histograms[3].log_count_entries[16].rle_end_bits, Some(1027));
    assert_eq!(histograms[3].log_count_entries[16].next_index, 20);
    assert_eq!(histograms[3].log_count_error_index, None);
    assert_eq!(histograms[3].population_entries.len(), 23);
    assert_eq!(histograms[3].population_error_index, None);
    assert_eq!(
        histograms[3]
            .population_entries
            .iter()
            .filter(|entry| entry.count != 0 || entry.omitted || entry.copied)
            .map(|entry| (
                entry.index,
                entry.start_bits,
                entry.end_bits,
                entry.bitcount,
                entry.extra_bits,
                entry.count,
                entry.copied,
                entry.omitted,
            ))
            .collect::<Vec<_>>(),
        vec![
            (0, 1038, 1038, 0, None, 2912, false, true),
            (2, 1038, 1038, 0, Some(0), 32, false, false),
            (4, 1038, 1038, 0, Some(0), 256, false, false),
            (8, 1038, 1039, 1, Some(1), 768, false, false),
            (12, 1039, 1039, 0, Some(0), 16, false, false),
            (14, 1039, 1039, 0, Some(0), 16, false, false),
            (16, 1039, 1039, 0, None, 0, true, false),
            (17, 1039, 1039, 0, None, 0, true, false),
            (18, 1039, 1039, 0, None, 0, true, false),
            (19, 1039, 1039, 0, None, 0, true, false),
            (20, 1039, 1039, 0, Some(0), 32, false, false),
            (22, 1039, 1039, 0, Some(0), 64, false, false),
        ]
    );
    assert_eq!(
        histograms[3].final_counts.as_deref(),
        Some(
            &[
                2912, 0, 32, 0, 256, 0, 0, 0, 768, 0, 0, 0, 16, 0, 16, 0, 0, 0, 0, 0, 32, 0, 64,
            ][..]
        )
    );
    assert_eq!(histograms[3].total_count_before_omit, Some(1184));
    assert_eq!(histograms[3].omit_count, Some(2912));
    let alias = histograms[3].alias_table.as_ref().unwrap();
    assert_eq!(alias.table_size, 64);
    assert_eq!(alias.entry_size, 64);
    assert_eq!(alias.distribution_len, 23);
    assert_eq!(alias.nonzero_symbols, 8);
    assert_eq!(alias.count_sum, 4096);
    assert_eq!(alias.first_nonzero_symbol, Some(0));
    assert_eq!(alias.last_nonzero_symbol, Some(22));
    assert_eq!(alias.table_checksum, 5351005287173771891);
    assert_eq!(histograms[3].error_stage, None);
    assert_eq!(histograms[3].error_bits, None);
    assert_eq!(histograms[3].error, None);
    assert_eq!(plan.modular_global_tree_start_bits, Some(206));
    assert_eq!(plan.modular_global_tree_start_absolute_bits, Some(398));
    assert_eq!(plan.modular_global_tree_start_remaining_bits, Some(834));
    assert_eq!(plan.modular_global_tree_direct_error, None);
    assert_eq!(plan.modular_global_tree_error, None);
    assert_vardct_global_cursor_in_payload(global, global.section.section.payload_size);
}

#[test]
fn generated_vardct_intensity_target_scales_opsin_plan_when_available() {
    let (Some(cjxl), Some(djxl)) = (reference_cjxl(), reference_djxl()) else {
        eprintln!("skipping generated VarDCT intensity fixture; reference tools are not built");
        return;
    };

    let input = unique_temp_path("jxl-rs-vardct-intensity-source", "ppm");
    let encoded = unique_temp_path("jxl-rs-vardct-intensity", "jxl");
    let reference_output = unique_temp_path("jxl-rs-vardct-intensity-reference", "ppm");
    write_split_vardct_source_ppm(&input);

    let cjxl_output = Command::new(&cjxl)
        .arg(&input)
        .arg(&encoded)
        .args([
            "-d",
            "1.0",
            "-m",
            "0",
            "--container=0",
            "--intensity_target=510",
            "--quiet",
        ])
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&input);
    assert!(
        cjxl_output.status.success(),
        "reference cjxl failed: {}",
        String::from_utf8_lossy(&cjxl_output.stderr)
    );

    let djxl_output = Command::new(&djxl)
        .arg(&encoded)
        .arg(&reference_output)
        .arg("--quiet")
        .output()
        .unwrap();
    assert!(
        djxl_output.status.success(),
        "reference djxl failed: {}",
        String::from_utf8_lossy(&djxl_output.stderr)
    );
    let reference = std::fs::read(&reference_output).unwrap();
    let reference = parse_ppm_rgb(&reference);
    let _ = std::fs::remove_file(&reference_output);

    let encoded_bytes = std::fs::read(&encoded).unwrap();
    let _ = std::fs::remove_file(&encoded);
    let (_, codestream) = parse_file(&encoded_bytes).unwrap();
    assert!((codestream.metadata.tone_mapping.intensity_target - 510.0).abs() < 1.0e-3);
    assert!(codestream.transform_data.opsin_inverse_matrix.is_none());
    let plan = codestream.first_frame_vardct_plan.as_ref().unwrap();
    assert!((plan.opsin_params.inverse_matrix[0][0] - 5.5157833).abs() < 1.0e-6);
    assert!((plan.opsin_params.inverse_matrix[1][1] - 2.2093852).abs() < 1.0e-6);
    assert!((plan.opsin_params.inverse_matrix[2][2] - 0.9729641).abs() < 1.0e-6);
    assert!((plan.opsin_params.opsin_biases[0] + 0.0037930732).abs() < 1.0e-9);
    assert!((plan.opsin_params.opsin_biases_cbrt[0] + 0.15595423).abs() < 1.0e-7);

    let srgb8_image = assemble_vardct_srgb8_image(plan).unwrap().unwrap();
    let anchor_indices = [
        0usize,
        1,
        2,
        ((95 * 320 + 159) * 3) as usize,
        ((95 * 320 + 159) * 3 + 1) as usize,
        ((95 * 320 + 159) * 3 + 2) as usize,
        ((191 * 320 + 319) * 3) as usize,
        ((191 * 320 + 319) * 3 + 1) as usize,
        ((191 * 320 + 319) * 3 + 2) as usize,
    ];
    let metrics = srgb8_oracle_metrics(&srgb8_image, &reference, &anchor_indices);
    assert_eq!(metrics.max_abs_error, 255);
    assert_eq!(metrics.sum_abs_error, 20670678);
    assert_eq!(metrics.checksum, 13324128505421059030);
    assert_eq!(metrics.anchors, vec![0, 0, 0, 14, 10, 0, 29, 28, 0]);
    assert_eq!(
        metrics.reference_anchors,
        vec![0, 0, 0, 127, 126, 124, 253, 255, 255]
    );
}

#[test]
fn generated_vardct_no_gaborish_skips_filter_when_available() {
    let (Some(cjxl), Some(djxl)) = (reference_cjxl(), reference_djxl()) else {
        eprintln!("skipping generated no-Gaborish VarDCT fixture; reference tools are not built");
        return;
    };

    let input = unique_temp_path("jxl-rs-vardct-no-gaborish-source", "ppm");
    let encoded = unique_temp_path("jxl-rs-vardct-no-gaborish", "jxl");
    let reference_output = unique_temp_path("jxl-rs-vardct-no-gaborish-reference", "ppm");
    write_split_vardct_source_ppm(&input);

    let cjxl_output = Command::new(&cjxl)
        .arg(&input)
        .arg(&encoded)
        .args([
            "-d",
            "1.0",
            "-m",
            "0",
            "--container=0",
            "--gaborish=0",
            "--quiet",
        ])
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&input);
    assert!(
        cjxl_output.status.success(),
        "reference cjxl failed: {}",
        String::from_utf8_lossy(&cjxl_output.stderr)
    );

    let djxl_output = Command::new(&djxl)
        .arg(&encoded)
        .arg(&reference_output)
        .arg("--quiet")
        .output()
        .unwrap();
    assert!(
        djxl_output.status.success(),
        "reference djxl failed: {}",
        String::from_utf8_lossy(&djxl_output.stderr)
    );
    let reference = std::fs::read(&reference_output).unwrap();
    let reference = parse_ppm_rgb(&reference);
    let _ = std::fs::remove_file(&reference_output);

    let encoded_bytes = std::fs::read(&encoded).unwrap();
    let _ = std::fs::remove_file(&encoded);
    let (_, codestream) = parse_file(&encoded_bytes).unwrap();
    let plan = codestream.first_frame_vardct_plan.as_ref().unwrap();
    assert!(!plan.loop_filter.gab);
    assert_eq!(plan.loop_filter.gab_weights, None);

    let xyb_image = assemble_vardct_xyb_image(plan).unwrap().unwrap();
    assert_eq!(xyb_image.width, 320);
    assert_eq!(xyb_image.height, 192);
    assert_eq!(xyb_image.groups_assembled, 2);
    assert_eq!(xyb_image.groups_missing, 0);
    let xyb_summary = xyb_image
        .channels
        .iter()
        .map(|channel| {
            channel
                .iter()
                .enumerate()
                .filter(|(_, sample)| **sample != 0.0)
                .fold((0usize, 0u64), |(count, checksum), (index, sample)| {
                    let checksum = checksum
                        .wrapping_mul(1_099_511_628_211)
                        .wrapping_add(index as u64)
                        .rotate_left(11)
                        ^ sample.to_bits() as u64;
                    (count + 1, checksum)
                })
        })
        .collect::<Vec<_>>();
    let xyb_anchors = [
        xyb_image.sample(0, 0, 0).unwrap().to_bits(),
        xyb_image.sample(1, 0, 0).unwrap().to_bits(),
        xyb_image.sample(2, 0, 0).unwrap().to_bits(),
        xyb_image.sample(0, 319, 191).unwrap().to_bits(),
        xyb_image.sample(1, 319, 191).unwrap().to_bits(),
        xyb_image.sample(2, 319, 191).unwrap().to_bits(),
    ];

    let rgb_image = assemble_vardct_linear_rgb_image(plan).unwrap().unwrap();
    let rgb_summary = rgb_image
        .channels
        .iter()
        .map(|channel| {
            channel
                .iter()
                .enumerate()
                .filter(|(_, sample)| **sample != 0.0)
                .fold((0usize, 0u64), |(count, checksum), (index, sample)| {
                    let checksum = checksum
                        .wrapping_mul(1_099_511_628_211)
                        .wrapping_add(index as u64)
                        .rotate_left(11)
                        ^ sample.to_bits() as u64;
                    (count + 1, checksum)
                })
        })
        .collect::<Vec<_>>();
    let rgb_anchors = [
        rgb_image.channels[0][0].to_bits(),
        rgb_image.channels[1][0].to_bits(),
        rgb_image.channels[2][0].to_bits(),
        rgb_image.channels[0][(191 * 320 + 319) as usize].to_bits(),
        rgb_image.channels[1][(191 * 320 + 319) as usize].to_bits(),
        rgb_image.channels[2][(191 * 320 + 319) as usize].to_bits(),
    ];

    let srgb8_image = assemble_vardct_srgb8_image(plan).unwrap().unwrap();
    let anchor_indices = [
        0usize,
        1,
        2,
        ((95 * 320 + 159) * 3) as usize,
        ((95 * 320 + 159) * 3 + 1) as usize,
        ((95 * 320 + 159) * 3 + 2) as usize,
        ((191 * 320 + 319) * 3) as usize,
        ((191 * 320 + 319) * 3 + 1) as usize,
        ((191 * 320 + 319) * 3 + 2) as usize,
    ];
    let metrics = srgb8_oracle_metrics(&srgb8_image, &reference, &anchor_indices);
    assert_eq!(
        xyb_summary,
        vec![
            (54781, 2020519470673849507),
            (54803, 10888162461340476259),
            (54955, 13716982194859290094),
        ]
    );
    assert_eq!(
        xyb_anchors,
        [
            3082885028, 977405121, 3128468381, 952178596, 1036992499, 984350354
        ]
    );
    assert_eq!(
        rgb_summary,
        vec![
            (61440, 3171998921660256096),
            (61440, 11927304132132683606),
            (61440, 16709797738514973326),
        ]
    );
    assert_eq!(
        rgb_anchors,
        [
            941880548, 951553088, 3107379552, 1015089770, 1014597649, 3159069755
        ]
    );
    assert_eq!(metrics.max_abs_error, 255);
    assert_eq!(metrics.sum_abs_error, 20512148);
    assert_eq!(metrics.checksum, 18332893120240527486);
    assert_eq!(metrics.anchors, vec![0, 0, 0, 18, 17, 0, 34, 33, 0]);
    assert_eq!(
        metrics.reference_anchors,
        vec![0, 0, 0, 125, 127, 127, 255, 255, 254]
    );
}

fn parse_ppm_rgb(bytes: &[u8]) -> PpmRgb {
    let (magic, offset) = netpbm_token(bytes, 0);
    assert_eq!(magic, b"P6");
    let (width, offset) = netpbm_token(bytes, offset);
    let (height, offset) = netpbm_token(bytes, offset);
    let (maxval, mut offset) = netpbm_token(bytes, offset);
    let maxval = parse_ascii_u32(maxval);
    assert!(matches!(maxval, 255 | 65535));
    assert!(
        offset < bytes.len() && bytes[offset].is_ascii_whitespace(),
        "PPM header was not followed by binary sample data"
    );
    offset += 1;

    let width = parse_ascii_u32(width);
    let height = parse_ascii_u32(height);
    let bytes_per_sample = if maxval > 255 { 2 } else { 1 };
    let expected_bytes = width as usize * height as usize * 3 * bytes_per_sample;
    let data = &bytes[offset..];
    assert_eq!(data.len(), expected_bytes);
    let samples = if bytes_per_sample == 2 {
        data.chunks_exact(2)
            .map(|chunk| u16::from_be_bytes([chunk[0], chunk[1]]))
            .collect()
    } else {
        data.iter().copied().map(u16::from).collect()
    };
    PpmRgb {
        width,
        height,
        maxval,
        samples,
    }
}

fn srgb8_oracle_metrics(
    image: &VarDctSrgb8Image,
    reference: &PpmRgb,
    anchor_indices: &[usize],
) -> Srgb8OracleMetrics {
    assert_eq!(image.width, reference.width);
    assert_eq!(image.height, reference.height);
    assert_eq!(image.pixels.len(), reference.samples.len());

    let mut max_abs_error = 0u8;
    let mut sum_abs_error = 0u64;
    let mut checksum = 0u64;
    for (index, (&actual, &expected)) in image
        .pixels
        .iter()
        .zip(reference.samples.iter())
        .enumerate()
    {
        let expected = ppm_sample_to_srgb8(expected, reference.maxval);
        let error = actual.abs_diff(expected);
        max_abs_error = max_abs_error.max(error);
        sum_abs_error += u64::from(error);
        checksum = checksum
            .wrapping_mul(1_099_511_628_211)
            .wrapping_add(index as u64)
            .rotate_left(11)
            ^ u64::from(actual);
    }

    Srgb8OracleMetrics {
        max_abs_error,
        sum_abs_error,
        checksum,
        anchors: anchor_indices
            .iter()
            .map(|&index| image.pixels[index])
            .collect(),
        reference_anchors: anchor_indices
            .iter()
            .map(|&index| ppm_sample_to_srgb8(reference.samples[index], reference.maxval))
            .collect(),
    }
}

fn ppm_sample_to_srgb8(sample: u16, maxval: u32) -> u8 {
    match maxval {
        255 => u8::try_from(sample).unwrap(),
        65535 => ((u32::from(sample) * 255 + 32767) / 65535) as u8,
        _ => unreachable!("unsupported PPM maxval"),
    }
}

fn netpbm_token(bytes: &[u8], mut offset: usize) -> (&[u8], usize) {
    loop {
        while offset < bytes.len() && bytes[offset].is_ascii_whitespace() {
            offset += 1;
        }
        if offset < bytes.len() && bytes[offset] == b'#' {
            while offset < bytes.len() && bytes[offset] != b'\n' {
                offset += 1;
            }
            continue;
        }
        break;
    }
    let start = offset;
    while offset < bytes.len() && !bytes[offset].is_ascii_whitespace() {
        offset += 1;
    }
    (&bytes[start..offset], offset)
}

fn parse_ascii_u32(bytes: &[u8]) -> u32 {
    std::str::from_utf8(bytes).unwrap().parse().unwrap()
}

fn assert_vardct_global_cursor_in_payload(
    global: &jxl_codec::VarDctGlobalMetadata,
    payload_size: u32,
) {
    let cursor = global.cursor;
    let payload_bits = payload_size as usize * 8;
    assert!(cursor.dc_dequant_end_bits > 0);
    assert!(cursor.quantizer_end_bits > cursor.dc_dequant_end_bits);
    assert!(cursor.block_context_end_bits >= cursor.quantizer_end_bits);
    assert!(cursor.color_correlation_end_bits >= cursor.block_context_end_bits);
    assert!(cursor.color_correlation_end_bits <= payload_bits);
    assert_eq!(cursor.color_correlation_end_bits, global.bits_consumed);
}

fn write_progressive_squeeze_source_ppm(path: &Path) {
    let width = 128u32;
    let height = 128u32;
    let mut state = 2u32;
    let mut bytes = format!("P6\n{width} {height}\n255\n").into_bytes();
    for _ in 0..width * height * 3 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        bytes.push((state >> 24) as u8);
    }
    std::fs::write(path, bytes).unwrap();
}

fn write_split_vardct_source_ppm(path: &Path) {
    let width = 320u32;
    let height = 192u32;
    let mut bytes = format!("P6\n{width} {height}\n255\n").into_bytes();
    for y in 0..height {
        for x in 0..width {
            let checker = (((x / 16) ^ (y / 16)) & 1) * 48;
            bytes.push(((x * 255 / (width - 1)) ^ checker) as u8);
            bytes.push(((y * 255 / (height - 1)) ^ checker) as u8);
            bytes.push((((x + y) * 255 / (width + height - 2)) ^ checker) as u8);
        }
    }
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn parses_non_default_fixture_metadata() {
    let pq = parse_fixture("reference/libjxl/testdata/jxl/pq_gradient.jxl");
    assert_eq!(pq.basic_info.bits_per_sample, 16);
    assert_eq!(pq.metadata.color_encoding.color_space, ColorSpace::Gray);
    assert_eq!(
        pq.metadata.color_encoding.transfer_function,
        TransferFunction::Pq
    );
    assert_eq!(pq.metadata.tone_mapping.intensity_target, 10_000.0);
    assert!(!pq.metadata.xyb_encoded);

    let animation =
        parse_fixture("reference/libjxl/testdata/jxl/blending/cropped_traffic_light.jxl");
    assert!(animation.basic_info.have_animation);
    assert_eq!(animation.basic_info.num_extra_channels, 1);
    assert_eq!(animation.basic_info.alpha_bits, 8);
    assert_eq!(
        animation.metadata.extra_channels[0].channel_type,
        ExtraChannelType::Alpha
    );
    assert_eq!(animation.metadata.animation.unwrap().tps_numerator, 100);
}

#[test]
fn parses_checked_in_fixture_first_frame_headers() {
    let splines = parse_fixture("reference/libjxl/testdata/jxl/splines.jxl");
    let frame = splines.first_frame.as_ref().unwrap();
    assert_eq!(frame.encoding, FrameEncoding::Modular);
    assert_eq!(frame.frame_type, FrameType::Regular);
    assert_eq!(frame.frame_size.width, 2048);
    assert_eq!(frame.frame_size.height, 2048);
    assert_eq!(frame.group_layout.group_dim, 1024);
    assert_eq!(frame.group_layout.num_groups, 4);
    assert_eq!(frame.blending_info.mode, BlendMode::Replace);

    let pq = parse_fixture("reference/libjxl/testdata/jxl/pq_gradient.jxl");
    let frame = pq.first_frame.as_ref().unwrap();
    assert_eq!(frame.encoding, FrameEncoding::Modular);
    assert_eq!(frame.color_transform, ColorTransform::None);
    assert_eq!(frame.frame_size.width, 1088);
    assert_eq!(frame.frame_size.height, 64);
    assert_eq!(frame.group_layout.groups_x, 3);
    assert_eq!(frame.group_layout.groups_y, 1);

    let animation =
        parse_fixture("reference/libjxl/testdata/jxl/blending/cropped_traffic_light.jxl");
    let frame = animation.first_frame.as_ref().unwrap();
    assert_eq!(frame.encoding, FrameEncoding::Modular);
    assert_eq!(frame.frame_size.width, 60);
    assert_eq!(frame.frame_size.height, 105);
    assert_eq!(frame.extra_channel_upsampling, vec![1]);
    assert_eq!(frame.extra_channel_blending_info.len(), 1);
    assert_eq!(frame.animation_frame.duration, 300);

    let container =
        parse_fixture("reference/libjxl/testdata/jxl/boxes/square-extended-size-container.jxl");
    let frame = container.first_frame.as_ref().unwrap();
    assert_eq!(frame.encoding, FrameEncoding::VarDct);
    assert_eq!(frame.color_transform, ColorTransform::Xyb);
    assert_eq!(frame.frame_size.width, 8);
    assert_eq!(frame.frame_size.height, 8);
    assert_eq!(frame.group_layout.num_groups, 1);
    assert!(frame.loop_filter.gab);
}

#[test]
fn parses_checked_in_fixture_first_frame_toc() {
    let splines = parse_fixture("reference/libjxl/testdata/jxl/splines.jxl");
    let frame_data = splines.first_frame_data.as_ref().unwrap();
    assert_eq!(frame_data.toc.entries.len(), 7);
    assert!(!frame_data.toc.has_permutation);
    assert_eq!(frame_data.payload_size, 60);
    assert_eq!(frame_data.sections[0].kind, FrameSectionKind::DcGlobal);
    assert_eq!(frame_data.sections[0].size, 56);
    assert_eq!(
        frame_data.sections[3].kind,
        FrameSectionKind::AcGroup { pass: 0, group: 0 }
    );

    let pq = parse_fixture("reference/libjxl/testdata/jxl/pq_gradient.jxl");
    let frame_data = pq.first_frame_data.as_ref().unwrap();
    assert_eq!(frame_data.toc.entries.len(), 6);
    assert_eq!(frame_data.payload_size, 107);
    assert_eq!(frame_data.sections[0].kind, FrameSectionKind::DcGlobal);
    assert_eq!(
        frame_data.sections[5].kind,
        FrameSectionKind::AcGroup { pass: 0, group: 2 }
    );

    let container =
        parse_fixture("reference/libjxl/testdata/jxl/boxes/square-extended-size-container.jxl");
    let frame_data = container.first_frame_data.as_ref().unwrap();
    assert_eq!(frame_data.toc.entries.len(), 1);
    assert_eq!(frame_data.sections[0].kind, FrameSectionKind::Combined);
    assert_eq!(frame_data.payload_size, 45);
}

#[test]
fn parses_checked_in_fixture_vardct_metadata() {
    let container =
        parse_fixture("reference/libjxl/testdata/jxl/boxes/square-extended-size-container.jxl");
    let vardct = container.first_frame_vardct.as_ref().unwrap();
    let vardct_plan = container.first_frame_vardct_plan.as_ref().unwrap();

    assert_eq!(vardct.width, 8);
    assert_eq!(vardct.height, 8);
    assert_eq!(vardct.group_dim, 256);
    assert_eq!(vardct.groups_x, 1);
    assert_eq!(vardct.groups_y, 1);
    assert_eq!(vardct.dc_groups_x, 1);
    assert_eq!(vardct.dc_groups_y, 1);
    assert_eq!(vardct.sections.len(), 1);
    assert_eq!(vardct.sections[0].section_kind, FrameSectionKind::Combined);
    assert_eq!(vardct.sections[0].payload_size, 45);
    assert!(vardct.is_combined);
    assert_eq!(
        vardct.global_section.as_ref().unwrap().section_kind,
        FrameSectionKind::Combined
    );
    assert!(vardct.ac_global_section.is_none());
    assert!(vardct.dc_group_sections.is_empty());
    assert!(vardct.ac_group_sections.is_empty());
    assert!(vardct_plan.frame.is_combined);
    let global = vardct_plan.global.as_ref().unwrap();
    assert!(global.bits_consumed > 0);
    assert!(global.bits_consumed <= vardct.sections[0].payload_size as usize * 8);
    assert_vardct_global_cursor_in_payload(global, vardct.sections[0].payload_size);
    assert!(global.dc_dequant.all_default);
    assert_eq!(global.dc_dequant.coefficients, None);
    assert!(global.quantizer.global_scale > 0);
    assert!(global.quantizer.quant_dc > 0);
    assert!(global.quantizer.scale > 0.0);
    assert!(global.quantizer.inv_global_scale > 0.0);
    assert!(global.quantizer.inv_quant_dc > 0.0);
    assert!(global.block_context_map.context_map_size > 0);
    assert!(global.block_context_map.num_contexts > 0);
    assert!(global.block_context_map.num_contexts <= 16);
    assert!(global.block_context_map.num_dc_contexts > 0);
    assert!(global.color_correlation.color_factor > 0);
    assert!(global.color_correlation.base_correlation_x.abs() <= 4.0);
    assert!(global.color_correlation.base_correlation_b.abs() <= 4.0);
    assert_eq!(
        vardct_plan
            .global_payload
            .as_ref()
            .unwrap()
            .section
            .section_kind,
        FrameSectionKind::Combined
    );
    assert_eq!(
        vardct_plan
            .global_payload
            .as_ref()
            .unwrap()
            .payload_range
            .len(),
        45
    );
    assert!(vardct_plan.ac_global_payload.is_none());
    assert!(vardct_plan.dc_group_payloads.is_empty());
    assert!(vardct_plan.ac_group_payloads.is_empty());
    assert_eq!(vardct.ac_groups.len(), 1);
    assert_eq!(vardct.ac_groups[0].group, 0);
    assert_eq!(vardct.ac_groups[0].x, 0);
    assert_eq!(vardct.ac_groups[0].y, 0);
    assert_eq!(vardct.ac_groups[0].width, 8);
    assert_eq!(vardct.ac_groups[0].height, 8);
    assert_eq!(vardct.dc_groups.len(), 1);
    assert_eq!(
        vardct.ac_groups_intersecting_region(ImageRegion {
            x: 4,
            y: 4,
            width: 1,
            height: 1,
        }),
        vec![0]
    );
    assert!(
        vardct
            .ac_sections_for_region(ImageRegion {
                x: 4,
                y: 4,
                width: 1,
                height: 1,
            })
            .is_empty()
    );
    assert!(
        vardct
            .dc_sections_for_region(ImageRegion {
                x: 4,
                y: 4,
                width: 1,
                height: 1,
            })
            .is_empty()
    );

    let pq = parse_fixture("reference/libjxl/testdata/jxl/pq_gradient.jxl");
    assert!(pq.first_frame_vardct.is_none());
    assert!(pq.first_frame_vardct_plan.is_none());
}

#[test]
fn parses_checked_in_fixture_modular_global_metadata() {
    let pq = parse_fixture("reference/libjxl/testdata/jxl/pq_gradient.jxl");
    let modular = pq.first_frame_modular.as_ref().unwrap();
    assert_eq!(modular.global.section_kind, FrameSectionKind::DcGlobal);
    assert!(modular.global.has_global_tree);
    assert_eq!(modular.global.global_tree.as_ref().unwrap().nodes.len(), 3);
    assert_eq!(modular.global.global_tree_contexts, Some(2));
    assert!(modular.global.group_header.use_global_tree);
    assert!(modular.global.group_header.weighted_predictor.all_default);
    assert_eq!(modular.global.group_header.transforms.len(), 1);
    let transform = &modular.global.group_header.transforms[0];
    assert_eq!(transform.id, TransformId::Palette);
    assert_eq!(transform.begin_c, 0);
    assert_eq!(transform.num_c, Some(1));
    assert_eq!(transform.nb_colors, Some(17));
    assert_eq!(transform.nb_deltas, Some(0));
    assert_eq!(modular.channel_plan.width, 1088);
    assert_eq!(modular.channel_plan.height, 64);
    assert_eq!(modular.channel_plan.nb_meta_channels, 1);
    assert_eq!(modular.channel_plan.channels.len(), 2);
    assert_eq!(modular.channel_plan.channels[0].width, 17);
    assert_eq!(modular.channel_plan.channels[0].height, 1);
    assert_eq!(modular.channel_plan.channels[0].hshift, -1);
    assert_eq!(modular.channel_plan.channels[1].width, 1088);
    assert_eq!(modular.channel_plan.channels[1].height, 64);
    assert_eq!(modular.groups.len(), 4);
    assert!(modular.groups[0].header.is_none());
    assert!(modular.groups[0].channels.is_empty());
    assert_eq!(
        modular.groups[0].section_kind,
        FrameSectionKind::DcGroup { group: 0 }
    );
    assert_eq!(modular.groups[0].stream_id, 2);
    assert_eq!(
        modular.groups[1].section_kind,
        FrameSectionKind::AcGroup { pass: 0, group: 0 }
    );
    assert_eq!(modular.groups[1].stream_id, 21);
    assert_eq!(modular.groups[1].bits_consumed, 4);
    assert!(modular.groups[1].header.as_ref().unwrap().use_global_tree);
    assert!(modular.groups[1].local_tree.is_none());
    assert_eq!(modular.groups[1].channels.len(), 1);
    assert_eq!(modular.groups[1].channels[0].channel_index, 1);
    assert_eq!(modular.groups[1].channels[0].width, 512);
    assert_eq!(modular.groups[1].channels[0].height, 64);
    assert_eq!(modular.groups[3].channels[0].x0, 1024);
    assert_eq!(modular.groups[3].channels[0].width, 64);
    assert_eq!(modular.groups[3].stream_id, 23);
    let residuals = modular.residuals.as_ref().unwrap();
    let planned_residual_streams = modular
        .groups
        .iter()
        .filter(|group| group.payload_size != 0 && !group.channels.is_empty())
        .map(|group| group.stream_id)
        .collect::<Vec<_>>();
    let decoded_residual_streams = residuals
        .groups
        .iter()
        .map(|group| group.stream_id)
        .collect::<Vec<_>>();
    assert_eq!(decoded_residual_streams, planned_residual_streams);
    let global = residuals.global.as_ref().unwrap();
    assert_eq!(global.stream_id, 0);
    assert_eq!(global.channels.len(), 1);
    assert_eq!(global.channels[0].channel_index, 0);
    assert_eq!(global.channels[0].x0, 0);
    assert_eq!(global.channels[0].y0, 0);
    assert_eq!(global.channels[0].samples.len(), 17);
    assert_eq!(global.channels[0].samples.iter().min(), Some(&6682));
    assert_eq!(global.channels[0].samples.iter().max(), Some(&58853));
    assert_eq!(residuals.groups.len(), 3);
    assert_eq!(residuals.groups[0].stream_id, 21);
    assert_eq!(residuals.groups[0].channels[0].x0, 0);
    assert_eq!(residuals.groups[0].channels[0].y0, 0);
    assert_eq!(residuals.groups[0].channels[0].samples.len(), 512 * 64);
    assert_eq!(
        residuals.groups[0].channels[0].samples.iter().min(),
        Some(&0)
    );
    assert_eq!(
        residuals.groups[0].channels[0].samples.iter().max(),
        Some(&7)
    );
    assert_eq!(residuals.groups[1].stream_id, 22);
    assert_eq!(residuals.groups[1].channels[0].x0, 512);
    assert_eq!(residuals.groups[1].channels[0].y0, 0);
    assert_eq!(
        residuals.groups[1].channels[0].samples.iter().min(),
        Some(&8)
    );
    assert_eq!(
        residuals.groups[1].channels[0].samples.iter().max(),
        Some(&15)
    );
    assert_eq!(residuals.groups[2].stream_id, 23);
    assert_eq!(residuals.groups[2].channels[0].x0, 1024);
    assert_eq!(residuals.groups[2].channels[0].y0, 0);
    assert_eq!(residuals.groups[2].channels[0].samples.len(), 64 * 64);
    assert!(
        residuals.groups[2].channels[0]
            .samples
            .iter()
            .all(|sample| *sample == 16)
    );
    let image = modular.image.as_ref().unwrap();
    assert_eq!(image.width, 1088);
    assert_eq!(image.height, 64);
    assert_eq!(image.channels.len(), 1);
    assert_eq!(image.channels[0].width, 1088);
    assert_eq!(image.channels[0].height, 64);
    assert_eq!(image.channels[0].samples.len(), 1088 * 64);
    assert_eq!(image.channels[0].samples.iter().min(), Some(&6682));
    assert_eq!(image.channels[0].samples.iter().max(), Some(&58853));

    let icc = parse_fixture("crates/jxl-codec/tests/generated/icc_rec2020_lossless.jxl");
    let modular = icc.first_frame_modular.as_ref().unwrap();
    assert_eq!(modular.global.section_kind, FrameSectionKind::Combined);
    assert_eq!(
        modular.global.global_tree.as_ref().unwrap().nodes.len(),
        309
    );
    assert_eq!(modular.global.global_tree_contexts, Some(155));
    assert_eq!(modular.global.group_header.transforms.len(), 1);
    let transform = &modular.global.group_header.transforms[0];
    assert_eq!(transform.id, TransformId::Rct);
    assert_eq!(transform.rct_type, Some(10));
    assert_eq!(modular.channel_plan.nb_meta_channels, 0);
    assert_eq!(modular.channel_plan.channels.len(), 3);
    assert_eq!(modular.channel_plan.channels[0].width, 64);
    assert!(modular.groups.is_empty());
    let residuals = modular.residuals.as_ref().unwrap();
    assert!(residuals.groups.is_empty());
    let global = residuals.global.as_ref().unwrap();
    assert_eq!(global.stream_id, 0);
    assert_eq!(global.channels.len(), 3);
    assert_eq!(global.channels[0].samples.len(), 64 * 64);
    assert_eq!(global.channels[0].samples.iter().min(), Some(&0));
    assert_eq!(global.channels[0].samples.iter().max(), Some(&14482));
    assert_eq!(global.channels[1].samples.iter().min(), Some(&-4651));
    assert_eq!(global.channels[1].samples.iter().max(), Some(&9364));
    assert_eq!(global.channels[2].samples.iter().min(), Some(&-3228));
    assert_eq!(global.channels[2].samples.iter().max(), Some(&7676));
    let image = modular.image.as_ref().unwrap();
    assert_eq!(image.width, 64);
    assert_eq!(image.height, 64);
    assert_eq!(image.channels.len(), 3);
    assert_eq!(image.channels[0].samples.iter().min(), Some(&0));
    assert_eq!(image.channels[0].samples.iter().max(), Some(&13717));
    assert_eq!(image.channels[1].samples.iter().min(), Some(&0));
    assert_eq!(image.channels[1].samples.iter().max(), Some(&14482));
    assert_eq!(image.channels[2].samples.iter().min(), Some(&0));
    assert_eq!(image.channels[2].samples.iter().max(), Some(&14045));

    let splines = parse_fixture("reference/libjxl/testdata/jxl/splines.jxl");
    assert!(splines.first_frame_modular.is_none());
}

#[test]
fn parses_generated_icc_profile_and_continues_to_first_frame() {
    let image = parse_fixture("crates/jxl-codec/tests/generated/icc_rec2020_lossless.jxl");
    assert!(image.metadata.color_encoding.want_icc);
    let icc = image.icc_profile.as_ref().unwrap();
    assert_eq!(icc.len(), 832);
    assert_eq!(&icc[36..40], b"acsp");
    assert_eq!(u32::from_be_bytes([icc[0], icc[1], icc[2], icc[3]]), 832);

    let frame = image.first_frame.as_ref().unwrap();
    assert_eq!(frame.encoding, FrameEncoding::Modular);
    assert_eq!(frame.color_transform, ColorTransform::None);
    assert_eq!(frame.frame_size.width, 64);
    assert_eq!(frame.frame_size.height, 64);
    assert_eq!(frame.group_layout.num_groups, 1);

    let frame_data = image.first_frame_data.as_ref().unwrap();
    assert_eq!(frame_data.toc.entries.len(), 1);
    assert_eq!(frame_data.sections[0].kind, FrameSectionKind::Combined);
    assert_eq!(frame_data.payload_size, 17_986);
}

fn workspace_path(relative: impl AsRef<Path>) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

fn reference_jxlinfo() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("JXL_RS_REFERENCE_JXLINFO") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }

    let default = workspace_path("reference/libjxl/build-rs-oracle/tools/jxlinfo");
    default.is_file().then_some(default)
}

fn reference_djxl() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("JXL_RS_REFERENCE_DJXL") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }

    let default = workspace_path("reference/libjxl/build-rs-oracle/tools/djxl");
    default.is_file().then_some(default)
}

fn reference_cjxl() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("JXL_RS_REFERENCE_CJXL") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }

    let default = workspace_path("reference/libjxl/build-rs-oracle/tools/cjxl");
    default.is_file().then_some(default)
}

fn reference_vardct_trace() -> Option<PathBuf> {
    let path = std::env::var("JXL_RS_REFERENCE_TRACE").ok()?;
    let path = PathBuf::from(path);
    path.is_file().then_some(path)
}

fn compare_reference_vardct_trace(encoded: &Path, plan: &jxl_codec::VarDctDecodePlan) {
    let Some(trace) = reference_vardct_trace() else {
        eprintln!(
            "skipping split VarDCT internal trace comparison; set JXL_RS_REFERENCE_TRACE to a trace tool"
        );
        return;
    };

    let output = Command::new(&trace).arg(encoded).output().unwrap();
    assert!(
        output.status.success(),
        "reference trace failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("reference trace output was not UTF-8");

    assert_trace_field(
        &stdout,
        "global_tree_bits",
        &format!(
            "{}..{}",
            plan.modular_global_tree_direct_start_bits.unwrap(),
            plan.modular_global_tree_direct_tree_end_bits.unwrap()
        ),
    );
    assert_trace_field(
        &stdout,
        "residual_contexts",
        &plan
            .modular_global_tree_direct_residual_context_count
            .unwrap()
            .to_string(),
    );
    assert_trace_field(
        &stdout,
        "residual_context_map",
        &join_u8(&plan.modular_global_tree_direct_residual_context_map_entries),
    );
    assert_trace_field(
        &stdout,
        "residual_histograms",
        &plan
            .modular_global_tree_direct_residual_histogram_count
            .unwrap()
            .to_string(),
    );
    assert_trace_field(
        &stdout,
        "residual_histogram_bits",
        &plan
            .modular_global_tree_direct_residual_ans_histograms
            .iter()
            .map(|histogram| {
                let end = histogram
                    .end_bits
                    .or(histogram.error_bits)
                    .unwrap_or(histogram.start_bits);
                format!("{}..{}", histogram.start_bits, end)
            })
            .collect::<Vec<_>>()
            .join(","),
    );
    if let (Some(index), Some(error_bits)) = (
        plan.modular_global_tree_direct_residual_failed_histogram_index,
        plan.modular_global_tree_direct_error_bits,
    ) {
        assert_trace_field(
            &stdout,
            "residual_histogram_error",
            &format!("{index}@{error_bits}"),
        );
    } else {
        assert!(
            !stdout
                .lines()
                .any(|line| line.starts_with("residual_histogram_error=")),
            "unexpected reference trace residual_histogram_error: {stdout}"
        );
    }
}

fn assert_trace_field(stdout: &str, key: &str, expected: &str) {
    let prefix = format!("{key}=");
    let actual = stdout
        .lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("reference trace did not emit {key}=...; output:\n{stdout}"));
    assert_eq!(actual.trim(), expected, "reference trace field {key}");
}

fn join_u8(values: &[u8]) -> String {
    values
        .iter()
        .map(u8::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn unique_temp_path(prefix: &str, extension: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "{prefix}-{}-{nanos}.{extension}",
        std::process::id()
    ))
}

fn parse_fixture(path: &str) -> jxl_codec::Codestream {
    let bytes = std::fs::read(workspace_path(path)).unwrap();
    let (_, codestream) = parse_file(&bytes).unwrap();
    codestream
}
