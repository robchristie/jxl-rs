use std::{
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use jxl_codec::{
    BlendMode, ColorSpace, ColorTransform, DecodeConfig, ExtraChannelType, FileFormat,
    FrameEncoding, FrameSectionKind, FrameType, ImageRegion, ModularGroupExecution,
    TransferFunction, TransformId, parse_file, parse_file_with_config,
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
    samples: Vec<u16>,
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
    let _ = std::fs::remove_file(&reference_output);

    let encoded_bytes = std::fs::read(&encoded).unwrap();
    let _ = std::fs::remove_file(&encoded);
    let (_, codestream) = parse_file(&encoded_bytes).unwrap();
    let plan = codestream.first_frame_vardct_plan.as_ref().unwrap();

    assert!(
        !plan.frame.is_combined,
        "generated VarDCT fixture unexpectedly used a combined section"
    );
    assert!(plan.global_payload.is_some());
    assert!(plan.ac_global_payload.is_some());
    assert!(!plan.dc_group_payloads.is_empty());
    assert!(!plan.ac_group_payloads.is_empty());

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
        assert_eq!(metadata.parse_error, None);
        assert_eq!(metadata.cursor.var_dct_dc_end_bits, Some(6));
        assert_eq!(metadata.cursor.modular_dc_start_bits, Some(6));
        assert_eq!(metadata.cursor.modular_dc_end_bits, None);
        assert_eq!(metadata.cursor.ac_metadata_start_bits, None);
        assert_eq!(metadata.cursor.ac_metadata_end_bits, None);
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
    assert_eq!(plan.modular_global_tree_direct_start_bits, Some(206));
    assert_eq!(plan.modular_global_tree_direct_tree_end_bits, Some(520));
    assert_eq!(plan.modular_global_tree_direct_tree_node_count, Some(31));
    assert_eq!(plan.modular_global_tree_direct_error_bits, Some(1035));
    assert_eq!(plan.modular_global_tree_start_bits, Some(220));
    assert_eq!(
        plan.modular_global_tree_direct_error,
        Some(jxl_codec::Error::Truncated)
    );
    assert_eq!(plan.modular_global_tree_error, None);
    assert_vardct_global_cursor_in_payload(global, global.section.section.payload_size);
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
        samples,
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
