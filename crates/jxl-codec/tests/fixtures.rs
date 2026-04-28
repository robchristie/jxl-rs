use std::{
    path::{Path, PathBuf},
    process::Command,
};

use jxl_codec::{
    BlendMode, ColorSpace, ColorTransform, ExtraChannelType, FileFormat, FrameEncoding,
    FrameSectionKind, FrameType, TransferFunction, TransformId, parse_file,
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

fn parse_fixture(path: &str) -> jxl_codec::Codestream {
    let bytes = std::fs::read(workspace_path(path)).unwrap();
    let (_, codestream) = parse_file(&bytes).unwrap();
    codestream
}
