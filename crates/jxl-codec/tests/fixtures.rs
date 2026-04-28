use std::{
    path::{Path, PathBuf},
    process::Command,
};

use jxl_codec::{
    BlendMode, ColorSpace, ColorTransform, ExtraChannelType, FileFormat, FrameEncoding, FrameType,
    TransferFunction, parse_file,
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
