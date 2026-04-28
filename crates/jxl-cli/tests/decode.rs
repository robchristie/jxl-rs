use std::{
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn decode_cli_writes_rgba8_pam() {
    let input = workspace_path("crates/jxl-codec/tests/generated/icc_rec2020_lossless.jxl");
    let output = unique_temp_path("jxl-cli-decode", "pam");

    let result = Command::new(env!("CARGO_BIN_EXE_jxl-decode-rs"))
        .arg(&input)
        .arg(&output)
        .output()
        .unwrap();

    assert!(
        result.status.success(),
        "jxl-decode-rs failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );

    let bytes = std::fs::read(&output).unwrap();
    let _ = std::fs::remove_file(&output);
    let pam = parse_pam_rgba(&bytes);
    assert_eq!(pam.width, 64);
    assert_eq!(pam.height, 64);
    assert_eq!(pam.maxval, 255);
    assert_eq!(pam.samples.len(), 64 * 64 * 4);
    assert!(pam.samples.chunks_exact(4).all(|pixel| pixel[3] == 255));
}

#[test]
fn decode_cli_writes_rgba16_pam() {
    let input = workspace_path("crates/jxl-codec/tests/generated/icc_rec2020_lossless.jxl");
    let output = unique_temp_path("jxl-cli-decode-16", "pam");

    let result = Command::new(env!("CARGO_BIN_EXE_jxl-decode-rs"))
        .arg(&input)
        .arg(&output)
        .args(["--bits", "16"])
        .output()
        .unwrap();

    assert!(
        result.status.success(),
        "jxl-decode-rs failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );

    let bytes = std::fs::read(&output).unwrap();
    let _ = std::fs::remove_file(&output);
    let pam = parse_pam_rgba(&bytes);
    assert_eq!(pam.width, 64);
    assert_eq!(pam.height, 64);
    assert_eq!(pam.maxval, 65535);
    assert_eq!(pam.samples.len(), 64 * 64 * 4);
    assert!(pam.samples.chunks_exact(4).all(|pixel| pixel[3] == 65535));
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PamRgba {
    width: u32,
    height: u32,
    maxval: u32,
    samples: Vec<u16>,
}

fn parse_pam_rgba(bytes: &[u8]) -> PamRgba {
    let header_end = bytes
        .windows(7)
        .position(|window| window == b"ENDHDR\n")
        .map(|index| index + 7)
        .expect("PAM header did not contain ENDHDR");
    let header = std::str::from_utf8(&bytes[..header_end]).unwrap();
    assert!(header.starts_with("P7\n"));
    let mut width = None;
    let mut height = None;
    let mut depth = None;
    let mut maxval = None;
    let mut tupltype = None;
    for line in header.lines() {
        let mut fields = line.splitn(2, ' ');
        match (fields.next(), fields.next()) {
            (Some("WIDTH"), Some(value)) => width = Some(value.parse::<u32>().unwrap()),
            (Some("HEIGHT"), Some(value)) => height = Some(value.parse::<u32>().unwrap()),
            (Some("DEPTH"), Some(value)) => depth = Some(value.parse::<u32>().unwrap()),
            (Some("MAXVAL"), Some(value)) => maxval = Some(value.parse::<u32>().unwrap()),
            (Some("TUPLTYPE"), Some(value)) => tupltype = Some(value),
            _ => {}
        }
    }
    assert_eq!(depth, Some(4));
    let maxval = maxval.unwrap();
    assert!(matches!(maxval, 255 | 65535));
    assert_eq!(tupltype, Some("RGB_ALPHA"));
    let width = width.unwrap();
    let height = height.unwrap();
    let data = &bytes[header_end..];
    let samples = if maxval > 255 {
        assert_eq!(data.len(), width as usize * height as usize * 4 * 2);
        data.chunks_exact(2)
            .map(|chunk| u16::from_be_bytes([chunk[0], chunk[1]]))
            .collect()
    } else {
        assert_eq!(data.len(), width as usize * height as usize * 4);
        data.iter().copied().map(u16::from).collect()
    };
    PamRgba {
        width,
        height,
        maxval,
        samples,
    }
}

fn workspace_path(relative: impl AsRef<Path>) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
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
