use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
};

use clap::{Parser, ValueEnum};

#[derive(Debug, Parser)]
#[command(about = "Decode supported JPEG XL still images to RGBA PAM")]
struct Args {
    /// JPEG XL file to decode.
    input: PathBuf,

    /// Output PAM path. Use '-' to write to stdout.
    output: PathBuf,

    /// Output sample depth.
    #[arg(long = "bits", value_enum, default_value_t = OutputBits::Eight)]
    bits: OutputBits,

    /// Alias for `--bits 8`.
    #[arg(long, conflicts_with = "rgba16")]
    rgba8: bool,

    /// Alias for `--bits 16`.
    #[arg(long, conflicts_with = "rgba8")]
    rgba16: bool,

    /// Decode a region of interest as x,y,width,height.
    #[arg(long, value_parser = parse_roi)]
    roi: Option<jxl::Rect>,

    /// Select exactly one VarDCT AC pass for progressive RGB/RGBA output.
    #[arg(long = "vardct-pass")]
    vardct_pass: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum OutputBits {
    #[value(name = "8")]
    Eight,
    #[value(name = "16")]
    Sixteen,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("jxl-decode-rs: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let input = fs::read(&args.input)?;
    let bits = if args.rgba16 {
        OutputBits::Sixteen
    } else {
        OutputBits::Eight
    };
    let bits = if args.rgba8 || args.rgba16 {
        bits
    } else {
        args.bits
    };

    if args.output.as_os_str() == "-" {
        let mut stdout = io::stdout().lock();
        decode_and_write_pam(&mut stdout, &input, bits, args.roi, args.vardct_pass)?;
    } else {
        let mut output = fs::File::create(&args.output)?;
        decode_and_write_pam(&mut output, &input, bits, args.roi, args.vardct_pass)?;
    }
    Ok(())
}

fn decode_and_write_pam(
    writer: impl Write,
    input: &[u8],
    bits: OutputBits,
    roi: Option<jxl::Rect>,
    vardct_pass: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut decoder = jxl::Decoder::new();
    if let Some(roi) = roi {
        decoder = decoder.roi(roi);
    }
    if let Some(pass) = vardct_pass {
        decoder = decoder.vardct_pass(pass);
    }

    match bits {
        OutputBits::Eight => write_pam8(writer, &decoder.decode_rgba8(input)?)?,
        OutputBits::Sixteen => write_pam16(writer, &decoder.decode_rgba16(input)?)?,
    }
    Ok(())
}

fn parse_roi(value: &str) -> Result<jxl::Rect, String> {
    let fields = value
        .split(',')
        .map(|field| {
            field
                .parse::<u32>()
                .map_err(|_| format!("invalid ROI component `{field}`"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let [x, y, width, height]: [u32; 4] = fields
        .try_into()
        .map_err(|_| "ROI must have four comma-separated fields: x,y,width,height".to_string())?;
    if width == 0 || height == 0 {
        return Err("ROI width and height must be nonzero".to_string());
    }
    Ok(jxl::Rect {
        x,
        y,
        width,
        height,
    })
}

fn write_pam8(mut writer: impl Write, image: &jxl::RgbaImage) -> io::Result<()> {
    writeln!(
        writer,
        "P7\nWIDTH {}\nHEIGHT {}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR",
        image.width, image.height
    )?;
    writer.write_all(&image.pixels)
}

fn write_pam16(mut writer: impl Write, image: &jxl::Rgba16Image) -> io::Result<()> {
    writeln!(
        writer,
        "P7\nWIDTH {}\nHEIGHT {}\nDEPTH 4\nMAXVAL 65535\nTUPLTYPE RGB_ALPHA\nENDHDR",
        image.width, image.height
    )?;
    for sample in &image.pixels {
        writer.write_all(&sample.to_be_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_roi_rejects_zero_width_or_height() {
        assert_eq!(
            parse_roi("1,2,0,4").unwrap_err(),
            "ROI width and height must be nonzero"
        );
        assert_eq!(
            parse_roi("1,2,3,0").unwrap_err(),
            "ROI width and height must be nonzero"
        );
    }
}
