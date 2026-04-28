use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
};

use clap::{Parser, ValueEnum};

#[derive(Debug, Parser)]
#[command(about = "Decode supported JPEG XL still images to raw RGBA PAM")]
struct Args {
    /// JPEG XL file to decode.
    input: PathBuf,

    /// Output PAM path. Use '-' to write to stdout.
    output: PathBuf,

    /// Output sample depth.
    #[arg(long = "bits", value_enum, default_value_t = OutputBits::Eight)]
    bits: OutputBits,
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

    if args.output.as_os_str() == "-" {
        let mut stdout = io::stdout().lock();
        decode_and_write_pam(&mut stdout, &input, args.bits)?;
    } else {
        let mut output = fs::File::create(&args.output)?;
        decode_and_write_pam(&mut output, &input, args.bits)?;
    }
    Ok(())
}

fn decode_and_write_pam(
    writer: impl Write,
    input: &[u8],
    bits: OutputBits,
) -> Result<(), Box<dyn std::error::Error>> {
    match bits {
        OutputBits::Eight => write_pam8(writer, &jxl::decode_rgba8(input)?)?,
        OutputBits::Sixteen => write_pam16(writer, &jxl::decode_rgba16(input)?)?,
    }
    Ok(())
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
