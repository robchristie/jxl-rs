use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
};

use clap::Parser;

#[derive(Debug, Parser)]
#[command(about = "Decode supported JPEG XL still images to raw RGBA8 PAM")]
struct Args {
    /// JPEG XL file to decode.
    input: PathBuf,

    /// Output PAM path. Use '-' to write to stdout.
    output: PathBuf,
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
    let image = jxl::decode_rgba8(&input)?;

    if args.output.as_os_str() == "-" {
        let mut stdout = io::stdout().lock();
        write_pam(&mut stdout, &image)?;
    } else {
        let mut output = fs::File::create(&args.output)?;
        write_pam(&mut output, &image)?;
    }
    Ok(())
}

fn write_pam(mut writer: impl Write, image: &jxl::RgbaImage) -> io::Result<()> {
    writeln!(
        writer,
        "P7\nWIDTH {}\nHEIGHT {}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR",
        image.width, image.height
    )?;
    writer.write_all(&image.pixels)
}
