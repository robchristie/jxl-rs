use crate::codestream::{CODESTREAM_SIGNATURE, Codestream, parse_codestream_with_config};
use crate::decode::DecodeConfig;
use crate::error::{Error, Result};

pub const CONTAINER_SIGNATURE: [u8; 12] = [
    0x00, 0x00, 0x00, 0x0c, b'J', b'X', b'L', b' ', 0x0d, 0x0a, 0x87, 0x0a,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signature {
    NotEnoughBytes,
    Invalid,
    Codestream,
    Container,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    NakedCodestream,
    Container,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Container {
    pub boxes: Vec<BoxRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoxRecord {
    pub box_type: [u8; 4],
    pub header_size: u64,
    pub content_size: Option<u64>,
    pub offset: u64,
}

impl BoxRecord {
    pub fn box_type_string(&self) -> String {
        String::from_utf8_lossy(&self.box_type).into_owned()
    }

    pub fn total_size(&self) -> Option<u64> {
        self.content_size.map(|size| size + self.header_size)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContainerBox<'a> {
    Borrowed {
        record: BoxRecord,
        contents: &'a [u8],
    },
    Unbounded {
        record: BoxRecord,
        contents: &'a [u8],
    },
}

impl<'a> ContainerBox<'a> {
    pub fn record(&self) -> &BoxRecord {
        match self {
            Self::Borrowed { record, .. } | Self::Unbounded { record, .. } => record,
        }
    }

    pub fn contents(&self) -> &'a [u8] {
        match self {
            Self::Borrowed { contents, .. } | Self::Unbounded { contents, .. } => contents,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedCodestream {
    pub format: FileFormat,
    pub container: Option<Container>,
    pub codestream: Vec<u8>,
}

pub fn check_signature(input: &[u8]) -> Signature {
    if input.is_empty() {
        return Signature::NotEnoughBytes;
    }

    if input[0] == 0xff {
        if input.len() < CODESTREAM_SIGNATURE.len() {
            return Signature::NotEnoughBytes;
        }
        return if input.starts_with(&CODESTREAM_SIGNATURE) {
            Signature::Codestream
        } else {
            Signature::Invalid
        };
    }

    if input[0] == 0 {
        if input.len() < CONTAINER_SIGNATURE.len() {
            return Signature::NotEnoughBytes;
        }
        return if input.starts_with(&CONTAINER_SIGNATURE) {
            Signature::Container
        } else {
            Signature::Invalid
        };
    }

    Signature::Invalid
}

pub fn parse_file(input: &[u8]) -> Result<(ExtractedCodestream, Codestream)> {
    parse_file_with_config(input, DecodeConfig::default())
}

pub fn parse_file_with_config(
    input: &[u8],
    config: DecodeConfig,
) -> Result<(ExtractedCodestream, Codestream)> {
    let extracted = extract_codestream(input)?;
    let codestream = parse_codestream_with_config(&extracted.codestream, config)?;
    Ok((extracted, codestream))
}

pub fn extract_codestream(input: &[u8]) -> Result<ExtractedCodestream> {
    match check_signature(input) {
        Signature::NotEnoughBytes => Err(Error::Truncated),
        Signature::Invalid => Err(Error::InvalidSignature),
        Signature::Codestream => Ok(ExtractedCodestream {
            format: FileFormat::NakedCodestream,
            container: None,
            codestream: input.to_vec(),
        }),
        Signature::Container => extract_container_codestream(input),
    }
}

fn extract_container_codestream(input: &[u8]) -> Result<ExtractedCodestream> {
    let mut boxes = Vec::new();
    let mut codestream = Vec::new();
    let mut cursor = 0usize;
    let mut box_index = 0usize;
    let mut saw_ftyp = false;
    let mut saw_jxlc = false;
    let mut saw_jxlp = false;
    let mut expected_jxlp_index = 0u32;
    let mut last_codestream_seen = false;

    while cursor < input.len() {
        let parsed = read_box(input, cursor)?;
        let record = parsed.record().clone();
        let contents = parsed.contents();
        box_index += 1;

        match &record.box_type {
            b"JXL " if box_index != 1 || contents != [0x0d, 0x0a, 0x87, 0x0a] => {
                return Err(Error::InvalidContainer("invalid signature box"));
            }
            b"JXL " => {}
            b"ftyp" => {
                if box_index != 2 {
                    return Err(Error::InvalidContainer("ftyp box must come second"));
                }
                if record.header_size != 8
                    || record.content_size != Some(12)
                    || contents != b"jxl \0\0\0\0jxl "
                {
                    return Err(Error::InvalidContainer("invalid ftyp box"));
                }
                saw_ftyp = true;
            }
            b"jxlc" => {
                if saw_jxlc || saw_jxlp {
                    return Err(Error::InvalidContainer(
                        "jxlc cannot appear more than once or be mixed with jxlp",
                    ));
                }
                saw_jxlc = true;
                last_codestream_seen = true;
                codestream.extend_from_slice(contents);
            }
            b"jxlp" => {
                if saw_jxlc {
                    return Err(Error::InvalidContainer("jxlp cannot appear after jxlc"));
                }
                if last_codestream_seen {
                    return Err(Error::InvalidContainer(
                        "jxlp appears after final codestream part",
                    ));
                }
                if contents.len() < 4 {
                    return Err(Error::InvalidContainer("jxlp box missing index"));
                }

                let raw_index =
                    u32::from_be_bytes(contents[..4].try_into().expect("slice length checked"));
                let is_last = (raw_index & 0x8000_0000) != 0;
                let part_index = raw_index & 0x7fff_ffff;
                if part_index != expected_jxlp_index {
                    return Err(Error::InvalidContainer("non-contiguous jxlp index"));
                }

                saw_jxlp = true;
                expected_jxlp_index += 1;
                last_codestream_seen = is_last;
                codestream.extend_from_slice(&contents[4..]);
            }
            _ => {}
        }

        boxes.push(record);
        cursor = match parsed.record().content_size {
            Some(size) => next_box_offset(parsed.record(), size)?,
            None => input.len(),
        };
    }

    if !saw_ftyp {
        return Err(Error::InvalidContainer("missing ftyp box"));
    }
    if codestream.is_empty() {
        return Err(Error::InvalidContainer("missing codestream box"));
    }
    if saw_jxlp && !last_codestream_seen {
        return Err(Error::InvalidContainer("missing final jxlp box"));
    }

    Ok(ExtractedCodestream {
        format: FileFormat::Container,
        container: Some(Container { boxes }),
        codestream,
    })
}

fn next_box_offset(record: &BoxRecord, content_size: u64) -> Result<usize> {
    u64_to_usize(record.offset, "box offset overflow")?
        .checked_add(u64_to_usize(record.header_size, "box offset overflow")?)
        .and_then(|start| {
            start.checked_add(u64_to_usize(content_size, "box offset overflow").ok()?)
        })
        .ok_or(Error::InvalidContainer("box offset overflow"))
}

fn u64_to_usize(value: u64, msg: &'static str) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::InvalidContainer(msg))
}

fn read_box(input: &[u8], offset: usize) -> Result<ContainerBox<'_>> {
    if input.len() - offset < 8 {
        return Err(Error::Truncated);
    }

    let size32 = u32::from_be_bytes(input[offset..offset + 4].try_into().unwrap());
    let box_type = input[offset + 4..offset + 8].try_into().unwrap();
    let mut header_size = 8u64;
    let content_size = match size32 {
        0 => None,
        1 => {
            if input.len() - offset < 16 {
                return Err(Error::Truncated);
            }
            let size64 = u64::from_be_bytes(input[offset + 8..offset + 16].try_into().unwrap());
            if size64 < 16 {
                return Err(Error::InvalidContainer(
                    "extended box size is smaller than header",
                ));
            }
            header_size = 16;
            Some(size64 - header_size)
        }
        size if size < 8 => {
            return Err(Error::InvalidContainer("box size is smaller than header"));
        }
        size => Some(u64::from(size) - header_size),
    };

    let header_size_usize = u64_to_usize(header_size, "box offset overflow")?;
    let content_start = offset
        .checked_add(header_size_usize)
        .ok_or(Error::InvalidContainer("box offset overflow"))?;
    let content_end = match content_size {
        Some(size) => {
            let size = u64_to_usize(size, "box offset overflow")?;
            content_start
                .checked_add(size)
                .ok_or(Error::InvalidContainer("box offset overflow"))?
        }
        None => input.len(),
    };

    if content_end > input.len() {
        return Err(Error::Truncated);
    }

    let record = BoxRecord {
        box_type,
        header_size,
        content_size,
        offset: offset as u64,
    };
    let contents = &input[content_start..content_end];
    Ok(match content_size {
        Some(_) => ContainerBox::Borrowed { record, contents },
        None => ContainerBox::Unbounded { record, contents },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifies_signatures_like_libjxl() {
        assert_eq!(check_signature(&[]), Signature::NotEnoughBytes);
        assert_eq!(check_signature(&[0xff]), Signature::NotEnoughBytes);
        assert_eq!(check_signature(&[0xff, 0x0a]), Signature::Codestream);
        assert_eq!(check_signature(&[0xff, 0xd8]), Signature::Invalid);
        assert_eq!(check_signature(&CONTAINER_SIGNATURE), Signature::Container);
    }

    #[test]
    fn extracts_naked_codestream_without_copying_container_state() {
        let input = [0xff, 0x0a, 0x21];
        let extracted = extract_codestream(&input).unwrap();

        assert_eq!(extracted.format, FileFormat::NakedCodestream);
        assert!(extracted.container.is_none());
        assert_eq!(extracted.codestream, input);
    }

    #[test]
    fn extracts_container_with_exact_ftyp() {
        let input = container_with_ftyp(b"jxl \0\0\0\0jxl ");
        let extracted = extract_codestream(&input).unwrap();

        assert_eq!(extracted.format, FileFormat::Container);
        assert_eq!(extracted.codestream, [0xff, 0x0a]);
    }

    #[test]
    fn rejects_ftyp_with_extra_bytes() {
        let input = container_with_ftyp(b"jxl \0\0\0\0jxl extra");

        assert_eq!(
            extract_codestream(&input).unwrap_err(),
            Error::InvalidContainer("invalid ftyp box")
        );
    }

    #[test]
    fn rejects_ftyp_with_nonzero_minor_version() {
        let input = container_with_ftyp(b"jxl \0\0\0\x01jxl ");

        assert_eq!(
            extract_codestream(&input).unwrap_err(),
            Error::InvalidContainer("invalid ftyp box")
        );
    }

    #[test]
    fn rejects_ftyp_missing_compatible_brand() {
        let input = container_with_ftyp(b"jxl \0\0\0\0bad ");

        assert_eq!(
            extract_codestream(&input).unwrap_err(),
            Error::InvalidContainer("invalid ftyp box")
        );
    }

    #[test]
    fn rejects_ftyp_not_second() {
        let mut input = Vec::new();
        push_box(&mut input, *b"JXL ", &[0x0d, 0x0a, 0x87, 0x0a]);
        push_box(&mut input, *b"free", &[]);
        push_box(&mut input, *b"ftyp", b"jxl \0\0\0\0jxl ");
        push_box(&mut input, *b"jxlc", &[0xff, 0x0a]);

        assert_eq!(
            extract_codestream(&input).unwrap_err(),
            Error::InvalidContainer("ftyp box must come second")
        );
    }

    #[test]
    fn rejects_box_offset_overflow() {
        let record = BoxRecord {
            box_type: *b"free",
            header_size: 16,
            content_size: Some(u64::MAX),
            offset: 1,
        };

        assert_eq!(
            next_box_offset(&record, u64::MAX).unwrap_err(),
            Error::InvalidContainer("box offset overflow")
        );
    }

    fn container_with_ftyp(ftyp_contents: &[u8]) -> Vec<u8> {
        let mut input = Vec::new();
        push_box(&mut input, *b"JXL ", &[0x0d, 0x0a, 0x87, 0x0a]);
        push_box(&mut input, *b"ftyp", ftyp_contents);
        push_box(&mut input, *b"jxlc", &[0xff, 0x0a]);
        input
    }

    fn push_box(output: &mut Vec<u8>, box_type: [u8; 4], contents: &[u8]) {
        let size = u32::try_from(8 + contents.len()).unwrap();
        output.extend_from_slice(&size.to_be_bytes());
        output.extend_from_slice(&box_type);
        output.extend_from_slice(contents);
    }
}
