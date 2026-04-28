use crate::bitstream::BitReader;
use crate::entropy::{AnsSymbolReader, decode_histograms};
use crate::error::{Error, Result};

const NUM_ICC_CONTEXTS: usize = 41;
const ICC_HEADER_SIZE: usize = 128;
const ICC_DECODED_LIMIT: usize = 1 << 28;

pub fn read_icc_profile(reader: &mut BitReader<'_>) -> Result<Vec<u8>> {
    let encoded_size = reader.read_u64()? as usize;
    if encoded_size > ICC_DECODED_LIMIT {
        return Err(Error::InvalidCodestream("encoded ICC profile is too large"));
    }

    let (code, context_map) = decode_histograms(reader, NUM_ICC_CONTEXTS, false)?;
    let mut symbol_reader = AnsSymbolReader::new(code, reader)?;
    let mut decompressed = Vec::with_capacity(encoded_size);
    for index in 0..encoded_size {
        let previous = if index > 0 {
            decompressed[index - 1]
        } else {
            0
        };
        let second_previous = if index > 1 {
            decompressed[index - 2]
        } else {
            0
        };
        let context = icc_ans_context(index, previous, second_previous);
        let value = symbol_reader.read_hybrid_uint(context, reader, &context_map)?;
        if value > u8::MAX.into() {
            return Err(Error::InvalidCodestream("ICC byte exceeds u8"));
        }
        decompressed.push(value as u8);
    }
    if !symbol_reader.check_final_state() {
        return Err(Error::InvalidCodestream("invalid ICC ANS final state"));
    }

    unpredict_icc(&decompressed)
}

fn unpredict_icc(encoded: &[u8]) -> Result<Vec<u8>> {
    let mut position = 0;
    let output_size = decode_varint(encoded, &mut position)? as usize;
    if output_size > ICC_DECODED_LIMIT {
        return Err(Error::InvalidCodestream("decoded ICC profile is too large"));
    }
    let commands_size = decode_varint(encoded, &mut position)? as usize;
    let mut command_position = position;
    check_bounds(position, commands_size, encoded.len())?;
    let commands_end = command_position + commands_size;
    position = commands_end;

    let mut result = Vec::with_capacity(output_size);
    let mut header = initial_header_prediction(output_size as u32);
    for index in 0..=ICC_HEADER_SIZE {
        if result.len() == output_size {
            if command_position != commands_end {
                return Err(Error::InvalidCodestream("unused ICC commands"));
            }
            if position != encoded.len() {
                return Err(Error::InvalidCodestream("unused ICC data"));
            }
            validate_icc_size(&result)?;
            return Ok(result);
        }
        if index == ICC_HEADER_SIZE {
            break;
        }
        predict_header(&result, &mut header, index);
        let byte = *encoded
            .get(position)
            .ok_or(Error::InvalidCodestream("truncated ICC header"))?;
        position += 1;
        result.push(byte.wrapping_add(header[index]));
    }
    if command_position >= commands_end {
        return Err(Error::InvalidCodestream("truncated ICC commands"));
    }

    let num_tags_encoded = decode_varint(encoded, &mut command_position)?;
    if num_tags_encoded != 0 {
        let num_tags = num_tags_encoded
            .checked_sub(1)
            .ok_or(Error::InvalidCodestream("invalid ICC tag count"))?;
        if num_tags > u32::MAX.into() {
            return Err(Error::InvalidCodestream("ICC tag count exceeds u32"));
        }
        append_u32(num_tags as u32, &mut result);
        let mut previous_tag_start = ICC_HEADER_SIZE as u64 + num_tags * 12;
        let mut previous_tag_size = 0u64;
        loop {
            if result.len() > output_size {
                return Err(Error::InvalidCodestream(
                    "ICC output exceeded expected size",
                ));
            }
            if command_position > commands_end {
                return Err(Error::InvalidCodestream("ICC command overread"));
            }
            if command_position == commands_end {
                break;
            }
            let command = encoded[command_position];
            command_position += 1;
            let tag_code = command & 63;
            let tag = match tag_code {
                0 => break,
                COMMAND_TAG_UNKNOWN => {
                    check_bounds(position, 4, encoded.len())?;
                    let tag = read_tag(encoded, position)?;
                    position += 4;
                    tag
                }
                COMMAND_TAG_TRC => *RTRC_TAG,
                COMMAND_TAG_XYZ => *RXYZ_TAG,
                code => {
                    let index = usize::from(code.saturating_sub(COMMAND_TAG_STRING_FIRST));
                    *TAG_STRINGS
                        .get(index)
                        .ok_or(Error::InvalidCodestream("unknown ICC tag code"))?
                }
            };
            result.extend_from_slice(&tag);

            let mut tag_size = previous_tag_size;
            if matches!(
                tag,
                RXYZ_TAG_VALUE
                    | GXYZ_TAG_VALUE
                    | BXYZ_TAG_VALUE
                    | KXYZ_TAG_VALUE
                    | WTPT_TAG_VALUE
                    | BKPT_TAG_VALUE
                    | LUMI_TAG_VALUE
            ) {
                tag_size = 20;
            }

            let tag_start = if command & FLAG_BIT_OFFSET != 0 {
                decode_varint(encoded, &mut command_position)?
            } else {
                previous_tag_start + previous_tag_size
            };
            if tag_start > u32::MAX.into() || tag_size > u32::MAX.into() {
                return Err(Error::InvalidCodestream("ICC tag offset exceeds u32"));
            }
            append_u32(tag_start as u32, &mut result);
            if command & FLAG_BIT_SIZE != 0 {
                tag_size = decode_varint(encoded, &mut command_position)?;
            }
            if tag_size > u32::MAX.into() {
                return Err(Error::InvalidCodestream("ICC tag size exceeds u32"));
            }
            append_u32(tag_size as u32, &mut result);
            previous_tag_start = tag_start;
            previous_tag_size = tag_size;

            if tag_code == COMMAND_TAG_TRC {
                append_tag(*GTRC_TAG, &mut result);
                append_u32(tag_start as u32, &mut result);
                append_u32(tag_size as u32, &mut result);
                append_tag(*BTRC_TAG, &mut result);
                append_u32(tag_start as u32, &mut result);
                append_u32(tag_size as u32, &mut result);
            }

            if tag_code == COMMAND_TAG_XYZ {
                let second_start = tag_start
                    .checked_add(tag_size)
                    .ok_or(Error::InvalidCodestream("ICC tag offset overflow"))?;
                let third_start = second_start
                    .checked_add(tag_size)
                    .ok_or(Error::InvalidCodestream("ICC tag offset overflow"))?;
                if third_start > u32::MAX.into() {
                    return Err(Error::InvalidCodestream("ICC tag offset exceeds u32"));
                }
                append_tag(*GXYZ_TAG, &mut result);
                append_u32(second_start as u32, &mut result);
                append_u32(tag_size as u32, &mut result);
                append_tag(*BXYZ_TAG, &mut result);
                append_u32(third_start as u32, &mut result);
                append_u32(tag_size as u32, &mut result);
            }
        }
    }

    loop {
        if result.len() > output_size {
            return Err(Error::InvalidCodestream(
                "ICC output exceeded expected size",
            ));
        }
        if command_position > commands_end {
            return Err(Error::InvalidCodestream("ICC command overread"));
        }
        if command_position == commands_end {
            break;
        }
        let command = encoded[command_position];
        command_position += 1;
        match command {
            COMMAND_INSERT => {
                let count = decode_varint(encoded, &mut command_position)? as usize;
                check_bounds(position, count, encoded.len())?;
                result.extend_from_slice(&encoded[position..position + count]);
                position += count;
            }
            COMMAND_SHUFFLE2 | COMMAND_SHUFFLE4 => {
                let count = decode_varint(encoded, &mut command_position)? as usize;
                check_bounds(position, count, encoded.len())?;
                let mut shuffled = encoded[position..position + count].to_vec();
                if command == COMMAND_SHUFFLE2 {
                    shuffle(&mut shuffled, 2);
                } else {
                    shuffle(&mut shuffled, 4);
                }
                result.extend_from_slice(&shuffled);
                position += count;
            }
            COMMAND_PREDICT => {
                check_bounds(command_position, 2, commands_end)?;
                let flags = encoded[command_position];
                command_position += 1;
                let width = usize::from(flags & 3) + 1;
                if width == 3 {
                    return Err(Error::InvalidCodestream("invalid ICC predictor width"));
                }
                let order = i32::from((flags & 12) >> 2);
                if order == 3 {
                    return Err(Error::InvalidCodestream("invalid ICC predictor order"));
                }
                let mut stride = width;
                if flags & 16 != 0 {
                    stride = decode_varint(encoded, &mut command_position)? as usize;
                    if stride < width {
                        return Err(Error::InvalidCodestream("invalid ICC predictor stride"));
                    }
                }
                if result.is_empty() || ((result.len() - 1) >> 2) < stride {
                    return Err(Error::InvalidCodestream("invalid ICC predictor stride"));
                }

                let count = decode_varint(encoded, &mut command_position)? as usize;
                check_bounds(position, count, encoded.len())?;
                let mut shuffled = encoded[position..position + count].to_vec();
                if width > 1 {
                    shuffle(&mut shuffled, width);
                }
                let start = result.len();
                for (index, value) in shuffled.iter().enumerate() {
                    let predicted =
                        linear_predict_icc_value(&result, start, index, stride, width, order);
                    result.push(predicted.wrapping_add(*value));
                }
                position += count;
            }
            COMMAND_XYZ => {
                append_tag(*XYZ_TAG, &mut result);
                result.extend_from_slice(&[0; 4]);
                check_bounds(position, 12, encoded.len())?;
                result.extend_from_slice(&encoded[position..position + 12]);
                position += 12;
            }
            value
                if (COMMAND_TYPE_START_FIRST
                    ..COMMAND_TYPE_START_FIRST + TYPE_STRINGS.len() as u8)
                    .contains(&value) =>
            {
                append_tag(
                    TYPE_STRINGS[usize::from(value - COMMAND_TYPE_START_FIRST)],
                    &mut result,
                );
                result.extend_from_slice(&[0; 4]);
            }
            _ => return Err(Error::InvalidCodestream("unknown ICC command")),
        }
    }

    if position != encoded.len() {
        return Err(Error::InvalidCodestream("unused ICC data"));
    }
    if result.len() != output_size {
        return Err(Error::InvalidCodestream("invalid ICC output size"));
    }
    validate_icc_size(&result)?;
    Ok(result)
}

fn validate_icc_size(icc: &[u8]) -> Result<()> {
    if icc.len() < 4 {
        return Err(Error::InvalidCodestream("ICC profile is too short"));
    }
    let declared = u32::from_be_bytes([icc[0], icc[1], icc[2], icc[3]]) as usize;
    if declared != icc.len() {
        return Err(Error::InvalidCodestream("ICC profile size mismatch"));
    }
    if icc.len() >= 40 && &icc[36..40] != b"acsp" {
        return Err(Error::InvalidCodestream(
            "ICC profile missing acsp signature",
        ));
    }
    Ok(())
}

fn decode_varint(input: &[u8], position: &mut usize) -> Result<u64> {
    let mut result = 0u64;
    for index in 0..10 {
        let byte = *input
            .get(*position)
            .ok_or(Error::InvalidCodestream("truncated ICC varint"))?;
        *position += 1;
        result |= u64::from(byte & 127) << (7 * index);
        if byte & 128 == 0 {
            return Ok(result);
        }
    }
    Err(Error::InvalidCodestream("ICC varint is too long"))
}

fn check_bounds(position: usize, count: usize, size: usize) -> Result<()> {
    let end = position
        .checked_add(count)
        .ok_or(Error::InvalidCodestream("ICC bounds overflow"))?;
    if end > size {
        return Err(Error::InvalidCodestream("ICC out of bounds"));
    }
    Ok(())
}

fn shuffle(data: &mut [u8], width: usize) {
    let height = data.len().div_ceil(width);
    let mut result = vec![0; data.len()];
    let mut start = 0;
    let mut input_index = 0;
    for output in &mut result {
        *output = data[input_index];
        input_index += height;
        if input_index >= data.len() {
            start += 1;
            input_index = start;
        }
    }
    data.copy_from_slice(&result);
}

fn initial_header_prediction(size: u32) -> [u8; ICC_HEADER_SIZE] {
    let mut copy = [0; ICC_HEADER_SIZE];
    copy[8] = 4;
    copy[12..16].copy_from_slice(b"mntr");
    copy[16..20].copy_from_slice(b"RGB ");
    copy[20..24].copy_from_slice(b"XYZ ");
    copy[36..40].copy_from_slice(b"acsp");
    copy[70] = 246;
    copy[71] = 214;
    copy[73] = 1;
    copy[78] = 211;
    copy[79] = 45;
    copy[0..4].copy_from_slice(&size.to_be_bytes());
    copy
}

fn predict_header(icc: &[u8], header: &mut [u8; ICC_HEADER_SIZE], position: usize) {
    if position == 8 && icc.len() >= 8 {
        header[80] = icc[4];
        header[81] = icc[5];
        header[82] = icc[6];
        header[83] = icc[7];
    }
    if position == 41 && icc.len() >= 41 {
        if icc[40] == b'A' {
            header[41] = b'P';
            header[42] = b'P';
            header[43] = b'L';
        }
        if icc[40] == b'M' {
            header[41] = b'S';
            header[42] = b'F';
            header[43] = b'T';
        }
    }
    if position == 42 && icc.len() >= 42 {
        if icc[40] == b'S' && icc[41] == b'G' {
            header[42] = b'I';
            header[43] = b' ';
        }
        if icc[40] == b'S' && icc[41] == b'U' {
            header[42] = b'N';
            header[43] = b'W';
        }
    }
}

fn linear_predict_icc_value(
    data: &[u8],
    start: usize,
    index: usize,
    stride: usize,
    width: usize,
    order: i32,
) -> u8 {
    let position = start + index;
    match width {
        1 => {
            let p1 = i32::from(data[position - stride]);
            let p2 = i32::from(data[position - stride * 2]);
            let p3 = i32::from(data[position - stride * 3]);
            predict_value(p1, p2, p3, order) as u8
        }
        2 => {
            let position = start + (index & !1);
            let p1 = u16::from_be_bytes([data[position - stride], data[position - stride + 1]]);
            let p2 =
                u16::from_be_bytes([data[position - stride * 2], data[position - stride * 2 + 1]]);
            let p3 =
                u16::from_be_bytes([data[position - stride * 3], data[position - stride * 3 + 1]]);
            let predicted =
                predict_value(i32::from(p1), i32::from(p2), i32::from(p3), order) as u16;
            if index & 1 != 0 {
                (predicted & 255) as u8
            } else {
                (predicted >> 8) as u8
            }
        }
        _ => {
            let position = start + (index & !3);
            let p1 = read_u32(data, position - stride);
            let p2 = read_u32(data, position - stride * 2);
            let p3 = read_u32(data, position - stride * 3);
            let predicted = predict_value_u32(p1, p2, p3, order);
            let shift_bytes = 3 - (index & 3);
            (predicted >> (shift_bytes * 8)) as u8
        }
    }
}

fn predict_value(p1: i32, p2: i32, p3: i32, order: i32) -> i32 {
    match order {
        0 => p1,
        1 => p1.wrapping_mul(2).wrapping_sub(p2),
        2 => p1
            .wrapping_mul(3)
            .wrapping_sub(p2.wrapping_mul(3))
            .wrapping_add(p3),
        _ => 0,
    }
}

fn predict_value_u32(p1: u32, p2: u32, p3: u32, order: i32) -> u32 {
    match order {
        0 => p1,
        1 => p1.wrapping_mul(2).wrapping_sub(p2),
        2 => p1
            .wrapping_mul(3)
            .wrapping_sub(p2.wrapping_mul(3))
            .wrapping_add(p3),
        _ => 0,
    }
}

fn read_u32(data: &[u8], position: usize) -> u32 {
    if position + 4 > data.len() {
        0
    } else {
        u32::from_be_bytes([
            data[position],
            data[position + 1],
            data[position + 2],
            data[position + 3],
        ])
    }
}

fn read_tag(data: &[u8], position: usize) -> Result<[u8; 4]> {
    check_bounds(position, 4, data.len())?;
    Ok([
        data[position],
        data[position + 1],
        data[position + 2],
        data[position + 3],
    ])
}

fn append_u32(value: u32, data: &mut Vec<u8>) {
    data.extend_from_slice(&value.to_be_bytes());
}

fn append_tag(tag: [u8; 4], data: &mut Vec<u8>) {
    data.extend_from_slice(&tag);
}

fn byte_kind1(byte: u8) -> usize {
    if byte.is_ascii_alphabetic() {
        return 0;
    }
    if byte.is_ascii_digit() || byte == b'.' || byte == b',' {
        return 1;
    }
    if byte == 0 {
        return 2;
    }
    if byte == 1 {
        return 3;
    }
    if byte < 16 {
        return 4;
    }
    if byte == 255 {
        return 6;
    }
    if byte > 240 {
        return 5;
    }
    7
}

fn byte_kind2(byte: u8) -> usize {
    if byte.is_ascii_alphabetic() {
        return 0;
    }
    if byte.is_ascii_digit() || byte == b'.' || byte == b',' {
        return 1;
    }
    if byte < 16 {
        return 2;
    }
    if byte > 240 {
        return 3;
    }
    4
}

fn icc_ans_context(index: usize, previous: u8, second_previous: u8) -> usize {
    if index <= ICC_HEADER_SIZE {
        0
    } else {
        1 + byte_kind1(previous) + byte_kind2(second_previous) * 8
    }
}

const COMMAND_TAG_UNKNOWN: u8 = 1;
const COMMAND_TAG_TRC: u8 = 2;
const COMMAND_TAG_XYZ: u8 = 3;
const COMMAND_TAG_STRING_FIRST: u8 = 4;
const COMMAND_INSERT: u8 = 1;
const COMMAND_SHUFFLE2: u8 = 2;
const COMMAND_SHUFFLE4: u8 = 3;
const COMMAND_PREDICT: u8 = 4;
const COMMAND_XYZ: u8 = 10;
const COMMAND_TYPE_START_FIRST: u8 = 16;
const FLAG_BIT_OFFSET: u8 = 64;
const FLAG_BIT_SIZE: u8 = 128;

const CPRT_TAG: &[u8; 4] = b"cprt";
const WTPT_TAG: &[u8; 4] = b"wtpt";
const BKPT_TAG: &[u8; 4] = b"bkpt";
const RXYZ_TAG: &[u8; 4] = b"rXYZ";
const GXYZ_TAG: &[u8; 4] = b"gXYZ";
const BXYZ_TAG: &[u8; 4] = b"bXYZ";
const KXYZ_TAG: &[u8; 4] = b"kXYZ";
const RTRC_TAG: &[u8; 4] = b"rTRC";
const GTRC_TAG: &[u8; 4] = b"gTRC";
const BTRC_TAG: &[u8; 4] = b"bTRC";
const KTRC_TAG: &[u8; 4] = b"kTRC";
const CHAD_TAG: &[u8; 4] = b"chad";
const DESC_TAG: &[u8; 4] = b"desc";
const CHRM_TAG: &[u8; 4] = b"chrm";
const DMND_TAG: &[u8; 4] = b"dmnd";
const DMDD_TAG: &[u8; 4] = b"dmdd";
const LUMI_TAG: &[u8; 4] = b"lumi";
const XYZ_TAG: &[u8; 4] = b"XYZ ";
const TEXT_TAG: &[u8; 4] = b"text";
const MLUC_TAG: &[u8; 4] = b"mluc";
const PARA_TAG: &[u8; 4] = b"para";
const CURV_TAG: &[u8; 4] = b"curv";
const SF32_TAG: &[u8; 4] = b"sf32";
const GBD_TAG: &[u8; 4] = b"gbd ";

const RXYZ_TAG_VALUE: [u8; 4] = *RXYZ_TAG;
const GXYZ_TAG_VALUE: [u8; 4] = *GXYZ_TAG;
const BXYZ_TAG_VALUE: [u8; 4] = *BXYZ_TAG;
const KXYZ_TAG_VALUE: [u8; 4] = *KXYZ_TAG;
const WTPT_TAG_VALUE: [u8; 4] = *WTPT_TAG;
const BKPT_TAG_VALUE: [u8; 4] = *BKPT_TAG;
const LUMI_TAG_VALUE: [u8; 4] = *LUMI_TAG;

const TAG_STRINGS: &[[u8; 4]] = &[
    *CPRT_TAG, *WTPT_TAG, *BKPT_TAG, *RXYZ_TAG, *GXYZ_TAG, *BXYZ_TAG, *KXYZ_TAG, *RTRC_TAG,
    *GTRC_TAG, *BTRC_TAG, *KTRC_TAG, *CHAD_TAG, *DESC_TAG, *CHRM_TAG, *DMND_TAG, *DMDD_TAG,
    *LUMI_TAG,
];

const TYPE_STRINGS: &[[u8; 4]] = &[
    *XYZ_TAG, *DESC_TAG, *TEXT_TAG, *MLUC_TAG, *PARA_TAG, *CURV_TAG, *SF32_TAG, *GBD_TAG,
];
