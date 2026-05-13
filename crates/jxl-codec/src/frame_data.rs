use crate::bitstream::{BitReader, bits_offset};
use crate::entropy::{AnsSymbolReader, decode_histograms};
use crate::error::{Error, Result};
use crate::frame::{FrameEncoding, FrameGroupLayout, FrameHeader, FrameSize};
use std::ops::Range;

const BLOCK_DIM: u32 = 8;
const PERMUTATION_CONTEXTS: usize = 8;
const MAX_TOC_ENTRIES: u32 = 65_536;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameData {
    pub toc: FrameToc,
    pub sections: Vec<FrameSection>,
    pub payload_start_offset: usize,
    pub payload_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameToc {
    pub entries: Vec<TocEntry>,
    pub has_permutation: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TocEntry {
    pub logical_id: usize,
    pub physical_index: usize,
    pub kind: FrameSectionKind,
    pub offset: usize,
    pub size: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameSection {
    pub logical_id: usize,
    pub physical_index: usize,
    pub kind: FrameSectionKind,
    pub codestream_offset: usize,
    pub size: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameSectionKind {
    Combined,
    DcGlobal,
    DcGroup { group: usize },
    AcGlobal,
    AcGroup { pass: usize, group: usize },
}

pub fn section_payload<'a>(codestream: &'a [u8], section: &FrameSection) -> Result<&'a [u8]> {
    codestream
        .get(section_payload_range(section)?)
        .ok_or(Error::InvalidCodestream("frame section outside codestream"))
}

pub fn section_payload_range(section: &FrameSection) -> Result<Range<usize>> {
    let start = section.codestream_offset;
    let end = start
        .checked_add(section.size as usize)
        .ok_or(Error::InvalidCodestream("frame section range overflow"))?;
    Ok(start..end)
}

pub fn read_frame_data(
    reader: &mut BitReader<'_>,
    frame_header: &FrameHeader,
    codestream_base_offset: usize,
) -> Result<FrameData> {
    let toc_entries = num_toc_entries(
        frame_header.group_layout.num_groups as usize,
        frame_header.group_layout.num_dc_groups as usize,
        frame_header.passes.num_passes as usize,
    )?;
    let toc = read_toc(reader, toc_entries, &frame_header.group_layout)?;

    if !reader.is_byte_aligned() {
        return Err(Error::InvalidCodestream(
            "frame payload is not byte-aligned",
        ));
    }
    let payload_start_offset = codestream_base_offset + reader.bytes_consumed_floor();
    let mut sections = Vec::with_capacity(toc.entries.len());
    for entry in &toc.entries {
        let offset = usize::try_from(entry.size)
            .ok()
            .and_then(|_| payload_start_offset.checked_add(entry.offset))
            .ok_or(Error::InvalidCodestream("frame section offset overflow"))?;
        sections.push(FrameSection {
            logical_id: entry.logical_id,
            physical_index: entry.physical_index,
            kind: entry.kind,
            codestream_offset: offset,
            size: entry.size,
        });
    }

    let payload_size = sections.iter().try_fold(0usize, |sum, section| {
        sum.checked_add(section.size as usize)
            .ok_or(Error::InvalidCodestream("frame payload size overflow"))
    })?;
    reader.skip_aligned_bytes(payload_size)?;

    Ok(FrameData {
        toc,
        sections,
        payload_start_offset,
        payload_size,
    })
}

pub fn num_toc_entries(
    num_groups: usize,
    num_dc_groups: usize,
    num_passes: usize,
) -> Result<usize> {
    if num_groups == 1 && num_passes == 1 {
        Ok(1)
    } else {
        let ac_entries = num_groups
            .checked_mul(num_passes)
            .ok_or(Error::InvalidCodestream("TOC entry count overflow"))?;
        2usize
            .checked_add(num_dc_groups)
            .and_then(|entries| entries.checked_add(ac_entries))
            .ok_or(Error::InvalidCodestream("TOC entry count overflow"))
    }
}

pub fn compute_frame_group_layout(header: &FrameHeader) -> Result<FrameGroupLayout> {
    compute_group_layout(
        header.frame_size,
        header.dc_level,
        header.group_size_shift,
        header.upsampling,
        header.chroma_subsampling.max_h_shift,
        header.chroma_subsampling.max_v_shift,
        header.encoding == FrameEncoding::Modular,
    )
}

pub(crate) fn compute_group_layout(
    mut frame_size: FrameSize,
    dc_level: u32,
    group_size_shift: u32,
    upsampling: u32,
    max_h_shift: u8,
    max_v_shift: u8,
    modular_mode: bool,
) -> Result<FrameGroupLayout> {
    if dc_level != 0 {
        let divisor = 1u32 << (3 * dc_level);
        frame_size.width = frame_size.width.div_ceil(divisor);
        frame_size.height = frame_size.height.div_ceil(divisor);
    }

    let group_dim = 128 << group_size_shift;
    let xsize = frame_size.width.div_ceil(upsampling);
    let ysize = frame_size.height.div_ceil(upsampling);
    let mut xsize_blocks = xsize.div_ceil(BLOCK_DIM << max_h_shift) << max_h_shift;
    let mut ysize_blocks = ysize.div_ceil(BLOCK_DIM << max_v_shift) << max_v_shift;
    let mut xsize_padded = xsize_blocks * BLOCK_DIM;
    let mut ysize_padded = ysize_blocks * BLOCK_DIM;
    if modular_mode {
        xsize_padded = xsize;
        ysize_padded = ysize;
    }
    if xsize_padded == 0 {
        xsize_blocks = 0;
    }
    if ysize_padded == 0 {
        ysize_blocks = 0;
    }

    let groups_x = xsize.div_ceil(group_dim);
    let groups_y = ysize.div_ceil(group_dim);
    let dc_groups_x = xsize_blocks.div_ceil(group_dim);
    let dc_groups_y = ysize_blocks.div_ceil(group_dim);
    let num_groups = checked_group_count(groups_x, groups_y, "group count overflow")?;
    if num_groups > MAX_TOC_ENTRIES {
        return Err(Error::InvalidCodestream("too many frame groups"));
    }
    let num_dc_groups = checked_group_count(dc_groups_x, dc_groups_y, "DC group count overflow")?;
    if num_dc_groups > MAX_TOC_ENTRIES {
        return Err(Error::InvalidCodestream("too many frame DC groups"));
    }
    Ok(FrameGroupLayout {
        group_dim,
        groups_x,
        groups_y,
        num_groups,
        dc_group_dim: group_dim * BLOCK_DIM,
        dc_groups_x,
        dc_groups_y,
        num_dc_groups,
    })
}

fn checked_group_count(x: u32, y: u32, overflow_msg: &'static str) -> Result<u32> {
    x.checked_mul(y)
        .ok_or(Error::InvalidCodestream(overflow_msg))
}

fn read_toc(
    reader: &mut BitReader<'_>,
    toc_entries: usize,
    group_layout: &FrameGroupLayout,
) -> Result<FrameToc> {
    if toc_entries == 0 || toc_entries > MAX_TOC_ENTRIES as usize {
        return Err(Error::InvalidCodestream("invalid TOC entry count"));
    }

    let permutation = if reader.read_bool()? {
        Some(decode_permutation(reader, toc_entries)?)
    } else {
        None
    };
    reader.jump_to_byte_boundary()?;

    let mut sizes = Vec::with_capacity(toc_entries);
    for _ in 0..toc_entries {
        sizes.push(reader.read_u32_selector(
            bits_offset(10, 0),
            bits_offset(14, 1024),
            bits_offset(22, 17_408),
            bits_offset(30, 4_211_712),
        )?);
    }
    reader.jump_to_byte_boundary()?;

    let logical_ids = physical_to_logical_ids(permutation.as_deref(), toc_entries)?;
    let mut offset = 0usize;
    let mut entries = Vec::with_capacity(toc_entries);
    for (physical_index, size) in sizes.into_iter().enumerate() {
        let logical_id = logical_ids[physical_index];
        let kind = section_kind(
            logical_id,
            group_layout.num_groups as usize,
            group_layout.num_dc_groups as usize,
            toc_entries,
        )?;
        entries.push(TocEntry {
            logical_id,
            physical_index,
            kind,
            offset,
            size,
        });
        offset = offset
            .checked_add(size as usize)
            .ok_or(Error::InvalidCodestream("TOC offset overflow"))?;
    }

    Ok(FrameToc {
        entries,
        has_permutation: permutation.is_some(),
    })
}

fn physical_to_logical_ids(
    permutation: Option<&[usize]>,
    toc_entries: usize,
) -> Result<Vec<usize>> {
    if let Some(permutation) = permutation {
        if permutation.len() != toc_entries {
            return Err(Error::InvalidCodestream("invalid TOC permutation length"));
        }
        let mut logical_ids = vec![0; toc_entries];
        for (logical_id, &physical_index) in permutation.iter().enumerate() {
            if physical_index >= toc_entries {
                return Err(Error::InvalidCodestream("invalid TOC permutation entry"));
            }
            logical_ids[physical_index] = logical_id;
        }
        Ok(logical_ids)
    } else {
        Ok((0..toc_entries).collect())
    }
}

fn section_kind(
    logical_id: usize,
    num_groups: usize,
    num_dc_groups: usize,
    toc_entries: usize,
) -> Result<FrameSectionKind> {
    if toc_entries == 1 {
        return if logical_id == 0 {
            Ok(FrameSectionKind::Combined)
        } else {
            Err(Error::InvalidCodestream("invalid single-section TOC id"))
        };
    }

    let ac_global_index = num_dc_groups + 1;
    if logical_id == 0 {
        Ok(FrameSectionKind::DcGlobal)
    } else if logical_id < ac_global_index {
        Ok(FrameSectionKind::DcGroup {
            group: logical_id - 1,
        })
    } else if logical_id == ac_global_index {
        Ok(FrameSectionKind::AcGlobal)
    } else {
        let ac_index = logical_id - ac_global_index - 1;
        let pass = ac_index / num_groups;
        let group = ac_index % num_groups;
        Ok(FrameSectionKind::AcGroup { pass, group })
    }
}

fn decode_permutation(reader: &mut BitReader<'_>, size: usize) -> Result<Vec<usize>> {
    let (code, context_map) = decode_histograms(reader, PERMUTATION_CONTEXTS, false)?;
    let mut symbol_reader = AnsSymbolReader::new(code, reader, 0)?;
    let mut lehmer = vec![0u32; size];
    let end =
        symbol_reader.read_hybrid_uint(coeff_order_context(size), reader, &context_map)? as usize;
    if end > size {
        return Err(Error::InvalidCodestream("invalid TOC permutation size"));
    }
    let mut last = 0usize;
    for (index, value) in lehmer.iter_mut().enumerate().take(end) {
        let code =
            symbol_reader.read_hybrid_uint(coeff_order_context(last), reader, &context_map)?;
        if code as usize >= size - index {
            return Err(Error::InvalidCodestream("invalid TOC Lehmer code"));
        }
        *value = code;
        last = code as usize;
    }
    if !symbol_reader.check_final_state() {
        return Err(Error::InvalidCodestream(
            "invalid TOC permutation ANS state",
        ));
    }
    decode_lehmer_code(&lehmer)
}

fn coeff_order_context(value: usize) -> usize {
    if value == 0 {
        0
    } else {
        (usize::BITS as usize - value.leading_zeros() as usize).min(PERMUTATION_CONTEXTS - 1)
    }
}

fn decode_lehmer_code(code: &[u32]) -> Result<Vec<usize>> {
    let size = code.len();
    if size == 0 {
        return Err(Error::InvalidCodestream("empty Lehmer code"));
    }
    let log2_size = usize::BITS as usize - (size - 1).leading_zeros() as usize;
    let padded_size = 1usize << log2_size;
    let mut tree = vec![0u32; padded_size];
    for (index, value) in tree.iter_mut().enumerate() {
        *value = value_of_lowest_one_bit(index + 1) as u32;
    }

    let mut permutation = vec![0; size];
    for (index, &lehmer) in code.iter().enumerate() {
        if lehmer as usize + index >= size {
            return Err(Error::InvalidCodestream("invalid Lehmer code"));
        }
        let mut rank = lehmer + 1;
        let mut bit = padded_size;
        let mut next = 0usize;
        for _ in 0..=log2_size {
            let candidate = next + bit;
            bit >>= 1;
            if tree[candidate - 1] < rank {
                next = candidate;
                rank -= tree[candidate - 1];
            }
        }
        permutation[index] = next;

        next += 1;
        while next <= padded_size {
            tree[next - 1] -= 1;
            next += value_of_lowest_one_bit(next);
        }
    }
    Ok(permutation)
}

fn value_of_lowest_one_bit(value: usize) -> usize {
    value & value.wrapping_neg()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_bounded_section_payload() {
        let codestream = [0, 1, 2, 3, 4, 5];
        let section = frame_section(2, 3);

        assert_eq!(section_payload_range(&section).unwrap(), 2..5);
        assert_eq!(section_payload(&codestream, &section).unwrap(), &[2, 3, 4]);
    }

    #[test]
    fn rejects_section_range_overflow() {
        let section = frame_section(usize::MAX - 1, 4);

        assert_eq!(
            section_payload_range(&section).unwrap_err(),
            Error::InvalidCodestream("frame section range overflow")
        );
    }

    #[test]
    fn rejects_section_payload_outside_codestream() {
        let codestream = [0, 1, 2, 3];
        let section = frame_section(2, 3);

        assert_eq!(
            section_payload(&codestream, &section).unwrap_err(),
            Error::InvalidCodestream("frame section outside codestream")
        );
    }

    #[test]
    fn rejects_group_count_overflow_before_toc() {
        assert_eq!(
            compute_group_layout(
                FrameSize {
                    width: 1 << 30,
                    height: 1 << 30,
                },
                0,
                1,
                1,
                0,
                0,
                false,
            )
            .unwrap_err(),
            Error::InvalidCodestream("group count overflow")
        );
    }

    #[test]
    fn rejects_too_many_frame_groups_before_toc() {
        assert_eq!(
            compute_group_layout(
                FrameSize {
                    width: 65_537 * 256,
                    height: 1,
                },
                0,
                1,
                1,
                0,
                0,
                false,
            )
            .unwrap_err(),
            Error::InvalidCodestream("too many frame groups")
        );
    }

    #[test]
    fn rejects_toc_entry_count_overflow() {
        assert_eq!(
            num_toc_entries(usize::MAX, 1, 2).unwrap_err(),
            Error::InvalidCodestream("TOC entry count overflow")
        );
    }

    fn frame_section(codestream_offset: usize, size: u32) -> FrameSection {
        FrameSection {
            logical_id: 0,
            physical_index: 0,
            kind: FrameSectionKind::Combined,
            codestream_offset,
            size,
        }
    }
}
