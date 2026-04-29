use crate::bitstream::{BitReader, bits_offset, val};
use crate::error::{Error, Result};

const ANS_LOG_TAB_SIZE: usize = 12;
const ANS_TAB_SIZE: usize = 1 << ANS_LOG_TAB_SIZE;
const ANS_TAB_MASK: u32 = (ANS_TAB_SIZE - 1) as u32;
const ANS_SIGNATURE: u32 = 0x13;
const ANS_MAX_ALPHABET_SIZE: usize = 256;
const PREFIX_MAX_BITS: usize = 15;
const PREFIX_MAX_ALPHABET_SIZE: usize = 1 << PREFIX_MAX_BITS;
const HUFFMAN_TABLE_BITS: usize = 8;
const WINDOW_SIZE: usize = 1 << 20;
const WINDOW_MASK: usize = WINDOW_SIZE - 1;
const NUM_SPECIAL_DISTANCES: usize = 120;
const SPECIAL_DISTANCES: [(i32, i32); NUM_SPECIAL_DISTANCES] = [
    (0, 1),
    (1, 0),
    (1, 1),
    (-1, 1),
    (0, 2),
    (2, 0),
    (1, 2),
    (-1, 2),
    (2, 1),
    (-2, 1),
    (2, 2),
    (-2, 2),
    (0, 3),
    (3, 0),
    (1, 3),
    (-1, 3),
    (3, 1),
    (-3, 1),
    (2, 3),
    (-2, 3),
    (3, 2),
    (-3, 2),
    (0, 4),
    (4, 0),
    (1, 4),
    (-1, 4),
    (4, 1),
    (-4, 1),
    (3, 3),
    (-3, 3),
    (2, 4),
    (-2, 4),
    (4, 2),
    (-4, 2),
    (0, 5),
    (3, 4),
    (-3, 4),
    (4, 3),
    (-4, 3),
    (5, 0),
    (1, 5),
    (-1, 5),
    (5, 1),
    (-5, 1),
    (2, 5),
    (-2, 5),
    (5, 2),
    (-5, 2),
    (4, 4),
    (-4, 4),
    (3, 5),
    (-3, 5),
    (5, 3),
    (-5, 3),
    (0, 6),
    (6, 0),
    (1, 6),
    (-1, 6),
    (6, 1),
    (-6, 1),
    (2, 6),
    (-2, 6),
    (6, 2),
    (-6, 2),
    (4, 5),
    (-4, 5),
    (5, 4),
    (-5, 4),
    (3, 6),
    (-3, 6),
    (6, 3),
    (-6, 3),
    (0, 7),
    (7, 0),
    (1, 7),
    (-1, 7),
    (5, 5),
    (-5, 5),
    (7, 1),
    (-7, 1),
    (4, 6),
    (-4, 6),
    (6, 4),
    (-6, 4),
    (2, 7),
    (-2, 7),
    (7, 2),
    (-7, 2),
    (3, 7),
    (-3, 7),
    (7, 3),
    (-7, 3),
    (5, 6),
    (-5, 6),
    (6, 5),
    (-6, 5),
    (8, 0),
    (4, 7),
    (-4, 7),
    (7, 4),
    (-7, 4),
    (8, 1),
    (8, 2),
    (6, 6),
    (-6, 6),
    (8, 3),
    (5, 7),
    (-5, 7),
    (7, 5),
    (-7, 5),
    (8, 4),
    (6, 7),
    (-6, 7),
    (7, 6),
    (-7, 6),
    (8, 5),
    (7, 7),
    (-7, 7),
    (8, 6),
    (8, 7),
];

#[derive(Debug, Clone)]
pub(crate) struct HybridUintConfig {
    split_exponent: u32,
    split_token: u32,
    msb_in_token: u32,
    lsb_in_token: u32,
}

impl HybridUintConfig {
    fn new(split_exponent: u32, msb_in_token: u32, lsb_in_token: u32) -> Self {
        Self {
            split_exponent,
            split_token: 1 << split_exponent,
            msb_in_token,
            lsb_in_token,
        }
    }

    fn decode_token(&self, token: usize, reader: &mut BitReader<'_>) -> Result<u32> {
        let split_token = self.split_token as usize;
        if token < split_token {
            return Ok(token as u32);
        }

        let nbits = self.split_exponent - (self.msb_in_token + self.lsb_in_token)
            + (((token - split_token) as u32) >> (self.msb_in_token + self.lsb_in_token));
        if nbits > 29 {
            return Err(Error::InvalidCodestream("invalid hybrid uint token"));
        }

        let low = token as u32 & ((1 << self.lsb_in_token) - 1);
        let shifted_token = (token as u32) >> self.lsb_in_token;
        let bits = reader.read_bits(nbits as usize)? as u32;
        Ok(
            (((((1 << self.msb_in_token) | (shifted_token & ((1 << self.msb_in_token) - 1)))
                << nbits)
                | bits)
                << self.lsb_in_token)
                | low,
        )
    }
}

impl Default for HybridUintConfig {
    fn default() -> Self {
        Self::new(4, 2, 0)
    }
}

#[derive(Debug, Clone)]
struct Lz77Params {
    enabled: bool,
    min_symbol: u32,
    min_length: u32,
    length_config: HybridUintConfig,
    distance_context: usize,
}

impl Default for Lz77Params {
    fn default() -> Self {
        Self {
            enabled: false,
            min_symbol: 224,
            min_length: 3,
            length_config: HybridUintConfig::new(0, 0, 0),
            distance_context: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AnsCode {
    alias_tables: Vec<AliasEntry>,
    huffman_data: Vec<HuffmanDecodingData>,
    uint_config: Vec<HybridUintConfig>,
    use_prefix_code: bool,
    log_alpha_size: usize,
    lz77: Lz77Params,
}

#[derive(Debug, Clone, Copy, Default)]
struct AliasEntry {
    cutoff: u32,
    right_value: u32,
    freq0: u32,
    offsets1: u32,
    freq1_xor_freq0: u32,
}

#[derive(Debug, Clone, Copy)]
struct AliasSymbol {
    value: usize,
    offset: u32,
    freq: u32,
}

#[derive(Debug, Clone, Copy, Default)]
struct HuffmanCode {
    bits: u8,
    value: u16,
}

#[derive(Debug, Clone, Default)]
struct HuffmanDecodingData {
    table: Vec<HuffmanCode>,
}

#[derive(Debug, Clone)]
pub(crate) struct AnsSymbolReader {
    code: AnsCode,
    state: u32,
    log_entry_size: usize,
    entry_size_minus_1: u32,
    lz77_window: Option<Vec<u32>>,
    num_decoded: usize,
    num_to_copy: usize,
    copy_pos: usize,
    lz77_threshold: usize,
    lz77_min_length: usize,
    special_distances: [usize; NUM_SPECIAL_DISTANCES],
    num_special_distances: usize,
}

impl AnsSymbolReader {
    pub(crate) fn new(
        code: AnsCode,
        reader: &mut BitReader<'_>,
        distance_multiplier: usize,
    ) -> Result<Self> {
        let state = if code.use_prefix_code {
            ANS_SIGNATURE << 16
        } else {
            reader.read_bits(32)? as u32
        };
        let log_entry_size = ANS_LOG_TAB_SIZE.saturating_sub(code.log_alpha_size);
        let entry_size_minus_1 = if code.use_prefix_code {
            0
        } else {
            (1 << log_entry_size) - 1
        };

        let lz77_window = code.lz77.enabled.then(|| vec![0; WINDOW_SIZE]);
        let lz77_threshold = if code.lz77.enabled {
            code.lz77.min_symbol as usize
        } else {
            usize::MAX
        };
        let lz77_min_length = code.lz77.min_length as usize;
        let num_special_distances = if distance_multiplier == 0 {
            0
        } else {
            NUM_SPECIAL_DISTANCES
        };
        let mut special_distances = [0; NUM_SPECIAL_DISTANCES];
        for (index, (offset, multiplier)) in SPECIAL_DISTANCES.iter().copied().enumerate() {
            let distance = i64::from(offset) + distance_multiplier as i64 * i64::from(multiplier);
            special_distances[index] = distance.max(1) as usize;
        }

        Ok(Self {
            code,
            state,
            log_entry_size,
            entry_size_minus_1,
            lz77_window,
            num_decoded: 0,
            num_to_copy: 0,
            copy_pos: 0,
            lz77_threshold,
            lz77_min_length,
            special_distances,
            num_special_distances,
        })
    }

    pub(crate) fn read_hybrid_uint(
        &mut self,
        context: usize,
        reader: &mut BitReader<'_>,
        context_map: &[u8],
    ) -> Result<u32> {
        let clustered_context = usize::from(
            *context_map
                .get(context)
                .ok_or(Error::InvalidCodestream("invalid entropy context"))?,
        );
        self.read_hybrid_uint_clustered(clustered_context, reader)
    }

    pub(crate) fn read_hybrid_uint_clustered(
        &mut self,
        context: usize,
        reader: &mut BitReader<'_>,
    ) -> Result<u32> {
        if self.code.lz77.enabled && self.num_to_copy > 0 {
            let value = self.copy_from_lz77_window();
            return Ok(value);
        }

        let token = self.read_symbol(context, reader)?;
        if self.code.lz77.enabled && token >= self.lz77_threshold {
            self.num_to_copy =
                self.code
                    .lz77
                    .length_config
                    .decode_token(token - self.lz77_threshold, reader)? as usize
                    + self.lz77_min_length;
            let distance_context = self.code.lz77.distance_context;
            let distance_token = self.read_symbol(distance_context, reader)?;
            let mut distance = self.code.uint_config[distance_context]
                .decode_token(distance_token, reader)? as usize;
            if distance < self.num_special_distances {
                distance = self.special_distances[distance];
            } else {
                distance = distance + 1 - self.num_special_distances;
            }
            distance = distance.min(self.num_decoded).min(WINDOW_SIZE);
            self.copy_pos = self.num_decoded.saturating_sub(distance);
            if distance == 0
                && let Some(window) = self.lz77_window.as_mut()
            {
                let to_fill = self.num_to_copy.min(WINDOW_SIZE);
                window[..to_fill].fill(0);
            }
            if self.num_to_copy < self.lz77_min_length {
                return Err(Error::InvalidCodestream("invalid LZ77 copy length"));
            }
            return Ok(self.copy_from_lz77_window());
        }

        let value = self.code.uint_config[context].decode_token(token, reader)?;
        if let Some(window) = self.lz77_window.as_mut() {
            window[self.num_decoded & WINDOW_MASK] = value;
            self.num_decoded += 1;
        }
        Ok(value)
    }

    fn copy_from_lz77_window(&mut self) -> u32 {
        let window = self
            .lz77_window
            .as_mut()
            .expect("LZ77 copy requested without a window");
        let value = window[self.copy_pos & WINDOW_MASK];
        self.copy_pos += 1;
        self.num_to_copy -= 1;
        window[self.num_decoded & WINDOW_MASK] = value;
        self.num_decoded += 1;
        value
    }

    fn read_symbol(&mut self, context: usize, reader: &mut BitReader<'_>) -> Result<usize> {
        if self.code.use_prefix_code {
            self.read_symbol_huffman(context, reader)
        } else {
            self.read_symbol_ans(context, reader)
        }
    }

    fn read_symbol_huffman(&mut self, context: usize, reader: &mut BitReader<'_>) -> Result<usize> {
        let table = self
            .code
            .huffman_data
            .get(context)
            .ok_or(Error::InvalidCodestream("invalid Huffman context"))?;
        table.read_symbol(reader).map(usize::from)
    }

    fn read_symbol_ans(&mut self, context: usize, reader: &mut BitReader<'_>) -> Result<usize> {
        let res = self.state & ANS_TAB_MASK;
        let table_offset = context
            .checked_shl(self.code.log_alpha_size as u32)
            .ok_or(Error::InvalidCodestream("entropy table offset overflow"))?;
        let table = self
            .code
            .alias_tables
            .get(table_offset..table_offset + (1 << self.code.log_alpha_size))
            .ok_or(Error::InvalidCodestream("invalid ANS context"))?;
        let symbol = lookup_alias(table, res, self.log_entry_size, self.entry_size_minus_1);
        self.state = symbol.freq * (self.state >> ANS_LOG_TAB_SIZE) + symbol.offset;
        if self.state < (1 << 16) {
            self.state = (self.state << 16) | reader.read_bits(16)? as u32;
        }
        Ok(symbol.value)
    }

    pub(crate) fn check_final_state(&self) -> bool {
        self.state == (ANS_SIGNATURE << 16)
    }
}

pub(crate) fn decode_histograms(
    reader: &mut BitReader<'_>,
    num_contexts: usize,
    disallow_lz77: bool,
) -> Result<(AnsCode, Vec<u8>)> {
    let mut lz77 = read_lz77_params(reader)?;
    let mut context_count = num_contexts;
    if lz77.enabled {
        context_count += 1;
        lz77.length_config = decode_uint_config(8, reader)?;
    }
    if lz77.enabled && disallow_lz77 {
        return Err(Error::InvalidCodestream("LZ77 is not allowed here"));
    }

    let mut context_map = vec![0; context_count];
    let num_histograms = if context_count > 1 {
        decode_context_map(reader, &mut context_map)?
    } else {
        1
    };
    lz77.distance_context = usize::from(*context_map.last().unwrap_or(&0));

    let use_prefix_code = reader.read_bool()?;
    let log_alpha_size = if use_prefix_code {
        PREFIX_MAX_BITS
    } else {
        reader.read_bits(2)? as usize + 5
    };
    let mut uint_config = Vec::with_capacity(num_histograms);
    for _ in 0..num_histograms {
        uint_config.push(decode_uint_config(log_alpha_size, reader)?);
    }

    let max_alphabet_size = 1 << log_alpha_size;
    if use_prefix_code && max_alphabet_size > PREFIX_MAX_ALPHABET_SIZE {
        return Err(Error::InvalidCodestream("prefix alphabet is too large"));
    }
    if !use_prefix_code && max_alphabet_size > ANS_MAX_ALPHABET_SIZE {
        return Err(Error::InvalidCodestream("ANS alphabet is too large"));
    }

    let (alias_tables, huffman_data) = if use_prefix_code {
        let mut data = Vec::with_capacity(num_histograms);
        let mut alphabet_sizes = Vec::with_capacity(num_histograms);
        for _ in 0..num_histograms {
            let alphabet_size = decode_var_len_uint16(reader)? + 1;
            if alphabet_size > max_alphabet_size {
                return Err(Error::InvalidCodestream("prefix alphabet is too large"));
            }
            alphabet_sizes.push(alphabet_size);
        }
        for alphabet_size in alphabet_sizes {
            data.push(HuffmanDecodingData::read_from_bitstream(
                alphabet_size,
                reader,
            )?);
        }
        (Vec::new(), data)
    } else {
        let table_size = 1 << log_alpha_size;
        let mut tables = vec![AliasEntry::default(); num_histograms * table_size];
        for context in 0..num_histograms {
            let mut counts = read_histogram(ANS_LOG_TAB_SIZE, reader)?;
            if counts.len() > max_alphabet_size {
                return Err(Error::InvalidCodestream("ANS alphabet is too large"));
            }
            while counts.last() == Some(&0) {
                counts.pop();
            }
            init_alias_table(
                counts,
                ANS_LOG_TAB_SIZE,
                log_alpha_size,
                &mut tables[context * table_size..(context + 1) * table_size],
            )?;
        }
        (tables, Vec::new())
    };

    Ok((
        AnsCode {
            alias_tables,
            huffman_data,
            uint_config,
            use_prefix_code,
            log_alpha_size,
            lz77,
        },
        context_map,
    ))
}

fn read_lz77_params(reader: &mut BitReader<'_>) -> Result<Lz77Params> {
    let enabled = reader.read_bool()?;
    if !enabled {
        return Ok(Lz77Params::default());
    }

    Ok(Lz77Params {
        enabled,
        min_symbol: reader.read_u32_selector(val(224), val(512), val(4096), bits_offset(15, 8))?,
        min_length: reader.read_u32_selector(
            val(3),
            val(4),
            bits_offset(2, 5),
            bits_offset(8, 9),
        )?,
        ..Lz77Params::default()
    })
}

pub(crate) fn decode_context_map(
    reader: &mut BitReader<'_>,
    context_map: &mut [u8],
) -> Result<usize> {
    let is_simple = reader.read_bool()?;
    if is_simple {
        let bits_per_entry = reader.read_bits(2)? as usize;
        if bits_per_entry == 0 {
            context_map.fill(0);
        } else {
            for entry in context_map.iter_mut() {
                *entry = reader.read_bits(bits_per_entry)? as u8;
            }
        }
    } else {
        let use_mtf = reader.read_bool()?;
        let (code, nested_context_map) = decode_histograms(reader, 1, context_map.len() <= 2)?;
        let mut symbol_reader = AnsSymbolReader::new(code, reader, 0)?;
        let mut max_symbol = 0;
        for entry in context_map.iter_mut() {
            let symbol = symbol_reader.read_hybrid_uint(0, reader, &nested_context_map)?;
            if symbol >= 256 {
                return Err(Error::InvalidCodestream("invalid context-map cluster"));
            }
            max_symbol = max_symbol.max(symbol as usize);
            *entry = symbol as u8;
        }
        if !symbol_reader.check_final_state() {
            return Err(Error::InvalidCodestream("invalid context-map ANS state"));
        }
        if use_mtf {
            inverse_move_to_front(context_map);
        }
        if max_symbol >= 256 {
            return Err(Error::InvalidCodestream("invalid context-map cluster"));
        }
    }

    let num_histograms = usize::from(*context_map.iter().max().unwrap_or(&0)) + 1;
    verify_context_map(context_map, num_histograms)?;
    Ok(num_histograms)
}

fn verify_context_map(context_map: &[u8], num_histograms: usize) -> Result<()> {
    let mut seen = vec![false; num_histograms];
    let mut num_found = 0;
    for &entry in context_map {
        let entry = usize::from(entry);
        if entry >= num_histograms {
            return Err(Error::InvalidCodestream("invalid context-map histogram"));
        }
        if !seen[entry] {
            seen[entry] = true;
            num_found += 1;
        }
    }
    if num_found != num_histograms {
        return Err(Error::InvalidCodestream("incomplete context map"));
    }
    Ok(())
}

fn inverse_move_to_front(data: &mut [u8]) {
    let mut table = [0u8; 256];
    for (index, item) in table.iter_mut().enumerate() {
        *item = index as u8;
    }
    for value in data {
        let index = usize::from(*value);
        let decoded = table[index];
        for i in (1..=index).rev() {
            table[i] = table[i - 1];
        }
        table[0] = decoded;
        *value = decoded;
    }
}

fn decode_uint_config(
    log_alpha_size: usize,
    reader: &mut BitReader<'_>,
) -> Result<HybridUintConfig> {
    let split_exponent = reader.read_bits(ceil_log2_nonzero(log_alpha_size + 1))? as u32;
    let mut msb_in_token = 0;
    let mut lsb_in_token = 0;
    if split_exponent != log_alpha_size as u32 {
        let nbits = ceil_log2_nonzero(split_exponent as usize + 1);
        msb_in_token = reader.read_bits(nbits)? as u32;
        if msb_in_token > split_exponent {
            return Err(Error::InvalidCodestream("invalid hybrid uint config"));
        }
        let nbits = ceil_log2_nonzero((split_exponent - msb_in_token) as usize + 1);
        lsb_in_token = reader.read_bits(nbits)? as u32;
    }
    if lsb_in_token + msb_in_token > split_exponent {
        return Err(Error::InvalidCodestream("invalid hybrid uint config"));
    }
    Ok(HybridUintConfig::new(
        split_exponent,
        msb_in_token,
        lsb_in_token,
    ))
}

fn read_histogram(precision_bits: usize, reader: &mut BitReader<'_>) -> Result<Vec<i32>> {
    let range = 1i32 << precision_bits;
    if reader.read_bool()? {
        let num_symbols = reader.read_bits(1)? as usize + 1;
        let mut symbols = Vec::with_capacity(num_symbols);
        let mut max_symbol = 0;
        for _ in 0..num_symbols {
            let symbol = decode_var_len_uint8(reader)?;
            max_symbol = max_symbol.max(symbol);
            symbols.push(symbol);
        }
        let mut counts = vec![0; max_symbol + 1];
        if num_symbols == 1 {
            counts[symbols[0]] = range;
        } else {
            if symbols[0] == symbols[1] {
                return Err(Error::InvalidCodestream(
                    "duplicate simple histogram symbol",
                ));
            }
            counts[symbols[0]] = reader.read_bits(precision_bits)? as i32;
            counts[symbols[1]] = range - counts[symbols[0]];
        }
        return Ok(counts);
    }

    if reader.read_bool()? {
        let alphabet_size = decode_var_len_uint8(reader)? + 1;
        if alphabet_size > range as usize {
            return Err(Error::InvalidCodestream(
                "flat histogram alphabet is too large",
            ));
        }
        return Ok(create_flat_histogram(alphabet_size, range));
    }

    let upper_bound_log = floor_log2_nonzero(ANS_LOG_TAB_SIZE + 1);
    let mut log = 0;
    while log < upper_bound_log {
        if !reader.read_bool()? {
            break;
        }
        log += 1;
    }
    let shift = (reader.read_bits(log)? as u32 | (1 << log)) - 1;
    if shift > ANS_LOG_TAB_SIZE as u32 + 1 {
        return Err(Error::InvalidCodestream("invalid histogram shift"));
    }

    let length = decode_var_len_uint8(reader)? + 3;
    let mut counts = vec![0; length];
    let mut logcounts = vec![0i32; length];
    let mut same = vec![0; length];
    let mut omit_log = -1;
    let mut omit_pos = None;
    let mut i = 0;
    while i < length {
        let idx = reader.peek_bits(7)? as usize;
        let (bits, value) = HISTOGRAM_LOGCOUNT_HUFFMAN[idx];
        reader.skip_bits(bits as usize)?;
        logcounts[i] = i32::from(value) - 1;
        if logcounts[i] == ANS_LOG_TAB_SIZE as i32 {
            let rle_length = decode_var_len_uint8(reader)?;
            same[i] = rle_length + 5;
            i += rle_length + 4;
            continue;
        }
        if logcounts[i] > omit_log {
            omit_log = logcounts[i];
            omit_pos = Some(i);
        }
        i += 1;
    }

    let omit_pos = omit_pos.ok_or(Error::InvalidCodestream("invalid histogram"))?;
    if omit_pos + 1 < length && logcounts[omit_pos + 1] == ANS_LOG_TAB_SIZE as i32 {
        return Err(Error::InvalidCodestream("invalid histogram RLE"));
    }

    let mut total_count = 0;
    let mut prev = 0;
    let mut numsame = 0;
    for i in 0..length {
        if same[i] != 0 {
            numsame = same[i] - 1;
            prev = if i > 0 { counts[i - 1] } else { 0 };
        }
        if numsame > 0 {
            counts[i] = prev;
            numsame -= 1;
        } else {
            let code = logcounts[i];
            if i == omit_pos || code < 0 {
                continue;
            } else if shift == 0 || code == 0 {
                counts[i] = 1 << code;
            } else {
                let bitcount = get_population_count_precision(code as u32, shift);
                counts[i] = (1 << code)
                    + ((reader.read_bits(bitcount as usize)? as i32) << (code as u32 - bitcount));
            }
        }
        total_count += counts[i];
    }
    counts[omit_pos] = range - total_count;
    if counts[omit_pos] <= 0 {
        return Err(Error::InvalidCodestream("invalid histogram count"));
    }
    Ok(counts)
}

fn init_alias_table(
    mut distribution: Vec<i32>,
    log_range: usize,
    log_alpha_size: usize,
    table: &mut [AliasEntry],
) -> Result<()> {
    let range = 1u32 << log_range;
    let table_size = 1usize << log_alpha_size;
    if table_size > range as usize || table.len() != table_size {
        return Err(Error::InvalidCodestream("invalid ANS table size"));
    }
    while distribution.last() == Some(&0) {
        distribution.pop();
    }
    if distribution.is_empty() {
        distribution.push(range as i32);
    }
    if distribution.len() > table_size {
        return Err(Error::InvalidCodestream("ANS distribution is too large"));
    }

    let entry_size = range >> log_alpha_size;
    let mut single_symbol = None;
    let mut sum = 0;
    for (symbol, &count) in distribution.iter().enumerate() {
        sum += count;
        if count == ANS_TAB_SIZE as i32 {
            if single_symbol.is_some() {
                return Err(Error::InvalidCodestream("invalid ANS distribution"));
            }
            single_symbol = Some(symbol);
        }
    }
    if sum != range as i32 {
        return Err(Error::InvalidCodestream("invalid ANS distribution sum"));
    }
    if let Some(symbol) = single_symbol {
        let symbol = symbol as u32;
        for (index, entry) in table.iter_mut().enumerate() {
            entry.right_value = symbol;
            entry.cutoff = 0;
            entry.offsets1 = entry_size * index as u32;
            entry.freq0 = 0;
            entry.freq1_xor_freq0 = ANS_TAB_SIZE as u32;
        }
        return Ok(());
    }

    let mut underfull = Vec::new();
    let mut overfull = Vec::new();
    let mut cutoffs = vec![0u32; table_size];
    for (index, &count) in distribution.iter().enumerate() {
        if count < 0 {
            return Err(Error::InvalidCodestream("negative ANS count"));
        }
        cutoffs[index] = count as u32;
        if cutoffs[index] > entry_size {
            overfull.push(index as u32);
        } else if cutoffs[index] < entry_size {
            underfull.push(index as u32);
        }
    }
    for (index, cutoff) in cutoffs
        .iter_mut()
        .enumerate()
        .take(table_size)
        .skip(distribution.len())
    {
        *cutoff = 0;
        underfull.push(index as u32);
    }

    while let Some(overfull_index) = overfull.pop() {
        let underfull_index = underfull
            .pop()
            .ok_or(Error::InvalidCodestream("invalid ANS distribution"))?;
        let underfull_by = entry_size - cutoffs[underfull_index as usize];
        cutoffs[overfull_index as usize] -= underfull_by;
        table[underfull_index as usize].right_value = overfull_index;
        table[underfull_index as usize].offsets1 = cutoffs[overfull_index as usize];
        if cutoffs[overfull_index as usize] < entry_size {
            underfull.push(overfull_index);
        } else if cutoffs[overfull_index as usize] > entry_size {
            overfull.push(overfull_index);
        }
    }

    for index in 0..table_size {
        if cutoffs[index] == entry_size {
            table[index].right_value = index as u32;
            table[index].offsets1 = 0;
            table[index].cutoff = 0;
        } else {
            table[index].offsets1 = table[index]
                .offsets1
                .checked_sub(cutoffs[index])
                .ok_or(Error::InvalidCodestream("invalid ANS alias offsets"))?;
            table[index].cutoff = cutoffs[index];
        }
        let freq0 = distribution.get(index).copied().unwrap_or(0) as u32;
        let right = table[index].right_value as usize;
        let freq1 = distribution.get(right).copied().unwrap_or(0) as u32;
        table[index].freq0 = freq0;
        table[index].freq1_xor_freq0 = freq1 ^ freq0;
    }
    Ok(())
}

fn lookup_alias(
    table: &[AliasEntry],
    value: u32,
    log_entry_size: usize,
    entry_size_minus_1: u32,
) -> AliasSymbol {
    let index = (value >> log_entry_size) as usize;
    let position = value & entry_size_minus_1;
    let entry = table[index];
    let greater = position >= entry.cutoff;
    AliasSymbol {
        value: if greater {
            entry.right_value as usize
        } else {
            index
        },
        offset: if greater {
            entry.offsets1 + position
        } else {
            position
        },
        freq: if greater {
            entry.freq0 ^ entry.freq1_xor_freq0
        } else {
            entry.freq0
        },
    }
}

impl HuffmanDecodingData {
    fn read_from_bitstream(alphabet_size: usize, reader: &mut BitReader<'_>) -> Result<Self> {
        if alphabet_size > PREFIX_MAX_ALPHABET_SIZE {
            return Err(Error::InvalidCodestream("prefix alphabet is too large"));
        }
        let mut data = Self::default();
        if alphabet_size <= 1 {
            data.table = vec![HuffmanCode { bits: 0, value: 0 }; 1 << HUFFMAN_TABLE_BITS];
            return Ok(data);
        }

        let simple_code_or_skip = reader.read_bits(2)? as usize;
        if simple_code_or_skip == 1 {
            data.table = vec![HuffmanCode::default(); 1 << HUFFMAN_TABLE_BITS];
            read_simple_huffman_code(alphabet_size, reader, &mut data.table)?;
            return Ok(data);
        }

        let mut code_lengths = vec![0u8; alphabet_size];
        let mut code_length_code_lengths = [0u8; CODE_LENGTH_CODES];
        let mut space = 32i32;
        let mut num_codes = 0;
        for &code_len_index in CODE_LENGTH_CODE_ORDER
            .iter()
            .take(CODE_LENGTH_CODES)
            .skip(simple_code_or_skip)
        {
            if space <= 0 {
                break;
            }
            let index = reader.peek_bits(4)? as usize;
            let huff = CODE_LENGTH_HUFFMAN[index];
            reader.skip_bits(huff.bits as usize)?;
            let value = huff.value as u8;
            code_length_code_lengths[code_len_index as usize] = value;
            if value != 0 {
                space -= 32 >> value;
                num_codes += 1;
            }
        }
        if !(num_codes == 1 || space == 0) {
            return Err(Error::InvalidCodestream("invalid Huffman code lengths"));
        }
        read_huffman_code_lengths(&code_length_code_lengths, &mut code_lengths, reader)?;

        let mut counts = [0u16; PREFIX_MAX_BITS + 1];
        for &length in &code_lengths {
            counts[length as usize] += 1;
        }
        let mut table = vec![HuffmanCode::default(); alphabet_size + 376];
        let table_size =
            build_huffman_table(&mut table, HUFFMAN_TABLE_BITS, &code_lengths, &mut counts)?;
        table.truncate(table_size);
        data.table = table;
        Ok(data)
    }

    fn read_symbol(&self, reader: &mut BitReader<'_>) -> Result<u16> {
        let mut index = reader.peek_bits(HUFFMAN_TABLE_BITS)? as usize;
        let mut code = *self
            .table
            .get(index)
            .ok_or(Error::InvalidCodestream("invalid Huffman table lookup"))?;
        if code.bits as usize > HUFFMAN_TABLE_BITS {
            reader.skip_bits(HUFFMAN_TABLE_BITS)?;
            let nbits = code.bits as usize - HUFFMAN_TABLE_BITS;
            index = index
                .checked_add(code.value as usize)
                .and_then(|base| base.checked_add(reader.peek_bits(nbits).ok()? as usize))
                .ok_or(Error::InvalidCodestream("invalid Huffman table lookup"))?;
            code = *self
                .table
                .get(index)
                .ok_or(Error::InvalidCodestream("invalid Huffman table lookup"))?;
        }
        reader.skip_bits(code.bits as usize)?;
        Ok(code.value)
    }
}

fn read_simple_huffman_code(
    alphabet_size: usize,
    reader: &mut BitReader<'_>,
    table: &mut [HuffmanCode],
) -> Result<()> {
    let max_bits = if alphabet_size > 1 {
        floor_log2_nonzero(alphabet_size - 1) + 1
    } else {
        0
    };
    let mut num_symbols = reader.read_bits(2)? as usize + 1;
    let mut symbols = [0u16; 4];
    for symbol in symbols.iter_mut().take(num_symbols) {
        *symbol = reader.read_bits(max_bits)? as u16;
        if usize::from(*symbol) >= alphabet_size {
            return Err(Error::InvalidCodestream("invalid simple Huffman symbol"));
        }
    }
    for i in 0..num_symbols.saturating_sub(1) {
        for j in i + 1..num_symbols {
            if symbols[i] == symbols[j] {
                return Err(Error::InvalidCodestream("duplicate simple Huffman symbol"));
            }
        }
    }
    if num_symbols == 4 {
        num_symbols += reader.read_bits(1)? as usize;
    }

    let mut table_size = 1;
    match num_symbols {
        1 => {
            table[0] = HuffmanCode {
                bits: 0,
                value: symbols[0],
            }
        }
        2 => {
            symbols[..2].sort_unstable();
            table[0] = HuffmanCode {
                bits: 1,
                value: symbols[0],
            };
            table[1] = HuffmanCode {
                bits: 1,
                value: symbols[1],
            };
            table_size = 2;
        }
        3 => {
            if symbols[1] > symbols[2] {
                symbols.swap(1, 2);
            }
            table[0] = HuffmanCode {
                bits: 1,
                value: symbols[0],
            };
            table[2] = HuffmanCode {
                bits: 1,
                value: symbols[0],
            };
            table[1] = HuffmanCode {
                bits: 2,
                value: symbols[1],
            };
            table[3] = HuffmanCode {
                bits: 2,
                value: symbols[2],
            };
            table_size = 4;
        }
        4 => {
            symbols[..4].sort_unstable();
            table[0] = HuffmanCode {
                bits: 2,
                value: symbols[0],
            };
            table[2] = HuffmanCode {
                bits: 2,
                value: symbols[1],
            };
            table[1] = HuffmanCode {
                bits: 2,
                value: symbols[2],
            };
            table[3] = HuffmanCode {
                bits: 2,
                value: symbols[3],
            };
            table_size = 4;
        }
        5 => {
            if symbols[2] > symbols[3] {
                symbols.swap(2, 3);
            }
            table[0] = HuffmanCode {
                bits: 1,
                value: symbols[0],
            };
            table[1] = HuffmanCode {
                bits: 2,
                value: symbols[1],
            };
            table[2] = HuffmanCode {
                bits: 1,
                value: symbols[0],
            };
            table[3] = HuffmanCode {
                bits: 3,
                value: symbols[2],
            };
            table[4] = HuffmanCode {
                bits: 1,
                value: symbols[0],
            };
            table[5] = HuffmanCode {
                bits: 2,
                value: symbols[1],
            };
            table[6] = HuffmanCode {
                bits: 1,
                value: symbols[0],
            };
            table[7] = HuffmanCode {
                bits: 3,
                value: symbols[3],
            };
            table_size = 8;
        }
        _ => return Err(Error::InvalidCodestream("invalid simple Huffman code")),
    }

    while table_size != 1 << HUFFMAN_TABLE_BITS {
        let (left, right) = table.split_at_mut(table_size);
        right[..table_size].copy_from_slice(left);
        table_size <<= 1;
    }
    Ok(())
}

fn read_huffman_code_lengths(
    code_length_code_lengths: &[u8; CODE_LENGTH_CODES],
    code_lengths: &mut [u8],
    reader: &mut BitReader<'_>,
) -> Result<()> {
    let mut symbol = 0;
    let mut prev_code_len = DEFAULT_CODE_LENGTH;
    let mut repeat = 0i32;
    let mut repeat_code_len = 0;
    let mut space = 32768i32;
    let mut table = [HuffmanCode::default(); 32];
    let mut counts = [0u16; PREFIX_MAX_BITS + 1];
    for &length in code_length_code_lengths {
        counts[length as usize] += 1;
    }
    build_huffman_table(&mut table, 5, code_length_code_lengths, &mut counts)?;

    while symbol < code_lengths.len() && space > 0 {
        let index = reader.peek_bits(5)? as usize;
        let code_len = table.get(index).ok_or(Error::InvalidCodestream(
            "invalid code-length Huffman lookup",
        ))?;
        reader.skip_bits(code_len.bits as usize)?;
        let code_len = code_len.value as u8;
        if code_len < CODE_LENGTH_REPEAT_CODE {
            repeat = 0;
            code_lengths[symbol] = code_len;
            symbol += 1;
            if code_len != 0 {
                prev_code_len = code_len;
                space -= 32768 >> code_len;
            }
        } else {
            let extra_bits = code_len - 14;
            let old_repeat = repeat;
            let mut new_len = 0;
            if code_len == CODE_LENGTH_REPEAT_CODE {
                new_len = prev_code_len;
            }
            if repeat_code_len != new_len {
                repeat = 0;
                repeat_code_len = new_len;
            }
            if repeat > 0 {
                repeat -= 2;
                repeat <<= extra_bits;
            }
            repeat += reader.read_bits(extra_bits as usize)? as i32 + 3;
            let repeat_delta = repeat - old_repeat;
            if repeat_delta < 0 || symbol + repeat_delta as usize > code_lengths.len() {
                return Err(Error::InvalidCodestream("invalid Huffman repeat"));
            }
            for code_length in &mut code_lengths[symbol..symbol + repeat_delta as usize] {
                *code_length = repeat_code_len;
            }
            symbol += repeat_delta as usize;
            if repeat_code_len != 0 {
                space -= repeat_delta << (15 - repeat_code_len);
            }
        }
    }

    if space != 0 {
        return Err(Error::InvalidCodestream("invalid Huffman code lengths"));
    }
    code_lengths[symbol..].fill(0);
    Ok(())
}

fn build_huffman_table(
    table: &mut [HuffmanCode],
    root_bits: usize,
    code_lengths: &[u8],
    counts: &mut [u16; PREFIX_MAX_BITS + 1],
) -> Result<usize> {
    if code_lengths.len() > (1 << PREFIX_MAX_BITS) {
        return Err(Error::InvalidCodestream("Huffman alphabet is too large"));
    }

    let mut offsets = [0u16; PREFIX_MAX_BITS + 1];
    let mut max_length = 1;
    let mut sum = 0u16;
    for length in 1..=PREFIX_MAX_BITS {
        offsets[length] = sum;
        if counts[length] != 0 {
            sum = sum.wrapping_add(counts[length]);
            max_length = length;
        }
    }

    let mut sorted = vec![0u16; code_lengths.len()];
    for (symbol, &length) in code_lengths.iter().enumerate() {
        if length != 0 {
            let offset = &mut offsets[length as usize];
            sorted[*offset as usize] = symbol as u16;
            *offset = offset.wrapping_add(1);
        }
    }

    let mut table_bits = root_bits;
    let mut table_size = 1usize << table_bits;
    let mut total_size = table_size;
    if offsets[PREFIX_MAX_BITS] == 1 {
        let code = HuffmanCode {
            bits: 0,
            value: sorted[0],
        };
        table[..total_size].fill(code);
        return Ok(total_size);
    }

    if table_bits > max_length {
        table_bits = max_length;
        table_size = 1usize << table_bits;
    }
    let mut key = 0usize;
    let mut symbol = 0usize;
    let mut code_bits = 1usize;
    let mut step = 2usize;
    while code_bits <= table_bits {
        while counts[code_bits] != 0 {
            let code = HuffmanCode {
                bits: code_bits as u8,
                value: sorted[symbol],
            };
            replicate_value(&mut table[key..], step, table_size, code);
            key = next_huffman_key(key, code_bits);
            symbol += 1;
            counts[code_bits] -= 1;
        }
        code_bits += 1;
        step <<= 1;
    }

    while total_size != table_size {
        let (left, right) = table.split_at_mut(table_size);
        right[..table_size].copy_from_slice(left);
        table_size <<= 1;
    }

    let mask = total_size - 1;
    let mut low = None;
    let mut len = root_bits + 1;
    step = 2;
    while len <= max_length {
        while counts[len] != 0 {
            if low != Some(key & mask) {
                let table_start = total_size;
                table_bits = next_table_bit_size(counts, len, root_bits);
                table_size = 1 << table_bits;
                total_size += table_size;
                if total_size > table.len() {
                    return Err(Error::InvalidCodestream("Huffman table overflow"));
                }
                low = Some(key & mask);
                let root = key & mask;
                table[root].bits = (table_bits + root_bits) as u8;
                table[root].value = (table_start - root) as u16;
            }
            let code = HuffmanCode {
                bits: (len - root_bits) as u8,
                value: sorted[symbol],
            };
            let table_start = total_size - table_size;
            replicate_value(
                &mut table[table_start + (key >> root_bits)..],
                step,
                table_size,
                code,
            );
            key = next_huffman_key(key, len);
            symbol += 1;
            counts[len] -= 1;
        }
        len += 1;
        step <<= 1;
    }

    Ok(total_size)
}

fn replicate_value(table: &mut [HuffmanCode], step: usize, end: usize, code: HuffmanCode) {
    let mut offset = end;
    while offset > 0 {
        offset -= step;
        table[offset] = code;
    }
}

fn next_table_bit_size(
    counts: &[u16; PREFIX_MAX_BITS + 1],
    mut len: usize,
    root_bits: usize,
) -> usize {
    let mut left = 1usize << (len - root_bits);
    while len < PREFIX_MAX_BITS {
        if left <= counts[len] as usize {
            break;
        }
        left -= counts[len] as usize;
        len += 1;
        left <<= 1;
    }
    len - root_bits
}

fn next_huffman_key(key: usize, len: usize) -> usize {
    let mut step = 1usize << (len - 1);
    while key & step != 0 {
        step >>= 1;
    }
    if step == 0 {
        return key;
    }
    (key & (step - 1)) + step
}

fn decode_var_len_uint8(reader: &mut BitReader<'_>) -> Result<usize> {
    if reader.read_bool()? {
        let nbits = reader.read_bits(3)? as usize;
        if nbits == 0 {
            Ok(1)
        } else {
            Ok(reader.read_bits(nbits)? as usize + (1 << nbits))
        }
    } else {
        Ok(0)
    }
}

fn decode_var_len_uint16(reader: &mut BitReader<'_>) -> Result<usize> {
    if reader.read_bool()? {
        let nbits = reader.read_bits(4)? as usize;
        if nbits == 0 {
            Ok(1)
        } else {
            Ok(reader.read_bits(nbits)? as usize + (1 << nbits))
        }
    } else {
        Ok(0)
    }
}

fn create_flat_histogram(length: usize, total_count: i32) -> Vec<i32> {
    let count = total_count / length as i32;
    let remainder = total_count % length as i32;
    let mut result = vec![count; length];
    for item in result.iter_mut().take(remainder as usize) {
        *item += 1;
    }
    result
}

fn get_population_count_precision(logcount: u32, shift: u32) -> u32 {
    let value =
        (logcount as i32).min(shift as i32 - ((ANS_LOG_TAB_SIZE as i32 - logcount as i32) >> 1));
    value.max(0) as u32
}

fn ceil_log2_nonzero(value: usize) -> usize {
    usize::BITS as usize - (value - 1).leading_zeros() as usize
}

fn floor_log2_nonzero(value: usize) -> usize {
    usize::BITS as usize - 1 - value.leading_zeros() as usize
}

const CODE_LENGTH_CODES: usize = 18;
const DEFAULT_CODE_LENGTH: u8 = 8;
const CODE_LENGTH_REPEAT_CODE: u8 = 16;
const CODE_LENGTH_CODE_ORDER: [u8; CODE_LENGTH_CODES] =
    [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];

const CODE_LENGTH_HUFFMAN: [HuffmanCode; 16] = [
    HuffmanCode { bits: 2, value: 0 },
    HuffmanCode { bits: 2, value: 4 },
    HuffmanCode { bits: 2, value: 3 },
    HuffmanCode { bits: 3, value: 2 },
    HuffmanCode { bits: 2, value: 0 },
    HuffmanCode { bits: 2, value: 4 },
    HuffmanCode { bits: 2, value: 3 },
    HuffmanCode { bits: 4, value: 1 },
    HuffmanCode { bits: 2, value: 0 },
    HuffmanCode { bits: 2, value: 4 },
    HuffmanCode { bits: 2, value: 3 },
    HuffmanCode { bits: 3, value: 2 },
    HuffmanCode { bits: 2, value: 0 },
    HuffmanCode { bits: 2, value: 4 },
    HuffmanCode { bits: 2, value: 3 },
    HuffmanCode { bits: 4, value: 5 },
];

const HISTOGRAM_LOGCOUNT_HUFFMAN: [(u8, u8); 128] = [
    (3, 10),
    (7, 12),
    (3, 7),
    (4, 3),
    (3, 6),
    (3, 8),
    (3, 9),
    (4, 5),
    (3, 10),
    (4, 4),
    (3, 7),
    (4, 1),
    (3, 6),
    (3, 8),
    (3, 9),
    (4, 2),
    (3, 10),
    (5, 0),
    (3, 7),
    (4, 3),
    (3, 6),
    (3, 8),
    (3, 9),
    (4, 5),
    (3, 10),
    (4, 4),
    (3, 7),
    (4, 1),
    (3, 6),
    (3, 8),
    (3, 9),
    (4, 2),
    (3, 10),
    (6, 11),
    (3, 7),
    (4, 3),
    (3, 6),
    (3, 8),
    (3, 9),
    (4, 5),
    (3, 10),
    (4, 4),
    (3, 7),
    (4, 1),
    (3, 6),
    (3, 8),
    (3, 9),
    (4, 2),
    (3, 10),
    (5, 0),
    (3, 7),
    (4, 3),
    (3, 6),
    (3, 8),
    (3, 9),
    (4, 5),
    (3, 10),
    (4, 4),
    (3, 7),
    (4, 1),
    (3, 6),
    (3, 8),
    (3, 9),
    (4, 2),
    (3, 10),
    (7, 13),
    (3, 7),
    (4, 3),
    (3, 6),
    (3, 8),
    (3, 9),
    (4, 5),
    (3, 10),
    (4, 4),
    (3, 7),
    (4, 1),
    (3, 6),
    (3, 8),
    (3, 9),
    (4, 2),
    (3, 10),
    (5, 0),
    (3, 7),
    (4, 3),
    (3, 6),
    (3, 8),
    (3, 9),
    (4, 5),
    (3, 10),
    (4, 4),
    (3, 7),
    (4, 1),
    (3, 6),
    (3, 8),
    (3, 9),
    (4, 2),
    (3, 10),
    (6, 11),
    (3, 7),
    (4, 3),
    (3, 6),
    (3, 8),
    (3, 9),
    (4, 5),
    (3, 10),
    (4, 4),
    (3, 7),
    (4, 1),
    (3, 6),
    (3, 8),
    (3, 9),
    (4, 2),
    (3, 10),
    (5, 0),
    (3, 7),
    (4, 3),
    (3, 6),
    (3, 8),
    (3, 9),
    (4, 5),
    (3, 10),
    (4, 4),
    (3, 7),
    (4, 1),
    (3, 6),
    (3, 8),
    (3, 9),
    (4, 2),
];
