use crate::error::{Error, Result};

/// Bounds-checked little-endian, least-significant-bit-first reader.
///
/// JPEG XL field bundles are written in the same bit order as libjxl's
/// `BitReader`: the first bit consumed is the low bit of the current byte.
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    bytes: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit_pos: 0 }
    }

    pub fn bits_consumed(&self) -> usize {
        self.bit_pos
    }

    pub fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_bits(1)? != 0)
    }

    pub fn peek_bits(&self, bits: usize) -> Result<u64> {
        if bits > 56 {
            return Err(Error::InvalidCodestream("bit reads are limited to 56 bits"));
        }

        let end = self
            .bit_pos
            .checked_add(bits)
            .ok_or(Error::InvalidCodestream("bit position overflow"))?;
        if end > self.bytes.len() * 8 {
            return Err(Error::Truncated);
        }

        let mut value = 0u64;
        for out_bit in 0..bits {
            let pos = self.bit_pos + out_bit;
            let bit = (self.bytes[pos / 8] >> (pos % 8)) & 1;
            value |= u64::from(bit) << out_bit;
        }
        Ok(value)
    }

    pub fn read_bits(&mut self, bits: usize) -> Result<u64> {
        let value = self.peek_bits(bits)?;
        self.skip_bits(bits)?;
        Ok(value)
    }

    pub fn read_u32_selector(
        &mut self,
        d0: U32Distribution,
        d1: U32Distribution,
        d2: U32Distribution,
        d3: U32Distribution,
    ) -> Result<u32> {
        let selector = self.read_bits(2)? as usize;
        let distribution = [d0, d1, d2, d3][selector];
        match distribution {
            U32Distribution::Direct(value) => Ok(value),
            U32Distribution::BitsOffset { bits, offset } => {
                let value = self.read_bits(bits as usize)? as u32;
                offset
                    .checked_add(value)
                    .ok_or(Error::InvalidCodestream("u32 field overflow"))
            }
        }
    }

    pub fn read_u64(&mut self) -> Result<u64> {
        let selector = self.read_bits(2)?;
        match selector {
            0 => Ok(0),
            1 => Ok(1 + self.read_bits(4)?),
            2 => Ok(17 + self.read_bits(8)?),
            _ => {
                let mut result = self.read_bits(12)?;
                let mut shift = 12;
                while self.read_bool()? {
                    if shift == 60 {
                        result |= self.read_bits(4)? << shift;
                        break;
                    }
                    result |= self.read_bits(8)? << shift;
                    shift += 8;
                }
                Ok(result)
            }
        }
    }

    pub fn read_f16(&mut self) -> Result<f32> {
        let bits = self.read_bits(16)? as u16;
        let value = half::f16::from_bits(bits);
        if value.is_nan() || value.is_infinite() {
            return Err(Error::InvalidCodestream("invalid f16 value"));
        }
        Ok(f32::from(value))
    }

    pub fn read_enum(&mut self, valid_values: &[u32]) -> Result<u32> {
        let value =
            self.read_u32_selector(val(0), val(1), bits_offset(4, 2), bits_offset(6, 18))?;
        if valid_values.contains(&value) {
            Ok(value)
        } else {
            Err(Error::InvalidCodestream("invalid enum value"))
        }
    }

    pub fn read_name(&mut self) -> Result<String> {
        let name_length = self.read_u32_selector(
            val(0),
            U32Distribution::BitsOffset { bits: 4, offset: 0 },
            bits_offset(5, 16),
            bits_offset(10, 48),
        )? as usize;
        let mut bytes = Vec::with_capacity(name_length);
        for _ in 0..name_length {
            bytes.push(self.read_bits(8)? as u8);
        }

        String::from_utf8(bytes).map_err(|_| Error::InvalidCodestream("invalid UTF-8 name"))
    }

    pub fn jump_to_byte_boundary(&mut self) -> Result<()> {
        let padding_bits = self.bit_pos % 8;
        if padding_bits == 0 {
            return Ok(());
        }

        let padding = self.read_bits(8 - padding_bits)?;
        if padding != 0 {
            return Err(Error::InvalidCodestream("non-zero padding bits"));
        }
        Ok(())
    }

    pub fn skip_bits(&mut self, bits: usize) -> Result<()> {
        let end = self
            .bit_pos
            .checked_add(bits)
            .ok_or(Error::InvalidCodestream("bit position overflow"))?;
        if end > self.bytes.len() * 8 {
            return Err(Error::Truncated);
        }
        self.bit_pos = end;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum U32Distribution {
    Direct(u32),
    BitsOffset { bits: u8, offset: u32 },
}

pub const fn val(value: u32) -> U32Distribution {
    U32Distribution::Direct(value)
}

pub const fn bits_offset(bits: u8, offset: u32) -> U32Distribution {
    U32Distribution::BitsOffset { bits, offset }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_bits_lsb_first() {
        let mut reader = BitReader::new(&[0b1010_0110, 0b0000_0011]);

        assert_eq!(reader.read_bits(1).unwrap(), 0);
        assert_eq!(reader.read_bits(3).unwrap(), 0b011);
        assert_eq!(reader.read_bits(4).unwrap(), 0b1010);
        assert_eq!(reader.read_bits(2).unwrap(), 0b11);
        assert_eq!(reader.bits_consumed(), 10);
    }

    #[test]
    fn rejects_out_of_bounds_reads() {
        let mut reader = BitReader::new(&[0]);

        assert_eq!(reader.read_bits(9), Err(Error::Truncated));
    }

    #[test]
    fn jumps_to_byte_boundary_through_zero_padding() {
        let mut reader = BitReader::new(&[0b0000_0101, 0b1100_0011]);

        assert_eq!(reader.read_bits(3).unwrap(), 0b101);
        reader.jump_to_byte_boundary().unwrap();
        assert_eq!(reader.bits_consumed(), 8);
        assert_eq!(reader.read_bits(4).unwrap(), 0b0011);
    }

    #[test]
    fn rejects_non_zero_byte_boundary_padding() {
        let mut reader = BitReader::new(&[0b0000_1001]);

        assert_eq!(reader.read_bits(3).unwrap(), 0b001);
        assert_eq!(
            reader.jump_to_byte_boundary(),
            Err(Error::InvalidCodestream("non-zero padding bits"))
        );
    }
}
