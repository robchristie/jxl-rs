use crate::bitstream::BitReader;
use crate::error::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct CustomTransformData {
    pub opsin_inverse_matrix: Option<OpsinInverseMatrix>,
    pub custom_weights_mask: u32,
    pub upsampling2_weights: Option<Vec<f32>>,
    pub upsampling4_weights: Option<Vec<f32>>,
    pub upsampling8_weights: Option<Vec<f32>>,
}

impl Default for CustomTransformData {
    fn default() -> Self {
        Self {
            opsin_inverse_matrix: None,
            custom_weights_mask: 0,
            upsampling2_weights: None,
            upsampling4_weights: None,
            upsampling8_weights: None,
        }
    }
}

impl CustomTransformData {
    pub fn is_default(&self) -> bool {
        self.opsin_inverse_matrix.is_none()
            && self.custom_weights_mask == 0
            && self.upsampling2_weights.is_none()
            && self.upsampling4_weights.is_none()
            && self.upsampling8_weights.is_none()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OpsinInverseMatrix {
    pub inverse_matrix: [[f32; 3]; 3],
    pub opsin_biases: [f32; 3],
    pub quant_biases: [f32; 4],
}

pub fn read_custom_transform_data(
    reader: &mut BitReader<'_>,
    xyb_encoded: bool,
) -> Result<CustomTransformData> {
    if reader.read_bool()? {
        return Ok(CustomTransformData::default());
    }

    let opsin_inverse_matrix = if xyb_encoded {
        read_opsin_inverse_matrix(reader)?
    } else {
        None
    };
    let custom_weights_mask = reader.read_bits(3)? as u32;

    let upsampling2_weights = if custom_weights_mask & 0x1 != 0 {
        Some(read_f16_vec(reader, 15)?)
    } else {
        None
    };
    let upsampling4_weights = if custom_weights_mask & 0x2 != 0 {
        Some(read_f16_vec(reader, 55)?)
    } else {
        None
    };
    let upsampling8_weights = if custom_weights_mask & 0x4 != 0 {
        Some(read_f16_vec(reader, 210)?)
    } else {
        None
    };

    Ok(CustomTransformData {
        opsin_inverse_matrix,
        custom_weights_mask,
        upsampling2_weights,
        upsampling4_weights,
        upsampling8_weights,
    })
}

fn read_opsin_inverse_matrix(reader: &mut BitReader<'_>) -> Result<Option<OpsinInverseMatrix>> {
    if reader.read_bool()? {
        return Ok(None);
    }

    let mut inverse_matrix = [[0.0; 3]; 3];
    for row in &mut inverse_matrix {
        for value in row {
            *value = reader.read_f16()?;
        }
    }

    let mut opsin_biases = [0.0; 3];
    for value in &mut opsin_biases {
        *value = reader.read_f16()?;
    }

    let mut quant_biases = [0.0; 4];
    for value in &mut quant_biases {
        *value = reader.read_f16()?;
    }

    Ok(Some(OpsinInverseMatrix {
        inverse_matrix,
        opsin_biases,
        quant_biases,
    }))
}

fn read_f16_vec(reader: &mut BitReader<'_>, len: usize) -> Result<Vec<f32>> {
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        values.push(reader.read_f16()?);
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_all_default_transform_data() {
        let mut reader = BitReader::new(&[1]);
        let transform = read_custom_transform_data(&mut reader, true).unwrap();

        assert!(transform.is_default());
    }

    #[test]
    fn reads_non_default_without_xyb_opsin_matrix() {
        // all_default = false, custom_weights_mask = 0.
        let mut reader = BitReader::new(&[0]);
        let transform = read_custom_transform_data(&mut reader, false).unwrap();

        assert!(transform.is_default());
        assert_eq!(reader.bits_consumed(), 4);
    }
}
