//! Minimal tensor operations for inference - optimized.

/// Quantized weight storage - supports multiple formats
#[derive(Clone)]
pub enum QuantizedWeights {
    /// F32 full precision weights
    F32(Vec<f32>),
    /// F16 weights stored as bytes (16-bit float)
    F16(Vec<u8>),
    /// Q8_0: 2 bytes scale (fp16) + 32 bytes int8 per block
    Q8_0(Vec<u8>, usize, usize),
    /// Q4_K: Complex super-block format (256 elements per block)
    #[allow(non_camel_case_types)]
    Q4_K(Vec<u8>, usize, usize),
}

impl QuantizedWeights {
    /// Get rows and cols for Q8_0 and Q4_K variants
    pub fn dims(&self) -> (usize, usize) {
        match self {
            QuantizedWeights::F32(data) => {
                // We need to track dims separately for F32
                (0, 0)
            }
            QuantizedWeights::F16(_) => (0, 0),
            QuantizedWeights::Q8_0(_, rows, cols) => (*rows, *cols),
            QuantizedWeights::Q4_K(_, rows, cols) => (*rows, *cols),
        }
    }

    /// Matrix-vector multiplication with quantized weights
    pub fn matvec(&self, vec: &[f32], out: &mut [f32]) {
        match self {
            QuantizedWeights::F32(data) => {
                let (rows, cols) = (out.len(), vec.len());
                matvec(data, rows, cols, vec, out);
            }
            QuantizedWeights::F16(data) => {
                let (rows, cols) = (out.len(), vec.len());
                matvec_f16(data, rows, cols, vec, out);
            }
            QuantizedWeights::Q8_0(data, rows, cols) => {
                matvec_q8_0(data, *rows, *cols, vec, out);
            }
            QuantizedWeights::Q4_K(data, rows, cols) => {
                matvec_q4_k(data, *rows, *cols, vec, out);
            }
        }
    }
}

/// Matrix-vector multiplication: out = mat * vec (optimized)
#[inline]
pub fn matvec(mat: &[f32], rows: usize, cols: usize, vec: &[f32], out: &mut [f32]) {
    assert_eq!(mat.len(), rows * cols);
    assert_eq!(vec.len(), cols);
    assert_eq!(out.len(), rows);

    for i in 0..rows {
        let mut sum = 0.0f32;
        let base = i * cols;

        let mut j = 0;
        let cols4 = (cols / 4) * 4;

        while j < cols4 {
            sum += mat[base + j] * vec[j];
            sum += mat[base + j + 1] * vec[j + 1];
            sum += mat[base + j + 2] * vec[j + 2];
            sum += mat[base + j + 3] * vec[j + 3];
            j += 4;
        }

        while j < cols {
            sum += mat[base + j] * vec[j];
            j += 1;
        }

        out[i] = sum;
    }
}

/// Convert f32 to f16 bits
pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let s = (bits >> 31) & 0x1;
    let e = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let m = bits & 0x7FFFFF;

    if e <= 0 {
        // Denormal or zero
        0
    } else if e >= 31 {
        // Inf or overflow
        if m == 0 {
            0x7C00
        } else {
            0x7FFF
        }
    } else {
        // Normalized
        let m_shifted = m >> 13;
        ((s as u16) << 15) | ((e as u16) << 10) | ((m_shifted >> 0) as u16)
    }
}

/// Softmax over the last dimension (optimized)
#[inline]
pub fn softmax(x: &mut [f32], dim: usize) {
    let batch = x.len() / dim;
    for b in 0..batch {
        let offset = b * dim;
        let slice = &mut x[offset..offset + dim];

        // Find max
        let mut max_val = f32::NEG_INFINITY;
        for &v in slice.iter() {
            if v > max_val {
                max_val = v;
            }
        }

        // Exp and sum
        let mut sum = 0.0f32;
        for v in slice.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }

        // Normalize
        let inv_sum = 1.0 / sum;
        for v in slice.iter_mut() {
            *v *= inv_sum;
        }
    }
}

/// Silu activation (optimized)
#[inline]
pub fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

/// RMSNorm (optimized)
#[inline]
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let len = x.len();

    // Compute RMS
    let mut sum = 0.0f32;
    for &v in x {
        sum += v * v;
    }
    let rms = 1.0 / (sum / len as f32 + eps).sqrt();

    // Normalize and scale with loop unrolling
    let mut i = 0;
    let len4 = (len / 4) * 4;

    while i < len4 {
        out[i] = x[i] * rms * weight[i];
        out[i + 1] = x[i + 1] * rms * weight[i + 1];
        out[i + 2] = x[i + 2] * rms * weight[i + 2];
        out[i + 3] = x[i + 3] * rms * weight[i + 3];
        i += 4;
    }

    while i < len {
        out[i] = x[i] * rms * weight[i];
        i += 1;
    }
}

/// Matrix-vector multiplication with F16 weights (direct, no dequant)
#[inline]
pub fn matvec_f16(weight_data: &[u8], rows: usize, cols: usize, vec: &[f32], out: &mut [f32]) {
    assert_eq!(out.len(), rows);
    assert_eq!(vec.len(), cols);

    let bytes_per_row = cols * 2; // 2 bytes per f16

    for row in 0..rows {
        let mut sum = 0.0f32;
        let base = row * bytes_per_row;

        for col in 0..cols {
            let bits =
                u16::from_le_bytes([weight_data[base + col * 2], weight_data[base + col * 2 + 1]]);
            let val = f16_to_f32(bits);
            sum += val * vec[col];
        }

        out[row] = sum;
    }
}

/// Matrix-vector multiplication with Q8_0 quantized weights
///
/// Q8_0 format: Each block of 32 elements has:
/// - 2 bytes: fp16 scale (d)
/// - 32 bytes: int8 quantized values (qs)
///
/// This avoids full f32 dequantization by processing blocks directly.
#[inline]
pub fn matvec_q8_0(weight_data: &[u8], rows: usize, cols: usize, vec: &[f32], out: &mut [f32]) {
    assert_eq!(out.len(), rows);
    assert_eq!(vec.len(), cols);

    let block_size = 32;
    let bytes_per_block = 2 + block_size; // scale + quantized values

    for row in 0..rows {
        let mut sum = 0.0f32;
        let mut col = 0;

        while col < cols {
            let block_start = row * ((cols + block_size - 1) / block_size) * bytes_per_block
                + (col / block_size) * bytes_per_block;

            // Read scale (fp16 -> f32)
            let scale = f16_to_f32(u16::from_le_bytes([
                weight_data[block_start],
                weight_data[block_start + 1],
            ]));

            // Process up to 32 elements in this block
            let remaining = (cols - col).min(block_size);
            let mut block_offset = 2; // skip scale bytes

            for _ in 0..remaining {
                let q = weight_data[block_start + block_offset] as i8 as f32;
                sum += q * vec[col] * scale;
                col += 1;
                block_offset += 1;
            }
        }

        out[row] = sum;
    }
}

/// Matrix-vector multiplication with Q4_K quantized weights
///
/// Q4_K format (QK_K = 256):
/// - 2 bytes: d (super-block scale, fp16)
/// - 2 bytes: dmin (super-block min scale, fp16)  
/// - 12 bytes: scales (6-bit quantized for 8 sub-blocks: 4 scales + 4 mins packed)
/// - 128 bytes: qs (4-bit quantized values: 2 elements per byte)
///
/// Each sub-block (32 elements) has:
/// - 1 nibble for scale (from scales[i])
/// - 1 nibble for min (from scales[i+4])
/// - 16 bytes of 4-bit quantized values
#[inline]
pub fn matvec_q4_k(weight_data: &[u8], rows: usize, cols: usize, vec: &[f32], out: &mut [f32]) {
    assert_eq!(out.len(), rows);
    assert_eq!(vec.len(), cols);

    const QK_K: usize = 256; // Super-block size
    const SUB_BLOCKS: usize = 8; // 8 sub-blocks of 32 elements
    const BLOCK_SIZE: usize = 32;

    // Bytes per 256-element super-block: 2(d) + 2(dmin) + 12(scales) + 128(qs) = 144
    const BYTES_PER_BLOCK: usize = 2 + 2 + 12 + (QK_K / 2);

    for row in 0..rows {
        let mut sum = 0.0f32;
        let mut col = 0;

        while col < cols {
            // Position in weight data for this row's super-block
            let super_block_idx = col / QK_K;
            let block_start = row * ((cols + QK_K - 1) / QK_K) * BYTES_PER_BLOCK
                + super_block_idx * BYTES_PER_BLOCK;

            // Read super-block scales
            let d = f16_to_f32(u16::from_le_bytes([
                weight_data[block_start],
                weight_data[block_start + 1],
            ]));
            let dmin = f16_to_f32(u16::from_le_bytes([
                weight_data[block_start + 2],
                weight_data[block_start + 3],
            ]));

            // Read scales (12 bytes, each nibble is 6-bit quantized)
            // scales[0-3] are the scales, scales[4-7] are the mins
            let mut scales = [0i32; 8];
            for i in 0..6 {
                let byte_idx = 4 + i;
                let low = (weight_data[block_start + byte_idx] & 0x0F) as i32;
                let high = ((weight_data[block_start + byte_idx] >> 4) & 0x0F) as i32;
                let scale_low = (low as f32 - 16.0) / 8.0; // Dequantize 6-bit
                let scale_high = (high as f32 - 16.0) / 8.0;
                if i < 4 {
                    scales[i] = (scale_low * d * 64.0) as i32;
                } else {
                    scales[i] = (scale_high * dmin * 64.0) as i32;
                }
            }
            // scales[6] = d * 64, scales[7] = dmin * 64 (fallback for remaining)
            scales[6] = (d * 64.0) as i32;
            scales[7] = (dmin * 64.0) as i32;

            // Process 256 elements (or remaining if at end)
            let remaining_total = cols - col;
            let elems_in_block = remaining_total.min(QK_K);
            let sub_blocks_in_block = (elems_in_block + BLOCK_SIZE - 1) / BLOCK_SIZE;

            for sub_block in 0..sub_blocks_in_block {
                let sub_block_start = col + sub_block * BLOCK_SIZE;
                let sub_block_elems = (elems_in_block - sub_block * BLOCK_SIZE).min(BLOCK_SIZE);
                let scale_idx = sub_block.min(7);
                let scale = scales[scale_idx] as f32;

                // Read 4-bit quantized values (16 bytes for 32 elements)
                let qs_start = block_start + 16 + sub_block * 16;

                for i in 0..sub_block_elems {
                    let byte_idx = qs_start + i / 2;
                    let q = if i % 2 == 0 {
                        (weight_data[byte_idx] & 0x0F) as i8
                    } else {
                        ((weight_data[byte_idx] >> 4) & 0x0F) as i8
                    };

                    // Q: [-8, 7], dequantize to [-1, 1] roughly, then scale
                    let q_f = q as f32 / 8.0;
                    sum += q_f * scale * vec[sub_block_start + i];
                }
            }

            col += elems_in_block;
        }

        out[row] = sum / 64.0; // Normalize by the 64 we multiplied scales by
    }
}

/// Convert f16 bits to f32 (inline for performance)
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let s = (bits >> 15) & 0x1;
    let e = (bits >> 10) & 0x1f;
    let m = bits & 0x3ff;

    if e == 0 {
        if m == 0 {
            if s == 1 {
                -0.0f32
            } else {
                0.0f32
            }
        } else {
            let val = (s as f32) * (m as f32) * f32::from_bits(0x38800000);
            if s == 1 {
                -val
            } else {
                val
            }
        }
    } else if e == 31 {
        if m == 0 {
            if s == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        }
    } else {
        f32::from_bits(((s as u32) << 31) | ((e as u32 + 127 - 15) << 23) | ((m as u32) << 13))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matvec() {
        let mat = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let vec = vec![1.0, 1.0];
        let mut out = vec![0.0, 0.0];

        matvec(&mat, 2, 2, &vec, &mut out);

        assert!((out[0] - 3.0).abs() < 1e-5);
        assert!((out[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax(&mut x, 3);

        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(x[2] > x[1]);
        assert!(x[1] > x[0]);
    }

    #[test]
    fn test_f16_to_f32() {
        // Test 1.0 in f16
        let bits = 0x3C00u16;
        let val = f16_to_f32(bits);
        assert!((val - 1.0).abs() < 1e-3);

        // Test 2.0 in f16
        let bits = 0x4000u16;
        let val = f16_to_f32(bits);
        assert!((val - 2.0).abs() < 1e-3);
    }

    #[test]
    fn test_matvec_q8_0_simple() {
        // Simple test: 2x4 matrix [[1,2,3,4], [5,6,7,8]]
        let rows = 2;
        let cols = 4;
        let block_size = 32;
        let bytes_per_block = 2 + block_size;
        let blocks_per_row = 1; // 4 cols / 32 = 1 block

        // Create Q8_0 data: scale + 32 int8 values per block
        // Scale = 1.0 in fp16 = 0x3C00
        let mut weight_data = vec![0u8; rows * blocks_per_row * bytes_per_block];

        // Row 0: scale=1.0, values [1,2,3,4, 0,0,...]
        weight_data[0] = 0x00; // scale low byte
        weight_data[1] = 0x3C; // scale high byte
        weight_data[2] = 1;
        weight_data[3] = 2;
        weight_data[4] = 3;
        weight_data[5] = 4;

        // Row 1: scale=1.0, values [5,6,7,8, 0,0,...]
        let row1_offset = blocks_per_row * bytes_per_block;
        weight_data[row1_offset + 0] = 0x00;
        weight_data[row1_offset + 1] = 0x3C;
        weight_data[row1_offset + 2] = 5;
        weight_data[row1_offset + 3] = 6;
        weight_data[row1_offset + 4] = 7;
        weight_data[row1_offset + 5] = 8;

        let vec = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0, 0.0];

        matvec_q8_0(&weight_data, rows, cols, &vec, &mut out);

        println!("out[0] = {}, out[1] = {}", out[0], out[1]);

        // Expected: row0 = 1+2+3+4 = 10, row1 = 5+6+7+8 = 26
        assert!(
            (out[0] - 10.0).abs() < 0.1,
            "out[0] = {} expected 10",
            out[0]
        );
        assert!(
            (out[1] - 26.0).abs() < 0.1,
            "out[1] = {} expected 26",
            out[1]
        );
    }

    #[test]
    fn test_matvec_f32_vs_q8_0() {
        use std::time::Instant;

        let rows = 128;
        let cols = 256;

        // Create simple f32 matrix
        let mut mat_f32 = vec![0.0f32; rows * cols];
        for i in 0..mat_f32.len() {
            mat_f32[i] = ((i % 100) as f32 - 50.0) / 10.0;
        }

        // Create simple input vector
        let vec: Vec<f32> = (0..cols).map(|i| (i as f32 % 10.0) / 5.0 - 1.0).collect();

        // Dequantize to Q8_0
        let block_size = 32;
        let bytes_per_block = 2 + block_size;
        let blocks_per_row = (cols + block_size - 1) / block_size;
        let mut mat_q8_0 = vec![0u8; rows * blocks_per_row * bytes_per_block];

        for row in 0..rows {
            for block in 0..blocks_per_row {
                let block_offset = row * blocks_per_row * bytes_per_block + block * bytes_per_block;
                let col_start = block * block_size;
                let block_cols = (cols - col_start).min(block_size);

                // Find max absolute value in this block
                let mut max_abs = 0.001f32;
                for i in 0..block_cols {
                    let val = mat_f32[row * cols + col_start + i].abs();
                    if val > max_abs {
                        max_abs = val;
                    }
                }

                let scale = max_abs / 127.0;

                // Convert scale to f16 bits properly
                let scale_bits = f32_to_f16(scale);
                mat_q8_0[block_offset] = (scale_bits & 0xFF) as u8;
                mat_q8_0[block_offset + 1] = ((scale_bits >> 8) & 0xFF) as u8;

                // Write quantized values
                for i in 0..block_cols {
                    let val = (mat_f32[row * cols + col_start + i] / scale)
                        .round()
                        .max(-127.0)
                        .min(127.0) as i8;
                    mat_q8_0[block_offset + 2 + i] = val as u8;
                }
            }
        }

        let mut out_f32 = vec![0.0f32; rows];
        let mut out_q8_0 = vec![0.0f32; rows];

        let start = Instant::now();
        matvec(&mat_f32, rows, cols, &vec, &mut out_f32);
        let f32_time = start.elapsed();

        let start = Instant::now();
        matvec_q8_0(&mat_q8_0, rows, cols, &vec, &mut out_q8_0);
        let q8_0_time = start.elapsed();

        // Check accuracy
        let mut max_err = 0.0f32;
        let mut total_err = 0.0f32;
        for i in 0..rows {
            let err = (out_f32[i] - out_q8_0[i]).abs();
            if err > max_err {
                max_err = err;
            }
            total_err += err;
        }
        let avg_err = total_err / rows as f32;

        println!("matvec F32: {:?}", f32_time);
        println!("matvec Q8_0: {:?}", q8_0_time);
        println!("Max error: {}, Avg error: {}", max_err, avg_err);
        println!("First 5 f32: {:?}", &out_f32[..5]);
        println!("First 5 q8_0: {:?}", &out_q8_0[..5]);

        // For small test, error should be reasonable
        assert!(max_err < 50.0, "Q8_0 error too large: {}", max_err);
    }
}

// Simple random for testing
fn rand_simple() -> f32 {
    static mut SEED: u64 = 12345;
    unsafe {
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        (SEED >> 33) as f32 / (u32::MAX >> 9) as f32
    }
}
