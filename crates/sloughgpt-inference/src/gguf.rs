//! GGUF model file format parser.
//! GGUF v3 specification: https://github.com/ggerganov/gguf

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum GGUfError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid GGUF magic number")]
    InvalidMagic,
    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),
    #[error("Missing tensor: {0}")]
    MissingTensor(String),
    #[error("Invalid tensor data")]
    InvalidTensorData,
}

pub const GGUF_MAGIC: u32 = 0x46554747;
pub const GGUF_VERSION_3: u32 = 3;

#[derive(Debug, Clone)]
pub enum GGufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Uint64(u64),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GGufValue>),
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: TensorType,
    pub offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    F32,
    F16,
    Q8_0,
    Q4K,
    Q4_0,
    Q5_0,
    Q5K,
    Q2K,
    Q3K,
    Q6K,
    I8,
    I16,
    I32,
    I64,
    F64,
}

impl TensorType {
    pub fn from_u32(val: u32) -> Option<Self> {
        match val {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4K),
            4 => Some(Self::Q5_0),
            5 => Some(Self::Q5K),
            6 => Some(Self::Q8_0),
            7 => Some(Self::Q8_0),
            8 => Some(Self::Q2K),
            9 => Some(Self::Q3K),
            10 => Some(Self::Q4K),
            11 => Some(Self::Q5K),
            12 => Some(Self::Q6K),
            13 => Some(Self::I8),
            14 => Some(Self::I16),
            15 => Some(Self::I32),
            16 => Some(Self::I64),
            17 => Some(Self::F64),
            _ => None,
        }
    }
}

pub struct GGufReader {
    file: BufReader<File>,
    pub version: u32,
    pub tensors: Vec<TensorInfo>,
    pub metadata: HashMap<String, GGufValue>,
    data_offset: usize,
}

impl GGufReader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GGUfError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut reader = Self {
            file: reader,
            version: 0,
            tensors: Vec::new(),
            metadata: HashMap::new(),
            data_offset: 0,
        };
        reader.read_header()?;
        Ok(reader)
    }

    fn read_header(&mut self) -> Result<(), GGUfError> {
        let mut magic = [0u8; 4];
        self.file.read_exact(&mut magic)?;
        let magic_val = u32::from_le_bytes(magic);
        println!("Magic: 0x{:08x} (expected 0x{:08x})", magic_val, GGUF_MAGIC);

        if magic_val != GGUF_MAGIC {
            return Err(GGUfError::InvalidMagic);
        }

        self.version = self.read_u32()?;
        println!("Version: {}", self.version);

        if self.version != GGUF_VERSION_3 {
            return Err(GGUfError::UnsupportedVersion(self.version));
        }

        let tensor_count = self.read_u64()? as usize;
        let metadata_kv_count = self.read_u64()? as usize;
        println!(
            "Tensor count: {}, Metadata KV count: {}",
            tensor_count, metadata_kv_count
        );

        let alignment = self.read_u32()?;
        println!("Alignment: {}", alignment);

        let current_pos = self.file.stream_position()?;
        println!("Position after alignment field: {}", current_pos);

        let padding = (8 - (current_pos % 8)) % 8;
        if padding > 0 {
            println!(
                "Skipping {} bytes of padding (8-byte header alignment)",
                padding
            );
            self.file.seek(SeekFrom::Current(padding as i64))?;
        }

        let kv_start = self.file.stream_position()?;
        println!("First 8 bytes at KV start: {:02x?}", {
            let mut buf = [0u8; 8];
            let _ = self.file.read_exact(&mut buf);
            self.file.seek(SeekFrom::Start(kv_start))?;
            buf
        });
        println!("Starting metadata at position: {:?}", kv_start);

        for i in 0..metadata_kv_count {
            println!("Reading metadata {}...", i);
            self.read_metadata_kv(i == 0)?;
            println!("Metadata {} read OK", i);
        }

        for _i in 0..tensor_count {
            let tensor = self.read_tensor_info()?;
            let n_elements: usize = tensor.shape.iter().product();
            if n_elements > 1_000_000_000 {
                println!(
                    "ERROR: Tensor '{}' has {} elements (too large!)",
                    tensor.name, n_elements
                );
                return Err(GGUfError::InvalidTensorData);
            }
            if _i < 5 || _i >= tensor_count - 3 {
                println!(
                    "Tensor {}: {} {:?} = {} elements",
                    _i, tensor.name, tensor.shape, n_elements
                );
            }
            self.tensors.push(tensor);
        }

        self.data_offset = self.file.stream_position()? as usize;

        let alignment = self.metadata_get::<u32>("general.alignment").unwrap_or(32);
        let aligned_offset =
            (self.data_offset + alignment as usize - 1) & !(alignment as usize - 1);
        let padding = aligned_offset - self.data_offset;

        if padding > 0 {
            self.file.seek(SeekFrom::Current(padding as i64))?;
            self.data_offset = aligned_offset;
        }

        Ok(())
    }

    fn read_tensor_info(&mut self) -> Result<TensorInfo, GGUfError> {
        let name_len = self.read_u64()? as usize;
        let mut name_buf = vec![0u8; name_len];
        self.file.read_exact(&mut name_buf)?;
        let name = String::from_utf8_lossy(&name_buf).to_string();

        let n_dims = self.read_u32()? as usize;
        let mut shape = Vec::with_capacity(n_dims);

        for _ in 0..n_dims {
            shape.push(self.read_u64()? as usize);
        }

        let dtype_val = self.read_u32()?;
        let dtype = TensorType::from_u32(dtype_val).ok_or(GGUfError::InvalidTensorData)?;

        let offset = self.read_u64()? as usize;

        Ok(TensorInfo {
            name,
            shape,
            dtype,
            offset,
        })
    }

    fn read_metadata_kv(&mut self, first_entry: bool) -> Result<(), GGUfError> {
        let start_pos = self.file.stream_position()?;

        let (key_len, key) = if first_entry {
            // First entry has no length prefix, key is 20 bytes
            let key_len = 20;
            let mut key_buf = vec![0u8; key_len];
            self.file.read_exact(&mut key_buf)?;
            let key = String::from_utf8_lossy(&key_buf).to_string();
            println!(
                "  Key: {:?} ({} bytes, file pos {})",
                key, key_len, start_pos
            );
            (key_len, key)
        } else {
            // Subsequent entries have u64 length prefix
            let key_len_raw = self.read_u64()? as usize;
            println!("  Key len from u64 prefix: {}", key_len_raw);

            let mut key_buf = vec![0u8; key_len_raw];
            self.file.read_exact(&mut key_buf)?;
            let key = String::from_utf8_lossy(&key_buf).to_string();
            println!(
                "  Key: {:?} ({} bytes, file pos {})",
                key, key_len_raw, start_pos
            );
            (key_len_raw, key)
        };

        let type_pos = self.file.stream_position()?;
        let mut type_bytes = [0u8; 4];
        self.file.read_exact(&mut type_bytes)?;
        let value_type = u32::from_le_bytes(type_bytes);
        println!(
            "  Type bytes at {}: {:02x?} = {}",
            type_pos, type_bytes, value_type
        );

        let value = match value_type {
            0 => GGufValue::Uint8(self.read_u8()?),
            1 => GGufValue::Int8(self.read_i8()?),
            2 => GGufValue::Uint16(self.read_u16()?),
            3 => GGufValue::Int16(self.read_i16()?),
            4 => GGufValue::Uint32(self.read_u32()?),
            5 => GGufValue::Int32(self.read_i32()?),
            6 => GGufValue::Float32(self.read_f32()?),
            7 => GGufValue::Bool(self.read_u8()? != 0),
            8 => GGufValue::String(self.read_string()?),
            9 => {
                let elem_type = self.read_u32()?;
                let arr_len = self.read_u64()? as usize;
                println!("  Array: {} elements, elem_type={}", arr_len, elem_type);
                if arr_len > 1000000 {
                    println!("  ERROR: Array too large!");
                    return Err(GGUfError::InvalidTensorData);
                }
                let mut arr = Vec::with_capacity(arr_len);
                for _ in 0..arr_len {
                    let val = match elem_type {
                        0 => GGufValue::Uint8(self.read_u8()?),
                        1 => GGufValue::Int8(self.read_i8()?),
                        2 => GGufValue::Uint16(self.read_u16()?),
                        3 => GGufValue::Int16(self.read_i16()?),
                        4 => GGufValue::Uint32(self.read_u32()?),
                        5 => GGufValue::Int32(self.read_i32()?),
                        6 => GGufValue::Float32(self.read_f32()?),
                        8 => GGufValue::String(self.read_string()?),
                        _ => return Err(GGUfError::InvalidTensorData),
                    };
                    arr.push(val);
                }
                GGufValue::Array(arr)
            }
            10 => GGufValue::Uint64(self.read_u64()?),
            11 => GGufValue::Int64(self.read_i64()?),
            12 => GGufValue::Float64(self.read_f64()?),
            _ => {
                println!("  ERROR: Unknown type {}", value_type);
                return Err(GGUfError::InvalidTensorData);
            }
        };

        println!("  Value read OK, inserting into metadata");
        self.metadata.insert(key, value);
        Ok(())
    }

    pub fn metadata_get<T: MetadataTryFrom>(&self, key: &str) -> Option<T> {
        self.metadata.get(key).and_then(|v| T::try_from_gguf(v))
    }

    pub fn read_tensor_data(&mut self, tensor: &TensorInfo) -> Result<Vec<f32>, GGUfError> {
        let file_offset = self.data_offset + tensor.offset;
        self.file.seek(SeekFrom::Start(file_offset as u64))?;

        match tensor.dtype {
            TensorType::F32 => {
                let n = tensor.shape.iter().product::<usize>();
                let mut data = vec![0f32; n];
                self.file.read_exact(bytemuck::cast_slice_mut(&mut data))?;
                Ok(data)
            }
            TensorType::F16 => {
                let n = tensor.shape.iter().product::<usize>();
                let mut raw = vec![0u16; n];
                self.file.read_exact(bytemuck::cast_slice_mut(&mut raw))?;
                Ok(raw.iter().map(|&x| self.f16_to_f32(x)).collect())
            }
            TensorType::Q4K => self.dequantize_q4_k(tensor),
            TensorType::Q6K => self.dequantize_q6_k(tensor),
            TensorType::Q8_0 => self.dequantize_q8_0(tensor),
            _ => Ok(Vec::new()),
        }
    }

    fn dequantize_q4_k(&mut self, tensor: &TensorInfo) -> Result<Vec<f32>, GGUfError> {
        let n_dims = tensor.shape.len();
        let nrows = if n_dims > 1 {
            tensor.shape[n_dims - 2]
        } else {
            1
        };
        let ncols = if n_dims > 0 {
            tensor.shape[n_dims - 1]
        } else {
            1
        };
        let n_elements = nrows * ncols;

        let block_size = 32usize;
        let n_blocks = (n_elements + block_size - 1) / block_size;

        let mut data = vec![0.0f32; n_elements];

        for _b in 0..n_blocks {
            let mut scale_min_buf = [0u8; 4];
            self.file.read_exact(&mut scale_min_buf)?;
            let scale = self.f16_to_f32(u16::from_le_bytes([scale_min_buf[0], scale_min_buf[1]]));
            let min_val = self.f16_to_f32(u16::from_le_bytes([scale_min_buf[2], scale_min_buf[3]]));

            let mut quant_buf = [0u8; 16];
            self.file.read_exact(&mut quant_buf)?;

            for i in 0..block_size {
                let global_idx = _b * block_size + i;
                if global_idx >= n_elements {
                    break;
                }
                let quant_idx = i / 2;
                let q = if i % 2 == 0 {
                    (quant_buf[quant_idx] & 0x0F) as i8
                } else {
                    ((quant_buf[quant_idx] >> 4) & 0x0F) as i8
                };
                let d = if q >= 8 { q as f32 - 16.0 } else { q as f32 };
                data[global_idx] = d * scale + min_val;
            }
        }

        Ok(data)
    }

    fn dequantize_q6_k(&mut self, tensor: &TensorInfo) -> Result<Vec<f32>, GGUfError> {
        let n_dims = tensor.shape.len();
        let nrows = if n_dims > 1 {
            tensor.shape[n_dims - 2]
        } else {
            1
        };
        let ncols = if n_dims > 0 {
            tensor.shape[n_dims - 1]
        } else {
            1
        };
        let n_elements = nrows * ncols;

        let block_size = 32usize;
        let n_blocks = (n_elements + block_size - 1) / block_size;

        let mut data = vec![0.0f32; n_elements];

        for _b in 0..n_blocks {
            let mut scale_buf = [0u8; 2];
            self.file.read_exact(&mut scale_buf)?;
            let scale = self.f16_to_f32(u16::from_le_bytes(scale_buf));

            let mut ql_buf = [0u8; 16];
            self.file.read_exact(&mut ql_buf)?;

            let mut qh_buf = [0u8; 16];
            self.file.read_exact(&mut qh_buf)?;

            let mut sc_buf = [0u8; 2];
            self.file.read_exact(&mut sc_buf)?;
            let extra_scale = self.f16_to_f32(u16::from_le_bytes(sc_buf));

            for i in 0..block_size {
                let global_idx = _b * block_size + i;
                if global_idx >= n_elements {
                    break;
                }

                let q = if i < 16 {
                    let v = ql_buf[i];
                    let lo = (v & 0x0F) as i8;
                    let hi = ((v >> 4) & 0x0F) as i8;
                    let dl = if lo >= 8 { lo as f32 - 16.0 } else { lo as f32 };
                    let dh = if hi >= 8 { hi as f32 - 16.0 } else { hi as f32 };
                    dl + dh * extra_scale
                } else {
                    let idx = i - 16;
                    let v = qh_buf[idx / 2];
                    let qh = if idx % 2 == 0 {
                        v & 0x0F
                    } else {
                        (v >> 4) & 0x0F
                    };
                    let d = if qh >= 8 { qh as f32 - 16.0 } else { qh as f32 };
                    d * extra_scale
                };

                data[global_idx] = q * scale;
            }
        }

        Ok(data)
    }

    fn dequantize_q8_0(&mut self, tensor: &TensorInfo) -> Result<Vec<f32>, GGUfError> {
        let n_dims = tensor.shape.len();
        let nrows = if n_dims > 1 {
            tensor.shape[n_dims - 2]
        } else {
            1
        };
        let ncols = if n_dims > 0 {
            tensor.shape[n_dims - 1]
        } else {
            1
        };
        let n_elements = nrows * ncols;

        let block_size = 32usize;
        let n_blocks = (n_elements + block_size - 1) / block_size;

        let mut data = vec![0.0f32; n_elements];

        for _b in 0..n_blocks {
            let mut scale_bytes = [0u8; 2];
            self.file.read_exact(&mut scale_bytes)?;
            let scale = self.f16_to_f32(u16::from_le_bytes(scale_bytes));

            let mut quant_buf = vec![0i8; block_size];
            self.file
                .read_exact(bytemuck::cast_slice_mut(&mut quant_buf))?;

            for i in 0..block_size {
                let global_idx = _b * block_size + i;
                if global_idx >= n_elements {
                    break;
                }
                data[global_idx] = quant_buf[i] as f32 * scale;
            }
        }

        Ok(data)
    }

    fn f16_to_f32(&self, bits: u16) -> f32 {
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

    fn read_u8(&mut self) -> Result<u8, GGUfError> {
        let mut buf = [0u8; 1];
        self.file.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    fn read_i8(&mut self) -> Result<i8, GGUfError> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16, GGUfError> {
        let mut buf = [0u8; 2];
        self.file.read_exact(&mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    fn read_i16(&mut self) -> Result<i16, GGUfError> {
        Ok(self.read_u16()? as i16)
    }

    fn read_u32(&mut self) -> Result<u32, GGUfError> {
        let mut buf = [0u8; 4];
        self.file.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_i32(&mut self) -> Result<i32, GGUfError> {
        Ok(self.read_u32()? as i32)
    }

    fn read_u64(&mut self) -> Result<u64, GGUfError> {
        let mut buf = [0u8; 8];
        self.file.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_i64(&mut self) -> Result<i64, GGUfError> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> Result<f32, GGUfError> {
        let mut buf = [0u8; 4];
        self.file.read_exact(&mut buf)?;
        Ok(f32::from_le_bytes(buf))
    }

    fn read_f64(&mut self) -> Result<f64, GGUfError> {
        let mut buf = [0u8; 8];
        self.file.read_exact(&mut buf)?;
        Ok(f64::from_le_bytes(buf))
    }

    fn read_string(&mut self) -> Result<String, GGUfError> {
        let len = self.read_u64()? as usize;
        println!("String length: {} (0x{:x})", len, len);
        if len > 1_000_000 {
            println!("ERROR: String length too large!");
            return Err(GGUfError::InvalidTensorData);
        }
        let mut buf = vec![0u8; len];
        self.file.read_exact(&mut buf)?;
        if let Some(pos) = buf.iter().position(|&b| b == 0) {
            buf.truncate(pos);
        }
        Ok(String::from_utf8_lossy(&buf).to_string())
    }

    fn read_cstring(&mut self) -> Result<String, GGUfError> {
        let mut buf = Vec::new();
        loop {
            let b = self.read_u8()?;
            if b == 0 {
                break;
            }
            buf.push(b);
        }
        Ok(String::from_utf8_lossy(&buf).to_string())
    }
}

pub trait MetadataTryFrom: Sized {
    fn try_from_gguf(val: &GGufValue) -> Option<Self>;
}

impl MetadataTryFrom for u32 {
    fn try_from_gguf(val: &GGufValue) -> Option<Self> {
        match val {
            GGufValue::Uint32(v) => Some(*v),
            GGufValue::Int32(v) => Some(*v as u32),
            _ => None,
        }
    }
}

impl MetadataTryFrom for i32 {
    fn try_from_gguf(val: &GGufValue) -> Option<Self> {
        match val {
            GGufValue::Int32(v) => Some(*v),
            GGufValue::Uint32(v) => Some(*v as i32),
            _ => None,
        }
    }
}

impl MetadataTryFrom for u64 {
    fn try_from_gguf(val: &GGufValue) -> Option<Self> {
        match val {
            GGufValue::Uint64(v) => Some(*v),
            GGufValue::Int64(v) => Some(*v as u64),
            _ => None,
        }
    }
}

impl MetadataTryFrom for f32 {
    fn try_from_gguf(val: &GGufValue) -> Option<Self> {
        match val {
            GGufValue::Float32(v) => Some(*v),
            _ => None,
        }
    }
}

impl MetadataTryFrom for String {
    fn try_from_gguf(val: &GGufValue) -> Option<Self> {
        match val {
            GGufValue::String(s) => Some(s.clone()),
            _ => None,
        }
    }
}

impl MetadataTryFrom for bool {
    fn try_from_gguf(val: &GGufValue) -> Option<Self> {
        match val {
            GGufValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

impl MetadataTryFrom for usize {
    fn try_from_gguf(val: &GGufValue) -> Option<Self> {
        match val {
            GGufValue::Uint32(v) => Some(*v as usize),
            GGufValue::Int32(v) => Some(*v as usize),
            GGufValue::Uint64(v) => Some(*v as usize),
            GGufValue::Int64(v) => Some(*v as usize),
            _ => None,
        }
    }
}
