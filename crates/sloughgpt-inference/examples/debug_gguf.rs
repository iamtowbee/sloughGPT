use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

fn main() {
    let file = File::open("/Users/mac/models/tinyllama.Q4_K_M.gguf").unwrap();
    let mut reader = BufReader::new(file);
    
    // Read header
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).unwrap();
    println!("Magic: {:02x?}", magic);
    
    let mut version_buf = [0u8; 4];
    reader.read_exact(&mut version_buf).unwrap();
    println!("Version bytes: {:02x?}", version_buf);
    
    let mut tensor_count_buf = [0u8; 8];
    reader.read_exact(&mut tensor_count_buf).unwrap();
    println!("Tensor count bytes: {:02x?}", tensor_count_buf);
    
    let mut metadata_count_buf = [0u8; 8];
    reader.read_exact(&mut metadata_count_buf).unwrap();
    println!("Metadata count bytes: {:02x?}", metadata_count_buf);
    
    let mut alignment_buf = [0u8; 4];
    reader.read_exact(&mut alignment_buf).unwrap();
    println!("Alignment bytes: {:02x?}", alignment_buf);
    
    let pos = reader.stream_position().unwrap();
    println!("\nPosition after header: {}", pos);
    
    // Skip padding
    let alignment = u32::from_le_bytes(alignment_buf);
    println!("Alignment value: {}", alignment);
    let padding = ((alignment - (pos as u32 % alignment)) % alignment) as i64;
    println!("Padding: {}", padding);
    if padding > 0 {
        reader.seek(SeekFrom::Current(padding)).unwrap();
    }
    
    let pos = reader.stream_position().unwrap();
    println!("Position after padding: {}", pos);
    
    // Read first 5 metadata keys as raw bytes
    println!("\nFirst 5 metadata entries (raw bytes):");
    for i in 0..5 {
        println!("\n--- Entry {} ---", i);
        
        // Read key length (u32)
        let mut key_len_buf = [0u8; 4];
        if reader.read_exact(&mut key_len_buf).is_err() {
            println!("End of file");
            break;
        }
        let key_len = u32::from_le_bytes(key_len_buf);
        println!("Key length: {} (0x{:08x})", key_len, key_len);
        
        // Read key bytes
        let mut key_buf = vec![0u8; key_len as usize];
        if reader.read_exact(&mut key_buf).is_err() {
            println!("Failed to read key");
            break;
        }
        println!("Key bytes: {:02x?}", key_buf);
        println!("Key string: {:?}", String::from_utf8_lossy(&key_buf));
        
        // Read type (u32)
        let mut type_buf = [0u8; 4];
        if reader.read_exact(&mut type_buf).is_err() {
            println!("Failed to read type");
            break;
        }
        let type_val = u32::from_le_bytes(type_buf);
        println!("Type: {} (0x{:08x})", type_val, type_val);
    }
}
