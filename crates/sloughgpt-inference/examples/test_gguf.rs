use sloughgpt_inference::gguf::GGufReader;

fn main() {
    let paths = [
        "/Users/mac/models/tinyllama.Q4_K_M.gguf",
        "/Users/mac/models/llama3.2-1b-q8_0.gguf",
    ];
    
    for path in paths {
        println!("\n=== Testing: {} ===\n", path);
        match GGufReader::open(path) {
            Ok(mut reader) => {
                println!("\nModel loaded successfully!");
                println!("  Version: {}", reader.version);
                println!("  Tensors: {}", reader.tensors.len());
                println!("  Metadata keys: {}", reader.metadata.len());
                
                if let Some(vocab) = reader.metadata_get::<u32>("llama.vocab_size") {
                    println!("  Vocab size: {}", vocab);
                }
                if let Some(embed) = reader.metadata_get::<u32>("llama.embedding_length") {
                    println!("  Embedding dim: {}", embed);
                }
                if let Some(layers) = reader.metadata_get::<u32>("llama.block_count") {
                    println!("  Layers: {}", layers);
                }
            }
            Err(e) => {
                println!("FAILED: {}", e);
            }
        }
    }
}
