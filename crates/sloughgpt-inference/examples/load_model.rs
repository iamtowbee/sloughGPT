use sloughgpt_inference::transformer::Transformer;

fn main() {
    let paths = [
        (
            "/Users/mac/models/tinyllama.Q4_K_M.gguf",
            "TinyLlama Q4_K_M",
        ),
        (
            "/Users/mac/models/llama3.2-1b-q8_0.gguf",
            "Llama3.2-1B Q8_0",
        ),
    ];

    for (path, name) in paths {
        println!("\n=== Loading: {} ===\n", name);
        match Transformer::load_from_gguf(path) {
            Ok(transformer) => {
                println!("Model loaded successfully!");
                println!(
                    "  Config: {} layers, {} dim, {} vocab",
                    transformer.config.num_layers,
                    transformer.config.embedding_dim,
                    transformer.config.vocab_size
                );
            }
            Err(e) => {
                println!("FAILED: {}", e);
            }
        }
    }
}
