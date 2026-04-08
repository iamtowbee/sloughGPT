//! Benchmark the transformer inference core.

use sloughgpt_inference::transformer::{benchmark, Transformer, TransformerConfig};

fn main() {
    println!("SloughGPT Inference Core Benchmark");
    println!("==================================\n");

    // Test small transformer
    let config = TransformerConfig {
        vocab_size: 1000,
        embedding_dim: 256,
        num_layers: 4,
        num_heads: 4,
        head_dim: 64,
        intermediate_size: 512,
        context_size: 128,
        rms_eps: 1e-5,
        rope_freq_base: 10000.0,
        rope_freq_scale: 1.0,
    };

    println!(
        "Config: {} layers, {} dim, {} vocab",
        config.num_layers, config.embedding_dim, config.vocab_size
    );

    println!("\nBuilding transformer...");
    let transformer = Transformer::new(config);
    println!("Built!\n");

    // Warmup
    println!("Warming up...");
    let tokens = vec![0usize; 1];
    for _ in 0..10 {
        let _ = transformer.forward(&tokens);
    }
    println!("Done warming up.\n");

    // Benchmark
    println!("Running benchmark (100 tokens)...\n");
    let (elapsed, tps) = benchmark(&transformer, 100);

    println!("Results:");
    println!("  Total time: {:.3}s", elapsed);
    println!("  Tokens/sec: {}", tps);

    // Memory usage estimate
    let model_size_mb = (256 * 1000 + // embedding
        4 * (256 * 256 * 4) + // attention weights
        4 * (256 * 512 * 3) + // FFN weights
        256 * 1000) as f64
        / 1_000_000.0; // lm_head

    println!("  Est. model size: {} MB", model_size_mb);

    println!("\nDone!");
}
