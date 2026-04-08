//! Minimal transformer for inference.

use crate::gguf::GGufReader;
use crate::tensor::{matvec, rms_norm, silu, softmax};

#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub context_size: usize,
    pub rms_eps: f32,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            embedding_dim: 2048,
            num_layers: 24,
            num_heads: 16,
            head_dim: 128,
            intermediate_size: 5632,
            context_size: 4096,
            rms_eps: 1e-5,
            rope_freq_base: 10000.0,
            rope_freq_scale: 1.0,
        }
    }
}

pub struct Transformer {
    pub config: TransformerConfig,
    embedding: Vec<f32>,
    layers: Vec<TransformerLayer>,
    lm_head: Vec<f32>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    kv_cache: Option<KVCache>,
}

pub struct KVCache {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub seq_len: usize,
}

impl KVCache {
    pub fn new(num_layers: usize, num_heads: usize, head_dim: usize, max_seq: usize) -> Self {
        let k = vec![vec![0.0f32; num_heads * head_dim]; num_layers];
        let v = vec![vec![0.0f32; num_heads * head_dim]; num_layers];
        Self { k, v, seq_len: 0 }
    }
}

struct TransformerLayer {
    attention: Attention,
    feed_forward: FeedForward,
    input_layernorm: Vec<f32>,
    post_attention_layernorm: Vec<f32>,
}

struct Attention {
    num_heads: usize,
    head_dim: usize,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    o_proj: Vec<f32>,
}

struct FeedForward {
    gate_proj: Vec<f32>,
    up_proj: Vec<f32>,
    down_proj: Vec<f32>,
    hidden_dim: usize,
    output_dim: usize,
}

impl Transformer {
    pub fn new(config: TransformerConfig) -> Self {
        let dim = config.embedding_dim;
        let hidden_dim = config.intermediate_size;
        let num_heads = config.num_heads;
        let head_dim = config.head_dim;
        let num_layers = config.num_layers;
        let context_size = config.context_size;
        let vocab_size = config.vocab_size;
        let rope_freq_base = config.rope_freq_base;

        let embedding = vec![0.0f32; vocab_size * dim];
        let lm_head = vec![0.0f32; dim * vocab_size];

        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(TransformerLayer::new(dim, num_heads, head_dim, hidden_dim));
        }

        let max_seq = context_size;
        let rope_len = max_seq.max(512);
        let mut rope_cos = vec![0.0f32; rope_len * head_dim / 2];
        let mut rope_sin = vec![0.0f32; rope_len * head_dim / 2];
        compute_rope_freqs(
            &mut rope_cos,
            &mut rope_sin,
            rope_len,
            head_dim,
            rope_freq_base,
        );

        Self {
            config,
            embedding,
            layers,
            lm_head,
            rope_cos,
            rope_sin,
            kv_cache: Some(KVCache::new(num_layers, num_heads, head_dim, context_size)),
        }
    }

    pub fn from_gguf(reader: &mut GGufReader) -> Result<Self, String> {
        let config = TransformerConfig {
            vocab_size: reader.metadata_get("llama.vocab_size").unwrap_or(32000),
            embedding_dim: reader
                .metadata_get("llama.embedding_length")
                .unwrap_or(2048),
            num_layers: reader.metadata_get("llama.block_count").unwrap_or(22),
            num_heads: reader
                .metadata_get("llama.attention.head_count")
                .unwrap_or(16),
            head_dim: reader
                .metadata_get("llama.attention.head_count_kv")
                .unwrap_or(16),
            intermediate_size: reader
                .metadata_get("llama.feed_forward_length")
                .unwrap_or(5632),
            context_size: reader.metadata_get("llama.context_length").unwrap_or(2048),
            rms_eps: reader
                .metadata_get("llama.attention.layer_norm_rms_epsilon")
                .unwrap_or(1e-5),
            rope_freq_base: reader
                .metadata_get("llama.rope.freq_base")
                .unwrap_or(10000.0),
            rope_freq_scale: reader.metadata_get("llama.rope.scale").unwrap_or(1.0),
        };

        let mut transformer = Self::new(config);

        let tensor_names: Vec<String> = reader.tensors.iter().map(|t| t.name.clone()).collect();
        let tensors_clone = reader.tensors.clone();

        let find_idx = |name: &str| tensors_clone.iter().position(|t| t.name.contains(name));

        if let Some(idx) = find_idx("token_embd") {
            if let Ok(data) = reader.read_tensor_data(&tensors_clone[idx]) {
                if data.len() == transformer.embedding.len() {
                    transformer.embedding = data;
                }
            }
        }

        if let Some(idx) = find_idx("output") {
            if let Ok(data) = reader.read_tensor_data(&tensors_clone[idx]) {
                if data.len() == transformer.lm_head.len() {
                    transformer.lm_head = data;
                }
            }
        }

        for (i, layer) in transformer.layers.iter_mut().enumerate() {
            let prefix = format!("blk.{}", i);

            if let Some(idx) = find_idx(&format!("{}.attn_q", prefix)) {
                if let Ok(data) = reader.read_tensor_data(&tensors_clone[idx]) {
                    layer.attention.q_proj = data;
                }
            }
            if let Some(idx) = find_idx(&format!("{}.attn_k", prefix)) {
                if let Ok(data) = reader.read_tensor_data(&tensors_clone[idx]) {
                    layer.attention.k_proj = data;
                }
            }
            if let Some(idx) = find_idx(&format!("{}.attn_v", prefix)) {
                if let Ok(data) = reader.read_tensor_data(&tensors_clone[idx]) {
                    layer.attention.v_proj = data;
                }
            }
            if let Some(idx) = find_idx(&format!("{}.attn_output", prefix)) {
                if let Ok(data) = reader.read_tensor_data(&tensors_clone[idx]) {
                    layer.attention.o_proj = data;
                }
            }
            if let Some(idx) = find_idx(&format!("{}.attn_norm", prefix)) {
                if let Ok(data) = reader.read_tensor_data(&tensors_clone[idx]) {
                    layer.input_layernorm = data;
                }
            }
            if let Some(idx) = find_idx(&format!("{}.ffn_norm", prefix)) {
                if let Ok(data) = reader.read_tensor_data(&tensors_clone[idx]) {
                    layer.post_attention_layernorm = data;
                }
            }
            if let Some(idx) = find_idx(&format!("{}.ffn_gate", prefix)) {
                if let Ok(data) = reader.read_tensor_data(&tensors_clone[idx]) {
                    layer.feed_forward.gate_proj = data;
                }
            }
            if let Some(idx) = find_idx(&format!("{}.ffn_up", prefix)) {
                if let Ok(data) = reader.read_tensor_data(&tensors_clone[idx]) {
                    layer.feed_forward.up_proj = data;
                }
            }
            if let Some(idx) = find_idx(&format!("{}.ffn_down", prefix)) {
                if let Ok(data) = reader.read_tensor_data(&tensors_clone[idx]) {
                    layer.feed_forward.down_proj = data;
                }
            }
        }

        Ok(transformer)
    }

    pub fn forward_single(&mut self, token: usize) -> Vec<f32> {
        let dim = self.config.embedding_dim;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let hidden_size = num_heads * head_dim;

        // Get embedding for single token
        let src = token * dim;
        let mut hidden = self.embedding[src..src + dim].to_vec();

        // RMSNorm before attention
        let mut normed = vec![0.0f32; dim];
        rms_norm(&hidden, &self.embedding, self.config.rms_eps, &mut normed);

        // Process each layer with KV cache
        let cache = self.kv_cache.as_mut().unwrap();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Pre-attention RMSNorm
            rms_norm(&hidden, &layer.input_layernorm, 1e-5, &mut normed);

            // Project to Q, K, V
            let mut q = vec![0.0f32; hidden_size];
            let mut k = vec![0.0f32; hidden_size];
            let mut v = vec![0.0f32; hidden_size];

            matvec(&layer.attention.q_proj, dim, dim, &normed, &mut q);
            matvec(&layer.attention.k_proj, dim, dim, &normed, &mut k);
            matvec(&layer.attention.v_proj, dim, dim, &normed, &mut v);

            // Apply RoPE
            let pos = cache.seq_len.min(self.rope_cos.len() / head_dim);
            for h in 0..num_heads {
                for d in 0..head_dim / 2 {
                    let q0 = q[h * head_dim + d];
                    let q1 = q[h * head_dim + d + head_dim / 2];
                    let cos = self
                        .rope_cos
                        .get(pos * head_dim / 2 + d)
                        .copied()
                        .unwrap_or(0.0);
                    let sin = self
                        .rope_sin
                        .get(pos * head_dim / 2 + d)
                        .copied()
                        .unwrap_or(0.0);
                    q[h * head_dim + d] = q0 * cos - q1 * sin;
                    q[h * head_dim + d + head_dim / 2] = q0 * sin + q1 * cos;

                    let k0 = k[h * head_dim + d];
                    let k1 = k[h * head_dim + d + head_dim / 2];
                    k[h * head_dim + d] = k0 * cos - k1 * sin;
                    k[h * head_dim + d + head_dim / 2] = k0 * sin + k1 * cos;
                }
            }

            // Cache K and V
            cache.k[layer_idx].copy_from_slice(&k);
            cache.v[layer_idx].copy_from_slice(&v);

            // Attention with cached K,V
            let mut attn_out = vec![0.0f32; hidden_size];
            let scale = 1.0 / (head_dim as f32).sqrt();

            // Simplified attention for single token
            for h in 0..num_heads {
                let q_off = h * head_dim;
                let k_off = h * head_dim;
                let v_off = h * head_dim;

                // Q @ K^T (only need score with current K)
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q[q_off + d] * k[k_off + d] * scale;
                }
                score = score.exp();

                // Weighted sum of V
                for d in 0..head_dim {
                    attn_out[q_off + d] = score * v[v_off + d];
                }
            }

            // Output projection
            let mut proj_out = vec![0.0f32; dim];
            matvec(&layer.attention.o_proj, dim, dim, &attn_out, &mut proj_out);

            // Residual
            for i in 0..dim {
                hidden[i] += proj_out[i];
            }

            // FFN
            rms_norm(&hidden, &layer.post_attention_layernorm, 1e-5, &mut normed);
            let ff_out = layer.feed_forward.forward(&normed, dim);
            for i in 0..dim {
                hidden[i] += ff_out[i];
            }
        }

        cache.seq_len += 1;

        // Final RMSNorm
        let mut final_hidden = vec![0.0f32; dim];
        rms_norm(
            &hidden,
            &self.embedding,
            self.config.rms_eps,
            &mut final_hidden,
        );

        // LM head
        let mut logits = vec![0.0f32; self.config.vocab_size];
        matvec(
            &self.lm_head,
            self.config.vocab_size,
            dim,
            &final_hidden,
            &mut logits,
        );

        logits
    }

    pub fn reset_cache(&mut self) {
        if let Some(cache) = self.kv_cache.as_mut() {
            cache.seq_len = 0;
        }
    }

    pub fn load_from_gguf(path: &str) -> Result<Self, String> {
        println!("Opening GGUF file: {}", path);
        let mut reader = GGufReader::open(path).map_err(|e| e.to_string())?;
        println!("GGUF file opened successfully");
        println!("GGUF Model Info:");
        println!("  Version: {}", reader.version);
        println!("  Tensors: {}", reader.tensors.len());
        println!("  Metadata keys: {}", reader.metadata.len());

        println!("GGUF Model Info:");
        println!("  Version: {}", reader.version);
        println!("  Tensors: {}", reader.tensors.len());

        // Get model metadata
        let vocab_size = reader
            .metadata_get::<u32>("llama.vocab_size")
            .unwrap_or(32000);
        let embedding_dim = reader
            .metadata_get::<u32>("llama.embedding_length")
            .unwrap_or(2048);
        let num_layers = reader
            .metadata_get::<u32>("llama.block_count")
            .unwrap_or(22);
        let num_heads = reader
            .metadata_get::<u32>("llama.attention.head_count")
            .unwrap_or(16);
        let head_dim = embedding_dim / num_heads;
        let intermediate_size = reader
            .metadata_get::<u32>("llama.feed_forward_length")
            .unwrap_or(5632);
        let context_size = reader
            .metadata_get::<u32>("llama.context_length")
            .unwrap_or(2048);
        let rms_eps = reader
            .metadata_get::<f32>("llama.attention.layer_norm_rms_epsilon")
            .unwrap_or(1e-5);
        let rope_freq_base = reader
            .metadata_get::<f32>("llama.rope.freq_base")
            .unwrap_or(10000.0);

        println!(
            "  Vocab: {}, Embedding: {}, Layers: {}, Heads: {}",
            vocab_size, embedding_dim, num_layers, num_heads
        );

        let config = TransformerConfig {
            vocab_size: vocab_size as usize,
            embedding_dim: embedding_dim as usize,
            num_layers: num_layers as usize,
            num_heads: num_heads as usize,
            head_dim: head_dim as usize,
            intermediate_size: intermediate_size as usize,
            context_size: context_size as usize,
            rms_eps,
            rope_freq_base,
            rope_freq_scale: 1.0,
        };

        let mut transformer = Self::new(config);

        // Clone tensors for reading
        let tensor_names: Vec<String> = reader.tensors.iter().map(|t| t.name.clone()).collect();
        let tensors_clone = reader.tensors.clone();

        let find_idx = |name: &str| tensors_clone.iter().position(|t| t.name.contains(name));

        // Load embedding
        if let Some(idx) = find_idx("token_embd") {
            let tensor = &tensors_clone[idx];
            println!(
                "Embedding tensor: {} {:?} {:?}",
                tensor.name, tensor.shape, tensor.dtype
            );
            let expected = tensor.shape.iter().product::<usize>();
            if expected > 100_000_000 {
                println!("ERROR: Embedding tensor too large: {}", expected);
            } else {
                println!("Loading embedding ({} elements)...", expected);
                match reader.read_tensor_data(&tensor) {
                    Ok(data) => {
                        println!("Loaded {} elements", data.len());
                        if data.len() == transformer.embedding.len() {
                            transformer.embedding = data;
                        }
                    }
                    Err(e) => println!("Failed to load: {}", e),
                }
            }
        }

        // Load output
        if let Some(idx) = find_idx("output") {
            println!("Loading output...");
            if let Ok(data) = reader.read_tensor_data(&tensors_clone[idx]) {
                if data.len() == transformer.lm_head.len() {
                    transformer.lm_head = data;
                }
            }
        }

        // Load layer weights
        for (i, layer) in transformer.layers.iter_mut().enumerate() {
            let prefix = format!("blk.{}", i);

            if let Some(idx) = find_idx(&format!("{}.attn_q", prefix)) {
                if let Ok(data) = reader.read_tensor_data(&tensors_clone[idx]) {
                    layer.attention.q_proj = data;
                }
            }
            if let Some(idx) = find_idx(&format!("{}.attn_k", prefix)) {
                if let Ok(data) = reader.read_tensor_data(&tensors_clone[idx]) {
                    layer.attention.k_proj = data;
                }
            }
            if let Some(idx) = find_idx(&format!("{}.attn_v", prefix)) {
                if let Ok(data) = reader.read_tensor_data(&tensors_clone[idx]) {
                    layer.attention.v_proj = data;
                }
            }
            if let Some(idx) = find_idx(&format!("{}.attn_output", prefix)) {
                if let Ok(data) = reader.read_tensor_data(&tensors_clone[idx]) {
                    layer.attention.o_proj = data;
                }
            }
            if let Some(idx) = find_idx(&format!("{}.attn_norm", prefix)) {
                if let Ok(data) = reader.read_tensor_data(&tensors_clone[idx]) {
                    layer.input_layernorm = data;
                }
            }
            if let Some(idx) = find_idx(&format!("{}.ffn_norm", prefix)) {
                if let Ok(data) = reader.read_tensor_data(&tensors_clone[idx]) {
                    layer.post_attention_layernorm = data;
                }
            }
            if let Some(idx) = find_idx(&format!("{}.ffn_gate", prefix)) {
                if let Ok(data) = reader.read_tensor_data(&tensors_clone[idx]) {
                    layer.feed_forward.gate_proj = data;
                }
            }
            if let Some(idx) = find_idx(&format!("{}.ffn_up", prefix)) {
                if let Ok(data) = reader.read_tensor_data(&tensors_clone[idx]) {
                    layer.feed_forward.up_proj = data;
                }
            }
            if let Some(idx) = find_idx(&format!("{}.ffn_down", prefix)) {
                if let Ok(data) = reader.read_tensor_data(&tensors_clone[idx]) {
                    layer.feed_forward.down_proj = data;
                }
            }

            if i % 5 == 0 || i == num_layers as usize - 1 {
                println!("Loaded layer {}/{}", i + 1, num_layers);
            }
        }

        println!("Model loaded successfully!");
        Ok(transformer)
    }

    pub fn forward(&self, tokens: &[usize]) -> Vec<f32> {
        let seq_len = tokens.len();
        let dim = self.config.embedding_dim;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;

        // Token embeddings
        let mut hidden = vec![0.0f32; seq_len * dim];
        for (i, &token) in tokens.iter().enumerate() {
            let src = token * dim;
            let dst = i * dim;
            hidden[dst..dst + dim].copy_from_slice(&self.embedding[src..src + dim]);
        }

        // Process layers
        for layer in &self.layers {
            hidden = layer.forward(
                &hidden,
                seq_len,
                dim,
                num_heads,
                head_dim,
                &self.rope_cos,
                &self.rope_sin,
            );
        }

        // Final RMSNorm
        let mut final_hidden = vec![0.0f32; seq_len * dim];
        rms_norm(
            &hidden,
            &self.embedding,
            self.config.rms_eps,
            &mut final_hidden,
        );

        // LM head
        let mut logits = vec![0.0f32; self.config.vocab_size];
        let last_hidden = &final_hidden[(seq_len - 1) * dim..seq_len * dim];
        matvec(
            &self.lm_head,
            self.config.vocab_size,
            dim,
            last_hidden,
            &mut logits,
        );

        logits
    }

    pub fn sample(&self, logits: &[f32], temperature: f32, top_k: usize, top_p: f32) -> usize {
        if temperature <= 0.0 || temperature.is_nan() {
            return logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
        }

        let scale = 1.0 / temperature;
        let mut probs: Vec<f32> = logits.iter().map(|x| x * scale).collect();

        // Top-k filtering
        if top_k > 0 && top_k < probs.len() {
            let mut sorted = probs.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let threshold = sorted[top_k];
            for p in probs.iter_mut() {
                if *p < threshold {
                    *p = f32::NEG_INFINITY;
                }
            }
        }

        // Convert to probabilities
        let max_val = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for p in probs.iter_mut() {
            *p = (*p - max_val).exp();
            sum += *p;
        }
        for p in probs.iter_mut() {
            *p /= sum;
        }

        // Top-p (nucleus) sampling
        if top_p < 1.0 {
            let mut cumsum = 0.0f32;
            let mut filtered: Vec<(usize, f32)> = probs
                .iter()
                .enumerate()
                .filter(|(_, p)| p.is_finite())
                .map(|(i, &p)| (i, p))
                .collect();
            filtered.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            for (i, p) in filtered {
                cumsum += p;
                if cumsum > top_p {
                    probs[i] = f32::NEG_INFINITY;
                }
            }

            let max_val = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for p in probs.iter_mut() {
                if p.is_finite() {
                    *p = (*p - max_val).exp();
                    sum += *p;
                }
            }
            for p in probs.iter_mut() {
                if p.is_finite() {
                    *p /= sum;
                }
            }
        }

        // Sample
        let r: f32 = fastrand();
        let mut cumsum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= r {
                return i;
            }
        }

        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

impl TransformerLayer {
    fn new(
        embedding_dim: usize,
        num_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
    ) -> Self {
        let dim = embedding_dim;
        Self {
            attention: Attention::new(dim, num_heads, head_dim),
            feed_forward: FeedForward::new(dim, intermediate_size),
            input_layernorm: vec![1.0; dim],
            post_attention_layernorm: vec![1.0; dim],
        }
    }

    fn forward(
        &self,
        input: &[f32],
        seq_len: usize,
        dim: usize,
        num_heads: usize,
        head_dim: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> Vec<f32> {
        // Pre-attention RMSNorm (per token)
        let mut normed = vec![0.0f32; seq_len * dim];
        for i in 0..seq_len {
            let src = i * dim;
            let dst = i * dim;
            rms_norm(
                &input[src..src + dim],
                &self.input_layernorm,
                1e-5,
                &mut normed[dst..dst + dim],
            );
        }

        // Attention
        let attn_out = self.attention.forward(
            &normed, seq_len, dim, num_heads, head_dim, rope_cos, rope_sin,
        );

        // Residual
        let mut hidden: Vec<f32> = input
            .iter()
            .zip(attn_out.iter())
            .map(|(a, b)| a + b)
            .collect();

        // Post-attention RMSNorm (per token)
        for i in 0..seq_len {
            let src = i * dim;
            let dst = i * dim;
            rms_norm(
                &hidden[src..src + dim],
                &self.post_attention_layernorm,
                1e-5,
                &mut normed[dst..dst + dim],
            );
        }

        // FeedForward with SwiGLU (per token)
        let mut ff_out = vec![0.0f32; seq_len * dim];
        for i in 0..seq_len {
            let src = i * dim;
            let dst = i * dim;
            let token_ff = self.feed_forward.forward(&normed[src..src + dim], dim);
            ff_out[dst..dst + dim].copy_from_slice(&token_ff);
        }

        // Final residual
        hidden
            .iter_mut()
            .zip(ff_out.iter())
            .for_each(|(h, f)| *h += f);

        hidden
    }
}

impl Attention {
    fn new(dim: usize, num_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            head_dim,
            q_proj: vec![0.0; dim * dim],
            k_proj: vec![0.0; dim * dim],
            v_proj: vec![0.0; dim * dim],
            o_proj: vec![0.0; dim * dim],
        }
    }

    fn forward(
        &self,
        input: &[f32],
        seq_len: usize,
        dim: usize,
        num_heads: usize,
        head_dim: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> Vec<f32> {
        let hidden_size = num_heads * head_dim;

        // Project to Q, K, V
        let mut q = vec![0.0f32; seq_len * hidden_size];
        let mut k = vec![0.0f32; seq_len * hidden_size];
        let mut v = vec![0.0f32; seq_len * hidden_size];

        let mut q_full = vec![0.0f32; seq_len * dim];
        let mut k_full = vec![0.0f32; seq_len * dim];
        let mut v_full = vec![0.0f32; seq_len * dim];

        for i in 0..seq_len {
            let src = i * dim;
            matvec(
                &self.q_proj,
                dim,
                dim,
                &input[src..src + dim],
                &mut q_full[src..src + dim],
            );
            matvec(
                &self.k_proj,
                dim,
                dim,
                &input[src..src + dim],
                &mut k_full[src..src + dim],
            );
            matvec(
                &self.v_proj,
                dim,
                dim,
                &input[src..src + dim],
                &mut v_full[src..src + dim],
            );
        }

        // Reshape and apply RoPE
        for h in 0..num_heads {
            for i in 0..seq_len {
                let q_off = i * hidden_size + h * head_dim;
                let k_off = i * hidden_size + h * head_dim;
                let q_src = i * dim + h * head_dim;
                let k_src = i * dim + h * head_dim;

                // Apply RoPE to query and key
                for j in 0..head_dim / 2 {
                    let q0 = q_full[q_src + j];
                    let q1 = q_full[q_src + j + head_dim / 2];
                    let k0 = k_full[k_src + j];
                    let k1 = k_full[k_src + j + head_dim / 2];

                    let cos = rope_cos.get(j).copied().unwrap_or(0.0);
                    let sin = rope_sin.get(j).copied().unwrap_or(0.0);

                    q[q_off + j] = q0 * cos - q1 * sin;
                    q[q_off + j + head_dim / 2] = q0 * sin + q1 * cos;
                    k[k_off + j] = k0 * cos - k1 * sin;
                    k[k_off + j + head_dim / 2] = k0 * sin + k1 * cos;
                }
            }
        }

        // Attention scores: Q @ K^T
        let mut scores = vec![0.0f32; seq_len * seq_len];
        let k_transposed: Vec<f32> = k.chunks(head_dim).flat_map(|h| h.iter().copied()).collect();

        for i in 0..seq_len {
            for j in 0..seq_len {
                let q_row = i * hidden_size;
                let k_row = j * head_dim;
                let mut sum = 0.0f32;
                for h in 0..num_heads {
                    for d in 0..head_dim {
                        sum += q[q_row + h * head_dim + d] * k[j * hidden_size + h * head_dim + d];
                    }
                }
                scores[i * seq_len + j] = sum / (head_dim as f32).sqrt();
            }
        }

        // Softmax
        softmax(&mut scores, seq_len);

        // Apply attention to values
        let mut attn_output = vec![0.0f32; seq_len * hidden_size];
        for i in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for j in 0..seq_len {
                        sum += scores[i * seq_len + j] * v[j * hidden_size + h * head_dim + d];
                    }
                    attn_output[i * hidden_size + h * head_dim + d] = sum;
                }
            }
        }

        // Output projection
        let mut output = vec![0.0f32; seq_len * dim];
        let hidden_size = num_heads * head_dim;

        for i in 0..seq_len {
            let attn_start = i * hidden_size;
            let attn_end = attn_start + hidden_size;
            let attn_chunk = &attn_output[attn_start..attn_end];

            // Reshape from [num_heads, head_dim] to [dim]
            let mut token_hidden = vec![0.0f32; dim];
            for h in 0..num_heads {
                for d in 0..head_dim {
                    token_hidden[h * head_dim + d] = attn_chunk[h * head_dim + d];
                }
            }

            let mut o_out = vec![0.0f32; dim];
            matvec(&self.o_proj, dim, dim, &token_hidden, &mut o_out);
            output[i * dim..(i + 1) * dim].copy_from_slice(&o_out);
        }

        output
    }
}

impl FeedForward {
    fn new(dim: usize, intermediate_size: usize) -> Self {
        Self {
            gate_proj: vec![0.0; intermediate_size * dim],
            up_proj: vec![0.0; intermediate_size * dim],
            down_proj: vec![0.0; dim * intermediate_size],
            hidden_dim: intermediate_size,
            output_dim: dim,
        }
    }

    fn forward(&self, input: &[f32], dim: usize) -> Vec<f32> {
        let hidden_dim = self.hidden_dim;

        // Gate: [hidden_dim, dim] @ [dim] = [hidden_dim]
        let mut gate = vec![0.0f32; hidden_dim];
        matvec(&self.gate_proj, hidden_dim, dim, input, &mut gate);
        silu(&mut gate);

        // Up: [hidden_dim, dim] @ [dim] = [hidden_dim]
        let mut up = vec![0.0f32; hidden_dim];
        matvec(&self.up_proj, hidden_dim, dim, input, &mut up);

        // SwiGLU: gate * up
        for i in 0..hidden_dim {
            gate[i] *= up[i];
        }

        // Down: [dim, hidden_dim] @ [hidden_dim] = [dim]
        let mut down = vec![0.0f32; dim];
        matvec(&self.down_proj, dim, hidden_dim, &gate, &mut down);

        down
    }
}

fn compute_rope_freqs(
    cos: &mut [f32],
    sin: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    base: f32,
) {
    let inv_freq = 1.0 / base;
    for pos in 0..seq_len {
        for i in 0..head_dim / 2 {
            let theta = (i as f32) * inv_freq;
            let freq = theta * (pos as f32);
            cos[pos * head_dim / 2 + i] = freq.cos();
            sin[pos * head_dim / 2 + i] = freq.sin();
        }
    }
}

// Simple fast random for sampling
fn fastrand() -> f32 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hash, Hasher};
    static STATE: std::sync::OnceLock<RandomState> = std::sync::OnceLock::new();
    let rs = STATE.get_or_init(|| RandomState::new());
    let mut hasher = rs.build_hasher();
    std::time::Instant::now().hash(&mut hasher);
    (hasher.finish() as f32) / (u64::MAX as f32)
}

pub fn benchmark(transformer: &Transformer, num_tokens: usize) -> (f64, f64) {
    use std::time::Instant;

    let tokens = vec![1usize; 1];

    let start = Instant::now();
    let mut generated = Vec::with_capacity(num_tokens);

    for _ in 0..num_tokens {
        let logits = transformer.forward(&tokens);
        let next = transformer.sample(&logits, 0.7, 40, 0.9);
        generated.push(next);
    }

    let elapsed = start.elapsed().as_secs_f64();
    let tps = num_tokens as f64 / elapsed;

    (elapsed, tps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_creation() {
        let config = TransformerConfig::default();
        let transformer = Transformer::new(config);
        assert_eq!(transformer.layers.len(), 24);
    }

    #[test]
    fn test_forward() {
        let transformer = Transformer::new(TransformerConfig {
            vocab_size: 100,
            embedding_dim: 64,
            num_layers: 2,
            num_heads: 4,
            head_dim: 16,
            intermediate_size: 128,
            context_size: 32,
            rms_eps: 1e-5,
            rope_freq_base: 10000.0,
            rope_freq_scale: 1.0,
        });

        let tokens = vec![1, 2, 3];
        let logits = transformer.forward(&tokens);
        assert_eq!(logits.len(), 100);
    }
}
