//! PyO3 bindings for SloughGPT Inference Core

use pyo3::prelude::*;
use pyo3::pyclass;

#[pyclass]
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    #[pyo3(get, set)]
    pub vocab_size: usize,
    #[pyo3(get, set)]
    pub embedding_dim: usize,
    #[pyo3(get, set)]
    pub num_layers: usize,
    #[pyo3(get, set)]
    pub num_heads: usize,
    #[pyo3(get, set)]
    pub head_dim: usize,
    #[pyo3(get, set)]
    pub intermediate_size: usize,
    #[pyo3(get, set)]
    pub context_size: usize,
    #[pyo3(get, set)]
    pub rms_eps: f32,
    #[pyo3(get, set)]
    pub rope_freq_base: f32,
}

#[pymethods]
impl TransformerConfig {
    #[new]
    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        context_size: usize,
        rms_eps: f32,
        rope_freq_base: f32,
    ) -> Self {
        Self {
            vocab_size,
            embedding_dim,
            num_layers,
            num_heads,
            head_dim,
            intermediate_size,
            context_size,
            rms_eps,
            rope_freq_base,
        }
    }
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
        }
    }
}

#[pyclass]
pub struct Transformer {
    inner: crate::transformer::Transformer,
}

#[pymethods]
impl Transformer {
    #[new]
    pub fn new(config: TransformerConfig) -> Self {
        let cfg = crate::transformer::TransformerConfig {
            vocab_size: config.vocab_size,
            embedding_dim: config.embedding_dim,
            num_layers: config.num_layers,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
            intermediate_size: config.intermediate_size,
            context_size: config.context_size,
            rms_eps: config.rms_eps,
            rope_freq_base: config.rope_freq_base,
            rope_freq_scale: 1.0,
        };
        Self {
            inner: crate::transformer::Transformer::new(cfg),
        }
    }

    pub fn forward(&self, tokens: Vec<usize>) -> Vec<f32> {
        self.inner.forward(&tokens)
    }

    pub fn forward_single(&mut self, token: usize) -> Vec<f32> {
        self.inner.forward_single(token)
    }

    pub fn reset_cache(&mut self) {
        self.inner.reset_cache();
    }

    #[staticmethod]
    pub fn load_from_gguf(path: &str) -> PyResult<Self> {
        let inner = crate::transformer::Transformer::load_from_gguf(path)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(Self { inner })
    }

    pub fn sample(&self, logits: Vec<f32>, temperature: f32, top_k: usize, top_p: f32) -> usize {
        self.inner.sample(&logits, temperature, top_k, top_p)
    }

    #[getter]
    pub fn config(&self) -> TransformerConfig {
        TransformerConfig {
            vocab_size: self.inner.config.vocab_size,
            embedding_dim: self.inner.config.embedding_dim,
            num_layers: self.inner.config.num_layers,
            num_heads: self.inner.config.num_heads,
            head_dim: self.inner.config.head_dim,
            intermediate_size: self.inner.config.intermediate_size,
            context_size: self.inner.config.context_size,
            rms_eps: self.inner.config.rms_eps,
            rope_freq_base: self.inner.config.rope_freq_base,
        }
    }
}

pub fn benchmark(transformer: &Transformer, num_tokens: usize) -> (f64, f64) {
    crate::transformer::benchmark(&transformer.inner, num_tokens)
}

#[pymodule]
fn sloughgpt_inference(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TransformerConfig>()?;
    m.add_class::<Transformer>()?;
    Ok(())
}
