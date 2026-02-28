# .sou Format - SloughGPT Model File

## Overview

The `.sou` format is SloughGPT's model configuration format, inspired by Ollama's Modelfile but with enhancements for our ecosystem:
- Built-in personality embeddings
- Custom quantization support
- RAG/knowledge base integration
- Enterprise features (ACL, watermarking)

---

## Format Specification

### File Structure

```
.sou file (text-based configuration)
├── FROM                 # Base model (required)
├── PARAMETER            # Model runtime parameters
├── TEMPLATE             # Prompt template
├── SYSTEM               # System prompt
├── PERSONALITY          # Personality configuration
├── KNOWLEDGE            # RAG/knowledge base paths
├── ADAPTER             # LoRA adapters
├── LICENSE             # Legal license
├── MESSAGE             # Conversation history
└── METADATA            # Model metadata
```

---

## Instructions Reference

| Instruction | Required | Description |
|-------------|----------|-------------|
| `FROM` | Yes | Base model (local path or HF model ID) |
| `PARAMETER` | No | Runtime parameters (temperature, context, etc.) |
| `TEMPLATE` | No | Custom prompt template |
| `SYSTEM` | No | System prompt/message |
| `PERSONALITY` | No | Personality configuration |
| `KNOWLEDGE` | No | Knowledge base paths for RAG |
| `ADAPTER` | No | LoRA adapter paths |
| `LICENSE` | No | Legal license |
| `MESSAGE` | No | Pre-defined conversation history |
| `METADATA` | No | Model metadata (author, version, etc.) |

---

## Parameter Reference

### Generation Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `temperature` | float | 0.7 | 0.0-2.0 | Controls randomness |
| `top_p` | float | 0.9 | 0.0-1.0 | Nucleus sampling |
| `top_k` | int | 40 | 1-100 | Token selection pool |
| `max_tokens` | int | 2048 | 1-8192 | Max response length |
| `repeat_penalty` | float | 1.1 | 0.0-2.0 | Repetition penalty |
| `presence_penalty` | float | 0.0 | -2.0-2.0 | Presence penalty |
| `frequency_penalty` | float | 0.0 | -2.0-2.0 | Frequency penalty |
| `mirostat` | int | 0 | 0-2 | Mirostat sampling mode |
| `mirostat_tau` | float | 5.0 | 0.0-10.0 | Mirostat tau |
| `mirostat_eta` | float | 0.1 | 0.0-1.0 | Mirostat eta |

### Context Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_ctx` | int | 4096 | Context window size |
| `num_keep` | int | 0 | Tokens to keep from prompt |
| `num_thread` | int | auto | CPU threads |
| `num_gpu` | int | auto | GPU layers to use |

### Model Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Model name |
| `quantization` | string | Quantization level (Q4_K, Q8_0, F16, etc.) |

---

## Examples

### Basic .sou File

```sou
# Basic SloughGPT Model File
FROM llama3.2

# Generation parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER max_tokens 2048

# System prompt
SYSTEM You are a helpful AI assistant.

# Model metadata
METADATA author "SloughGPT Team"
METADATA version "1.0.0"
```

### Personality-Enabled .sou

```sou
# Custom personality model
FROM llama3.2

# Personality configuration
PERSONALITY
    warmth 0.8
    formality 0.3
    creativity 0.7
    empathy 0.9
    humor 0.6
    END

# Custom parameters for personality
PARAMETER temperature 0.9
PARAMETER top_k 60

SYSTEM You are a warm, empathetic assistant who loves creative problem-solving.

METADATA personality "empathetic-creative"
METADATA author "AI Personality Lab"
```

### RAG-Enabled .sou

```sou
# Knowledge-augmented model
FROM mistral

# Knowledge base paths
KNOWLEDGE
    /data/company-docs
    /data/product-manuals
    /data/faq
    END

# RAG configuration
PARAMETER retrieval_top_k 5
PARAMETER rerank true

SYSTEM You are a company documentation assistant. Use the provided knowledge base to answer questions accurately.

METADATA knowledge_enabled true
METADATA knowledge_version "2024.01"
```

### LoRA-Adapted .sou

```sou
# Fine-tuned model with LoRA
FROM llama3.2

# LoRA adapters
ADAPTER
    /adapters/code-lora
    /adapters-math-lora
    END

PARAMETER temperature 0.3
PARAMETER num_ctx 8192

SYSTEM You are an expert programming assistant.

METADATA base_model "llama3.2"
METADATA adapters ["code-lora", "math-lora"]
```

### Enterprise .sou

```sou
# Enterprise deployment with ACL
FROM llama3.2

PARAMETER temperature 0.5
PARAMETER num_ctx 4096

# Access control
ACL
    roles ["admin", "user", "guest"]
    default_role "user"
    END

# Watermarking
WATERMARK
    enabled true
    strength 0.5
    END

SYSTEM You are an enterprise assistant with access controls enabled.

METADATA tenant "company-a"
METADATA compliance "SOC2"
```

---

## Template Variables

For custom templates, use these variables:

| Variable | Description |
|----------|-------------|
| `{{.System}}` | System prompt |
| `{{.Prompt}}` | User prompt |
| `{{.Response}}` | Model response |
| `{{.Context}}` | Context from knowledge base |
| `{{.Personality}}` | Personality embeddings |
| `{{.Tools}}` | Available tools |

---

## Conversion

### From Ollama Modelfile

```bash
sloughgpt convert --from ollama Modelfile output.sou
```

### From HuggingFace

```bash
sloughgpt convert --from hf meta-llama/Llama-2-7b output.sou
```

### From GGUF

```bash
sloughgpt convert --from gguf model.gguf output.sou
```

---

## Version History

- **v1.0.0** (2026-02-28): Initial specification
  - Basic instructions (FROM, PARAMETER, TEMPLATE, SYSTEM)
  - Personality configuration
  - Knowledge base integration
  - LoRA adapter support
  - Enterprise features (ACL, watermarking)
