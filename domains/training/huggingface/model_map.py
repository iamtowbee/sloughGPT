# HuggingFace Integration - Model Map

Maps HuggingFace models to SloughGPT architecture.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class ModelSize(Enum):
    """Model size categories."""
    SMALL = "small"      # < 1B params
    MEDIUM = "medium"   # 1-10B params
    LARGE = "large"     # 10-70B params
    XLARGE = "xlarge"   # > 70B params


@dataclass
class ModelInfo:
    """Information about a HuggingFace model."""
    model_id: str
    name: str
    organization: str
    size: ModelSize
    parameters: int
    context_length: int
    recommended_precision: str
    quantization_recommended: str
    requires_special_tokenizer: bool
    supports_chat_template: bool
    description: str


# Model registry
HF_MODELS: Dict[str, ModelInfo] = {
    # LLaMA models
    "meta-llama/Llama-2-7b-hf": ModelInfo(
        model_id="meta-llama/Llama-2-7b-hf",
        name="LLaMA 2 7B",
        organization="Meta",
        size=ModelSize.MEDIUM,
        parameters=7_000_000_000,
        context_length=4096,
        recommended_precision="bf16",
        quantization_recommended="q4_k_m",
        requires_special_tokenizer=True,
        supports_chat_template=True,
        description="LLaMA 2 base model, 7 billion parameters",
    ),
    "meta-llama/Llama-2-7b-chat-hf": ModelInfo(
        model_id="meta-llama/Llama-2-7b-chat-hf",
        name="LLaMA 2 7B Chat",
        organization="Meta",
        size=ModelSize.MEDIUM,
        parameters=7_000_000_000,
        context_length=4096,
        recommended_precision="bf16",
        quantization_recommended="q4_k_m",
        requires_special_tokenizer=True,
        supports_chat_template=True,
        description="LLaMA 2 chat model, optimized for dialogue",
    ),
    "meta-llama/Llama-2-13b-hf": ModelInfo(
        model_id="meta-llama/Llama-2-13b-hf",
        name="LLaMA 2 13B",
        organization="Meta",
        size=ModelSize.LARGE,
        parameters=13_000_000_000,
        context_length=4096,
        recommended_precision="bf16",
        quantization_recommended="q4_k_m",
        requires_special_tokenizer=True,
        supports_chat_template=True,
        description="LLaMA 2 base model, 13 billion parameters",
    ),
    "meta-llama/Llama-2-13b-chat-hf": ModelInfo(
        model_id="meta-llama/Llama-2-13b-chat-hf",
        name="LLaMA 2 13B Chat",
        organization="Meta",
        size=ModelSize.LARGE,
        parameters=13_000_000_000,
        context_length=4096,
        recommended_precision="bf16",
        quantization_recommended="q4_k_m",
        requires_special_tokenizer=True,
        supports_chat_template=True,
        description="LLaMA 2 chat model, 13 billion parameters",
    ),
    "meta-llama/Llama-3-8b": ModelInfo(
        model_id="meta-llama/Llama-3-8b",
        name="LLaMA 3 8B",
        organization="Meta",
        size=ModelSize.MEDIUM,
        parameters=8_000_000_000,
        context_length=8192,
        recommended_precision="bf16",
        quantization_recommended="q4_k_m",
        requires_special_tokenizer=True,
        supports_chat_template=True,
        description="LLaMA 3 base model, 8 billion parameters",
    ),
    "meta-llama/Llama-3-8b-instruct": ModelInfo(
        model_id="meta-llama/Llama-3-8b-instruct",
        name="LLaMA 3 8B Instruct",
        organization="Meta",
        size=ModelSize.MEDIUM,
        parameters=8_000_000_000,
        context_length=8192,
        recommended_precision="bf16",
        quantization_recommended="q4_k_m",
        requires_special_tokenizer=True,
        supports_chat_template=True,
        description="LLaMA 3 instruction-tuned model",
    ),
    "meta-llama/Llama-3.1-8b-instruct": ModelInfo(
        model_id="meta-llama/Llama-3.1-8b-instruct",
        name="LLaMA 3.1 8B Instruct",
        organization="Meta",
        size=ModelSize.MEDIUM,
        parameters=8_000_000_000,
        context_length=128_000,
        recommended_precision="bf16",
        quantization_recommended="q4_k_m",
        requires_special_tokenizer=True,
        supports_chat_template=True,
        description="LLaMA 3.1 with 128K context",
    ),
    
    # Mistral models
    "mistralai/Mistral-7B-v0.1": ModelInfo(
        model_id="mistralai/Mistral-7B-v0.1",
        name="Mistral 7B",
        organization="Mistral AI",
        size=ModelSize.MEDIUM,
        parameters=7_000_000_000,
        context_length=8192,
        recommended_precision="bf16",
        quantization_recommended="q4_k_m",
        requires_special_tokenizer=False,
        supports_chat_template=True,
        description="Mistral 7B base model",
    ),
    "mistralai/Mistral-7B-Instruct-v0.2": ModelInfo(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        name="Mistral 7B Instruct",
        organization="Mistral AI",
        size=ModelSize.MEDIUM,
        parameters=7_000_000_000,
        context_length=32768,
        recommended_precision="bf16",
        quantization_recommended="q4_k_m",
        requires_special_tokenizer=False,
        supports_chat_template=True,
        description="Mistral 7B instruction-tuned",
    ),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": ModelInfo(
        model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        name="Mixtral 8x7B",
        organization="Mistral AI",
        size=ModelSize.XLARGE,
        parameters=12_000_000_000,
        context_length=32768,
        recommended_precision="bf16",
        quantization_recommended="q4_k_m",
        requires_special_tokenizer=False,
        supports_chat_template=True,
        description="Mixtral mixture of experts (8x7B)",
    ),
    
    # Google models
    "google/gemma-2b": ModelInfo(
        model_id="google/gemma-2b",
        name="Gemma 2B",
        organization="Google",
        size=ModelSize.SMALL,
        parameters=2_000_000_000,
        context_length=8192,
        recommended_precision="bf16",
        quantization_recommended="q4_k_m",
        requires_special_tokenizer=False,
        supports_chat_template=True,
        description="Google Gemma 2B base model",
    ),
    "google/gemma-7b": ModelInfo(
        model_id="google/gemma-7b",
        name="Gemma 7B",
        organization="Google",
        size=ModelSize.MEDIUM,
        parameters=7_000_000_000,
        context_length=8192,
        recommended_precision="bf16",
        quantization_recommended="q4_k_m",
        requires_special_tokenizer=False,
        supports_chat_template=True,
        description="Google Gemma 7B base model",
    ),
    "google/gemma-2-9b-it": ModelInfo(
        model_id="google/gemma-2-9b-it",
        name="Gemma 2 9B Instruct",
        organization="Google",
        size=ModelSize.MEDIUM,
        parameters=9_000_000_000,
        context_length=8192,
        recommended_precision="bf16",
        quantization_recommended="q4_k_m",
        requires_special_tokenizer=False,
        supports_chat_template=True,
        description="Google Gemma 2 9B instruction-tuned",
    ),
    
    # Qwen models
    "Qwen/Qwen2-7B-Instruct": ModelInfo(
        model_id="Qwen/Qwen2-7B-Instruct",
        name="Qwen 2 7B Instruct",
        organization="Alibaba",
        size=ModelSize.MEDIUM,
        parameters=7_000_000_000,
        context_length=32768,
        recommended_precision="bf16",
        quantization_recommended="q4_k_m",
        requires_special_tokenizer=False,
        supports_chat_template=True,
        description="Qwen 2 instruction-tuned model",
    ),
    "Qwen/Qwen1.5-72B-Chat": ModelInfo(
        model_id="Qwen/Qwen1.5-72B-Chat",
        name="Qwen 1.5 72B Chat",
        organization="Alibaba",
        size=ModelSize.LARGE,
        parameters=72_000_000_000,
        context_length=32768,
        recommended_precision="bf16",
        quantization_recommended="q4_k_m",
        requires_special_tokenizer=False,
        supports_chat_template=True,
        description="Qwen 1.5 chat model, 72B parameters",
    ),
    
    # Microsoft models
    "microsoft/Phi-3-mini-128k-instruct": ModelInfo(
        model_id="microsoft/Phi-3-mini-128k-instruct",
        name="Phi-3 Mini 128K",
        organization="Microsoft",
        size=ModelSize.SMALL,
        parameters=4_000_000_000,
        context_length=128_000,
        recommended_precision="bf16",
        quantization_recommended="q4_k_m",
        requires_special_tokenizer=False,
        supports_chat_template=True,
        description="Microsoft Phi-3 with 128K context",
    ),
    
    # Meta OPT models
    "facebook/opt-1.3b": ModelInfo(
        model_id="facebook/opt-1.3b",
        name="OPT 1.3B",
        organization="Meta",
        size=ModelSize.SMALL,
        parameters=1_300_000_000,
        context_length=2048,
        recommended_precision="fp32",
        quantization_recommended="q4_k_m",
        requires_special_tokenizer=False,
        supports_chat_template=False,
        description="OPT base model, 1.3B parameters",
    ),
    
    # Falcon models
    "tiiuae/falcon-7b": ModelInfo(
        model_id="tiiuae/falcon-7b",
        name="Falcon 7B",
        organization="TII",
        size=ModelSize.MEDIUM,
        parameters=7_000_000_000,
        context_length=2048,
        recommended_precision="bf16",
        quantization_recommended="q4_k_m",
        requires_special_tokenizer=False,
        supports_chat_template=False,
        description="Falcon base model, 7B parameters",
    ),
}


def get_model_info(model_id: str) -> Optional[ModelInfo]:
    """Get model information by ID."""
    return HF_MODELS.get(model_id)


def search_models(
    organization: Optional[str] = None,
    size: Optional[ModelSize] = None,
    context_length_min: Optional[int] = None,
) -> List[ModelInfo]:
    """Search models by criteria."""
    results = []
    
    for model in HF_MODELS.values():
        if organization and model.organization.lower() != organization.lower():
            continue
        if size and model.size != size:
            continue
        if context_length_min and model.context_length < context_length_min:
            continue
        
        results.append(model)
    
    return results


def get_recommended_quantization(model_id: str) -> str:
    """Get recommended quantization for a model."""
    info = get_model_info(model_id)
    if info:
        return info.quantization_recommended
    return "q4_k_m"  # Default


def get_model_requirements(model_id: str) -> Dict[str, Any]:
    """Get memory and compute requirements for a model."""
    info = get_model_info(model_id)
    
    if not info:
        return {"error": "Model not in registry"}
    
    # Calculate approximate memory
    precision_bytes = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
    }
    
    bytes_per_param = precision_bytes.get(info.recommended_precision, 2)
    
    # Model weights + KV cache + activations (rough estimate)
    model_memory = info.parameters * bytes_per_param
    kv_cache = info.parameters * 0.1  # ~10% for KV cache
    activation_memory = info.parameters * 0.2  # ~20% for activations
    
    total_memory = model_memory + kv_cache + activation_memory
    
    return {
        "model_id": model_id,
        "name": info.name,
        "parameters_billions": info.parameters / 1e9,
        "context_length": info.context_length,
        "recommended_precision": info.recommended_precision,
        "quantization": info.quantization_recommended,
        "memory_fp32_gb": (info.parameters * 4) / 1e9,
        "memory_bf16_gb": total_memory / 1e9,
        "memory_q4_gb": (info.parameters * 0.5) / 1e9,
        "size_category": info.size.value,
    }


# =============================================================================
# SloughGPT Integration
# =============================================================================

def map_to_sloughgpt_config(model_id: str) -> Dict[str, Any]:
    """
    Map HF model to SloughGPT TrainingConfig.
    
    This provides sensible defaults for training/fine-tuning.
    """
    info = get_model_info(model_id)
    
    if not info:
        return {
            "warning": "Model not in registry, using defaults",
            "n_embed": 512,
            "n_layer": 12,
            "n_head": 8,
        }
    
    # Estimate config from model size
    if info.size == ModelSize.SMALL:
        n_embed = 2048
        n_layer = 16
        n_head = 16
    elif info.size == ModelSize.MEDIUM:
        n_embed = 4096
        n_layer = 32
        n_head = 32
    elif info.size == ModelSize.LARGE:
        n_embed = 6144
        n_layer = 48
        n_head = 48
    else:  # XLARGE
        n_embed = 8192
        n_layer = 64
        n_head = 64
    
    return {
        "model_id": model_id,
        "name": info.name,
        "n_embed": n_embed,
        "n_layer": n_layer,
        "n_head": n_head,
        "context_length": info.context_length,
        "precision": info.recommended_precision,
        "quantization": info.quantization_recommended,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    """Demo CLI."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python model_map.py <model_id>")
        print("\nExamples:")
        print("   python model_map.py meta-llama/Llama-2-7b-chat-hf")
        print("   python model_map.py mistralai/Mistral-7B-Instruct-v0.2")
        print()
        
        # List all models
        print("Available models:")
        for model_id, info in HF_MODELS.items():
            print(f"  {model_id}")
            print(f"    {info.description}")
        return
    
    model_id = sys.argv[1]
    
    info = get_model_info(model_id)
    if not info:
        print(f"Model not in registry: {model_id}")
        return
    
    print(f"=== {info.name} ===")
    print(f"Organization: {info.organization}")
    print(f"Parameters: {info.parameters:,}")
    print(f"Context Length: {info.context_length:,}")
    print(f"Precision: {info.recommended_precision}")
    print(f"Quantization: {info.quantization_recommended}")
    print()
    
    reqs = get_model_requirements(model_id)
    print("Memory Requirements:")
    print(f"  FP32: {reqs['memory_fp32_gb']:.1f} GB")
    print(f"  BF16: {reqs['memory_bf16_gb']:.1f} GB")
    print(f"  Q4:   {reqs['memory_q4_gb']:.1f} GB")


if __name__ == "__main__":
    main()


__all__ = [
    "ModelSize",
    "ModelInfo",
    "HF_MODELS",
    "get_model_info",
    "search_models",
    "get_recommended_quantization",
    "get_model_requirements",
    "map_to_sloughgpt_config",
]
