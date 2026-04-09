"""
SloughGPT Inference Module

.sou Soul Unit format - the living identity format for trained AI models.
"""

from .sou_format import (
    SoulProfile,
    PersonalityCore,
    BehavioralTraits,
    CognitiveSignature,
    EmotionalRange,
    GenerationParams,
    ContextParams,
    SouParser,
    create_soul_profile,
    export_to_sou,
    import_from_sou,
    generate_sample_dialogue,
    SOU_MAGIC,
    SOU_VERSION,
    SOU_TRADEMARK,
)

from .quantization import (
    QuantizationType as QType,
    QuantizationInfo,
    Quantizer,
    SouModelQuantizer,
    QUANTIZATION_PRESETS,
    get_quantization_preset,
)

from .loader import (
    InferenceConfig,
    SouModelLoader,
    SouInferenceEngine,
    load_model,
    generate,
    chat,
)

from .engine import (
    InferenceEngine,
    KVCache,
    GenerationRequest,
    BatchedRequest,
    create_engine,
)

from .llama_engine import (
    LlamaInferenceEngine,
    LlamaInferenceConfig,
    OllamaInferenceEngine,
    find_gguf_models,
    LLAMA_CPP_AVAILABLE,
    LLAMA_CPP_ERROR,
)

__all__ = [
    "SoulProfile",
    "PersonalityCore",
    "BehavioralTraits",
    "CognitiveSignature",
    "EmotionalRange",
    "GenerationParams",
    "ContextParams",
    "SouParser",
    "create_soul_profile",
    "export_to_sou",
    "import_from_sou",
    "generate_sample_dialogue",
    "SOU_MAGIC",
    "SOU_VERSION",
    "SOU_TRADEMARK",
    "QType",
    "QuantizationInfo",
    "Quantizer",
    "SouModelQuantizer",
    "QUANTIZATION_PRESETS",
    "get_quantization_preset",
    "InferenceConfig",
    "SouModelLoader",
    "SouInferenceEngine",
    "load_model",
    "generate",
    "chat",
    "InferenceEngine",
    "KVCache",
    "GenerationRequest",
    "BatchedRequest",
    "create_engine",
    "LlamaInferenceEngine",
    "LlamaInferenceConfig",
    "OllamaInferenceEngine",
    "find_gguf_models",
    "LLAMA_CPP_AVAILABLE",
    "LLAMA_CPP_ERROR",
]
