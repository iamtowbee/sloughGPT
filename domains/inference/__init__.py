"""
SloughGPT Inference Module

.sou format model loading and inference.
"""

from .sou_format import (
    SouModelFile,
    SouParser,
    GenerationParameters,
    ContextParameters,
    PersonalityConfig,
    ACLConfig,
    WatermarkConfig,
    QuantizationType,
    create_default_sou,
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


__all__ = [
    # .sou Format
    "SouModelFile",
    "SouParser",
    "GenerationParameters",
    "ContextParameters",
    "PersonalityConfig",
    "ACLConfig",
    "WatermarkConfig",
    "QuantizationType",
    "create_default_sou",
    # Quantization
    "QType",
    "QuantizationInfo",
    "Quantizer",
    "SouModelQuantizer",
    "QUANTIZATION_PRESETS",
    "get_quantization_preset",
    # Loader
    "InferenceConfig",
    "SouModelLoader",
    "SouInferenceEngine",
    "load_model",
    "generate",
    "chat",
    # Engine
    "InferenceEngine",
    "KVCache",
    "GenerationRequest",
    "BatchedRequest",
    "create_engine",
]
