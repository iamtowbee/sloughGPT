# HuggingFace Integration Module

"""
Load models from HuggingFace in two modes:
1. API Mode: Use HF Inference API (no download)
2. Local Mode: Download and run locally
"""

from .api_loader import (
    HFAPIConfig,
    HuggingFaceAPILoader,
    HFInferenceClient,
    create_api_client,
    generate_via_api,
    chat_via_api,
)

from .local_loader import (
    HFLocalConfig,
    HuggingFaceLocalLoader,
    HuggingFaceLocalClient,
    download_model,
    load_model,
    generate_local,
)

from .model_map import (
    ModelSize,
    ModelInfo,
    HF_MODELS,
    get_model_info,
    search_models,
    get_recommended_quantization,
    get_model_requirements,
    map_to_sloughgpt_config,
)


__all__ = [
    # API Loader
    "HFAPIConfig",
    "HuggingFaceAPILoader",
    "HFInferenceClient",
    "create_api_client",
    "generate_via_api",
    "chat_via_api",
    # Local Loader
    "HFLocalConfig",
    "HuggingFaceLocalLoader",
    "HuggingFaceLocalClient",
    "download_model",
    "load_model",
    "generate_local",
    # Model Map
    "ModelSize",
    "ModelInfo",
    "HF_MODELS",
    "get_model_info",
    "search_models",
    "get_recommended_quantization",
    "get_model_requirements",
    "map_to_sloughgpt_config",
]
