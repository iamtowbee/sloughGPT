"""Model serving infrastructure for SloughGPT."""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import asyncio
import torch
import time
from datetime import datetime
from threading import Lock
import logging
from concurrent.futures import ThreadPoolExecutor

try:
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class ModelConfig:
    model_name: str
    model_path: str
    device: str = "auto"
    torch_dtype: str = "float16"
    max_length: int = 2048
    batch_size: int = 1
    trust_remote_code: bool = True
    use_cache: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


@dataclass
class GenerationRequest:
    prompt: str
    max_tokens: int = 100
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    user_id: Optional[int] = None


@dataclass
class GenerationResponse:
    text: str
    tokens_generated: int
    generation_time: float
    model_name: str
    finish_reason: str
    request_id: str


class ModelServer:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.device_map = {}
        self.request_queue = asyncio.Queue()
        self.processing = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = Lock()
        self.generation_count = 0
        
    async def load_model(self, config: ModelConfig) -> bool:
        """Load a model with the given configuration."""
        if not HAS_TRANSFORMERS:
            logging.error("Transformers library not available")
            return False
        
        try:
            # Determine device
            if config.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            else:
                device = config.device
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_path,
                trust_remote_code=config.trust_remote_code
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                "torch_dtype": getattr(torch, config.torch_dtype),
                "trust_remote_code": config.trust_remote_code,
                "use_cache": config.use_cache,
            }
            
            if device == "cuda":
                model_kwargs["device_map"] = "auto"
                model_kwargs["load_in_4bit"] = True  # Use 4-bit quantization for GPU
            elif device == "mps":
                model_kwargs["device_map"] = "mps"
            else:
                model_kwargs["device_map"] = "cpu"
            
            model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                **model_kwargs
            )
            
            # Store model and tokenizer
            with self.lock:
                self.models[config.model_name] = model
                self.tokenizers[config.model_name] = tokenizer
                self.model_configs[config.model_name] = config
                self.device_map[config.model_name] = device
            
            logging.info(f"Model {config.model_name} loaded successfully on {device}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load model {config.model_name}: {e}")
            return False
    
    async def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory."""
        try:
            with self.lock:
                if model_name in self.models:
                    del self.models[model_name]
                if model_name in self.tokenizers:
                    del self.tokenizers[model_name]
                if model_name in self.model_configs:
                    del self.model_configs[model_name]
                if model_name in self.device_map:
                    del self.device_map[model_name]
            
            # Clear CUDA cache if applicable
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logging.info(f"Model {model_name} unloaded successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to unload model {model_name}: {e}")
            return False
    
    async def generate(self, request: GenerationRequest, model_name: str) -> GenerationResponse:
        """Generate text using the specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        self.generation_count += 1
        request_id = f"gen_{self.generation_count}_{int(time.time())}"
        
        start_time = time.time()
        
        try:
            # Get model components
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
            config = self.model_configs[model_name]
            device = self.device_map[model_name]
            
            # Tokenize input
            inputs = tokenizer(
                request.prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_length
            )
            
            # Move to device
            if device != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Configure generation
            generation_config = GenerationConfig(
                max_new_tokens=request.max_tokens,
                temperature=request.temperature or config.temperature,
                top_p=request.top_p or config.top_p,
                top_k=request.top_k or config.top_k,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
            )
            
            # Generate
            with torch.no_grad():
                if request.stream:
                    # For streaming, we'll generate in chunks
                    text_chunks = []
                    async for chunk in self._stream_generate(model, inputs, generation_config):
                        text_chunks.append(chunk)
                    
                    generated_text = "".join(text_chunks)
                else:
                    outputs = model.generate(**inputs, generation_config=generation_config)
                    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            generation_time = time.time() - start_time
            tokens_generated = len(tokenizer.encode(generated_text))
            
            # Determine finish reason
            if tokens_generated >= request.max_tokens:
                finish_reason = "length"
            else:
                finish_reason = "stop"
            
            return GenerationResponse(
                text=generated_text,
                tokens_generated=tokens_generated,
                generation_time=generation_time,
                model_name=model_name,
                finish_reason=finish_reason,
                request_id=request_id
            )
            
        except Exception as e:
            logging.error(f"Generation failed for request {request_id}: {e}")
            raise
    
    async def _stream_generate(self, model, inputs, generation_config):
        """Generate text in a streaming fashion."""
        # Simplified streaming - in production, use proper streaming generation
        outputs = model.generate(**inputs, generation_config=generation_config)
        
        # Get tokenizer
        model_name = None
        for name, m in self.models.items():
            if m is model:
                model_name = name
                break
        
        if model_name is None:
            return
        
        tokenizer = self.tokenizers[model_name]
        full_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Yield in chunks
        chunk_size = max(1, len(full_text) // 10)
        for i in range(0, len(full_text), chunk_size):
            await asyncio.sleep(0.1)  # Simulate streaming delay
            yield full_text[i:i + chunk_size]
    
    async def batch_generate(self, requests: List[GenerationRequest], model_name: str) -> List[GenerationResponse]:
        """Generate text for multiple requests in batch."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        config = self.model_configs[model_name]
        
        # Group requests by similar parameters for better batching
        batch_size = min(config.batch_size, len(requests))
        results = []
        
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            batch_tasks = []
            
            for request in batch:
                task = asyncio.create_task(self.generate(request, model_name))
                batch_tasks.append(task)
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    # Create error response
                    error_response = GenerationResponse(
                        text="",
                        tokens_generated=0,
                        generation_time=0.0,
                        model_name=model_name,
                        finish_reason="error",
                        request_id="error"
                    )
                    results.append(error_response)
                else:
                    results.append(result)
        
        return results
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a loaded model."""
        if model_name not in self.model_configs:
            return None
        
        config = self.model_configs[model_name]
        device = self.device_map[model_name]
        model = self.models[model_name]
        
        return {
            "model_name": model_name,
            "model_path": config.model_path,
            "device": device,
            "max_length": config.max_length,
            "batch_size": config.batch_size,
            "parameters": sum(p.numel() for p in model.parameters()),
            "memory_usage": self._get_memory_usage(model, device),
            "loaded_at": datetime.now().isoformat()
        }
    
    def list_loaded_models(self) -> List[str]:
        """List all loaded models."""
        return list(self.models.keys())
    
    def _get_memory_usage(self, model, device: str) -> Dict[str, float]:
        """Get memory usage information for a model."""
        try:
            if device == "cuda" and torch.cuda.is_available():
                # GPU memory usage
                allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                cached = torch.cuda.memory_reserved() / (1024**3)  # GB
                return {
                    "allocated_gb": allocated,
                    "cached_gb": cached,
                    "total_gb": allocated + cached
                }
            elif device == "mps" and torch.backends.mps.is_available():
                # MPS memory usage (approximation)
                param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
                return {
                    "estimated_gb": param_size,
                    "type": "estimated"
                }
            else:
                # CPU memory usage (approximation)
                param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
                return {
                    "estimated_gb": param_size,
                    "type": "estimated"
                }
        except Exception as e:
            logging.warning(f"Could not get memory usage: {e}")
            return {"error": str(e)}
    
    async def optimize_model(self, model_name: str, optimization_type: str = "quantization") -> bool:
        """Optimize a loaded model for better performance."""
        if model_name not in self.models:
            return False
        
        try:
            model = self.models[model_name]
            device = self.device_map[model_name]
            
            if optimization_type == "quantization" and device == "cuda":
                # Apply dynamic quantization
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                self.models[model_name] = model
                
                logging.info(f"Model {model_name} quantized successfully")
                return True
            
            elif optimization_type == "compilation":
                # Use torch.compile for optimization (PyTorch 2.0+)
                if hasattr(torch, 'compile'):
                    model = torch.compile(model)
                    self.models[model_name] = model
                    logging.info(f"Model {model_name} compiled successfully")
                    return True
            
            elif optimization_type == "cpu_offload":
                # Move to CPU if GPU memory is constrained
                if device == "cuda":
                    model = model.cpu()
                    self.models[model_name] = model
                    self.device_map[model_name] = "cpu"
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    logging.info(f"Model {model_name} moved to CPU")
                    return True
            
            return False
            
        except Exception as e:
            logging.error(f"Model optimization failed: {e}")
            return False


class ModelManager:
    def __init__(self):
        self.server = ModelServer()
        self.default_configs = {
            "sloughgpt-base": ModelConfig(
                model_name="sloughgpt-base",
                model_path="microsoft/DialoGPT-small",
                device="auto",
                max_length=1024
            ),
            "sloughgpt-medium": ModelConfig(
                model_name="sloughgpt-medium", 
                model_path="microsoft/DialoGPT-medium",
                device="auto",
                max_length=2048
            ),
            "sloughgpt-large": ModelConfig(
                model_name="sloughgpt-large",
                model_path="microsoft/DialoGPT-large", 
                device="auto",
                max_length=2048
            )
        }
    
    async def initialize_default_models(self) -> None:
        """Initialize default models."""
        logging.info("Initializing default models...")
        
        for model_name, config in self.default_configs.items():
            success = await self.server.load_model(config)
            if success:
                logging.info(f"Loaded default model: {model_name}")
            else:
                logging.warning(f"Failed to load default model: {model_name}")
    
    async def generate_text(self, request: GenerationRequest, model_name: str = "sloughgpt-base") -> GenerationResponse:
        """Generate text using specified model."""
        return await self.server.generate(request, model_name)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of model server."""
        loaded_models = self.server.list_loaded_models()
        
        return {
            "status": "healthy" if loaded_models else "no_models_loaded",
            "loaded_models": loaded_models,
            "total_models": len(loaded_models),
            "device_available": {
                "cuda": torch.cuda.is_available(),
                "mps": torch.backends.mps.is_available(),
                "cpu": True
            }
        }


# Global model manager instance
model_manager = ModelManager()