# HuggingFace Integration - API Loader

Use HuggingFace Inference API without downloading models.
"""

import os
import json
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union, Generator

logger = logging.getLogger("sloughgpt.huggingface")


@dataclass
class HFAPIConfig:
    """Configuration for HuggingFace API."""
    model: str = "meta-llama/Llama-2-7b-chat-hf"
    api_key: Optional[str] = None
    timeout: int = 120
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    
    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("HF_API_KEY")
        if self.api_key is None:
            logger.warning("No HF_API_KEY set - some features may be limited")


class HuggingFaceAPILoader:
    """
    Load and run models via HuggingFace Inference API.
    
    Benefits:
    - No model download required
    - GPU handled by HF
    - Pay-per-use pricing
    - Always latest model version
    
    Limitations:
    - Requires internet
    - Costs per request
    - No custom fine-tuning
    """
    
    API_BASE = "https://api-inference.huggingface.co"
    
    # Model compatibility matrix
    SUPPORTED_TASKS = {
        "text-generation": [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-3-8b-instruct",
            "meta-llama/Llama-3.1-8b-instruct",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "google/gemma-2-9b-it",
            "Qwen/Qwen2-7B-Instruct",
            "microsoft/Phi-3-mini-128k-instruct",
        ],
        "chat-completion": [
            "meta-llama/Llama-3-8b-instruct",
            "meta-llama/Llama-3.1-8b-instruct",
            "mistralai/Mistral-7B-Instruct-v0.2",
        ]
    }
    
    def __init__(self, config: HFAPIConfig):
        self.config = config
        self.api_key = config.api_key
        self.headers = {}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        self.headers["Content-Type"] = "application/json"
    
    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request to HuggingFace."""
        import urllib.request
        import urllib.error
        
        url = f"{self.API_BASE}/models/{self.config.model}"
        
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            url,
            data=data,
            headers=self.headers,
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            raise Exception(f"HF API Error: {e.code} - {error_body}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Max tokens to generate
            **kwargs: Override config parameters
            
        Returns:
            Generated text
        """
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens or self.config.max_tokens,
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "top_k": kwargs.get("top_k", self.config.top_k),
                "repetition_penalty": kwargs.get("repetition_penalty", self.config.repetition_penalty),
                "return_full_text": False,
            },
            "options": {
                "use_cache": True,
                "wait_for_model": True,
            }
        }
        
        result = self._make_request(payload)
        
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "")
        elif isinstance(result, dict) and "generated_text" in result:
            return result["generated_text"]
        
        return str(result)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Generation parameters
            
        Returns:
            Assistant response
        """
        # Convert messages to prompt format
        prompt = self._format_chat_prompt(messages)
        return self.generate(prompt, **kwargs)
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into prompt."""
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    def stream_generate(
        self,
        prompt: str,
        **kwargs
    ) -> Generator[str, None, None]:
        """Generate text with streaming (if supported)."""
        # Note: Streaming requires different API endpoint
        result = self.generate(prompt, **kwargs)
        for char in result:
            yield char
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information from HF."""
        import urllib.request
        
        url = f"https://huggingface.co/api/models/{self.config.model}"
        
        req = urllib.request.Request(url)
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode('utf-8'))
    
    def is_model_supported(self) -> bool:
        """Check if current model is supported."""
        for models in self.SUPPORTED_TASKS.values():
            if self.config.model in models:
                return True
        return False


class HFInferenceClient:
    """
    High-level client for HF Inference API.
    
    Simple interface for common use cases.
    """
    
    def __init__(self, model: str = "meta-llama/Llama-2-7b-chat-hf", **kwargs):
        config = HFAPIConfig(model=model, **kwargs)
        self.loader = HuggingFaceAPILoader(config)
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Quick generate call."""
        return self.loader.generate(prompt, **kwargs)
    
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> "HFInferenceClient":
        """Create client from pretrained model name."""
        return cls(model=model_name, **kwargs)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_api_client(model: str = "meta-llama/Llama-2-7b-chat-hf", **kwargs) -> HuggingFaceAPILoader:
    """Create HF API loader."""
    config = HFAPIConfig(model=model, **kwargs)
    return HuggingFaceAPILoader(config)


def generate_via_api(prompt: str, model: str = "meta-llama/Llama-2-7b-chat-hf", **kwargs) -> str:
    """Quick generate via API."""
    client = create_api_client(model)
    return client.generate(prompt, **kwargs)


def chat_via_api(messages: List[Dict[str, str]], model: str = "meta-llama/Llama-2-7b-chat-hf", **kwargs) -> str:
    """Quick chat via API."""
    client = create_api_client(model)
    return client.chat(messages, **kwargs)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Demo CLI."""
    import sys
    
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        print("⚠️  Set HF_API_KEY environment variable for full access")
        print("   export HF_API_KEY=your_token_here")
    
    if len(sys.argv) < 2:
        print("Usage: python api_loader.py <model> [prompt]")
        print("\nExamples:")
        print("   python api_loader.py meta-llama/Llama-2-7b-chat-hf 'Hello!'")
        print("   python api_loader.py mistralai/Mistral-7B-Instruct-v0.2 'Write a story'")
        return
    
    model = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello, how are you?"
    
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print("-" * 40)
    
    try:
        client = create_api_client(model)
        
        if not client.is_model_supported():
            print(f"⚠️  Model may not be fully supported")
        
        print("Generating...")
        result = client.generate(prompt, max_new_tokens=100)
        print(f"\nResult: {result}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()


__all__ = [
    "HFAPIConfig",
    "HuggingFaceAPILoader",
    "HFInferenceClient",
    "create_api_client",
    "generate_via_api",
    "chat_via_api",
]
