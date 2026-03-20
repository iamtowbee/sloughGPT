"""
SloughGPT SDK Client
Main client for interacting with the SloughGPT API.
"""

from __future__ import annotations

import sys
import os
import importlib.util

import requests
from typing import List, Optional, Dict, Any, Iterator, Union, TYPE_CHECKING
import time
import json

if TYPE_CHECKING:
    from models import (
        GenerateRequest,
        GenerationResult,
        ChatRequest,
        ChatMessage,
        ChatResult,
        BatchRequest,
        BatchResult,
        ModelInfo,
        DatasetInfo,
        HealthStatus,
        SystemInfo,
        MetricsData,
    )
else:
    _models_spec = importlib.util.spec_from_file_location("models", os.path.join(os.path.dirname(__file__), "models.py"))
    models = importlib.util.module_from_spec(_models_spec)
    _models_spec.loader.exec_module(models)

    GenerateRequest = models.GenerateRequest
    GenerateResult = models.GenerationResult
    ChatRequest = models.ChatRequest
    ChatMessage = models.ChatMessage
    ChatResult = models.ChatResult
    BatchRequest = models.BatchRequest
    BatchResult = models.BatchResult
    ModelInfo = models.ModelInfo
    DatasetInfo = models.DatasetInfo
    HealthStatus = models.HealthStatus
    SystemInfo = models.SystemInfo
    MetricsData = models.MetricsData


class SloughGPTClient:
    """
    Python client for the SloughGPT API.
    
    Example usage:
    
    ```python
    from sloughgpt_sdk import SloughGPTClient
    
    client = SloughGPTClient(base_url="http://localhost:8000")
    
    # Check health
    health = client.health()
    print(f"Status: {health.status}")
    
    # Generate text
    result = client.generate("Hello, how are you?")
    print(result.generated_text)
    
    # Chat
    chat_result = client.chat([
        ChatMessage.user("Hello!"),
        ChatMessage.assistant("Hi! How can I help you?"),
        ChatMessage.user("Tell me a joke"),
    ])
    print(chat_result.message.content)
    
    # Stream generation
    for token in client.generate_stream("Once upon a time"):
        print(token, end="", flush=True)
    ```
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30,
        verify_ssl: bool = True,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the SloughGPT client.
        
        Args:
            base_url: Base URL of the SloughGPT API server.
            api_key: Optional API key for authentication.
            timeout: Request timeout in seconds.
            verify_ssl: Whether to verify SSL certificates.
            headers: Additional headers to include with requests.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        
        self._headers = headers or {}
        if api_key:
            self._headers["X-API-Key"] = api_key
        
        self._session = requests.Session()
        self._session.headers.update(self._headers)
    
    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> requests.Response:
        """Make an HTTP request to the API."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        kwargs.setdefault("timeout", self.timeout)
        kwargs.setdefault("verify", self.verify_ssl)
        
        response = self._session.request(method, url, **kwargs)
        response.raise_for_status()
        return response
    
    # Health & Status
    
    def health(self) -> HealthStatus:
        """Check API health status."""
        response = self._request("GET", "/health")
        return HealthStatus.from_response(response.json())
    
    def liveness(self) -> Dict[str, Any]:
        """Check if the server is alive (liveness probe)."""
        response = self._request("GET", "/health/live")
        return response.json()
    
    def readiness(self) -> Dict[str, Any]:
        """Check if the server is ready (readiness probe)."""
        response = self._request("GET", "/health/ready")
        return response.json()
    
    def info(self) -> SystemInfo:
        """Get detailed system information."""
        response = self._request("GET", "/info")
        return SystemInfo.from_response(response.json())
    
    # Text Generation
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The input prompt.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0-2).
            top_k: Top-k sampling parameter.
            top_p: Nucleus sampling parameter.
            **kwargs: Additional generation parameters.
        
        Returns:
            GenerationResult with generated text.
        """
        request = GenerateRequest(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **kwargs
        )
        
        start_time = time.time()
        response = self._request("POST", "/generate", json=request.to_dict())
        elapsed_ms = (time.time() - start_time) * 1000
        
        result = GenerationResult.from_response(response.json(), prompt)
        if result.inference_time_ms is None:
            result.inference_time_ms = elapsed_ms
        return result
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate text with streaming response.
        
        Args:
            prompt: The input prompt.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional generation parameters.
        
        Yields:
            Generated tokens as they arrive.
        """
        request = GenerateRequest(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
        
        response = self._request(
            "POST",
            "/generate/stream",
            json=request.to_dict(),
            stream=True
        )
        
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data:"):
                data = line[5:].strip()
                if data and data != "[DONE]":
                    try:
                        yield data
                    except json.JSONDecodeError:
                        yield data
    
    # Chat Completions
    
    def chat(
        self,
        messages: Union[List[ChatMessage], List[Dict[str, str]]],
        model: Optional[str] = None,
        temperature: float = 0.8,
        max_new_tokens: int = 100,
        **kwargs
    ) -> ChatResult:
        """
        Generate a chat completion.
        
        Args:
            messages: List of chat messages.
            model: Model to use (optional).
            temperature: Sampling temperature.
            max_new_tokens: Maximum tokens to generate.
            **kwargs: Additional parameters.
        
        Returns:
            ChatResult with assistant's response.
        """
        chat_messages = []
        for m in messages:
            if isinstance(m, ChatMessage):
                chat_messages.append(m)
            elif isinstance(m, dict):
                chat_messages.append(ChatMessage(
                    role=m.get("role", "user"),
                    content=m.get("content", "")
                ))
        
        request = ChatRequest(
            messages=chat_messages,
            model=model,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        
        response = self._request("POST", "/chat/completions", json=request.to_dict())
        return ChatResult.from_response(response.json())
    
    def chat_stream(
        self,
        messages: Union[List[ChatMessage], List[Dict[str, str]]],
        **kwargs
    ) -> Iterator[str]:
        """
        Generate a chat completion with streaming.
        
        Args:
            messages: List of chat messages.
            **kwargs: Additional parameters.
        
        Yields:
            Generated tokens as they arrive.
        """
        chat_messages = []
        for m in messages:
            if isinstance(m, ChatMessage):
                chat_messages.append(m)
            elif isinstance(m, dict):
                chat_messages.append(ChatMessage(
                    role=m.get("role", "user"),
                    content=m.get("content", "")
                ))
        
        request = ChatRequest(messages=chat_messages, stream=True, **kwargs)
        
        response = self._request(
            "POST",
            "/chat/completions",
            json=request.to_dict(),
            stream=True
        )
        
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data:"):
                data = line[5:].strip()
                if data and data != "[DONE]":
                    yield data
    
    # Batch Processing
    
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.8,
    ) -> BatchResult:
        """
        Generate text for multiple prompts in batch.
        
        Args:
            prompts: List of input prompts.
            max_new_tokens: Maximum tokens per generation.
            temperature: Sampling temperature.
        
        Returns:
            BatchResult with all generated texts.
        """
        request = BatchRequest(
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        response = self._request("POST", "/generate/batch", json=request.to_dict())
        return BatchResult.from_response(response.json(), prompts)
    
    # Models
    
    def list_models(self) -> List[ModelInfo]:
        """List available models."""
        response = self._request("GET", "/models")
        data = response.json()
        models = data.get("models", data) if isinstance(data, dict) else data
        return [ModelInfo.from_dict(m) for m in models]
    
    def get_model(self, model_id: str) -> ModelInfo:
        """Get information about a specific model."""
        response = self._request("GET", f"/models/{model_id}")
        return ModelInfo.from_dict(response.json())
    
    def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load a model into memory."""
        response = self._request("POST", "/models/load", json={"model_id": model_id})
        return response.json()
    
    def list_hf_models(self, query: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """List available HuggingFace models."""
        params = {"limit": limit}
        if query:
            params["q"] = query
        
        response = self._request("GET", "/models/hf", params=params)
        return response.json().get("models", [])
    
    # Datasets
    
    def list_datasets(self) -> List[DatasetInfo]:
        """List available datasets."""
        response = self._request("GET", "/datasets")
        data = response.json()
        datasets = data.get("datasets", data) if isinstance(data, dict) else data
        return [DatasetInfo.from_dict(d) for d in datasets]
    
    def get_dataset(self, dataset_id: str) -> DatasetInfo:
        """Get information about a specific dataset."""
        response = self._request("GET", f"/datasets/{dataset_id}")
        return DatasetInfo.from_dict(response.json())
    
    # Metrics
    
    def metrics(self) -> MetricsData:
        """Get API metrics."""
        response = self._request("GET", "/metrics")
        return MetricsData.from_response(response.json())
    
    def metrics_prometheus(self) -> str:
        """Get metrics in Prometheus format."""
        response = self._request(
            "GET",
            "/metrics",
            headers={"Accept": "text/plain"}
        )
        return response.text
    
    # Authentication
    
    def login(self, username: str, password: str) -> Dict[str, Any]:
        """Login and get access token."""
        response = self._request("POST", "/auth/login", json={
            "username": username,
            "password": password
        })
        return response.json()
    
    def refresh_token(self) -> Dict[str, Any]:
        """Refresh access token."""
        response = self._request("POST", "/auth/refresh")
        return response.json()
    
    # Context Manager
    
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self._session.close()
    
    # Convenience Methods
    
    def quick_generate(self, prompt: str) -> str:
        """Quick generation with default settings."""
        return self.generate(prompt).generated_text
    
    def quick_chat(self, user_message: str) -> str:
        """Quick chat with a single user message."""
        result = self.chat([ChatMessage.user(user_message)])
        return result.message.content


class AsyncSloughGPTClient:
    """
    Async Python client for the SloughGPT API.
    
    Example usage:
    
    ```python
    import asyncio
    from sloughgpt_sdk import AsyncSloughGPTClient
    
    async def main():
        async with AsyncSloughGPTClient() as client:
            result = await client.generate("Hello!")
            print(result.generated_text)
    
    asyncio.run(main())
    ```
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30,
        verify_ssl: bool = True,
        headers: Optional[Dict[str, str]] = None,
    ):
        """Initialize the async client."""
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self._headers = headers or {}
        if api_key:
            self._headers["X-API-Key"] = api_key
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an async HTTP request."""
        import httpx
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        kwargs.setdefault("timeout", self.timeout)
        
        async with httpx.AsyncClient(verify=self.verify_ssl, headers=self._headers) as client:
            response = await client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
    
    async def health(self) -> HealthStatus:
        """Check API health."""
        data = await self._request("GET", "/health")
        return HealthStatus.from_response(data)
    
    async def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate text."""
        from .models import GenerateRequest
        
        request = GenerateRequest(prompt=prompt, **kwargs)
        data = await self._request("POST", "/generate", json=request.to_dict())
        return GenerationResult.from_response(data, prompt)
    
    async def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResult:
        """Generate chat completion."""
        data = await self._request("POST", "/chat/completions", json={
            "messages": [m.to_dict() if isinstance(m, ChatMessage) else m for m in messages],
            **kwargs
        })
        return ChatResult.from_response(data)
    
    async def list_models(self) -> List[ModelInfo]:
        """List available models."""
        data = await self._request("GET", "/models")
        models = data.get("models", data) if isinstance(data, dict) else data
        return [ModelInfo.from_dict(m) for m in models]
    
    async def metrics(self) -> MetricsData:
        """Get API metrics."""
        data = await self._request("GET", "/metrics")
        return MetricsData.from_response(data)
    
    async def __aenter__(self):
        """Enter async context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager."""
        pass
