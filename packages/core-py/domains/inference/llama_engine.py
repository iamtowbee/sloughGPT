"""
Llama.cpp Inference Engine
High-performance local inference using llama.cpp with GGUF models.
"""

import os
import sys
import time
import threading
import json
import requests
from pathlib import Path
from typing import Optional, List, AsyncIterator, Dict, Any, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

LLAMA_CPP_AVAILABLE = False
LLAMA_CPP_PYTHON_AVAILABLE = False
LLAMA_CPP_ERROR: Optional[str] = None
Llama = None

try:
    from llama_cpp import Llama as _Llama
    from llama_cpp.llama import LlamaCompletionTokenChunk

    LLAMA_CPP_AVAILABLE = True
    LLAMA_CPP_PYTHON_AVAILABLE = True
    Llama = _Llama
except ImportError as e:
    LLAMA_CPP_ERROR = str(e)
    logger.warning(f"llama-cpp-python not installed: {e}. Falling back to llama-cli subprocess.")

LLAMA_CLI_PATH = "/usr/local/bin/llama-cli"
LLAMA_CLI_AVAILABLE = Path(LLAMA_CLI_PATH).exists() if not LLAMA_CPP_PYTHON_AVAILABLE else False


@dataclass
class LlamaInferenceConfig:
    """Configuration for llama.cpp inference."""

    model_path: str
    n_ctx: int = 4096
    n_threads: int = 6
    n_gpu_layers: int = 0
    n_batch: int = 512
    repeat_penalty: float = 1.1
    rope_freq_base: float = 10000.0
    rope_freq_scale: float = 1.0
    verbose: bool = False


class LlamaInferenceEngine:
    """
    Inference engine using llama.cpp for high-performance CPU/GPU inference.

    Features:
    - GGUF model support
    - Streaming generation
    - KV cache (via llama.cpp)
    - Configurable thread count
    - Memory efficient
    - Fallback to llama-cli subprocess if llama-cpp-python unavailable
    """

    def __init__(self, config: LlamaInferenceConfig):
        self.config = config
        self._llama: Optional[Llama] = None
        self._cli_engine: Optional[LlamaCLIInferenceEngine] = None
        self._lock = threading.Lock()
        self._model_loaded = False

        if not LLAMA_CPP_PYTHON_AVAILABLE:
            if LLAMA_CLI_AVAILABLE:
                logger.info("llama-cpp-python not available, using llama-cli fallback")
                self._cli_engine = LlamaCLIInferenceEngine(
                    model_path=config.model_path,
                    n_ctx=config.n_ctx,
                    n_threads=config.n_threads,
                    n_gpu_layers=config.n_gpu_layers,
                )
            else:
                raise ImportError(
                    "Neither llama-cpp-python nor llama-cli available. "
                    "Install llama-cpp-python or brew install llama.cpp"
                )

    def load_model(self) -> bool:
        """Load the model into memory."""
        if self._model_loaded:
            return True

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return False

        try:
            logger.info(f"Loading model: {model_path}")
            self._llama = Llama(
                model_path=str(model_path),
                n_ctx=self.config.n_ctx,
                n_threads=self.config.n_threads,
                n_gpu_layers=self.config.n_gpu_layers,
                n_batch=self.config.n_batch,
                rope_freq_base=self.config.rope_freq_base,
                rope_freq_scale=self.config.rope_freq_scale,
                verbose=self.config.verbose,
            )
            self._model_loaded = True
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate text synchronously."""
        if self._cli_engine:
            return self._cli_engine.generate(
                prompt, max_tokens, temperature, top_p, repeat_penalty, stop
            )

        if not self._model_loaded:
            if not self.load_model():
                return ""

        with self._lock:
            try:
                result = self._llama(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    stop=stop or [],
                    echo=False,
                )
                return result["choices"][0]["text"]
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                return ""

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
    ) -> AsyncIterator[str]:
        """Generate text with streaming."""
        if self._cli_engine:
            text = self._cli_engine.generate(
                prompt, max_tokens, temperature, top_p, repeat_penalty, stop
            )
            for word in text.split():
                yield word + " "
            return

        if not self._model_loaded:
            if not self.load_model():
                return

        def _stream():
            with self._lock:
                try:
                    for chunk in self._llama(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repeat_penalty=repeat_penalty,
                        stop=stop or [],
                        echo=False,
                        stream=True,
                    ):
                        token = chunk["choices"][0]["text"]
                        yield token
                except Exception as e:
                    logger.error(f"Streaming failed: {e}")

        return _stream()

    def benchmark(
        self, prompt: str = "The quick brown fox", num_tokens: int = 50
    ) -> Dict[str, Any]:
        """Benchmark inference speed."""
        if self._cli_engine:
            return self._cli_engine.benchmark(prompt, num_tokens)

        if not self._model_loaded:
            if not self.load_model():
                return {"error": "Failed to load model"}

        start = time.perf_counter()
        text = self.generate(prompt, max_tokens=num_tokens, temperature=0)
        end = time.perf_counter()

        elapsed = end - start
        tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0

        return {
            "tokens_generated": len(text.split()),
            "time_seconds": elapsed,
            "tokens_per_second": tokens_per_sec,
            "text_preview": text[:100],
            "backend": "llama-cpp-python",
        }

    def unload(self):
        """Unload model from memory."""
        with self._lock:
            if self._cli_engine:
                self._cli_engine.unload()
                self._cli_engine = None
            if self._llama is not None:
                del self._llama
                self._llama = None
                self._model_loaded = False
                logger.info("Model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded


class OllamaInferenceEngine:
    """
    Inference engine using Ollama API as backend.

    This provides a reference implementation to benchmark against while
    building our native inference core.
    """

    def __init__(
        self,
        model: str = "llama3.2:1b",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url
        self._available = None

    def check_connection(self) -> bool:
        """Check if Ollama is running."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except:
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Generate text via Ollama API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        }

        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def benchmark(
        self, prompt: str = "The quick brown fox", num_tokens: int = 50
    ) -> Dict[str, Any]:
        """Benchmark inference speed."""
        start = time.perf_counter()
        result = self.generate(prompt, max_tokens=num_tokens, temperature=0)
        end = time.perf_counter()

        elapsed = end - start
        tokens = result.get("eval_count", 0)

        return {
            "tokens_generated": tokens,
            "time_seconds": elapsed,
            "tokens_per_second": tokens / elapsed if elapsed > 0 else 0,
            "text": result.get("response", "")[:100],
            "backend": "ollama",
            "model": self.model,
        }


class LlamaCLIInferenceEngine:
    """
    Inference engine using llama-cli subprocess.

    This is a fallback when llama-cpp-python is not available.
    Less efficient due to subprocess overhead but provides direct llama.cpp access.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = 6,
        n_gpu_layers: int = 0,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self._process: Optional[subprocess.Popen] = None
        self._input_buffer = ""
        self._output_buffer = ""
        self._lock = threading.Lock()

    def _start_interactive(self):
        """Start llama-cli in interactive mode."""
        if self._process is not None:
            return

        cmd = [
            LLAMA_CLI_PATH,
            "-m",
            self.model_path,
            "-c",
            str(self.n_ctx),
            "-t",
            str(self.n_threads),
            "-ngl",
            str(self.n_gpu_layers),
            "-i",  # interactive mode
            "--log-disable",
        ]

        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate text using llama-cli."""
        if not Path(self.model_path).exists():
            logger.error(f"Model not found: {self.model_path}")
            return ""

        cmd = [
            LLAMA_CLI_PATH,
            "-m",
            self.model_path,
            "-c",
            str(self.n_ctx),
            "-t",
            str(self.n_threads),
            "-ngl",
            str(self.n_gpu_layers),
            "-p",
            prompt,
            "-n",
            str(max_tokens),
            "--temp",
            str(temperature),
            "--top-p",
            str(top_p),
            "--repeat-penalty",
            str(repeat_penalty),
            "--log-disable",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )

            output = result.stdout
            for line in output.split("\n"):
                if "[ Prompt:" in line:
                    continue
                if "[ Generation:" in line:
                    continue
                if line.startswith("> ") or line.startswith("llama"):
                    continue
                if line.strip():
                    return line.strip()

            return output.split(">")[-1].strip() if ">" in output else output.strip()

        except subprocess.TimeoutExpired:
            logger.error("Generation timed out")
            return ""
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""

    def benchmark(
        self, prompt: str = "The quick brown fox", num_tokens: int = 50
    ) -> Dict[str, Any]:
        """Benchmark inference speed."""
        start = time.perf_counter()
        text = self.generate(prompt, max_tokens=num_tokens, temperature=0)
        end = time.perf_counter()

        elapsed = end - start
        tokens = len(text.split()) if text else 0

        return {
            "tokens_generated": tokens,
            "time_seconds": elapsed,
            "tokens_per_second": tokens / elapsed if elapsed > 0 else 0,
            "text_preview": text[:100] if text else "",
            "backend": "llama-cli",
        }

    def unload(self):
        """Clean up subprocess."""
        if self._process:
            self._process.terminate()
            self._process = None

    @property
    def is_loaded(self) -> bool:
        return Path(self.model_path).exists()


import subprocess


def find_gguf_models(search_paths: Optional[List[str]] = None) -> List[Path]:
    """Find all GGUF model files in common locations."""
    if search_paths is None:
        search_paths = [
            str(Path.home() / "models"),
            str(Path.home() / ".ollama" / "models"),
            "/tmp",
            "./models",
        ]

    models = []
    for path_str in search_paths:
        path = Path(path_str)
        if path.exists():
            models.extend(path.glob("**/*.gguf"))

    return models


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test llama.cpp inference")
    parser.add_argument("--model", "-m", help="Path to GGUF model")
    parser.add_argument("--prompt", "-p", default="Hello world", help="Prompt")
    parser.add_argument("--tokens", "-t", type=int, default=50, help="Max tokens")
    parser.add_argument("--threads", type=int, default=6, help="CPU threads")
    parser.add_argument("--list", "-l", action="store_true", help="List available models")

    args = parser.parse_args()

    if args.list:
        models = find_gguf_models()
        print("Available GGUF models:")
        for m in models:
            size = m.stat().st_size / (1024 * 1024)
            print(f"  {m} ({size:.1f} MB)")
        sys.exit(0)

    if not args.model:
        models = find_gguf_models()
        if models:
            args.model = str(models[0])
            print(f"Using model: {args.model}")
        else:
            print("No model found. Use --model or download a GGUF model.")
            sys.exit(1)

    config = LlamaInferenceConfig(
        model_path=args.model,
        n_threads=args.threads,
    )

    engine = LlamaInferenceEngine(config)

    print(f"Generating {args.tokens} tokens...")
    result = engine.benchmark(args.prompt, args.tokens)

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Speed: {result['tokens_per_second']:.1f} tok/s")
        print(f"Time: {result['time_seconds']:.2f}s")
        print(f"Output: {result['text_preview']}...")
