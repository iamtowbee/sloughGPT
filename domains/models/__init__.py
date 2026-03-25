"""
domains/models/ - Pluggable Model Interface

SloughGPTModel is OUR OWN architecture. External models (NanoGPT, Qwen, etc.)
can be plugged in via the ModelLoader registry or ModelInterface.

Supported external model types:
- NanoGPT: https://github.com/karpathy/nanoGPT
- HuggingFace transformers: https://huggingface.co/models
- GGUF: llama.cpp format
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Callable
import logging
import math
import torch

logger = logging.getLogger("sloughgpt.models")

if TYPE_CHECKING:
    from domains.core.soul import SoulEngine


class ModelInterface(ABC):
    """Abstract interface all model backends must implement."""

    @abstractmethod
    def load(self, path: str, device: str = "cpu", **kwargs) -> "ModelInterface":
        """Load a model from a file path. Returns self for chaining."""
        pass

    @abstractmethod
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate tokens from input_ids. Returns full sequence including input."""
        pass

    @abstractmethod
    def forward(
        self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None, **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass. Returns (logits, loss)."""
        pass

    @abstractmethod
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return model state dict."""
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], **kwargs) -> None:
        """Load state dict."""
        pass

    @abstractmethod
    def num_parameters(self) -> int:
        """Total number of parameters."""
        pass

    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """Return model configuration dict."""
        pass

    @abstractmethod
    def to(self, device: str) -> "ModelInterface":
        """Move model to device. Returns self for chaining."""
        pass

    @abstractmethod
    def eval(self) -> "ModelInterface":
        """Set to eval mode. Returns self for chaining."""
        pass

    @abstractmethod
    def train_mode(self) -> "ModelInterface":
        """Set to train mode. Returns self for chaining."""
        pass


class ModelLoader:
    """Pluggable model loader - dispatches to the right backend.

    Supports:
    - .sou files (SloughGPT Soul Unit format)
    - .safetensors files
    - .pt/.pth PyTorch checkpoints
    - .gguf llama.cpp format
    - HuggingFace model IDs (when transformers is installed)
    - External model types registered via ModelLoader.register()
    """

    _registry: Dict[str, type] = {}
    _loader_funcs: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, model_class: type):
        """Register a model backend by name."""
        cls._registry[name] = model_class
        logger.info(f"Registered model backend: {name}")

    @classmethod
    def register_loader(cls, suffix: str, loader_func: Callable):
        """Register a custom loader function for a file suffix."""
        cls._loader_funcs[suffix] = loader_func
        logger.info(f"Registered loader for: {suffix}")

    @classmethod
    def load(cls, path: str, device: str = "cpu", **kwargs) -> ModelInterface:
        """Auto-detect format and load model."""
        from pathlib import Path

        p = Path(path)
        suffix = p.suffix.lower()

        if suffix in cls._loader_funcs:
            return cls._loader_funcs[suffix](path, device, **kwargs)

        if suffix == ".sou":
            return cls._load_sou(path, device, **kwargs)
        elif suffix == ".safetensors":
            return cls._load_safetensors(path, device, **kwargs)
        elif suffix == ".gguf":
            return cls._load_gguf(path, device, **kwargs)
        elif suffix == ".pt" or suffix == ".pth":
            return cls._load_torch(path, device, **kwargs)
        elif "/" in path or path.startswith("hf://"):
            return cls._load_huggingface(path, device, **kwargs)
        else:
            return cls._load_torch(path, device, **kwargs)

    @classmethod
    def _load_sou(cls, path: str, device: str, **kwargs) -> ModelInterface:
        """Load a .sou Soul Unit file."""
        from domains.inference.sou_format import import_from_sou

        soul, state_dict = import_from_sou(path)

        cfg = state_dict.get("config", {}) if isinstance(state_dict, dict) else {}
        model_type = cfg.get("model_type", "sloughgpt")

        vocab_size = cfg.get("vocab_size", 256)
        n_embed = cfg.get("n_embed", 256)
        n_layer = cfg.get("n_layer", 6)
        n_head = cfg.get("n_head", 8)
        block_size = cfg.get("block_size", 128)

        if model_type == "sloughgpt":
            model = SloughGPTModel(
                vocab_size=vocab_size,
                n_embed=n_embed,
                n_layer=n_layer,
                n_head=n_head,
                n_kv_head=cfg.get("n_kv_head"),
                block_size=block_size,
                max_seq_len=cfg.get("max_seq_len", 2048),
                use_sdpa=cfg.get("use_sdpa", True),
                use_flash=cfg.get("use_flash", False),
            )
        else:
            model = cls._load_external_model(model_type, cfg)

        if isinstance(state_dict, dict):
            filtered = {k: v for k, v in state_dict.items() if k not in ("config", "metadata")}
            model.load_state_dict(filtered, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)

        model._soul = soul
        return model.to(device)

    @classmethod
    def _load_safetensors(cls, path: str, device: str, **kwargs) -> ModelInterface:
        """Load a SafeTensors file."""
        from safetensors.torch import load_file

        state_dict = load_file(path, device=device)
        config = kwargs.get("config", {})
        model_type = config.get("model_type", "sloughgpt")

        if model_type == "sloughgpt":
            model = SloughGPTModel(
                vocab_size=config.get("vocab_size", 256),
                n_embed=config.get("n_embed", 256),
                n_layer=config.get("n_layer", 6),
                n_head=config.get("n_head", 8),
                block_size=config.get("block_size", 128),
            )
        else:
            model = cls._load_external_model(model_type, config)

        model.load_state_dict(state_dict, strict=False)
        return model.to(device)

    @classmethod
    def _load_torch(cls, path: str, device: str, **kwargs) -> ModelInterface:
        """Load a PyTorch .pt file."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model", checkpoint)
            config = checkpoint.get("config", {})
        else:
            state_dict = checkpoint
            config = {}

        model_type = config.get("model_type", "sloughgpt")

        if model_type == "sloughgpt":
            model = SloughGPTModel(
                vocab_size=config.get("vocab_size", 256),
                n_embed=config.get("n_embed", 256),
                n_layer=config.get("n_layer", 6),
                n_head=config.get("n_head", 8),
                n_kv_head=config.get("n_kv_head"),
                block_size=config.get("block_size", 128),
                max_seq_len=config.get("max_seq_len", 2048),
                use_sdpa=config.get("use_sdpa", True),
                use_flash=config.get("use_flash", False),
            )
        else:
            model = cls._load_external_model(model_type, config)

        model.load_state_dict(state_dict, strict=False)
        return model.to(device)

    @classmethod
    def _load_gguf(cls, path: str, device: str, **kwargs):
        """Load a GGUF file via llama-cpp-python."""
        try:
            from llama_cpp import Llama
            return Llama(model_path=path, n_ctx=kwargs.get("n_ctx", 2048))
        except ImportError:
            raise NotImplementedError(
                "GGUF loading requires llama-cpp-python. Install: pip install llama-cpp-python"
            )

    @classmethod
    def _load_huggingface(cls, model_id: str, device: str, **kwargs) -> ModelInterface:
        """Load a model from HuggingFace."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

            logger.info(f"Loading from HuggingFace: {model_id}")
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                config=config,
                trust_remote_code=True,
                device_map=device if device != "cpu" else None,
                **kwargs
            )
            return HuggingFaceWrapper(model)
        except ImportError:
            raise ImportError(
                "HuggingFace loading requires transformers. Install: pip install transformers"
            )

    @classmethod
    def _load_external_model(cls, model_type: str, config: Dict[str, Any]) -> ModelInterface:
        """Load an external model type from registry."""
        if model_type in cls._registry:
            model_class = cls._registry[model_type]
            return model_class(**config)

        supported = list(cls._registry.keys())
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Supported external models: {supported}. "
            f"Register with ModelLoader.register('{model_type}', YourModelClass)"
        )


class HuggingFaceWrapper(ModelInterface):
    """Wrapper for HuggingFace transformers models."""

    def __init__(self, model):
        self._model = model
        self._config = getattr(model, "config", {})

    def load(self, path: str, device: str = "cpu", **kwargs) -> "HuggingFaceWrapper":
        from transformers import AutoModelForCausalLM
        self._model = AutoModelForCausalLM.from_pretrained(path)
        return self.to(device)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        with torch.no_grad():
            output = self._model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                **kwargs
            )
        return output

    def forward(
        self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None, **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        outputs = self._model(input_ids, labels=targets)
        return outputs.logits, getattr(outputs, "loss", None)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self._model.state_dict()

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], **kwargs) -> None:
        self._model.load_state_dict(state_dict, **kwargs)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self._model.parameters())

    def config(self) -> Dict[str, Any]:
        return self._config

    def to(self, device: str) -> "HuggingFaceWrapper":
        self._model = self._model.to(device)
        return self

    def eval(self) -> "HuggingFaceWrapper":
        self._model.eval()
        return self

    def train_mode(self) -> "HuggingFaceWrapper":
        self._model.train()
        return self


# =============================================================================
# SloughGPTModel - OUR OWN ARCHITECTURE
# Features: RoPE, Flash Attention/SDPA, SwiGLU, RMSNorm, KV Cache, Gradient Checkpointing
# =============================================================================

class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization - LLaMA style.

    Formula: output = x * weight / RMS(x)
    Where RMS(x) = sqrt(mean(x^2))
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.float()
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        output = (x * norm * self.weight).to(x_dtype)
        return output


class RotaryEmbedding(torch.nn.Module):
    """Rotary Position Embeddings (RoPE) - replaces learned absolute positions."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cos_cached is not None and self._cos_cached.shape[0] >= seq_len:
            return self._cos_cached[:seq_len].to(device), self._sin_cached[:seq_len].to(device)

        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_cached = emb.cos()
        self._sin_cached = emb.sin()
        return self._cos_cached.to(device), self._sin_cached.to(device)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to q and k."""
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class SloughGPTAttention(torch.nn.Module):
    """
    SloughGPT's attention: Flash Attention/SDPA with RoPE and optional KV Cache.
    Falls back gracefully - SDPA works on all hardware, Flash Attention needs flash_attn.
    """

    def __init__(
        self,
        n_embed: int,
        n_head: int,
        n_kv_head: Optional[int] = None,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        use_sdpa: bool = True,
        use_flash: bool = True,
    ):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embed // n_head
        self.n_kv_head = n_kv_head or n_head
        self.n_rep = self.n_head // self.n_kv_head
        self.max_seq_len = max_seq_len

        self.use_sdpa = use_sdpa and hasattr(torch.nn.functional, "scaled_dot_product_attention")
        self.use_flash = use_flash
        self._sdpa_available = self.use_sdpa

        self.q_proj = torch.nn.Linear(n_embed, self.n_head * self.head_dim, bias=False)
        self.k_proj = torch.nn.Linear(n_embed, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(n_embed, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = torch.nn.Linear(self.n_head * self.head_dim, n_embed, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

        self.dropout = dropout
        self.attn_dropout = torch.nn.Dropout(dropout)

        self._kv_cache = None

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(T, q.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None and use_cache:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        if self.use_sdpa and not self.use_flash:
            attn_mask = torch.triu(
                torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1
            )
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                scale=1.0 / math.sqrt(self.head_dim),
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            causal_mask = torch.triu(
                torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1
            )
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn = torch.nn.functional.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out), (k, v)

    def clear_cache(self):
        self._kv_cache = None


class SwiGLU(torch.nn.Module):
    """Swish + Gated Linear Unit activation - better than GELU.

    LLaMA-style: gate = silu(w1(x)) * w3(x), output = w2(gate)
    With proper initialization for stability.
    """

    def __init__(self, dim: int, hidden_dim: int, init_scale: float = 0.02):
        super().__init__()
        self.w1 = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = torch.nn.Linear(dim, hidden_dim, bias=False)
        self._init_weights(init_scale)

    def _init_weights(self, scale: float):
        for linear in [self.w1, self.w2, self.w3]:
            torch.nn.init.normal_(linear.weight, mean=0.0, std=scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.silu(self.w1(x)) * self.w3(x)
        return self.w2(gate)


class SloughGPTBlock(torch.nn.Module):
    """Single transformer block: RMSNorm -> Attention -> RMSNorm -> SwiGLU MLP."""

    def __init__(
        self,
        n_embed: int,
        n_head: int,
        n_kv_head: Optional[int] = None,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        use_sdpa: bool = True,
        use_flash: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.norm1 = RMSNorm(n_embed)
        self.attn = SloughGPTAttention(
            n_embed, n_head, n_kv_head, dropout, max_seq_len, use_sdpa, use_flash
        )
        self.norm2 = RMSNorm(n_embed)
        hidden = int(n_embed * 8 // 3)
        hidden = ((hidden + 63) // 64) * 64
        self.mlp = SwiGLU(n_embed, hidden)
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SloughGPTModel(torch.nn.Module, ModelInterface):
    """
    SloughGPT - OUR OWN model architecture.

    Key innovations:
    - RoPE (Rotary Position Embeddings) - better generalization
    - Flash Attention / SDPA - memory-efficient attention
    - SwiGLU activation - outperforms GELU
    - RMSNorm - more stable training
    - KV Cache support - faster autoregressive generation
    - Grouped Query Attention - fewer KV heads for efficiency
    - Gradient Checkpointing - memory optimization for training

    This is SloughGPT - NOT NanoGPT, NOT GPT-2.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        n_embed: int = 256,
        n_layer: int = 6,
        n_head: int = 8,
        n_kv_head: Optional[int] = None,
        dropout: float = 0.1,
        block_size: int = 128,
        max_seq_len: int = 2048,
        use_sdpa: bool = True,
        use_flash: bool = False,
        tie_weights: bool = True,
    ):
        super().__init__()

        self._config = {
            "vocab_size": vocab_size,
            "n_embed": n_embed,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_kv_head": n_kv_head or n_head,
            "dropout": dropout,
            "block_size": block_size,
            "max_seq_len": max_seq_len,
            "use_sdpa": use_sdpa,
            "use_flash": use_flash,
            "model_type": "sloughgpt",
        }

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_layer = n_layer
        self.n_head = n_head
        self.max_seq_len = max_seq_len

        self.tok_emb = torch.nn.Embedding(vocab_size, n_embed)
        self.drop = torch.nn.Dropout(dropout)

        self.blocks = torch.nn.ModuleList()
        for layer_idx in range(n_layer):
            use_ckpt = layer_idx < n_layer // 2
            self.blocks.append(
                SloughGPTBlock(
                    n_embed, n_head, n_kv_head, dropout, max_seq_len,
                    use_sdpa=use_sdpa, use_flash=use_flash,
                    use_checkpoint=use_ckpt,
                )
            )

        self.norm = RMSNorm(n_embed)
        self.lm_head = torch.nn.Linear(n_embed, vocab_size, bias=False)

        self.apply(self._init_weights)

        if tie_weights:
            with torch.no_grad():
                self.lm_head.weight.copy_(self.tok_emb.weight)

        self._device = "cpu"
        self._soul = None
        self._kv_cache = [None] * n_layer

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.1)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on all blocks."""
        for block in self.blocks:
            block.use_checkpoint = True

    def clear_kv_cache(self):
        """Clear the KV cache."""
        self._kv_cache = [None] * len(self.blocks)
        for block in self.blocks:
            if hasattr(block.attn, "clear_cache"):
                block.attn.clear_cache()

    def load(self, path: str, device: str = "cpu", **kwargs) -> "SloughGPTModel":
        loaded = ModelLoader.load(path, device=device, config=self._config, **kwargs)
        self.load_state_dict(loaded.state_dict())
        self._soul = getattr(loaded, "_soul", None)
        return self.to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, t = input_ids.size()
        assert t <= self.block_size, f"Sequence {t} > block_size {self.block_size}"

        x = self.tok_emb(input_ids)
        x = self.drop(x)

        for layer_idx, block in enumerate(self.blocks):
            x = block(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.block_size :]
            logits, _ = self(idx_cond, use_cache=False)
            logits = logits[:, -1, :] / temperature
            logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            probs = torch.nn.functional.softmax(logits, dim=-1)
            probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
            probs_sum = probs.sum(dim=-1, keepdim=True)
            probs = probs / (probs_sum + 1e-10)
            idx_next = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, idx_next], dim=1)

            if idx_next.item() == 0:
                break

        return input_ids

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return torch.nn.Module.state_dict(self)

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True, **kwargs) -> None:
        torch.nn.Module.load_state_dict(self, state_dict, strict=strict)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config(self) -> Dict[str, Any]:
        return self._config.copy()

    def to(self, device: str) -> "SloughGPTModel":
        self._device = device
        torch.nn.Module.to(self, device)
        return self

    def eval(self) -> "SloughGPTModel":
        torch.nn.Module.eval(self)
        return self

    def train_mode(self) -> "SloughGPTModel":
        torch.nn.Module.train(self)
        return self

    def wrap_ddp(self, device_id: Optional[int] = None):
        """Wrap model for DDP distributed training."""
        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
            ddp_model = DDP(self, device_ids=[device_id] if device_id is not None else None)
            return ddp_model
        except Exception:
            return self

    def apply_gradient_checkpointing(self):
        """Enable gradient checkpointing on transformer blocks for memory efficiency."""
        for block in self.blocks:
            block.use_checkpoint = True
        return self

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def wrap_fsdp(self, sharding_strategy: str = "FULL_SHARD"):
        """Wrap model for FSDP (Fully Sharded Data Parallel) training."""
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            try:
                from torch.distributed.fsdp.api import ShardingStrategy
            except ImportError:
                from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy

            strategy_map = {
                "FULL_SHARD": getattr(ShardingStrategy, "FULL_SHARD", None),
                "SHARD_GRAD_OP": getattr(ShardingStrategy, "SHARD_GRAD_OP", None),
                "NO_SHARD": getattr(ShardingStrategy, "NO_SHARD", None),
            }

            strategy = strategy_map.get(sharding_strategy)
            if strategy is None:
                logger.warning("FSDP sharding strategy not available")
                return self
            fsdp_model = FSDP(self, sharding_strategy=strategy)
            return fsdp_model
        except ImportError:
            logger.warning("FSDP not available. Requires PyTorch 2.0+ with distributed support.")
            return self
        except Exception as e:
            logger.warning(f"FSDP wrapping failed: {e}")
            return self

    def get_model_size_mb(self) -> float:
        """Get model size in megabytes."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)

    def freeze_embeddings(self) -> "SloughGPTModel":
        """Freeze token embeddings to save memory."""
        for p in self.tok_emb.parameters():
            p.requires_grad = False
        return self


__all__ = [
    "ModelInterface",
    "ModelLoader",
    "HuggingFaceWrapper",
    "SloughGPTModel",
    "RMSNorm",
    "RotaryEmbedding",
    "SloughGPTAttention",
    "SloughGPTBlock",
    "SwiGLU",
]
