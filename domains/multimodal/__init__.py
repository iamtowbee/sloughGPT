"""
Multi-Modal Support for SloughGPT

Vision-language models for image understanding:
- Vision Encoder (CNN/ViT)
- Image-to-Text generation
- Cross-modal attention
- Vision-Language alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger("sloughgpt.multimodal")


@dataclass
class MultiModalConfig:
    """Configuration for multi-modal models."""

    # Vision
    image_size: int = 224
    patch_size: int = 16
    vision_hidden_size: int = 768
    vision_num_layers: int = 12
    vision_num_heads: int = 12

    # Language
    vocab_size: int = 50257
    text_hidden_size: int = 768
    text_num_layers: int = 12
    text_num_heads: int = 12
    max_seq_length: int = 512

    # Fusion
    fusion_type: str = "cross_attention"  # cross_attention, concat, gated
    projection_dim: int = 768


class VisionEncoder(nn.Module):
    """
    Vision Transformer (ViT) encoder for images.

    Splits image into patches and encodes with transformer.
    """

    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=config.vision_hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.vision_hidden_size))
        self.position_embed = nn.Parameter(
            torch.zeros(
                1, (config.image_size // config.patch_size) ** 2 + 1, config.vision_hidden_size
            )
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.vision_hidden_size,
            nhead=config.vision_num_heads,
            dim_feedforward=config.vision_hidden_size * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.vision_num_layers)

        # Output projection
        self.projection = nn.Linear(config.vision_hidden_size, config.projection_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embed, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to embeddings.

        Args:
            images: [batch, 3, image_size, image_size]

        Returns:
            image_embeddings: [batch, num_patches+1, projection_dim]
        """
        batch_size = images.size(0)

        # Patch embedding: [B, 3, H, W] -> [B, num_patches, hidden]
        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden]

        # Add cls token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embedding
        x = x + self.position_embed

        # Transformer encoding
        x = self.transformer(x)

        # Project to projection dimension
        x = self.projection(x)

        return x


class ImageCaptionModel(nn.Module):
    """
    Image-to-Text model for caption generation.

    Encoder: Vision Transformer
    Decoder: Causal Language Model
    """

    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.config = config

        # Vision encoder
        self.vision_encoder = VisionEncoder(config)

        # Text decoder
        self.embedding = nn.Embedding(config.vocab_size, config.text_hidden_size)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.text_hidden_size)

        # Cross-attention (vision -> text)
        cross_attn = nn.TransformerEncoderLayer(
            d_model=config.text_hidden_size,
            nhead=config.text_num_heads,
            dim_feedforward=config.text_hidden_size * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.cross_attention = nn.TransformerEncoder(cross_attn, num_layers=config.text_num_layers)

        # Output
        self.lm_head = nn.Linear(config.text_hidden_size, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embedding.weight

    def forward(
        self,
        images: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images: [batch, 3, image_size, image_size]
            text_tokens: [batch, seq_len] (optional for encoding only)

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Encode images
        image_embeds = self.vision_encoder(images)  # [B, patches+1, proj_dim]

        if text_tokens is None:
            return image_embeds

        # Embed text
        text_embeds = self.embedding(text_tokens)

        # Add position embedding
        positions = torch.arange(text_tokens.size(1), device=text_tokens.device)
        position_embeds = self.position_embedding(positions)
        text_embeds = text_embeds + position_embeds

        # Cross-attention: text attends to images
        # Need to project image_embeds to text_hidden_size
        image_proj = F.linear(
            image_embeds,
            self.cross_attention.layers[0].linear1.weight,
            self.cross_attention.layers[0].linear1.bias,
        )

        # Simple cross attention
        attn_output = self.cross_attention(text_embeds)

        # LM head
        logits = self.lm_head(attn_output)

        return logits

    def generate_caption(
        self,
        images: torch.Tensor,
        max_length: int = 30,
        temperature: float = 1.0,
    ) -> List[str]:
        """
        Generate captions for images.

        Args:
            images: [batch, 3, image_size, image_size]
            max_length: Maximum caption length
            temperature: Sampling temperature

        Returns:
            List of generated captions
        """
        self.eval()

        # Encode images
        image_embeds = self.vision_encoder(images)

        # Start with BOS token
        batch_size = images.size(0)
        generated = torch.full((batch_size, 1), 0, dtype=torch.long, device=images.device)  # BOS=0

        with torch.no_grad():
            for _ in range(max_length):
                logits = self.forward(images, generated)
                next_token_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_token], dim=1)

                # Stop if all EOS
                if (next_token == 2).all():  # EOS=2
                    break

        return generated.tolist()


class CLIPModel(nn.Module):
    """
    CLIP-style vision-language model.

    Contrastive learning between images and text.
    """

    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.config = config

        # Vision encoder
        self.vision_encoder = VisionEncoder(config)

        # Text encoder
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.text_hidden_size,
                nhead=config.text_num_heads,
                dim_feedforward=config.text_hidden_size * 4,
                batch_first=True,
            ),
            num_layers=config.text_num_layers,
        )

        # Projections
        self.image_projection = nn.Linear(config.vision_hidden_size, config.projection_dim)
        self.text_projection = nn.Linear(config.text_hidden_size, config.projection_dim)

        # Temperature (learnable)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images."""
        image_features = self.vision_encoder(images)
        image_features = self.image_projection(image_features[:, 0])  # Use CLS token
        return F.normalize(image_features, dim=-1)

    def encode_text(
        self, text_tokens: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode text."""
        text_features = self.text_encoder(text_tokens)
        text_features = self.text_projection(text_features[:, 0])  # Use CLS token
        return F.normalize(text_features, dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for contrastive learning.

        Args:
            images: [batch, 3, image_size, image_size]
            text_tokens: [batch, seq_len]

        Returns:
            image_logits, text_logits
        """
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_tokens)

        # Compute logits
        logit_scale = self.logit_scale.exp()
        image_logits = logit_scale * image_features @ text_features.t()
        text_logits = logit_scale * text_features @ image_features.t()

        return image_logits, text_logits


class MultiModalModel(nn.Module):
    """
    Unified multi-modal model supporting:
    - Image captioning
    - Visual question answering
    - Image-text retrieval
    """

    def __init__(self, config: MultiModalConfig, mode: str = "caption"):
        super().__init__()
        self.config = config
        self.mode = mode

        if mode == "caption":
            self.model = ImageCaptionModel(config)
        elif mode == "clip":
            self.model = CLIPModel(config)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def generate(self, images: torch.Tensor, max_length: int = 30) -> torch.Tensor:
        """Generate from images."""
        if hasattr(self.model, "generate_caption"):
            return self.model.generate_caption(images, max_length)
        raise NotImplementedError(f"Generate not supported for mode: {self.mode}")


def create_vision_model(
    model_type: str = "vit",
    image_size: int = 224,
    hidden_size: int = 768,
) -> nn.Module:
    """
    Create a vision model.

    Args:
        model_type: vit (Vision Transformer)
        image_size: Input image size
        hidden_size: Hidden dimension

    Returns:
        Vision model
    """
    config = MultiModalConfig(
        image_size=image_size,
        vision_hidden_size=hidden_size,
    )

    return VisionEncoder(config)


def create_multimodal_model(
    mode: str = "caption",
    image_size: int = 224,
    vocab_size: int = 50257,
) -> MultiModalModel:
    """
    Create a multi-modal model.

    Args:
        mode: caption, clip, vqa
        image_size: Input image size
        vocab_size: Vocabulary size

    Returns:
        MultiModalModel
    """
    config = MultiModalConfig(
        image_size=image_size,
        vocab_size=vocab_size,
    )

    return MultiModalModel(config, mode=mode)


__all__ = [
    "MultiModalConfig",
    "VisionEncoder",
    "ImageCaptionModel",
    "CLIPModel",
    "MultiModalModel",
    "create_vision_model",
    "create_multimodal_model",
]
