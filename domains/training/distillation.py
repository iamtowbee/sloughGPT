"""
Knowledge Distillation for SloughGPT

Implements knowledge distillation for model compression:
- Temperature-based distillation
- Label smoothing
- Feature-based distillation
- Progressive distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger("sloughgpt.distillation")


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""

    # Temperature
    temperature: float = 4.0  # Higher = softer probabilities
    temperature_schedule: Optional[List[float]] = None

    # Loss weights
    alpha: float = 0.5  # Weight for student loss (hard labels)
    beta: float = 0.5  # Weight for distillation loss (soft labels)
    gamma: float = 0.0  # Weight for feature distillation

    # Distillation type
    distillation_type: str = "logits"  # logits, hidden_states, attention
    use_label_smoothing: bool = False
    label_smoothing: float = 0.1

    # Progressive distillation
    progressive: bool = False
    stage_weights: Optional[List[float]] = None

    # Feature distillation
    hidden_layer_mapping: Optional[Dict[int, int]] = None  # student_layer -> teacher_layer


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.

    Combines:
    - Hard label loss (cross-entropy with labels)
    - Soft label loss (KL divergence with teacher outputs)
    - Feature loss (MSE on hidden states)
    """

    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing if config.use_label_smoothing else 0.0
        )

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        student_hidden: Optional[torch.Tensor] = None,
        teacher_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute distillation loss.

        Args:
            student_logits: Student model logits [batch, seq, vocab]
            teacher_logits: Teacher model logits [batch, seq, vocab]
            labels: Ground truth labels [batch, seq]
            student_hidden: Student hidden states [batch, seq, hidden]
            teacher_hidden: Teacher hidden states [batch, seq, hidden]

        Returns:
            Total loss and individual loss components
        """
        losses = {}

        # Soft label loss (KL divergence)
        if self.config.beta > 0:
            # Apply temperature scaling
            student_soft = F.log_softmax(student_logits / self.config.temperature, dim=-1)
            teacher_soft = F.softmax(teacher_logits / self.config.temperature, dim=-1)

            soft_loss = self.kl_loss(student_soft, teacher_soft)
            # Scale by temperature^2 as per Hinton et al.
            soft_loss = soft_loss * (self.config.temperature**2)
            losses["soft_loss"] = soft_loss.item()

        # Hard label loss (cross-entropy)
        if self.config.alpha > 0 and labels is not None:
            # Reshape for sequence classification
            student_flat = student_logits.view(-1, student_logits.size(-1))
            labels_flat = labels.view(-1)

            hard_loss = self.ce_loss(student_flat, labels_flat)
            losses["hard_loss"] = hard_loss.item()

        # Feature distillation loss
        if self.config.gamma > 0 and student_hidden is not None and teacher_hidden is not None:
            # Match hidden state dimensions if needed
            if student_hidden.size(-1) != teacher_hidden.size(-1):
                # Project student hidden to teacher dimension
                student_hidden = self._project_hidden(student_hidden, teacher_hidden.size(-1))

            feat_loss = self.mse_loss(student_hidden, teacher_hidden)
            losses["feature_loss"] = feat_loss.item()

        # Combine losses
        total_loss = torch.tensor(0.0, device=student_logits.device)

        if self.config.alpha > 0 and labels is not None:
            total_loss = total_loss + self.config.alpha * hard_loss

        if self.config.beta > 0:
            total_loss = total_loss + self.config.beta * soft_loss

        if self.config.gamma > 0 and student_hidden is not None:
            total_loss = total_loss + self.config.gamma * feat_loss

        losses["total_loss"] = total_loss.item()

        return total_loss, losses

    def _project_hidden(
        self,
        student_hidden: torch.Tensor,
        target_dim: int,
    ) -> torch.Tensor:
        """Project student hidden states to match teacher dimension."""
        if not hasattr(self, "projection"):
            # Create projection layer lazily
            self.projection = nn.Linear(student_hidden.size(-1), target_dim).to(
                student_hidden.device
            )

        return self.projection(student_hidden)


class DistillationTrainer:
    """
    Trainer for knowledge distillation.

    Handles:
    - Loading teacher and student models
    - Computing distillation loss
    - Progressive distillation schedules
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        config: DistillationConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.config = config
        self.device = device

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        # Student optimizer
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=1e-4,
        )

        self.loss_fn = DistillationLoss(config)

        # For progressive distillation
        self.current_stage = 0
        self.stage_weights = config.stage_weights or [1.0]

    def step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Perform one distillation step.

        Args:
            inputs: Input tokens [batch, seq]
            labels: Target labels [batch, seq]

        Returns:
            Dictionary of losses
        """
        # Get teacher predictions
        with torch.no_grad():
            teacher_outputs = self.teacher(inputs)
            teacher_logits = (
                teacher_outputs if isinstance(teacher_outputs, torch.Tensor) else teacher_outputs[0]
            )

        # Get student predictions
        student_outputs = self.student(inputs)
        student_logits = (
            student_outputs if isinstance(student_outputs, torch.Tensor) else student_outputs[0]
        )

        # Handle different sequence dimensions
        if teacher_logits.size(1) != student_logits.size(1):
            # Truncate to shorter sequence
            min_len = min(teacher_logits.size(1), student_logits.size(1))
            teacher_logits = teacher_logits[:, :min_len, :]
            student_logits = student_logits[:, :min_len, :]
            labels = labels[:, :min_len]

        # Compute distillation loss
        loss, losses = self.loss_fn(
            student_logits,
            teacher_logits,
            labels,
        )

        # Backward and update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return losses

    def distill_logits(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float = 4.0,
    ) -> torch.Tensor:
        """
        Distill from teacher logits to student.

        Args:
            student_logits: [batch, seq, vocab]
            teacher_logits: [batch, seq, vocab]
            temperature: Temperature for softening

        Returns:
            Distillation loss
        """
        student_soft = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

        loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean")
        loss = loss * (temperature**2)

        return loss

    def distill_hidden_states(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        projection: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Distill from teacher hidden states to student.

        Args:
            student_hidden: [batch, seq, hidden]
            teacher_hidden: [batch, seq, hidden]
            projection: Optional projection layer

        Returns:
            Feature distillation loss
        """
        if projection is not None:
            student_hidden = projection(student_hidden)

        return F.mse_loss(student_hidden, teacher_hidden)

    def distill_attention(
        self,
        student_attn: torch.Tensor,
        teacher_attn: torch.Tensor,
    ) -> torch.Tensor:
        """
        Distill attention patterns.

        Args:
            student_attn: [batch, heads, seq, seq]
            teacher_attn: [batch, heads, seq, seq]

        Returns:
            Attention distillation loss
        """
        # Align dimensions if needed
        if student_attn.size(1) != teacher_attn.size(1):
            # Reduce student heads to match teacher
            ratio = teacher_attn.size(1) // student_attn.size(1)
            student_attn = student_attn.repeat(1, ratio, 1, 1) / ratio

        return F.mse_loss(student_attn, teacher_attn)


class ProgressiveDistiller:
    """
    Progressive knowledge distillation.

    Distills layer-by-layer from teacher to student.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        config: DistillationConfig,
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.config = config

        # Identify layers
        self.teacher_layers = self._get_layers(teacher_model)
        self.student_layers = self._get_layers(student_model)

        # Create layer mappings
        self.layer_mapping = self._create_layer_mapping()

    def _get_layers(self, model: nn.Module) -> List[nn.Module]:
        """Extract transformer layers from model."""
        layers = []
        for name, module in model.named_modules():
            if "block" in name.lower() or "layer" in name.lower():
                if len(list(module.children())) > 0:
                    layers.append(module)
        return layers

    def _create_layer_mapping(self) -> Dict[int, int]:
        """Create mapping from student layers to teacher layers."""
        num_teacher = len(self.teacher_layers)
        num_student = len(self.student_layers)

        if num_student >= num_teacher:
            return {i: i for i in range(num_student)}

        # Map student layers to teacher layers
        mapping = {}
        for i in range(num_student):
            teacher_idx = int(i * num_teacher / num_student)
            mapping[i] = teacher_idx

        return mapping

    def distill_intermediate(
        self,
        inputs: torch.Tensor,
        intermediate_losses: List[torch.Tensor],
    ) -> torch.Tensor:
        """Distill intermediate representations."""
        total_loss = torch.tensor(0.0)

        # Get intermediate representations
        # (This would require hooks to capture intermediate states)

        return total_loss


def create_distillation_trainer(
    teacher_model: nn.Module,
    student_model: nn.Module,
    temperature: float = 4.0,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> DistillationTrainer:
    """
    Create a distillation trainer.

    Args:
        teacher_model: Larger model to learn from
        student_model: Smaller model to train
        temperature: Temperature for softening
        alpha: Weight for hard labels
        beta: Weight for soft labels

    Returns:
        DistillationTrainer instance
    """
    config = DistillationConfig(
        temperature=temperature,
        alpha=alpha,
        beta=beta,
    )

    return DistillationTrainer(teacher_model, student_model, config)


__all__ = [
    "DistillationConfig",
    "DistillationLoss",
    "DistillationTrainer",
    "ProgressiveDistiller",
    "create_distillation_trainer",
]
