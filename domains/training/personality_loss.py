"""
Personality Contrastive Loss for SloughGPT

Training loss functions for personality alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import logging

logger = logging.getLogger("sloughgpt.personality")


class PersonalityContrastiveLoss(nn.Module):
    """
    Contrastive loss for personality training.
    
    Pulls samples with similar personalities together,
    pushes dissimilar ones apart in embedding space.
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        margin: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(
        self,
        embeddings: torch.Tensor,
        traits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embeddings: [batch_size, embed_dim] - text embeddings
            traits: [batch_size, num_traits] - personality traits
            
        Returns:
            Loss scalar
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create labels from traits (similar = similar traits)
        # traits: [batch_size, num_traits]
        trait_sim = torch.matmul(traits, traits.T)  # [batch, batch]
        
        # Positive pairs: high trait similarity
        labels = (trait_sim > self.margin).float()
        
        # Mask out diagonal
        mask = torch.eye(embeddings.size(0), device=embeddings.device)
        labels = labels * (1 - mask)
        
        # Compute loss
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Mean log-likelihood for pos positive pairs
        loss = -(labels * log_prob).sum() / (labels.sum() + 1e-8)
        
        return loss


class PersonalityMSELoss(nn.Module):
    """
    MSE loss for predicting personality traits from embeddings.
    """
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE between predicted and target traits."""
        return F.mse_loss(predictions, targets)


class PersonalityTripletLoss(nn.Module):
    """
    Triplet loss for personality alignment.
    
    Ensures anchor-positive distance < anchor-negative distance.
    """
    
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """Triplet loss computation."""
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class PersonalityLoss(nn.Module):
    """
    Combined personality loss for training.
    
    Combines multiple loss functions:
    - Contrastive loss for embedding alignment
    - MSE loss for trait prediction
    - Optional triplet loss
    """
    
    def __init__(
        self,
        contrastive_weight: float = 1.0,
        mse_weight: float = 0.5,
        triplet_weight: float = 0.3,
        temperature: float = 0.1,
    ):
        super().__init__()
        
        self.contrastive_weight = contrastive_weight
        self.mse_weight = mse_weight
        self.triplet_weight = triplet_weight
        
        self.contrastive_loss = PersonalityContrastiveLoss(temperature=temperature)
        self.mse_loss = PersonalityMSELoss()
        self.triplet_loss = PersonalityTripletLoss()
    
    def forward(
        self,
        embeddings: torch.Tensor,
        traits: torch.Tensor,
        trait_predictions: Optional[torch.Tensor] = None,
        positive_emb: Optional[torch.Tensor] = None,
        negative_emb: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined personality loss.
        
        Args:
            embeddings: Text embeddings [batch, dim]
            traits: Target personality traits [batch, num_traits]
            trait_predictions: Predicted traits from model [batch, num_traits]
            positive_emb: Positive samples for triplet [batch, dim]
            negative_emb: Negative samples for triplet [batch, dim]
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        total_loss = 0.0
        
        # Contrastive loss
        if self.contrastive_weight > 0:
            cont_loss = self.contrastive_loss(embeddings, traits)
            losses["contrastive"] = cont_loss
            total_loss += self.contrastive_weight * cont_loss
        
        # MSE loss for trait prediction
        if self.mse_weight > 0 and trait_predictions is not None:
            mse = self.mse_loss(trait_predictions, traits)
            losses["mse"] = mse
            total_loss += self.mse_weight * mse
        
        # Triplet loss
        if self.triplet_weight > 0 and positive_emb is not None and negative_emb is not None:
            trip = self.triplet_loss(embeddings, positive_emb, negative_emb)
            losses["triplet"] = trip
            total_loss += self.triplet_weight * trip
        
        losses["total"] = total_loss
        return losses


class ArchetypeAlignmentLoss(nn.Module):
    """
    Loss for aligning outputs to personality archetypes.
    
    Ensures model outputs match target archetype distributions.
    """
    
    def __init__(self, num_archetypes: int = 8):
        super().__init__()
        self.num_archetypes = num_archetypes
        
        # Predefined archetype trait vectors
        self.register_buffer(
            "archetypes",
            torch.tensor([
                [0.9, 0.8, 0.5, 0.5],  # sage
                [0.9, -0.5, 0.6, 0.9],  # innocent
                [0.5, 0.9, 0.9, 0.5],  # explorer
                [0.9, 0.9, 0.8, 0.5],  # caregiver
                [0.9, 0.8, 0.7, 0.9],  # ruler
                [0.9, 0.9, 0.7, 0.8],  # rebel
                [0.9, 0.8, 0.7, 0.5],  # magician
                [0.7, 0.9, 0.8, 0.9],  # jester
            ], dtype=torch.float32)
        )
    
    def forward(
        self,
        traits: torch.Tensor,
        target_archetype: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute archetype alignment loss.
        
        Args:
            traits: Predicted traits [batch, num_traits]
            target_archetype: One-hot target archetype [batch]
            
        Returns:
            Loss scalar
        """
        # Compute similarity to each archetype
        traits_norm = F.normalize(traits, p=2, dim=1)
        archetypes_norm = F.normalize(self.archetypes, p=2, dim=1)
        
        sim = torch.matmul(traits_norm, archetypes_norm.T)  # [batch, num_archetypes]
        
        if target_archetype is not None:
            # Supervised: align to target archetype
            target_sim = (sim * target_archetype).sum(dim=1)
            loss = 1 - target_sim.mean()
        else:
            # Unsupervised: find closest archetype and minimize distance
            max_sim = sim.max(dim=1)[0]
            loss = 1 - max_sim.mean()
        
        return loss


class PersonalityFineTuningLoss(nn.Module):
    """
    Complete loss for personality fine-tuning.
    
    Combines:
    - Language modeling loss (next token prediction)
    - Personality alignment loss
    - Optional KL divergence for smoothness
    """
    
    def __init__(
        self,
        lm_weight: float = 1.0,
        personality_weight: float = 0.5,
        kl_weight: float = 0.1,
    ):
        super().__init__()
        
        self.lm_weight = lm_weight
        self.personality_weight = personality_weight
        self.kl_weight = kl_weight
        
        self.personality_loss = PersonalityLoss()
        self.archetype_loss = ArchetypeAlignmentLoss()
    
    def forward(
        self,
        lm_logits: torch.Tensor,  # [batch, seq, vocab]
        lm_targets: torch.Tensor,  # [batch, seq]
        traits: torch.Tensor,  # [batch, num_traits]
        trait_predictions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute complete fine-tuning loss.
        
        Args:
            lm_logits: Language model output logits
            lm_targets: Target tokens for LM
            traits: Target personality traits
            trait_predictions: Predicted traits from model
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        total_loss = 0.0
        
        # Language modeling loss (cross-entropy)
        if self.lm_weight > 0:
            lm_loss = F.cross_entropy(
                lm_logits.view(-1, lm_logits.size(-1)),
                lm_targets.view(-1),
                ignore_index=-100,
            )
            losses["lm"] = lm_loss
            total_loss += self.lm_weight * lm_loss
        
        # Personality loss
        if self.personality_weight > 0:
            # Get embeddings from logits (mean pooling)
            embeddings = lm_logits.mean(dim=1)  # [batch, vocab] -> [batch, embed]
            
            pers_losses = self.personality_loss(
                embeddings=embeddings,
                traits=traits,
                trait_predictions=trait_predictions,
            )
            
            for k, v in pers_losses.items():
                losses[f"personality_{k}"] = v
                total_loss += self.personality_weight * v
        
        losses["total"] = total_loss
        return losses


# =============================================================================
# Convenience Functions
# =============================================================================

def create_personality_loss(
    loss_type: str = "combined",
    **kwargs,
) -> nn.Module:
    """
    Create personality loss by type.
    
    Args:
        loss_type: "contrastive", "mse", "triplet", "combined", "finetuning"
        **kwargs: Loss-specific parameters
        
    Returns:
        Loss module
    """
    if loss_type == "contrastive":
        return PersonalityContrastiveLoss(**kwargs)
    elif loss_type == "mse":
        return PersonalityMSELoss()
    elif loss_type == "triplet":
        return PersonalityTripletLoss(**kwargs)
    elif loss_type == "combined":
        return PersonalityLoss(**kwargs)
    elif loss_type == "finetuning":
        return PersonalityFineTuningLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


__all__ = [
    "PersonalityContrastiveLoss",
    "PersonalityMSELoss",
    "PersonalityTripletLoss",
    "PersonalityLoss",
    "ArchetypeAlignmentLoss",
    "PersonalityFineTuningLoss",
    "create_personality_loss",
]
