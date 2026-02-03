#!/usr/bin/env python3
"""
SloughGPT Neural Network Implementation
Enhanced transformer architecture with quantization and optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass

from config import ModelConfig

@dataclass
class AttentionOutput:
    """Output from attention layer"""
    hidden_states: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

class MultiHeadAttention(nn.Module):
    """Multi-head attention with optimizations"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query, Key, Value projections
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        
        # Output projection
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.attention_dropout)
        
        # Layer normalization
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose tensor for attention computation"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> AttentionOutput:
        
        batch_size, seq_length, hidden_size = hidden_states.size()
        
        # Linear transformations
        mixed_query_layer = self.query(hidden_states)
        
        # Reuse past key/value if provided
        if past_key_value is not None:
            past_key, past_value = past_key_value
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)
            
            # Concat with past
            key_layer = torch.cat([past_key, key_layer], dim=-2)
            value_layer = torch.cat([past_value, value_layer], dim=-2)
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        
# Apply attention mask if provided
        if attention_mask is not None:
                # Ensure attention mask has correct shape for attention scores
                # attention_scores shape: (batch_size, num_heads, seq_len, seq_len)
                if attention_mask.dim() == 3:
                    # Expand mask to match attention scores shape
                    attention_mask = attention_mask.unsqueeze(1).expand(-1, self.num_attention_heads, -1, -1)
                elif attention_mask.dim() == 2:
                    # Convert 2D mask to 4D
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
                    attention_mask = attention_mask.expand(-1, self.num_attention_heads, -1, -1)
                
                # Only add if shapes match
                if attention_scores.shape == attention_mask.shape:
                    attention_scores = attention_scores + attention_mask
                else:
                    # Create proper causal mask if shapes don't match
                    seq_len = attention_scores.size(-1)
                    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
                    # Match the exact shape of attention_scores
                    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(
                        attention_scores.size(0), self.num_attention_heads, -1, -1
                    ).to(attention_scores.device)
                    
                    # Only mask if shapes actually match
                    if causal_mask.shape == attention_scores.shape:
                        attention_scores = attention_scores.masked_fill(causal_mask, float('-inf'))
                    else:
                        # Fall back to basic causal masking
                        batch_size, num_heads, seq_q, seq_k = attention_scores.shape
                        causal_mask_simple = torch.triu(torch.ones(seq_q, seq_k), diagonal=1).bool()
                        causal_mask_simple = causal_mask_simple.unsqueeze(0).unsqueeze(0).expand(
                            batch_size, num_heads, -1, -1
                        ).to(attention_scores.device)
                        attention_scores = attention_scores.masked_fill(causal_mask_simple, float('-inf'))
        
        # Normalize attention scores
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Compute context layer
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape and project back
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        attention_output = self.dense(context_layer)
        attention_output = self.dropout(attention_output)
        
        # Residual connection and layer norm
        attention_output = self.LayerNorm(attention_output + hidden_states)
        
        # Update past key value for caching
        past_key_value = (key_layer, value_layer) if use_cache else None
        
        return AttentionOutput(
            hidden_states=attention_output,
            attention_weights=attention_probs,
            past_key_value=past_key_value
        )

class FeedForward(nn.Module):
    """Feed-forward network with optimizations"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Intermediate layer
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        
        # Output layer
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout)
        
        # Layer normalization
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Activation function
        self.activation = F.gelu if config.activation == "gelu" else F.relu
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Feed forward
        intermediate_output = self.activation(self.intermediate(hidden_states))
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        
        # Residual connection and layer norm
        return self.LayerNorm(layer_output + hidden_states)

class TransformerLayer(nn.Module):
    """Transformer layer with attention and feed-forward"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        # Self-attention
        attention_output = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        
        # Feed-forward
        output = self.feed_forward(attention_output.hidden_states)
        
        return output, attention_output.past_key_value

class SloughGPT(nn.Module):
    """Main SloughGPT model with enhanced features"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Map config attributes to expected names
        hidden_size = getattr(config, 'hidden_size', getattr(config, 'd_model', 512))
        num_attention_heads = getattr(config, 'num_attention_heads', getattr(config, 'n_heads', 8))
        num_hidden_layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layers', 6))
        intermediate_size = getattr(config, 'intermediate_size', getattr(config, 'd_ff', 2048))
        hidden_dropout = getattr(config, 'hidden_dropout', getattr(config, 'dropout', 0.1))
        max_position_embeddings = getattr(config, 'max_position_embeddings', getattr(config, 'max_seq_length', 1024))
        layer_norm_eps = getattr(config, 'layer_norm_eps', 1e-6)
        attention_dropout = getattr(config, 'attention_dropout', 0.1)
        activation = getattr(config, 'activation', 'gelu')
        vocab_size = getattr(config, 'vocab_size', 50257)
        
        # Store the mapped values for use by components
        self.config.hidden_size = hidden_size
        self.config.num_attention_heads = num_attention_heads  
        self.config.num_hidden_layers = num_hidden_layers
        self.config.intermediate_size = intermediate_size
        self.config.hidden_dropout = hidden_dropout
        self.config.max_position_embeddings = max_position_embeddings
        self.config.layer_norm_eps = layer_norm_eps
        self.config.attention_dropout = attention_dropout
        self.config.activation = activation
        self.config.vocab_size = vocab_size
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Layer normalization
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(hidden_dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(num_hidden_layers)
        ])
        
        # Output projection
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie weights
        self.lm_head.weight = self.token_embeddings.weight
        
        # Performance metrics
        self.forward_time = 0.0
        self.memory_usage = 0
        
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask for input"""
        batch_size, seq_length = input_ids.size()
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        
        # Expand for batch
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Convert to float and mask
        return mask.to(self.device).float().masked_fill(mask, float('-inf'))
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
                use_cache: bool = False) -> Dict[str, Any]:
        """
        Enhanced forward pass with error handling and optimization
        """
        start_time = time.time()
        
        try:
            batch_size, seq_length = input_ids.size()
            
            # Create position ids
            position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
            # Embeddings
            token_embeds = self.token_embeddings(input_ids)
            position_embeds = self.position_embeddings(position_ids)
            hidden_states = self.LayerNorm(token_embeds + position_embeds)
            hidden_states = self.dropout(hidden_states)
            
            # Create attention mask if not provided
            if attention_mask is None:
                attention_mask = self.create_attention_mask(input_ids)
            
            # Process through transformer layers
            all_past_key_values = [] if use_cache else None
            
            for i, layer in enumerate(self.layers):
                past_key_value = past_key_values[i] if past_key_values is not None else None
                
                hidden_states, pkv = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache
                )
                
                if use_cache and pkv is not None and all_past_key_values is not None:
                    all_past_key_values.append(pkv)
            
            # Compute logits
            logits = self.lm_head(hidden_states)
            
            # Update performance metrics
            self.forward_time = time.time() - start_time
            if torch.cuda.is_available():
                self.memory_usage = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
            return {
                'logits': logits,
                'hidden_states': hidden_states,
                'past_key_values': tuple(all_past_key_values) if use_cache and all_past_key_values else None,
                'forward_time': self.forward_time,
                'memory_usage': self.memory_usage
            }
            
        except Exception as e:
            logging.error(f"Forward pass failed: {e}")
            raise
    
    def generate(self, 
                input_ids: torch.Tensor,
                max_length: int = 50,
                temperature: float = 1.0,
                top_k: int = 50,
                top_p: float = 0.9,
                do_sample: bool = True,
                pad_token_id: Optional[int] = None) -> torch.Tensor:
        """
        Enhanced text generation with multiple sampling strategies
        """
        self.eval()
        
        with torch.no_grad():
            batch_size = input_ids.size(0)
            past_key_values = None
            
            for _ in range(max_length):
                # Forward pass
                outputs = self(input_ids, past_key_values=past_key_values, use_cache=True)
                
                logits = outputs['logits'][:, -1, :]  # Get last token logits
                past_key_values = outputs['past_key_values']
                
                # Apply temperature
                logits = logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append to input
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Check for EOS token
                if pad_token_id is not None and (next_token == pad_token_id).all():
                    break
        
        return input_ids
    
    def count_parameters(self) -> int:
        """Count total parameters in the model"""
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            'model_type': 'SloughGPT',
            'vocab_size': self.config.vocab_size,
            'hidden_size': self.config.hidden_size,
            'num_layers': self.config.num_hidden_layers,
            'num_attention_heads': self.config.num_attention_heads,
            'max_position_embeddings': self.config.max_position_embeddings,
            'total_parameters': self.count_parameters(),
            'trainable_parameters': self.count_trainable_parameters(),
            'device': str(self.device),
            'memory_usage_mb': self.memory_usage,
            'forward_time_ms': self.forward_time * 1000
        }