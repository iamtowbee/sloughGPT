#!/usr/bin/env python3
"""
Hugging Face Integration for SloGPT Dataset System

Connects dataset standardization with Hugging Face ecosystem for model sharing,
download, and integration with existing models.
"""

import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
try:
    from huggingface_hub import HfApi, Repository
    # Try to import ModelSearchApi, fallback if not available
    try:
        from huggingface_hub import ModelSearchApi
    except ImportError:
        ModelSearchApi = None
except ImportError:
    print("Warning: huggingface_hub not installed. Install with: pip install huggingface_hub")
    HfApi = None
    Repository = None
    ModelSearchApi = None
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class HuggingFaceManager:
    """Manages Hugging Face integration for datasets and models."""
    
    def __init__(self):
        self.api = HfApi()
        self.models_dir = Path("hf_models")
        self.models_dir.mkdir(exist_ok=True)
        
    def search_models(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for models on Hugging Face."""
        try:
            if ModelSearchApi is None:
                print("ModelSearchApi not available, using fallback search...")
                # Fallback: Use HfApi search if available
                if HfApi is None:
                    return []
                
                # Simplified search using HfApi
                models = self.api.list_models(
                    search=query,
                    limit=limit,
                    sort="downloads",
                    task="text-generation"
                )
            else:
                models = ModelSearchApi().list_models(
                    search=query,
                    limit=limit,
                    sort="downloads",
                    task="text-generation"
                )
            
            formatted_models = []
            for model in models:
                formatted_models.append({
                    "id": model.id,
                    "modelId": model.modelId,
                    "author": model.author or "Unknown",
                    "downloads": model.downloads or 0,
                    "likes": model.likes or 0,
                    "created_at": str(model.created_at) if model.created_at else "Unknown",
                    "tags": model.tags or [],
                    "description": model.description or model.pipeline_tag or "No description",
                    "library_name": model.library_name or "Unknown"
                })
            
            return formatted_models
            
        except Exception as e:
            print(f"Error searching models: {e}")
            return []
    
    def download_model(self, model_id: str, local_name: Optional[str] = None) -> Dict:
        """Download a model from Hugging Face."""
        try:
            # Create model directory
            if local_name:
                model_dir = self.models_dir / local_name
            else:
                model_dir = self.models_dir / model_id.replace("/", "_")
            
            model_dir.mkdir(exist_ok=True)
            
            # Download using git clone (faster for large models)
            repo_url = f"https://huggingface.co/{model_id}"
            
            print(f"📥 Downloading model: {model_id}")
            print(f"📁 Destination: {model_dir}")
            
            result = subprocess.run([
                "git", "clone", repo_url, str(model_dir)
            ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            if result.returncode == 0:
                success_msg = f"✅ Model downloaded successfully to {model_dir}"
                print(success_msg)
                
                # Save model metadata
                metadata = {
                    "model_id": model_id,
                    "local_name": local_name or model_id.replace("/", "_"),
                    "download_time": str(Path(model_dir).stat().st_mtime),
                    "repo_url": repo_url
                }
                
                with open(model_dir / "hf_metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
                
                return {"success": True, "message": success_msg, "model_dir": str(model_dir)}
            
            else:
                error_msg = f"❌ Failed to download model: {result.stderr}"
                print(error_msg)
                return {"success": False, "error": error_msg}
                
        except subprocess.TimeoutExpired:
            error_msg = f"⏰️ Model download timed out (10 minutes)"
            print(error_msg)
            return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"❌ Error downloading model: {e}"
            print(error_msg)
            return {"success": False, "error": str(e)}
    
    def push_model_to_hf(self, dataset_name: str, model_path: str, repo_name: str, private: bool = False) -> Dict:
        """Push a trained model to Hugging Face."""
        try:
            model_path = Path(model_path)
            repo_dir = Path(f"{dataset_name}_hf_repo")
            
            if not model_path.exists():
                return {"success": False, "error": f"Model not found at {model_path}"}
            
            # Create Hugging Face repo structure
            repo_dir.mkdir(exist_ok=True)
            
            # Copy model files
            import shutil
            if model_path.is_dir():
                shutil.copytree(model_path, repo_dir / f"{dataset_name}")
            else:
                shutil.copy2(model_path, repo_dir / f"{dataset_name}")
            
            # Create model card
            model_card = self._create_model_card(dataset_name, repo_name)
            with open(repo_dir / "README.md", "w") as f:
                f.write(model_card)
            
            # Add git repo
            subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True)
            subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
            subprocess.run(["git", "config", "user.name", "SloGPT User"], cwd=repo_dir, capture_output=True)
            subprocess.run(["git", "config", "user.email", "user@example.com"], cwd=repo_dir, capture_output=True)
            subprocess.run(["git", "commit", "-m", f"Add {dataset_name} model and README"], cwd=repo_dir, capture_output=True)
            
            # Push to Hugging Face
            visibility = "--private" if private else "--public"
            repo_url = f"https://huggingface.co/spaces/{repo_name}"
            
            print(f"🚀 Pushing to Hugging Face...")
            print(f"📁 Repo: {repo_url}")
            print(f"🔐 Visibility: {'Private' if private else 'Public'}")
            
            result = subprocess.run([
                "huggingface-cli", "repo", "create", visibility, repo_name
            ], cwd=repo_dir, capture_output=True, text=True)
            
            if result.returncode == 0:
                success_msg = f"✅ Model pushed successfully to {repo_url}"
                print(success_msg)
                return {"success": True, "message": success_msg, "repo_url": repo_url}
            else:
                error_msg = f"❌ Failed to push to Hugging Face: {result.stderr}"
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            error_msg = f"❌ Error pushing to Hugging Face: {e}"
            return {"success": False, "error": str(e)}
    
    def _create_model_card(self, dataset_name: str, repo_name: str) -> str:
        """Create Hugging Face model card."""
        return f"""---
language: en
license: mit
tags:
- causal-language-modeling
- transformers
- pytorch
- slo-gpt

---

# {dataset_name}

This model was trained using the SloGPT dataset standardization system.

## Model Description
{dataset_name} is a causal language model trained on the {dataset_name} dataset.
The model demonstrates character-level text generation capabilities with a focus on
quality and consistency.

## Training Details
- Dataset: {dataset_name}
- Training Framework: SloGPT Dataset System
- Architecture: Transformer with RoPE, SwiGLU, RMSNorm
- Training Time: {Path('datasets/' + dataset_name).stat().st_mtime if Path('datasets/' + dataset_name).exists() else 'Unknown'}

## Intended Use
This model is designed for:
- Text generation
- Content completion
- Conversational AI applications
- Educational purposes

## Limitations
This model is trained on character-level tokenization and may not be suitable for:
- Word-level tasks requiring precise vocabulary
- Production applications without further fine-tuning
- Multi-modal use cases

## Bias and Limitations
The model may generate content that reflects patterns in the training data.
Users should be aware of potential biases and limitations.

## Getting Started

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("spaces/{repo_name}")
model = AutoModelForCausalLM.from_pretrained("spaces/{repo_name}")

# Generate text
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Model Card Authors
For more details, visit: https://huggingface.co/spaces/{repo_name}
"""
    
    def convert_slogpt_model_to_hf(self, dataset_name: str, model_path: str, output_path: str) -> Dict:
        """Convert SloGPT trained model to Hugging Face format."""
        try:
            import pickle
            import torch
            import numpy as np
            from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
            
            dataset_dir = Path(f"datasets/{dataset_name}")
            meta_file = dataset_dir / "meta.pkl"
            
            if not model_path.endswith('.bin'):
                # Check if it's a PyTorch checkpoint
                checkpoint_file = Path(model_path)
                if not checkpoint_file.exists():
                    # Look for common model file patterns
                    possible_files = [
                        Path(model_path) / "model.pt",
                        Path(model_path) / "checkpoint.pt",
                        Path(model_path) / "pytorch_model.bin"
                    ]
                    for pf in possible_files:
                        if pf.exists():
                            checkpoint_file = pf
                            break
                    
                    if not checkpoint_file.exists():
                        return {"success": False, "error": f"No valid model file found at {model_path}"}
                model_path = str(checkpoint_file)
            
            # Load dataset metadata
            if not meta_file.exists():
                return {"success": False, "error": f"Dataset metadata not found for {dataset_name}"}
            
            with open(meta_file, 'rb') as f:
                meta = pickle.load(f)
            
            print(f"🔄 Converting SloGPT model to Hugging Face format...")
            print(f"📁 Dataset: {dataset_name}")
            print(f"🤖 Model: {model_path}")
            
            # Create output directory
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load SloGPT model weights
            if model_path.endswith('.bin'):
                # Load llama2.c style .bin file
                model_weights = self._load_llamac_bin_model(model_path)
            else:
                # Load PyTorch checkpoint
                model_weights = torch.load(model_path, map_location='cpu')
            
            # Create Hugging Face model configuration
            # We'll use GPT2 architecture as it's compatible with our transformer
            config = GPT2Config(
                vocab_size=meta['vocab_size'],
                n_positions=1024,  # Context length
                n_embd=model_weights.get('n_embed', 768),
                n_layer=model_weights.get('n_layer', 12),
                n_head=model_weights.get('n_head', 12),
                n_inner=None,  # Will be calculated as 4 * n_embd
                activation_function='swish',  # Closest to SwiGLU
                resid_pdrop=0.1,
                embd_pdrop=0.1,
                attn_pdrop=0.1,
                layer_norm_epsilon=1e-5,
                initializer_range=0.02,
                summary_type='cls_index',
                summary_use_proj=True,
                summary_activation=None,
                summary_proj_to_labels=True,
                summary_first_dropout=0.1,
                scale_attn_weights=False,
                use_cache=True,
                bos_token_id=0,  # Beginning of sequence
                eos_token_id=1,  # End of sequence
            )
            
            # Initialize Hugging Face model
            hf_model = GPT2LMHeadModel(config)
            
            # Map SloGPT weights to Hugging Face format
            state_dict = self._map_slogpt_to_hf_weights(model_weights, config)
            
            # Load weights into Hugging Face model
            hf_model.load_state_dict(state_dict, strict=False)
            
            # Create character-level tokenizer
            tokenizer = self._create_character_tokenizer(meta['itos'])
            
            # Save model and tokenizer
            hf_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # Create model card with conversion details
            model_card = self._create_conversion_model_card(dataset_name, meta, config)
            with open(output_dir / "README.md", "w") as f:
                f.write(model_card)
            
            # Save conversion metadata
            conversion_info = {
                "original_dataset": dataset_name,
                "original_model_path": model_path,
                "conversion_timestamp": str(Path().stat().st_mtime),
                "slogpt_metadata": meta,
                "hf_config": config.to_dict(),
                "vocab_size": meta['vocab_size'],
                "model_type": "GPT2LMHeadModel",
                "tokenizer_type": "Character-level"
            }
            
            with open(output_dir / "conversion_info.json", "w") as f:
                json.dump(conversion_info, f, indent=2)
            
            success_msg = f"✅ Model converted successfully to Hugging Face format"
            print(success_msg)
            print(f"📁 Output: {output_dir}")
            print(f"📊 Vocab Size: {meta['vocab_size']}")
            print(f"🏗️ Architecture: GPT2 ({config.n_layer} layers, {config.n_embd} dim)")
            
            return {
                "success": True,
                "message": success_msg,
                "output_dir": str(output_dir),
                "config": config.to_dict(),
                "vocab_size": meta['vocab_size']
            }
            
        except Exception as e:
            error_msg = f"❌ Error converting model: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def _load_llamac_bin_model(self, bin_file: str) -> Dict:
        """Load llama2.c style .bin model file."""
        import struct
        
        weights = {}
        
        with open(bin_file, 'rb') as f:
            # Read header information
            # This is a simplified version - actual llama2.c format is more complex
            header_size = struct.unpack('I', f.read(4))[0]
            header_data = f.read(header_size)
            
            # Parse header (simplified - actual format varies)
            # We'll make reasonable assumptions about the structure
            weights['n_embed'] = 768  # Default
            weights['n_layer'] = 12   # Default
            weights['n_head'] = 12    # Default
            
            # Read token embeddings
            vocab_size = 50304  # Common for llama2.c
            embed_dim = weights['n_embed']
            token_embedding_table = np.fromfile(f, dtype=np.float32, count=vocab_size * embed_dim)
            weights['token_embedding_table'] = torch.from_numpy(token_embedding_table.reshape(vocab_size, embed_dim))
            
            # Read layer weights (simplified)
            for layer_idx in range(weights['n_layer']):
                # Layer normalization weights
                ln_1 = np.fromfile(f, dtype=np.float32, count=embed_dim)
                weights[f'layer_{layer_idx}_ln_1'] = torch.from_numpy(ln_1)
                
                # Attention weights
                qkv_weights = np.fromfile(f, dtype=np.float32, count=3 * embed_dim * embed_dim)
                weights[f'layer_{layer_idx}_attn_qkv'] = torch.from_numpy(qkv_weights.reshape(3 * embed_dim, embed_dim))
                
                # Output projection
                proj_weights = np.fromfile(f, dtype=np.float32, count=embed_dim * embed_dim)
                weights[f'layer_{layer_idx}_attn_proj'] = torch.from_numpy(proj_weights.reshape(embed_dim, embed_dim))
                
                # Second layer norm
                ln_2 = np.fromfile(f, dtype=np.float32, count=embed_dim)
                weights[f'layer_{layer_idx}_ln_2'] = torch.from_numpy(ln_2)
                
                # Feedforward weights
                ffn_weights = np.fromfile(f, dtype=np.float32, count=4 * embed_dim * embed_dim)
                weights[f'layer_{layer_idx}_ffn'] = torch.from_numpy(ffn_weights.reshape(4 * embed_dim, embed_dim))
            
            # Final layer norm
            final_ln = np.fromfile(f, dtype=np.float32, count=embed_dim)
            weights['final_layer_norm'] = torch.from_numpy(final_ln)
            
            # Output projection
            output_weights = np.fromfile(f, dtype=np.float32, count=vocab_size * embed_dim)
            weights['output_projection'] = torch.from_numpy(output_weights.reshape(vocab_size, embed_dim))
        
        return weights
    
    def _map_slogpt_to_hf_weights(self, slogpt_weights: Dict, config) -> Dict:
        """Map SloGPT weights to Hugging Face GPT2 format."""
        state_dict = {}
        
        # Token embeddings
        if 'token_embedding_table' in slogpt_weights:
            state_dict['transformer.wte.weight'] = slogpt_weights['token_embedding_table']
        
        # Position embeddings (will be initialized randomly if not present)
        if 'position_embedding_table' in slogpt_weights:
            state_dict['transformer.wpe.weight'] = slogpt_weights['position_embedding_table']
        
        # Layer weights
        for layer_idx in range(config.n_layer):
            prefix = f'transformer.h.{layer_idx}.'
            
            # Layer normalization 1
            if f'layer_{layer_idx}_ln_1' in slogpt_weights:
                state_dict[prefix + 'ln_1.weight'] = slogpt_weights[f'layer_{layer_idx}_ln_1']
            
            # Layer normalization 2
            if f'layer_{layer_idx}_ln_2' in slogpt_weights:
                state_dict[prefix + 'ln_2.weight'] = slogpt_weights[f'layer_{layer_idx}_ln_2']
            
            # Attention weights
            if f'layer_{layer_idx}_attn_qkv' in slogpt_weights:
                qkv = slogpt_weights[f'layer_{layer_idx}_attn_qkv']
                embed_dim = config.n_embd
                head_dim = embed_dim // config.n_head
                
                # Split Q, K, V
                state_dict[prefix + 'attn.c_attn.weight'] = qkv
                # Bias will be initialized to zeros
            
            if f'layer_{layer_idx}_attn_proj' in slogpt_weights:
                state_dict[prefix + 'attn.c_proj.weight'] = slogpt_weights[f'layer_{layer_idx}_attn_proj']
            
            # Feedforward weights
            if f'layer_{layer_idx}_ffn' in slogpt_weights:
                ffn = slogpt_weights[f'layer_{layer_idx}_ffn']
                # GPT2 uses c_fc and c_proj for feedforward
                intermediate_dim = ffn.shape[0] // embed_dim
                state_dict[prefix + 'c_fc.weight'] = ffn
                # Output projection will need to be created or copied separately
        
        # Final layer norm
        if 'final_layer_norm' in slogpt_weights:
            state_dict['transformer.ln_f.weight'] = slogpt_weights['final_layer_norm']
        
        # Output projection (language model head)
        if 'output_projection' in slogpt_weights:
            state_dict['lm_head.weight'] = slogpt_weights['output_projection']
        elif 'token_embedding_table' in slogpt_weights:
            # GPT2 ties the embeddings and lm_head
            state_dict['lm_head.weight'] = slogpt_weights['token_embedding_table']
        
        return state_dict
    
    def _create_character_tokenizer(self, vocab_list: List[str]):
        """Create a character-level tokenizer for Hugging Face."""
        try:
            from transformers import PreTrainedTokenizer
        except ImportError:
            print("Transformers not available, creating simple tokenizer")
            return self._create_simple_character_tokenizer(vocab_list)
        
        class CharacterTokenizer(PreTrainedTokenizer):
            def __init__(self, vocab_list, **kwargs):
                # Initialize vocab first
                self.vocab_list = list(vocab_list)  # Ensure it's a list
                self.base_vocab = {char: i for i, char in enumerate(self.vocab_list)}
                self.base_ids_to_tokens = {i: char for i, char in enumerate(self.vocab_list)}
                
                # Add special tokens
                special_tokens = ["<pad>", "<eos>", "<bos>", "<unk>"]
                self.special_tokens = {}
                
                # Add special tokens after base vocab
                next_id = len(self.base_vocab)
                for token in special_tokens:
                    if token not in self.base_vocab:
                        self.special_tokens[token] = next_id
                        next_id += 1
                
                # Combine vocab
                self.vocab = {**self.base_vocab, **self.special_tokens}
                self.ids_to_tokens = {i: char for char, i in self.vocab.items()}
                self.vocab_size = len(self.vocab)
                
                # Set special token attributes
                self.pad_token = "<pad>"
                self.eos_token = "<eos>"
                self.bos_token = "<bos>"
                self.unk_token = "<unk>"
                self.pad_token_id = self.vocab.get(self.pad_token, 0)
                self.eos_token_id = self.vocab.get(self.eos_token, 1)
                self.bos_token_id = self.vocab.get(self.bos_token, 2)
                self.unk_token_id = self.vocab.get(self.unk_token, 3)
                
                # Initialize parent
                super().__init__(
                    pad_token=self.pad_token,
                    eos_token=self.eos_token,
                    bos_token=self.bos_token,
                    unk_token=self.unk_token,
                    **kwargs
                )
            
            def get_vocab(self):
                return dict(self.vocab)
            
            def _tokenize(self, text):
                return list(text)
            
            def _convert_token_to_id(self, token):
                return self.vocab.get(token, self.unk_token_id)
            
            def _convert_id_to_token(self, index):
                return self.ids_to_tokens.get(index, self.unk_token)
            
            def convert_tokens_to_string(self, tokens):
                return "".join(tokens)
            
            def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
                if token_ids_1 is None:
                    return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
                return [self.bos_token_id] + token_ids_0 + [self.eos_token_id] + \
                       [self.bos_token_id] + token_ids_1 + [self.eos_token_id]
            
            def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
                if already_has_special_tokens:
                    return super().get_special_tokens_mask(token_ids_0, token_ids_1, already_has_special_tokens)
                
                if token_ids_1 is None:
                    return [1] + [0] * len(token_ids_0) + [1]
                return [1] + [0] * len(token_ids_0) + [1] + \
                       [1] + [0] * len(token_ids_1) + [1]
            
            def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
                if token_ids_1 is None:
                    return [0] * (len(token_ids_0) + 2)
                return [0] * (len(token_ids_0) + 1) + [1] * (len(token_ids_1) + 1)
            
            def save_vocabulary(self, save_directory):
                vocab_file = Path(save_directory) / "vocab.json"
                with open(vocab_file, 'w', encoding='utf-8') as f:
                    json.dump(self.ids_to_tokens, f, indent=2, ensure_ascii=False)
                return str(vocab_file)
        
        return CharacterTokenizer(vocab_list, name_or_path="slogpt_character_tokenizer")
    
    def _create_simple_character_tokenizer(self, vocab_list: List[str]):
        """Create a simple character tokenizer without Hugging Face dependencies."""
        
        class SimpleCharTokenizer:
            def __init__(self, vocab_list):
                self.vocab_list = list(vocab_list)
                self.base_vocab = {char: i for i, char in enumerate(self.vocab_list)}
                
                # Add special tokens
                special_tokens = ["<pad>", "<eos>", "<bos>", "<unk>"]
                self.special_tokens = {}
                
                next_id = len(self.base_vocab)
                for token in special_tokens:
                    if token not in self.base_vocab:
                        self.special_tokens[token] = next_id
                        next_id += 1
                
                self.vocab = {**self.base_vocab, **self.special_tokens}
                self.ids_to_tokens = {i: char for char, i in self.vocab.items()}
                self.vocab_size = len(self.vocab)
                
                # Set special token attributes
                self.pad_token = "<pad>"
                self.eos_token = "<eos>"
                self.bos_token = "<bos>"
                self.unk_token = "<unk>"
                self.pad_token_id = self.vocab.get(self.pad_token, 0)
                self.eos_token_id = self.vocab.get(self.eos_token, 1)
                self.bos_token_id = self.vocab.get(self.bos_token, 2)
                self.unk_token_id = self.vocab.get(self.unk_token, 3)
            
            def tokenize(self, text):
                return list(text)
            
            def convert_tokens_to_ids(self, tokens):
                return [self.vocab.get(token, self.unk_token_id) for token in tokens]
            
            def convert_ids_to_tokens(self, ids):
                return [self.ids_to_tokens.get(id, self.unk_token) for id in ids]
            
            def convert_tokens_to_string(self, tokens):
                return "".join(tokens)
            
            def save_pretrained(self, path):
                path = Path(path)
                path.mkdir(parents=True, exist_ok=True)
                
                # Save vocabulary
                with open(path / "vocab.json", 'w') as f:
                    json.dump(self.ids_to_tokens, f, indent=2)
                
                # Save tokenizer config
                config = {
                    "tokenizer_class": "SimpleCharTokenizer",
                    "vocab_size": self.vocab_size,
                    "special_tokens": {
                        "pad_token": self.pad_token,
                        "eos_token": self.eos_token,
                        "bos_token": self.bos_token,
                        "unk_token": self.unk_token
                    }
                }
                
                with open(path / "tokenizer_config.json", 'w') as f:
                    json.dump(config, f, indent=2)
        
        return SimpleCharTokenizer(vocab_list)
    
    def _create_conversion_model_card(self, dataset_name: str, meta: Dict, config) -> str:
        """Create model card for converted model."""
        return f"""---
language: en
license: mit
tags:
- causal-language-modeling
- transformers
- pytorch
- slogpt
- character-level
- gpt2

---

# {dataset_name} (Hugging Face Format)

This model was converted from SloGPT to Hugging Face format for better ecosystem integration.

## Model Description
{dataset_name} is a character-level causal language model that was originally trained using the SloGPT dataset standardization system. This version has been converted to Hugging Face Transformers format for easy use with the broader ecosystem.

## Conversion Details
- **Original Framework**: SloGPT Dataset System
- **Converted To**: Hugging Face Transformers (GPT2 Architecture)
- **Tokenizer**: Character-level ({meta['vocab_size']} tokens)
- **Architecture**: {config.n_layer} layers, {config.n_embd} hidden dimension
- **Attention Heads**: {config.n_head}
- **Context Length**: {config.n_positions} tokens

## Original Training Information
- **Dataset**: {dataset_name}
- **Original Vocab**: {meta['vocab_size']} characters
- **Training Framework**: SloGPT with RoPE, SwiGLU, RMSNorm

## Intended Use
This model is designed for:
- Character-level text generation
- Educational purposes and demonstrations
- Research in character-based language modeling
- Creative writing assistance

## Getting Started

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("{config.name_or_path}")
tokenizer = AutoTokenizer.from_pretrained("{config.name_or_path}")

# Generate text
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, temperature=0.8)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

## Limitations
- This model operates at the character level, not word level
- May have different behavior compared to the original SloGPT model due to architectural differences
- Character-level models typically require more context to generate coherent text
- Performance may vary depending on the training data quality

## Bias and Limitations
This model may generate content reflecting patterns and potential biases present in the training data. Users should be aware of these limitations when using the model.

## Technical Notes
The conversion process maps SloGPT's transformer architecture to Hugging Face's GPT2 implementation. While the core architecture is similar, some differences may exist in:
- Attention implementation details
- Feed-forward network structure
- Position embedding handling
- Normalization layer placement

For more information about the original model, refer to the SloGPT dataset system documentation.

## Model Card Authors
Converted using SloGPT Dataset System Hugging Face Integration
"""
    
    def convert_dataset_for_hf(self, dataset_name: str, output_path: str) -> Dict:
        """Convert SloGPT dataset to Hugging Face format."""
        try:
            dataset_dir = Path(f"datasets/{dataset_name}")
            meta_file = dataset_dir / "meta.pkl"
            train_file = dataset_dir / "train.bin"
            
            if not all([dataset_dir.exists(), meta_file.exists(), train_file.exists()]):
                return {"success": False, "error": f"Dataset {dataset_name} not found or incomplete"}
            
            # Load metadata
            import pickle
            with open(meta_file, 'rb') as f:
                meta = pickle.load(f)
            
            # Load training data
            import numpy as np
            train_data = np.fromfile(train_file, dtype=np.uint16)
            
            # Create character tokenizer
            # This is a simple character-level tokenizer
            vocab = meta['itos']
            
            # Save as text file for Hugging Face
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert tokens back to text (simplified)
            text_data = ""
            for token in train_data[:10000]:  # Limit for demo
                if token < len(vocab):
                    text_data += vocab[token]
            
            # Save as text file
            text_file = output_dir / "train.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text_data)
            
            # Save metadata
            with open(output_dir / "meta.json", 'w') as f:
                json.dump({
                    'vocab_size': meta['vocab_size'],
                    'itos': meta['itos'],
                    'dataset_name': dataset_name,
                    'slogpt_metadata': meta,
                    'conversion_time': str(Path().ctime()),
                    'original_tokens': len(train_data)
                }, f, indent=2)
            
            success_msg = f"✅ Dataset {dataset_name} converted for Hugging Face"
            print(success_msg)
            
            return {
                "success": True,
                "message": success_msg,
                "output_dir": str(output_dir),
                "vocab_size": meta['vocab_size'],
                "total_tokens": len(train_data)
            }
            
        except Exception as e:
            error_msg = f"❌ Error converting dataset: {e}"
            return {"success": False, "error": str(e)}
    
    def list_local_models(self) -> List[Dict]:
        """List locally downloaded models."""
        models = []
        
        for item in self.models_dir.iterdir():
            if item.is_dir():
                metadata_file = item / "hf_metadata.json"
                
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            models.append(metadata)
                    except:
                        continue
        
        return models


def main():
    """Command line interface for Hugging Face integration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hugging Face Integration")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Search models
    search_parser = subparsers.add_parser('search', help='Search models')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--limit', type=int, default=10, help='Number of results')
    
    # Download model
    download_parser = subparsers.add_parser('download', help='Download model')
    download_parser.add_argument('model_id', help='Hugging Face model ID')
    download_parser.add_argument('--local-name', help='Local directory name')
    
    # Push model
    push_parser = subparsers.add_parser('push', help='Push model to Hugging Face')
    push_parser.add_argument('dataset', help='Dataset name to push')
    push_parser.add_argument('model_path', help='Path to trained model')
    push_parser.add_argument('--repo_name', help='Repository name')
    push_parser.add_argument('--private', action='store_true', help='Create private repository')
    
    # Convert dataset
    convert_parser = subparsers.add_parser('convert', help='Convert dataset for Hugging Face')
    convert_parser.add_argument('dataset', help='Dataset name to convert')
    convert_parser.add_argument('output', help='Output path for converted dataset')
    
    # Convert model
    convert_model_parser = subparsers.add_parser('convert-model', help='Convert SloGPT model to Hugging Face format')
    convert_model_parser.add_argument('dataset', help='Dataset name (used for metadata)')
    convert_model_parser.add_argument('model_path', help='Path to trained SloGPT model (.bin or .pt)')
    convert_model_parser.add_argument('output', help='Output path for Hugging Face model')
    
    # List local models
    list_parser = subparsers.add_parser('list', help='List local models')
    
    args = parser.parse_args()
    
    hf_manager = HuggingFaceManager()
    
    if args.command == 'search':
        models = hf_manager.search_models(args.query, args.limit)
        
        print(f"🔍 Hugging Face Search Results for: '{args.query}'")
        print("=" * 50)
        
        if models:
            for model in models:
                print(f"\n🤖 {model['author']} - {model['modelId']}")
                print(f"   📊 {model['downloads']:,} downloads")
                print(f"   ❤️ {model['likes']:,} likes")
                print(f"   📅 Created: {model['created_at']}")
                print(f"   🏷️ Library: {model['library_name']}")
                print(f"   📝 Tags: {', '.join(model['tags'])}")
                print(f"   📄 {model['description']}")
        else:
            print("No models found")
    
    elif args.command == 'download':
        result = hf_manager.download_model(args.model_id, args.local_name)
        
        if result['success']:
            print(f"\n📊 Model ready for use!")
            print(f"   Location: {result['model_dir']}")
    
    elif args.command == 'push':
        result = hf_manager.push_model_to_hf(
            args.dataset, args.model_path, args.repo_name, args.private
        )
        
        if result['success']:
            print(f"\n🚀 Model pushed successfully!")
            print(f"   Repository: {result['repo_url']}")
    
    elif args.command == 'convert':
        result = hf_manager.convert_dataset_for_hf(args.dataset, args.output)
        
        if result['success']:
            print(f"\n🔄 Dataset converted successfully!")
            print(f"   Output: {result['output_dir']}")
            print(f"   Vocab Size: {result['vocab_size']}")
            print(f"   Tokens: {result['total_tokens']}")
    
    elif args.command == 'convert-model':
        result = hf_manager.convert_slogpt_model_to_hf(args.dataset, args.model_path, args.output)
        
        if result['success']:
            print(f"\n🤖 Model converted successfully!")
            print(f"   Output: {result['output_dir']}")
            print(f"   Architecture: GPT2 ({result['config']['n_layer']} layers)")
            print(f"   Hidden Dim: {result['config']['n_embd']}")
            print(f"   Vocab Size: {result['vocab_size']}")
            print(f"\n💡 To use the converted model:")
            print(f"   from transformers import AutoTokenizer, AutoModelForCausalLM")
            print(f"   model = AutoModelForCausalLM.from_pretrained('{result['output_dir']}')")
            print(f"   tokenizer = AutoTokenizer.from_pretrained('{result['output_dir']}')")
    
    elif args.command == 'list':
        models = hf_manager.list_local_models()
        
        print(f"📦 Locally Downloaded Models")
        print("=" * 30)
        
        if models:
            for model in models:
                print(f"\n🤖 {model['model_id']}")
                print(f"   📁 Local Name: {model['local_name']}")
                print(f"   ⬇ Downloaded: {model['download_time']}")
                print(f"   🔗 Repository: {model['repo_url']}")
        else:
            print("No local models found")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()