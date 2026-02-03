#!/usr/bin/env python3
"""
Hugging Face Integration Demo
Demonstrates the complete workflow of converting SloGPT models to Hugging Face format.
"""

import os
import sys
import torch
import pickle
import numpy as np
from pathlib import Path

# Add the dataset system to path
sys.path.insert(0, str(Path(__file__).parent))

from create_dataset_fixed import create_dataset
from huggingface_integration import HuggingFaceManager


def demo_hf_integration():
    """Demonstrate Hugging Face integration capabilities."""
    print("🤖 Hugging Face Integration Demo")
    print("=" * 50)
    
    # Step 1: Create a demonstration dataset
    print("\n📝 Step 1: Creating Demo Dataset")
    print("-" * 30)
    
    demo_text = """
    In a kingdom far away, there lived a brave knight named Sir Reginald. 
    He was known throughout the land for his courage and wisdom.
    Every morning, Sir Reginald would practice his sword fighting in the castle courtyard.
    The villagers would gather to watch him train.
    His armor gleamed in the sunlight like polished silver.
    One day, a dragon appeared in the nearby mountains.
    Sir Reginald knew he had to protect his kingdom.
    With his trusty sword and loyal steed, he rode towards the mountain.
    The adventure was about to begin...
    """
    
    result = create_dataset("hf_demo", demo_text)
    
    if result and result.get('success'):
        print("✅ Demo dataset created successfully")
    else:
        print("❌ Failed to create demo dataset")
        return False
    
    # Step 2: Load dataset metadata
    print("\n📊 Step 2: Analyzing Dataset")
    print("-" * 30)
    
    dataset_dir = Path("datasets/hf_demo")
    meta_file = dataset_dir / "meta.pkl"
    
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f)
    
    print(f"   Vocab Size: {meta['vocab_size']}")
    print(f"   Character Set: {list(meta['itos'].values())[:15]}...")
    
    # Step 3: Create mock trained model
    print("\n🏗️ Step 3: Creating Mock Trained Model")
    print("-" * 30)
    
    # Create realistic model weights
    vocab_size = meta['vocab_size']
    n_embed = 256
    n_layer = 4
    n_head = 8
    
    model_weights = {
        'n_embed': n_embed,
        'n_layer': n_layer,
        'n_head': n_head,
        'vocab_size': vocab_size,
    }
    
    # Token embeddings
    model_weights['token_embedding_table'] = torch.randn(vocab_size, n_embed) * 0.02
    
    # Position embeddings
    model_weights['position_embedding_table'] = torch.randn(1024, n_embed) * 0.02
    
    # Layer weights
    for i in range(n_layer):
        model_weights[f'layer_{i}_ln_1'] = torch.ones(n_embed)
        model_weights[f'layer_{i}_attn_qkv'] = torch.randn(3 * n_embed, n_embed) * 0.02
        model_weights[f'layer_{i}_attn_proj'] = torch.randn(n_embed, n_embed) * 0.02
        model_weights[f'layer_{i}_ln_2'] = torch.ones(n_embed)
        model_weights[f'layer_{i}_ffn'] = torch.randn(4 * n_embed, n_embed) * 0.02
    
    # Final layers
    model_weights['final_layer_norm'] = torch.ones(n_embed)
    model_weights['output_projection'] = torch.randn(vocab_size, n_embed) * 0.02
    
    # Save mock model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    demo_model_dir = models_dir / "hf_demo"
    demo_model_dir.mkdir(exist_ok=True)
    
    model_file = demo_model_dir / "model.pt"
    torch.save(model_weights, model_file)
    
    print(f"✅ Mock model saved to {model_file}")
    print(f"   Architecture: {n_layer} layers, {n_embed} dim, {n_head} heads")
    
    # Step 4: Convert to Hugging Face format
    print("\n🔄 Step 4: Converting to Hugging Face Format")
    print("-" * 30)
    
    hf_manager = HuggingFaceManager()
    
    output_dir = "hf_converted_demo"
    
    try:
        # Mock GPT2Config class for demo
        class MockGPT2Config:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
            
            def to_dict(self):
                return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
        # Test weight mapping (the core conversion logic)
        config = MockGPT2Config(
            vocab_size=vocab_size,
            n_embd=n_embed,
            n_layer=n_layer,
            n_head=n_head,
            n_positions=1024
        )
        
        mapped_weights = hf_manager._map_slogpt_to_hf_weights(model_weights, config)
        
        print("✅ Weight mapping successful!")
        print(f"   Mapped {len(mapped_weights)} weight tensors")
        
        for key in list(mapped_weights.keys())[:5]:  # Show first 5
            print(f"   ✓ {key}: {mapped_weights[key].shape}")
        
        print(f"   ... and {len(mapped_weights) - 5} more tensors")
        
        # Step 5: Create tokenizer
        print("\n🔤 Step 5: Creating Character Tokenizer")
        print("-" * 30)
        
        try:
            # Use simple tokenizer for demo
            tokenizer = hf_manager._create_simple_character_tokenizer(list(meta['itos'].values()))
            
            print("✅ Character tokenizer created!")
            print(f"   Vocabulary: {tokenizer.vocab_size} tokens")
            
            # Test tokenization
            test_text = "Hello dragon!"
            tokens = tokenizer.tokenize(test_text)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            
            print(f"   Sample: '{test_text}'")
            print(f"   Tokens: {tokens}")
            print(f"   IDs: {token_ids}")
            
        except Exception as e:
            print(f"⚠️ Tokenizer creation issue: {e}")
        
        # Step 6: Save converted model files
        print("\n💾 Step 6: Saving Hugging Face Model Files")
        print("-" * 30)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save model weights
        torch.save(mapped_weights, output_path / "pytorch_model.bin")
        
        # Save config
        import json
        config_dict = config.to_dict()
        config_dict.update({
            "model_type": "gpt2",
            "torch_dtype": "float32",
            "transformers_version": "4.21.0",
            "use_cache": True
        })
        
        with open(output_path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
        
        # Create README
        readme_content = f"""---
language: en
license: mit
tags:
- causal-language-modeling
- transformers
- pytorch
- slogpt
- character-level

# SloGPT Demo Model (Hugging Face Format)

This is a demonstration model converted from SloGPT to Hugging Face format.

## Model Details
- **Original Dataset**: hf_demo
- **Vocabulary Size**: {vocab_size} characters
- **Architecture**: GPT2 ({n_layer} layers, {n_embed} hidden)
- **Tokenizer**: Character-level
- **Context Length**: 1024 tokens

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./{output_dir}")
tokenizer = AutoTokenizer.from_pretrained("./{output_dir}")

# Generate text
prompt = "Sir Reginald"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50, temperature=0.8)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

## Notes
This is a demonstration model showing the conversion process from SloGPT to Hugging Face format.
The model operates at the character level and was trained on a small demo dataset.
"""
        
        with open(output_path / "README.md", 'w') as f:
            f.write(readme_content)
        
        print(f"✅ Hugging Face model saved to {output_dir}/")
        print("   Files created:")
        for file in output_path.glob("*"):
            print(f"     📄 {file.name}")
        
        # Step 7: Summary
        print("\n🎉 Demo Summary")
        print("-" * 30)
        print("✅ Dataset creation: SUCCESS")
        print("✅ Model weights generation: SUCCESS")
        print("✅ Weight mapping to HF format: SUCCESS")
        print("✅ Character tokenizer creation: SUCCESS")
        print("✅ Hugging Face model files: SUCCESS")
        
        print(f"\n💡 Your converted model is ready at: {output_dir}/")
        print("🚀 You can now use it with the Hugging Face Transformers library!")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_demo():
    """Clean up demo files."""
    print("\n🧹 Cleaning up demo files...")
    
    cleanup_items = [
        "datasets/hf_demo",
        "models/hf_demo",
        "hf_converted_demo"
    ]
    
    for item in cleanup_items:
        path = Path(item)
        if path.exists():
            if path.is_dir():
                import shutil
                shutil.rmtree(path)
                print(f"   🗑️ Removed {item}/")
            else:
                path.unlink()
                print(f"   🗑️ Removed {item}")


def main():
    """Run the Hugging Face integration demo."""
    success = demo_hf_integration()
    
    if success:
        print(f"\n🎊 HUGGING FACE INTEGRATION DEMO COMPLETED SUCCESSFULLY!")
        print("\n📚 What we demonstrated:")
        print("   • Creating SloGPT datasets")
        print("   • Generating realistic model weights")
        print("   • Converting to Hugging Face format")
        print("   • Creating character-level tokenizers")
        print("   • Saving compatible model files")
        
        print(f"\n🔧 Commands you can now use:")
        print(f"   python3 huggingface_integration.py convert-model <dataset> <model_path> <output>")
        print(f"   python3 huggingface_integration.py convert <dataset> <output>")
        print(f"   python3 huggingface_integration.py search <query>")
        
    else:
        print(f"\n❌ Demo encountered issues. Check the error messages above.")
    
    # Ask if user wants to cleanup
    try:
        response = input(f"\n🧹 Clean up demo files? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            cleanup_demo()
    except KeyboardInterrupt:
        print("\n\n👋 Demo completed. Files preserved for inspection.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)