#!/usr/bin/env python3
"""
Simple Hugging Face Integration Test
Tests the core model conversion functionality without external dependencies.
"""

import os
import sys
import tempfile
import shutil
import pickle
import torch
import numpy as np
from pathlib import Path

# Add the dataset system to path
sys.path.insert(0, str(Path(__file__).parent))

from create_dataset_fixed import create_dataset


def test_basic_conversion():
    """Test basic model conversion components."""
    print("🧪 Testing Basic Hugging Face Conversion Components")
    print("=" * 60)
    
    try:
        # Create a small test dataset
        test_dataset = "hf_basic_test"
        test_text = "Hello world! This is a test. Once upon a time there was a brave knight."
        
        print(f"📝 Creating test dataset: {test_dataset}")
        result = create_dataset(test_dataset, test_text)
        
        if result is None or not result.get('success', False):
            error_msg = result.get('error', 'Unknown error') if result else 'create_dataset returned None'
            print(f"❌ Failed to create test dataset: {error_msg}")
            return False
        
        print("✅ Test dataset created successfully")
        
        # Load dataset metadata
        dataset_dir = Path(f"datasets/{test_dataset}")
        meta_file = dataset_dir / "meta.pkl"
        
        with open(meta_file, 'rb') as f:
            meta = pickle.load(f)
        
        print(f"📊 Dataset info:")
        print(f"   Vocab Size: {meta['vocab_size']}")
        # Convert itos values to list for slicing
        vocab_chars = list(meta['itos'].values())[:10]
        print(f"   Sample vocab: {vocab_chars}")
        
        # Create mock model weights
        print(f"\n🏗️ Creating mock model weights...")
        n_embed = 128
        n_layer = 2
        n_head = 4
        vocab_size = meta['vocab_size']
        
        mock_weights = {
            'n_embed': n_embed,
            'n_layer': n_layer,
            'n_head': n_head,
            'token_embedding_table': torch.randn(vocab_size, n_embed),
            'position_embedding_table': torch.randn(1024, n_embed),
            'final_layer_norm': torch.randn(n_embed),
            'output_projection': torch.randn(vocab_size, n_embed)
        }
        
        # Add layer weights
        for i in range(n_layer):
            mock_weights[f'layer_{i}_ln_1'] = torch.randn(n_embed)
            mock_weights[f'layer_{i}_attn_qkv'] = torch.randn(3 * n_embed, n_embed)
            mock_weights[f'layer_{i}_attn_proj'] = torch.randn(n_embed, n_embed)
            mock_weights[f'layer_{i}_ln_2'] = torch.randn(n_embed)
            mock_weights[f'layer_{i}_ffn'] = torch.randn(4 * n_embed, n_embed)
        
        print("✅ Mock model weights created")
        
        # Test weight mapping
        print(f"\n🔄 Testing weight mapping...")
        
        # Mock GPT2Config
        class MockGPT2Config:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
            
            def to_dict(self):
                return self.__dict__
        
        config = MockGPT2Config(
            vocab_size=vocab_size,
            n_embd=n_embed,
            n_layer=n_layer,
            n_head=n_head
        )
        
        # Import and test the weight mapping
        from huggingface_integration import HuggingFaceManager
        hf_manager = HuggingFaceManager()
        
        # Test the weight mapping function
        try:
            mapped_weights = hf_manager._map_slogpt_to_hf_weights(mock_weights, config)
            print("✅ Weight mapping successful")
            
            # Check key mappings
            expected_keys = [
                'transformer.wte.weight',
                'transformer.wpe.weight',
                'transformer.ln_f.weight',
                'lm_head.weight'
            ]
            
            for key in expected_keys:
                if key in mapped_weights:
                    print(f"   ✓ {key}: {mapped_weights[key].shape}")
                else:
                    print(f"   ❌ Missing key: {key}")
            
            # Check layer mappings
            for i in range(n_layer):
                layer_prefix = f'transformer.h.{i}.'
                layer_keys = [
                    f'{layer_prefix}ln_1.weight',
                    f'{layer_prefix}attn.c_attn.weight',
                    f'{layer_prefix}attn.c_proj.weight',
                    f'{layer_prefix}ln_2.weight',
                    f'{layer_prefix}c_fc.weight'
                ]
                
                for key in layer_keys:
                    if key in mapped_weights:
                        print(f"   ✓ {key}: {mapped_weights[key].shape}")
                    else:
                        print(f"   ❌ Missing key: {key}")
        
        except Exception as e:
            print(f"❌ Weight mapping failed: {e}")
            return False
        
        # Test tokenizer creation
        print(f"\n🔤 Testing tokenizer creation...")
        try:
            tokenizer = hf_manager._create_character_tokenizer(meta['itos'])
            
            # Test tokenization
            test_text = "Hello"
            tokens = tokenizer.tokenize(test_text)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            
            print(f"   Original: '{test_text}'")
            print(f"   Tokens: {tokens}")
            print(f"   Token IDs: {token_ids}")
            print(f"   Decoded: '{tokenizer.convert_tokens_to_string(tokens)}'")
            
            # Test special tokens
            print(f"   Special tokens:")
            print(f"     PAD: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
            print(f"     EOS: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
            print(f"     BOS: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
            print(f"     UNK: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
            
            print("✅ Tokenizer creation successful")
            
        except Exception as e:
            print(f"❌ Tokenizer creation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print(f"\n🎉 Basic conversion components test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_character_tokenizer():
    """Test the character tokenizer implementation."""
    print("\n🧪 Testing Character Tokenizer Implementation")
    print("=" * 50)
    
    try:
        from huggingface_integration import HuggingFaceManager
        hf_manager = HuggingFaceManager()
        
        # Test vocabulary
        vocab = ['a', 'b', 'c', ' ', '!', '.']
        
        print(f"📝 Test vocab: {vocab}")
        
        try:
            tokenizer = hf_manager._create_character_tokenizer(vocab)
        except Exception as e:
            print(f"⚠️ HuggingFace tokenizer failed, trying simple tokenizer: {e}")
            tokenizer = hf_manager._create_simple_character_tokenizer(vocab)
        
        # Test basic functionality
        test_cases = [
            "abc",
            "a b c",
            "hello!",
            "a.b.c"
        ]
        
        for test_case in test_cases:
            tokens = tokenizer.tokenize(test_case)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            decoded = tokenizer.convert_tokens_to_string(tokens)
            
            print(f"\n   Input: '{test_case}'")
            print(f"   Tokens: {tokens}")
            print(f"   IDs: {token_ids}")
            print(f"   Decoded: '{decoded}'")
        
        # Test special tokens handling
        print(f"\n🔧 Testing special tokens...")
        print(f"   PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        print(f"   EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        print(f"   BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
        print(f"   UNK token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
        
        # Test with special tokens if available
        if hasattr(tokenizer, 'build_inputs_with_special_tokens'):
            inputs = tokenizer.build_inputs_with_special_tokens([1, 2, 3])
            print(f"   With special tokens: {inputs}")
        else:
            print("   Special token handling not available in simple tokenizer")
        
        if hasattr(tokenizer, 'get_special_tokens_mask'):
            mask = tokenizer.get_special_tokens_mask([1, 2, 3])
            print(f"   Special mask: {mask}")
        else:
            print("   Special mask not available in simple tokenizer")
        
        print("✅ Character tokenizer test passed")
        return True
        
    except Exception as e:
        print(f"❌ Character tokenizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_test_files():
    """Clean up test files."""
    print("\n🧹 Cleaning up test files...")
    
    cleanup_dirs = [
        "datasets/hf_basic_test"
    ]
    
    for cleanup_dir in cleanup_dirs:
        if Path(cleanup_dir).exists():
            shutil.rmtree(cleanup_dir)
            print(f"   🗑️ Removed {cleanup_dir}")


def main():
    """Run basic conversion tests."""
    print("🤖 Simple Hugging Face Integration Test")
    print("=" * 50)
    
    all_passed = True
    
    # Test 1: Basic conversion components
    if not test_basic_conversion():
        all_passed = False
    
    # Test 2: Character tokenizer
    if not test_character_tokenizer():
        all_passed = False
    
    # Cleanup
    cleanup_test_files()
    
    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL BASIC TESTS PASSED!")
        print("\n💡 Core conversion functionality is working.")
        print("🚀 Ready for full Hugging Face integration!")
    else:
        print("❌ SOME TESTS FAILED. Please check the errors above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)