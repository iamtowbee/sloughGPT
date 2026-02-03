#!/usr/bin/env python3
"""
Hugging Face Integration Test Script
Tests the complete workflow of converting SloGPT models to Hugging Face format
and using them with the Transformers ecosystem.
"""

import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path

# Add the dataset system to path
sys.path.insert(0, str(Path(__file__).parent))

from huggingface_integration import HuggingFaceManager
from create_dataset_fixed import create_dataset
from train_simple import train_dataset


def test_model_conversion():
    """Test the complete model conversion workflow."""
    print("🧪 Testing Hugging Face Model Conversion")
    print("=" * 50)
    
    try:
        # Create a small test dataset
        test_dataset = "hf_conversion_test"
        test_text = """
        Once upon a time, there was a little village nestled in the mountains.
        The villagers were kind and hardworking, always helping each other.
        Every morning, the baker would wake up early to bake fresh bread.
        The smell of freshly baked bread would fill the entire village.
        Children would run to the bakery with coins in their pockets.
        Life was simple and beautiful in this mountain village.
        """
        
        print(f"📝 Creating test dataset: {test_dataset}")
        result = create_dataset(test_dataset, test_text)
        
        if not result['success']:
            print(f"❌ Failed to create test dataset: {result.get('error')}")
            return False
        
        # Train a simple model
        print(f"\n🚀 Training model on {test_dataset}")
        
        # Use the simple trainer with minimal steps for testing
        import subprocess
        result = subprocess.run([
            "python3", "train_simple.py", test_dataset, "--steps", "50", "--eval_interval", "25"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"❌ Training failed: {result.stderr}")
            return False
        
        print("✅ Model training completed")
        
        # Find the trained model file
        model_dir = Path(f"models/{test_dataset}")
        model_files = list(model_dir.glob("**/*.pt")) + list(model_dir.glob("**/*.bin"))
        
        if not model_files:
            print(f"❌ No trained model found in {model_dir}")
            return False
        
        model_file = model_files[0]
        print(f"📁 Found trained model: {model_file}")
        
        # Test model conversion
        print(f"\n🔄 Converting model to Hugging Face format...")
        hf_manager = HuggingFaceManager()
        
        output_dir = f"hf_converted_models/{test_dataset}"
        conversion_result = hf_manager.convert_slogpt_model_to_hf(
            test_dataset, str(model_file), output_dir
        )
        
        if not conversion_result['success']:
            print(f"❌ Model conversion failed: {conversion_result.get('error')}")
            return False
        
        print("✅ Model conversion completed successfully")
        
        # Test loading the converted model with Transformers
        print(f"\n🧪 Testing Hugging Face model loading...")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Load the converted model
            tokenizer = AutoTokenizer.from_pretrained(output_dir)
            model = AutoModelForCausalLM.from_pretrained(output_dir)
            
            print("✅ Model loaded successfully with Transformers")
            
            # Test text generation
            print(f"\n💬 Testing text generation...")
            prompt = "Once upon a time"
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Generate with short output for testing
            outputs = model.generate(
                **inputs, 
                max_length=50, 
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"📝 Generated text: {generated_text}")
            
            print("✅ Text generation successful")
            
        except Exception as e:
            print(f"❌ Failed to use converted model: {e}")
            return False
        
        # Test model metadata
        print(f"\n📊 Checking model metadata...")
        conversion_info_file = Path(output_dir) / "conversion_info.json"
        if conversion_info_file.exists():
            import json
            with open(conversion_info_file, 'r') as f:
                info = json.load(f)
            
            print(f"✅ Conversion metadata found")
            print(f"   Original Dataset: {info['original_dataset']}")
            print(f"   Model Type: {info['model_type']}")
            print(f"   Vocab Size: {info['vocab_size']}")
        
        print(f"\n🎉 All Hugging Face conversion tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_conversion():
    """Test dataset conversion functionality."""
    print("\n🧪 Testing Dataset Conversion")
    print("=" * 30)
    
    try:
        hf_manager = HuggingFaceManager()
        
        # Use the test dataset from the previous test
        test_dataset = "hf_conversion_test"
        output_dir = f"hf_converted_datasets/{test_dataset}"
        
        result = hf_manager.convert_dataset_for_hf(test_dataset, output_dir)
        
        if not result['success']:
            print(f"❌ Dataset conversion failed: {result.get('error')}")
            return False
        
        print("✅ Dataset conversion successful")
        print(f"   Output: {result['output_dir']}")
        print(f"   Vocab Size: {result['vocab_size']}")
        print(f"   Total Tokens: {result['total_tokens']}")
        
        # Check if files were created
        output_path = Path(output_dir)
        if output_path.exists():
            files = list(output_path.glob("*"))
            print(f"✅ Created {len(files)} files:")
            for file in files:
                print(f"   📄 {file.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset conversion test failed: {e}")
        return False


def test_hf_search_functionality():
    """Test Hugging Face model search."""
    print("\n🧪 Testing Hugging Face Search")
    print("=" * 30)
    
    try:
        hf_manager = HuggingFaceManager()
        
        # Search for small text generation models
        results = hf_manager.search_models("gpt2", limit=3)
        
        if not results:
            print("❌ No search results returned")
            return False
        
        print(f"✅ Found {len(results)} models")
        for i, model in enumerate(results):
            print(f"   {i+1}. {model['modelId']} ({model['downloads']:,} downloads)")
        
        return True
        
    except Exception as e:
        print(f"❌ Search functionality test failed: {e}")
        return False


def cleanup_test_files():
    """Clean up test files."""
    print("\n🧹 Cleaning up test files...")
    
    cleanup_dirs = [
        "datasets/hf_conversion_test",
        "models/hf_conversion_test", 
        "hf_converted_models/hf_conversion_test",
        "hf_converted_datasets/hf_conversion_test"
    ]
    
    for cleanup_dir in cleanup_dirs:
        if Path(cleanup_dir).exists():
            shutil.rmtree(cleanup_dir)
            print(f"   🗑️ Removed {cleanup_dir}")


def main():
    """Run all Hugging Face integration tests."""
    print("🤖 Hugging Face Integration Test Suite")
    print("=" * 50)
    
    all_passed = True
    
    # Test 1: Model conversion
    if not test_model_conversion():
        all_passed = False
    
    # Test 2: Dataset conversion
    if not test_dataset_conversion():
        all_passed = False
    
    # Test 3: Search functionality
    if not test_hf_search_functionality():
        all_passed = False
    
    # Cleanup
    cleanup_test_files()
    
    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Hugging Face integration is working correctly.")
        print("\n💡 You can now use these commands:")
        print("   python3 huggingface_integration.py convert-model <dataset> <model_path> <output>")
        print("   python3 huggingface_integration.py convert <dataset> <output>")
        print("   python3 huggingface_integration.py search <query>")
    else:
        print("❌ SOME TESTS FAILED. Please check the errors above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)