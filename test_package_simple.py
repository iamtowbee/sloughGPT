#!/usr/bin/env python3
"""
Simple test script for SloughGPT package structure without pip install
"""

import sys
import os

def test_package_imports():
    """Test package import functionality directly"""
    print("🧪 Testing SloughGPT Package Structure")
    print("=" * 50)
    
    # Add current directory to Python path
    current_dir = "/Users/mac/sloughGPT"
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Test imports
    print("\n🔍 Testing module imports...")
    
    success_count = 0
    total_tests = 0
    
    # Test configuration
    total_tests += 1
    try:
        from sloughgpt.config import ModelConfig
        config = ModelConfig()
        print("✅ sloughgpt.config import successful")
        success_count += 1
    except ImportError as e:
        print(f"❌ sloughgpt.config import failed: {e}")
    except Exception as e:
        print(f"❌ sloughgpt.config creation failed: {e}")
    
    # Test neural network
    total_tests += 1
    try:
        from sloughgpt.neural_network import SloughGPT
        from sloughgpt.config import ModelConfig
        config = ModelConfig()
        model = SloughGPT(config)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ sloughgpt.neural_network import successful")
        print(f"   Parameters: {param_count:,}")
        print(f"   Device: {model.device}")
        success_count += 1
    except ImportError as e:
        print(f"❌ sloughgpt.neural_network import failed: {e}")
    except Exception as e:
        print(f"❌ sloughgpt.neural_network creation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test exceptions
    total_tests += 1
    try:
        from sloughgpt.core.exceptions import SloughGPTError, create_error
        error = create_error(SloughGPTError, "Test error message")
        print("✅ sloughgpt.core.exceptions import successful")
        success_count += 1
    except ImportError as e:
        print(f"❌ sloughgpt.core.exceptions import failed: {e}")
    except Exception as e:
        print(f"❌ sloughgpt.core.exceptions creation failed: {e}")
    
    # Test file structure
    print("\n📁 Checking package structure...")
    structure_tests = [
        "sloughgpt/__init__.py",
        "sloughgpt/config.py", 
        "sloughgpt/neural_network.py",
        "sloughgpt/core/__init__.py",
        "sloughgpt/core/config.py",
        "sloughgpt/core/exceptions.py",
    ]
    
    structure_success = 0
    for file_path in structure_tests:
        total_tests += 1
        full_path = f"/Users/mac/sloughGPT/{file_path}"
        if os.path.exists(full_path):
            print(f"✅ {file_path} exists")
            structure_success += 1
        else:
            print(f"❌ {file_path} missing")
    
    success_count += structure_success
    
    # Summary
    print(f"\n📊 Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 SloughGPT package structure is ready!")
        return True
    else:
        print("❌ Package structure needs fixes")
        return False

if __name__ == "__main__":
    success = test_package_imports()
    sys.exit(0 if success else 1)