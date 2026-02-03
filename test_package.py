#!/usr/bin/env python3
"""
Test script for SloughGPT package structure
"""

import sys
import os
import subprocess

def test_package_imports():
    """Test package import functionality"""
    print("🧪 Testing SloughGPT Package Structure")
    print("=" * 50)
    
    # Install the package in development mode
    print("📦 Installing package in development mode...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", "."
        ], capture_output=True, text=True, cwd="/Users/mac/sloughGPT")
        
        if result.returncode == 0:
            print("✅ Package installation successful")
        else:
            print(f"❌ Package installation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False
    
    # Test imports
    print("\n🔍 Testing module imports...")
    
    test_cases = [
        ("sloughgpt.config", "ModelConfig"),
        ("sloughgpt.neural_network", "SloughGPT"), 
        ("sloughgpt.core.exceptions", "SloughGPTError"),
    ]
    
    success_count = 0
    for module_name, class_name in test_cases:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✅ {module_name}.{class_name} import successful")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module_name}.{class_name} import failed: {e}")
        except AttributeError as e:
            print(f"❌ {module_name}.{class_name} not found: {e}")
    
    # Test functionality
    print("\n⚡ Testing functionality...")
    try:
        from sloughgpt.config import ModelConfig
        from sloughgpt.neural_network import SloughGPT
        
        # Test configuration
        config = ModelConfig()
        print("✅ Configuration creation successful")
        
        # Test model creation  
        model = SloughGPT(config)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Neural network creation successful")
        print(f"   Parameters: {param_count:,}")
        print(f"   Device: {model.device}")
        
        success_count += 1
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    total_tests = len(test_cases) + 1  # +1 for functionality test
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