#!/usr/bin/env python3
"""
SloGPT System Integration Demo

Demonstrates all components working together in a complete workflow.
"""

import subprocess
import sys
import time
from pathlib import Path


def create_demo_dataset():
    """Create a demonstration dataset."""
    print("📝 Creating demo dataset...")
    
    demo_text = """
    SloGPT is a modular transformer training framework with advanced dataset management.
    
    This is demonstration data for the quality scoring system.
    The model can learn simple patterns and generate text continuations.
    
    Features demonstrated:
    - Universal dataset creation from any source
    - Quality scoring and validation
    - Performance monitoring
    - Web interface for management
    - Distributed training support
    - CLI shortcuts for frictionless usage
    
    The system eliminates terminal gymnastics while providing enterprise-level
    capabilities for dataset management and model training.
    """
    
    cmd = "python3 create_dataset_fixed.py demo_data \"SloGPT demo dataset with quality scoring demo\""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Demo dataset created successfully")
        return True
    else:
        print(f"❌ Demo dataset creation failed: {result.stderr}")
        return False


def demonstrate_quality_scoring():
    """Demonstrate dataset quality scoring."""
    print("🔍 Demonstrating quality scoring...")
    
    cmd = "python3 quality_scorer.py demo_data --output demo_quality_report.json"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Quality scoring completed")
        return True
    else:
        print(f"❌ Quality scoring failed: {result.stderr}")
        return False


def start_web_interface():
    """Start web interface."""
    print("🌐 Starting web interface...")
    
    cmd = "python3 web_interface.py"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Web interface started successfully")
        print("🚀 Open http://localhost:5000 in your browser")
        return True
    else:
        print(f"❌ Web interface failed: {result.stderr}")
        return False


def start_analytics_dashboard():
    """Start analytics dashboard."""
    print("📊 Starting analytics dashboard...")
    
    cmd = "python3 anality_dashboard.py"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Analytics dashboard started successfully")
        print("🚀 Open http://localhost:5001 in your browser")
        return True
    else:
        print(f"❌ Analytics dashboard failed: {result.stderr}")
        return False


def demonstrate_huggingface_integration():
    """Demonstrate Hugging Face integration."""
    print("🤖 Demonstrating Hugging Face integration...")
    
    cmd = "python3 huggingface_integration.py search --query 'text generation' --limit 3"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Hugging Face search completed")
        return True
    else:
        print(f"❌ Hugging Face search failed: {result.stderr}")
        return False


def start_simple_training():
    """Start simple training workflow."""
    print("🚀 Starting simple training workflow...")
    
    # Create dataset
    if not create_demo_dataset():
        print("❌ Cannot start training without dataset")
        return False
    
    # Start training
    cmd = "python3 train_simple.py demo_data --batch_size 16 --max_iters 100"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Simple training started successfully")
        return True
    else:
        print(f"❌ Simple training failed: {result.stderr}")
        return False


def run_performance_monitoring():
    """Run performance monitoring."""
    print("📈 Starting performance monitoring...")
    
    cmd = "python3 performance_optimizer.py monitor --interval 15"
    
    try:
        subprocess.run(cmd, shell=True, timeout=30)  # Run for 30 seconds
        print("✅ Performance monitoring completed (30s demo)")
        return True
    except subprocess.TimeoutExpired:
        print("⏰ Performance monitoring demo completed (30s timeout)")
        return True
    except Exception as e:
        print(f"❌ Performance monitoring failed: {e}")
        return False


def demonstrate_cli_shortcuts():
    """Demonstrate CLI shortcuts."""
    print("⚡ Demonstrating CLI shortcuts...")
    
    print("📋 Available shortcuts:")
    print("  slo new <name> <text>     - Create dataset")
    print("  slo train <name>            - Train model")
    print("  slo list                  - List datasets")
    print("  slo validate <name>          - Validate dataset")
    print("  slo monitor               - Start monitoring")
    print("  slo help                  - Show help")
    
    print("\n🎯 Testing CLI shortcuts...")
    
    # Test dataset creation
    cmd = "python3 train_simple.py --help"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    print(f"CLI help output:")
    if result.returncode == 0:
        print(result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
        return True
    
    return True


def run_comprehensive_test():
    """Run comprehensive system test."""
    print("🧪 Running comprehensive system test...")
    
    cmd = "python3 final_system_test.py"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Comprehensive test completed successfully")
        return True
    else:
            print(f"❌ Comprehensive test failed: {result.stderr}")
            return False


def main():
    """Main demonstration script."""
    print("🚀 SloGPT System Integration Demo")
    print("=" * 60)
    
    print("""
This demo will showcase all components working together:
    1. 📝 Dataset creation and management
    2. 🔍 Quality scoring and validation
    3. 🌐 Web interface
    4. 📊 Analytics dashboard  
    5. 🤖 Hugging Face integration
    6. 🚀 Simple training workflow
    7. ⚡ CLI shortcuts
    8. 📈 Performance monitoring
    9. 🧪 Comprehensive system test

Each component will be demonstrated with real usage examples.
    """)
    
    demonstrations = [
        ("Dataset Creation", create_demo_dataset),
        ("Quality Scoring", demonstrate_quality_scoring),
        ("Web Interface", start_web_interface),
        ("Analytics Dashboard", start_analytics_dashboard),
        ("Hugging Face", demonstrate_huggingface_integration),
        ("Simple Training", start_simple_training),
        ("CLI Shortcuts", demonstrate_cli_shortcuts),
        ("Performance Monitor", run_performance_monitoring),
        ("Comprehensive Test", run_comprehensive_test)
    ]
    
    print(f"🎯 Available Demonstrations:")
    for i, (name, func) in enumerate(demonstrations, 1):
        print(f"  {i}. {name}")
    
    print(f"\n🚀 Select a demonstration (1-{len(demonstrations)}):")
    
    try:
        choice = input("Enter choice (number): ").strip()
        choice = int(choice)
        
        if 1 <= choice <= len(demonstrations):
            name, func = demonstrations[choice - 1]
            print(f"\n🚀 Running demonstration: {name}")
            
            success = func()
            
            if success:
                print(f"\n✅ {name} demonstration completed successfully")
            else:
                print(f"\n❌ {name} demonstration failed")
            
        else:
            print(f"❌ Invalid choice. Please select 1-{len(demonstrations)}")
    
    except KeyboardInterrupt:
        print(f"\n🛑 Demo interrupted")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")


if __name__ == '__main__':
    main()