#!/usr/bin/env python3
"""
Quick Start Guide for Standardized Multi-Dataset Training

This script demonstrates how to use the new dataset standardization system.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and show results."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"📝 Command: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Demonstrate the dataset standardization workflow."""
    
    print("🚀 SloGPT Dataset Standardization Demo")
    print("This guide shows how to prepare and train on multiple datasets easily!")
    
    # Step 1: Discover existing datasets
    print(f"\n{'='*60}")
    print("📊 STEP 1: Discover existing datasets")
    print('='*60)
    
    run_command("python dataset_manager.py discover", "Discovering datasets in datasets/ folder")
    run_command("python dataset_manager.py list", "Listing all registered datasets")
    
    # Step 2: Create sample datasets for demonstration
    print(f"\n{'='*60}")
    print("📝 STEP 2: Creating sample datasets")
    print('='*60)
    
    # Create sample data directories
    Path("demo_data").mkdir(exist_ok=True)
    
    # Sample code data
    with open("demo_data/sample_code.py", "w") as f:
        f.write("""
# Sample Python code for dataset
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class DataProcessor:
    def __init__(self, name):
        self.name = name
    
    def process(self, data):
        return [x * 2 for x in data]
""")
    
    # Sample text data
    with open("demo_data/sample_text.txt", "w") as f:
        f.write("""
Artificial Intelligence and Machine Learning

Machine learning is a subset of artificial intelligence that enables computers
to learn and improve from experience without being explicitly programmed.
It focuses on developing computer programs that can access data and use it
to learn for themselves.

Deep learning is a type of machine learning that trains a computer to perform
human-like tasks, such as identifying images and sounds.
Natural language processing helps computers understand human language.
""")
    
    # Step 3: Prepare datasets using universal preparer
    print(f"\n{'='*60}")
    print("⚙️  STEP 3: Prepare datasets with universal preparer")
    print('='*60)
    
    run_command(
        "python universal_prepare.py --name demo_code --source demo_data/sample_code.py",
        "Preparing code dataset"
    )
    
    run_command(
        "python universal_prepare.py --name demo_text --source demo_data/sample_text.txt", 
        "Preparing text dataset"
    )
    
    # Step 4: Register datasets
    print(f"\n{'='*60}")
    print("📋 STEP 4: Register datasets")
    print('='*60)
    
    run_command(
        "python dataset_manager.py register --name demo_code --path datasets/demo_code",
        "Registering code dataset"
    )
    
    run_command(
        "python dataset_manager.py register --name demo_text --path datasets/demo_text", 
        "Registering text dataset"
    )
    
    # Step 5: List all datasets
    run_command("python dataset_manager.py list", "Listing registered datasets")
    
    # Step 6: Create mixed dataset
    print(f"\n{'='*60}")
    print("🔀 STEP 6: Create mixed dataset")
    print('='*60)
    
    run_command(
        "python dataset_manager.py mix --ratios demo_code:0.6,demo_text:0.4 --output demo_mixed",
        "Creating mixed dataset (60% code, 40% text)"
    )
    
    # Step 7: Generate training configuration
    print(f"\n{'='*60}")
    print("⚙️  STEP 7: Generate training configuration")
    print('='*60)
    
    run_command(
        "python dataset_manager.py generate-config --ratios demo_code:0.6,demo_text:0.4 --output demo_training_config.json",
        "Generating training configuration"
    )
    
    # Step 8: Show training commands
    print(f"\n{'='*60}")
    print("🚀 STEP 8: Training commands")
    print('='*60)
    
    print("✅ You can now train on your datasets using:")
    print()
    print("1️⃣  Train on individual dataset:")
    print("   python train.py --dataset=demo_code")
    print("   python train.py --dataset=demo_text")
    print("   python train.py --dataset=demo_mixed")
    print()
    print("2️⃣  Train on multiple datasets:")
    print("   python train.py dataset=multi datasets='{\"demo_code\": 0.6, \"demo_text\": 0.4}'")
    print()
    print("3️⃣  Train using configuration file:")
    print("   python train.py config=demo_training_config.json")
    print()
    
    # Step 9: Advanced usage examples
    print(f"\n{'='*60}")
    print("🎓 ADVANCED USAGE EXAMPLES")
    print('='*60)
    
    print("📁 Process entire directories:")
    print("   python universal_prepare.py --name myproject --source ./src --recursive")
    print()
    print("📄 Use configuration file for batch preparation:")
    print("   python universal_prepare.py --config datasets.yaml")
    print()
    print("🔀 Create complex mixed datasets:")
    print("   python dataset_manager.py mix --config datasets.yaml --output web_code_mixed")
    print()
    print("📊 Dataset management:")
    print("   python dataset_manager.py list")
    print("   python dataset_manager.py discover --path ./custom_datasets")
    print("   python dataset_manager.py unregister --name old_dataset")
    print()
    
    print("🎉 Dataset standardization setup complete!")
    print("You now have a unified system for managing and training on multiple datasets.")


if __name__ == "__main__":
    main()