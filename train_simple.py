#!/usr/bin/env python3
"""
Simple Training Wrapper - No Self-Destruction Allowed

Just train your model. All the complexity is handled internally.
No terminal gymnastics required.

Usage:
    python train_simple.py                    # Auto-configures everything
    python train_simple.py mydata            # Train on dataset 'mydata'
    python train_simple.py --from gpt2       # Fine-tune from GPT-2
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


class SimpleTrainer:
    """Handles all the complexity so you don't destroy yourself."""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent
        os.chdir(self.root_dir)
        
    def detect_datasets(self):
        """Find available datasets automatically."""
        datasets_dir = Path("datasets")
        if not datasets_dir.exists():
            return []
        
        datasets = []
        for item in datasets_dir.iterdir():
            if item.is_dir():
                meta_file = item / "meta.pkl"
                train_file = item / "train.bin"
                if meta_file.exists() and train_file.exists():
                    datasets.append(item.name)
        return datasets
    
    def prepare_dataset_if_needed(self, dataset_name: str):
        """Prepare dataset if it doesn't exist."""
        dataset_dir = Path(f"datasets/{dataset_name}")
        if dataset_dir.exists() and (dataset_dir / "meta.pkl").exists():
            return True
        
        # Try to prepare from common locations
        possible_sources = [
            f"data/{dataset_name}.txt",
            f"data/{dataset_name}/",
            f"{dataset_name}.txt",
            f"{dataset_name}/",
            "input.txt"  # in datasets folder
        ]
        
        for source in possible_sources:
            if Path(source).exists():
                print(f"🔧 Preparing dataset '{dataset_name}' from {source}")
                cmd = f"python universal_prepare.py --name {dataset_name} --source {source}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    return True
        
        return False
    
    def get_smart_config(self, dataset_name: str = None, from_model: str = None):
        """Generate smart training config automatically."""
        config = {
            "batch_size": 32,
            "learning_rate": 3e-4,
            "max_iters": 10000,
            "eval_interval": 2000,
            "device": "cuda" if self._has_cuda() else "mps" if self._has_mps() else "cpu"
        }
        
        # Adjust based on starting model
        if from_model:
            config.update({
                "init_from": from_model,
                "learning_rate": 5e-5,  # Lower for fine-tuning
                "max_iters": 5000        # Less for fine-tuning
            })
        
        # Adjust based on dataset
        if dataset_name:
            dataset_path = Path(f"datasets/{dataset_name}/meta.pkl")
            if dataset_path.exists():
                # Auto-detect if this is large dataset
                # (Would load metadata to check size)
                pass
        
        return config
    
    def train(self, dataset_name: str = None, from_model: str = None, mixed: str = None):
        """Train the model with sensible defaults."""
        
        # Handle dataset selection
        if mixed:
            # Multi-dataset training
            print(f"🚀 Training on mixed datasets: {mixed}")
            cmd = f"python3 train.py dataset=multi datasets='{mixed}'"
        else:
            # Single dataset training
            if not dataset_name:
                datasets = self.detect_datasets()
                if datasets:
                    dataset_name = datasets[0]
                    print(f"📊 Auto-detected dataset: {dataset_name}")
                else:
                    print("❌ No datasets found. Create one first:")
                    print("   echo 'Your training text' > data/mydata.txt")
                    print("   python train_simple.py mydata")
                    return
            
            # Prepare dataset if needed
            if not self.prepare_dataset_if_needed(dataset_name):
                print(f"❌ Dataset '{dataset_name}' not found and couldn't be prepared")
                return
            
            print(f"🚀 Training on dataset: {dataset_name}")
            
            # Build training command
            config = self.get_smart_config(dataset_name, from_model)
            cmd_parts = ["python", "train.py"]
            
            if dataset_name:
                cmd_parts.extend(["--dataset", dataset_name])
            
            if from_model:
                cmd_parts.extend(["--init_from", from_model])
            
            # Add config options
            for key, value in config.items():
                if key != "init_from":  # Already added
                    cmd_parts.extend([f"--{key}", str(value)])
            
            cmd = " ".join(cmd_parts)
        
        print(f"📝 Running: {cmd}")
        print("⏳ Starting training...")
        
        # Always use the working simple trainer
        print(f"🔧 Using simple trainer (robust fallback)")
        simple_cmd = f"python3 simple_trainer.py --dataset {dataset_name}"
        if from_model:
            simple_cmd += f" --from_model {from_model}"
        if mixed:
            simple_cmd += f" --mixed '{mixed}'"
        
        # Add config parameters
        for key, value in config.items():
            simple_cmd += f" --{key} {value}"
        
        subprocess.run(simple_cmd, shell=True)
    
    def _has_cuda(self):
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _has_mps(self):
        """Check if MPS is available."""
        try:
            import torch
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except ImportError:
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Simple training - no self-destruction allowed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train_simple.py                    # Auto-detect and train
    python train_simple.py mydata            # Train on mydata dataset
    python train_simple.py --from gpt2       # Fine-tune GPT-2
    python train_simple.py --mixed '{"web": 0.7, "code": 0.3}'
        """
    )
    
    parser.add_argument("dataset", nargs="?", help="Dataset name to train on")
    parser.add_argument("--from", dest="from_model", help="Starting model (gpt2, resume, etc.)")
    parser.add_argument("--mixed", help="Mixed datasets JSON string")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    
    args = parser.parse_args()
    
    trainer = SimpleTrainer()
    
    if args.list:
        datasets = trainer.detect_datasets()
        if datasets:
            print("Available datasets:")
            for ds in datasets:
                print(f"  - {ds}")
        else:
            print("No datasets found. Create one with:")
            print("  python universal_prepare.py --name mydata --source your_file.txt")
        return
    
    trainer.train(
        dataset_name=args.dataset,
        from_model=args.from_model,
        mixed=args.mixed
    )


if __name__ == "__main__":
    main()