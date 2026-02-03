#!/usr/bin/env python3
"""
Fixed Dataset Registry and Manager for SloGPT

Centralized dataset management for multi-dataset training.
Provides:
- Dataset registration and discovery
- Unified dataset loading interface
- Multi-dataset mixing and sampling
- Dataset metadata management
- Training configuration generation

Usage:
    python dataset_manager.py register --name mydata --path datasets/mydata
    python dataset_manager.py list
    python dataset_manager.py mix --config datasets.yaml
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import yaml
import numpy as np


class DatasetInfo:
    """Dataset information container."""
    
    def __init__(self, name: str, path: Path, meta: Optional[Dict[str, Any]] = None):
        self.name = name
        self.path = path
        self.meta = meta or {}
        self.vocab_size = self.meta.get('vocab_size', 0) if self.meta else 0
        self.train_tokens = self.meta.get('train_tokens', 0) if self.meta else 0
        self.val_tokens = self.meta.get('val_tokens', 0) if self.meta else 0
        self.total_characters = self.meta.get('total_characters', 0) if self.meta else 0
        self.source_files = self.meta.get('source_files', []) if self.meta else []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'path': str(self.path),
            'vocab_size': self.vocab_size,
            'train_tokens': self.train_tokens,
            'val_tokens': self.val_tokens,
            'total_characters': self.total_characters,
            'source_files': self.source_files,
            'meta': self.meta
        }


class DatasetRegistry:
    """Central registry for all datasets."""
    
    def __init__(self, registry_file: str = "datasets/registry.json"):
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.datasets: Dict[str, DatasetInfo] = {}
        self.load_registry()
    
    def load_registry(self):
        """Load dataset registry from file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    for name, info in data.items():
                        self.datasets[name] = DatasetInfo(
                            name=info['name'],
                            path=Path(info['path']),
                            meta=info.get('meta', {})
                        )
                print(f"Loaded {len(self.datasets)} datasets from registry")
            except Exception as e:
                print(f"Warning: Could not load registry: {e}")
                self.datasets = {}
    
    def save_registry(self):
        """Save dataset registry to file."""
        data = {name: info.to_dict() for name, info in self.datasets.items()}
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(self.datasets)} datasets to registry")
    
    def register_dataset(self, name: str, path: Union[str, Path]) -> DatasetInfo:
        """Register a new dataset."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")
        
        # Load metadata
        meta_file = path / "meta.pkl"
        meta = {}
        if meta_file.exists():
            with open(meta_file, 'rb') as f:
                meta = pickle.load(f)
        
        dataset_info = DatasetInfo(name, path, meta)
        self.datasets[name] = dataset_info
        self.save_registry()
        
        print(f"Registered dataset '{name}' from {path}")
        return dataset_info
    
    def unregister_dataset(self, name: str):
        """Remove a dataset from registry."""
        if name in self.datasets:
            del self.datasets[name]
            self.save_registry()
            print(f"Unregistered dataset '{name}'")
        else:
            print(f"Dataset '{name}' not found in registry")
    
    def get_dataset(self, name: str) -> Optional[DatasetInfo]:
        """Get dataset information."""
        return self.datasets.get(name)
    
    def list_datasets(self) -> List[DatasetInfo]:
        """List all registered datasets."""
        return list(self.datasets.values())
    
    def discover_datasets(self, search_path: str = "datasets"):
        """Automatically discover datasets in search path."""
        search_path = Path(search_path)
        discovered = 0
        
        for dataset_dir in search_path.iterdir():
            if dataset_dir.is_dir():
                meta_file = dataset_dir / "meta.pkl"
                train_file = dataset_dir / "train.bin"
                
                if meta_file.exists() and train_file.exists():
                    name = dataset_dir.name
                    if name not in self.datasets:
                        self.register_dataset(name, dataset_dir)
                        discovered += 1
        
        if discovered > 0:
            print(f"Discovered and registered {discovered} new datasets")
        else:
            print("No new datasets discovered")
        
        return discovered


class MultiDatasetMixer:
    """Handles mixing of multiple datasets for training."""
    
    def __init__(self, registry: DatasetRegistry):
        self.registry = registry
    
    def create_mixed_dataset(self, 
                           mixing_ratios: Dict[str, float],
                           output_name: str,
                           output_dir: Optional[str] = None) -> DatasetInfo:
        """Create a mixed dataset from multiple sources."""
        # Validate datasets exist
        for name in mixing_ratios:
            if name not in self.registry.datasets:
                raise ValueError(f"Dataset '{name}' not found in registry")
        
        # Calculate total tokens for each dataset
        dataset_info = {}
        total_train_tokens = 0
        
        for name, ratio in mixing_ratios.items():
            info = self.registry.get_dataset(name)
            if info is None:
                raise ValueError(f"Dataset '{name}' not found in registry")
            
            dataset_info[name] = {
                'info': info,
                'ratio': ratio,
                'tokens': int(info.train_tokens * ratio)
            }
            total_train_tokens += dataset_info[name]['tokens']
        
        print(f"Creating mixed dataset '{output_name}'")
        print(f"Total train tokens: {total_train_tokens:,}")
        
        # Create output directory
        output_dir = Path(output_dir or f"datasets/{output_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and mix datasets
        mixed_train = []
        mixed_val = []
        
        for name, data in dataset_info.items():
            info = data['info']
            print(f"Loading {name}: {data['tokens']:,} tokens")
            
            # Load train data
            train_path = info.path / "train.bin"
            train_data = np.fromfile(train_path, dtype=np.uint16)
            
            # Sample required tokens
            if len(train_data) > data['tokens']:
                indices = np.random.choice(len(train_data), data['tokens'], replace=False)
                train_data = train_data[indices]
            
            mixed_train.append(train_data)
            
            # Load validation data (proportionally)
            val_path = info.path / "val.bin"
            val_data = np.fromfile(val_path, dtype=np.uint16)
            val_tokens = int(len(val_data) * data['ratio'])
            if len(val_data) > val_tokens:
                indices = np.random.choice(len(val_data), val_tokens, replace=False)
                val_data = val_data[indices]
            mixed_val.append(val_data)
        
        # Concatenate all data
        final_train = np.concatenate(mixed_train)
        final_val = np.concatenate(mixed_val)
        
        # Save mixed dataset
        final_train.tofile(output_dir / "train.bin")
        final_val.tofile(output_dir / "val.bin")
        
        # Create combined metadata
        combined_meta = {
            'vocab_size': max(data['info'].vocab_size for data in dataset_info.values()),
            'mixed_datasets': mixing_ratios,
            'source_datasets': [name for name in mixing_ratios],
            'total_train_tokens': len(final_train),
            'total_val_tokens': len(final_val),
            'creation_method': 'mixed'
        }
        
        with open(output_dir / "meta.pkl", "wb") as f:
            pickle.dump(combined_meta, f)
        
        # Register the mixed dataset
        mixed_info = self.registry.register_dataset(output_name, output_dir)
        
        print(f"Mixed dataset created: {output_name}")
        print(f"  Train tokens: {len(final_train):,}")
        print(f"  Val tokens: {len(final_val):,}")
        print(f"  Source datasets: {list(mixing_ratios.keys())}")
        
        return mixed_info
    
    def generate_training_config(self, 
                                mixing_ratios: Dict[str, float],
                                output_file: str = "multi_dataset_config.json") -> str:
        """Generate training configuration for multi-dataset training."""
        config = {
            'dataset': 'multi',
            'datasets': mixing_ratios,
            'datasets_dir': 'datasets',
            'training': {
                'batch_size': 32,
                'learning_rate': 3e-4,
                'max_iters': 10000
            }
        }
        
        config_path = Path(output_file)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Training configuration saved to: {config_path}")
        print(f"Use: python train.py config={config_path}")
        
        return str(config_path)


class DatasetManager:
    """Main dataset management interface."""
    
    def __init__(self):
        self.registry = DatasetRegistry()
        self.mixer = MultiDatasetMixer(self.registry)
    
    def register(self, name: str, path: str):
        """Register a dataset."""
        return self.registry.register_dataset(name, path)
    
    def list(self):
        """List all datasets."""
        datasets = self.registry.list_datasets()
        if not datasets:
            print("No datasets registered")
            return
        
        print(f"Registered datasets ({len(datasets)}):")
        print("-" * 80)
        for info in datasets:
            print(f"Name: {info.name}")
            print(f"Path: {info.path}")
            print(f"Vocab size: {info.vocab_size:,}")
            print(f"Train tokens: {info.train_tokens:,}")
            print(f"Val tokens: {info.val_tokens:,}")
            print(f"Source files: {len(info.source_files)}")
            print("-" * 80)
    
    def discover(self):
        """Discover new datasets."""
        return self.registry.discover_datasets()
    
    def mix(self, config_file: Optional[str] = None, ratios: Optional[str] = None, output: str = "mixed_dataset"):
        """Create mixed dataset."""
        if config_file:
            # Load from config file
            with open(config_file, 'r') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            mixing_ratios = config.get('training', {}).get('mixing_ratios', {})
            if not mixing_ratios:
                # Use individual dataset weights
                mixing_ratios = {d['name']: d.get('weight', 1.0) for d in config.get('datasets', [])}
        elif ratios:
            # Parse ratios string format: "dataset1:0.5,dataset2:0.3"
            mixing_ratios = {}
            for pair in ratios.split(','):
                name, ratio = pair.split(':')
                mixing_ratios[name.strip()] = float(ratio.strip())
        else:
            print("Error: Either --config or --ratios required")
            return None
        
        return self.mixer.create_mixed_dataset(mixing_ratios, output)
    
    def generate_config(self, ratios: str, output: str = "multi_config.json"):
        """Generate training configuration."""
        mixing_ratios = {}
        for pair in ratios.split(','):
            name, ratio = pair.split(':')
            mixing_ratios[name.strip()] = float(ratio.strip())
        
        return self.mixer.generate_training_config(mixing_ratios, output)


def main():
    parser = argparse.ArgumentParser(description="Dataset manager for SloGPT")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Register command
    register_parser = subparsers.add_parser('register', help='Register a dataset')
    register_parser.add_argument('--name', required=True, help='Dataset name')
    register_parser.add_argument('--path', required=True, help='Dataset path')
    
    # List command
    subparsers.add_parser('list', help='List all datasets')
    
    # Discover command
    discover_parser = subparsers.add_parser('discover', help='Discover datasets')
    discover_parser.add_argument('--path', default='datasets', help='Search path')
    
    # Mix command
    mix_parser = subparsers.add_parser('mix', help='Create mixed dataset')
    mix_parser.add_argument('--config', help='Configuration file')
    mix_parser.add_argument('--ratios', help='Mixing ratios (name:ratio,name:ratio)')
    mix_parser.add_argument('--output', default='mixed_dataset', help='Output dataset name')
    
    # Generate config command
    config_parser = subparsers.add_parser('generate-config', help='Generate training config')
    config_parser.add_argument('--ratios', required=True, help='Mixing ratios (name:ratio,name:ratio)')
    config_parser.add_argument('--output', default='multi_config.json', help='Output config file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = DatasetManager()
    
    if args.command == 'register':
        manager.register(args.name, args.path)
    elif args.command == 'list':
        manager.list()
    elif args.command == 'discover':
        manager.discover()
    elif args.command == 'mix':
        manager.mix(args.config, args.ratios, args.output)
    elif args.command == 'generate-config':
        manager.generate_config(args.ratios, args.output)


if __name__ == "__main__":
    main()