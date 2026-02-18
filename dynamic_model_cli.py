#!/usr/bin/env python3
"""
Dynamic Model Selection CLI for SloughGPT

Interactive CLI that allows users to choose training models dynamically
"""

import argparse
import sys
import torch
import json
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


class DynamicModelCLI:
    """Interactive CLI for dynamic model selection."""
    
    def __init__(self):
        self.registry = self._get_model_registry()
        self.models = self._load_models_from_registry()
        self.questions = []
        self.responses = {}
        
    def _get_model_registry(self):
        """Get the model registry."""
        from domains.training.model_registry import registry
        return registry
    
    def _load_models_from_registry(self) -> Dict[str, Dict]:
        """Load models dynamically from registry."""
        models = {}
        
        for model_info in self.registry.list_models():
            models[model_info.id] = {
                'name': model_info.name,
                'type': model_info.id,
                'description': model_info.description,
                'default_params': model_info.default_config,
                'tags': model_info.tags,
            }
        
        return models
    
    def interactive_model_selection(self) -> Dict[str, Any]:
        """Interactive model selection process."""
        
        print("🦁 SloughGPT Dynamic Model Selection CLI")
        print("=" * 50)
        print()
        
        # Group models by tag
        grouped = {}
        for key, model in self.models.items():
            tag = model.get('tags', ['other'])[0] if model.get('tags') else 'other'
            if tag not in grouped:
                grouped[tag] = []
            grouped[tag].append((key, model))
        
        # Show available models by group
        print("📊 Available Models:")
        num = 1
        model_map = {}
        
        for tag, models in sorted(grouped.items()):
            print(f"\n  [{tag.upper()}]")
            for key, model in models:
                print(f"    {num}. {model['name']} ({key})")
                print(f"       {model['description']}")
                model_map[str(num)] = key
                num += 1
        
        print()
        
        # Let user choose model
        while True:
            choice = input(f"Choose model (1-{num-1}), or 'search <query>': ").strip()
            
            if choice.startswith('search '):
                query = choice[7:]
                print(f"\n🔍 Search results for '{query}':")
                for key, model in self.models.items():
                    if (query.lower() in model['name'].lower() or 
                        query.lower() in model['description'].lower() or
                        query.lower() in ' '.join(model.get('tags', []))):
                        print(f"  - {key}: {model['name']}")
                continue
            
            if choice in model_map:
                model_id = model_map[choice]
                return self.models[model_id]
            
            if choice in self.models:
                return self.models[choice]
            
            print(f"✗ Choose 1-{num-1} or model ID")
    
    def interactive_data_selection(self) -> Dict[str, Any]:
        """Interactive data type and source selection."""
        
        print("\n📁 Data Selection")
        print("=" * 30)
        
        # Step 1: Choose data type
        data_types = {
            '1': ('dataset', 'Existing dataset'),
            '2': ('text', 'Plain text'),
            '3': ('text_file', 'Text file/directory'),
            '4': ('jsonl', 'JSON Lines file'),
            '5': ('json', 'JSON file'),
            '6': ('csv', 'CSV file'),
            '7': ('web', 'Web scraping'),
        }
        
        print("Choose data type:")
        for num, (type_, desc) in data_types.items():
            print(f"  {num}: {desc}")
        
        while True:
            data_choice = input("Choose data type (1-7): ").strip()
            if data_choice in data_types:
                data_type, _ = data_types[data_choice]
                break
            print(f"Choose 1-7")
        
        # Step 2: Specify data source
        print(f"\n📂 Specifying {data_type} data source...")
        
        if data_type == 'dataset':
            return self._dataset_source()
        elif data_type == 'text':
            return self._text_source()
        elif data_type in ['text_file', 'jsonl', 'json', 'csv']:
            return self._file_source(data_type)
        elif data_type == 'web':
            return self._web_source()
        
        return {'type': 'text', 'source': 'demo data'}
    
    def _dataset_source(self) -> Dict[str, Any]:
        """Dataset source selection."""
        from domains.training import DatasetRegistry
        
        # Show available datasets
        registry = DatasetRegistry()
        datasets = [d.name for d in registry.list_datasets()]
        
        if datasets:
            print("Available datasets:")
            for i, ds in enumerate(datasets, 1):
                print(f"  {i}: {ds}")
            
            while True:
                choice = input("Choose dataset number, or 'list' again: ").strip()
                if choice == 'list':
                    datasets = [d.name for d in registry.list_datasets()]
                    continue
                if choice.isdigit() and 1 <= int(choice) <= len(datasets):
                    return {'type': 'dataset', 'name': datasets[int(choice)-1]}
                print("Choose a valid dataset number")
        else:
            print("No datasets found. Create one first:")
            print("  python cli.py dataset create mydata 'your text'")
            return {'type': 'text', 'source': 'demo data'}
    
    def _text_source(self) -> Dict[str, Any]:
        """Text source selection."""
        text = input("Enter training text: ").strip()
        if text:
            return {'type': 'text', 'source': text}
        return {'type': 'text', 'source': 'demo data'}
    
    def _file_source(self, file_type: str) -> Dict[str, Any]:
        """File source selection."""
        path = input(f"Enter path to {file_type} file/directory: ").strip()
        
        from domains.training import UniversalDataLoader
        from pathlib import Path
        
        path_obj = Path(path)
        if path_obj.exists():
            return {
                'type': file_type,
                'path': str(path_obj),
                'validated': True
            }
        else:
            print(f"File not found: {path}")
            print("Creating demo dataset instead...")
            return {'type': 'text', 'source': 'demo data'}
    
    def _web_source(self) -> Dict[str, Any]:
        """Web scraping source."""
        url = input("Enter website URL to scrape: ").strip()
        if url:
            return {
                'type': 'web',
                'url': url,
                'max_tokens': 5000
            }
        return {'type': 'text', 'source': 'demo data'}
    
    def interactive_model_config(self, model_type: str) -> Dict[str, Any]:
        """Interactive model configuration."""
        from domains.training import UniversalDataLoader
        
        print(f"\n🔧 Model Configuration: {model_type}")
        print("=" * 30)
        
        # Get model type specific defaults
        base_model = self.models[model_type]
        default_params = base_model.get('default_params', {})
        
        print(f"Default config:")
        for key, value in default_params.items():
            print(f"  {key}: {value}")
        
        # Allow customization
        use_defaults = input("Use default settings? (y/n): ").lower() == 'y'
        
        if use_defaults:
            return default_params
        
        # Interactive config for each parameter
        config = default_params.copy()
        
        param_explanations = {
            'vocab_size': "Vocabulary size - number of unique tokens",
            'n_embed': "Embedding dimension - model width", 
            'n_layer': "Number of layers - model depth",
            'n_head': "Attention heads - parallel processing",
        }
        
        for param, default in default_params.items():
            explanation = param_explanations.get(param, param)
            new_val = input(f"{explanation} (default: {default}): ").strip()
            
            if new_val:
                try:
                    if param == 'lora_rank':
                        config[param] = int(new_val)
                    else:
                        new_val_int = int(new_val)
                        if param.startswith('n_') or param == 'vocab_size':
                            config[param] = new_val_int
                except ValueError:
                    print(f"  Keeping default: {default}")
        
        return config
    
    def run(self) -> None:
        """Run the interactive CLI."""
        try:
            print()
            model_info = self.interactive_model_selection()
            data_info = self.interactive_data_selection()
            config = self.interactive_model_config(model_info['type'])
            
            # Execute training
            print("\n🚀 Starting training...")
            
            # Determine data source
            data_source = None
            if data_info['type'] == 'dataset':
                data_source = f"datasets/{data_info['name']}"
            elif 'path' in data_info:
                data_source = data_info['path']
            else:
                data_source = data_info['source']
            
            print(f"\n📊 Training Summary:")
            print(f"  Model: {model_info['name']} ({model_info['type']})")
            print(f"  Data: {data_info['type']}")
            print(f"  Source: {data_source}")
            print(f"  Config: {json.dumps(config, indent=2)}")
            
            # Create model from registry
            model = self.registry.create_model(model_info['type'], config)
            
            # Use unified training
            from domains.training.unified_training import TrainingConfig, TrainingPipeline
            
            training_config = TrainingConfig(
                data_path=data_source,
                model_id=model_info['type'],
                epochs=config.get('epochs', 3),
                batch_size=config.get('batch_size', 8),
                learning_rate=config.get('learning_rate', 1e-4),
                vocab_size=config.get('vocab_size', 500),
                n_embed=config.get('n_embed', 128),
                n_layer=config.get('n_layer', 3),
                n_head=config.get('n_head', 4),
                output_path=config.get('output_path', 'models/trained_model.pt'),
                max_batches=100,
            )
            
            pipeline = TrainingPipeline(training_config)
            results = pipeline.run()
            
            print("\n✅ Training complete!")
            print(f"   Model: {model_info['name']}")
            print(f"   Parameters: {results['parameters']:,}")
            print(f"   Final loss: {results['final_loss']:.4f}")
            
            # Update RAG with training knowledge
            try:
                from domains.infrastructure import RAGSystem
                rag = RAGSystem()
                rag.add_document(
                    f"Successfully trained {model_info['name']} on {data_info['type']} data. Final loss: {results['final_loss']:.4f}",
                    {
                        'type': 'training_completion',
                        'model': model_info['type'],
                        'data_format': results['data_format'],
                        'loss': results['final_loss'],
                        'timestamp': datetime.datetime.now().isoformat()
                    }
                )
                print("  ✓ Updated RAG knowledge base")
                
            except ImportError:
                pass
                
        except KeyboardInterrupt:
            print("\n👋 Training interrupted")
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    cli = DynamicModelCLI()
    cli.run()
