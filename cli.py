"""
SloughGPT CLI
Command-line interface for SloughGPT
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path


def cmd_chat(args):
    """Start an interactive chat session."""
    import requests
    
    base_url = f"http://{args.host}:{args.port}"
    
    print(f"SloughGPT Chat ({base_url})")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        try:
            response = requests.post(
                f"{base_url}/infer",
                json={
                    "prompt": user_input,
                    "max_length": args.max_tokens,
                    "temperature": args.temperature,
                    "model": args.model
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"SloughGPT: {data['text']}\n")
            else:
                print(f"Error: {response.text}\n")
                
        except Exception as e:
            print(f"Error: {e}\n")


def cmd_models(args):
    """List available models."""
    import requests
    
    base_url = f"http://{args.host}:{args.port}"
    
    try:
        response = requests.get(f"{base_url}/models")
        if response.status_code == 200:
            data = response.json()
            print("Available Models:")
            for model in data.get('models', []):
                print(f"  - {model['id']}: {model['name']}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")


def cmd_train(args):
    """Start a training job."""
    if not args.api:
        import torch
        import sys
        sys.path.insert(0, '.')
        
        # Load config
        from config_loader import load_config, merge_args_with_config
        config = load_config(args.config)
        config = merge_args_with_config(config, args)
        
        # Setup tracking
        tracker = None
        if config.tracking.enabled:
            from domains.training.tracking import ExperimentTracker, TrackerBackend, TrackingConfig
            
            backend = TrackerBackend.WANDB if config.tracking.backend == "wandb" else TrackerBackend.MLFLOW
            tracking_config = TrackingConfig(
                backend=backend,
                experiment_name=f"{config.model.name}_training",
                project=config.tracking.project,
                entity=config.tracking.entity,
            )
            tracker = ExperimentTracker(config=tracking_config)
            tracker.start_run(run_name=f"run_{args.dataset}_{args.epochs}ep")
            tracker.log_config({
                "model": config.model.__dict__,
                "training": config.training.__dict__,
                "lora": config.lora.__dict__,
            })
            print(f"Tracking enabled: {config.tracking.backend}")
        
        # Use train_pipeline for full-featured training
        from domains.training.train_pipeline import SloughGPTTrainer
        
        print("=" * 50)
        print("SLOUGHGPT TRAINING")
        print("=" * 50)
        print(f"Dataset: {config.data.dataset}")
        print(f"Epochs: {config.training.epochs}")
        print(f"Batch: {config.training.batch_size}")
        print(f"LR: {config.training.learning_rate}")
        print(f"LoRA: {config.lora.enabled}")
        print(f"Tracking: {config.tracking.enabled}")
        print("=" * 50)
        
        trainer = SloughGPTTrainer(
            data_path=config.data.data_path,
            vocab_size=config.model.vocab_size,
            n_embed=config.model.n_embed,
            n_layer=config.model.n_layer,
            n_head=config.model.n_head,
            block_size=config.model.block_size,
            use_lora=config.lora.enabled,
            lora_rank=config.lora.rank,
            lora_alpha=config.lora.alpha,
            batch_size=config.training.batch_size,
            epochs=config.training.epochs,
            lr=config.training.learning_rate,
        )
        
        # Train
        model = trainer.train()
        
        # Save
        save_path = f"{config.checkpoint.save_dir}/{config.model.name}.pt"
        trainer.save(save_path)
        print(f"\nModel saved to {save_path}")
        
        if tracker:
            tracker.end_run()
        
        print("Training complete!")
        return
    
    import requests
    
    base_url = f"http://{args.host}:{args.port}"
    
    try:
        response = requests.post(
            f"{base_url}/training/start",
            params={
                "dataset": args.dataset,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "use_lora": args.use_lora
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Training started: {data['job_id']}")
            print(f"Status: {data['status']}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")


def cmd_status(args):
    """Get system status."""
    import requests
    
    base_url = f"http://{args.host}:{args.port}"
    
    try:
        response = requests.get(f"{base_url}/status")
        if response.status_code == 200:
            data = response.json()
            print(json.dumps(data, indent=2))
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")


def cmd_health(args):
    """Check API health."""
    import requests
    
    base_url = f"http://{args.host}:{args.port}"
    
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"Status: {data['status']}")
            print(f"Version: {data.get('version', 'unknown')}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")


def cmd_generate(args):
    """Generate text from prompt."""
    import requests
    
    base_url = f"http://{args.host}:{args.port}"
    
    try:
        response = requests.post(
            f"{base_url}/infer",
            json={
                "prompt": args.prompt,
                "max_length": args.max_tokens,
                "temperature": args.temperature,
                "model": args.model or "sloughgpt"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(data['text'])
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")


def cmd_datasets(args):
    """List datasets."""
    import requests
    
    base_url = f"http://{args.host}:{args.port}"
    
    try:
        response = requests.get(f"{base_url}/datasets")
        if response.status_code == 200:
            data = response.json()
            print("Datasets:")
            for ds in data.get('datasets', []):
                print(f"  - {ds['name']}: {ds.get('size', 0)} bytes")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")


def cmd_export(args):
    """Export a model."""
    import requests
    
    base_url = f"http://{args.host}:{args.port}"
    
    try:
        response = requests.get(f"{base_url}/export/model/{args.model}")
        if response.status_code == 200:
            data = response.json()
            print(f"Model: {data['model']}")
            print(f"Format: {data['format']}")
            print(f"Size: {data['size_bytes']} bytes")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SloughGPT CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--host", default="localhost", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument("-c", "--config", default="config.yaml", help="Config file")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")
    chat_parser.add_argument("--model", default="sloughgpt", help="Model to use")
    chat_parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens")
    chat_parser.add_argument("--temperature", type=float, default=0.8, help="Temperature")
    chat_parser.set_defaults(func=cmd_chat)
    
    # Models command
    models_parser = subparsers.add_parser("models", help="List models")
    models_parser.set_defaults(func=cmd_models)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Start training")
    train_parser.add_argument("--dataset", default="shakespeare", help="Dataset")
    train_parser.add_argument("--epochs", type=int, default=3, help="Epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    train_parser.add_argument("--use-lora", action="store_true", help="Use LoRA")
    train_parser.add_argument("--api", action="store_true", help="Use API server instead of local")
    train_parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    train_parser.add_argument("--max-steps", type=int, default=None, help="Max steps per epoch (for quick testing)")
    train_parser.set_defaults(func=cmd_train)
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Get system status")
    status_parser.set_defaults(func=cmd_status)
    
    # Health command
    health_parser = subparsers.add_parser("health", help="Check API health")
    health_parser.set_defaults(func=cmd_health)
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text")
    gen_parser.add_argument("prompt", help="Prompt text")
    gen_parser.add_argument("--model", help="Model name")
    gen_parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens")
    gen_parser.add_argument("--temperature", type=float, default=0.8, help="Temperature")
    gen_parser.set_defaults(func=cmd_generate)
    
    # Datasets command
    datasets_parser = subparsers.add_parser("datasets", help="List datasets")
    datasets_parser.set_defaults(func=cmd_datasets)
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export model")
    export_parser.add_argument("model", help="Model name")
    export_parser.set_defaults(func=cmd_export)
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
