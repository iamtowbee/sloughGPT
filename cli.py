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
    """List available models - local version."""
    print("=" * 50)
    print("Available Models")
    print("=" * 50)
    print("  nanogpt: NanoGPT - Custom GPT model")
    print("  gpt2: GPT-2 - HuggingFace model")
    print("  llama: LLaMA - Meta model")
    print("\nNote: Use 'quick' command to train a custom model")


def cmd_quick(args):
    """Quick training and generation - no API needed."""
    import sys
    sys.path.insert(0, '.')
    
    from domains.training.train_pipeline import SloughGPTTrainer
    
    print("=" * 50)
    print("SloughGPT Quick Start")
    print("=" * 50)
    
    # Create trainer
    trainer = SloughGPTTrainer(
        data_path=args.dataset,
        n_embed=args.embed,
        n_layer=args.layers,
        n_head=args.heads,
        block_size=args.block,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        max_steps=args.steps,
    )
    
    # Train
    print("\nTraining...")
    trainer.train()
    
    # Generate
    print("\nGenerating text...")
    text = trainer.generate(args.prompt, max_tokens=args.max_tokens, temperature=args.temperature)
    print(f"\nPrompt: {args.prompt}")
    print(f"Generated: {text[:200]}...")
    
    # Save
    trainer.save(args.output)
    print(f"\nModel saved to {args.output}")


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
            tracker.log_params({
                "model": str(config.model.__dict__),
                "training": str(config.training.__dict__),
                "lora": str(config.lora.__dict__),
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
    """Get system status - local version."""
    from pathlib import Path
    
    print("=" * 50)
    print("SloughGPT System Status")
    print("=" * 50)
    
    # Check models
    models_dir = Path("models")
    if models_dir.exists():
        models = list(models_dir.rglob("*.pt")) + list(models_dir.rglob("*.pth"))
        print(f"\nModels: {len(models)} found")
        for m in models[:5]:
            print(f"  - {m}")
    else:
        print("\nModels: directory not found")
    
    # Check datasets
    datasets_dir = Path("datasets")
    if datasets_dir.exists():
        datasets = list(datasets_dir.iterdir())
        print(f"\nDatasets: {len(datasets)} found")
        for d in datasets[:5]:
            print(f"  - {d.name}")
    else:
        print("\nDatasets: directory not found")
    
    print("\n" + "=" * 50)
    print("Run 'python3 cli.py quick --help' to train a model")
    print("=" * 50)


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


def cmd_info(args):
    """Show model checkpoint info."""
    import torch
    from pathlib import Path
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return
    
    print("=" * 50)
    print(f"Model: {model_path}")
    print("=" * 50)
    
    checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
    
    if 'model' in checkpoint:
        model = checkpoint['model']
        if hasattr(model, 'state_dict'):
            state = model.state_dict()
            print(f"\nState dict keys: {len(state)}")
            total_params = sum(p.numel() for p in state.values() if isinstance(p, torch.Tensor))
            print(f"Total parameters: {total_params:,}")
        elif isinstance(model, dict):
            print(f"Model dict keys: {len(model)}")
            total_params = sum(p.numel() for p in model.values() if isinstance(p, torch.Tensor))
            print(f"Total parameters: {total_params:,}")
    
    if 'chars' in checkpoint:
        print(f"Vocab size: {len(checkpoint['chars'])}")
    if 'stoi' in checkpoint:
        print(f"Char-to-int map size: {len(checkpoint['stoi'])}")
    if 'itos' in checkpoint:
        print(f"Int-to-char map size: {len(checkpoint['itos'])}")
    if 'training_info' in checkpoint:
        print(f"Training info: {checkpoint['training_info']}")
    
    print("=" * 50)


def cmd_serve(args):
    """Start a simple HTTP inference server."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json
    import torch
    
    model = None
    stoi = {}
    itos = {}
    
    # Try to load model
    model_path = args.model or "models/sloughgpt.pt"
    if Path(model_path).exists():
        print(f"Loading model from {model_path}...")
        try:
            checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
            stoi = checkpoint.get('stoi', {})
            itos = checkpoint.get('itos', {})
            print("Model loaded!")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
    
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/health':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "model": "sloughgpt"}).encode())
            else:
                self.send_response(404)
                self.end_headers()
        
        def do_POST(self):
            if self.path == '/generate':
                content_length = int(self.headers['Content-Length'])
                body = self.rfile.read(content_length)
                data = json.loads(body)
                
                prompt = data.get('prompt', '')
                max_tokens = data.get('max_tokens', 100)
                temperature = data.get('temperature', 0.8)
                
                # Simple generation (placeholder)
                text = f"Generated: {prompt[:50]}... (model not fully loaded)"
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"text": text}).encode())
            else:
                self.send_response(404)
                self.end_headers()
        
        def log_message(self, format, *args):
            print(f"[Server] {args[0]}")
    
    print("=" * 50)
    print(f"Starting server on {args.host}:{args.port}")
    print("=" * 50)
    
    server = HTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
        server.shutdown()


def cmd_generate(args):
    """Generate text from prompt - tries local first, then API."""
    import torch
    from pathlib import Path
    from domains.training.models.nanogpt import NanoGPT
    
    # Try local first
    model_path = Path("models/sloughgpt.pt")
    if model_path.exists():
        try:
            checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
            
            # Get config from checkpoint
            training_info = checkpoint.get('training_info', {})
            vocab_size = training_info.get('vocab_size', 65)
            n_embed = training_info.get('n_embed', 128)
            n_layer = training_info.get('n_layer', 4)
            n_head = training_info.get('n_head', 4)
            block_size = training_info.get('block_size', 64)
            
            # Create model and load weights
            model = NanoGPT(vocab_size=vocab_size, n_embed=n_embed, n_layer=n_layer, n_head=n_head, block_size=block_size)
            
            # Load state dict if present
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)
            
            stoi = checkpoint.get('stoi', {})
            itos = checkpoint.get('itos', {})
            
            idx = torch.tensor([[stoi.get(c, 0) for c in args.prompt]], dtype=torch.long)
            
            model.eval()
            with torch.no_grad():
                for _ in range(args.max_tokens):
                    idx_cond = idx[:, -128:]
                    logits, _ = model(idx_cond)
                    logits = logits[:, -1, :] / args.temperature
                    probs = torch.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    idx = torch.cat([idx, idx_next], dim=1)
            
            generated = ''.join([itos.get(i, '') for i in idx[0].tolist()])
            result = generated[len(args.prompt):]
            print(f"Generated: {result[:500]}")
            return
        except Exception as e:
            print(f"Local generation failed: {e}")
    
    # Fall back to API
    import requests
    
    base_url = f"http://{args.host}:{args.port}"
    
    try:
        response = requests.post(
            f"{base_url}/generate",
            json={
                "prompt": args.prompt,
                "max_new_tokens": args.max_tokens,
                "temperature": args.temperature,
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Generated: {data['text'][:500]}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")


def cmd_datasets(args):
    """List datasets - local version."""
    from pathlib import Path
    
    datasets_dir = Path("datasets")
    
    if not datasets_dir.exists():
        print("No datasets directory found")
        return
    
    print("=" * 50)
    print("Datasets")
    print("=" * 50)
    
    for ds in sorted(datasets_dir.iterdir()):
        if ds.is_dir():
            size = sum(f.stat().st_size for f in ds.rglob("*") if f.is_file())
            print(f"  {ds.name}: {size / 1024:.1f} KB")
        else:
            print(f"  {ds.name}: {ds.stat().st_size} bytes")


def cmd_personalities(args):
    """List available personalities."""
    from domains.ai_personality import PERSONALITIES, PersonalityType
    
    print("=" * 50)
    print("Available Personalities")
    print("=" * 50)
    
    for ptype, personality in PERSONALITIES.items():
        print(f"\n{ptype.value.upper()}: {personality.name}")
        print(f"  Description: {personality.description}")
        print(f"  Traits: {personality.traits}")


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
    
    # Quick command - train and generate locally
    quick_parser = subparsers.add_parser("quick", help="Quick train & generate (no API)")
    quick_parser.add_argument("--dataset", default="datasets/shakespeare/input.txt", help="Training data")
    quick_parser.add_argument("--prompt", default="The king", help="Generation prompt")
    quick_parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    quick_parser.add_argument("--steps", type=int, default=50, help="Max training steps")
    quick_parser.add_argument("--embed", type=int, default=128, help="Embedding size")
    quick_parser.add_argument("--layers", type=int, default=4, help="Number of layers")
    quick_parser.add_argument("--heads", type=int, default=4, help="Number of heads")
    quick_parser.add_argument("--block", type=int, default=128, help="Block size")
    quick_parser.add_argument("--batch", type=int, default=32, help="Batch size")
    quick_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    quick_parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    quick_parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    quick_parser.add_argument("--output", default="models/quick.pt", help="Output model path")
    quick_parser.set_defaults(func=cmd_quick)
    
    # Models command
    models_parser = subparsers.add_parser("models", help="List models")
    models_parser.set_defaults(func=cmd_models)
    
    # Personalities command
    personalities_parser = subparsers.add_parser("personalities", help="List available personalities")
    personalities_parser.set_defaults(func=cmd_personalities)
    
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
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show model checkpoint info")
    info_parser.add_argument("model", nargs="?", default="models/sloughgpt.pt", help="Model path")
    info_parser.set_defaults(func=cmd_info)
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start HTTP inference server")
    serve_parser.add_argument("--host", default="localhost", help="Host")
    serve_parser.add_argument("--port", type=int, default=8080, help="Port")
    serve_parser.add_argument("--model", help="Model path")
    serve_parser.set_defaults(func=cmd_serve)
    
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
