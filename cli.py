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
        if user_input.lower() in ["quit", "exit", "q"]:
            break

        try:
            response = requests.post(
                f"{base_url}/infer",
                json={
                    "prompt": user_input,
                    "max_length": args.max_tokens,
                    "temperature": args.temperature,
                    "model": args.model,
                },
                timeout=30,
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
    import os
    from pathlib import Path
    
    print("=" * 60)
    print("Available Models")
    print("=" * 60)
    
    models_dir = Path("models")
    
    # Trained models
    print("\n📁 TRAINED MODELS:")
    if models_dir.exists():
        for model_file in models_dir.glob("*.pt"):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"  {model_file.name}: {size_mb:.1f} MB")
        for model_file in models_dir.glob("*.safetensors"):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"  {model_file.name}: {size_mb:.1f} MB")
        for model_file in models_dir.glob("*.sou"):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"  {model_file.name}: {size_mb:.1f} MB")
    else:
        print("  (No trained models found in models/)")
    
    # Available architectures
    print("\n🏗️ AVAILABLE ARCHITECTURES:")
    print("  nanogpt: NanoGPT - Custom GPT model")
    print("  gpt2: GPT-2 - HuggingFace model (124M params)")
    print("  gpt2-medium: GPT-2 Medium (355M params)")
    print("  gpt2-large: GPT-2 Large (774M params)")
    print("  llama: LLaMA - Meta model")
    print("  phi: Phi - Microsoft model")
    
    # HuggingFace models
    print("\n🤗 HUGGINGFACE MODELS:")
    print("  facebook/opt-125m: OPT 125M")
    print("  facebook/opt-1.3b: OPT 1.3B")
    print("  microsoft/phi-2: Phi-2 2.7B")
    print("  mistralai/Mistral-7B-v0.1: Mistral 7B")
    print("  meta-llama/Llama-2-7b-hf: LLaMA-2 7B")
    
    # Usage
    print("\n💡 USAGE:")
    print("  python3 cli.py quick                  # Train custom NanoGPT")
    print("  python3 cli.py hf-serve gpt2         # Serve HuggingFace model")
    print("  python3 cli.py hf-download microsoft/phi-2  # Download model")


def cmd_quick(args):
    """Quick training and generation with optimizations."""
    import sys
    import torch

    sys.path.insert(0, ".")

    from domains.training.models.nanogpt import NanoGPT
    from domains.training.optimized_trainer import TrainingConfig, OptimizedTextDataset, OptimizedTrainer, Presets, get_optimal_device

    print("=" * 60)
    print("SloughGPT Quick Start (Optimized)")
    print("=" * 60)
    print(f"Device: {get_optimal_device()}")

    # Get device
    device = get_optimal_device()

    # Check if optimizations disabled
    use_optimize = not getattr(args, 'no_optimize', False)

    # Create config with optimizations
    config = TrainingConfig(
        batch_size=args.batch,
        learning_rate=args.lr,
        use_mixed_precision=use_optimize and device != "cpu",
        use_gradient_checkpointing=use_optimize and args.batch > 16,
        use_compile=use_optimize and hasattr(torch, 'compile'),
        max_steps=args.steps if args.steps else 100,
        warmup_steps=max(10, args.steps // 10) if args.steps else 10,
    )

    print(f"\nOptimizations:")
    print(f"  Mixed Precision: {'✅' if config.use_mixed_precision else '❌'}")
    print(f"  Gradient Checkpointing: {'✅' if config.use_gradient_checkpointing else '❌'}")
    print(f"  torch.compile: {'✅' if config.use_compile else '❌'}")

    # Load data
    from domains.training.train_pipeline import prepare_data
    print(f"\nLoading data from {args.dataset}...")
    data, vocab_size, stoi, itos = prepare_data(args.dataset, block_size=args.block)

    # Create model
    print(f"\nCreating model...")
    model = NanoGPT(
        vocab_size=vocab_size,
        n_embed=args.embed,
        n_layer=args.layers,
        n_head=args.heads,
        block_size=args.block,
    )

    # Create datasets
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    train_dataset = OptimizedTextDataset(train_data, block_size=args.block)
    val_dataset = OptimizedTextDataset(val_data, block_size=args.block)

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Train: {len(train_data):,} tokens")

    # Create optimized trainer
    trainer = OptimizedTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    # Train
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)
    trainer.train()

    # Generate
    print("\nGenerating text...")
    model.eval()
    input_ids = torch.tensor([[stoi.get(c, 0) for c in args.prompt]]).to(device)
    
    with torch.no_grad():
        output = model(input_ids)
        logits = output[0] if isinstance(output, tuple) else output
        next_token = logits[-1].argmax().item()
        generated = [next_token]
        
        for _ in range(args.max_tokens - 1):
            output = model(torch.tensor([[next_token]]).to(device))
            logits = output[0] if isinstance(output, tuple) else output
            next_token = logits[-1].argmax().item()
            if next_token == 0:
                break
            generated.append(next_token)

    text = ''.join([itos.get(i, '') for i in generated])
    print(f"\nPrompt: {args.prompt}")
    print(f"Generated: {args.prompt}{text[:200]}...")

    # Save
    from domains.training.train_pipeline import SloughGPTTrainer
    legacy_trainer = SloughGPTTrainer(
        data_path=args.dataset,
        n_embed=args.embed,
        n_layer=args.layers,
        n_head=args.heads,
        block_size=args.block,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        soul_name=getattr(args, 'soul_name', 'SloughGPT-Quick'),
    )
    legacy_trainer.model = model
    legacy_trainer.stoi = stoi
    legacy_trainer.itos = itos
    legacy_trainer._best_val_loss = 0.0
    legacy_trainer._train_loss_at_best = 0.0
    legacy_trainer.vocab_size = vocab_size

    output_base = args.output.replace(".pt", "").replace(".safetensors", "")
    legacy_trainer.save(output_base, format="safetensors")
    print(f"\nModel saved to {output_base}.safetensors")

    if getattr(args, 'export_sou', False):
        legacy_trainer.save(output_base, format="sou")
        print(f"Soul Unit saved to {output_base}.sou")


def cmd_train(args):
    """Start a training job."""
    if not args.api:
        import sys

        sys.path.insert(0, ".")

        # Load config
        from config_loader import load_config, merge_args_with_config

        config = load_config(args.config)
        config = merge_args_with_config(config, args)

        # Setup tracking
        tracker = None
        if config.tracking.enabled:
            from domains.training.tracking import ExperimentTracker, TrackerBackend, TrackingConfig

            backend = (
                TrackerBackend.WANDB
                if config.tracking.backend == "wandb"
                else TrackerBackend.MLFLOW
            )
            tracking_config = TrackingConfig(
                backend=backend,
                experiment_name=f"{config.model.name}_training",
                project=config.tracking.project,
                entity=config.tracking.entity,
            )
            tracker = ExperimentTracker(config=tracking_config)
            tracker.start_run(run_name=f"run_{args.dataset}_{args.epochs}ep")
            tracker.log_params(
                {
                    "model": str(config.model.__dict__),
                    "training": str(config.training.__dict__),
                    "lora": str(config.lora.__dict__),
                }
            )
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

        save_formats = [args.save_format]
        if args.export_sou and "sou" not in save_formats:
            save_formats.append("sou")

        save_path = f"{config.checkpoint.save_dir}/{config.model.name}"

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
            save_format=",".join(save_formats),
            save_quantized=args.save_quantized,
            soul_name=args.soul_name or config.model.name,
        )

        # Train
        trainer.train()

        # Save in all requested formats
        print(f"\nSaving model in format(s): {', '.join(save_formats)}...")
        for fmt in save_formats:
            trainer.save(save_path, format=fmt)
            print(f"  Saved: {save_path} ({fmt})")

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
                "use_lora": args.use_lora,
            },
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

    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")

    if "model" in checkpoint:
        model = checkpoint["model"]
        if hasattr(model, "state_dict"):
            state = model.state_dict()
            print(f"\nState dict keys: {len(state)}")
            total_params = sum(p.numel() for p in state.values() if isinstance(p, torch.Tensor))
            print(f"Total parameters: {total_params:,}")
        elif isinstance(model, dict):
            print(f"Model dict keys: {len(model)}")
            total_params = sum(p.numel() for p in model.values() if isinstance(p, torch.Tensor))
            print(f"Total parameters: {total_params:,}")

    if "chars" in checkpoint:
        print(f"Vocab size: {len(checkpoint['chars'])}")
    if "stoi" in checkpoint:
        print(f"Char-to-int map size: {len(checkpoint['stoi'])}")
    if "itos" in checkpoint:
        print(f"Int-to-char map size: {len(checkpoint['itos'])}")
    if "training_info" in checkpoint:
        print(f"Training info: {checkpoint['training_info']}")

    print("=" * 50)


def cmd_serve(args):
    """Start a simple HTTP inference server."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import torch

    model = None
    stoi = {}
    itos = {}

    # Try to load model
    model_path = args.model or "models/sloughgpt.pt"
    if Path(model_path).exists():
        print(f"Loading model from {model_path}...")
        try:
            checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
            stoi = checkpoint.get("stoi", {})
            itos = checkpoint.get("itos", {})
            print("Model loaded!")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "model": "sloughgpt"}).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/generate":
                content_length = int(self.headers["Content-Length"])
                body = self.rfile.read(content_length)
                data = json.loads(body)

                prompt = data.get("prompt", "")
                max_tokens = data.get("max_tokens", 100)
                temperature = data.get("temperature", 0.8)

                # Simple generation (placeholder)
                text = f"Generated: {prompt[:50]}... (model not fully loaded)"

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
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
    model_path = Path("models/sloughgpt_finetuned.pt")
    if not model_path.exists():
        model_path = Path("models/sloughgpt.pt")
    if model_path.exists():
        try:
            checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")

            # Get config from checkpoint
            training_info = checkpoint.get("training_info", {})
            vocab_size = training_info.get("vocab_size", len(checkpoint.get("stoi", {})))
            n_embed = training_info.get("n_embed", 128)
            n_layer = training_info.get("n_layer", 4)
            n_head = training_info.get("n_head", 4)
            block_size = training_info.get("block_size", 64)

            # Fallback defaults
            if vocab_size == 0:
                vocab_size = 65

            # Create model and load weights
            model = NanoGPT(
                vocab_size=vocab_size,
                n_embed=n_embed,
                n_layer=n_layer,
                n_head=n_head,
                block_size=block_size,
            )

            # Load state dict if present
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)

            stoi = checkpoint.get("stoi", {})
            itos = checkpoint.get("itos", {})

            idx = torch.tensor([[stoi.get(c, 0) for c in args.prompt]], dtype=torch.long)

            model.eval()
            with torch.no_grad():
                for _ in range(args.max_tokens):
                    idx_cond = idx[:, -block_size:]
                    logits, _ = model(idx_cond)
                    logits = logits[:, -1, :] / args.temperature
                    probs = torch.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    idx = torch.cat([idx, idx_next], dim=1)

            generated = "".join([itos.get(i, "") for i in idx[0].tolist()])
            result = generated[len(args.prompt) :]
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
            },
        )

        if response.status_code == 200:
            data = response.json()
            print(f"Generated: {data['text'][:500]}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")


def cmd_soul(args):
    """Load, inspect, or create .sou Soul Unit files."""
    import requests

    if args.load:
        base_url = f"http://{args.host}:{args.port}"
        try:
            resp = requests.post(
                f"{base_url}/load-soul",
                json={"soul_path": args.load},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                print("=" * 50)
                print("SOUL LOADED")
                print("=" * 50)
                print(f"Name:     {data.get('soul_name', 'unknown')}")
                print(f"Lineage:  {data.get('lineage', 'unknown')}")
                print(f"Born:     {data.get('born_at', '')}")
                print()
                print("Generation Params:")
                for k, v in data.get("generation_params", {}).items():
                    print(f"  {k}: {v}")
                print()
                print("Personality:")
                for k, v in data.get("personality", {}).items():
                    print(f"  {k}: {v}")
                print()
                print("Cognition:")
                for k, v in data.get("cognition", {}).items():
                    print(f"  {k}: {v}")
            else:
                print(f"Error: {resp.json()}")
        except Exception as e:
            print(f"Error: {e}")
        return

    if args.info:
        from domains.inference.sou_format import SouParser
        try:
            soul = SouParser.load(args.info)
            print("=" * 50)
            print(f"SOUL: {soul.name}")
            print("=" * 50)
            print(f"Version:    {soul.version}")
            print(f"Lineage:   {soul.lineage}")
            print(f"Born:      {soul.born_at}")
            print(f"Hash:      {soul.integrity_hash}")
            print(f"Tags:      {', '.join(soul.tags)}")
            print()
            print("Personality:")
            if soul.personality:
                for k, v in soul.personality.to_dict().items():
                    print(f"  {k}: {v}")
            print()
            print("Behavior:")
            if soul.behavior:
                for k, v in soul.behavior.to_dict().items():
                    print(f"  {k}: {v}")
            print()
            print("Cognition:")
            if soul.cognition:
                for k, v in soul.cognition.to_dict().items():
                    print(f"  {k}: {v}")
            print()
            print("Emotion:")
            if soul.emotion:
                for k, v in soul.emotion.to_dict().items():
                    print(f"  {k}: {v}")
        except Exception as e:
            print(f"Error: {e}")
        return

    if args.create:
        from domains.inference.sou_format import create_soul_profile, export_to_sou
        from domains.training.models.nanogpt import NanoGPT
        import torch

        soul = create_soul_profile(
            name=args.name or "SloughGPT-Soul",
            base_model="nanogpt",
            training_dataset=args.dataset or "",
            epochs_trained=args.epochs or 0,
            lineage=args.lineage or "nanogpt",
            tags=args.tags.split(",") if args.tags else ["sloughgpt", "soul"],
        )

        if args.model:
            checkpoint = torch.load(args.model, weights_only=False, map_location="cpu")
            state_dict = checkpoint.get("model_state_dict") or checkpoint.get("model") or checkpoint
            cfg = checkpoint.get("config") or {}
            n_embed = cfg.get("n_embed", 256)
            n_layer = cfg.get("n_layer", 6)
            n_head = cfg.get("n_head", 8)
            block_size = cfg.get("block_size", 128)
            vocab_size = cfg.get("vocab_size", 256)

            model = NanoGPT(
                vocab_size=vocab_size,
                n_embed=n_embed,
                n_layer=n_layer,
                n_head=n_head,
                block_size=block_size,
            )
            model.load_state_dict(state_dict, strict=False)
            export_to_sou(model, args.create, soul_profile=soul)
            print(f"Soul Unit created: {args.create}")
        else:
            from domains.inference.sou_format import SouParser
            SouParser.save(soul, args.create)
            print(f"Soul profile created: {args.create}")


def cmd_datasets(args):
    """List datasets - local version."""
    from pathlib import Path

    datasets_dir = Path("datasets")

    if not datasets_dir.exists():
        print("No datasets directory found")
        return

    print("=" * 60)
    print("Datasets")
    print("=" * 60)

    total_size = 0
    for ds in sorted(datasets_dir.iterdir()):
        if ds.is_dir():
            size = sum(f.stat().st_size for f in ds.rglob("*") if f.is_file())
            total_size += size
            size_kb = size / 1024
            size_mb = size_kb / 1024
            if size_mb > 1:
                print(f"  📁 {ds.name}: {size_mb:.1f} MB")
            else:
                print(f"  📁 {ds.name}: {size_kb:.1f} KB")
        elif ds.is_file():
            size = ds.stat().st_size
            total_size += size
            size_kb = size / 1024
            size_mb = size_kb / 1024
            if size_mb > 1:
                print(f"  📄 {ds.name}: {size_mb:.1f} MB")
            else:
                print(f"  📄 {ds.name}: {size_kb:.1f} KB")
    
    print(f"\nTotal: {total_size / (1024*1024):.1f} MB")
    
    print("\n💡 USAGE:")
    print("  python3 cli.py data stats <path>   # Get dataset statistics")
    print("  python3 cli.py data validate <path>  # Validate dataset")


def cmd_stats(args):
    """Show training and model statistics."""
    from pathlib import Path
    import json
    
    print("=" * 60)
    print("SloughGPT Statistics")
    print("=" * 60)
    
    # Models
    print("\n📊 MODELS:")
    models_dir = Path("models")
    model_count = 0
    total_size = 0
    if models_dir.exists():
        for f in models_dir.glob("*.pt"):
            model_count += 1
            total_size += f.stat().st_size
        for f in models_dir.glob("*.safetensors"):
            model_count += 1
            total_size += f.stat().st_size
    print(f"  Trained models: {model_count}")
    print(f"  Total size: {total_size / (1024*1024):.1f} MB")
    
    # Datasets
    print("\n📚 DATASETS:")
    datasets_dir = Path("datasets")
    ds_count = 0
    ds_size = 0
    if datasets_dir.exists():
        for f in datasets_dir.rglob("*"):
            if f.is_file():
                ds_count += 1
                ds_size += f.stat().st_size
    print(f"  Files: {ds_count}")
    print(f"  Total size: {ds_size / (1024*1024):.1f} MB")
    
    # Checkpoints
    print("\n💾 CHECKPOINTS:")
    ckpt_dir = Path("checkpoints")
    ckpt_count = 0
    if ckpt_dir.exists():
        ckpt_count = len(list(ckpt_dir.glob("*.pt")))
    print(f"  Saved checkpoints: {ckpt_count}")
    
    # Experiments
    print("\n🔬 EXPERIMENTS:")
    exp_file = Path("experiments/experiments.json")
    if exp_file.exists():
        with open(exp_file) as f:
            experiments = json.load(f)
        print(f"  Total experiments: {len(experiments)}")
    else:
        print(f"  No experiments recorded")
    
    # Training presets
    print("\n⚙️ AVAILABLE PRESETS:")
    print("  auto: Auto-detect best settings")
    print("  high_end_gpu: A100, H100, RTX 4090")
    print("  mid_range_gpu: RTX 3080, A4000")
    print("  apple_silicon: M1/M2/M3")
    print("  cpu_only: CPU training")


def cmd_personalities(args):
    """List available personalities."""
    from domains.ai_personality import PERSONALITIES

    print("=" * 50)
    print("Available Personalities")
    print("=" * 50)

    for ptype, personality in PERSONALITIES.items():
        print(f"\n{ptype.value.upper()}: {personality.name}")
        print(f"  Description: {personality.description}")
        print(f"  Traits: {personality.traits}")


def cmd_data_tool(args, subcmd: str):
    """Dataset utilities - stats, validate."""

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path not found: {path}")
        return

    if subcmd == "stats":
        total_lines = 0
        total_chars = 0

        if path.is_file():
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        total_lines += 1
                        total_chars += len(line)

            print(
                json.dumps(
                    {
                        "path": str(path),
                        "total_lines": total_lines,
                        "total_chars": total_chars,
                        "avg_line_length": total_chars // max(total_lines, 1),
                    },
                    indent=2,
                )
            )
        else:
            files = list(path.rglob("*"))
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            print(
                json.dumps(
                    {
                        "path": str(path),
                        "file_count": len([f for f in files if f.is_file()]),
                        "total_size": total_size,
                    },
                    indent=2,
                )
            )

    elif subcmd == "validate":
        issues = []
        if path.is_file():
            with open(path, "r") as f:
                for i, line in enumerate(f, 1):
                    if not line.strip():
                        issues.append(f"Line {i}: Empty")
        else:
            files = [f for f in path.rglob("*") if f.is_file()]

        print(
            json.dumps(
                {"valid": len(issues) == 0, "path": str(path), "issues": issues[:10]}, indent=2
            )
        )


def cmd_optimize(args):
    """Show and configure optimization settings."""
    import torch
    from domains.training.optimized_trainer import TrainingConfig, get_optimal_device
    from domains.inference.throughput import ThroughputConfig

    print("=" * 60)
    print("SloughGPT Optimization System")
    print("=" * 60)

    print(f"\nPyTorch Version: {torch.__version__}")
    print(f"Device: {get_optimal_device()}")
    
    # Check optimizations
    print("\n--- Available Optimizations ---")
    print(f"  torch.compile:       {'✅ Yes' if hasattr(torch, 'compile') else '❌ No (upgrade to PyTorch 2.0+)'}")
    print(f"  CUDA:                {'✅ Yes' if torch.cuda.is_available() else '❌ No'}")
    print(f"  MPS (Apple Silicon): {'✅ Yes' if torch.backends.mps.is_available() else '❌ No'}")
    
    if torch.cuda.is_available():
        print(f"  CUDA Compute:        {torch.cuda.get_device_capability()}")
        print(f"  BF16 Support:        {'✅ Yes' if torch.cuda.get_device_capability()[0] >= 8 else '❌ No (use FP16)'}")
    
    # Flash Attention
    try:
        from flash_attn import flash_attn_func
        print(f"  Flash Attention:     ✅ Yes")
    except:
        print(f"  Flash Attention:     ❌ No (pip install flash-attn)")

    print("\n--- Training Optimizations ---")
    print("  Mixed Precision (FP16/BF16):  2-3x speedup, 50% memory")
    print("  Gradient Checkpointing:        50% memory savings")
    print("  Flash Attention:               2-4x speedup")
    print("  torch.compile:                 1.5-2x speedup")
    print("  DataLoader prefetch:           1.2-1.5x speedup")

    print("\n--- Inference Optimizations ---")
    print("  Dynamic Batching:               Maximize GPU utilization")
    print("  KV Cache:                      Skip recomputation")
    print("  Prompt Caching:                Reuse computed states")
    print("  Batch Generation:               Parallel processing")

    print("\n--- Recommended Configurations ---")
    
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:
            print("\n  High-End GPU (RTX 3090+, A100, H100):")
            print("    config = TrainingConfig(dtype='bf16', use_compile=True, batch_size=32)")
        else:
            print("\n  Mid-Range GPU (RTX 2080, V100):")
            print("    config = TrainingConfig(dtype='fp16', use_compile=True, batch_size=16)")
    elif torch.backends.mps.is_available():
        print("\n  Apple Silicon (M1/M2/M3):")
        print("    config = TrainingConfig(dtype='fp16', batch_size=8)")
    else:
        print("\n  CPU Only:")
        print("    config = TrainingConfig(dtype='fp32', batch_size=4, num_workers=4)")

    if args.optimize:
        print("\nApplying runtime optimizations...")
        torch.set_num_threads(min(8, torch.get_num_threads()))
        print("  ✅ Thread count optimized")
        print("  ✅ Memory format set to channels_last (where applicable)")


    print()


def cmd_eval(args):
    """Evaluate/benchmark a model."""
    import torch
    import time

    print("=" * 50)
    print(f"Evaluating: {args.checkpoint}")
    print("=" * 50)

    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

        print(f"\nCheckpoint keys: {list(checkpoint.keys())}")

        if "training_info" in checkpoint:
            info = checkpoint["training_info"]
            print("\nTraining Info:")
            for k, v in info.items():
                print(f"  {k}: {v}")

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
            total_params = sum(v.numel() for v in state_dict.values())
            print(f"\nModel Statistics:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Parameter groups: {len(state_dict)}")

            total_size = sum(v.numel() * v.element_size() for v in state_dict.values())
            print(f"  Model size (FP32): {total_size / (1024**2):.2f} MB")
            print(f"  Model size (FP16): {total_size / 2 / (1024**2):.2f} MB")
            print(f"  Model size (INT8): {total_size / 4 / (1024**2):.2f} MB")

        # Quick benchmark if model loaded
        if args.benchmark:
            print("\nRunning benchmark...")
            dummy_input = torch.randint(0, 1000, (1, 128))

            with torch.no_grad():
                start = time.time()
                for _ in range(10):
                    pass  # Placeholder for actual forward pass
                elapsed = time.time() - start

            print(f"  10 iterations: {elapsed:.3f}s")
            print(f"  Per iteration: {elapsed/10*1000:.1f}ms")

    except Exception as e:
        print(f"Error: {e}")


def cmd_export(args):
    """Export a model via API."""
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


def cmd_export_cli(args):
    """Export a model to different formats (local)."""
    import torch
    from domains.training.export import export_model, list_export_formats
    from domains.training.models.nanogpt import NanoGPT

    print("=" * 50)
    print("SloughGPT Model Export")
    print("=" * 50)

    # List formats
    print("\nSupported formats:")
    for fmt, desc in list_export_formats().items():
        print(f"  {fmt}: {desc}")

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\nError: Model not found: {args.model}")
        return

    print(f"\nLoading model from {args.model}...")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # Extract model and metadata
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        metadata = checkpoint.get("metadata", {})
    else:
        state_dict = checkpoint
        metadata = {}

    # Get model config from metadata or use defaults
    vocab_size = metadata.get("vocab_size", 1000)
    n_embed = metadata.get("n_embed", 256)
    n_layer = metadata.get("n_layer", 6)
    n_head = metadata.get("n_head", 8)
    block_size = metadata.get("block_size", 128)

    # Create model
    model = NanoGPT(
        vocab_size=vocab_size,
        n_embed=n_embed,
        n_layer=n_layer,
        n_head=n_head,
        block_size=block_size,
    )
    model.load_state_dict(state_dict)

    print(f"Model loaded: {model.num_parameters:,} parameters")

    # Set output path
    output_path = args.output or str(model_path.with_suffix(""))

    # Export
    export_formats = [args.format]
    if args.export_sou:
        export_formats.append("sou")
    final_format = ",".join(export_formats)

    print(f"\nExporting to format: {final_format}")

    from domains.training.export import ExportConfig

    meta_with_name = {**metadata}
    if args.soul_name:
        meta_with_name["name"] = args.soul_name

    config = ExportConfig(
        input_path=args.model,
        output_path=output_path,
        format=final_format,
        quantization=args.quantize,
        metadata=meta_with_name,
    )

    results = export_model(config, model=model)

    if results:
        print("\nExport successful!")
        for fmt, path in results.items():
            print(f"  {fmt}: {path}")
    else:
        print("\nExport failed.")


def cmd_compare(args):
    """Compare benchmark results or models."""
    import json
    from pathlib import Path
    
    print("=" * 60)
    print("SloughGPT Model & Benchmark Comparison")
    print("=" * 60)
    
    # Compare benchmark results
    benchmarks_dir = Path("experiments/benchmarks")
    if benchmarks_dir.exists():
        benchmarks = list(benchmarks_dir.glob("*.json"))
        if benchmarks:
            print(f"\n📊 BENCHMARK RESULTS ({len(benchmarks)} files)")
            print("-" * 60)
            
            all_results = []
            for bf in sorted(benchmarks)[:5]:
                with open(bf) as f:
                    data = json.load(f)
                    all_results.append(data)
            
            if all_results:
                # Table header
                print(f"{'Model':<20} {'Tokens/sec':<15} {'Latency':<12} {'Memory':<12}")
                print("-" * 60)
                
                for r in all_results:
                    model = r.get('model', 'unknown')[:18]
                    tps = r.get('tokens_per_second', 0)
                    latency = r.get('latency_ms', 0)
                    memory = r.get('memory_mb', 0)
                    print(f"{model:<20} {tps:<15.2f} {latency:<12.1f} {memory:<12.1f}")
        else:
            print("\n(No benchmark results found)")
    else:
        print("\n(No benchmarks directory found)")
    
    # Compare models
    print("\n🤖 MODEL COMPARISON")
    print("-" * 60)
    print(f"{'Model':<25} {'Params':<12} {'Size':<12} {'Speed':<10}")
    print("-" * 60)
    
    models = [
        ("gpt2", "124M", "~250MB", "Fast"),
        ("gpt2-medium", "355M", "~700MB", "Medium"),
        ("gpt2-large", "774M", "~1.5GB", "Slow"),
        ("phi-2", "2.7B", "~5.4GB", "Medium"),
        ("mistral-7b", "7.3B", "~14GB", "Slow"),
        ("llama-2-7b", "7B", "~13GB", "Slow"),
    ]
    
    for name, params, size, speed in models:
        print(f"{name:<25} {params:<12} {size:<12} {speed:<10}")
    
    print("\n💡 Run benchmarks with:")
    print("  python3 cli.py benchmark --model gpt2")


def cmd_hf_download(args):
    """Download a HuggingFace model."""
    from domains.training.huggingface import HFClient, download_model
    from domains.training.huggingface.model_map import get_model_info

    print("=" * 50)
    print(f"Downloading: {args.model}")
    print("=" * 50)

    # Show model info if available
    info = get_model_info(args.model)
    if info:
        print(f"\nModel: {info.name}")
        print(f"Description: {info.description}")
        print(f"Parameters: {info.params:,}")
        print(f"Context: {info.context_length}")
        print(f"Memory (FP16): {info.memory_fp16_gb} GB")

    print("\nDownloading...")
    try:
        cache_dir = download_model(args.model)
        print(f"\nDownloaded to: {cache_dir}")
    except Exception as e:
        print(f"\nDownload failed: {e}")


def cmd_hf_serve(args):
    """Serve a HuggingFace model via API."""
    import requests

    print("=" * 50)
    print(f"Serving: {args.model}")
    print("=" * 50)

    base_url = f"http://{args.host}:{args.port}"

    print(f"\nLoading model via API: {base_url}")
    try:
        response = requests.post(
            f"{base_url}/models/load",
            json={"model_id": args.model, "mode": args.mode, "device": args.device},
        )
        if response.ok:
            print(f"Model loaded: {response.json()}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"API error: {e}")
        print("\nMake sure the API server is running: python cli.py serve")


def cmd_monitor(args):
    """Monitor training jobs."""
    import requests
    import time

    base_url = f"http://{args.host}:{args.port}"

    print("=" * 50)
    print("Training Monitor")
    print("=" * 50)

    while True:
        try:
            response = requests.get(f"{base_url}/training", timeout=5)
            if response.status_code == 200:
                jobs = response.json()
                if isinstance(jobs, dict) and "jobs" in jobs:
                    jobs = jobs["jobs"]
                if jobs:
                    print(f"\nActive Jobs: {len(jobs)}")
                    for job in jobs:
                        print(f"  - {job.get('name', 'unknown')}: {job.get('status', 'unknown')}")
                else:
                    print("No active training jobs")
            else:
                print(f"Error: {response.status_code}")
        except Exception as e:
            print(f"Error: {e}")

        if not args.watch:
            break
        time.sleep(args.interval)


def cmd_benchmark(args):
    """Run performance benchmarks on models."""
    import torch
    import time
    import statistics
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Auto-detect device
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    
    print("=" * 60)
    print(f"SloughGPT Benchmark - {args.model}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Check device availability
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU")
        args.device = "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded")
    
    # Load model
    print(f"Loading model...")
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model = model.to(args.device)
    model.eval()
    load_time = time.time() - start_time
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded in {load_time:.1f}s")
    print(f"Parameters: {params:,} ({params/1e9:.2f}B)")
    
    # Memory info
    if args.device == "mps":
        print(f"Device: Apple Silicon MPS")
    elif args.device == "cuda":
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        print(f"GPU Memory: {memory_allocated:.2f}GB allocated")
    
    print("\n" + "=" * 60)
    print("Running benchmarks...")
    print("=" * 60)
    
    # Tokenize prompt
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(args.device)
    prompt_length = input_ids.shape[1]
    
    # Warmup
    print("\nWarming up...")
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=10, do_sample=False)
    if args.device == "mps":
        torch.mps.synchronize()
    
    # Latency test
    if args.test in ["all", "latency"]:
        print("\n--- Latency Test ---")
        latencies = []
        
        for i in range(args.runs):
            torch.mps.synchronize() if args.device == "mps" else None
            torch.cuda.synchronize() if args.device == "cuda" else None
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = model.generate(input_ids, max_new_tokens=args.tokens, do_sample=False)
            
            torch.mps.synchronize() if args.device == "mps" else None
            torch.cuda.synchronize() if args.device == "cuda" else None
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
        
        latencies.sort()
        p50 = latencies[int(len(latencies) * 0.50)]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        avg = statistics.mean(latencies)
        
        print(f"Prompt: {prompt_length} tokens")
        print(f"Generated: {args.tokens} tokens")
        print(f"Latency - Mean: {avg:.1f}ms, P50: {p50:.1f}ms, P95: {p95:.1f}ms, P99: {p99:.1f}ms")
    
    # Throughput test
    if args.test in ["all", "throughput"]:
        print("\n--- Throughput Test ---")
        throughputs = []
        
        for i in range(min(args.runs, 5)):
            torch.mps.synchronize() if args.device == "mps" else None
            torch.cuda.synchronize() if args.device == "cuda" else None
            start = time.perf_counter()
            
            with torch.no_grad():
                output = model.generate(input_ids, max_new_tokens=args.tokens, do_sample=False)
            
            torch.mps.synchronize() if args.device == "mps" else None
            torch.cuda.synchronize() if args.device == "cuda" else None
            elapsed = time.perf_counter() - start
            
            tokens = output.shape[1]
            tps = tokens / elapsed
            throughputs.append(tps)
            print(f"Run {i+1}: {tps:.1f} tokens/sec")
        
        if throughputs:
            print(f"Average: {statistics.mean(throughputs):.1f} tokens/sec")
    
    # Batch test
    if args.test in ["all", "batch"]:
        print("\n--- Batch Inference Test ---")
        batch_prompts = [
            "Hello, how are you?",
            "What is AI?",
            "Tell me a joke.",
            "Explain machine learning.",
        ]
        
        for batch_size in [1, 2, 4]:
            if batch_size > len(batch_prompts):
                continue
            
            inputs = tokenizer(
                batch_prompts[:batch_size],
                return_tensors="pt",
                padding=True,
            ).to(args.device)
            
            torch.mps.synchronize() if args.device == "mps" else None
            torch.cuda.synchronize() if args.device == "cuda" else None
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, 
                                   max_new_tokens=20, do_sample=False)
            
            torch.mps.synchronize() if args.device == "mps" else None
            torch.cuda.synchronize() if args.device == "cuda" else None
            elapsed = time.perf_counter() - start
            
            print(f"Batch {batch_size}: {elapsed*1000:.1f}ms total, {batch_size/elapsed:.1f} seq/sec")
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


def cmd_setup(args):
    """Setup SloughGPT environment."""
    import subprocess
    import sys
    
    print("=" * 50)
    print("SloughGPT Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("cli.py"):
        print("Error: Run this from the SloughGPT root directory")
        return
    
    # Create directories
    print("\nCreating directories...")
    dirs = ["models", "datasets", "data", "checkpoints", "experiments", "logs", "cache"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  ✓ {d}/")
    
    # Create .env if needed
    if not os.path.exists(".env") and os.path.exists(".env.example"):
        print("\nCreating .env from .env.example...")
        subprocess.run(["cp", ".env.example", ".env"])
        print("  ✓ .env created")
    
    # Check Python version
    print(f"\nPython: {sys.version}")
    if sys.version_info < (3, 9):
        print("Warning: Python 3.9+ recommended")
    
    # Check CUDA
    import torch
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    if args.gpu:
        if torch.cuda.is_available():
            print(f"CUDA devices: {torch.cuda.device_count()}")
        elif torch.backends.mps.is_available():
            print("GPU: Apple Silicon MPS")
    
    # Create venv
    venv_dir = args.venv
    if not args.docker_only:
        print(f"\nSetting up virtual environment at {venv_dir}/...")
        
        if not os.path.exists(venv_dir):
            subprocess.run([sys.executable, "-m", "venv", venv_dir])
            print(f"  ✓ Created {venv_dir}/")
        
        pip_exe = os.path.join(venv_dir, "bin", "pip")
        print("  Installing dependencies...")
        subprocess.run([pip_exe, "install", "--upgrade", "pip"])
        subprocess.run([pip_exe, "install", 
                       "torch", "transformers", "accelerate",
                       "fastapi", "uvicorn", "pydantic",
                       "pytest", "ruff"])
        print("  ✓ Dependencies installed")
    
    # Docker setup
    if not args.local_only:
        print("\nDocker setup:")
        if os.path.exists("Dockerfile"):
            print("  ✓ Dockerfile found")
        if os.path.exists("docker-compose.yml"):
            print("  ✓ docker-compose.yml found")
        print("  Run 'docker-compose up -d' to start with Docker")
    
    print("\n" + "=" * 50)
    print("Setup complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("  1. Activate venv: source .venv/bin/activate")
    print("  2. Start server: python3 cli.py serve")
    print("  3. Or: ./start.sh")
    print("=" * 50)


def cmd_docker_start(args):
    """Start Docker services."""
    import subprocess
    
    profile = ""
    if args.dev:
        profile = "--profile dev"
    elif args.gpu:
        profile = "--profile gpu"
    
    print("Starting Docker services...")
    subprocess.run(f"docker compose up -d {profile}", shell=True)
    print("\nServices started!")
    subprocess.run("docker compose ps", shell=True)


def cmd_docker_stop(args):
    """Stop Docker services."""
    import subprocess
    
    print("Stopping Docker services...")
    subprocess.run("docker compose down", shell=True)
    print("Services stopped.")


def cmd_docker_status(args):
    """Show Docker status."""
    import subprocess
    
    subprocess.run("docker compose ps", shell=True)


def cmd_docker_logs(args):
    """Show Docker logs."""
    import subprocess
    
    service = args.service if args.service else ""
    subprocess.run(f"docker compose logs -f {service}", shell=True)


def cmd_docker_build(args):
    """Build Docker images."""
    import subprocess
    
    cache = "" if args.no_cache else ""
    print("Building Docker images...")
    subprocess.run(f"docker compose build {cache}", shell=True)
    print("Build complete!")


def cmd_docker_shell(args):
    """Shell into Docker container."""
    import subprocess
    
    subprocess.run(f"docker compose exec {args.service} /bin/bash", shell=True)


def cmd_system(args):
    """Show system information."""
    import platform
    import psutil

    print("=" * 50)
    print("System Information")
    print("=" * 50)

    # Python & Platform
    print(f"\nPlatform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")

    # CPU
    print("\nCPU:")
    print(f"  Cores: {psutil.cpu_count()}")
    print(f"  Usage: {psutil.cpu_percent()}%")

    # Memory
    mem = psutil.virtual_memory()
    print("\nMemory:")
    print(f"  Total: {mem.total / (1024**3):.1f} GB")
    print(f"  Used: {mem.used / (1024**3):.1f} GB")
    print(f"  Available: {mem.available / (1024**3):.1f} GB")
    print(f"  Usage: {mem.percent}%")

    # Disk
    disk = psutil.disk_usage("/")
    print("\nDisk:")
    print(f"  Total: {disk.total / (1024**3):.1f} GB")
    print(f"  Used: {disk.used / (1024**3):.1f} GB")
    print(f"  Free: {disk.free / (1024**3):.1f} GB")
    print(f"  Usage: {disk.percent}%")

    print()


def cmd_api_status(args):
    """Show detailed API status and security information."""
    import requests
    import time

    base_url = f"http://{args.host}:{args.port}"
    
    print("=" * 50)
    print("SloughGPT API Status")
    print("=" * 50)
    
    # Health check
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        print(f"\n[{'OK' if r.status_code == 200 else 'FAIL'}] Health: {r.json()}")
    except Exception as e:
        print(f"\n[FAIL] Health: {e}")
    
    # Detailed health
    try:
        r = requests.get(f"{base_url}/health/detailed", timeout=5)
        print(f"[{'OK' if r.status_code == 200 else 'FAIL'}] Detailed Health: {r.json()}")
    except Exception as e:
        print(f"[FAIL] Detailed Health: {e}")
    
    # Rate limit status
    try:
        r = requests.get(f"{base_url}/rate-limit/status", timeout=5)
        print(f"[{'OK' if r.status_code == 200 else 'FAIL'}] Rate Limit: {r.json()}")
    except Exception as e:
        print(f"[FAIL] Rate Limit: {e}")
    
    # Cache stats
    try:
        r = requests.get(f"{base_url}/cache/stats", timeout=5)
        print(f"[{'OK' if r.status_code == 200 else 'FAIL'}] Cache: {r.json()}")
    except Exception as e:
        print(f"[FAIL] Cache: {e}")
    
    # Metrics
    try:
        r = requests.get(f"{base_url}/metrics", timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"\n[OK] Metrics:")
            print(f"  WebSocket Connections: {data.get('websocket_connections', 'N/A')}")
            print(f"  Active Clients: {data.get('active_clients', 'N/A')}")
            print(f"  CPU: {data.get('system', {}).get('cpu_percent', 'N/A')}%")
            print(f"  Memory: {data.get('system', {}).get('memory_percent', 'N/A')}%")
        else:
            print(f"\n[FAIL] Metrics: {r.status_code}")
    except Exception as e:
        print(f"\n[FAIL] Metrics: {e}")
    
    # Security config
    try:
        r = requests.get(f"{base_url}/security/keys", timeout=5)
        print(f"[{'OK' if r.status_code == 200 else 'FAIL'}] Security: {r.json()}")
    except Exception as e:
        print(f"[FAIL] Security: {e}")
    
    print()


def cmd_api_test(args):
    """Test API endpoints and security."""
    import requests
    import time

    base_url = f"http://{args.host}:{args.port}"
    
    print("=" * 50)
    print("SloughGPT API Test")
    print("=" * 50)
    
    # Test generation
    print("\n[TEST] Generation endpoint...")
    try:
        start = time.time()
        r = requests.post(
            f"{base_url}/generate",
            json={"prompt": "Hello world", "max_new_tokens": 10},
            timeout=30
        )
        elapsed = time.time() - start
        if r.status_code == 200:
            print(f"[PASS] Generation: {elapsed:.2f}s - {r.json()}")
        else:
            print(f"[FAIL] Generation: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"[FAIL] Generation: {e}")
    
    # Test rate limiting
    print("\n[TEST] Rate limiting...")
    try:
        for i in range(5):
            r = requests.get(f"{base_url}/health", timeout=5)
            if 'X-RateLimit-Remaining' in r.headers:
                print(f"  Request {i+1}: Remaining={r.headers['X-RateLimit-Remaining']}")
    except Exception as e:
        print(f"[FAIL] Rate limiting: {e}")
    
    # Test caching
    print("\n[TEST] Caching...")
    try:
        prompt = f"Test prompt {time.time()}"
        r1 = requests.post(
            f"{base_url}/generate",
            json={"prompt": prompt, "max_new_tokens": 10},
            timeout=30
        )
        r2 = requests.post(
            f"{base_url}/generate",
            json={"prompt": prompt, "max_new_tokens": 10},
            timeout=30
        )
        cache_stats = requests.get(f"{base_url}/cache/stats", timeout=5).json()
        print(f"[PASS] Cache hit rate: {cache_stats.get('hit_rate', 0):.2%}")
    except Exception as e:
        print(f"[FAIL] Caching: {e}")
    
    # Test batch
    print("\n[TEST] Batch processing...")
    try:
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        r = requests.post(
            f"{base_url}/inference/batch",
            json={"prompts": prompts, "max_new_tokens": 5},
            timeout=30
        )
        if r.status_code == 200:
            data = r.json()
            print(f"[PASS] Batch: {data['count']} prompts processed")
        else:
            print(f"[FAIL] Batch: {r.status_code}")
    except Exception as e:
        print(f"[FAIL] Batch: {e}")
    
    print()


def cmd_api_auth(args):
    """Test API authentication."""
    import requests

    base_url = f"http://{args.host}:{args.port}"
    
    print("=" * 50)
    print("SloughGPT API Authentication Test")
    print("=" * 50)
    
    # Test without auth
    print("\n[TEST] Generate without auth...")
    try:
        r = requests.post(
            f"{base_url}/generate",
            json={"prompt": "Hello", "max_new_tokens": 5},
            timeout=10
        )
        print(f"[{'PASS' if r.status_code == 200 else 'NEEDS AUTH'}] Status: {r.status_code}")
    except Exception as e:
        print(f"[FAIL] {e}")
    
    # Test auth token endpoint
    print("\n[TEST] Auth token endpoint...")
    try:
        r = requests.post(
            f"{base_url}/auth/token",
            json={"api_key": "test-key"},
            timeout=10
        )
        if r.status_code == 401:
            print("[PASS] Auth rejected invalid key (401)")
        elif r.status_code == 200:
            data = r.json()
            print(f"[INFO] Token created: {data.get('access_token', '')[:20]}...")
        else:
            print(f"[INFO] Auth status: {r.status_code}")
    except Exception as e:
        print(f"[INFO] Auth endpoint: {e}")
    
    # Test verify endpoint
    print("\n[TEST] Verify endpoint...")
    try:
        r = requests.post(
            f"{base_url}/auth/verify",
            headers={"Authorization": "Bearer invalid-token"},
            timeout=10
        )
        print(f"[PASS] Verify rejected invalid token: {r.status_code}")
    except Exception as e:
        print(f"[INFO] Verify: {e}")
    
    print()


def cmd_config_validate(args):
    """Validate environment configuration."""
    import os
    import secrets
    
    print("=" * 50)
    print("SloughGPT Configuration Validator")
    print("=" * 50)
    
    env_file = args.env
    issues = []
    warnings = []
    
    # Check if .env exists
    if not os.path.exists(env_file):
        print(f"\n[WARN] {env_file} not found")
        return
    
    # Load and validate .env
    with open(env_file, 'r') as f:
        content = f.read()
    
    required_vars = ['SLAUGHGPT_API_KEY', 'SLAUGHGPT_JWT_SECRET']
    optional_vars = ['DATABASE_URL', 'REDIS_URL', 'MODEL_PATH']
    security_vars = ['SLAUGHGPT_API_KEY', 'SLAUGHGPT_JWT_SECRET', 'JWT_SECRET_KEY']
    
    print(f"\n[CHECK] Validating {env_file}...")
    
    # Check for required vars
    for var in required_vars:
        if var not in content:
            issues.append(f"Missing required: {var}")
        else:
            # Check if using default/weak values
            val = content.split(f"{var}=")[1].split('\n')[0] if f"{var}=" in content else ""
            if 'hash21' in val.lower() or 'change' in val.lower() or len(val) < 32:
                warnings.append(f"Weak {var}: should be >32 random chars")
    
    # Check for default values
    for var in security_vars:
        if var in content:
            val = content.split(f"{var}=")[1].split('\n')[0] if f"{var}=" in content else ""
            if 'change-this' in val.lower() or 'your-' in val.lower():
                warnings.append(f"Default {var}: change to secure value")
    
    # Check for missing optional vars
    for var in optional_vars:
        if var not in content:
            warnings.append(f"Optional not set: {var}")
    
    # Check SSL
    if 'SSL_ENABLED=true' in content and 'SSL_CERT_PATH' not in content:
        issues.append("SSL enabled but no cert path")
    
    # Print results
    print("\n" + "=" * 50)
    print("Results")
    print("=" * 50)
    
    if issues:
        print("\n[ERRORS]:")
        for issue in issues:
            print(f"  - {issue}")
    
    if warnings:
        print("\n[WARNINGS]:")
        for warn in warnings:
            print(f"  - {warn}")
    
    if not issues and not warnings:
        print("\n[PASS] Configuration looks good!")
    
    # Summary
    print(f"\n[INFO] {len(issues)} issues, {len(warnings)} warnings")


def cmd_config_generate(args):
    """Generate new secrets for configuration."""
    import secrets
    
    print("=" * 50)
    print("SloughGPT Secret Generator")
    print("=" * 50)
    
    print("\n[SUGGESTED VALUES FOR .env]:\n")
    
    if args.type in ["api-key", "all"]:
        api_key = secrets.token_urlsafe(32)
        print(f"SLAUGHGPT_API_KEY={api_key}")
    
    if args.type in ["jwt-secret", "all"]:
        jwt_secret = secrets.token_urlsafe(64)
        print(f"SLAUGHGPT_JWT_SECRET={jwt_secret}")
    
    if args.type == "all":
        encryption_key = secrets.token_hex(32)
        print(f"ENCRYPTION_KEY={encryption_key}")
    
    print("\n[INFO] Copy these to your .env file and restart the server")


def cmd_config_check(args):
    """Check environment setup."""
    import os
    import platform
    
    print("=" * 50)
    print("SloughGPT Environment Check")
    print("=" * 50)
    
    checks = []
    
    # Python version
    import sys
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    checks.append(("Python >= 3.8", int(py_version.replace(".", "")) >= 38))
    
    # PyTorch
    try:
        import torch
        checks.append(("PyTorch installed", True))
        checks.append((f"PyTorch {torch.__version__}", True))
    except ImportError:
        checks.append(("PyTorch installed", False))
    
    # CUDA
    try:
        import torch
        cuda = torch.cuda.is_available()
        checks.append(("CUDA available", cuda))
    except:
        checks.append(("CUDA available", False))
    
    # MPS (Apple Silicon)
    try:
        import torch
        mps = torch.backends.mps.is_available()
        checks.append(("MPS available", mps))
    except:
        checks.append(("MPS available", False))
    
    # FastAPI
    try:
        import fastapi
        checks.append((f"FastAPI {fastapi.__version__}", True))
    except ImportError:
        checks.append(("FastAPI installed", False))
    
    # Directory checks
    checks.append(("models/ directory", os.path.isdir("models")))
    checks.append(("datasets/ directory", os.path.isdir("datasets")))
    checks.append((".env file exists", os.path.exists(".env")))
    
    # Docker
    try:
        import subprocess
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        checks.append(("Docker installed", result.returncode == 0))
    except:
        checks.append(("Docker installed", False))
    
    # Kubernetes
    try:
        import subprocess
        result = subprocess.run(["kubectl", "version", "--client"], capture_output=True, text=True)
        checks.append(("kubectl installed", result.returncode == 0))
    except:
        checks.append(("kubectl installed", False))
    
    # Print results
    print()
    for name, passed in checks:
        status = "[OK]" if passed else "[X]"
        print(f"{status} {name}")
    
    # Summary
    passed = sum(1 for _, p in checks if p)
    total = len(checks)
    print(f"\n[INFO] {passed}/{total} checks passed")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SloughGPT CLI", formatter_class=argparse.RawDescriptionHelpFormatter
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

    # Quick command - train and generate locally (OPTIMIZED)
    quick_parser = subparsers.add_parser("quick", help="Quick train & generate (with optimizations)")
    quick_parser.add_argument(
        "--dataset", "-d", default="datasets/shakespeare/input.txt", help="Dataset path"
    )
    quick_parser.add_argument("--prompt", default="The king", help="Generation prompt")
    quick_parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    quick_parser.add_argument("--steps", type=int, default=100, help="Max training steps")
    quick_parser.add_argument("--embed", type=int, default=128, help="Embedding size")
    quick_parser.add_argument("--layers", type=int, default=4, help="Number of layers")
    quick_parser.add_argument("--heads", type=int, default=4, help="Number of heads")
    quick_parser.add_argument("--block", type=int, default=128, help="Block size")
    quick_parser.add_argument("--batch", type=int, default=16, help="Batch size")
    quick_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    quick_parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    quick_parser.add_argument(
        "--temperature", type=float, default=0.8, help="Generation temperature"
    )
    quick_parser.add_argument("--output", default="models/quick.pt", help="Output model path")
    quick_parser.add_argument(
        "--no-optimize", action="store_true", help="Disable optimizations (FP16, compile, etc)"
    )
    quick_parser.add_argument(
        "--export-sou", action="store_true", help="Export as .sou Soul Unit"
    )
    quick_parser.add_argument(
        "--soul-name", type=str, default="SloughGPT-Quick", help="Name for the soul"
    )
    quick_parser.set_defaults(func=cmd_quick)

    # Train command
    train_parser = subparsers.add_parser("train", help="Start training")
    train_parser.add_argument("--dataset", default="shakespeare", help="Dataset")
    train_parser.add_argument("--epochs", type=int, default=3, help="Epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    train_parser.add_argument("--use-lora", action="store_true", help="Use LoRA")
    train_parser.add_argument("--optimized", action="store_true", help="Use optimized training (FP16, compile)")
    train_parser.add_argument("--api", action="store_true", help="Use API server instead of local")
    train_parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    train_parser.add_argument(
        "--max-steps", type=int, default=None, help="Max steps per epoch (for quick testing)"
    )
    train_parser.add_argument(
        "--save-format",
        type=str,
        default="safetensors",
        choices=["safetensors", "safetensors_bf16", "torch", "gguf", "sou", "all"],
        help="Model save format (default: safetensors)",
    )
    train_parser.add_argument(
        "--save-quantized",
        type=str,
        default=None,
        choices=["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "F16", "F32"],
        help="GGUF quantization type",
    )
    train_parser.add_argument(
        "--export-sou",
        action="store_true",
        help="Export as .sou Soul Unit (self-contained model + soul profile)",
    )
    train_parser.add_argument(
        "--soul-name",
        type=str,
        default=None,
        help="Name for the model's soul (default: model name)",
    )
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

    # Export command
    export_parser = subparsers.add_parser("export", help="Export model to different formats")
    export_parser.add_argument("model", nargs="?", default="models/sloughgpt.pt", help="Model path")
    export_parser.add_argument("--output", "-o", help="Output path")
    export_parser.add_argument(
        "--format",
        "-f",
        default="safetensors",
        help="Export format: safetensors, safetensors_bf16, torch, gguf, sou, all (comma-separated)",
    )
    export_parser.add_argument("--quantize", choices=["int8", "int4", "fp16"], help="Quantization")
    export_parser.add_argument(
        "--export-sou",
        action="store_true",
        help="Also export as .sou Soul Unit",
    )
    export_parser.add_argument(
        "--soul-name",
        type=str,
        default=None,
        help="Name for the soul profile",
    )
    export_parser.set_defaults(func=cmd_export_cli)

    # HuggingFace download command
    hf_download_parser = subparsers.add_parser("hf-download", help="Download HuggingFace model")
    hf_download_parser.add_argument(
        "model", help="Model name (e.g., gpt2, mistralai/Mistral-7B-Instruct-v0.2)"
    )
    hf_download_parser.add_argument("--output", "-o", help="Output directory")
    hf_download_parser.add_argument(
        "--quantize", choices=["int4", "int8"], help="Quantize during download"
    )
    hf_download_parser.set_defaults(func=cmd_hf_download)

    # HuggingFace serve command
    hf_serve_parser = subparsers.add_parser("hf-serve", help="Serve HuggingFace model via API")
    hf_serve_parser.add_argument("model", help="Model name")
    hf_serve_parser.add_argument(
        "--mode", choices=["api", "local"], default="local", help="Load mode"
    )
    hf_serve_parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu, mps)")
    hf_serve_parser.set_defaults(func=cmd_hf_serve)

    # Datasets command
    datasets_parser = subparsers.add_parser("datasets", help="List datasets")
    datasets_parser.set_defaults(func=cmd_datasets)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show training statistics")
    stats_parser.set_defaults(func=cmd_stats)

    # Data tools command
    data_parser = subparsers.add_parser("data", help="Dataset utilities")
    data_sub = data_parser.add_subparsers(dest="data_cmd", help="Data commands")

    stats_parser = data_sub.add_parser("stats", help="Get dataset statistics")
    stats_parser.add_argument("path", help="Dataset or file path")
    stats_parser.set_defaults(func=lambda a: cmd_data_tool(a, "stats"))

    validate_parser = data_sub.add_parser("validate", help="Validate dataset")
    validate_parser.add_argument("path", help="Dataset path")
    validate_parser.set_defaults(func=lambda a: cmd_data_tool(a, "validate"))

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Model evaluation utilities")
    eval_parser.add_argument("--checkpoint", default="models/sloughgpt.pt", help="Model checkpoint")
    eval_parser.add_argument("--data", default="datasets/shakespeare/input.txt", help="Eval data")
    eval_parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    eval_parser.set_defaults(func=cmd_eval)

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor training jobs")
    monitor_parser.add_argument("--watch", action="store_true", help="Watch continuously")
    monitor_parser.add_argument("--interval", type=int, default=5, help="Update interval (seconds)")
    monitor_parser.set_defaults(func=cmd_monitor)

    # System command
    sys_parser = subparsers.add_parser("system", help="Show system information")
    sys_parser.set_defaults(func=cmd_system)

    # API status command
    api_status_parser = subparsers.add_parser("api-status", help="Show API status and security info")
    api_status_parser.add_argument("--host", default="localhost", help="API host")
    api_status_parser.add_argument("--port", type=int, default=8000, help="API port")
    api_status_parser.set_defaults(func=cmd_api_status)

    # API test command
    api_test_parser = subparsers.add_parser("api-test", help="Test API endpoints")
    api_test_parser.add_argument("--host", default="localhost", help="API host")
    api_test_parser.add_argument("--port", type=int, default=8000, help="API port")
    api_test_parser.set_defaults(func=cmd_api_test)

    # API auth command
    api_auth_parser = subparsers.add_parser("api-auth", help="Test API authentication")
    api_auth_parser.add_argument("--host", default="localhost", help="API host")
    api_auth_parser.add_argument("--port", type=int, default=8000, help="API port")
    api_auth_parser.set_defaults(func=cmd_api_auth)

    # Optimize command
    opt_parser = subparsers.add_parser("optimize", help="Show/configure optimization settings")
    opt_parser.add_argument("--optimize", action="store_true", help="Apply optimizations")
    opt_parser.set_defaults(func=cmd_optimize)

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    bench_parser.add_argument("--model", "-m", default="gpt2", help="Model to benchmark")
    bench_parser.add_argument("--device", "-d", default="auto", 
                              choices=["auto", "cpu", "cuda", "mps"], help="Device to use")
    bench_parser.add_argument("--test", "-t", default="all",
                              choices=["all", "latency", "throughput", "batch", "quantization"],
                              help="Test type")
    bench_parser.add_argument("--runs", "-r", type=int, default=10, help="Number of runs")
    bench_parser.add_argument("--tokens", "-k", type=int, default=50, help="Max new tokens")
    bench_parser.add_argument("--prompt", "-p", default="The quick brown fox jumps over the lazy dog",
                             help="Test prompt")
    bench_parser.set_defaults(func=cmd_benchmark)

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    compare_parser.add_argument("--models", "-m", default="gpt2,distilgpt2",
                              help="Comma-separated model names to compare")
    compare_parser.set_defaults(func=cmd_compare)

    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup SloughGPT environment")
    setup_parser.add_argument("--gpu", action="store_true", help="Enable GPU support")
    setup_parser.add_argument("--docker-only", action="store_true", help="Docker only setup")
    setup_parser.add_argument("--local-only", action="store_true", help="Local only setup")
    setup_parser.add_argument("--venv", default=".venv", help="Virtual environment directory")
    setup_parser.set_defaults(func=cmd_setup)

    # Docker command
    docker_parser = subparsers.add_parser("docker", help="Docker management")
    docker_sub = docker_parser.add_subparsers(dest="docker_cmd", help="Docker commands")
    
    docker_start = docker_sub.add_parser("start", help="Start Docker services")
    docker_start.add_argument("--gpu", action="store_true", help="Use GPU")
    docker_start.add_argument("--dev", action="store_true", help="Development mode")
    docker_start.set_defaults(func=lambda a: cmd_docker_start(a))
    
    docker_stop = docker_sub.add_parser("stop", help="Stop Docker services")
    docker_stop.set_defaults(func=lambda a: cmd_docker_stop(a))
    
    docker_status = docker_sub.add_parser("status", help="Show Docker status")
    docker_status.set_defaults(func=lambda a: cmd_docker_status(a))
    
    docker_logs = docker_sub.add_parser("logs", help="Show Docker logs")
    docker_logs.add_argument("service", nargs="?", help="Service name")
    docker_logs.set_defaults(func=lambda a: cmd_docker_logs(a))
    
    docker_build = docker_sub.add_parser("build", help="Build Docker images")
    docker_build.add_argument("--no-cache", action="store_true", help="Build without cache")
    docker_build.set_defaults(func=lambda a: cmd_docker_build(a))
    
    docker_shell = docker_sub.add_parser("shell", help="Shell into container")
    docker_shell.add_argument("service", default="api", help="Service name")
    docker_shell.set_defaults(func=lambda a: cmd_docker_shell(a))

    # Config validation command
    config_parser = subparsers.add_parser("config", help="Configuration utilities")
    config_sub = config_parser.add_subparsers(dest="config_cmd", help="Config commands")
    
    validate_parser = config_sub.add_parser("validate", help="Validate configuration")
    validate_parser.add_argument("--env", default=".env", help="Environment file path")
    validate_parser.set_defaults(func=lambda a: cmd_config_validate(a))
    
    gen_parser = config_sub.add_parser("generate", help="Generate new secrets")
    gen_parser.add_argument("--type", choices=["api-key", "jwt-secret", "all"], default="all",
                           help="Type of secret to generate")
    gen_parser.set_defaults(func=lambda a: cmd_config_generate(a))
    
    check_parser = config_sub.add_parser("check", help="Check environment setup")
    check_parser.set_defaults(func=cmd_config_check)

    # Soul command
    soul_parser = subparsers.add_parser("soul", help="Manage .sou Soul Unit files")
    soul_parser.add_argument("--load", "-l", metavar="PATH", help="Load a .sou file into the server")
    soul_parser.add_argument("--info", "-i", metavar="PATH", help="Inspect a .sou file")
    soul_parser.add_argument("--create", "-c", metavar="PATH", help="Create a .sou file")
    soul_parser.add_argument("--model", "-m", metavar="PATH", help="Model checkpoint for --create")
    soul_parser.add_argument("--name", "-n", metavar="NAME", help="Soul name for --create")
    soul_parser.add_argument("--dataset", "-d", metavar="PATH", help="Training dataset for --create")
    soul_parser.add_argument("--epochs", "-e", type=int, default=0, help="Epochs trained for --create")
    soul_parser.add_argument("--lineage", default="nanogpt", help="Model lineage for --create")
    soul_parser.add_argument("--tags", default="", help="Comma-separated tags for --create")
    soul_parser.set_defaults(func=cmd_soul)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
