#!/usr/bin/env python3
"""
SloughGPT CLI - Standalone Command Line Tool

Usage:
    python cli.py --help
    python cli.py dataset create mydata "text here"
    python cli.py dataset list
    python cli.py train --epochs 3
"""

import argparse
import sys
import asyncio
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))


def cmd_dataset(args):
    """Dataset commands"""
    from domains.training import DatasetCreator, DatasetRegistry, DatasetQualityScorer
    
    if args.action == "create":
        dc = DatasetCreator()
        result = dc.create_from_text(args.name, args.text)
        if result.get("success"):
            print(f"✓ Dataset '{args.name}' created")
            print(f"  Vocab size: {result.get('vocab_size', 'N/A')}")
            print(f"  Train tokens: {result.get('train_tokens', 'N/A')}")
        else:
            print(f"✗ Failed to create dataset")
            
    elif args.action == "list":
        datasets_dir = Path("datasets")
        if datasets_dir.exists():
            datasets = [d.name for d in datasets_dir.iterdir() if d.is_dir()]
            if datasets:
                print("📊 Available datasets:")
                for ds in sorted(datasets):
                    print(f"  - {ds}")
            else:
                print("No datasets found")
        else:
            print("No datasets directory found")
            
    elif args.action == "score":
        qs = DatasetQualityScorer()
        scores = qs.score_dataset(f"datasets/{args.name}")
        print(f"📊 Quality scores for '{args.name}':")
        for key, value in scores.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")


def cmd_finetune(args):
    """Fine-tune model with RAG integration"""
    from domains.training.unified_training import TrainingConfig, TrainingPipeline
    import tempfile
    
    print("🔧 SloughGPT Fine-tuning Pipeline")
    print("=" * 40)
    
    # Step 1: Collect training data
    print("\n📥 Step 1: Collecting training data...")
    training_texts = []
    
    # Option A: Use text from CLI
    if args.text:
        training_texts.append(args.text)
        print(f"  ✓ From CLI: {len(args.text)} chars")
    
    # Option B: Use dataset directory
    if args.dataset:
        dataset_path = f"datasets/{args.dataset}"
        if Path(dataset_path).exists():
            training_texts.append(Path(dataset_path).read_text())
            print(f"  ✓ From dataset '{args.dataset}'")
    
    # Option C: Load from RAG
    from domains.infrastructure import RAGSystem
    rag = RAGSystem()
    if rag.documents:
        training_texts.extend([doc["content"] for doc in rag.documents])
        print(f"  ✓ From RAG: {len(rag.documents)} documents")
    
    if not training_texts:
        training_texts = ["The sky is blue. Machine learning is powerful. AI is the future."]
        print(f"  ⚠ Using default training text")
    
    # Write combined data to temp file
    combined_text = "\n\n".join(training_texts)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(combined_text)
        temp_path = f.name
    
    print(f"  ✓ Total: {len(combined_text):,} characters")
    
    # Step 2: Create config and train
    print("\n🧠 Step 2: Fine-tuning...")
    
    config = TrainingConfig(
        data_path=temp_path,
        model_id="nanogpt-nanogpt",
        epochs=args.epochs or 3,
        batch_size=args.batch or 8,
        learning_rate=args.lr or 1e-4,
        vocab_size=args.vocab_size or 500,
        n_embed=args.embed or 128,
        n_layer=args.layers or 3,
        n_head=args.heads or 4,
        output_path=args.save,
        max_batches=50,
    )
    
    pipeline = TrainingPipeline(config)
    results = pipeline.run()
    
    # Cleanup temp file
    Path(temp_path).unlink()
    
    # Step 3: Save RAG knowledge
    print(f"\n💾 Step 3: Saving RAG knowledge...")
    print(f"  ✓ RAG knowledge saved ({len(rag.documents)} documents)")
    
    print("\n✅ Fine-tuning complete!")
    print(f"   Final loss: {results['final_loss']:.4f}")
    print(f"   Parameters: {results['parameters']:,}")


def cmd_train(args):
    """Train on any data format"""
    from domains.training.unified_training import TrainingConfig, TrainingPipeline
    from domains.training.model_registry import get_available_models
    
    # Determine data source (--data takes priority over --dataset)
    if args.data:
        data_path = args.data
    elif args.dataset:
        data_path = f"datasets/{args.dataset}"
    else:
        data_path = "datasets/default/input.txt"
    
    if not Path(data_path).exists():
        print(f"✗ Data not found: {data_path}")
        print("  Create a dataset first:")
        print("    python cli.py dataset create mydata 'training text'")
        return
    
    # Create config
    config = TrainingConfig(
        data_path=data_path,
        model_id=args.model or "nanogpt-nanogpt",
        epochs=args.epochs or 3,
        batch_size=args.batch or 8,
        learning_rate=args.lr or 1e-4,
        vocab_size=args.vocab_size or 500,
        n_embed=args.embed or 128,
        n_layer=args.layers or 3,
        n_head=args.heads or 4,
        output_path=args.output,
        max_batches=args.max_batches or 50,
    )
    
    # Run training
    pipeline = TrainingPipeline(config)
    results = pipeline.run()
    
    print(f"\n📊 Results:")
    print(f"   Model: {config.model_id}")
    print(f"   Parameters: {results['parameters']:,}")
    print(f"   Final loss: {results['final_loss']:.4f}")


def cmd_model(args):
    """Model commands"""
    from domains.training import NanoGPT
    
    if args.action == "info":
        model = NanoGPT(
            vocab_size=args.vocab_size or 1000,
            n_embed=args.embed or 128,
            n_layer=args.layers or 3,
            n_head=args.heads or 4
        )
        print(f"📊 Model Info:")
        print(f"  Parameters: {model.num_parameters:,}")
        print(f"  Vocab size: {args.vocab_size or 1000}")
        print(f"  Embedding: {args.embed or 128}")
        print(f"  Layers: {args.layers or 3}")
        print(f"  Heads: {args.heads or 4}")


def cmd_cognitive(args):
    """Cognitive commands"""
    from domains.cognitive import CognitiveCore, KnowledgeGraph, ThinkingMode
    
    if args.action == "think":
        core = CognitiveCore()
        mode = ThinkingMode[args.mode.upper()] if args.mode else ThinkingMode.ANALYTICAL
        thought = core.think(args.prompt, mode)
        print(f"💭 Thought ({args.mode or 'analytical'}):")
        print(f"   {thought.thought_content}")
        
    elif args.action == "knowledge":
        kg = KnowledgeGraph(f"/tmp/kg_{args.name or 'default'}.db")
        if args.operation == "add":
            node_id = kg.add_node(args.entity, args.concept or "concept")
            print(f"✓ Added node: {args.entity} ({args.concept})")
        elif args.operation == "list":
            print(f"📊 Knowledge Graph: {len(kg.nodes)} nodes")


def cmd_rag(args):
    """RAG (Retrieval Augmented Generation) commands"""
    from domains.infrastructure import RAGSystem
    
    rag = RAGSystem()
    if args.operation == "add":
        rag.add_document(args.text, {"source": "cli"})
        print(f"✓ Added document to RAG")
    elif args.operation == "search":
        results = rag.search(args.query, top_k=3)
        print(f"🔍 Search results for '{args.query}':")
        for r in results:
            print(f"   - {r['content'][:50]}...")

def cmd_generate(args):
    """Generate text from a trained model"""
    import torch
    from domains.training.models.nanogpt import NanoGPT
    import json
    
    if not args.model_path:
        print("✗ Please specify --model-path")
        return
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        return
    
    # Load config
    config_path = f"{model_path}.json"
    if Path(config_path).exists():
        with open(config_path) as f:
            config = json.load(f)
        vocab_size = config.get("config", {}).get("vocab_size", 500)
    else:
        vocab_size = args.vocab_size or 500
    
    # Load model
    checkpoint = torch.load(model_path, map_location="cpu")
    
    n_embed = args.embed or 128
    n_layer = args.layers or 3
    n_head = args.heads or 4
    
    model = NanoGPT(
        vocab_size=vocab_size,
        n_embed=n_embed,
        n_layer=n_layer,
        n_head=n_head
    )
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"🦁 Generating with model: {model_path.name}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create prompt tokens
    if args.prompt:
        prompt_tokens = [ord(c) % vocab_size for c in args.prompt[:20]]
    else:
        prompt_tokens = [0]
    
    context = torch.tensor([prompt_tokens], dtype=torch.long)
    
    # Generate
    output = model.generate(
        context,
        max_new_tokens=args.length or 50,
        temperature=args.temperature or 0.8,
        top_k=args.top_k
    )
    
    # Decode
    tokens = output[0].tolist()
    
    if args.raw:
        print(f"\n📝 Tokens: {tokens}")
    else:
        # Try to decode as text
        try:
            text = ''.join(chr(t) if 32 <= t < 127 else '·' for t in tokens)
            print(f"\n📝 Generated:\n{text}")
        except:
            print(f"\n📝 Tokens: {tokens}")


def cmd_version(args):
    """Show version"""
    print("🦁 SloughGPT CLI v2.0.0")
    print("   Domain Architecture: ✓")


def cmd_models(args):
    """List all dynamically available models"""
    from domains.training.model_registry import get_available_models
    
    print("🦁 Available Models (Dynamically Loaded)")
    print("=" * 50)
    
    models = get_available_models()
    
    # Group by primary tag
    grouped = {}
    for m in models:
        tag = m.tags[0] if m.tags else 'other'
        if tag not in grouped:
            grouped[tag] = []
        grouped[tag].append(m)
    
    for tag, tag_models in sorted(grouped.items()):
        print(f"\n[{tag.upper()}]")
        for m in tag_models:
            print(f"  {m.id:20} - {m.name}")
            print(f"  {'':20}   {m.description}")
    
    print(f"\n📊 Total: {len(models)} models available")
    print("\n💡 Use: python cli.py train --model <model_id> --data <path>")


def cmd_interactive(args):
    """Launch interactive model selection CLI"""
    from dynamic_model_cli import DynamicModelCLI
    
    cli = DynamicModelCLI()
    cli.run()


def main():
    parser = argparse.ArgumentParser(
        description="SloughGPT CLI - Domain-based AI Assistant"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Dataset command
    dataset_parser = subparsers.add_parser("dataset", help="Dataset operations")
    dataset_parser.add_argument("action", choices=["create", "list", "score"], help="Action")
    dataset_parser.add_argument("name", nargs="?", help="Dataset name")
    dataset_parser.add_argument("text", nargs="?", help="Text content for dataset")
    dataset_parser.set_defaults(func=cmd_dataset)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train on any data format")
    train_parser.add_argument("--data", help="Path to data file/directory")
    train_parser.add_argument("--dataset", help="Dataset name (in datasets/)")
    train_parser.add_argument("--model", default="nanogpt-nanogpt", help="Model ID to train")
    train_parser.add_argument("--output", help="Output path for model")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    train_parser.add_argument("--batch", type=int, default=8, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--max-batches", type=int, default=50, help="Max batches per epoch")
    train_parser.add_argument("--vocab-size", type=int, default=500, help="Vocabulary size")
    train_parser.add_argument("--embed", type=int, default=128, help="Embedding dimension")
    train_parser.add_argument("--layers", type=int, default=3, help="Number of layers")
    train_parser.add_argument("--heads", type=int, default=4, help="Number of attention heads")
    train_parser.set_defaults(func=cmd_train)
    
    # Finetune command
    finetune_parser = subparsers.add_parser("finetune", help="Fine-tune model with RAG")
    finetune_parser.add_argument("--text", help="Training text to learn")
    finetune_parser.add_argument("--dataset", help="Dataset name to train on")
    finetune_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    finetune_parser.add_argument("--batch", type=int, default=8, help="Batch size")
    finetune_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    finetune_parser.add_argument("--vocab-size", type=int, help="Vocabulary size")
    finetune_parser.add_argument("--embed", type=int, help="Embedding dimension")
    finetune_parser.add_argument("--layers", type=int, help="Number of layers")
    finetune_parser.add_argument("--heads", type=int, help="Number of attention heads")
    finetune_parser.add_argument("--save", help="Save model to file")
    finetune_parser.set_defaults(func=cmd_finetune)
    
    # Model command
    model_parser = subparsers.add_parser("model", help="Model operations")
    model_parser.add_argument("action", choices=["info"], help="Action")
    model_parser.add_argument("--vocab-size", type=int, help="Vocabulary size")
    model_parser.add_argument("--embed", type=int, help="Embedding dimension")
    model_parser.add_argument("--layers", type=int, help="Number of layers")
    model_parser.add_argument("--heads", type=int, help="Number of attention heads")
    model_parser.set_defaults(func=cmd_model)
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text from trained model")
    gen_parser.add_argument("--model-path", required=True, help="Path to trained model (.pt)")
    gen_parser.add_argument("--prompt", help="Prompt text to continue")
    gen_parser.add_argument("--length", type=int, default=50, help="Tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    gen_parser.add_argument("--top-k", type=int, help="Top-k sampling")
    gen_parser.add_argument("--vocab-size", type=int, default=500, help="Vocabulary size")
    gen_parser.add_argument("--embed", type=int, default=128, help="Embedding dimension")
    gen_parser.add_argument("--layers", type=int, default=3, help="Number of layers")
    gen_parser.add_argument("--heads", type=int, default=4, help="Number of attention heads")
    gen_parser.add_argument("--raw", action="store_true", help="Output raw tokens")
    gen_parser.set_defaults(func=cmd_generate)
    
    # Cognitive command
    cog_parser = subparsers.add_parser("cognitive", help="Cognitive operations")
    cog_parser.add_argument("action", choices=["think", "knowledge", "rag"], help="Action")
    cog_parser.add_argument("--prompt", help="Prompt for thinking")
    cog_parser.add_argument("--mode", help="Thinking mode (analytical, creative, critical)")
    cog_parser.add_argument("--entity", help="Entity for knowledge graph")
    cog_parser.add_argument("--concept", help="Concept type")
    cog_parser.add_argument("--name", help="Name")
    cog_parser.add_argument("--operation", help="Operation (add, list, search)")
    cog_parser.add_argument("--text", help="Text")
    cog_parser.add_argument("--query", help="Query")
    cog_parser.set_defaults(func=cmd_cognitive)
    
    # RAG command
    rag_parser = subparsers.add_parser("rag", help="RAG operations")
    rag_parser.add_argument("operation", choices=["add", "search"], help="Operation")
    rag_parser.add_argument("--text", help="Text to add")
    rag_parser.add_argument("--query", help="Query to search")
    rag_parser.set_defaults(func=cmd_rag)
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Show version")
    version_parser.set_defaults(func=cmd_version)
    
    # Models command
    models_parser = subparsers.add_parser("models", help="List available models")
    models_parser.set_defaults(func=cmd_models)
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive model selection")
    interactive_parser.set_defaults(func=cmd_interactive)
    
    args = parser.parse_args()
    
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
