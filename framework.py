#!/usr/bin/env python3
"""
SLO Framework Builder - Raw Python AI Infrastructure

A comprehensive framework that builds all tooling, frameworks, libraries,
and infrastructure needed for the SLO cognitive AI system.

Usage:
    python framework.py --build          # Build all components
    python framework.py --serve         # Start API server
    python framework.py --chat           # Start interactive chat
    python framework.py --train         # Train model
    python framework.py --status         # Check system status
"""

import argparse
import asyncio
import json
import os
import sys
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# CORE FRAMEWORK COMPONENTS
# ============================================================================

class FrameworkConfig:
    """Framework configuration"""
    
    def __init__(self):
        self.project_name = "SloughGPT"
        self.version = "2.0.0"
        self.mode = "development"
        
        # Paths
        self.runs_dir = PROJECT_ROOT / "runs"
        self.models_dir = PROJECT_ROOT / "models"
        self.datasets_dir = PROJECT_ROOT / "datasets"
        self.configs_dir = PROJECT_ROOT / "configs"
        
        # Storage
        self.store_path = str(self.runs_dir / "store" / "slo_store.db")
        self.rag_path = str(self.runs_dir / "store" / "rag_store.db")
        
        # Ensure directories exist
        self.runs_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.datasets_dir.mkdir(exist_ok=True)
        
        # Logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.runs_dir / "framework.log"),
                logging.StreamHandler()
            ]
        )


class ComponentBuilder:
    """Builds and manages framework components"""
    
    def __init__(self, config: FrameworkConfig):
        self.config = config
        self.logger = logging.getLogger("framework.builder")
        self.components: Dict[str, Any] = {}
    
    def build_slo(self) -> Any:
        """Build the SLO (Self-Learning Organism)"""
        self.logger.info("Building SLO...")
        
        try:
            from domains.soul.cognitive import CognitiveSLO
            from domains.soul.foundation import SLOConfig
            
            slo_config = SLOConfig(
                name="slo",
                learning_rate=0.01,
                memory_capacity=1000
            )
            
            slo = CognitiveSLO(slo_config)
            
            self.components["slo"] = slo
            self.logger.info("SLO built successfully")
            
            return slo
            
        except Exception as e:
            self.logger.error(f"Failed to build SLO: {e}")
            raise
    
    def build_rag_engine(self) -> Any:
        """Build the RAG Engine"""
        self.logger.info("Building RAG Engine...")
        
        try:
            from domains.infrastructure.rag import RAGEngine
            
            rag = RAGEngine(
                store_path=self.config.rag_path,
                enable_persistence=True
            )
            
            self.components["rag"] = rag
            self.logger.info("RAG Engine built successfully")
            
            return rag
            
        except Exception as e:
            self.logger.error(f"Failed to build RAG Engine: {e}")
            raise
    
    def build_memory_system(self) -> Any:
        """Build memory management system"""
        self.logger.info("Building Memory System...")
        
        try:
            from domains.soul.foundation import HaulsStore
            from domains.soul.cognitive import (
                SessionMemory,
                EpisodicMemoryStore,
                CognitiveArchitecture
            )
            
            memory_system = {
                "hauls_store": HaulsStore(),
                "session_memory": SessionMemory(),
                "episodic_store": EpisodicMemoryStore(),
                "cognitive_arch": CognitiveArchitecture()
            }
            
            self.components["memory"] = memory_system
            self.logger.info("Memory System built successfully")
            
            return memory_system
            
        except Exception as e:
            self.logger.error(f"Failed to build Memory System: {e}")
            raise
    
    def build_learning_system(self) -> Any:
        """Build learning systems"""
        self.logger.info("Building Learning System...")
        
        try:
            from domains.soul.cognitive import (
                NeuralPlasticityEngine,
                MetaLearningEngine,
                DreamProcessingEngine
            )
            from domains.infrastructure.rag import (
                SpacedRepetitionScheduler,
                SLOKnowledgeGraph
            )
            
            learning_system = {
                "plasticity": NeuralPlasticityEngine(learning_rate=0.01),
                "meta_learner": MetaLearningEngine(),
                "dream_engine": DreamProcessingEngine(),
                "spaced_repetition": SpacedRepetitionScheduler(),
                "knowledge_graph": SLOKnowledgeGraph()
            }
            
            self.components["learning"] = learning_system
            self.logger.info("Learning System built successfully")
            
            return learning_system
            
        except Exception as e:
            self.logger.error(f"Failed to build Learning System: {e}")
            raise
    
    def build_cognitive_domain(self) -> Any:
        """Build cognitive domain"""
        self.logger.info("Building Cognitive Domain...")
        
        try:
            from domains.cognitive.base import CognitiveDomain
            
            cognitive = CognitiveDomain()
            
            self.components["cognitive"] = cognitive
            self.logger.info("Cognitive Domain built successfully")
            
            return cognitive
            
        except Exception as e:
            self.logger.error(f"Failed to build Cognitive Domain: {e}")
            raise
    
    def build_infrastructure(self) -> Any:
        """Build infrastructure domain"""
        self.logger.info("Building Infrastructure Domain...")
        
        try:
            from domains.infrastructure.base import InfrastructureDomain
            
            infrastructure = InfrastructureDomain()
            
            self.components["infrastructure"] = infrastructure
            self.logger.info("Infrastructure Domain built successfully")
            
            return infrastructure
            
        except Exception as e:
            self.logger.error(f"Failed to build Infrastructure: {e}")
            raise
    
    def build_all(self) -> Dict[str, Any]:
        """Build all components"""
        self.logger.info("=" * 50)
        self.logger.info("Building Complete Framework...")
        self.logger.info("=" * 50)
        
        # Build in dependency order
        self.build_memory_system()
        self.build_learning_system()
        self.build_rag_engine()
        self.build_slo()
        
        self.logger.info("=" * 50)
        self.logger.info("Framework built successfully!")
        self.logger.info("=" * 50)
        
        return self.components


class SLOService:
    """Main SLO Service - Unified Interface"""
    
    def __init__(self, config: Optional[FrameworkConfig] = None):
        self.config = config or FrameworkConfig()
        self.builder = ComponentBuilder(self.config)
        self.slo = None
        self.rag = None
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the service"""
        self.logger = logging.getLogger("slo.service")
        self.logger.info("Initializing SLO Service...")
        
        # Build all components
        components = self.builder.build_all()
        
        self.slo = components.get("slo")
        self.rag = components.get("rag")
        
        self.is_initialized = True
        self.logger.info("SLO Service initialized")
    
    async def think(self, query: str, mode: str = "auto") -> Dict[str, Any]:
        """Process a query through the SLO"""
        if not self.is_initialized:
            await self.initialize()
        
        return self.slo.think(query, mode)
    
    async def chat(self, message: str) -> Dict[str, Any]:
        """Chat with the SLO"""
        if not self.is_initialized:
            await self.initialize()
        
        return self.slo.chat(message)
    
    async def learn(self, content: str, metadata: Optional[Dict] = None) -> int:
        """Teach the SLO new knowledge"""
        if not self.is_initialized:
            await self.initialize()
        
        return self.slo.add_knowledge(content, metadata)
    
    async def learn_from_feedback(
        self,
        user_input: str,
        response: str,
        feedback: str
    ) -> None:
        """Learn from user feedback"""
        if not self.is_initialized:
            await self.initialize()
        
        self.slo.learn_from_feedback(user_input, response, feedback)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        return {
            "status": "running",
            "version": self.config.version,
            "components": {
                "slo": self.slo is not None,
                "rag": self.rag is not None
            },
            "knowledge_stats": self.slo.rag_engine.get_knowledge_stats() if self.slo else {},
            "memory_stats": {
                "session_turns": len(self.slo.cognitive_arch.session_memory.conversation) if self.slo else 0,
                "episodes": len(self.slo.cognitive_arch.episodic_store.episodes) if self.slo else 0
            }
        }


class CLIManager:
    """Command-line interface manager"""
    
    def __init__(self):
        self.config = FrameworkConfig()
        self.service = SLOService(self.config)
        self.logger = logging.getLogger("cli")
    
    async def handle_build(self, args):
        """Handle build command"""
        print("Building SLO Framework...")
        components = self.service.builder.build_all()
        
        print("\n✓ Built Components:")
        for name in components.keys():
            print(f"  - {name}")
        
        print(f"\n✓ Framework ready at: {self.config.store_path}")
    
    async def handle_serve(self, args):
        """Handle serve command"""
        port = args.port or 8000
        
        print(f"Starting API server on port {port}...")
        
        try:
            from domains.ui.api import create_app
            
            app = create_app(self.service)
            
            import uvicorn
            uvicorn.run(app, host="0.0.0.0", port=port)
            
        except ImportError:
            print("Starting simple HTTP server...")
            await self._start_simple_server(port)
    
    async def _start_simple_server(self, port: int):
        """Start a simple HTTP server"""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json
        
        service = self.service
        
        class Handler(BaseHTTPRequestHandler):
            async def do_POST(self):
                content_length = int(self.headers['Content-Length'])
                body = self.rfile.read(content_length)
                data = json.loads(body)
                
                query = data.get("query", "")
                result = await service.think(query)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                status = asyncio.run(service.get_status())
                self.wfile.write(json.dumps(status).encode())
        
        server = HTTPServer(('0.0.0.0', port), Handler)
        print(f"Server running at http://localhost:{port}")
        server.serve_forever()
    
    async def handle_chat(self, args):
        """Handle chat command"""
        print("=" * 50)
        print("SLO Interactive Chat")
        print("Type 'quit' to exit")
        print("=" * 50)
        
        await self.service.initialize()
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                result = await self.service.think(user_input, mode=args.mode)
                
                print(f"\nSLO: {result.get('final_response', '')[:500]}")
                
                if args.verbose:
                    print(f"\n[Debug] Systems: {result.get('systems_used', [])}")
                    print(f"[Debug] Confidence: {result.get('metacognition', {}).get('confidence', 0):.2f}")
            
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    async def handle_status(self, args):
        """Handle status command"""
        await self.service.initialize()
        status = await self.service.get_status()
        
        print("\n" + "=" * 50)
        print("SLO Framework Status")
        print("=" * 50)
        
        print(f"\nStatus: {status.get('status', 'unknown')}")
        print(f"Version: {status.get('version', 'unknown')}")
        
        print("\nComponents:")
        for name, is_ready in status.get('components', {}).items():
            status_str = "✓ Ready" if is_ready else "✗ Not Ready"
            print(f"  {name}: {status_str}")
        
        print("\nKnowledge:")
        stats = status.get('knowledge_stats', {})
        print(f"  Total Documents: {stats.get('total_documents', 0)}")
        
        print("\nMemory:")
        mem = status.get('memory_stats', {})
        print(f"  Session Turns: {mem.get('session_turns', 0)}")
        print(f"  Episodes: {mem.get('episodes', 0)}")
        
        print()
    
    async def handle_learn(self, args):
        """Handle learn command"""
        await self.service.initialize()
        
        if args.file:
            # Learn from file
            path = Path(args.file)
            if path.exists():
                with open(path) as f:
                    content = f.read()
                doc_id = await self.service.learn(content, {"source": "file", "file": str(path)})
                print(f"Learned from file: {args.file} (doc_id: {doc_id})")
        elif args.content:
            # Learn from command line
            doc_id = await self.service.learn(args.content, {"source": "cli"})
            print(f"Learned (doc_id: {doc_id})")
        else:
            print("Error: Provide --file or --content")
    
    async def handle_train(self, args):
        """Handle train command - uses existing training infrastructure"""
        import time
        import logging
        import numpy as np
        from datetime import datetime
        
        # Setup logging
        log_file = self.config.runs_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger("sloughgpt.training")
        
        print("=" * 60)
        print("  SLOUGHGPT TRAINING")
        print("=" * 60)
        print(f"Data:       {args.data}")
        print(f"Epochs:     {args.epochs}")
        print(f"Batch:      {args.batch}")
        print(f"Layers:     {args.layers}")
        print(f"Embed:      {args.embed}")
        print(f"LR:         {args.lr}")
        print(f"Output:     {args.output}")
        print("=" * 60)
        
        logger.info(f"Starting training: epochs={args.epochs}, batch={args.batch}")
        
        start_time = time.time()
        
        try:
            # Use existing training infrastructure
            from domains.training import TrainingConfig, Trainer
            import torch
            from pathlib import Path
            
            # Setup config
            config = TrainingConfig(
                data_path=args.data,
                epochs=args.epochs,
                batch_size=args.batch,
                learning_rate=args.lr,
                n_embed=args.embed,
                n_layer=args.layers,
                block_size=128,
                vocab_size=args.vocab,
                output_path=args.output,
                max_batches=200,
            )
            
            # Create trainer (handles data + model internally)
            print("\n[1/3] Setting up trainer...")
            trainer = Trainer(config)
            trainer.setup()
            
            num_params = sum(p.numel() for p in trainer.model.model.parameters())
            print(f"Model ready: {num_params:,} parameters")
            
            # Train with real progress
            print("\n[2/3] Training...")
            print("-" * 60)
            
            # Use CPU for now (MPS has compatibility issues with some operations)
            device = "cpu"
            print("Using: CPU (MPS compatibility issues)")
            
            epochs = args.epochs
            total_batches = min(config.max_batches, 200)
            
            trainer.model.model.train()
            optimizer = torch.optim.AdamW(trainer.model.model.parameters(), lr=args.lr)
            
            for epoch in range(epochs):
                epoch_start = time.time()
                epoch_loss = 0
                batch_count = 0
                
                batch_gen = trainer.data_loader.get_batch(args.batch, 128)
                
                for batch_idx, (x, y) in enumerate(batch_gen):
                    if batch_idx >= total_batches:
                        break
                    
                    x_t = torch.tensor(x.astype(np.int64), dtype=torch.long).to(device)
                    y_t = torch.tensor(y.astype(np.int64), dtype=torch.long).to(device)
                    
                    optimizer.zero_grad()
                    logits, loss = trainer.model.model(x_t, y_t)
                    
                    if loss is not None:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(trainer.model.model.parameters(), 1.0)
                        optimizer.step()
                        epoch_loss += loss.item()
                        batch_count += 1
                    
                    # Progress every 10 batches
                    if batch_idx > 0 and batch_idx % 10 == 0:
                        elapsed = time.time() - start_time
                        batches_done = epoch * total_batches + batch_idx
                        total_batches_all = epochs * total_batches
                        if batches_done > 0:
                            eta_seconds = (elapsed / batches_done) * (total_batches_all - batches_done)
                            eta_mins = int(eta_seconds // 60)
                            eta_secs = int(eta_seconds % 60)
                            eta_str = f"{eta_mins}m {eta_secs}s"
                        else:
                            eta_str = "calc..."
                        
                        pct = int((batches_done / total_batches_all) * 100)
                        bar_len = 25
                        bar_filled = int(bar_len * batches_done / total_batches_all)
                        bar = "█" * bar_filled + "░" * (bar_len - bar_filled)
                        
                        curr_loss = loss.item() if loss else 0
                        print(f"E{epoch+1} [{bar}] {pct}% | b{batch_idx} | loss:{curr_loss:.3f} | ETA:{eta_str}")
                
                epoch_time = time.time() - epoch_start
                avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
                print(f"Epoch {epoch+1} done | avg_loss:{avg_loss:.4f} | {epoch_time:.1f}s")
                logger.info(f"Epoch {epoch+1} loss:{avg_loss:.4f}")
            
            elapsed_total = time.time() - start_time
            
            # Save
            print("\n[3/3] Saving model...")
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': trainer.model.model.state_dict(),
                'config': config.to_dict(),
            }, output_dir / "sloughgpt.pt")
            
            print("-" * 60)
            print(f"✓ DONE! Time:{elapsed_total:.1f}s | Saved: {output_dir/'sloughgpt.pt'}")
            print(f"  Log: {log_file}")
            logger.info(f"Training completed in {elapsed_total:.1f}s")
            
        except Exception as e:
            import traceback
            logger.error(f"Training failed: {e}")
            print(f"\n✗ FAILED: {e}")
            traceback.print_exc()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SLO Framework - Build AI Infrastructure with Raw Python"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build framework components")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--port", type=int, help="Server port")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")
    chat_parser.add_argument("--mode", default="auto", choices=["fast", "auto", "deep"])
    chat_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check system status")
    
    # Learn command
    learn_parser = subparsers.add_parser("learn", help="Teach SLO new knowledge")
    learn_parser.add_argument("--content", type=str, help="Content to learn")
    learn_parser.add_argument("--file", type=str, help="File to learn from")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data", type=str, default="datasets/karpathy/corpus.jsonl", help="Data path")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    train_parser.add_argument("--batch", type=int, default=32, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--layers", type=int, default=4, help="Number of layers")
    train_parser.add_argument("--embed", type=int, default=256, help="Embedding dimension")
    train_parser.add_argument("--output", type=str, default="models/sloughgpt", help="Output directory")
    train_parser.add_argument("--vocab", type=int, default=5000, help="Vocabulary size")
    
    args = parser.parse_args()
    
    # Run the appropriate command
    cli = CLIManager()
    
    if args.command == "build":
        asyncio.run(cli.handle_build(args))
    elif args.command == "serve":
        asyncio.run(cli.handle_serve(args))
    elif args.command == "chat":
        asyncio.run(cli.handle_chat(args))
    elif args.command == "status":
        asyncio.run(cli.handle_status(args))
    elif args.command == "learn":
        asyncio.run(cli.handle_learn(args))
    elif args.command == "train":
        asyncio.run(cli.handle_train(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
