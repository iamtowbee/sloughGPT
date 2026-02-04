#!/usr/bin/env python3
"""
SloughGPT - Enterprise AI Framework Launcher

The main entry point for the SloughGPT enterprise platform.
This script provides a unified interface for all SloughGPT capabilities.
"""

import sys
import os
import argparse
import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from sloughgpt import (
        SloughGPTConfig, SloughGPT, SloughGPTTrainer,
        get_package_info, health_check
    )
    from sloughgpt.api_server import app
    from sloughgpt.admin import create_app, start_admin_server
    from sloughgpt.core.logging_system import get_logger, setup_logging
    from sloughgpt.core.database import get_database_manager, initialize_database
    FRAMEWORK_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  SloughGPT framework not available: {e}")
    FRAMEWORK_AVAILABLE = False

logger = get_logger(__name__)

class SloughGPTLauncher:
    """Main launcher for SloughGPT enterprise platform"""
    
    def __init__(self):
        self.config: Optional[SloughGPTConfig] = None
        self.setup_logging()
    
    def setup_logging(self):
        """Setup structured logging"""
        try:
            setup_logging(level="INFO")
            logger.info("SloughGPT launcher initialized")
        except Exception as e:
            print(f"⚠️  Failed to setup logging: {e}")
    
    async def initialize_framework(self, config_path: Optional[str] = None):
        """Initialize the SloughGPT framework"""
        try:
            # Load configuration
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                self.config = SloughGPTConfig(**config_data)
                logger.info(f"Loaded configuration from {config_path}")
            else:
                self.config = SloughGPTConfig()
                logger.info("Using default configuration")
            
            # Initialize database
            await initialize_database(self.config.database_config)
            logger.info("Database initialized")
            
            # Verify framework health
            health = await self._check_health()
            logger.info(f"Framework health: {health['status']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize framework: {e}")
            return False
    
    async def _check_health(self) -> Dict[str, Any]:
        """Check framework health"""
        try:
            if FRAMEWORK_AVAILABLE:
                return health_check()
            else:
                return {"status": "unavailable", "error": "Framework not loaded"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def start_api_server(self, host: str = "127.0.0.1", port: int = 8000):
        """Start the main API server"""
        try:
            if not FRAMEWORK_AVAILABLE:
                print("❌ Framework not available - cannot start API server")
                return False
            
            import uvicorn
            logger.info(f"Starting SloughGPT API server on {host}:{port}")
            
            config = uvicorn.Config(
                app=app,
                host=host,
                port=port,
                log_level="info",
                access_log=True
            )
            
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            return False
    
    async def start_admin_dashboard(self, host: str = "127.0.0.1", port: int = 8080):
        """Start the admin dashboard"""
        try:
            if not FRAMEWORK_AVAILABLE:
                print("❌ Framework not available - cannot start admin dashboard")
                return False
            
            logger.info(f"Starting SloughGPT admin dashboard on {host}:{port}")
            await start_admin_server(host=host, port=port)
            
        except Exception as e:
            logger.error(f"Failed to start admin dashboard: {e}")
            return False
    
    async def run_training(self, model_config: Dict[str, Any], data_path: str):
        """Run model training"""
        try:
            if not FRAMEWORK_AVAILABLE:
                print("❌ Framework not available - cannot run training")
                return False
            
            # Create configuration
            config = SloughGPTConfig(
                model_config=model_config,
                learning_config={"data_path": data_path}
            )
            
            # Initialize trainer
            trainer = SloughGPTTrainer(config)
            
            # Create and train model
            model = SloughGPT(config.model_config)
            
            logger.info(f"Starting model training with {len(model_config)} parameters")
            await trainer.train(model, data_path)
            
            logger.info("Training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def show_info(self):
        """Show package information"""
        try:
            if FRAMEWORK_AVAILABLE:
                info = get_package_info()
                print("🚀 SloughGPT Enterprise AI Framework")
                print("=" * 50)
                print(f"📦 Package: {info['name']} v{info['version']}")
                print(f"👤 Author: {info['author']}")
                print(f"📝 Description: {info['description']}")
                print("\n🔧 Component Status:")
                for component, count in info["components"].items():
                    status_icon = "✅" if count > 0 else "❌"
                    print(f"   {status_icon} {component}: {count} exports")
                print()
            else:
                print("❌ SloughGPT framework not available")
        except Exception as e:
            print(f"Error showing info: {e}")
    
    async def run_diagnostics(self):
        """Run system diagnostics"""
        print("🔍 Running SloughGPT System Diagnostics")
        print("=" * 50)
        
        # Check framework availability
        print(f"Framework Available: {'✅' if FRAMEWORK_AVAILABLE else '❌'}")
        
        if FRAMEWORK_AVAILABLE:
            # Check package info
            try:
                info = get_package_info()
                print(f"Package Version: {info['version']}")
            except Exception as e:
                print(f"Package Info Error: {e}")
            
            # Check health
            try:
                health = await self._check_health()
                print(f"Framework Health: {health['status']}")
                if "issues" in health:
                    for issue in health["issues"]:
                        print(f"  ⚠️  {issue}")
            except Exception as e:
                print(f"Health Check Error: {e}")
            
            # Check database
            try:
                db_manager = get_database_manager()
                if db_manager:
                    print("Database Connection: ✅")
                else:
                    print("Database Connection: ❌")
            except Exception as e:
                print(f"Database Error: {e}")
        
        # Check system resources
        try:
            import psutil
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            
            print(f"System CPU Cores: {cpu_count}")
            print(f"Available Memory: {memory.available // (1024**3):.1f} GB")
            print(f"Total Memory: {memory.total // (1024**3):.1f} GB")
            
        except ImportError:
            print("System Info: psutil not available")
        except Exception as e:
            print(f"System Info Error: {e}")
        
        # Check Python environment
        print(f"Python Version: {sys.version}")
        print(f"Python Path: {sys.executable}")
        
        # Check dependencies
        try:
            import torch
            print(f"PyTorch Version: {torch.__version__}")
            if torch.cuda.is_available():
                print(f"CUDA Available: ✅ ({torch.cuda.device_count()} GPUs)")
            else:
                print("CUDA Available: ❌")
        except ImportError:
            print("PyTorch: ❌ Not installed")
        
        try:
            import fastapi
            print(f"FastAPI Version: {fastapi.__version__}")
        except ImportError:
            print("FastAPI: ❌ Not installed")
        
        print("\n" + "=" * 50)

def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="SloughGPT Enterprise AI Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s serve                    # Start API server (default: 127.0.0.1:8000)
  %(prog)s serve --host 0.0.0.0 --port 8080
  %(prog)s admin                    # Start admin dashboard (default: 127.0.0.1:8080)
  %(prog)s admin --port 9000
  %(prog)s train --config model.json --data ./data
  %(prog)s info                     # Show package information
  %(prog)s health                    # Run system diagnostics
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start API server')
    serve_parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    serve_parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    serve_parser.add_argument('--config', help='Path to configuration file')
    
    # Admin command
    admin_parser = subparsers.add_parser('admin', help='Start admin dashboard')
    admin_parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    admin_parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    admin_parser.add_argument('--config', help='Path to configuration file')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--config', required=True, help='Path to model configuration file')
    train_parser.add_argument('--data', required=True, help='Path to training data')
    train_parser.add_argument('--output', help='Path to save trained model')
    
    # Info command
    subparsers.add_parser('info', help='Show package information')
    
    # Health command
    subparsers.add_parser('health', help='Run system diagnostics')
    
    # Version command
    subparsers.add_parser('version', help='Show version information')
    
    return parser

async def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize launcher
    launcher = SloughGPTLauncher()
    
    # Initialize framework if needed
    if args.command in ['serve', 'admin']:
        if not await launcher.initialize_framework(getattr(args, 'config', None)):
            print("❌ Failed to initialize SloughGPT framework")
            return 1
    
    # Execute command
    try:
        if args.command == 'serve':
            await launcher.start_api_server(args.host, args.port)
        elif args.command == 'admin':
            await launcher.start_admin_dashboard(args.host, args.port)
        elif args.command == 'train':
            # Load model configuration
            with open(args.config, 'r') as f:
                model_config = json.load(f)
            
            await launcher.run_training(model_config, args.data)
        elif args.command == 'info':
            launcher.show_info()
        elif args.command == 'health':
            await launcher.run_diagnostics()
        elif args.command == 'version':
            if FRAMEWORK_AVAILABLE:
                info = get_package_info()
                print(f"SloughGPT v{info['version']}")
            else:
                print("SloughGPT (framework not available)")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down SloughGPT...")
        return 0
    except Exception as e:
        logger.error(f"Command failed: {e}")
        return 1

if __name__ == "__main__":
    # Banner
    print("🚀 SloughGPT Enterprise AI Framework")
    print("=" * 50)
    
    if not FRAMEWORK_AVAILABLE:
        print("⚠️  Framework components not available")
        print("   Please ensure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        print()
    
    # Run main function
    sys.exit(asyncio.run(main()))