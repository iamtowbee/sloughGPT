"""
New-style configurator using SLOConfig (Dockerfile/Modelfile format).

Usage:
    python train.py config/small.config
    python train.py config/standard.config --batch_size=16
    python train.py config/large.config

Still supports legacy .py config files for backward compatibility.
"""

import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

def load_configuration(config_file=None):
    """Load configuration using new SLOConfig system or fall back to legacy."""
    
    # Try new SLOConfig system first
    if config_file and config_file.endswith('.config'):
        try:
            from packages.core.src.services.slo_config import load_config
            
            print(f"Loading SLOConfig: {config_file}")
            slo_config = load_config(config_file)
            
            # Validate configuration
            errors = slo_config.validate()
            if errors:
                print("Configuration validation errors:")
                for error in errors:
                    print(f"  ❌ {error}")
                return None
            
            print(f"✓ Loaded {len(slo_config.config)} sections from {config_file}")
            return slo_config
            
        except ImportError:
            print("SLOConfig system not available, falling back to legacy")
        except Exception as e:
            print(f"Error loading SLOConfig: {e}")
            return None
    
    # Fall back to legacy system
    return load_legacy_config(config_file)

def load_legacy_config(config_file=None):
    """Load legacy Python configuration files."""
    
    # Default config values (same as before)
    defaults = {
        'out_dir': 'out',
        'eval_interval': 2000,
        'log_interval': 1,
        'eval_iters': 200,
        'always_save_checkpoint': True,
        'init_from': 'scratch',
        'wandb_log': True,
        'wandb_project': 'slo',
        'wandb_run_name': 'slo',
        'dataset': 'gopt',
        'gradient_accumulation_steps': 1,
        'batch_size': 64,
        'block_size': 80,
        'n_layer': 6,
        'n_head': 6,
        'n_embd': 384,
        'dropout': 0.0,
        'bias': False,
        'use_rope': False,
        'use_swiglu': False,
        'use_rmsnorm': False,
        'learning_rate': 1e-3,
        'max_iters': 5000,
        'weight_decay': 1e-1,
        'beta1': 0.9,
        'beta2': 0.95,
        'grad_clip': 1.0,
        'decay_lr': True,
        'warmup_iters': 2000,
        'lr_decay_iters': 5000,
        'min_lr': 1e-4,
        'backend': 'nccl'
    }
    
    # Apply defaults to globals
    globals().update(defaults)
    
    # Load config file if provided
    if config_file and os.path.exists(config_file):
        print(f"Loading legacy config: {config_file}")
        with open(config_file, 'r') as f:
            exec(f.read(), globals())
    
    # Process command line overrides
    for arg in sys.argv[1:]:
        if '=' in arg and arg.startswith('--'):
            key, value = arg[2:].split('=', 1)
            if key in globals():
                try:
                    import ast
                    parsed_value = ast.literal_eval(value)
                    globals()[key] = parsed_value
                    print(f"Override: {key} = {parsed_value}")
                except:
                    globals()[key] = value
                    print(f"Override: {key} = {value}")
            else:
                print(f"Warning: Unknown parameter {key}")
    
    return None

def get_config_file():
    """Extract config file from command line arguments."""
    for arg in sys.argv[1:]:
        if not arg.startswith('--') and not arg.startswith('-') and '=' not in arg:
            return arg
    return None

# Main execution
if __name__ == "__main__":
    config_file = get_config_file()
    
    # Try to load configuration
    slo_config = load_configuration(config_file)
    
    if slo_config is None and config_file:
        print("❌ Failed to load configuration")
        sys.exit(1)
    
    # If we have SLOConfig, apply it to globals
    if slo_config:
        config_dict = slo_config.to_globals()
        globals().update(config_dict)
        
        # Process any remaining command line overrides
        for arg in sys.argv[1:]:
            if '=' in arg and arg.startswith('--'):
                key, value = arg[2:].split('=', 1)
                try:
                    import ast
                    parsed_value = ast.literal_eval(value)
                    globals()[key] = parsed_value
                    print(f"CLI override: {key} = {parsed_value}")
                except:
                    globals()[key] = value
                    print(f"CLI override: {key} = {value}")
    
    print("✓ Configuration loaded successfully")
else:
    # When imported, load config if available
    config_file = get_config_file()
    slo_config = load_configuration(config_file)
    
    if slo_config:
        config_dict = slo_config.to_globals()
        globals().update(config_dict)