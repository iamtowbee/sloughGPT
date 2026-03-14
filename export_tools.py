#!/usr/bin/env python3
"""Model export utilities for SloughGPT."""

import json
import torch
import argparse
from pathlib import Path
from typing import Dict, Any, Optional


def export_to_onnx(model, dummy_input, output_path: str, opset_version: int = 14) -> bool:
    """Export PyTorch model to ONNX format."""
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        return True
    except Exception as e:
        print(f"ONNX export failed: {e}")
        return False


def export_to_torchscript(model, dummy_input, output_path: str) -> bool:
    """Export PyTorch model to TorchScript."""
    try:
        model.eval()
        traced = torch.jit.trace(model, dummy_input)
        traced.save(output_path)
        return True
    except Exception as e:
        print(f"TorchScript export failed: {e}")
        return False


def export_checkpoint_info(checkpoint_path: str, output_path: Optional[str] = None) -> bool:
    """Export checkpoint metadata to JSON."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        info = {
            "checkpoint_path": checkpoint_path,
            "keys": list(checkpoint.keys()) if isinstance(checkpoint, dict) else [],
        }
        
        if 'training_info' in checkpoint:
            info["training_info"] = checkpoint['training_info']
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            info["model_keys"] = len(state_dict) if hasattr(state_dict, '__len__') else "unknown"
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(info, f, indent=2)
            print(f"Saved to {output_path}")
        else:
            print(json.dumps(info, indent=2))
        
        return True
    except Exception as e:
        print(f"Export failed: {e}")
        return False


def quantize_model(model_path: str, output_path: str, dtype: str = "int8") -> bool:
    """Quantize a model for faster inference."""
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if 'model' not in checkpoint:
            print("No model found in checkpoint")
            return False
        
        model_state = checkpoint['model']
        
        if dtype == "int8":
            for key in model_state:
                if isinstance(model_state[key], torch.Tensor):
                    model_state[key] = model_state[key].to(torch.int8)
        
        checkpoint['model'] = model_state
        checkpoint['quantized'] = True
        checkpoint['dtype'] = dtype
        
        torch.save(checkpoint, output_path)
        print(f"Quantized model saved to {output_path}")
        return True
    except Exception as e:
        print(f"Quantization failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Model Export Tools")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Export info
    info_parser = subparsers.add_parser("info", help="Export checkpoint info")
    info_parser.add_argument("checkpoint", help="Checkpoint path")
    info_parser.add_argument("-o", "--output", help="Output JSON path")
    
    # Quantize
    quant_parser = subparsers.add_parser("quantize", help="Quantize model")
    quant_parser.add_argument("input", help="Input checkpoint")
    quant_parser.add_argument("output", help="Output checkpoint")
    quant_parser.add_argument("--dtype", default="int8", choices=["int8", "int4"], help="Quantization dtype")
    
    args = parser.parse_args()
    
    if args.command == "info":
        export_checkpoint_info(args.checkpoint, args.output)
    elif args.command == "quantize":
        quantize_model(args.input, args.output, args.dtype)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
