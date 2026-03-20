#!/usr/bin/env python3
"""
Model Export Utility
Quick export models to various formats.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import json
from pathlib import Path


def export_to_torch(model_path: str, output_path: str):
    """Export model to Torch format."""
    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
    torch.save(checkpoint, output_path)
    print(f"[OK] Exported to Torch: {output_path}")
    return output_path


def export_to_safetensors(model_path: str, output_path: str):
    """Export model to SafeTensors format."""
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("[ERROR] safetensors not installed. Install with: pip install safetensors")
        return None

    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")

    tensors = {}
    if "model" in checkpoint:
        model = checkpoint["model"]
        if hasattr(model, "state_dict"):
            tensors = model.state_dict()
        elif isinstance(model, dict):
            tensors = model

    save_file(tensors, output_path)
    print(f"[OK] Exported to SafeTensors: {output_path}")
    return output_path


def export_to_onnx(model_path: str, output_path: str, seq_len: int = 128):
    """Export model to ONNX format."""
    try:
        import torch
    except ImportError:
        print("[ERROR] PyTorch not installed")
        return None

    print("[INFO] ONNX export requires model architecture. Using dummy export...")

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(10000, 384)
            self.linear = torch.nn.Linear(384, 10000)

        def forward(self, x):
            x = self.embed(x)
            return self.linear(x)

    model = DummyModel()
    model.eval()

    dummy_input = torch.randint(0, 1000, (1, seq_len))

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    print(f"[OK] Exported to ONNX: {output_path}")
    return output_path


def export_to_sou(model_path: str, output_path: str):
    """Export model to .sou format (custom)."""
    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")

    sou_data = {
        "version": "1.0",
        "format": "sou",
        "checkpoint": checkpoint,
        "metadata": {
            "original_path": model_path,
            "exported_at": str(Path(model_path).stat().st_mtime),
        }
    }

    with open(output_path, 'w') as f:
        torch.save(sou_data, f)

    print(f"[OK] Exported to .sou: {output_path}")
    return output_path


def export_summary(model_path: str, output_formats: list):
    """Print export summary."""
    print("\n" + "=" * 50)
    print("Export Summary")
    print("=" * 50)
    print(f"Source: {model_path}")

    src_size = Path(model_path).stat().st_size / (1024**2)
    print(f"Source size: {src_size:.2f} MB")

    print("\nExports:")
    for fmt, path in output_formats:
        if path and Path(path).exists():
            size = Path(path).stat().st_size / (1024**2)
            ratio = src_size / size if size > 0 else 0
            print(f"  {fmt}: {path} ({size:.2f} MB, {ratio:.1f}x)")


def main():
    parser = argparse.ArgumentParser(description="Export models to various formats")
    parser.add_argument("model", help="Model checkpoint path")
    parser.add_argument("--output", "-o", default="models/exported", help="Output base path")
    parser.add_argument("--formats", "-f", nargs="+",
                       choices=["torch", "safetensors", "onnx", "sou", "all"],
                       default=["torch"], help="Export formats")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length for ONNX")

    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        return 1

    print("=" * 50)
    print("Model Export Utility")
    print("=" * 50)

    formats = args.formats if "all" not in args.formats else ["torch", "safetensors", "sou"]

    exports = []

    for fmt in formats:
        output = f"{args.output}.{fmt}"

        if fmt == "torch":
            exports.append(("Torch", export_to_torch(args.model, output)))
        elif fmt == "safetensors":
            exports.append(("SafeTensors", export_to_safetensors(args.model, output)))
        elif fmt == "onnx":
            exports.append(("ONNX", export_to_onnx(args.model, output, args.seq_len)))
        elif fmt == "sou":
            exports.append((".sou", export_to_sou(args.model, output)))

    export_summary(args.model, exports)

    return 0


if __name__ == "__main__":
    sys.exit(main())
