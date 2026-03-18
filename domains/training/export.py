"""Model export utilities for SloughGPT."""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import json


@dataclass
class ExportConfig:
    """Configuration for model export."""

    input_path: str
    output_path: str
    format: str = "torch"
    quantization: Optional[str] = None
    include_tokenizer: bool = True
    metadata: Optional[Dict[str, Any]] = None


def export_to_torchscript(
    model,
    output_path: str,
    example_input: Optional[Any] = None,
) -> str:
    """Export model to TorchScript format."""
    import torch

    model.eval()

    if example_input is not None:
        traced = torch.jit.trace(model, example_input)
    else:
        traced = torch.jit.script(model)

    traced.save(output_path)
    return output_path


def export_to_onnx(
    model,
    output_path: str,
    example_input: Any,
    input_names: list = ["input"],
    output_names: list = ["output"],
    dynamic_axes: Optional[Dict[str, Any]] = None,
) -> str:
    """Export model to ONNX format."""
    import torch

    model.eval()
    torch.onnx.export(
        model,
        example_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes or {},
        opset_version=14,
    )
    return output_path


def export_to_sou(
    model,
    output_path: str,
    from_model: str = "sloughgpt",
    temperature: float = 0.7,
    metadata: Optional[Dict] = None,
) -> str:
    """Export model to .sou format."""
    from domains.inference.sou_format import export_to_sou as sou_export

    sou_export(
        model=model,
        output_path=output_path,
        from_model=from_model,
        metadata=metadata,
    )
    return output_path


def export_to_safetensors(model, output_path: str) -> str:
    """Export model weights to SafeTensors format."""
    from safetensors.torch import save_file

    state_dict = model.state_dict()
    save_file(state_dict, output_path)
    return output_path


def export_to_torch(model, output_path: str, metadata: Optional[Dict] = None) -> str:
    """Export model to PyTorch format with metadata."""
    import torch

    checkpoint = {"model": model.state_dict(), "metadata": metadata or {}}
    torch.save(checkpoint, output_path)
    return output_path


def export_model(
    config: ExportConfig,
    model=None,
    tokenizer=None,
    example_input=None,
) -> Dict[str, str]:
    """Export a model to various formats."""
    import torch

    results = {}

    formats = config.format.split(",") if "," in config.format else [config.format]

    for fmt in formats:
        fmt = fmt.strip().lower()

        if fmt == "torch":
            output = config.output_path.replace(".pt", ".pt")
            export_to_torch(model, output, config.metadata)
            results["torch"] = output

        elif fmt == "torchscript":
            output = config.output_path.replace(".pt", ".torchscript.pt")
            export_to_torchscript(model, output, example_input)
            results["torchscript"] = output

        elif fmt == "onnx":
            output = config.output_path.replace(".pt", ".onnx")
            if example_input is None:
                print("Warning: example_input required for ONNX export")
                continue
            export_to_onnx(model, output, example_input)
            results["onnx"] = output

        elif fmt == "safetensors":
            output = config.output_path.replace(".pt", ".safetensors")
            export_to_safetensors(model, output)
            results["safetensors"] = output

        elif fmt == "sou":
            output = config.output_path.replace(".pt", ".sou")
            export_to_sou(model, output, metadata=config.metadata)
            results["sou"] = output

    if tokenizer and config.include_tokenizer:
        tokenizer_path = config.output_path.replace(".pt", "_tokenizer.json")
        tokenizer.save_pretrained(os.path.dirname(tokenizer_path))
        results["tokenizer"] = os.path.dirname(tokenizer_path)

    return results


def list_export_formats() -> Dict[str, str]:
    """List supported export formats."""
    return {
        "torch": "PyTorch checkpoint (.pt)",
        "torchscript": "TorchScript traced model (.torchscript.pt)",
        "onnx": "ONNX model (.onnx)",
        "safetensors": "SafeTensors weights (.safetensors)",
        "sou": "SloughGPT optimized format (.sou)",
    }


__all__ = [
    "ExportConfig",
    "export_model",
    "export_to_torch",
    "export_to_torchscript",
    "export_to_onnx",
    "export_to_safetensors",
    "export_to_sou",
    "list_export_formats",
]
