"""Model export utilities for SloughGPT."""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ExportConfig:
    """Configuration for model export."""

    input_path: str
    output_path: str
    format: str = "safetensors"
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


def export_to_safetensors(
    model,
    output_path: str,
    metadata: Optional[Dict] = None,
    dtype: Optional[str] = None,
) -> str:
    """Export model weights to SafeTensors format (recommended default)."""
    from safetensors.torch import save_file

    state_dict = model.state_dict()
    if dtype == "bf16":
        state_dict = {k: v.to(dtype="bfloat16") for k, v in state_dict.items()}

    meta = metadata or {}
    meta["format"] = "safetensors"
    meta["format_version"] = "1.0"

    save_file(state_dict, output_path, metadata=meta)

    meta_path = output_path.replace(".safetensors", ".meta.json")
    with open(meta_path, "w") as f:
        import json
        json.dump(meta, f, indent=2)

    return output_path


def export_to_safetensors_bf16(
    model,
    output_path: str,
    metadata: Optional[Dict] = None,
) -> str:
    """Export model weights to SafeTensors with BF16 precision."""
    return export_to_safetensors(model, output_path, metadata, dtype="bf16")


def export_to_gguf(
    model,
    output_path: str,
    quantization: str = "Q4_0",
) -> str:
    """Export model to GGUF format with quantization.

    Requires: pip install gguf transformers
    """
    try:
        from gguf import GGUFWriter
    except ImportError:
        raise ImportError(
            "gguf not installed. Run: pip install gguf"
        )

    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("transformers not installed. Run: pip install transformers")

    state_dict = model.state_dict()

    with open(output_path, "wb") as f:
        writer = GGUFWriter(f, "sloughgpt", prepend_useless_ga=False)

        tensor_map = {
            "attn.c_attn.weight": "blk.{}.attn.c_attn.weight",
            "attn.c_attn.bias": "blk.{}.attn.c_attn.bias",
            "attn.c_proj.weight": "blk.{}.attn.c_proj.weight",
            "attn.c_proj.bias": "blk.{}.attn.c_proj.bias",
            "ln_1.weight": "blk.{}.attn.ln_1.weight",
            "ln_1.bias": "blk.{}.attn.ln_1.bias",
            "ln_2.weight": "blk.{}.attn.ln_2.weight",
            "ln_2.bias": "blk.{}.attn.ln_2.bias",
            "mlp.c_fc.weight": "blk.{}.ffn.c_fc.weight",
            "mlp.c_fc.bias": "blk.{}.ffn.c_fc.bias",
            "mlp.c_proj.weight": "blk.{}.ffn.c_proj.weight",
            "mlp.c_proj.bias": "blk.{}.ffn.c_proj.bias",
        }

        n_layer = 0
        for key in state_dict.keys():
            if "blocks." in key:
                layer_num = int(key.split("blocks.")[1].split(".")[0])
                n_layer = max(n_layer, layer_num + 1)

        writer.add_head_count(model.n_head if hasattr(model, "n_head") else 12)
        writer.add_n_positions(model.block_size if hasattr(model, "block_size") else 1024)
        writer.add_n_embd(model.n_embed if hasattr(model, "n_embed") else 768)
        writer.add_n_layer(n_layer or 12)
        writer.add_n_ffn(model.n_embed * 4 if hasattr(model, "n_embed") else 3072)
        writer.add_vocab_size(model.vocab_size if hasattr(model, "vocab_size") else 50257)

        for key, tensor in state_dict.items():
            mapped_key = key
            for src, dst in tensor_map.items():
                if src in key:
                    layer_idx = int(key.split("blk.")[1].split(".")[0]) if "blk." in key else 0
                    mapped_key = dst.format(layer_idx)
                    break
            writer.add_tensor(mapped_key, tensor)

        writer.write_header()
        writer.write_tensors()
        writer.finish()
        writer.close()

    return output_path


def export_to_torch(
    model,
    output_path: str,
    metadata: Optional[Dict] = None,
) -> str:
    """Export model to PyTorch format with metadata (legacy)."""
    import torch

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {"format": "torch"},
    }
    torch.save(checkpoint, output_path)
    return output_path


def export_to_sou(
    model,
    output_path: str,
    soul_profile = None,
    weights_only: bool = False,
) -> str:
    """Export model to .sou Soul Unit format."""
    from domains.inference.sou_format import export_to_sou as sou_export

    sou_export(
        model=model,
        output_path=output_path,
        soul_profile=soul_profile,
        weights_only=weights_only,
    )
    return output_path


def export_model(
    config: ExportConfig,
    model=None,
    tokenizer=None,
    example_input=None,
) -> Dict[str, str]:
    """Export a model to various formats."""
    results = {}

    formats = config.format.split(",") if "," in config.format else [config.format]

    for fmt in formats:
        fmt = fmt.strip().lower()
        output = None

        if fmt == "safetensors":
            output = config.output_path.replace(".pt", ".safetensors").replace(".pth", ".safetensors")
            export_to_safetensors(model, output, config.metadata)
            results["safetensors"] = output

        elif fmt == "safetensors_bf16":
            output = config.output_path.replace(".pt", "-bf16.safetensors").replace(".pth", "-bf16.safetensors")
            export_to_safetensors_bf16(model, output, config.metadata)
            results["safetensors_bf16"] = output

        elif fmt == "gguf":
            output = _gguf_path(config.output_path, config.quantization)
            export_to_gguf(model, output, config.quantization or "Q4_0")
            results["gguf"] = output

        elif fmt == "gguf_q4_0":
            output = config.output_path.replace(".pt", "-Q4_0.gguf")
            export_to_gguf(model, output, "Q4_0")
            results["gguf_q4_0"] = output

        elif fmt == "gguf_q4_1":
            output = config.output_path.replace(".pt", "-Q4_1.gguf")
            export_to_gguf(model, output, "Q4_1")
            results["gguf_q4_1"] = output

        elif fmt == "gguf_q5_0":
            output = config.output_path.replace(".pt", "-Q5_0.gguf")
            export_to_gguf(model, output, "Q5_0")
            results["gguf_q5_0"] = output

        elif fmt == "gguf_q5_1":
            output = config.output_path.replace(".pt", "-Q5_1.gguf")
            export_to_gguf(model, output, "Q5_1")
            results["gguf_q5_1"] = output

        elif fmt == "gguf_q8_0":
            output = config.output_path.replace(".pt", "-Q8_0.gguf")
            export_to_gguf(model, output, "Q8_0")
            results["gguf_q8_0"] = output

        elif fmt == "gguf_f16":
            output = config.output_path.replace(".pt", "-F16.gguf")
            export_to_gguf(model, output, "F16")
            results["gguf_f16"] = output

        elif fmt == "gguf_f32":
            output = config.output_path.replace(".pt", "-F32.gguf")
            export_to_gguf(model, output, "F32")
            results["gguf_f32"] = output

        elif fmt == "torch" or fmt == "pytorch":
            output = config.output_path.replace(".safetensors", ".pt")
            export_to_torch(model, output, config.metadata)
            results["torch"] = output

        elif fmt == "torchscript":
            output = config.output_path.replace(".pt", ".torchscript.pt")
            if example_input is None:
                print("Warning: example_input required for TorchScript export")
                continue
            export_to_torchscript(model, output, example_input)
            results["torchscript"] = output

        elif fmt == "onnx":
            output = config.output_path.replace(".pt", ".onnx")
            if example_input is None:
                print("Warning: example_input required for ONNX export")
                continue
            export_to_onnx(model, output, example_input)
            results["onnx"] = output

        elif fmt == "sou":
            from domains.inference.sou_format import create_soul_profile

            soul = create_soul_profile(
                name=config.metadata.get("name", Path(config.output_path).stem) if config.metadata else Path(config.output_path).stem,
                base_model="nanogpt",
                training_dataset=config.metadata.get("training_dataset", "") if config.metadata else "",
                epochs_trained=config.metadata.get("epochs_trained", 0) if config.metadata else 0,
                final_train_loss=config.metadata.get("final_train_loss", 0.0) if config.metadata else 0.0,
                final_val_loss=config.metadata.get("final_val_loss", 0.0) if config.metadata else 0.0,
                lineage="nanogpt",
                **({"lineage": config.metadata["lineage"]} if config.metadata and "lineage" in config.metadata else {}),
            )
            output = config.output_path.replace(".pt", ".sou").replace(".safetensors", ".sou")
            export_to_sou(model, output, soul_profile=soul)
            results["sou"] = output

        elif fmt == "all":
            output = config.output_path.replace(".pt", ".safetensors")
            export_to_safetensors(model, output, config.metadata)
            results["safetensors"] = output

            output = config.output_path.replace(".pt", ".torch")
            export_to_torch(model, output, config.metadata)
            results["torch"] = output

            if example_input is not None:
                output = config.output_path.replace(".pt", ".onnx")
                export_to_onnx(model, output, example_input)
                results["onnx"] = output

            try:
                output = config.output_path.replace(".pt", "-Q4_0.gguf")
                export_to_gguf(model, output, "Q4_0")
                results["gguf_q4_0"] = output
            except ImportError:
                pass

    if tokenizer and config.include_tokenizer:
        tokenizer_path = Path(config.output_path).parent / "tokenizer"
        tokenizer.save_pretrained(str(tokenizer_path))
        results["tokenizer"] = str(tokenizer_path)

    return results


def _gguf_path(output_path: str, quantization: Optional[str]) -> str:
    """Get GGUF output path with quantization suffix."""
    q = (quantization or "Q4_0").upper()
    stem = Path(output_path).stem
    parent = Path(output_path).parent
    return str(parent / f"{stem}-{q}.gguf")


def list_export_formats() -> Dict[str, str]:
    """List supported export formats."""
    return {
        "safetensors": "SafeTensors weights (.safetensors) - RECOMMENDED default",
        "safetensors_bf16": "SafeTensors with BF16 precision (-bf16.safetensors)",
        "gguf": "GGUF quantized model (.gguf) - use with --quantization",
        "gguf_q4_0": "GGUF 4-bit quantization (.gguf) - smallest, fastest",
        "gguf_q4_1": "GGUF 4-bit with improved quality (.gguf)",
        "gguf_q5_0": "GGUF 5-bit quantization (.gguf)",
        "gguf_q5_1": "GGUF 5-bit with improved quality (.gguf)",
        "gguf_q8_0": "GGUF 8-bit quantization (.gguf) - balanced",
        "gguf_f16": "GGUF 16-bit float (.gguf)",
        "gguf_f32": "GGUF 32-bit float (.gguf) - full precision",
        "torch": "PyTorch checkpoint (.pt) - legacy format",
        "torchscript": "TorchScript traced model (.torchscript.pt)",
        "onnx": "ONNX model (.onnx) - cross-platform",
        "sou": "SloughGPT Soul Unit (.sou) - self-contained model + living soul profile",
        "all": "Export all formats at once",
    }


__all__ = [
    "ExportConfig",
    "export_model",
    "export_to_torch",
    "export_to_safetensors",
    "export_to_safetensors_bf16",
    "export_to_gguf",
    "export_to_torchscript",
    "export_to_onnx",
    "export_to_sou",
    "list_export_formats",
]
