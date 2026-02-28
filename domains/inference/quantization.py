"""
Quantization Support for .sou Format

Supports various quantization levels for efficient inference.
Based on GGML/GGUF quantization schemes.
"""

import struct
import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional
import numpy as np


class QuantizationType(Enum):
    """Quantization types supported by .sou format."""
    F32 = ("f32", 32, 4.0)
    F16 = ("f16", 16, 2.0)
    Q8_0 = ("q8_0", 8, 1.0)
    Q6_K = ("q6_k", 6, 0.75)
    Q5_K_M = ("q5_k_m", 5, 0.66)
    Q5_K_S = ("q5_k_s", 5, 0.62)
    Q4_K_M = ("q4_k_m", 4, 0.57)
    Q4_K_S = ("q4_k_s", 4, 0.53)
    Q4_0 = ("q4_0", 4, 0.5)
    Q3_K_M = ("q3_k_m", 3, 0.44)
    Q3_K_S = ("q3_k_s", 3, 0.41)
    Q2_K = ("q2_k", 2, 0.29)
    
    def __init__(self, name: str, bits: float, bytes_per_param: float):
        self.name = name
        self.bits = bits
        self.bytes_per_param = bytes_per_param
    
    @classmethod
    def from_string(cls, s: str) -> "QuantizationType":
        """Parse quantization type from string."""
        s = s.lower().strip()
        for qtype in cls:
            if qtype.name == s or qtype.value == s:
                return qtype
        raise ValueError(f"Unknown quantization type: {s}")


@dataclass
class QuantizationInfo:
    """Information about quantization."""
    qtype: QuantizationType
    original_size: int
    quantized_size: int
    compression_ratio: float
    
    @property
    def memory_savings(self) -> float:
        """Return memory savings as percentage."""
        return (1 - self.compression_ratio) * 100


class Quantizer:
    """
    Quantizer for model weights.
    
    Implements various quantization schemes.
    """
    
    @staticmethod
    def quantize_fp32_to_q4(data: np.ndarray) -> Tuple[bytes, List[float]]:
        """
        Quantize FP32 to Q4_0 format.
        
        Uses simple per-tensor quantization:
        - Find max absolute value
        - Normalize to [-8, 8] range
        - Round to 4-bit integers
        """
        if data.size == 0:
            return b"", []
        
        # Find scale
        max_val = np.max(np.abs(data))
        if max_val == 0:
            scale = 1.0
        else:
            scale = max_val / 8.0
        
        # Quantize
        normalized = data / scale
        quantized = np.round(normalized).clip(-8, 7).astype(np.int8)
        
        # Pack 2 4-bit values into 1 byte
        packed = bytearray()
        scales = [float(scale)]
        
        for i in range(0, quantized.size - 1, 2):
            low = quantized[i] & 0x0F
            high = (quantized[i + 1] & 0x0F) << 4
            packed.append(low | high)
        
        # Handle odd length
        if quantized.size % 2 == 1:
            packed.append(quantized[-1] & 0x0F)
        
        return bytes(packed), scales
    
    @staticmethod
    def quantize_fp32_to_q8(data: np.ndarray) -> Tuple[bytes, List[float]]:
        """
        Quantize FP32 to Q8_0 format.
        
        Uses per-tensor quantization:
        - Find max absolute value
        - Normalize to [-127, 127] range
        - Round to 8-bit integers
        """
        if data.size == 0:
            return b"", []
        
        # Find scale
        max_val = np.max(np.abs(data))
        if max_val == 0:
            scale = 1.0
        else:
            scale = max_val / 127.0
        
        # Quantize
        normalized = data / scale
        quantized = np.round(normalized).clip(-127, 127).astype(np.int8)
        
        # Pack with scale
        packed = bytearray(quantized.tobytes())
        packed.extend(struct.pack('f', scale))
        
        return bytes(packed), [float(scale)]
    
    @staticmethod
    def dequantize_q4(packed: bytes, scales: List[float], size: int) -> np.ndarray:
        """Dequantize Q4 to FP32."""
        if not packed or not scales:
            return np.zeros(size, dtype=np.float32)
        
        scale = scales[0]
        data = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))
        
        # Convert from 4-bit to 8-bit
        result = np.zeros(size, dtype=np.float32)
        for i in range(min(size, len(data) // 4)):
            val = data[i * 4:(i + 1) * 4]
            signed = (val[0] | (val[1] << 4) | (val[2] << 8) | (val[3] << 12))
            if signed >= 8:
                signed -= 16
            result[i] = signed * scale
        
        return result
    
    @staticmethod
    def quantize(data: np.ndarray, qtype: QuantizationType) -> Tuple[bytes, List[float]]:
        """Quantize data to specified type."""
        if qtype == QuantizationType.F32:
            return data.tobytes(), [1.0]
        elif qtype == QuantizationType.F16:
            return data.astype(np.float16).tobytes(), [1.0]
        elif qtype == QuantizationType.Q8_0:
            return Quantizer.quantize_fp32_to_q8(data)
        elif qtype == QuantizationType.Q4_0:
            return Quantizer.quantize_fp32_to_q4(data)
        else:
            raise NotImplementedError(f"Quantization {qtype} not implemented")
    
    @staticmethod
    def get_quantized_size(original_size: int, qtype: QuantizationType) -> int:
        """Calculate quantized size in bytes."""
        return int(original_size * qtype.bytes_per_param)


class SouModelQuantizer:
    """
    Quantizer for .sou model files.
    """
    
    SUPPORTED_TYPES = {
        ".safetensors": "safetensors",
        ".bin": "bin",
        ".pt": "pytorch",
        ".pth": "pytorch",
    }
    
    @staticmethod
    def quantize_model(
        input_path: str,
        output_path: str,
        qtype: QuantizationType = QuantizationType.Q4_K_M,
    ) -> QuantizationInfo:
        """
        Quantize a model file.
        
        Args:
            input_path: Path to input model
            output_path: Path for quantized output
            qtype: Target quantization type
            
        Returns:
            QuantizationInfo with compression statistics
        """
        import os
        
        # Get file size
        original_size = os.path.getsize(input_path)
        
        # For now, create a placeholder
        # In production, this would load the model and quantize
        quantized_size = Quantizer.get_quantized_size(original_size, qtype)
        
        info = QuantizationInfo(
            qtype=qtype,
            original_size=original_size,
            quantized_size=quantized_size,
            compression_ratio=quantized_size / original_size,
        )
        
        return info
    
    @staticmethod
    def get_memory_requirements(
        num_parameters: int,
        qtype: QuantizationType,
        include_kv_cache: bool = True,
        kv_cache_size: int = 4096,
    ) -> dict:
        """
        Calculate memory requirements for a model.
        
        Args:
            num_parameters: Number of model parameters
            qtype: Quantization type
            include_kv_cache: Include KV cache memory
            kv_cache_size: KV cache size in tokens
            
        Returns:
            Dictionary with memory requirements
        """
        # Model weights
        weights_bytes = int(num_parameters * qtype.bytes_per_param)
        
        # KV cache (2 * layers * hidden * 2 * bytes_per_param)
        # Assuming 32 layers, 4096 hidden
        kv_bytes = 0
        if include_kv_cache:
            layers = 32
            hidden = 4096
            kv_bytes = 2 * layers * hidden * 2 * 2  # Q and K, FP16
        
        total_bytes = weights_bytes + kv_bytes
        
        return {
            "weights_mb": weights_bytes / (1024 * 1024),
            "kv_cache_mb": kv_bytes / (1024 * 1024),
            "total_mb": total_bytes / (1024 * 1024),
            "total_gb": total_bytes / (1024 * 1024 * 1024),
            "num_parameters": num_parameters,
            "quantization": qtype.name,
        }


# =============================================================================
# Quantization Presets
# =============================================================================

QUANTIZATION_PRESETS = {
    "fast": QuantizationType.Q4_0,
    "balanced": QuantizationType.Q4_K_M,
    "quality": QuantizationType.Q8_0,
    "best": QuantizationType.F16,
    "fp32": QuantizationType.F32,
}


def get_quantization_preset(name: str) -> QuantizationType:
    """Get quantization type by preset name."""
    return QUANTIZATION_PRESETS.get(name.lower(), QuantizationType.Q4_K_M)


# =============================================================================
# CLI Functions
# =============================================================================

def print_quantization_info(info: QuantizationInfo):
    """Print quantization info in readable format."""
    print(f"Quantization Type: {info.qtype.name}")
    print(f"Original Size: {info.original_size / (1024*1024):.2f} MB")
    print(f"Quantized Size: {info.quantized_size / (1024*1024):.2f} MB")
    print(f"Compression Ratio: {info.compression_ratio:.2%}")
    print(f"Memory Savings: {info.memory_savings:.1f}%")


def main():
    """Demo CLI."""
    import sys
    
    # Test quantization
    print("=== Quantization Demo ===\n")
    
    # Create sample data
    data = np.random.randn(1000).astype(np.float32)
    
    print(f"Original data size: {data.nbytes} bytes")
    
    # Quantize to Q4
    packed, scales = Quantizer.quantize_fp32_to_q4(data)
    print(f"Q4_0 size: {len(packed)} bytes (theoretical: {1000 * 0.5})")
    
    # Quantize to Q8
    packed, scales = Quantizer.quantize_fp32_to_q8(data)
    print(f"Q8_0 size: {len(packed)} bytes (theoretical: {1000 * 1.0})")
    
    print("\n=== Memory Requirements Demo ===\n")
    
    # 7B model
    info = SouModelQuantizer.get_memory_requirements(
        num_parameters=7_000_000_000,
        qtype=QuantizationType.Q4_K_M,
    )
    print(f"7B Model (Q4_K_M):")
    print(f"  Weights: {info['weights_mb']:.0f} MB")
    print(f"  Total (with KV): {info['total_mb']:.0f} MB")
    
    info = SouModelQuantizer.get_memory_requirements(
        num_parameters=7_000_000_000,
        qtype=QuantizationType.F16,
    )
    print(f"\n7B Model (F16):")
    print(f"  Weights: {info['weights_mb']:.0f} MB")
    print(f"  Total (with KV): {info['total_mb']:.0f} MB")


if __name__ == "__main__":
    main()


__all__ = [
    "QuantizationType",
    "QuantizationInfo",
    "Quantizer",
    "SouModelQuantizer",
    "QUANTIZATION_PRESETS",
    "get_quantization_preset",
]
