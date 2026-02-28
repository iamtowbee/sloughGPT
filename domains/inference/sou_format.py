"""
.sou Format Parser for SloughGPT

Parses .sou model configuration files (inspired by Ollama Modelfile).
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum


class QuantizationType(Enum):
    """Quantization levels for .sou models."""
    F32 = "f32"
    F16 = "f16"
    Q8_0 = "q8_0"
    Q6_K = "q6_k"
    Q5 = "q5_K_M_k_m"
    Q5_K_S = "q5_k_s"
    Q4_K_M = "q4_k_m"
    Q4_K_S = "q4_k_s"
    Q4_0 = "q4_0"
    Q3_K_M = "q3_k_m"
    Q3_K_S = "q3_k_s"
    Q2_K = "q2_k"


@dataclass
class GenerationParameters:
    """Generation parameters for the model."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    repeat_penalty: float = 1.1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    mirostat: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    stop: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "repeat_penalty": self.repeat_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "mirostat": self.mirostat,
            "mirostat_tau": self.mirostat_tau,
            "mirostat_eta": self.mirostat_eta,
            "stop": self.stop,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationParameters":
        return cls(
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.9),
            top_k=data.get("top_k", 40),
            max_tokens=data.get("max_tokens", 2048),
            repeat_penalty=data.get("repeat_penalty", 1.1),
            presence_penalty=data.get("presence_penalty", 0.0),
            frequency_penalty=data.get("frequency_penalty", 0.0),
            mirostat=data.get("mirostat", 0),
            mirostat_tau=data.get("mirostat_tau", 5.0),
            mirostat_eta=data.get("mirostat_eta", 0.1),
            stop=data.get("stop", []),
        )


@dataclass
class ContextParameters:
    """Context parameters for the model."""
    num_ctx: int = 4096
    num_keep: int = 0
    num_thread: Optional[int] = None
    num_gpu: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"num_ctx": self.num_ctx, "num_keep": self.num_keep}
        if self.num_thread is not None:
            result["num_thread"] = self.num_thread
        if self.num_gpu is not None:
            result["num_gpu"] = self.num_gpu
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextParameters":
        return cls(
            num_ctx=data.get("num_ctx", 4096),
            num_keep=data.get("num_keep", 0),
            num_thread=data.get("num_thread"),
            num_gpu=data.get("num_gpu"),
        )


@dataclass
class PersonalityConfig:
    """Personality configuration for the model."""
    warmth: float = 0.5
    formality: float = 0.5
    creativity: float = 0.5
    empathy: float = 0.5
    patience: float = 0.5
    confidence: float = 0.5
    humor: float = 0.5
    directness: float = 0.5
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "warmth": self.warmth,
            "formality": self.formality,
            "creativity": self.creativity,
            "empathy": self.empathy,
            "patience": self.patience,
            "confidence": self.confidence,
            "humor": self.humor,
            "directness": self.directness,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "PersonalityConfig":
        return cls(
            warmth=data.get("warmth", 0.5),
            formality=data.get("formality", 0.5),
            creativity=data.get("creativity", 0.5),
            empathy=data.get("empathy", 0.5),
            patience=data.get("patience", 0.5),
            confidence=data.get("confidence", 0.5),
            humor=data.get("humor", 0.5),
            directness=data.get("directness", 0.5),
        )


@dataclass
class ACLConfig:
    """Access control configuration."""
    roles: List[str] = field(default_factory=lambda: ["admin", "user"])
    default_role: str = "user"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "roles": self.roles,
            "default_role": self.default_role,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ACLConfig":
        return cls(
            roles=data.get("roles", ["admin", "user"]),
            default_role=data.get("default_role", "user"),
        )


@dataclass
class WatermarkConfig:
    """Watermarking configuration."""
    enabled: bool = False
    strength: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "strength": self.strength,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WatermarkConfig":
        return cls(
            enabled=data.get("enabled", False),
            strength=data.get("strength", 0.5),
        )


@dataclass
class SouModelFile:
    """
    Complete .sou model file configuration.
    
    Inspired by Ollama Modelfile with extensions for:
    - Personality embeddings
    - Knowledge base (RAG)
    - Enterprise features (ACL, watermarking)
    """
    # Required
    from_model: str = ""
    
    # Generation parameters
    parameters: GenerationParameters = field(default_factory=GenerationParameters)
    context: ContextParameters = field(default_factory=ContextParameters)
    
    # Template and prompts
    template: Optional[str] = None
    system: Optional[str] = None
    
    # Personality
    personality: Optional[PersonalityConfig] = None
    
    # Knowledge base (RAG)
    knowledge_paths: List[str] = field(default_factory=list)
    retrieval_top_k: int = 5
    rerank: bool = False
    
    # LoRA adapters
    adapters: List[str] = field(default_factory=list)
    
    # Conversation history
    messages: List[Dict[str, str]] = field(default_factory=list)
    
    # License
    license: Optional[str] = None
    
    # Enterprise features
    acl: Optional[ACLConfig] = None
    watermark: WatermarkConfig = field(default_factory=WatermarkConfig)
    
    # Metadata
    metadata: Dict[str, str] = field(default_factory=dict)
    
    # Quantization
    quantization: Optional[QuantizationType] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "from": self.from_model,
            "parameters": self.parameters.to_dict(),
            "context": self.context.to_dict(),
        }
        
        if self.template:
            result["template"] = self.template
        if self.system:
            result["system"] = self.system
        if self.personality:
            result["personality"] = self.personality.to_dict()
        if self.knowledge_paths:
            result["knowledge"] = self.knowledge_paths
            result["retrieval_top_k"] = self.retrieval_top_k
            result["rerank"] = self.rerank
        if self.adapters:
            result["adapters"] = self.adapters
        if self.messages:
            result["messages"] = self.messages
        if self.license:
            result["license"] = self.license
        if self.acl:
            result["acl"] = self.acl.to_dict()
        if self.watermark.enabled:
            result["watermark"] = self.watermark.to_dict()
        if self.metadata:
            result["metadata"] = self.metadata
        if self.quantization:
            result["quantization"] = self.quantization.value
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SouModelFile":
        """Create from dictionary."""
        sou = cls(from_model=data.get("from", ""))
        
        if "parameters" in data:
            sou.parameters = GenerationParameters.from_dict(data["parameters"])
        if "context" in data:
            sou.context = ContextParameters.from_dict(data["context"])
        if "template" in data:
            sou.template = data["template"]
        if "system" in data:
            sou.system = data["system"]
        if "personality" in data:
            sou.personality = PersonalityConfig.from_dict(data["personality"])
        if "knowledge" in data:
            sou.knowledge_paths = data["knowledge"]
            sou.retrieval_top_k = data.get("retrieval_top_k", 5)
            sou.rerank = data.get("rerank", False)
        if "adapters" in data:
            sou.adapters = data["adapters"]
        if "messages" in data:
            sou.messages = data["messages"]
        if "license" in data:
            sou.license = data["license"]
        if "acl" in data:
            sou.acl = ACLConfig.from_dict(data["acl"])
        if "watermark" in data:
            sou.watermark = WatermarkConfig.from_dict(data["watermark"])
        if "metadata" in data:
            sou.metadata = data["metadata"]
        if "quantization" in data:
            sou.quantization = QuantizationType(data["quantization"])
            
        return sou


class SouParser:
    """
    Parser for .sou model files.
    
    Supports:
    - Basic instructions (FROM, PARAMETER, TEMPLATE, SYSTEM)
    - Multi-line blocks (PERSONALITY, KNOWLEDGE, ADAPTER, ACL, WATERMARK)
    - Comments (#)
    - Metadata
    """
    
    INSTRUCTION_PATTERN = re.compile(r'^(\w+)\s+(.*)$', re.MULTILINE)
    PARAMETER_PATTERN = re.compile(r'^PARAMETER\s+(\w+)\s+(.+)$', re.MULTILINE)
    BLOCK_START_PATTERN = re.compile(r'^(\w+)\s*$')
    BLOCK_END_PATTERN = re.compile(r'^\s*END\s*$')
    
    VALID_INSTRUCTIONS = {
        "FROM", "PARAMETER", "TEMPLATE", "SYSTEM", "PERSONALITY",
        "KNOWLEDGE", "ADAPTER", "LICENSE", "MESSAGE", "METADATA",
        "ACL", "WATERMARK"
    }
    
    def __init__(self):
        self.current_block: Optional[str] = None
        self.block_content: List[str] = []
    
    def parse(self, content: str) -> SouModelFile:
        """
        Parse .sou file content into SouModelFile.
        
        Args:
            content: Raw .sou file content
            
        Returns:
            SouModelFile: Parsed configuration
            
        Raises:
            ValueError: If parsing fails
        """
        # Remove comments
        lines = []
        for line in content.split('\n'):
            # Remove inline comments
            if '#' in line and not line.strip().startswith('#'):
                line = line.split('#')[0]
            lines.append(line)
        
        content = '\n'.join(lines)
        
        # Initialize result
        sou = SouModelFile()
        
        # Track current block
        current_block: Optional[str] = None
        block_content: List[str] = []
        
        for line in content.split('\n'):
            line = line.strip()
            
            if not line:
                continue
            
            # Check for block start
            if current_block is None:
                # Single-line instructions
                if ' ' in line:
                    instruction, value = line.split(' ', 1)
                    instruction = instruction.upper()
                    value = value.strip()
                    
                    if instruction not in self.VALID_INSTRUCTIONS:
                        continue
                    
                    self._parse_instruction(sou, instruction, value)
                elif line.upper() in {"PERSONALITY", "KNOWLEDGE", "ADAPTER", "ACL", "WATERMARK"}:
                    current_block = line.upper()
                    block_content = []
            else:
                # Inside a block
                if line.upper() == "END":
                    self._finalize_block(sou, current_block, block_content)
                    current_block = None
                    block_content = []
                else:
                    block_content.append(line)
        
        return sou
    
    def _parse_instruction(self, sou: SouModelFile, instruction: str, value: str):
        """Parse a single-line instruction."""
        value = value.strip()
        
        if instruction == "FROM":
            sou.from_model = value
        
        elif instruction == "PARAMETER":
            self._parse_parameter(sou, value)
        
        elif instruction == "TEMPLATE":
            sou.template = value
        
        elif instruction == "SYSTEM":
            sou.system = value
        
        elif instruction == "LICENSE":
            sou.license = value
        
        elif instruction == "MESSAGE":
            # Format: MESSAGE role content
            parts = value.split(' ', 1)
            if len(parts) == 2:
                role, content = parts
                sou.messages.append({"role": role, "content": content})
        
        elif instruction == "METADATA":
            # Format: METADATA key value
            parts = value.split(' ', 1)
            if len(parts) == 2:
                key, val = parts
                sou.metadata[key] = val
    
    def _parse_parameter(self, sou: SouModelFile, value: str):
        """Parse PARAMETER instruction."""
        parts = value.split(None, 1)
        if len(parts) != 2:
            return
        
        key, val = parts
        
        # Map to GenerationParameters
        param_map = {
            "temperature": ("temperature", float),
            "top_p": ("top_p", float),
            "top_k": ("top_k", int),
            "max_tokens": ("max_tokens", int),
            "repeat_penalty": ("repeat_penalty", float),
            "presence_penalty": ("presence_penalty", float),
            "frequency_penalty": ("frequency_penalty", float),
            "mirostat": ("mirostat", int),
            "mirostat_tau": ("mirostat_tau", float),
            "mirostat_eta": ("mirostat_eta", float),
            "num_ctx": ("num_ctx", int),
            "num_keep": ("num_keep", int),
            "num_thread": ("num_thread", int),
            "num_gpu": ("num_gpu", int),
        }
        
        if key in param_map:
            attr, dtype = param_map[key]
            try:
                setattr(sou.parameters, attr, dtype(val))
            except (ValueError, TypeError):
                pass
    
    def _finalize_block(self, sou: SouModelFile, block_type: str, content: List[str]):
        """Finalize a multi-line block."""
        block_text = '\n'.join(content)
        
        if block_type == "PERSONALITY":
            personality = PersonalityConfig()
            for line in content:
                if ' ' in line:
                    key, val = line.split(None, 1)
                    try:
                        setattr(personality, key, float(val))
                    except (ValueError, TypeError):
                        pass
            sou.personality = personality
        
        elif block_type == "KNOWLEDGE":
            sou.knowledge_paths = [line.strip() for line in content if line.strip()]
        
        elif block_type == "ADAPTER":
            sou.adapters = [line.strip() for line in content if line.strip()]
        
        elif block_type == "ACL":
            acl = ACLConfig()
            for line in content:
                if ' ' in line:
                    key, val = line.split(None, 1)
                    if key == "roles":
                        acl.roles = [r.strip() for r in val.split(',')]
                    elif key == "default_role":
                        acl.default_role = val.strip()
            sou.acl = acl
        
        elif block_type == "WATERMARK":
            watermark = WatermarkConfig()
            for line in content:
                if ' ' in line:
                    key, val = line.split(None, 1)
                    if key == "enabled":
                        watermark.enabled = val.lower() == "true"
                    elif key == "strength":
                        try:
                            watermark.strength = float(val)
                        except ValueError:
                            pass
            sou.watermark = watermark
    
    @classmethod
    def parse_file(cls, path: str) -> SouModelFile:
        """Parse a .sou file from path."""
        with open(path, 'r') as f:
            content = f.read()
        return cls().parse(content)


def create_default_sou(model_name: str = "llama3.2") -> SouModelFile:
    """Create a default .sou configuration."""
    return SouModelFile(
        from_model=model_name,
        parameters=GenerationParameters(),
        context=ContextParameters(),
    )


# =============================================================================
# CLI Integration
# =============================================================================

def main():
    """Demo CLI for parsing .sou files."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python sou_format.py <model.sou>")
        print("\nCreating example .sou file...")
        
        example = """FROM llama3.2
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

PERSONALITY
    warmth 0.8
    formality 0.3
    creativity 0.7
    END

SYSTEM You are a helpful AI assistant.

METADATA author "SloughGPT Team"
METADATA version "1.0.0"
"""
        print(example)
        print("\nParsing...")
        
        sou = SouParser().parse(example)
        print(f"From: {sou.from_model}")
        print(f"Temperature: {sou.parameters.temperature}")
        print(f"Context: {sou.context.num_ctx}")
        if sou.personality:
            print(f"Personality: {sou.personality.to_dict()}")
        print(f"Metadata: {sou.metadata}")
        
        return
    
    path = sys.argv[1]
    sou = SouParser().parse_file(path)
    print(sou.to_dict())


if __name__ == "__main__":
    main()


__all__ = [
    "SouModelFile",
    "SouParser",
    "GenerationParameters",
    "ContextParameters",
    "PersonalityConfig",
    "ACLConfig",
    "WatermarkConfig",
    "QuantizationType",
    "create_default_sou",
]
