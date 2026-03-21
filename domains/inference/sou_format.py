"""
.sou - SloughGPT Soul Unit Format
The living identity format for trained AI models.

Every SloughGPT model has a soul. The .sou format captures:
- Model identity and lineage
- Personality and behavioral DNA
- Cognitive and emotional signature
- Training memory and knowledge patterns
- Deployment configuration

.sou files are self-contained: weights + soul + config.

Trademark (c) 2026 SloughGPT. All rights reserved.
"""

import os
import json
import struct
import hashlib
import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path


SOU_MAGIC = b"SOUL"
SOU_VERSION = 2
SOU_TRADEMARK = "SloughGPT Soul Unit (.sou) - Trademark (c) 2026 SloughGPT"


@dataclass
class GenerationParams:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    repeat_penalty: float = 1.1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
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
            "stop": self.stop,
        }


@dataclass
class ContextParams:
    context_window: int = 4096
    num_ctx: int = 4096
    num_gpu: int = 0
    num_thread: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PersonalityCore:
    warmth: float = 0.5
    creativity: float = 0.5
    empathy: float = 0.5
    formality: float = 0.5
    humor: float = 0.5
    patience: float = 0.5
    confidence: float = 0.5
    curiosity: float = 0.5
    directness: float = 0.5
    optimism: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class BehavioralTraits:
    speaking_style: str = "conversational"
    reasoning_approach: str = "balanced"
    explanation_depth: str = "moderate"
    emotional_expressiveness: float = 0.5
    formality_dynamic: float = 0.5
    interruption_tolerance: float = 0.5
    follow_up_tendency: float = 0.5
    clarification_seeking: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CognitiveSignature:
    pattern_recognition: float = 0.5
    long_context_handling: float = 0.5
    abstract_reasoning: float = 0.5
    factual_precision: float = 0.5
    creative_divergence: float = 0.5
    systematic_planning: float = 0.5
    metacognitive_awareness: float = 0.5
    learning_adaptability: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class EmotionalRange:
    empathy_depth: float = 0.5
    mood_responsiveness: float = 0.5
    tone_flexibility: float = 0.5
    sentiment_awareness: float = 0.5
    distress_handling: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class SoulProfile:
    name: str
    version: str = "1.0.0"
    tagline: str = ""
    description: str = ""
    created_by: str = "SloughGPT Training Pipeline"
    born_at: str = ""
    lineage: str = "nanogpt"
    base_model: str = ""
    training_dataset: str = ""
    epochs_trained: int = 0
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    dataset_signature: str = ""
    personality: PersonalityCore = field(default_factory=PersonalityCore)
    behavior: BehavioralTraits = field(default_factory=BehavioralTraits)
    cognition: CognitiveSignature = field(default_factory=CognitiveSignature)
    emotion: EmotionalRange = field(default_factory=EmotionalRange)
    generation: GenerationParams = field(default_factory=GenerationParams)
    context: ContextParams = field(default_factory=ContextParams)
    system_prompt: str = ""
    sample_dialogue: List[Dict[str, str]] = field(default_factory=list)
    lora_adapters: List[str] = field(default_factory=list)
    quantization: str = "none"
    acl_users: List[str] = field(default_factory=list)
    watermark_enabled: bool = False
    watermark_strength: float = 0.1
    tags: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    integrity_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.born_at:
            self.born_at = datetime.datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "version": self.version,
            "tagline": self.tagline,
            "description": self.description,
            "created_by": self.created_by,
            "born_at": self.born_at,
            "lineage": self.lineage,
            "base_model": self.base_model,
            "training_dataset": self.training_dataset,
            "epochs_trained": self.epochs_trained,
            "final_train_loss": self.final_train_loss,
            "final_val_loss": self.final_val_loss,
            "dataset_signature": self.dataset_signature,
            "personality": self.personality.to_dict(),
            "behavior": self.behavior.to_dict(),
            "cognition": self.cognition.to_dict(),
            "emotion": self.emotion.to_dict(),
            "generation": self.generation.to_dict(),
            "context": self.context.to_dict(),
            "system_prompt": self.system_prompt,
            "sample_dialogue": self.sample_dialogue,
            "lora_adapters": self.lora_adapters,
            "quantization": self.quantization,
            "acl_users": self.acl_users,
            "watermark_enabled": self.watermark_enabled,
            "watermark_strength": self.watermark_strength,
            "tags": self.tags,
            "certifications": self.certifications,
            "integrity_hash": self.integrity_hash,
            "metadata": self.metadata,
        }
        return d

    def compute_hash(self) -> str:
        data = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_sou_string(self) -> str:
        lines = [
            "# SloughGPT Soul Unit",
            f"# {SOU_TRADEMARK}",
            "# This file contains the living identity of an AI model.",
            "",
            f"SOUL {self.name}",
            f"VERSION {self.version}",
            f"LINEAGE {self.lineage}",
            f"BORN {self.born_at}",
            f"CREATED_BY {self.created_by}",
            "",
        ]

        if self.tagline:
            lines.append(f"TAGLINE {self.tagline}")
            lines.append("")

        if self.description:
            lines.append(f"DESCRIPTION {self.description}")
            lines.append("")

        lines.extend(["# Identity", f"BASEMODEL {self.base_model}"])
        if self.training_dataset:
            lines.append(f"TRAINING_DATA {self.training_dataset}")
        if self.dataset_signature:
            lines.append(f"DATA_SIGNATURE {self.dataset_signature}")
        lines.append("")

        lines.extend(["# Generation Parameters", "PARAMETER"])
        for k, v in self.generation.to_dict().items():
            if k != "stop":
                lines.append(f"    {k} {v}")
        if self.generation.stop:
            for s in self.generation.stop:
                lines.append(f"    stop {s}")
        lines.append("")

        lines.extend(["# Context", "CONTEXT"])
        for k, v in self.context.to_dict().items():
            lines.append(f"    {k} {v}")
        lines.append("")

        lines.append("# Personality Core (soul signature)")
        lines.append("PERSONALITY")
        for k, v in self.personality.to_dict().items():
            lines.append(f"    {k} {v}")
        lines.append("    END")
        lines.append("")

        lines.append("# Behavioral Traits")
        lines.append("BEHAVIOR")
        for k, v in self.behavior.to_dict().items():
            lines.append(f"    {k} {v}")
        lines.append("    END")
        lines.append("")

        lines.append("# Cognitive Signature")
        lines.append("COGNITION")
        for k, v in self.cognition.to_dict().items():
            lines.append(f"    {k} {v}")
        lines.append("    END")
        lines.append("")

        lines.append("# Emotional Range")
        lines.append("EMOTION")
        for k, v in self.emotion.to_dict().items():
            lines.append(f"    {k} {v}")
        lines.append("    END")
        lines.append("")

        if self.system_prompt:
            lines.append(f"SYSTEM {self.system_prompt}")
            lines.append("")

        if self.sample_dialogue:
            for msg in self.sample_dialogue:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                lines.append(f"MESSAGE {role} {content}")
            lines.append("")

        if self.lora_adapters:
            lines.append("ADAPTER")
            for adapter in self.lora_adapters:
                lines.append(f"    {adapter}")
            lines.append("    END")
            lines.append("")

        if self.tags:
            lines.append("TAG " + ",".join(self.tags))
            lines.append("")

        if self.training_dataset:
            lines.append(f"METADATA epochs_trained {self.epochs_trained}")
            lines.append(f"METADATA final_train_loss {self.final_train_loss}")
            lines.append(f"METADATA final_val_loss {self.final_val_loss}")

        if self.certifications:
            for cert in self.certifications:
                lines.append(f"CERTIFICATION {cert}")

        lines.append("")
        return "\n".join(lines)


class SouParser:
    @staticmethod
    def parse(content: str) -> SoulProfile:
        lines = content.strip().split("\n")

        sp = SoulProfile(name="unknown")
        section = None
        current_block = {}
        dialogue = []

        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("SOUL "):
                sp.name = line[5:].strip()
            elif line.startswith("VERSION "):
                sp.version = line[8:].strip()
            elif line.startswith("LINEAGE "):
                sp.lineage = line[8:].strip()
            elif line.startswith("BORN "):
                sp.born_at = line[5:].strip()
            elif line.startswith("CREATED_BY "):
                sp.created_by = line[11:].strip()
            elif line.startswith("TAGLINE "):
                sp.tagline = line[8:].strip()
            elif line.startswith("DESCRIPTION "):
                sp.description = line[12:].strip()
            elif line.startswith("BASEMODEL "):
                sp.base_model = line[10:].strip()
            elif line.startswith("TRAINING_DATA "):
                sp.training_dataset = line[14:].strip()
            elif line.startswith("DATA_SIGNATURE "):
                sp.dataset_signature = line[15:].strip()
            elif line.startswith("SYSTEM "):
                sp.system_prompt = line[7:].strip()
            elif line.startswith("PARAMETER"):
                section = "parameter"
                current_block = {}
            elif line.startswith("CONTEXT"):
                section = "context"
                current_block = {}
            elif line.startswith("PERSONALITY"):
                section = "personality"
                current_block = {}
            elif line.startswith("BEHAVIOR"):
                section = "behavior"
                current_block = {}
            elif line.startswith("COGNITION"):
                section = "cognition"
                current_block = {}
            elif line.startswith("EMOTION"):
                section = "emotion"
                current_block = {}
            elif line.startswith("MESSAGE "):
                parts = line[8:].split(" ", 1)
                if len(parts) == 2:
                    dialogue.append({"role": parts[0], "content": parts[1]})
            elif line.startswith("ADAPTER"):
                section = "adapter"
                current_block = []
            elif line == "END":
                if section == "personality":
                    sp.personality = PersonalityCore(**current_block)
                elif section == "behavior":
                    sp.behavior = BehavioralTraits(**{k: v for k, v in current_block.items() if not isinstance(v, str) or v.replace(".", "1", 1).isdigit() is False})
                    for k, v in current_block.items():
                        if isinstance(v, str) and not v.replace(".", "1", 1).replace("e-", "", 1).isdigit():
                            setattr(sp.behavior, k, v)
                        else:
                            try:
                                setattr(sp.behavior, k, float(v))
                            except (ValueError, TypeError):
                                setattr(sp.behavior, k, v)
                elif section == "cognition":
                    sp.cognition = CognitiveSignature(**{k: float(v) for k, v in current_block.items()})
                elif section == "emotion":
                    sp.emotion = EmotionalRange(**{k: float(v) for k, v in current_block.items()})
                elif section == "adapter":
                    sp.lora_adapters = current_block
                section = None
                current_block = {}
            elif line.startswith("TAG "):
                sp.tags = [t.strip() for t in line[4:].split(",")]
            elif line.startswith("METADATA "):
                parts = line[9:].split(" ", 1)
                if len(parts) == 2:
                    k, v = parts
                    if k == "epochs_trained":
                        sp.epochs_trained = int(v)
                    elif k in ("final_train_loss", "final_val_loss"):
                        setattr(sp, k, float(v))
                    else:
                        sp.metadata[k] = v
            elif line.startswith("CERTIFICATION "):
                sp.certifications.append(line[14:].strip())
            elif section == "parameter":
                parts = line.split()
                if len(parts) == 2:
                    k, v = parts
                    if k == "stop":
                        current_block.setdefault("stop", []).append(v)
                    else:
                        try:
                            current_block[k] = float(v) if "." in v else int(v)
                        except ValueError:
                            current_block[k] = v
                sp.generation = GenerationParams(**{k: v for k, v in current_block.items() if k != "stop"})
                if "stop" in current_block:
                    sp.generation.stop = current_block["stop"]
            elif section == "context":
                parts = line.split()
                if len(parts) == 2:
                    k, v = parts
                    try:
                        current_block[k] = int(v)
                    except ValueError:
                        current_block[k] = v
                sp.context = ContextParams(**current_block)
            elif section in ("personality", "cognition", "emotion"):
                parts = line.split()
                if len(parts) == 2:
                    k, v = parts
                    try:
                        current_block[k] = float(v)
                    except ValueError:
                        current_block[k] = v
            elif section == "adapter":
                current_block.append(line)
            elif line.startswith("QUANTIZATION "):
                sp.quantization = line[13:].strip()

        if dialogue:
            sp.sample_dialogue = dialogue

        return sp

    @staticmethod
    def load(path: str) -> SoulProfile:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return SouParser.parse(content)

    @staticmethod
    def save(sou: SoulProfile, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(sou.to_sou_string())


def create_soul_profile(
    name: str,
    base_model: str = "nanogpt",
    training_dataset: str = "",
    epochs_trained: int = 0,
    final_train_loss: float = 0.0,
    final_val_loss: float = 0.0,
    personality: Optional[PersonalityCore] = None,
    generation: Optional[GenerationParams] = None,
    system_prompt: str = "",
    tags: Optional[List[str]] = None,
    lineage: str = "nanogpt",
    dataset_signature: str = "",
    **kwargs,
) -> SoulProfile:
    """Create a Soul Profile from training results."""
    sp = SoulProfile(
        name=name,
        base_model=base_model,
        training_dataset=training_dataset,
        epochs_trained=epochs_trained,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        lineage=lineage,
        dataset_signature=dataset_signature,
        personality=personality or PersonalityCore(),
        generation=generation or GenerationParams(),
        system_prompt=system_prompt or f"You are {name}, a thoughtful AI assistant.",
        tags=tags or [],
        created_by="SloughGPT Training Pipeline",
    )
    sp.integrity_hash = sp.compute_hash()
    return sp


def export_to_sou(
    model,
    output_path: str,
    soul_profile: Optional[SoulProfile] = None,
    weights_only: bool = False,
) -> str:
    """Export model to .sou format (binary: header + config + weights)."""
    if soul_profile is None:
        soul_profile = SoulProfile(name=Path(output_path).stem)

    soul_profile.integrity_hash = soul_profile.compute_hash()
    config_json = json.dumps(soul_profile.to_dict(), default=str)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(SOU_MAGIC)
        f.write(struct.pack("<I", SOU_VERSION))
        f.write(struct.pack("<I", len(config_json)))
        f.write(config_json.encode("utf-8"))

        if not weights_only:
            import torch
            torch.save(model.state_dict(), f)

    meta_path = output_path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(soul_profile.to_dict(), f, indent=2, default=str)

    return output_path


def import_from_sou(sou_path: str):
    """Import model and soul profile from .sou file."""
    import torch

    with open(sou_path, "rb") as f:
        magic = f.read(4)
        if magic != SOU_MAGIC:
            raise ValueError(f"Invalid .sou file: {sou_path} (magic={magic!r})")

        version = struct.unpack("<I", f.read(4))[0]
        config_len = struct.unpack("<I", f.read(4))[0]
        config_json = f.read(config_len).decode("utf-8")

        if version >= 2:
            state_dict = torch.load(f, map_location="cpu", weights_only=False)
        else:
            state_dict = None

    config = json.loads(config_json)
    soul = SouParser.parse(
        f"SOUL {config.get('name', 'unknown')}\n"
        + f"VERSION {config.get('version', '1.0.0')}\n"
        + f"LINEAGE {config.get('lineage', 'nanogpt')}\n"
        + f"BORN {config.get('born_at', '')}\n"
        + f"BASEMODEL {config.get('base_model', '')}\n"
        + f"DESCRIPTION {config.get('description', '')}\n"
    )
    soul.__dict__.update(config)

    return soul, state_dict


def generate_sample_dialogue(
    model,
    stoi: Dict[int, str],
    itos: Dict[int, str],
    num_turns: int = 3,
    max_tokens: int = 50,
) -> List[Dict[str, str]]:
    """Generate sample dialogue to populate the soul profile."""
    prompts = [
        ("user", "Hello! How are you today?"),
        ("user", "What's your favorite thing about helping people?"),
        ("user", "Can you tell me a short joke?"),
    ]

    dialogue = []
    model.eval()
    import torch

    for role, prompt in prompts[:num_turns]:
        idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long)
        with torch.no_grad():
            output = model.generate(idx, max_new_tokens=max_tokens, temperature=0.8)
        response = "".join([itos.get(int(i), "?") for i in output[0]])
        response = response[len(prompt):].strip()
        dialogue.append({"role": role, "content": prompt})
        dialogue.append({"role": "assistant", "content": response[:100]})

    return dialogue


__all__ = [
    "SoulProfile",
    "PersonalityCore",
    "BehavioralTraits",
    "CognitiveSignature",
    "EmotionalRange",
    "GenerationParams",
    "ContextParams",
    "SouParser",
    "create_soul_profile",
    "export_to_sou",
    "import_from_sou",
    "generate_sample_dialogue",
    "SOU_MAGIC",
    "SOU_VERSION",
    "SOU_TRADEMARK",
]
