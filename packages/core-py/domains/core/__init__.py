"""
domains/core/ - Soul Core Architecture

SoulEngine is THE core model wrapper. All inference flows through here.
Cognitive and reasoning engines are first-class citizens, built INTO the soul.
"""

from .soul import SoulEngine, GenerationContext

__all__ = ["SoulEngine", "GenerationContext"]
