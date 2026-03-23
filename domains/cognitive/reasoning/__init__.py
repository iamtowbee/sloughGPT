"""
Cognitive Reasoning Domain

This module contains the reasoning engine and related components
for advanced logical and creative reasoning.
"""

from .advanced import (
    advanced_reasoning,
    ChainOfThought,
    CausalReasoning,
    ConstitutionalAI,
    ReasoningMode,
    ReasoningResult,
    ReActReasoning,
    SelfConsistency,
    SyllogismReasoning,
    ThoughtStep,
    TreeOfThoughts,
)

__all__ = [
    "ReasoningEngine",
    "advanced_reasoning",
    "ChainOfThought",
    "TreeOfThoughts",
    "SelfConsistency",
    "ConstitutionalAI",
    "CausalReasoning",
    "SyllogismReasoning",
    "ReActReasoning",
    "ReasoningMode",
    "ThoughtStep",
    "ReasoningResult",
]


class ReasoningEngine:
    """Advanced reasoning engine with multiple strategies."""

    def __init__(self) -> None:
        self.mode = ReasoningMode.CHAIN_OF_THOUGHT
        self.reasoning_history = []

    async def reason(self, premise: str, context: dict) -> str:
        """Perform reasoning on a premise."""
        result = await advanced_reasoning(
            problem=premise,
            mode=self.mode,
            llm_call=None,
        )
        self.reasoning_history.append(result)
        return result.conclusion

    async def set_mode(self, mode: ReasoningMode) -> None:
        """Set reasoning mode."""
        self.mode = mode

    async def get_history(self) -> list:
        """Get reasoning history."""
        return self.reasoning_history
