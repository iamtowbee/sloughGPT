"""
Reasoning Engine for SloughGPT

Advanced multi-step reasoning system with self-correction capabilities,
logical inference, and cognitive architecture integration.
"""

from .core.reasoning import (
    ReasoningEngine, ReasoningTask, ReasoningStep, ReasoningResult,
    ReasoningStrategy, CognitiveProcessor, InferenceEngine,
    create_reasoning_engine, run_reasoning_task
)
from .core.cognitive import (
    CognitiveProcessor, CognitiveState, CognitiveOperation,
    ThoughtProcess, MemoryRetrieval, ConceptAssociation,
    create_cognitive_processor, process_cognitive_task
)

__all__ = [
    # Main Reasoning Components
    'ReasoningEngine', 'ReasoningTask', 'ReasoningStep', 'ReasoningResult',
    'ReasoningStrategy', 'CognitiveProcessor', 'InferenceEngine',
    'create_reasoning_engine', 'run_reasoning_task',
    
    # Cognitive Components
    'CognitiveProcessor', 'CognitiveState', 'CognitiveOperation',
    'ThoughtProcess', 'MemoryRetrieval', 'ConceptAssociation',
    'create_cognitive_processor', 'process_cognitive_task'
]