"""
Cognitive Domain - Recovered Files Reference
This module provides references to the recovered cognitive files
"""

# Reference to recovered advanced cognitive files
# These files are kept in the root for now and can be gradually integrated

RECOVERED_FILES = {
    "advanced_reasoning_engine.py": {
        "description": "Advanced reasoning with Chain-of-Thought, Self-Reflective, Multi-Hop patterns",
        "location": "../advanced_reasoning_engine.py",
        "status": "reference",
    },
    "advanced_memory_consolidation.py": {
        "description": "Advanced memory consolidation system",
        "location": "../advanced_memory_consolidation.py",
        "status": "reference",
    },
    "stage2_cognitive_architecture.py": {
        "description": "Stage 2 cognitive architecture",
        "location": "../stage2_cognitive_architecture.py",
        "status": "reference",
    },
    "cross_layer_reasoning.py": {
        "description": "Cross-layer reasoning system",
        "location": "../cross_layer_reasoning.py",
        "status": "reference",
    },
    "epiphany_integration.py": {
        "description": "Epiphany/insight integration",
        "location": "../epiphany_integration.py",
        "status": "reference",
    },
    "sloughgpt_learning_system.py": {
        "description": "Learning system for SloughGPT",
        "location": "../sloughgpt_learning_system.py",
        "status": "reference",
    },
    "sloughgpt_neural_network.py": {
        "description": "Neural network implementation",
        "location": "../sloughgpt_neural_network.py",
        "status": "reference",
    },
    "slo_cognitive_core.py": {
        "description": "Core cognitive processing",
        "location": "../slo_cognitive_core.py",
        "status": "reference",
    },
    "slo_focused_cognitive.py": {
        "description": "Focused cognitive processing",
        "location": "../slo_focused_cognitive.py",
        "status": "reference",
    },
    "slo_knowledge_graph.py": {
        "description": "Knowledge graph implementation",
        "location": "../slo_knowledge_graph.py",
        "status": "reference",
    },
    "slo_metacognitive.py": {
        "description": "Metacognition module",
        "location": "../slo_metacognitive.py",
        "status": "reference",
    },
    "slo_multi_agent.py": {
        "description": "Multi-agent coordination",
        "location": "../slo_multi_agent.py",
        "status": "reference",
    },
    "slo_rag.py": {
        "description": "Retrieval-Augmented Generation",
        "location": "../slo_rag.py",
        "status": "reference",
    },
    "slo_spaced_repetition.py": {
        "description": "Spaced repetition learning",
        "location": "../slo_spaced_repetition.py",
        "status": "reference",
    },
}


def get_recovered_file(filename: str):
    """Get a recovered file reference."""
    return RECOVERED_FILES.get(filename)


__all__ = ["RECOVERED_FILES", "get_recovered_file"]
