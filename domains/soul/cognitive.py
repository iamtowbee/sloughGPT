"""
Stage 2: Cognitive SLO - Memory & Learning

Adds:
- CognitiveArchitecture: Multi-layered memory
- NeuralPlasticityEngine: Hebbian learning
- MetaLearningEngine: Learn how to learn
- DreamProcessingEngine: Sleep consolidation
"""

import random
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import defaultdict
import logging

from .foundation import FoundationSLO, SLOConfig, Experience, Thought, EvolutionStage

logger = logging.getLogger("slo.cognitive")


class CognitiveArchitecture:
    """
    Multi-layered memory system:
    - Sensory: Immediate input buffer
    - Working: Active processing (7±2 items)
    - Episodic: Autobiographical events
    - Semantic: Facts and concepts
    """
    
    def __init__(self):
        self.sensory_buffer: List[Any] = []
        self.working_memory: List[Any] = []
        self.episodic_memory: List[Dict] = []
        self.semantic_memory: Dict[str, Any] = {}
        
        self.working_capacity = 7  # Miller's law
    
    def process_sensory(self, input_data: Any) -> bool:
        """Process sensory input."""
        self.sensory_buffer.append({
            "data": input_data,
            "timestamp": datetime.now().isoformat(),
        })
        # Keep buffer small
        if len(self.sensory_buffer) > 100:
            self.sensory_buffer = self.sensory_buffer[-50:]
        return True
    
    def to_working(self, item: Any) -> bool:
        """Move item to working memory."""
        if len(self.working_memory) >= self.working_capacity:
            # FIFO eviction
            evicted = self.working_memory.pop(0)
            self._consolidate_to_episodic(evicted)
        
        self.working_memory.append(item)
        return True
    
    def _consolidate_to_episodic(self, item: Any) -> bool:
        """Consolidate working memory to episodic."""
        episode = {
            "content": item,
            "timestamp": datetime.now().isoformat(),
            "importance": random.random(),  # Simplified
        }
        self.episodic_memory.append(episode)
        return True
    
    def to_semantic(self, key: str, value: Any) -> bool:
        """Store in semantic memory."""
        if key in self.semantic_memory:
            # Strengthen existing
            self.semantic_memory[key]["strength"] += 0.1
        else:
            self.semantic_memory[key] = {
                "value": value,
                "strength": 1.0,
                "created": datetime.now().isoformat(),
            }
        return True
    
    def retrieve_semantic(self, key: str) -> Optional[Any]:
        """Retrieve from semantic memory."""
        if key in self.semantic_memory:
            self.semantic_memory[key]["last_accessed"] = datetime.now().isoformat()
            return self.semantic_memory[key]["value"]
        return None


class NeuralPlasticityEngine:
    """
    Hebbian learning: "Neurons that fire together, wire together"
    
    Implements synaptic plasticity for learning patterns.
    """
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.connections: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.activation_history: Dict[str, List[float]] = defaultdict(list)
    
    def activate(self, neuron_id: str, strength: float = 1.0) -> None:
        """Record neuron activation."""
        self.activation_history[neuron_id].append(strength)
        # Keep history limited
        if len(self.activation_history[neuron_id]) > 100:
            self.activation_history[neuron_id] = self.activation_history[neuron_id][-50:]
    
    def hebbian_learn(self, pre: str, post: str, reward: float = 1.0) -> float:
        """
        Hebbian learning rule: Δw = η * pre * post
        Strengthens connection between co-activated neurons.
        """
        pre_strength = self.activation_history[pre][-1] if self.activation_history[pre] else 1.0
        post_strength = self.activation_history[post][-1] if self.activation_history[post] else 1.0
        
        delta = self.learning_rate * pre_strength * post_strength * reward
        self.connections[pre][post] += delta
        
        return self.connections[pre][post]
    
    def get_connection_strength(self, pre: str, post: str) -> float:
        """Get connection strength between neurons."""
        return self.connections[pre][post]
    
    def prune_weak_connections(self, threshold: float = 0.01) -> int:
        """Remove weak connections (synaptic pruning)."""
        pruned = 0
        for pre in list(self.connections.keys()):
            for post in list(self.connections[pre].keys()):
                if abs(self.connections[pre][post]) < threshold:
                    del self.connections[pre][post]
                    pruned += 1
        return pruned


class MetaLearningEngine:
    """
    Learn how to learn better.
    Optimizes learning strategies based on performance.
    """
    
    def __init__(self):
        self.strategies: Dict[str, Dict] = {
            "rote": {"success": 0, "attempts": 0, "weight": 1.0},
            "spaced": {"success": 0, "attempts": 0, "weight": 1.0},
            "interleaved": {"success": 0, "attempts": 0, "weight": 1.0},
            "elaborative": {"success": 0, "attempts": 0, "weight": 1.0},
        }
        self.best_strategy = "spaced"
    
    def record_outcome(self, strategy: str, success: bool) -> None:
        """Record learning outcome for strategy."""
        if strategy in self.strategies:
            self.strategies[strategy]["attempts"] += 1
            if success:
                self.strategies[strategy]["success"] += 1
    
    def update_weights(self) -> None:
        """Update strategy weights based on performance."""
        for name, data in self.strategies.items():
            if data["attempts"] > 0:
                success_rate = data["success"] / data["attempts"]
                data["weight"] = 0.7 * data["weight"] + 0.3 * success_rate
        
        # Find best strategy
        self.best_strategy = max(
            self.strategies.keys(),
            key=lambda k: self.strategies[k]["weight"]
        )
    
    def get_strategy(self) -> str:
        """Get recommended learning strategy."""
        return self.best_strategy


class DreamProcessingEngine:
    """
    Sleep consolidation: Process and integrate memories.
    Runs during idle periods to strengthen important memories.
    """
    
    def __init__(self):
        self.dream_cycles = 0
        self.consolidated = 0
    
    def dream(self, memories: List[Experience], plasticity: NeuralPlasticityEngine) -> List[str]:
        """
        Process memories during 'sleep'.
        Returns insights generated during dreaming.
        """
        self.dream_cycles += 1
        insights = []
        
        # Replay important memories
        important = sorted(memories, key=lambda m: m.importance, reverse=True)[:10]
        
        for i, memory in enumerate(important):
            # Connect related memories (Hebbian)
            for j, other in enumerate(important):
                if i != j:
                    plasticity.hebbian_learn(memory.id, other.id)
        
        # Generate insights from patterns
        if len(important) >= 3:
            insight = f"Pattern detected across {len(important)} memories"
            insights.append(insight)
        
        self.consolidated += len(important)
        
        return insights


class CognitiveSLO(FoundationSLO):
    """
    Stage 2: Cognitive SLO
    
    Adds cognitive capabilities on top of foundation:
    - Multi-layered memory architecture
    - Neural plasticity (Hebbian learning)
    - Meta-learning optimization
    - Dream consolidation
    """
    
    def __init__(self, config: Optional[SLOConfig] = None):
        super().__init__(config)
        self.stage = EvolutionStage.COGNITIVE
        
        # Cognitive systems
        self.cognitive_arch = CognitiveArchitecture()
        self.plasticity = NeuralPlasticityEngine(
            learning_rate=self.config.learning_rate
        )
        self.meta_learner = MetaLearningEngine()
        self.dream_engine = DreamProcessingEngine()
        
        # Track learning progress
        self.learning_sessions = 0
        self.last_dream = datetime.now()
        
        logger.info("Cognitive SLO initialized")
    
    def process(self, input_data: Any) -> Thought:
        """Process with cognitive enhancement."""
        # Process through foundation first
        base_thought = super().process(input_data)
        
        # Cognitive processing
        content = str(input_data)
        
        # Sensory input
        self.cognitive_arch.process_sensory(input_data)
        
        # Working memory
        self.cognitive_arch.to_working(content)
        
        # Activate neurons for Hebbian learning
        tokens = content.split()[:10]
        for i, token in enumerate(tokens):
            self.plasticity.activate(token, 1.0)
            if i > 0:
                self.plasticity.hebbian_learn(tokens[i-1], token)
        
        # Learn semantically
        key_concepts = [t for t in tokens if len(t) > 4]
        for concept in key_concepts[:3]:
            self.cognitive_arch.to_semantic(concept, {"context": content[:100]})
        
        # Enhanced reasoning
        reasoning = base_thought.reasoning + [
            f"Cognitive layers: sensory={len(self.cognitive_arch.sensory_buffer)}, working={len(self.cognitive_arch.working_memory)}",
            f"Plastic connections: {sum(len(v) for v in self.plasticity.connections.values())}",
            f"Best learning strategy: {self.meta_learner.get_strategy()}",
        ]
        
        # Generate cognitive thought
        thought = Thought(
            content=base_thought.content,
            stage=self.stage,
            confidence=min(0.95, base_thought.confidence + 0.1),
            reasoning=reasoning,
            insights=self._generate_insights(content),
        )
        self.thoughts.append(thought)
        
        # Progress evolution
        self._evolution_progress = min(1.0, self._evolution_progress + 0.008)
        
        # Trigger dream if enough time passed
        if (datetime.now() - self.last_dream).seconds > 3600:
            self._dream()
        
        return thought
    
    def learn(self, experience: Experience) -> bool:
        """Enhanced learning with meta-cognition."""
        # Foundation learning
        success = super().learn(experience)
        
        # Apply best learning strategy
        strategy = self.meta_learner.get_strategy()
        self.learning_sessions += 1
        
        # Record outcome (simplified)
        self.meta_learner.record_outcome(strategy, success)
        
        # Update weights periodically
        if self.learning_sessions % 10 == 0:
            self.meta_learner.update_weights()
        
        return success
    
    def _generate_insights(self, content: str) -> List[str]:
        """Generate cognitive insights."""
        insights = []
        
        # Check semantic memory for related concepts
        for word in content.split()[:5]:
            if len(word) > 4:
                semantic = self.cognitive_arch.retrieve_semantic(word)
                if semantic:
                    insights.append(f"Related concept: {word}")
        
        return insights[:3]
    
    def _dream(self) -> List[str]:
        """Run dream consolidation."""
        self.last_dream = datetime.now()
        
        # Get recent experiences
        recent = self.experiences[-50:] if self.experiences else []
        
        # Dream processing
        insights = self.dream_engine.dream(recent, self.plasticity)
        
        # Prune weak connections
        pruned = self.plasticity.prune_weak_connections()
        
        logger.info(f"Dream complete: {len(insights)} insights, {pruned} connections pruned")
        
        return insights
    
    def get_status(self) -> Dict[str, Any]:
        """Get cognitive status."""
        status = super().get_status()
        status.update({
            "sensory_buffer": len(self.cognitive_arch.sensory_buffer),
            "working_memory": len(self.cognitive_arch.working_memory),
            "episodic_memory": len(self.cognitive_arch.episodic_memory),
            "semantic_memory": len(self.cognitive_arch.semantic_memory),
            "neural_connections": sum(len(v) for v in self.plasticity.connections.values()),
            "learning_sessions": self.learning_sessions,
            "best_strategy": self.meta_learner.best_strategy,
            "dream_cycles": self.dream_engine.dream_cycles,
        })
        return status
