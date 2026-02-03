#!/usr/bin/env python3
"""
Cross-Layer Reasoning Patterns

Advanced reasoning patterns that operate across multiple cognitive layers
for enhanced emergent understanding and epiphany generation.
"""

import sys
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from stage2_cognitive_architecture import CognitiveArchitecture, CognitiveState
from advanced_reasoning_engine import AdvancedReasoningEngine, ReasoningChain, ReasoningStep
from epiphany_integration import EpiphanyIntegrationBridge

@dataclass
class CrossLayerConnection:
    """Enhanced connection between cognitive layers"""
    source_layer: str
    target_layer: str
    connection_strength: float
    information_flow: str
    activation_threshold: float
    last_activation: float = 0.0
    activation_history: List[float] = field(default_factory=list)

@dataclass
class ResonancePattern:
    """Pattern for cognitive resonance detection"""
    pattern_id: str
    frequency: float
    phase: float
    amplitude: float
    coherence: float
    participating_layers: List[str]

class CrossLayerReasoningPatterns:
    """Advanced cross-layer reasoning patterns"""
    
    def __init__(self, reasoning_engine: AdvancedReasoningEngine):
        self.engine = reasoning_engine
        self.cognitive_arch = reasoning_engine.cognitive_arch
        
        # Cross-layer connections
        self.layer_connections = self._initialize_layer_connections()
        
        # Resonance patterns
        self.resonance_patterns = self._initialize_resonance_patterns()
        
        # Active resonances
        self.active_resonances = []
        self.resonance_history = []
        
        # Pattern matching
        self.pattern_network = self._build_pattern_network()
        
        print("🌊 Cross-Layer Reasoning Patterns initialized")
        print(f"🔗 Layer connections: {len(self.layer_connections)}")
        print(f"🎵 Resonance patterns: {len(self.resonance_patterns)}")
    
    def _initialize_layer_connections(self) -> Dict[str, CrossLayerConnection]:
        """Initialize enhanced cross-layer connections"""
        connections = {}
        
        # Define cognitive layers
        layers = ["sensory_memory", "working_memory", "episodic_memory", "semantic_memory"]
        
        # Create bi-directional connections with different strengths
        connection_configs = [
            ("sensory_memory", "working_memory", 0.9, "input_processing", 0.6),
            ("sensory_memory", "episodic_memory", 0.4, "pattern_recognition", 0.8),
            ("working_memory", "episodic_memory", 0.8, "context_transfer", 0.7),
            ("working_memory", "semantic_memory", 0.7, "concept_activation", 0.6),
            ("episodic_memory", "semantic_memory", 0.6, "memory_consolidation", 0.8),
            ("semantic_memory", "working_memory", 0.8, "concept_retrieval", 0.7),
            ("episodic_memory", "sensory_memory", 0.3, "association_recall", 0.9)
        ]
        
        for source, target, strength, flow_type, threshold in connection_configs:
            connection_id = f"{source}->{target}"
            connections[connection_id] = CrossLayerConnection(
                source_layer=source,
                target_layer=target,
                connection_strength=strength,
                information_flow=flow_type,
                activation_threshold=threshold
            )
            
            # Add reverse connection if not symmetrical
            reverse_id = f"{target}->{source}"
            if reverse_id not in connections:
                connections[reverse_id] = CrossLayerConnection(
                    source_layer=target,
                    target_layer=source,
                    connection_strength=strength * 0.8,  # Slightly weaker reverse
                    information_flow=f"reverse_{flow_type}",
                    activation_threshold=threshold * 1.2
                )
        
        return connections
    
    def _initialize_resonance_patterns(self) -> List[ResonancePattern]:
        """Initialize resonance patterns for cross-layer processing"""
        return [
            ResonancePattern(
                pattern_id="alpha_wave",
                frequency=8.0,
                phase=0.0,
                amplitude=0.8,
                coherence=0.9,
                participating_layers=["sensory_memory", "working_memory"]
            ),
            ResonancePattern(
                pattern_id="beta_sync",
                frequency=12.0,
                phase=np.pi/4,
                amplitude=0.6,
                coherence=0.8,
                participating_layers=["episodic_memory", "semantic_memory"]
            ),
            ResonancePattern(
                pattern_id="gamma_harmony",
                frequency=16.0,
                phase=np.pi/2,
                amplitude=0.7,
                coherence=0.85,
                participating_layers=["working_memory", "semantic_memory", "episodic_memory"]
            ),
            ResonancePattern(
                pattern_id="theta_coherence",
                frequency=4.0,
                phase=3*np.pi/2,
                amplitude=0.9,
                coherence=0.95,
                participating_layers=["semantic_memory", "episodic_memory", "working_memory"]
            )
        ]
    
    def _build_pattern_network(self) -> Dict[str, Dict[str, float]]:
        """Build pattern matching network"""
        return {
            "harmony_detection": {
                "frequency_analysis": 0.9,
                "phase_coherence": 0.8,
                "amplitude_resonance": 0.85
            },
            "pattern_completion": {
                "missing_piece_detection": 0.9,
                "structural_integrity": 0.8,
                "logical_flow": 0.85
            },
            "consciousness_evidence": {
                "self_reference": 0.8,
                "meta_cognition": 0.9,
                "recursive_awareness": 0.85
            },
            "creative_insight": {
                "novelty_generation": 0.9,
                "metaphor_mapping": 0.8,
                "abstraction_level": 0.85
            }
        }
    
    def process_cross_layer_reasoning(self, query: str) -> Tuple[ReasoningChain, Optional[str]]:
        """Process query using cross-layer reasoning patterns"""
        print(f"🌊 Cross-layer processing: {query[:50]}...")
        
        # Get current cognitive state
        cognitive_state = self.cognitive_arch.get_cognitive_statistics()
        
        # Detect active resonances
        active_resonances = self._detect_active_resonances(query, cognitive_state)
        
        if active_resonances:
            # Use resonance-based reasoning
            reasoning_chain, insight_type = self._generate_resonance_reasoning(query, active_resonances)
            
            print(f"🎵 Active resonances: {[r.pattern_id for r in active_resonances]}")
            return reasoning_chain, f"resonance_{insight_type}"
        
        # Fall back to enhanced standard reasoning
        return self._enhanced_standard_reasoning(query), None
    
    def _detect_active_resonances(self, query: str, cognitive_state: Dict[str, Any]) -> List[ResonancePattern]:
        """Detect active resonance patterns"""
        active_resonances = []
        
        # Analyze query for resonance triggers
        query_lower = query.lower()
        resonance_triggers = {
            "alpha_wave": ["sensory", "input", "perception", "immediate", "processing"],
            "beta_sync": ["memory", "recall", "sync", "harmony", "pattern"],
            "gamma_harmony": ["multiple", "integration", "complex", "synthesis", "whole"],
            "theta_coherence": ["deep", "understanding", "meaning", "structure", "why"]
        }
        
        for pattern in self.resonance_patterns:
            triggers = resonance_triggers.get(pattern.pattern_id, [])
            
            # Check trigger presence in query
            trigger_count = sum(1 for trigger in triggers if trigger in query_lower)
            
            # Check layer activation levels
            layer_activations = []
            for layer in pattern.participating_layers:
                activation = self._get_layer_activation_level(layer, cognitive_state)
                layer_activations.append(activation)
            
            avg_activation = np.mean(layer_activations) if layer_activations else 0.0
            
            # Calculate resonance strength
            frequency_factor = self._calculate_frequency_match(query, pattern.frequency)
            coherence_factor = pattern.coherence  # Static for now
            
            resonance_strength = (trigger_count / len(triggers)) * 0.5 + avg_activation * 0.3 + frequency_factor * 0.1 + coherence_factor * 0.1
            
            # Activate if above threshold
            if resonance_strength > 0.6:
                active_resonances.append(pattern)
        
        return active_resonances
    
    def _get_layer_activation_level(self, layer: str, cognitive_state: Dict[str, Any]) -> float:
        """Get activation level for specific layer"""
        try:
            if layer == "working_memory":
                # Based on current task and items
                working_state = cognitive_state.get("working_memory", {})
                task_active = working_state.get("current_task") is not None
                items_count = working_state.get("items", 0)
                
                task_factor = 1.0 if task_active else 0.3
                items_factor = min(items_count / 7.0, 1.0)  # 7 is capacity
                
                return (task_factor + items_factor) / 2.0
            
            elif layer == "episodic_memory":
                # Based on consolidation queue
                episodic_state = cognitive_state.get("episodic_memory", {})
                queue_size = episodic_state.get("consolidation_queue", 0)
                forgetting_size = episodic_state.get("forgetting_curve_size", 0)
                
                return min((queue_size + forgetting_size) / 30.0, 1.0)
            
            elif layer == "semantic_memory":
                # Based on concepts and clusters
                semantic_state = cognitive_state.get("semantic_memory", {})
                concepts_count = semantic_state.get("concepts_count", 0)
                clusters_count = semantic_state.get("semantic_clusters", 0)
                
                return min((concepts_count + clusters_count) / 150.0, 1.0)
            
            elif layer == "sensory_memory":
                # Based on buffer and attention
                sensory_state = cognitive_state.get("sensory_memory", {})
                buffer_size = sensory_state.get("buffer_size", 0)
                attention = sensory_state.get("attention_weights", 0)
                
                return min((buffer_size + attention) / 100.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_frequency_match(self, query: str, pattern_frequency: float) -> float:
        """Calculate how well query matches pattern frequency"""
        # Simple heuristic based on query characteristics
        query_length = len(query.split())
        
        if pattern_frequency <= 6:  # Low frequency patterns (alpha)
            return min(query_length / 10.0, 1.0) if query_length <= 10 else 0.3
        elif pattern_frequency <= 12:  # Medium frequency (beta, theta)
            return 0.8 if 5 <= query_length <= 15 else 0.5
        else:  # High frequency (gamma)
            return 0.6 if query_length >= 10 else 0.4
    
    def _generate_resonance_reasoning(self, query: str, active_resonances: List[ResonancePattern]) -> Tuple[ReasoningChain, str]:
        """Generate reasoning using active resonance patterns"""
        steps = []
        
        # Sort resonances by strength
        sorted_resonances = sorted(active_resonances, key=lambda x: x.amplitude * x.coherence, reverse=True)
        
        for i, resonance in enumerate(sorted_resonances):
            step_reasoning = self._generate_resonance_step(query, resonance, i)
            confidence = resonance.amplitude * resonance.coherence
            
            step = ReasoningStep(
                step_id=f"resonance_{resonance.pattern_id}_{i}",
                query=query,
                retrieved_docs=[],
                cognitive_context=f"resonance_{resonance.pattern_id}",
                reasoning=step_reasoning,
                confidence=confidence,
                timestamp=time.time()
            )
            steps.append(step)
        
        # Synthesize final answer
        final_answer = self._synthesize_resonance_answer(query, sorted_resonances, steps)
        total_confidence = np.mean([s.confidence for s in steps])
        
        reasoning_chain = ReasoningChain(
            chain_id=f"resonance_{int(time.time())}",
            original_query=query,
            steps=steps,
            final_answer=final_answer,
            total_confidence=total_confidence,
            metadata={
                "pattern": "cross_layer_resonance",
                "active_resonances": [r.pattern_id for r in sorted_resonances],
                "resonance_count": len(sorted_resonances)
            }
        )
        
        # Store resonance for learning
        self._record_resonance_usage(sorted_resonances)
        
        insight_type = "multi_resonance" if len(sorted_resonances) > 1 else sorted_resonances[0].pattern_id
        
        return reasoning_chain, insight_type
    
    def _generate_resonance_step(self, query: str, resonance: ResonancePattern, step_num: int) -> str:
        """Generate reasoning for a single resonance step"""
        step_templates = {
            "alpha_wave": f"Sensory alpha resonance: Direct perceptual processing reveals immediate patterns in '{query[:30]}...'",
            "beta_sync": f"Memory beta sync: Pattern matching and harmonic integration identifies connections in '{query[:30]}...'",
            "gamma_harmony": f"Semantic gamma harmony: Complex integration creates unified understanding of '{query[:30]}...'",
            "theta_coherence": f"Deep theta coherence: Fundamental structural analysis of '{query[:30]}...'"
        }
        
        template = step_templates.get(resonance.pattern_id, f"Resonance step {step_num}: {query}")
        
        # Add resonance-specific insights
        if resonance.pattern_id == "theta_coherence":
            template += "\nThis reveals deep structural relationships and fundamental principles."
        elif resonance.pattern_id == "gamma_harmony":
            template += "\nMultiple layers achieve harmonious integration."
        elif resonance.pattern_id == "beta_sync":
            template += "\nPatterns synchronize across memory systems."
        elif resonance.pattern_id == "alpha_wave":
            template += "\nImmediate perceptual insights emerge."
        
        return template
    
    def _synthesize_resonance_answer(self, query: str, resonances: List[ResonancePattern], steps: List[ReasoningStep]) -> str:
        """Synthesize final answer from resonance reasoning"""
        if len(resonances) == 1:
            return f"Cross-layer resonance analysis reveals that {query[:40]} is primarily understood through {resonances[0].pattern_id} patterns, indicating {self._get_resonance_meaning(resonances[0].pattern_id)}."
        
        elif len(resonances) >= 2:
            dominant_resonance = resonances[0]  # Sorted by strength
            secondary_resonances = resonances[1:]
            
            primary_meaning = self._get_resonance_meaning(dominant_resonance.pattern_id)
            secondary_aspects = [self._get_resonance_meaning(r.pattern_id) for r in secondary_resonances]
            
            secondary_aspects_str = ' + '.join(secondary_aspects)
            return f"Multi-layer resonance analysis shows that {query[:40]} involves {dominant_resonance.pattern_id} patterns ({primary_meaning}), complemented by {secondary_aspects_str}. This indicates complex cross-layer cognitive processing."
        
        return f"Cross-layer analysis of {query[:40]} reveals patterns across multiple cognitive dimensions."
    
    def _get_resonance_meaning(self, pattern_id: str) -> str:
        """Get cognitive meaning of resonance pattern"""
        meanings = {
            "alpha_wave": "immediate perceptual processing",
            "beta_sync": "pattern recognition and memory synchronization", 
            "gamma_harmony": "complex integration and unified understanding",
            "theta_coherence": "deep structural analysis and fundamental principles"
        }
        return meanings.get(pattern_id, "cognitive processing")
    
    def _record_resonance_usage(self, resonances: List[ResonancePattern]):
        """Record resonance usage for learning"""
        usage_event = {
            "timestamp": time.time(),
            "resonances": [r.pattern_id for r in resonances],
            "total_resonances": len(resonances),
            "dominant_frequency": np.mean([r.frequency for r in resonances]),
            "coherence_level": np.mean([r.coherence for r in resonances])
        }
        
        self.resonance_history.append(usage_event)
        
        # Update layer connections
        self._update_layer_connections(resonances)
    
    def _update_layer_connections(self, resonances: List[ResonancePattern]):
        """Update layer connections based on resonance usage"""
        current_time = time.time()
        
        for resonance in resonances:
            for i, layer in enumerate(resonance.participating_layers):
                if i < len(resonance.participating_layers) - 1:
                    connection_id = f"{layer}->{resonance.participating_layers[i+1]}"
                    
                    if connection_id in self.layer_connections:
                        connection = self.layer_connections[connection_id]
                        connection.last_activation = current_time
                        
                        # Strengthen connection based on successful resonance
                        connection.connection_strength = min(connection.connection_strength * 1.05, 1.0)
    
    def _enhanced_standard_reasoning(self, query: str) -> ReasoningChain:
        """Enhanced standard reasoning when no resonances detected"""
        # Use epiphany integration if available
        if hasattr(self.engine, 'epiphany_bridge') and self.engine.use_epiphany:
            return self.engine.epiphany_bridge.process_with_epiphany_integration(query, "hybrid")[0]
        
        # Fall back to hybrid reasoning
        return self.engine.reason(query, "hybrid")
    
    def get_cross_layer_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cross-layer reasoning statistics"""
        if not self.resonance_history:
            return {"status": "No cross-layer reasoning performed yet"}
        
        # Analyze resonance patterns
        pattern_usage = defaultdict(int)
        dominant_patterns = []
        
        for event in self.resonance_history:
            for resonance in event["resonances"]:
                pattern_usage[resonance] += 1
            
            if event["coherence_level"] > 0.8:
                dominant_patterns.append(event["resonances"][0])  # Most coherent pattern
        
        # Connection analysis
        active_connections = [conn_id for conn_id, conn in self.layer_connections.items() 
                            if time.time() - conn.last_activation < 3600]  # Last hour
        
        return {
            "total_cross_layer_reasoning": len(self.resonance_history),
            "resonance_patterns_used": dict(pattern_usage),
            "most_common_patterns": dict(pattern_usage.most_common(3)),
            "dominant_patterns": dominant_patterns,
            "average_coherence": np.mean([e["coherence_level"] for e in self.resonance_history]) if self.resonance_history else 0,
            "active_connections": len(active_connections),
            "connection_strength": {
                conn_id: {
                    "strength": conn.connection_strength,
                    "last_activation": conn.last_activation,
                    "information_flow": conn.information_flow
                }
                for conn_id, conn in self.layer_connections.items()
            }
        }

def main():
    """Test cross-layer reasoning patterns"""
    print("🌊 Cross-Layer Reasoning Patterns Test")
    print("=" * 60)
    
    # Initialize reasoning engine
    from advanced_reasoning_engine import AdvancedReasoningEngine
    engine = AdvancedReasoningEngine('runs/store/hauls_store.db')
    
    # Create cross-layer patterns
    cross_layer = CrossLayerReasoningPatterns(engine)
    
    # Test queries that might trigger resonances
    test_queries = [
        "What sensory patterns do you detect in my questions?",  # Should trigger alpha_wave
        "How can you synchronize memory recall with understanding?",  # Should trigger beta_sync  
        "Explain the deep structural unity of this concept",  # Should trigger theta_coherence
        "How do multiple cognitive systems achieve harmony?",  # Should trigger gamma_harmony
        "Create a comprehensive synthesis of these complex ideas",  # Multiple resonances
        "Why do certain insights feel immediately obvious?"  # Alpha wave
    ]
    
    print("\n🧪 Testing Cross-Layer Reasoning:\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. {query}")
        
        try:
            reasoning_chain, insight_type = cross_layer.process_cross_layer_reasoning(query)
            
            print(f"   ✨ Cross-layer reasoning: {insight_type}")
            print(f"   📊 Steps: {len(reasoning_chain.steps)}")
            print(f"   🎯 Confidence: {reasoning_chain.total_confidence:.3f}")
            print(f"   💡 Answer: {reasoning_chain.final_answer[:100]}...")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:50]}...")
        
        print()
    
    # Get statistics
    stats = cross_layer.get_cross_layer_statistics()
    print(f"\n📊 Cross-Layer Statistics: {json.dumps(stats, indent=2)}")

if __name__ == "__main__":
    main()