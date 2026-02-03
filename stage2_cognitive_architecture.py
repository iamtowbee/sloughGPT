#!/usr/bin/env python3
"""
Stage 2: Cognitive Architecture - Multi-layered Memory System
Extends Stage 1 foundation with cognitive capabilities
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from collections import deque, defaultdict
import random
import threading
import asyncio

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from hauls_store import HaulsStore, Document


@dataclass
class CognitiveState:
    """Current cognitive processing state"""
    attention_focus: str
    working_memory_load: float
    processing_depth: int
    neural_activity: float
    consolidation_level: float


@dataclass
class MemoryTrace:
    """Trace of memory access patterns for learning"""
    timestamp: float
    doc_id: str
    access_type: str
    attention_weight: float
    retention_strength: float


class SensoryMemory:
    """Immediate sensory input buffer"""
    
    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.attention_weights = {}
        self.decay_rate = 0.95
        
    def add_sensory_input(self, content: str, metadata: Dict[str, Any]) -> str:
        """Add sensory input to buffer"""
        sensory_id = f"sensory_{int(time.time() * 1000) % 10000}"
        
        self.buffer.append({
            'id': sensory_id,
            'content': content,
            'metadata': {
                **metadata,
                'type': 'sensory',
                'timestamp': time.time(),
                'attention_weight': 1.0
            }
        })
        
        return sensory_id
    
    def get_attention_focused(self) -> Optional[Dict[str, Any]]:
        """Get most attention-weighted sensory input"""
        if not self.buffer:
            return None
        
        # Apply attention weights (recent inputs have higher weight)
        for item in self.buffer:
            age = time.time() - item['metadata']['timestamp']
            decay = self.decay_rate ** age
            item['metadata']['attention_weight'] *= decay
        
        # Find highest attention item
        focused_item = max(self.buffer, key=lambda x: x['metadata']['attention_weight'])
        return focused_item
    
    def clear_old_inputs(self, max_age: float = 30.0):
        """Clear old sensory inputs"""
        current_time = time.time()
        while self.buffer and (current_time - self.buffer[0]['metadata']['timestamp']) > max_age:
            self.buffer.popleft()


class WorkingMemory:
    """Active working memory for current cognitive tasks"""
    
    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.items = []
        self.current_task = None
        self.chunks = []  # Chunking for complex tasks
        
    def add_to_working_memory(self, content: str, metadata: Dict[str, Any]) -> str:
        """Add item to working memory with capacity management"""
        working_id = f"working_{int(time.time() * 1000) % 10000}"
        
        item = {
            'id': working_id,
            'content': content,
            'metadata': {
                **metadata,
                'type': 'working',
                'timestamp': time.time(),
                'access_count': 1
            }
        }
        
        # Manage capacity with chunking
        if len(self.items) >= self.capacity:
            # Chunk older items or move to episodic
            self._manage_capacity_overflow()
        
        self.items.append(item)
        return working_id
    
    def get_relevant_context(self, query: str) -> List[str]:
        """Get context relevant to current query"""
        if not self.items:
            return []
        
        # Simple relevance scoring based on content similarity
        relevant = []
        query_words = set(query.lower().split())
        
        for item in self.items:
            content_words = set(item['content'].lower().split())
            overlap = len(query_words.intersection(content_words))
            if overlap > 0:
                relevant.append(item['content'])
                # Update access count
                item['metadata']['access_count'] += 1
        
        return relevant[-3:]  # Return last 3 relevant items
    
    def _manage_capacity_overflow(self):
        """Handle when working memory is full"""
        if len(self.items) >= self.capacity * 1.5:
            # Move oldest items to episodic memory consideration
            old_items = self.items[:self.capacity // 2]
            self.items = self.items[self.capacity // 2:]
            
            # Mark for episodic transfer
            for item in old_items:
                item['metadata']['transfer_to_episodic'] = True


class EpisodicMemory:
    """Long-term memory for specific experiences"""
    
    def __init__(self, hauls_store: HaulsStore):
        self.hauls_store = hauls_store
        self.consolidation_queue = []
        self.consolidation_threshold = 10
        self.forgetting_curve = {}
        
    def add_episode(self, content: str, metadata: Dict[str, Any]) -> str:
        """Add episodic memory event"""
        episode_id = self.hauls_store.add_document(content, {
            **metadata,
            'type': 'episodic',
            'timestamp': time.time(),
            'emotional_weight': self._calculate_emotional_weight(content)
        })
        
        # Track for consolidation
        self.consolidation_queue.append({
            'episode_id': episode_id,
            'timestamp': time.time()
        })
        
        return episode_id
    
    def recall_episodes(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Recall relevant episodic memories"""
        results = self.hauls_store.search(query, top_k=top_k * 2)
        
        # Filter for episodic memories and apply temporal weighting
        episodic_results = []
        current_time = time.time()
        
        for result in results:
            if result['metadata'].get('type') == 'episodic':
                # Apply temporal decay
                age = current_time - result['metadata']['timestamp']
                temporal_weight = self._apply_forgetting_curve(result['id'], age)
                
                episodic_result = result.copy()
                episodic_result['temporal_weight'] = temporal_weight
                episodic_result['relevance'] = result['similarity'] * temporal_weight
                episodic_results.append(episodic_result)
        
        # Sort by weighted relevance and return top_k
        episodic_results.sort(key=lambda x: x['relevance'], reverse=True)
        return episodic_results[:top_k]
    
    def _calculate_emotional_weight(self, content: str) -> float:
        """Calculate emotional significance of content"""
        # Simple heuristic - could be enhanced with NLP
        emotional_words = ['important', 'critical', 'urgent', 'emergency', 'special', 
                          'love', 'fear', 'joy', 'anger', 'sad', 'exciting']
        content_lower = content.lower()
        
        weight = 1.0
        for word in emotional_words:
            if word in content_lower:
                weight += 0.2
        
        return min(weight, 2.0)  # Cap emotional weight
    
    def _apply_forgetting_curve(self, memory_id: str, age: float) -> float:
        """Apply forgetting curve based on access patterns"""
        if memory_id not in self.forgetting_curve:
            self.forgetting_curve[memory_id] = 1.0
        
        # Exponential decay with access reinforcement
        decay_rate = 0.99  # Daily decay
        days_elapsed = age / (24 * 3600)  # Convert to days
        
        decayed_weight = self.forgetting_curve[memory_id] * (decay_rate ** days_elapsed)
        return max(decayed_weight, 0.1)  # Minimum weight
    
    def consolidate_memories(self) -> int:
        """Consolidate queued memories for long-term storage"""
        if len(self.consolidation_queue) < self.consolidation_threshold:
            return 0
        
        consolidated_count = 0
        current_time = time.time()
        
        for episode in self.consolidation_queue[:self.consolidation_threshold]:
            # Check if episode should be consolidated
            age = current_time - episode['timestamp']
            if age > 3600:  # 1 hour old
                # Strengthen memory during consolidation
                self._strengthen_memory(episode['episode_id'])
                consolidated_count += 1
        
        self.consolidation_queue = self.consolidation_queue[self.consolidation_threshold:]
        return consolidated_count
    
    def _strengthen_memory(self, episode_id: str):
        """Strengthen memory during consolidation"""
        if episode_id in self.forgetting_curve:
            self.forgetting_curve[episode_id] = min(self.forgetting_curve[episode_id] * 1.1, 2.0)


class SemanticMemory:
    """General knowledge and conceptual understanding"""
    
    def __init__(self, hauls_store: HaulsStore):
        self.hauls_store = hauls_store
        self.concept_network = {}
        self.semantic_clusters = {}
        
    def add_concept(self, concept: str, definition: str, relationships: List[str] = None) -> str:
        """Add semantic concept to knowledge base"""
        concept_id = self.hauls_store.add_document(definition, {
            'type': 'semantic',
            'concept': concept,
            'relationships': relationships or [],
            'timestamp': time.time()
        })
        
        # Build concept network
        self.concept_network[concept] = {
            'id': concept_id,
            'definition': definition,
            'relationships': relationships or [],
            'activation_strength': 1.0
        }
        
        return concept_id
    
    def reason_about_concepts(self, query: str) -> List[Dict[str, Any]]:
        """Perform reasoning about concepts"""
        # Search for relevant concepts
        relevant_concepts = []
        query_words = query.lower().split()
        
        for concept, info in self.concept_network.items():
            if any(word in concept.lower() for word in query_words):
                relevant_concepts.append({
                    'concept': concept,
                    'definition': info['definition'],
                    'relationships': info['relationships'],
                    'strength': info['activation_strength']
                })
        
        # Simple reasoning - could be enhanced with logic engines
        reasoning_results = []
        for concept_info in relevant_concepts:
            # Generate inferences
            inferences = self._generate_inferences(concept_info)
            reasoning_results.append({
                'concept': concept_info['concept'],
                'definition': concept_info['definition'],
                'inferences': inferences,
                'confidence': concept_info['strength']
            })
        
        return reasoning_results
    
    def _generate_inferences(self, concept_info: Dict[str, Any]) -> List[str]:
        """Generate inferences from concept information"""
        inferences = []
        
        # Simple pattern-based inferences
        concept = concept_info['concept'].lower()
        definition = concept_info['definition'].lower()
        
        # Relationship-based inferences
        for relationship in concept_info['relationships']:
            inferences.append(f"{concept} is related to {relationship}")
        
        # Definition-based inferences
        if 'type of' in definition:
            inferences.append(f"{concept} can be classified")
        if 'example' in definition:
            inferences.append(f"{concept} has practical applications")
        
        return inferences


class CognitiveArchitecture:
    """Main cognitive architecture integrating all memory layers"""
    
    def __init__(self, hauls_store_path: str = "runs/store/cognitive_hauls.db"):
        # Initialize memory systems
        self.hauls_store = HaulsStore(hauls_store_path)
        self.sensory_memory = SensoryMemory(capacity=50)
        self.working_memory = WorkingMemory(capacity=7)
        self.episodic_memory = EpisodicMemory(self.hauls_store)
        self.semantic_memory = SemanticMemory(self.hauls_store)
        
        # Cognitive state tracking
        self.cognitive_state = CognitiveState(
            attention_focus="",
            working_memory_load=0.0,
            processing_depth=0,
            neural_activity=0.0,
            consolidation_level=0.0
        )
        
        # Memory traces for learning
        self.memory_traces = []
        
        # Processing parameters
        self.attention_capacity = 4  # Miller's number ± 1
        self.consolidation_interval = 300  # 5 minutes
        self.learning_rate = 0.1
        
    def process_input(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Process input through all cognitive layers"""
        processing_start = time.time()
        
        # Layer 1: Sensory memory
        sensory_id = self.sensory_memory.add_sensory_input(content, metadata or {})
        
        # Layer 2: Attention allocation
        attention_result = self._allocate_attention(content)
        
        # Layer 3: Working memory integration
        working_id = self.working_memory.add_to_working_memory(
            attention_result['attended_content'], 
            {**(metadata or {}), 'sensory_id': sensory_id}
        )
        
        # Layer 4: Episodic memory formation
        if attention_result['emotional_weight'] > 1.2:
            # High emotional significance - store in episodic
            episode_id = self.episodic_memory.add_episode(
                attention_result['attended_content'],
                {**(metadata or {}), 'attention_weight': attention_result['emotional_weight']}
            )
        else:
            episode_id = None
        
        # Layer 5: Semantic integration
        semantic_concepts = self._extract_semantic_concepts(attention_result['attended_content'])
        for concept in semantic_concepts:
            self.semantic_memory.add_concept(
                concept['name'], 
                concept['definition'], 
                concept['relationships']
            )
        
        # Update cognitive state
        processing_time = time.time() - processing_start
        self._update_cognitive_state(processing_time, attention_result['complexity'])
        
        # Record memory trace
        trace = MemoryTrace(
            timestamp=processing_start,
            doc_id=working_id,
            access_type='cognitive_processing',
            attention_weight=attention_result['attention_weight'],
            retention_strength=1.0
        )
        self.memory_traces.append(trace)
        
        return working_id
    
    def _allocate_attention(self, content: str) -> Dict[str, Any]:
        """Allocate attention to content features"""
        # Calculate attention weights based on saliency
        features = self._extract_features(content)
        
        attention_weights = {}
        total_weight = 0
        
        for feature, weight in features.items():
            attention_weights[feature] = weight
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            for feature in attention_weights:
                attention_weights[feature] /= total_weight
        
        # Select top attention features
        attended_features = sorted(attention_weights.items(), key=lambda x: x[1], reverse=True)
        top_features = attended_features[:self.attention_capacity]
        
        attended_content = ' '.join([feature for feature, _ in top_features])
        
        return {
            'attended_content': attended_content,
            'attention_weights': attention_weights,
            'top_features': top_features,
            'complexity': len(features),
            'attention_weight': sum(weight for _, weight in top_features),
            'emotional_weight': self._calculate_emotional_saliency(content)
        }
    
    def _extract_features(self, content: str) -> Dict[str, float]:
        """Extract salient features for attention"""
        features = {}
        words = content.lower().split()
        
        # Word frequency (rare words get more attention)
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        for word in words:
            if word not in common_words:
                features[word] = features.get(word, 0) + 2.0
            else:
                features[word] = features.get(word, 0) + 0.5
        
        # Length-based attention
        if len(content) > 100:
            features['long_content'] = 2.0
        elif len(content) > 50:
            features['medium_content'] = 1.0
        
        # Question/important markers
        if '?' in content or '!' in content:
            features['question_exclamation'] = 3.0
        if any(word in content.lower() for word in ['important', 'critical', 'urgent']):
            features['important_marker'] = 2.5
        
        return features
    
    def _calculate_emotional_saliency(self, content: str) -> float:
        """Calculate emotional saliency of content"""
        emotional_words = {
            'love': 2.0, 'fear': 2.5, 'anger': 2.0, 'joy': 1.8, 'sad': 1.5,
            'exciting': 1.7, 'danger': 2.8, 'safe': 1.2, 'important': 2.0
        }
        
        content_lower = content.lower()
        saliency = 1.0
        
        for word, weight in emotional_words.items():
            if word in content_lower:
                saliency = max(saliency, weight)
        
        return saliency
    
    def _extract_semantic_concepts(self, content: str) -> List[Dict[str, str]]:
        """Extract semantic concepts from content"""
        # Simple concept extraction (could be enhanced with NLP)
        concepts = []
        
        # Capitalized words as potential concepts
        import re
        potential_concepts = re.findall(r'\b[A-Z][a-z]+\b', content)
        
        for concept in potential_concepts:
            if len(concept) > 2:  # Filter out short words
                concepts.append({
                    'name': concept,
                    'definition': f"{concept} appears in context: {content[:50]}...",
                    'relationships': []
                })
        
        return concepts
    
    def _update_cognitive_state(self, processing_time: float, complexity: int):
        """Update cognitive state based on processing"""
        # Update working memory load
        self.cognitive_state.working_memory_load = len(self.working_memory.items) / self.working_memory.capacity
        
        # Update processing depth
        self.cognitive_state.processing_depth = max(self.cognitive_state.processing_depth, complexity)
        
        # Update neural activity (simplified model)
        self.cognitive_state.neural_activity = min(
            self.cognitive_state.neural_activity * 0.9 + processing_time * 10,
            1.0
        )
        
        # Update consolidation level
        self.cognitive_state.consolidation_level = (
            len(self.episodic_memory.consolidation_queue) / 
            self.episodic_memory.consolidation_threshold
        )
    
    def retrieve_memories(self, query: str, memory_type: str = 'all') -> Dict[str, Any]:
        """Retrieve memories from appropriate systems"""
        results = {
            'query': query,
            'timestamp': time.time(),
            'cognitive_state': self.cognitive_state.__dict__,
            'memories': {}
        }
        
        # Get sensory memories
        if memory_type in ['all', 'sensory']:
            focused_sensory = self.sensory_memory.get_attention_focused()
            results['memories']['sensory'] = focused_sensory
        
        # Get working memory context
        if memory_type in ['all', 'working']:
            working_context = self.working_memory.get_relevant_context(query)
            results['memories']['working'] = working_context
        
        # Get episodic memories
        if memory_type in ['all', 'episodic']:
            episodic_memories = self.episodic_memory.recall_episodes(query)
            results['memories']['episodic'] = episodic_memories
        
        # Get semantic reasoning
        if memory_type in ['all', 'semantic']:
            semantic_reasoning = self.semantic_memory.reason_about_concepts(query)
            results['memories']['semantic'] = semantic_reasoning
        
        return results
    
    def consolidate_and_cleanup(self) -> Dict[str, Any]:
        """Perform memory consolidation and cleanup"""
        # Consolidate episodic memories
        consolidated_episodes = self.episodic_memory.consolidate_memories()
        
        # Clean old sensory inputs
        self.sensory_memory.clear_old_inputs()
        
        # Optimize working memory
        if len(self.working_memory.items) > self.working_memory.capacity:
            # Move least accessed to episodic
            self._optimize_working_memory()
        
        # Memory trace analysis
        recent_traces = [trace for trace in self.memory_traces 
                        if time.time() - trace.timestamp < 3600]  # Last hour
        
        return {
            'timestamp': time.time(),
            'consolidated_episodes': consolidated_episodes,
            'memory_traces_analyzed': len(recent_traces),
            'cognitive_load': self.cognitive_state.working_memory_load,
            'neural_activity': self.cognitive_state.neural_activity
        }
    
    def _optimize_working_memory(self):
        """Optimize working memory by moving unused items"""
        # Sort by access count and keep most accessed
        sorted_items = sorted(
            self.working_memory.items, 
            key=lambda x: x['metadata']['access_count'], 
            reverse=True
        )
        
        # Keep top items, move others to episodic consideration
        keep_items = sorted_items[:self.working_memory.capacity]
        move_items = sorted_items[self.working_memory.capacity:]
        
        self.working_memory.items = keep_items
        
        # Mark moved items for episodic transfer
        for item in move_items:
            item['metadata']['transfer_to_episodic'] = True
    
    def get_cognitive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cognitive system statistics"""
        return {
            'cognitive_state': self.cognitive_state.__dict__,
            'sensory_memory': {
                'buffer_size': len(self.sensory_memory.buffer),
                'capacity': self.sensory_memory.capacity,
                'attention_weights': len(self.sensory_memory.attention_weights)
            },
            'working_memory': {
                'items': len(self.working_memory.items),
                'capacity': self.working_memory.capacity,
                'current_task': self.working_memory.current_task
            },
            'episodic_memory': {
                'consolidation_queue': len(self.episodic_memory.consolidation_queue),
                'forgetting_curve_size': len(self.episodic_memory.forgetting_curve),
                'consolidation_threshold': self.episodic_memory.consolidation_threshold
            },
            'semantic_memory': {
                'concepts_count': len(self.semantic_memory.concept_network),
                'semantic_clusters': len(self.semantic_memory.semantic_clusters)
            },
            'memory_traces': len(self.memory_traces),
            'hauls_store_stats': self.hauls_store.get_stats()
        }


def main():
    """Cognitive Architecture CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SLO Cognitive Architecture - Stage 2')
    parser.add_argument('--test', action='store_true', help='Run cognitive architecture tests')
    parser.add_argument('--interactive', action='store_true', help='Interactive cognitive testing')
    parser.add_argument('--stats', action='store_true', help='Show cognitive statistics')
    
    args = parser.parse_args()
    
    if args.stats:
        # Show cognitive statistics
        cognitive = CognitiveArchitecture()
        stats = cognitive.get_cognitive_statistics()
        
        print("🧠 SLO Cognitive Architecture Statistics:")
        print("=" * 50)
        
        for category, data in stats.items():
            print(f"\n{category.upper()}:")
            if isinstance(data, dict):
                for key, value in data.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {data}")
    
    elif args.test:
        # Run cognitive tests
        test_cognitive_architecture()
    
    elif args.interactive:
        # Interactive mode
        run_interactive_cognitive_test()
    
    else:
        print("Use --stats, --test, or --interactive")


def test_cognitive_architecture():
    """Test cognitive architecture functionality"""
    print("🧪 Testing Cognitive Architecture...")
    
    cognitive = CognitiveArchitecture()
    
    # Test 1: Input processing
    print("\n1. Testing input processing...")
    test_inputs = [
        "This is an important message about safety protocols",
        "Quick note: check the system status",
        "I need to remember that quantum computing uses superposition",
        "Critical alert: security vulnerability discovered"
    ]
    
    processing_results = []
    for i, input_text in enumerate(test_inputs):
        result_id = cognitive.process_input(input_text, {'test_index': i})
        processing_results.append(result_id)
        print(f"  Processed input {i}: {result_id}")
    
    # Test 2: Memory retrieval
    print("\n2. Testing memory retrieval...")
    test_queries = [
        "safety protocols",
        "quantum computing",
        "security vulnerability"
    ]
    
    for query in test_queries:
        memories = cognitive.retrieve_memories(query)
        print(f"  Query: '{query}'")
        for memory_type, results in memories['memories'].items():
            if results:
                print(f"    {memory_type}: {len(results)} items")
            else:
                print(f"    {memory_type}: No results")
    
    # Test 3: Consolidation
    print("\n3. Testing memory consolidation...")
    consolidation = cognitive.consolidate_and_cleanup()
    print(f"  Consolidated {consolidation['consolidated_episodes']} episodes")
    print(f"  Analyzed {consolidation['memory_traces_analyzed']} memory traces")
    
    # Test 4: Statistics
    print("\n4. Getting cognitive statistics...")
    stats = cognitive.get_cognitive_statistics()
    
    performance_score = 0
    if stats['working_memory']['items'] > 0:
        performance_score += 25
        print("  ✅ Working memory active")
    if stats['episodic_memory']['consolidation_queue'] > 0:
        performance_score += 25
        print("  ✅ Episodic memory active")
    if stats['semantic_memory']['concepts_count'] > 0:
        performance_score += 25
        print("  ✅ Semantic memory active")
    if stats['cognitive_state']['neural_activity'] > 0.1:
        performance_score += 25
        print("  ✅ Neural activity detected")
    
    print(f"\n🎯 Cognitive Architecture Score: {performance_score}/100")
    
    if performance_score >= 75:
        print("🏆 EXCELLENT: Cognitive architecture fully functional")
    elif performance_score >= 50:
        print("✅ GOOD: Cognitive architecture functional with room for improvement")
    else:
        print("⚠️ NEEDS WORK: Cognitive architecture needs improvement")


def run_interactive_cognitive_test():
    """Interactive cognitive testing"""
    print("🧠 Interactive Cognitive Testing")
    print("Type 'quit' to exit")
    print("=" * 40)
    
    cognitive = CognitiveArchitecture()
    
    while True:
        try:
            user_input = input("\ncognitive> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Exiting interactive mode")
                break
            
            if not user_input:
                continue
            
            # Process input
            result_id = cognitive.process_input(user_input, {'interactive': True})
            print(f"📝 Processed: {result_id}")
            
            # Retrieve relevant memories
            memories = cognitive.retrieve_memories(user_input)
            print(f"🔍 Retrieved memories: {len(memories.get('memories', {}))} types")
            
            # Show working memory context
            working_context = cognitive.working_memory.get_relevant_context(user_input)
            if working_context:
                print(f"🧠 Working context: {len(working_context)} items")
            
        except KeyboardInterrupt:
            print("\n👋 Interrupted")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()