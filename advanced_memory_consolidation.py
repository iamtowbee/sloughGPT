#!/usr/bin/env python3
"""
Advanced Memory Consolidation Algorithms

Sophisticated memory management for long-term learning and pattern retention
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import json
from pathlib import Path
import sys

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from stage2_cognitive_architecture import CognitiveArchitecture, MemoryTrace

@dataclass
class MemoryTrace:
    """Enhanced memory trace with consolidation data"""
    content: str
    embedding: np.ndarray
    confidence: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    importance_score: float = 0.0
    decay_factor: float = 1.0
    consolidation_level: int = 0  # 0: short-term, 1: working, 2: long-term
    related_traces: List[str] = field(default_factory=list)

@dataclass
class ConsolidationPattern:
    """Pattern for memory consolidation"""
    pattern_id: str
    name: str
    frequency_weight: float
    recency_weight: float
    confidence_weight: float
    importance_threshold: float

class AdvancedMemoryConsolidation:
    """Advanced memory consolidation system"""
    
    def __init__(self, cognitive_arch: CognitiveArchitecture):
        self.cognitive_arch = cognitive_arch
        
        # Enhanced memory traces
        self.memory_traces: Dict[str, MemoryTrace] = {}
        self.consolidation_patterns = self._initialize_consolidation_patterns()
        
        # Consolidation parameters
        self.consolidation_interval = 300  # 5 minutes
        self.last_consolidation = time.time()
        self.consolidation_history: List[Dict[str, Any]] = []
        
        # Memory management
        self.max_memory_traces = 10000
        self.consolidation_threshold = 0.7
        self.decay_rate = 0.95  # Daily decay factor
        
        # Pattern recognition
        self.pattern_frequencies = defaultdict(int)
        self.pattern_recency = defaultdict(float)
        
        # Thread safety
        self.consolidation_lock = threading.Lock()
        
        print("🧠 Advanced Memory Consolidation initialized")
        print(f"📊 Max traces: {self.max_memory_traces}")
        print(f"⏰ Consolidation interval: {self.consolidation_interval}s")
    
    def _initialize_consolidation_patterns(self) -> List[ConsolidationPattern]:
        """Initialize memory consolidation patterns"""
        return [
            ConsolidationPattern(
                pattern_id="frequency_based",
                name="Frequency-Based Consolidation",
                frequency_weight=0.5,
                recency_weight=0.3,
                confidence_weight=0.2,
                importance_threshold=0.6
            ),
            ConsolidationPattern(
                pattern_id="recency_based",
                name="Recency-Based Consolidation", 
                frequency_weight=0.2,
                recency_weight=0.6,
                confidence_weight=0.2,
                importance_threshold=0.5
            ),
            ConsolidationPattern(
                pattern_id="confidence_weighted",
                name="Confidence-Weighted Consolidation",
                frequency_weight=0.2,
                recency_weight=0.2,
                confidence_weight=0.6,
                importance_threshold=0.8
            ),
            ConsolidationPattern(
                pattern_id="mixed",
                name="Mixed Consolidation",
                frequency_weight=0.33,
                recency_weight=0.33,
                confidence_weight=0.34,
                importance_threshold=0.65
            )
        ]
    
    def add_memory_trace(self, content: str, embedding: np.ndarray, confidence: float, 
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add enhanced memory trace with consolidation tracking"""
        current_time = time.time()
        trace_id = f"trace_{int(current_time * 1000)}_{hash(content) % 10000}"
        
        # Calculate initial importance score
        importance = self._calculate_initial_importance(content, confidence, metadata)
        
        # Create memory trace
        memory_trace = MemoryTrace(
            content=content,
            embedding=embedding,
            confidence=confidence,
            importance_score=importance,
            last_accessed=current_time,
            decay_factor=1.0
        )
        
        # Store with thread safety
        with self.consolidation_lock:
            self.memory_traces[trace_id] = memory_trace
            
            # Update pattern frequencies
            self._update_pattern_frequencies(content, trace_id)
            
            # Check if consolidation needed
            if len(self.memory_traces) >= self.max_memory_traces * 0.8:
                self._trigger_consolidation()
        
        return trace_id
    
    def _calculate_initial_importance(self, content: str, confidence: float, 
                                   metadata: Optional[Dict[str, Any]]) -> float:
        """Calculate initial importance score for memory trace"""
        importance = confidence * 0.4
        
        # Content length factor
        length_factor = min(len(content.split()) / 20.0, 1.0)
        importance += length_factor * 0.2
        
        # Keyword importance
        important_keywords = ["key", "important", "critical", "essential", "fundamental", "core", "main", "primary"]
        keyword_count = sum(1 for word in important_keywords if word.lower() in content.lower())
        importance += min(keyword_count * 0.1, 0.3)
        
        # Metadata importance
        if metadata:
            if metadata.get("epiphany", False):
                importance += 0.3  # Bonus for epiphanies
            if metadata.get("high_confidence", False) and confidence > 0.8:
                importance += 0.2
        
        return min(importance, 1.0)
    
    def _update_pattern_frequencies(self, content: str, trace_id: str):
        """Update pattern recognition frequencies"""
        # Extract content patterns
        content_lower = content.lower()
        
        # Update word patterns
        words = content_lower.split()
        for word in set(words):
            if len(word) > 3:  # Ignore short words
                self.pattern_frequencies[word] += 1
                self.pattern_recency[word] = time.time()
    
    def _trigger_consolidation(self):
        """Trigger memory consolidation process"""
        current_time = time.time()
        if current_time - self.last_consolidation < self.consolidation_interval:
            return
        
        print("🔄 Triggering memory consolidation...")
        
        # Run consolidation in background thread
        consolidation_thread = threading.Thread(
            target=self._perform_consolidation,
            daemon=True
        )
        consolidation_thread.start()
        
        self.last_consolidation = current_time
    
    def _perform_consolidation(self):
        """Perform actual memory consolidation"""
        with self.consolidation_lock:
            print(f"📚 Consolidating {len(self.memory_traces)} memory traces...")
            
            consolidation_start = time.time()
            
            # Apply decay to existing traces
            self._apply_memory_decay()
            
            # Calculate importance scores
            self._update_importance_scores()
            
            # Select traces for consolidation
            consolidation_candidates = self._select_consolidation_candidates()
            
            # Perform consolidation
            consolidated_traces = self._consolidate_memory_traces(consolidation_candidates)
            
            # Update memory store
            self._update_memory_store(consolidated_traces)
            
            # Record consolidation event
            consolidation_time = time.time() - consolidation_start
            self._record_consolidation_event(consolidated_traces, consolidation_time)
            
            print(f"✅ Consolidation complete in {consolidation_time:.2f}s")
            print(f"📊 {len(consolidated_traces)} traces consolidated")
    
    def _apply_memory_decay(self):
        """Apply time-based decay to memory traces"""
        current_time = time.time()
        days_elapsed = 1.0  # Assume daily consolidation
        
        for trace_id, trace in self.memory_traces.items():
            # Calculate decay based on time since last access
            time_since_access = (current_time - trace.last_accessed) / (24 * 3600)  # Days
            decay_factor = self.decay_rate ** time_since_access
            
            # Update trace
            trace.decay_factor = decay_factor
            trace.importance_score *= decay_factor
    
    def _update_importance_scores(self):
        """Update importance scores based on all consolidation patterns"""
        current_time = time.time()
        
        for pattern in self.consolidation_patterns:
            pattern_traces = self._filter_traces_by_pattern(pattern)
            
            for trace_id, trace in pattern_traces.items():
                # Calculate pattern-specific importance
                frequency_score = self._calculate_frequency_score(trace, pattern)
                recency_score = self._calculate_recency_score(trace, pattern, current_time)
                confidence_score = trace.confidence * pattern.confidence_weight
                
                # Combined importance
                pattern_importance = (
                    frequency_score * pattern.frequency_weight +
                    recency_score * pattern.recency_weight +
                    confidence_score * pattern.confidence_weight
                )
                
                # Update trace importance if higher than current
                if pattern_importance > trace.importance_score:
                    trace.importance_score = min(pattern_importance, 1.0)
    
    def _filter_traces_by_pattern(self, pattern: ConsolidationPattern) -> Dict[str, MemoryTrace]:
        """Filter memory traces based on consolidation pattern"""
        filtered_traces = {}
        
        for trace_id, trace in self.memory_traces.items():
            if trace.importance_score >= pattern.importance_threshold:
                filtered_traces[trace_id] = trace
        
        return filtered_traces
    
    def _calculate_frequency_score(self, trace: MemoryTrace, pattern: ConsolidationPattern) -> float:
        """Calculate frequency score for trace"""
        # Extract key terms from trace
        content_words = trace.content.lower().split()
        
        # Count pattern word matches
        pattern_matches = 0
        total_pattern_words = 0
        
        for word in set(content_words):
            if len(word) > 3:  # Focus on meaningful words
                total_pattern_words += 1
                if self.pattern_frequencies.get(word, 0) > 2:  # Frequent words
                    pattern_matches += 1
        
        return min(pattern_matches / max(total_pattern_words, 1), 1.0)
    
    def _calculate_recency_score(self, trace: MemoryTrace, pattern: ConsolidationPattern, 
                              current_time: float) -> float:
        """Calculate recency score for trace"""
        time_since_access = current_time - trace.last_accessed
        
        # Normalize to 0-1 range (more recent = higher score)
        max_age = 7 * 24 * 3600  # 7 days
        recency_score = max(0, 1 - (time_since_access / max_age))
        
        return recency_score
    
    def _select_consolidation_candidates(self) -> List[Tuple[str, MemoryTrace]]:
        """Select best candidates for consolidation"""
        # Sort by importance score
        sorted_traces = sorted(
            self.memory_traces.items(),
            key=lambda x: x[1].importance_score,
            reverse=True
        )
        
        # Select top candidates (20% of total, max 100)
        candidate_count = min(int(len(self.memory_traces) * 0.2), 100)
        candidates = sorted_traces[:candidate_count]
        
        print(f"🎯 Selected {len(candidates)} consolidation candidates")
        return candidates
    
    def _consolidate_memory_traces(self, candidates: List[Tuple[str, MemoryTrace]]) -> List[MemoryTrace]:
        """Consolidate selected memory traces"""
        consolidated_traces = []
        
        for trace_id, trace in candidates:
            # Promote consolidation level
            if trace.consolidation_level < 2:
                trace.consolidation_level = min(trace.consolidation_level + 1, 2)
                
                # Create consolidated trace with enhanced properties
                consolidated_trace = MemoryTrace(
                    content=trace.content,
                    embedding=trace.embedding,
                    confidence=trace.confidence,
                    importance_score=trace.importance_score,
                    consolidation_level=trace.consolidation_level,
                    decay_factor=1.0,  # Reset decay on consolidation
                    access_count=trace.access_count + 1,
                    last_accessed=time.time()
                )
                
                # Find related traces
                related_ids = self._find_related_traces(trace, candidates)
                consolidated_trace.related_traces = related_ids
                
                consolidated_traces.append(consolidated_trace)
        
        return consolidated_traces
    
    def _find_related_traces(self, target_trace: MemoryTrace, 
                             candidates: List[Tuple[str, MemoryTrace]]) -> List[str]:
        """Find related traces for consolidation"""
        related_ids = []
        
        target_embedding = target_trace.embedding
        target_words = set(target_trace.content.lower().split())
        
        for candidate_id, candidate_trace in candidates:
            if candidate_id != f"trace_{target_trace.content}_{hash(target_trace.content) % 10000}":
                # Calculate semantic similarity
                similarity = np.dot(target_embedding, candidate_trace.embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(candidate_trace.embedding) + 1e-8
                )
                
                # Check word overlap
                candidate_words = set(candidate_trace.content.lower().split())
                word_overlap = len(target_words.intersection(candidate_words))
                
                # Related if high similarity or word overlap
                if similarity > 0.7 or word_overlap > 3:
                    related_ids.append(candidate_id)
        
        return related_ids[:5]  # Top 5 related traces
    
    def _update_memory_store(self, consolidated_traces: List[MemoryTrace]):
        """Update memory store with consolidated traces"""
        for trace in consolidated_traces:
            # Find original trace and update
            original_id = f"trace_{int(trace.last_accessed - 86400) * 1000}_{hash(trace.content) % 10000}"
            
            if original_id in self.memory_traces:
                # Update existing trace
                self.memory_traces[original_id] = trace
            else:
                # Add new trace
                new_id = f"consolidated_{int(time.time() * 1000)}_{hash(trace.content) % 10000}"
                self.memory_traces[new_id] = trace
        
        # Remove traces below importance threshold
        traces_to_remove = []
        for trace_id, trace in self.memory_traces.items():
            if trace.importance_score < self.consolidation_threshold * 0.5:  # Remove very low importance
                traces_to_remove.append(trace_id)
        
        for trace_id in traces_to_remove:
            del self.memory_traces[trace_id]
        
        print(f"🗑️  Removed {len(traces_to_remove)} low-importance traces")
        print(f"📚 Memory store updated: {len(self.memory_traces)} total traces")
    
    def _record_consolidation_event(self, consolidated_traces: List[MemoryTrace], 
                                 consolidation_time: float):
        """Record consolidation event for analysis"""
        event = {
            "timestamp": time.time(),
            "consolidation_time": consolidation_time,
            "traces_consolidated": len(consolidated_traces),
            "memory_size_before": len(self.memory_traces) + len(consolidated_traces),
            "memory_size_after": len(self.memory_traces),
            "average_importance": np.mean([t.importance_score for t in consolidated_traces]),
            "consolidation_patterns_used": self._get_active_patterns(),
            "decay_applied": self.decay_rate
        }
        
        self.consolidation_history.append(event)
        
        # Keep only last 50 consolidation events
        if len(self.consolidation_history) > 50:
            self.consolidation_history = self.consolidation_history[-50:]
    
    def _get_active_patterns(self) -> List[str]:
        """Get currently active consolidation patterns"""
        # Based on current memory state, return which patterns are most active
        active_patterns = []
        
        # Analyze current memory distribution
        consolidation_levels = [trace.consolidation_level for trace in self.memory_traces.values()]
        
        if np.mean(consolidation_levels) > 1.5:
            active_patterns.append("confidence_weighted")
        if len(self.memory_traces) > self.max_memory_traces * 0.7:
            active_patterns.append("frequency_based")
        if np.mean([trace.importance_score for trace in self.memory_traces.values()]) > 0.8:
            active_patterns.append("mixed")
        
        return active_patterns if active_patterns else ["recency_based"]
    
    def get_consolidation_report(self) -> Dict[str, Any]:
        """Get comprehensive consolidation report"""
        with self.consolidation_lock:
            # Calculate current statistics
            total_traces = len(self.memory_traces)
            consolidation_levels = [trace.consolidation_level for trace in self.memory_traces.values()]
            importance_scores = [trace.importance_score for trace in self.memory_traces.values()]
            
            # Pattern analysis
            common_patterns = self._get_common_patterns()
            
            report = {
                "memory_statistics": {
                    "total_traces": total_traces,
                    "consolidated_traces": sum(1 for level in consolidation_levels if level >= 1),
                    "average_importance": np.mean(importance_scores) if importance_scores else 0,
                    "average_consolidation_level": np.mean(consolidation_levels) if consolidation_levels else 0,
                    "memory_utilization": total_traces / self.max_memory_traces
                },
                "consolidation_activity": {
                    "last_consolidation": self.last_consolidation,
                    "total_consolidations": len(self.consolidation_history),
                    "average_consolidation_time": np.mean([e["consolidation_time"] for e in self.consolidation_history]) if self.consolidation_history else 0,
                    "active_patterns": common_patterns
                },
                "pattern_frequencies": dict(list(self.pattern_frequencies.most_common(20))),
                "recent_consolidations": self.consolidation_history[-5:] if self.consolidation_history else []
            }
            
            return report
    
    def _get_common_patterns(self) -> List[str]:
        """Get most common consolidation patterns"""
        pattern_counts = defaultdict(int)
        
        for event in self.consolidation_history:
            for pattern in event.get("consolidation_patterns_used", []):
                pattern_counts[pattern] += 1
        
        # Return top 3 patterns
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        return [pattern for pattern, count in sorted_patterns[:3]]
    
    def optimize_consolidation_parameters(self):
        """Dynamically optimize consolidation parameters based on performance"""
        if len(self.consolidation_history) < 10:
            return  # Not enough data
        
        recent_events = self.consolidation_history[-10:]
        avg_consolidation_time = np.mean([e["consolidation_time"] for e in recent_events])
        memory_utilization = np.mean([e["memory_utilization"] for e in recent_events])
        
        # Adjust consolidation interval based on performance
        if avg_consolidation_time > 5.0:  # Taking too long
            self.consolidation_interval = min(self.consolidation_interval * 1.2, 600)  # Increase interval
            print(f"⏱️  Increasing consolidation interval to {self.consolidation_interval}s")
        elif avg_consolidation_time < 2.0:  # Very fast
            self.consolidation_interval = max(self.consolidation_interval * 0.8, 60)  # Decrease interval
            print(f"⚡ Decreasing consolidation interval to {self.consolidation_interval}s")
        
        # Adjust importance threshold based on memory utilization
        if memory_utilization > 0.9:  # High utilization
            self.consolidation_threshold = min(self.consolidation_threshold + 0.05, 0.9)
            print(f"📈 Raising importance threshold to {self.consolidation_threshold}")
        elif memory_utilization < 0.5:  # Low utilization
            self.consolidation_threshold = max(self.consolidation_threshold - 0.05, 0.5)
            print(f"📉 Lowering importance threshold to {self.consolidation_threshold}")

def main():
    """Test advanced memory consolidation"""
    print("🧠 Advanced Memory Consolidation Test")
    print("=" * 50)
    
    # Initialize with cognitive architecture
    from stage2_cognitive_architecture import CognitiveArchitecture
    cognitive_arch = CognitiveArchitecture()
    
    # Create consolidation system
    consolidation = AdvancedMemoryConsolidation(cognitive_arch)
    
    # Add test memory traces
    test_traces = [
        ("Hamlet is the Prince of Denmark who seeks revenge for his father's death", np.random.rand(64), 0.9),
        ("To be or not to be is the fundamental question of human existence", np.random.rand(64), 0.95),
        ("Shakespeare uses dramatic irony to create multiple layers of meaning", np.random.rand(64), 0.85),
        ("The balcony scene symbolizes love and forbidden desire", np.random.rand(64), 0.8),
        ("Cognitive dissonance occurs when actions contradict beliefs", np.random.rand(64), 0.75)
    ]
    
    print("📝 Adding test memory traces...")
    for i, (content, embedding, confidence) in enumerate(test_traces, 1):
        metadata = {"test_trace": True, "sequence": i}
        trace_id = consolidation.add_memory_trace(content, embedding, confidence, metadata)
        print(f"  {i}. {content[:50]}... (ID: {trace_id})")
    
    print(f"📊 Total traces added: {len(consolidation.memory_traces)}")
    
    # Wait for consolidation
    print("\n⏰ Waiting for automatic consolidation...")
    time.sleep(2)  # Allow time to pass
    
    # Trigger manual consolidation
    consolidation._trigger_consolidation()
    time.sleep(1)  # Allow consolidation to complete
    
    # Get report
    report = consolidation.get_consolidation_report()
    print("\n📊 Consolidation Report:")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()