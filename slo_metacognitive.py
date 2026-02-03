#!/usr/bin/env python3
"""
SloughGPT Metacognitive Abilities
Self-monitoring, self-correction, and confidence assessment
"""

import time
import json
import math
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import sqlite3
from pathlib import Path
import re
from collections import Counter

class ContradictionType(Enum):
    """Types of contradictions that can be detected"""
    FACTUAL = "factual"          # "The sky is green" vs "The sky is blue"
    TEMPORAL = "temporal"        # "It happened yesterday" vs "It will happen tomorrow"  
    LOGICAL = "logical"           # "All birds can fly" vs "Penguins are birds and can't fly"
    NUMERICAL = "numerical"       # "The population is 1000" vs "The population is 10000"

@dataclass
class Contradiction:
    """Represents a detected contradiction"""
    source1: str
    source2: str
    contradiction_type: ContradictionType
    severity: float  # 0.0 to 1.0
    description: str
    detected_at: float

@dataclass
class SelfAssessment:
    """Represents a self-assessment of a response"""
    response_id: str
    confidence: float  # 0.0 to 1.0
    completeness: float  # 0.0 to 1.0
    accuracy: float  # 0.0 to 1.0
    contradictions: List[Contradiction]
    self_correction_applied: bool
    original_response: str
    corrected_response: str
    assessment_timestamp: float

class SloughGPTMetacognitive:
    """SloughGPT's ability to monitor and correct its own thinking"""
    
    def __init__(self, db_path: str = "slo_metacognitive.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
        
        # Metacognitive parameters
        self.confidence_threshold = 0.7
        self.contradiction_threshold = 0.6
        self.correction_confidence = 0.8
        
    def _init_database(self):
        """Initialize SQLite database for metacognitive data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS self_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                response_id TEXT UNIQUE,
                confidence REAL,
                completeness REAL,
                accuracy REAL,
                contradictions TEXT,
                self_correction_applied BOOLEAN,
                original_response TEXT,
                corrected_response TEXT,
                assessment_timestamp REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contradictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source1 TEXT,
                source2 TEXT,
                contradiction_type TEXT,
                severity REAL,
                description TEXT,
                detected_at REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def assess_confidence(self, response: str, retrieved_context: List[str]) -> float:
        """Assess confidence based on retrieved context and contradictions"""
        if not retrieved_context:
            # Low confidence without context
            return 0.3
        
        base_confidence = 0.5
        
        # Factor in context quality
        context_quality = self._assess_context_quality(retrieved_context)
        confidence = base_confidence + (context_quality * 0.3)
        
        # Check for contradictions in retrieved info
        contradictions = self._detect_contradictions(retrieved_context)
        if contradictions:
            # Reduce confidence based on contradiction severity
            avg_severity = statistics.mean([c.severity for c in contradictions])
            confidence -= avg_severity * 0.4
        
        return max(0.1, min(1.0, confidence))
    
    def _assess_context_quality(self, context: List[str]) -> float:
        """Assess the quality of retrieved context"""
        if not context:
            return 0.0
        
        quality_scores = []
        
        for doc in context:
            score = 0.5  # Base score
            
            # Length factor (too short or too long is suspicious)
            length = len(doc.split())
            if 50 <= length <= 500:
                score += 0.2
            elif length < 20 or length > 1000:
                score -= 0.2
            
            # Structure factor (well-structured is better)
            if any(marker in doc.lower() for marker in [':', ';', '•', '-', '1.', '2.']):
                score += 0.1
            
            # Authority factor (mentions of sources, studies, etc.)
            if any(keyword in doc.lower() for keyword in ['study', 'research', 'according to', 'source']):
                score += 0.1
            
            # Consistency factor (no obvious contradictions within doc)
            if not self._has_internal_contradictions(doc):
                score += 0.1
            
            quality_scores.append(max(0.0, min(1.0, score)))
        
        return statistics.mean(quality_scores)
    
    def _detect_contradictions(self, texts: List[str]) -> List[Contradiction]:
        """Detect contradictions in a set of texts"""
        contradictions = []
        
        # Compare each pair of texts
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                text1, text2 = texts[i], texts[j]
                
                # Look for various types of contradictions
                contradiction = self._find_contradiction(text1, text2)
                if contradiction:
                    contradictions.append(contradiction)
        
        return contradictions
    
    def _find_contradiction(self, text1: str, text2: str) -> Optional[Contradiction]:
        """Find contradiction between two texts"""
        
        # Extract key facts from both texts
        facts1 = self._extract_facts(text1)
        facts2 = self._extract_facts(text2)
        
        # Check for factual contradictions
        for fact1 in facts1:
            for fact2 in facts2:
                if self._are_contradictory_facts(fact1, fact2):
                    return Contradiction(
                        source1=text1[:100] + "...",
                        source2=text2[:100] + "...",
                        contradiction_type=ContradictionType.FACTUAL,
                        severity=0.8,
                        description=f"Contradictory facts: {fact1} vs {fact2}",
                        detected_at=time.time()
                    )
        
        return None
    
    def _extract_facts(self, text: str) -> List[str]:
        """Extract key facts from text (simplified)"""
        facts = []
        text_lower = text.lower()
        
        # Look for patterns that indicate facts
        import re
        
        # Numbers with units
        number_facts = re.findall(r'\b\d+(?:\.\d+)?\s*(?:%|degrees|celsius|fahrenheit|kg|lbs|meters|feet)\b', text_lower)
        facts.extend(number_facts)
        
        # Simple statements
        statement_patterns = [
            r'\b(?:is|are|was|were)\s+[^.]+\.',
            r'\b(?:has|have|had)\s+[^.]+\.',
            r'\b(?:can|could|should|would)\s+[^.]+\.'
        ]
        
        for pattern in statement_patterns:
            statements = re.findall(pattern, text_lower)
            facts.extend(statements)
        
        return list(set(facts))  # Remove duplicates
    
    def _are_contradictory_facts(self, fact1: str, fact2: str) -> bool:
        """Check if two facts contradict each other"""
        fact1, fact2 = fact1.lower(), fact2.lower()
        
        # Direct negations
        if f"not {fact1}" in fact2 or f"not {fact2}" in fact1:
            return True
        
        # Opposite states
        opposites = {
            'hot': ['cold', 'cool', 'freezing'],
            'big': ['small', 'tiny', 'little'], 
            'fast': ['slow', 'sluggish'],
            'high': ['low', 'down'],
            'true': ['false', 'fake'],
            'alive': ['dead', 'deceased']
        }
        
        for key, opposites in opposites.items():
            if key in fact1 and any(opp in fact2 for opp in opposites):
                return True
            if key in fact2 and any(opp in fact1 for opp in opposites):
                return True
        
        # Numerical contradictions
        numbers1 = re.findall(r'\d+(?:\.\d+)?', fact1)
        numbers2 = re.findall(r'\d+(?:\.\d+)?', fact2)
        
        if numbers1 and numbers2:
            num1 = float(numbers1[0])
            num2 = float(numbers2[0])
            
            # If numbers differ significantly, likely contradiction
            if abs(num1 - num2) / max(num1, num2) > 0.5:  # More than 50% difference
                return True
        
        return False
    
    def _has_internal_contradictions(self, text: str) -> bool:
        """Check if text contains internal contradictions"""
        text_lower = text.lower()
        
        # Look for "but" followed by contradictory statements
        contradiction_patterns = [
            r'\b(?:although|however|but|yet)\s+.*\b(?:although|however|but|yet)\b',
            r'\b(?:always|never)\s+.*\b(?:sometimes|never|always)\b',
            r'\b(?:all|every|each)\s+.*\b(?:some|few|none)\b',
            r'\b(?:only|just)\s+.*\b(?:also|and|plus)\b'
        ]
        
        for pattern in contradiction_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def measure_completeness(self, response: str, retrieved_context: List[str]) -> float:
        """Measure how complete the response is relative to retrieved context"""
        if not retrieved_context:
            return 0.5  # Neutral when no context
        
        # Extract key concepts from context
        context_concepts = self._extract_key_concepts(retrieved_context)
        response_concepts = self._extract_key_concepts([response])
        
        # Calculate coverage
        if not context_concepts:
            return 0.5
        
        covered_concepts = len([c for c in context_concepts if c in response_concepts])
        completeness = covered_concepts / len(context_concepts)
        
        # Factor in response elaboration
        response_length = len(response.split())
        expected_length = len(' '.join(retrieved_context)) // len(retrieved_context) * 2
        
        if response_length < expected_length * 0.3:
            completeness -= 0.2  # Too brief
        elif response_length > expected_length * 2:
            completeness -= 0.1  # Too verbose
        
        return max(0.0, min(1.0, completeness))
    
    def _extract_key_concepts(self, texts: List[str]) -> List[str]:
        """Extract key concepts from texts"""
        if not texts:
            return []
        
        combined_text = ' '.join(texts).lower()
        
        # Simple concept extraction (nouns and important terms)
        import re
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'been', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        # Extract potential concepts (simplified)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', combined_text)
        concepts = [w for w in words if w not in stop_words and len(w) > 4]
        
        # Count frequency and return most common
        concept_counts = Counter(concepts)
        
        # Return top concepts
        return [concept for concept, count in concept_counts.most_common(10)]
    
    def self_correct_response(self, initial_response: str, contradictions: List[Contradiction]) -> str:
        """Apply self-correction to initial response"""
        if not contradictions:
            return initial_response
        
        correction_confidence = 1.0 - max([c.severity for c in contradictions])
        
        if correction_confidence < self.correction_confidence:
            return initial_response  # Not confident enough to correct
        
        # Apply different correction strategies
        corrected_response = initial_response
        
        # Strategy 1: Add qualification for contradictions
        for contradiction in contradictions:
            if contradiction.contradiction_type == ContradictionType.FACTUAL:
                qualifier = self._generate_qualifier(contradiction)
                corrected_response = f"[QUALIFIED] {corrected_response}\n\n{qualifier}"
        
        # Strategy 2: Add uncertainty for detected issues
        if self._contains_uncertainty_patterns(initial_response):
            corrected_response = self._add_uncertainty_note(corrected_response)
        
        # Strategy 3: Add verification suggestions
        if contradictions:
            corrected_response += f"\n\n[NOTE] This information may contain contradictions. Please verify with additional sources."
        
        return corrected_response
    
    def _generate_qualifier(self, contradiction: Contradiction) -> str:
        """Generate an appropriate qualifier for a contradiction"""
        if contradiction.contradiction_type == ContradictionType.FACTUAL:
            return "Note: There may be conflicting information in the sources used. Further verification is recommended."
        elif contradiction.contradiction_type == ContradictionType.TEMPORAL:
            return "Note: There may be temporal inconsistencies in the information provided."
        elif contradiction.contradiction_type == ContradictionType.LOGICAL:
            return "Note: There may be logical inconsistencies in the reasoning presented."
        else:
            return "Note: The sources may contain incompatible information."
    
    def _contains_uncertainty_patterns(self, text: str) -> bool:
        """Check if response already shows uncertainty"""
        uncertainty_indicators = [
            'maybe', 'perhaps', 'possibly', 'might', 'could be', 'uncertain',
            'not sure', 'approximately', 'roughly', 'estimate', 'guess'
        ]
        
        return any(indicator in text.lower() for indicator in uncertainty_indicators)
    
    def _add_uncertainty_note(self, response: str) -> str:
        """Add appropriate uncertainty note to response"""
        if len(response.split()) < 50:  # Short response
            return f"[ADDITIONAL CONTEXT] {response}\n\nThis response is based on limited information and may benefit from additional context."
        else:
            return f"[ENHANCED] {response}\n\nAdditional verification of these points with multiple sources is recommended."
    
    def perform_self_assessment(self, response: str, retrieved_context: List[str], 
                            query: str) -> SelfAssessment:
        """Perform complete self-assessment of a response"""
        
        # Calculate metrics
        confidence = self.assess_confidence(response, retrieved_context)
        completeness = self.measure_completeness(response, retrieved_context)
        contradictions = self._detect_contradictions(retrieved_context)
        
        # Calculate overall accuracy estimate
        accuracy = confidence * (1 - max([c.severity for c in contradictions], default=0))
        
        # Apply self-correction if needed
        contradictions_present = len(contradictions) > 0
        self_correction_applied = False
        corrected_response = response
        
        if contradictions_present or confidence < self.confidence_threshold:
            corrected_response = self.self_correct_response(response, contradictions)
            self_correction_applied = True
        
        # Create assessment
        assessment = SelfAssessment(
            response_id=f"resp_{int(time.time())}",
            confidence=confidence,
            completeness=completeness,
            accuracy=accuracy,
            contradictions=contradictions,
            self_correction_applied=self_correction_applied,
            original_response=response,
            corrected_response=corrected_response,
            assessment_timestamp=time.time()
        )
        
        # Save to database
        self._save_assessment(assessment)
        
        return assessment
    
    def _save_assessment(self, assessment: SelfAssessment):
        """Save self-assessment to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO self_assessments 
            (response_id, confidence, completeness, accuracy, contradictions, 
             self_correction_applied, original_response, corrected_response, assessment_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            assessment.response_id,
            assessment.confidence,
            assessment.completeness,
            assessment.accuracy,
            json.dumps([asdict(c) for c in assessment.contradictions]),
            assessment.self_correction_applied,
            assessment.original_response,
            assessment.corrected_response,
            assessment.assessment_timestamp
        ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Saved self-assessment for {assessment.response_id}: confidence={assessment.confidence:.2f}, corrections_applied={assessment.self_correction_applied}")
    
    def get_learning_insights(self, days: int = 7) -> Dict[str, Any]:
        """Get insights from recent self-assessments"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = time.time() - (days * 24 * 3600)
        
        cursor.execute("""
            SELECT confidence, completeness, accuracy, self_correction_applied, contradictions
            FROM self_assessments 
            WHERE assessment_timestamp > ?
            ORDER BY assessment_timestamp DESC
        """, (cutoff_time,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {"message": "No recent assessments found"}
        
        # Calculate insights
        confidences = [row[0] for row in rows]
        completeness_scores = [row[1] for row in rows]
        accuracy_scores = [row[2] for row in rows]
        corrections_needed = [row[3] for row in rows]
        
        insights = {
            "assessment_count": len(rows),
            "average_confidence": statistics.mean(confidences) if confidences else 0,
            "average_completeness": statistics.mean(completeness_scores) if completeness_scores else 0,
            "average_accuracy": statistics.mean(accuracy_scores) if accuracy_scores else 0,
            "correction_rate": sum(corrections_needed) / len(corrections_needed),
            "confidence_trend": "improving" if len(confidences) > 1 and confidences[-1] > confidences[0] else "stable",
            "recommendations": []
        }
        
        # Generate recommendations
        if insights["average_confidence"] < 0.7:
            insights["recommendations"].append("Consider improving context retrieval to increase confidence")
        
        if insights["correction_rate"] > 0.3:
            insights["recommendations"].append("High correction rate detected - review fact-checking mechanisms")
        
        if insights["average_completeness"] < 0.6:
            insights["recommendations"].append("Responses may be incomplete - work on comprehensive coverage")
        
        return insights
    
    def get_metacognitive_status(self) -> Dict[str, Any]:
        """Get current metacognitive system status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent statistics
        cursor.execute("""
            SELECT COUNT(*) as total_assessments,
                   AVG(confidence) as avg_confidence,
                   AVG(accuracy) as avg_accuracy,
                   SUM(CASE WHEN self_correction_applied = 1 THEN 1 ELSE 0 END) as corrections_count
            FROM self_assessments
        """)
        
        stats = cursor.fetchone()
        conn.close()
        
        return {
            "system_status": "active",
            "total_assessments": stats[0] or 0,
            "average_confidence": stats[1] or 0,
            "average_accuracy": stats[2] or 0,
            "corrections_needed": stats[3] or 0,
            "settings": {
                "confidence_threshold": self.confidence_threshold,
                "contradiction_threshold": self.contradiction_threshold,
                "correction_confidence": self.correction_confidence
            }
        }

if __name__ == "__main__":
    # Example usage
    metacognitive = SloughGPTMetacognitive()
    
    # Example assessment
    response = "The Earth orbits the Sun and is the third planet from the Sun."
    context = [
        "The Earth orbits the Sun.",
        "Mars is the fourth planet from the Sun.",
        "Venus is the second planet from the Sun."
    ]
    query = "Tell me about Earth's position in the solar system"
    
    assessment = metacognitive.perform_self_assessment(response, context, query)
    
    print("Self-Assessment Results:")
    print(f"Confidence: {assessment.confidence:.2f}")
    print(f"Completeness: {assessment.completeness:.2f}")
    print(f"Accuracy: {assessment.accuracy:.2f}")
    print(f"Corrections Applied: {assessment.self_correction_applied}")
    
    if assessment.contradictions:
        print(f"Contradictions Found: {len(assessment.contradictions)}")
        for contradiction in assessment.contradictions:
            print(f"  - {contradiction.description}")
    
    # Get status
    status = metacognitive.get_metacognitive_status()
    print(f"\nSystem Status: {json.dumps(status, indent=2)}")