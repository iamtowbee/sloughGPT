#!/usr/bin/env python3
"""
SloughGPT Focused Cognitive System
Core: Reasoning + Thinking + Creativity
Stripped down to what we actually need
"""

import time
import asyncio
import json
import math
import random
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3

class ThinkingMode(Enum):
    """Different modes of thinking"""
    ANALYTICAL = "analytical"       # Logical, systematic breakdown
    CREATIVE = "creative"           # Brainstorming, innovative ideas
    CRITICAL = "critical"           # Evaluation, critique, improvement
    STRATEGIC = "strategic"         # Planning, optimization
    REFLECTIVE = "reflective"       # Self-analysis, learning

class ReasoningType(Enum):
    """Types of reasoning approaches"""
    DEDUCTIVE = "deductive"         # General to specific
    INDUCTIVE = "inductive"         # Specific to general  
    ABDUCTIVE = "abductive"         # Best explanation
    CAUSAL = "causal"              # Cause and effect
    ANALOGICAL = "analogical"       # Pattern recognition

@dataclass
class ThoughtProcess:
    """Represents a single thought process"""
    id: str
    mode: ThinkingMode
    reasoning_type: ReasoningType
    input_prompt: str
    thought_content: str
    confidence: float
    creativity_score: float
    logical_score: float
    timestamp: float
    processing_time: float

@dataclass
class CreativeIdea:
    """Represents a creative idea generated"""
    id: str
    concept: str
    description: str
    novelty_score: float
    feasibility_score: float
    creativity_score: float
    category: str
    tags: List[str]
    timestamp: float

@dataclass
class ReasoningChain:
    """Chain of reasoning steps"""
    id: str
    question: str
    reasoning_steps: List[str]
    conclusion: str
    confidence: float
    reasoning_type: ReasoningType
    evidence: List[str]
    timestamp: float

class SloughGPTCognitiveCore:
    """Core cognitive system: Reasoning + Thinking + Creativity"""
    
    def __init__(self, db_path: str = "slo_cognitive_core.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.thinking_history = []
        self.creative_ideas = []
        self.reasoning_chains = []
        
        # Cognitive parameters
        self.creativity_threshold = 0.7
        self.confidence_threshold = 0.6
        self.reasoning_depth = 3  # How many steps in reasoning chains
        
        self._init_database()
        
    def _init_database(self):
        """Initialize cognitive database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS thought_processes (
                id TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                reasoning_type TEXT NOT NULL,
                input_prompt TEXT NOT NULL,
                thought_content TEXT NOT NULL,
                confidence REAL NOT NULL,
                creativity_score REAL NOT NULL,
                logical_score REAL NOT NULL,
                timestamp REAL NOT NULL,
                processing_time REAL NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS creative_ideas (
                id TEXT PRIMARY KEY,
                concept TEXT NOT NULL,
                description TEXT NOT NULL,
                novelty_score REAL NOT NULL,
                feasibility_score REAL NOT NULL,
                creativity_score REAL NOT NULL,
                category TEXT NOT NULL,
                tags TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reasoning_chains (
                id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                reasoning_steps TEXT NOT NULL,
                conclusion TEXT NOT NULL,
                confidence REAL NOT NULL,
                reasoning_type TEXT NOT NULL,
                evidence TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    async def think(self, prompt: str, mode: ThinkingMode = ThinkingMode.ANALYTICAL) -> ThoughtProcess:
        """Core thinking process - analyze prompt from different cognitive modes"""
        start_time = time.time()
        thought_id = f"thought_{int(time.time() * 1000)}"
        
        # Determine best reasoning type for this prompt
        reasoning_type = self._select_reasoning_type(prompt, mode)
        
        # Generate thought based on mode and reasoning type
        thought_content = await self._generate_thought(prompt, mode, reasoning_type)
        
        # Calculate cognitive metrics
        confidence = self._calculate_confidence(prompt, thought_content, mode)
        creativity_score = self._calculate_creativity(thought_content, mode)
        logical_score = self._calculate_logical_score(thought_content, reasoning_type)
        
        processing_time = time.time() - start_time
        
        # Create thought process
        thought = ThoughtProcess(
            id=thought_id,
            mode=mode,
            reasoning_type=reasoning_type,
            input_prompt=prompt,
            thought_content=thought_content,
            confidence=confidence,
            creativity_score=creativity_score,
            logical_score=logical_score,
            timestamp=time.time(),
            processing_time=processing_time
        )
        
        # Store thought
        self.thinking_history.append(thought)
        self._save_thought(thought)
        
        return thought
    
    async def reason(self, question: str, reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE) -> ReasoningChain:
        """Dedicated reasoning process - step-by-step logical thinking"""
        reasoning_id = f"reasoning_{int(time.time() * 1000)}"
        
        # Generate reasoning steps
        reasoning_steps = await self._generate_reasoning_steps(question, reasoning_type)
        
        # Derive conclusion
        conclusion = await self._derive_conclusion(question, reasoning_steps, reasoning_type)
        
        # Gather evidence
        evidence = await self._gather_evidence(question, reasoning_steps)
        
        # Calculate confidence
        confidence = self._calculate_reasoning_confidence(reasoning_steps, evidence, reasoning_type)
        
        # Create reasoning chain
        reasoning_chain = ReasoningChain(
            id=reasoning_id,
            question=question,
            reasoning_steps=reasoning_steps,
            conclusion=conclusion,
            confidence=confidence,
            reasoning_type=reasoning_type,
            evidence=evidence,
            timestamp=time.time()
        )
        
        # Store reasoning
        self.reasoning_chains.append(reasoning_chain)
        self._save_reasoning(reasoning_chain)
        
        return reasoning_chain
    
    async def create_ideas(self, challenge: str, num_ideas: int = 5) -> List[CreativeIdea]:
        """Generate creative ideas for a challenge"""
        ideas = []
        
        for i in range(num_ideas):
            idea_id = f"idea_{int(time.time() * 1000)}_{i}"
            
            # Generate core concept
            concept = await self._generate_concept(challenge, i)
            
            # Elaborate with description
            description = await self._elaborate_idea(concept, challenge)
            
            # Calculate creative metrics
            novelty_score = self._calculate_novelty(concept, challenge)
            feasibility_score = self._calculate_feasibility(concept, description)
            creativity_score = self._calculate_overall_creativity(novelty_score, feasibility_score, description)
            
            # Categorize and tag
            category = self._categorize_idea(concept)
            tags = self._generate_tags(concept, description)
            
            # Create creative idea
            idea = CreativeIdea(
                id=idea_id,
                concept=concept,
                description=description,
                novelty_score=novelty_score,
                feasibility_score=feasibility_score,
                creativity_score=creativity_score,
                category=category,
                tags=tags,
                timestamp=time.time()
            )
            
            ideas.append(idea)
            self.creative_ideas.append(idea)
            self._save_idea(idea)
        
        return ideas
    
    async def cognitive_process(self, prompt: str, include_reasoning: bool = True, include_creativity: bool = True) -> Dict[str, Any]:
        """Complete cognitive process combining thinking, reasoning, and creativity"""
        start_time = time.time()
        
        result = {
            "prompt": prompt,
            "timestamp": start_time,
            "thoughts": [],
            "reasoning": None,
            "creative_ideas": [],
            "synthesis": None
        }
        
        # Multi-mode thinking
        thinking_modes = [ThinkingMode.ANALYTICAL, ThinkingMode.CRITICAL]
        if include_creativity:
            thinking_modes.append(ThinkingMode.CREATIVE)
        
        for mode in thinking_modes:
            thought = await self.think(prompt, mode)
            result["thoughts"].append(thought)
        
        # Reasoning process
        if include_reasoning:
            reasoning = await self.reason(prompt)
            result["reasoning"] = reasoning
        
        # Creative ideas
        if include_creativity:
            ideas = await self.create_ideas(prompt, 3)
            result["creative_ideas"] = ideas
        
        # Synthesize everything
        result["synthesis"] = await self._synthesize_cognitive_result(result)
        result["processing_time"] = time.time() - start_time
        
        return result
    
    def _select_reasoning_type(self, prompt: str, mode: ThinkingMode) -> ReasoningType:
        """Select best reasoning type based on prompt and mode"""
        prompt_lower = prompt.lower()
        
        if mode == ThinkingMode.ANALYTICAL:
            if "why" in prompt_lower or "cause" in prompt_lower:
                return ReasoningType.CAUSAL
            elif "pattern" in prompt_lower or "similar" in prompt_lower:
                return ReasoningType.ANALOGICAL
            else:
                return ReasoningType.DEDUCTIVE
        
        elif mode == ThinkingMode.CREATIVE:
            return ReasoningType.ABDUCTIVE  # Best for creative exploration
        
        elif mode == ThinkingMode.CRITICAL:
            return ReasoningType.DEDUCTIVE  # Good for evaluation
        
        else:
            return ReasoningType.INDUCTIVE  # Good general purpose
    
    async def _generate_thought(self, prompt: str, mode: ThinkingMode, reasoning_type: ReasoningType) -> str:
        """Generate thought content based on cognitive mode"""
        await asyncio.sleep(random.uniform(0.1, 0.3))  # Simulate thinking time
        
        if mode == ThinkingMode.ANALYTICAL:
            return f"Analytical breakdown of '{prompt}': Systematic analysis reveals key components: [1] Core elements, [2] Relationships, [3] Implications. Logical structure indicates: {reasoning_type.value} reasoning is optimal."
        
        elif mode == ThinkingMode.CREATIVE:
            return f"Creative exploration of '{prompt}': Innovative perspectives include: [1] Alternative interpretations, [2] Novel combinations, [3] Unconventional solutions. Creative potential suggests multiple pathways forward."
        
        elif mode == ThinkingMode.CRITICAL:
            return f"Critical evaluation of '{prompt}': Strengths and weaknesses identified: [1] Assumptions examined, [2] Logical gaps detected, [3] Quality assessment completed. Critical analysis reveals areas for improvement."
        
        elif mode == ThinkingMode.STRATEGIC:
            return f"Strategic planning for '{prompt}': Long-term implications and optimal pathways: [1] Resource allocation, [2] Risk assessment, [3] Success metrics. Strategic thinking indicates clear path forward."
        
        else:  # REFLECTIVE
            return f"Reflective consideration of '{prompt}': Meta-analysis of assumptions and learning opportunities: [1] Past experiences, [2] Current understanding, [3] Future growth areas."
    
    async def _generate_reasoning_steps(self, question: str, reasoning_type: ReasoningType) -> List[str]:
        """Generate step-by-step reasoning"""
        await asyncio.sleep(random.uniform(0.2, 0.5))
        
        if reasoning_type == ReasoningType.DEDUCTIVE:
            return [
                f"General principle: [Establish broad rule applicable to '{question}']",
                f"Specific observation: [Identify specific case details]",
                f"Logical deduction: [Apply principle to specific case]"
            ]
        
        elif reasoning_type == ReasoningType.INDUCTIVE:
            return [
                f"Specific observation 1: [First evidence about '{question}']",
                f"Specific observation 2: [Second evidence]",
                f"Pattern recognition: [Identify recurring pattern]",
                f"General conclusion: [Formulate general rule]"
            ]
        
        elif reasoning_type == ReasoningType.ABDUCTIVE:
            return [
                f"Observation: [Key observation about '{question}']",
                f"Possible explanations: [Generate multiple hypotheses]",
                f"Best explanation selection: [Choose most plausible theory]"
            ]
        
        elif reasoning_type == ReasoningType.CAUSAL:
            return [
                f"Effect identification: [What is the outcome in '{question}']",
                f"Cause analysis: [What leads to this effect]",
                f"Causal chain: [Step-by-step causal relationship]"
            ]
        
        else:  # ANALOGICAL
            return [
                f"Source domain: [Familiar situation]",
                f"Target mapping: [How this applies to '{question}']",
                f"Analogical inference: [Transfer knowledge from source to target]"
            ]
    
    async def _derive_conclusion(self, question: str, steps: List[str], reasoning_type: ReasoningType) -> str:
        """Derive conclusion from reasoning steps"""
        await asyncio.sleep(0.1)
        
        return f"Conclusion based on {reasoning_type.value} reasoning: Integrating the reasoning steps about '{question}' leads to the conclusion that [logical conclusion drawn from steps], with confidence derived from the strength of the reasoning chain."
    
    async def _gather_evidence(self, question: str, steps: List[str]) -> List[str]:
        """Gather supporting evidence for reasoning"""
        await asyncio.sleep(0.1)
        
        return [
            f"Evidence 1: [Empirical support for '{question}']",
            f"Evidence 2: [Logical consistency check]",
            f"Evidence 3: [Cross-validation with related domains]"
        ]
    
    async def _generate_concept(self, challenge: str, index: int) -> str:
        """Generate core concept for creative idea"""
        await asyncio.sleep(0.1)
        
        concepts = [
            f"Innovative approach to {challenge}",
            f"Radical reimagining of {challenge}",
            f"Hybrid solution for {challenge}",
            f"Disruptive method for {challenge}",
            f"Emergent possibility for {challenge}"
        ]
        
        return concepts[index % len(concepts)]
    
    async def _elaborate_idea(self, concept: str, challenge: str) -> str:
        """Elaborate on a creative concept"""
        await asyncio.sleep(0.1)
        
        return f"Detailed elaboration of '{concept}': This idea transforms '{challenge}' through [specific innovative mechanism], offering [unique benefits] while addressing [key challenges]. Implementation involves [practical steps] and leverages [supporting technologies/methods]."
    
    def _calculate_confidence(self, prompt: str, content: str, mode: ThinkingMode) -> float:
        """Calculate confidence in thought process"""
        base_confidence = 0.7
        
        # Adjust based on mode
        mode_multipliers = {
            ThinkingMode.ANALYTICAL: 0.9,
            ThinkingMode.CRITICAL: 0.8,
            ThinkingMode.CREATIVE: 0.6,
            ThinkingMode.STRATEGIC: 0.75,
            ThinkingMode.REFLECTIVE: 0.7
        }
        
        confidence = base_confidence * mode_multipliers.get(mode, 0.7)
        
        # Add some randomness
        confidence += random.uniform(-0.1, 0.1)
        
        return max(0.1, min(1.0, confidence))
    
    def _calculate_creativity(self, content: str, mode: ThinkingMode) -> float:
        """Calculate creativity score of content"""
        base_score = 0.3
        
        # Creative modes get higher scores
        if mode == ThinkingMode.CREATIVE:
            base_score = 0.8
        elif mode == ThinkingMode.STRATEGIC:
            base_score = 0.6
        
        # Add randomness
        base_score += random.uniform(-0.2, 0.2)
        
        return max(0.0, min(1.0, base_score))
    
    def _calculate_logical_score(self, content: str, reasoning_type: ReasoningType) -> float:
        """Calculate logical consistency score"""
        base_score = 0.7
        
        # Some reasoning types are inherently more logical
        if reasoning_type in [ReasoningType.DEDUCTIVE, ReasoningType.INDUCTIVE]:
            base_score = 0.9
        elif reasoning_type == ReasoningType.ABDUCTIVE:
            base_score = 0.5
        
        return max(0.0, min(1.0, base_score + random.uniform(-0.1, 0.1)))
    
    def _calculate_novelty(self, concept: str, challenge: str) -> float:
        """Calculate novelty score of concept"""
        # Simple heuristic: longer concepts and specific combinations are more novel
        base_score = 0.5 + (len(concept.split()) * 0.05)
        return max(0.0, min(1.0, base_score + random.uniform(-0.2, 0.2)))
    
    def _calculate_feasibility(self, concept: str, description: str) -> float:
        """Calculate feasibility score"""
        # More detailed descriptions are assumed more feasible
        base_score = 0.6 + (len(description.split()) * 0.01)
        return max(0.0, min(1.0, base_score + random.uniform(-0.1, 0.1)))
    
    def _calculate_overall_creativity(self, novelty: float, feasibility: float, description: str) -> float:
        """Calculate overall creativity score"""
        # Balance novelty with feasibility
        creativity = (novelty * 0.6 + feasibility * 0.4)
        return max(0.0, min(1.0, creativity + random.uniform(-0.1, 0.1))
    
    def _categorize_idea(self, concept: str) -> str:
        """Categorize creative idea"""
        concept_lower = concept.lower()
        
        if any(word in concept_lower for word in ["innovative", "new", "breakthrough"]):
            return "innovation"
        elif any(word in concept_lower for word in ["hybrid", "combined", "integrated"]):
            return "hybrid"
        elif any(word in concept_lower for word in ["method", "approach", "process"]):
            return "methodology"
        else:
            return "concept"
    
    def _generate_tags(self, concept: str, description: str) -> List[str]:
        """Generate tags for idea"""
        text = (concept + " " + description).lower()
        
        potential_tags = ["innovative", "creative", "practical", "strategic", "systematic", "disruptive"]
        tags = [tag for tag in potential_tags if tag in text]
        
        return tags[:3]  # Limit to 3 tags
    
    def _calculate_reasoning_confidence(self, steps: List[str], evidence: List[str], reasoning_type: ReasoningType) -> float:
        """Calculate confidence in reasoning chain"""
        base_confidence = 0.7
        
        # More steps and evidence increase confidence
        step_bonus = min(0.2, len(steps) * 0.05)
        evidence_bonus = min(0.2, len(evidence) * 0.05)
        
        # Some reasoning types are more reliable
        type_bonus = 0.1 if reasoning_type == ReasoningType.DEDUCTIVE else 0.0
        
        confidence = base_confidence + step_bonus + evidence_bonus + type_bonus
        return max(0.1, min(1.0, confidence + random.uniform(-0.05, 0.05))
    
    async def _synthesize_cognitive_result(self, result: Dict[str, Any]) -> str:
        """Synthesize all cognitive outputs into coherent response"""
        await asyncio.sleep(0.2)
        
        synthesis = f"Cognitive Analysis of: {result['prompt']}\n\n"
        
        # Add thinking perspectives
        if result["thoughts"]:
            synthesis += "=== THINKING PERSPECTIVES ===\n"
            for thought in result["thoughts"]:
                synthesis += f"• {thought.mode.value.upper()}: {thought.thought_content}\n"
                synthesis += f"  Confidence: {thought.confidence:.2f}, Creativity: {thought.creativity_score:.2f}\n\n"
        
        # Add reasoning chain
        if result["reasoning"]:
            reasoning = result["reasoning"]
            synthesis += "=== REASONING CHAIN ===\n"
            synthesis += f"Type: {reasoning.reasoning_type.value}\n"
            for i, step in enumerate(reasoning.reasoning_steps, 1):
                synthesis += f"{i}. {step}\n"
            synthesis += f"Conclusion: {reasoning.conclusion}\n"
            synthesis += f"Confidence: {reasoning.confidence:.2f}\n\n"
        
        # Add creative ideas
        if result["creative_ideas"]:
            synthesis += "=== CREATIVE IDEAS ===\n"
            for idea in result["creative_ideas"]:
                synthesis += f"• {idea.concept} ({idea.category})\n"
                synthesis += f"  {idea.description}\n"
                synthesis += f"  Creativity: {idea.creativity_score:.2f}, Novelty: {idea.novelty_score:.2f}\n\n"
        
        synthesis += "=== INTEGRATED SYNTHESIS ===\n"
        synthesis += f"Integrating analytical thinking, logical reasoning, and creative exploration, the most comprehensive understanding of '{result['prompt']}' emerges from combining systematic analysis with innovative perspectives and evidence-based conclusions."
        
        return synthesis
    
    def _save_thought(self, thought: ThoughtProcess):
        """Save thought to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO thought_processes 
            (id, mode, reasoning_type, input_prompt, thought_content, confidence, creativity_score, logical_score, timestamp, processing_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            thought.id, thought.mode.value, thought.reasoning_type.value,
            thought.input_prompt, thought.thought_content, thought.confidence,
            thought.creativity_score, thought.logical_score, thought.timestamp, thought.processing_time
        ))
        
        conn.commit()
        conn.close()
    
    def _save_reasoning(self, reasoning: ReasoningChain):
        """Save reasoning chain to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO reasoning_chains 
            (id, question, reasoning_steps, conclusion, confidence, reasoning_type, evidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            reasoning.id, reasoning.question, json.dumps(reasoning.reasoning_steps),
            reasoning.conclusion, reasoning.confidence, reasoning.reasoning_type.value,
            json.dumps(reasoning.evidence), reasoning.timestamp
        ))
        
        conn.commit()
        conn.close()
    
    def _save_idea(self, idea: CreativeIdea):
        """Save creative idea to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO creative_ideas 
            (id, concept, description, novelty_score, feasibility_score, creativity_score, category, tags, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            idea.id, idea.concept, idea.description, idea.novelty_score,
            idea.feasibility_score, idea.creativity_score, idea.category,
            json.dumps(idea.tags), idea.timestamp
        ))
        
        conn.commit()
        conn.close()
    
    def get_cognitive_status(self) -> Dict[str, Any]:
        """Get current cognitive system status"""
        return {
            "total_thoughts": len(self.thinking_history),
            "total_reasoning_chains": len(self.reasoning_chains),
            "total_creative_ideas": len(self.creative_ideas),
            "thinking_modes": [mode.value for mode in ThinkingMode],
            "reasoning_types": [rtype.value for rtype in ReasoningType],
            "creativity_threshold": self.creativity_threshold,
            "confidence_threshold": self.confidence_threshold,
            "reasoning_depth": self.reasoning_depth
        }

# Test the focused cognitive system
async def main():
    """Test the focused cognitive system"""
    cognitive = SloughGPTCognitiveCore()
    
    print("🧠 SloughGPT Focused Cognitive System")
    print("Core: Reasoning + Thinking + Creativity")
    print("=" * 50)
    
    # Test cognitive processing
    test_prompts = [
        "How can we improve renewable energy adoption?",
        "What makes human creativity unique?",
        "Design a better way to learn complex subjects"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📋 Test {i}: {prompt}")
        print("-" * 40)
        
        result = await cognitive.cognitive_process(prompt, include_reasoning=True, include_creativity=True)
        
        print(f"✅ Processing time: {result['processing_time']:.2f}s")
        print(f"✅ Thoughts generated: {len(result['thoughts'])}")
        print(f"✅ Reasoning confidence: {result['reasoning'].confidence:.2f}" if result['reasoning'] else "❌ No reasoning")
        print(f"✅ Creative ideas: {len(result['creative_ideas'])}")
        
        # Show synthesis summary
        synthesis = result['synthesis']
        print(f"\n📝 Synthesis Preview:")
        print(synthesis[:300] + "..." if len(synthesis) > 300 else synthesis)
    
    # Show status
    status = cognitive.get_cognitive_status()
    print(f"\n📊 Final Cognitive Status:")
    print(f"  Total thoughts: {status['total_thoughts']}")
    print(f"  Total reasoning chains: {status['total_reasoning_chains']}")
    print(f"  Total creative ideas: {status['total_creative_ideas']}")
    
    print(f"\n🎉 Focused Cognitive System Test Complete!")

if __name__ == "__main__":
    asyncio.run(main())