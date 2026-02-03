"""Advanced reasoning engine with multi-step logic and self-correction."""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import uuid


@dataclass
class ReasoningStep:
    step_id: str
    description: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence: float
    timestamp: datetime
    corrections: Optional[List[str]] = None


@dataclass
class ReasoningContext:
    context_id: str
    user_id: int
    conversation_history: List[Dict[str, Any]]
    current_prompt: str
    metadata: Dict[str, Any]
    created_at: datetime


@dataclass
class ReasoningResult:
    context_id: str
    final_answer: str
    reasoning_steps: List[ReasoningStep]
    confidence: float
    corrections_made: int
    processing_time: float
    timestamp: datetime


class ReasoningEngine:
    def __init__(self):
        self.active_contexts: Dict[str, ReasoningContext] = {}
        self.reasoning_history: List[ReasoningResult] = []
        self.confidence_threshold = 0.7
        self.max_steps = 10
        
    def create_context(self, user_id: int, prompt: str, 
                      conversation_history: Optional[List[Dict[str, Any]]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new reasoning context."""
        context_id = str(uuid.uuid4())
        
        context = ReasoningContext(
            context_id=context_id,
            user_id=user_id,
            conversation_history=conversation_history or [],
            current_prompt=prompt,
            metadata=metadata or {},
            created_at=datetime.now()
        )
        
        self.active_contexts[context_id] = context
        return context_id
    
    def reason(self, context_id: str) -> ReasoningResult:
        """Perform multi-step reasoning on a context."""
        if context_id not in self.active_contexts:
            raise ValueError(f"Context {context_id} not found")
        
        context = self.active_contexts[context_id]
        start_time = datetime.now()
        
        reasoning_steps = []
        current_data = {
            "prompt": context.current_prompt,
            "conversation_history": context.conversation_history
        }
        
        corrections_made = 0
        step_count = 0
        
        # Multi-step reasoning process
        while step_count < self.max_steps:
            step_result = self._perform_reasoning_step(current_data, step_count + 1)
            reasoning_steps.append(step_result)
            
            # Check if we have a confident answer
            if step_result.confidence >= self.confidence_threshold:
                break
            
            # Check if we need correction
            if step_result.corrections:
                corrections_made += len(step_result.corrections)
                current_data.update(step_result.output_data)
            else:
                current_data = step_result.output_data
            
            step_count += 1
        
        # Generate final answer
        final_answer = self._generate_final_answer(reasoning_steps, current_data)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = ReasoningResult(
            context_id=context_id,
            final_answer=final_answer,
            reasoning_steps=reasoning_steps,
            confidence=reasoning_steps[-1].confidence if reasoning_steps else 0.0,
            corrections_made=corrections_made,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
        self.reasoning_history.append(result)
        
        # Clean up context
        del self.active_contexts[context_id]
        
        return result
    
    def _perform_reasoning_step(self, input_data: Dict[str, Any], step_number: int) -> ReasoningStep:
        """Perform a single reasoning step."""
        step_id = str(uuid.uuid4())
        
        # Analyze the input and determine what type of reasoning is needed
        prompt = input_data.get("prompt", "")
        
        if self._is_factual_question(prompt):
            output = self._analyze_factual_question(prompt, input_data)
            description = "Analyzing factual information"
        elif self._is_creative_task(prompt):
            output = self._analyze_creative_task(prompt, input_data)
            description = "Generating creative response"
        elif self._is_logical_reasoning(prompt):
            output = self._analyze_logical_reasoning(prompt, input_data)
            description = "Performing logical reasoning"
        else:
            output = self._analyze_general_query(prompt, input_data)
            description = "General analysis"
        
        # Check for potential errors and suggest corrections
        corrections = self._identify_corrections(input_data, output)
        
        # Calculate confidence
        confidence = self._calculate_confidence(output, corrections)
        
        return ReasoningStep(
            step_id=step_id,
            description=description,
            input_data=input_data,
            output_data=output,
            confidence=confidence,
            timestamp=datetime.now(),
            corrections=corrections
        )
    
    def _is_factual_question(self, prompt: str) -> bool:
        """Check if prompt is asking for factual information."""
        factual_indicators = ["what is", "who is", "when did", "where is", "how many", "define", "explain"]
        prompt_lower = prompt.lower()
        return any(indicator in prompt_lower for indicator in factual_indicators)
    
    def _is_creative_task(self, prompt: str) -> bool:
        """Check if prompt is a creative task."""
        creative_indicators = ["write", "create", "generate", "imagine", "design", "compose"]
        prompt_lower = prompt.lower()
        return any(indicator in prompt_lower for indicator in creative_indicators)
    
    def _is_logical_reasoning(self, prompt: str) -> bool:
        """Check if prompt requires logical reasoning."""
        logical_indicators = ["why", "how does", "what if", "compare", "analyze", "evaluate"]
        prompt_lower = prompt.lower()
        return any(indicator in prompt_lower for indicator in logical_indicators)
    
    def _analyze_factual_question(self, prompt: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze factual questions."""
        # Simulate factual analysis
        return {
            "analysis_type": "factual",
            "key_entities": self._extract_entities(prompt),
            "search_strategy": "knowledge_base_lookup",
            "confidence_factors": ["entity_recognition", "context_relevance"],
            "partial_answer": f"Based on the question '{prompt}', I need to look up specific information."
        }
    
    def _analyze_creative_task(self, prompt: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze creative tasks."""
        return {
            "analysis_type": "creative",
            "creative_requirements": self._extract_creative_requirements(prompt),
            "style_suggestions": ["engaging", "informative", "well-structured"],
            "confidence_factors": ["creativity_score", "clarity_of_request"],
            "partial_answer": f"Creating a response to '{prompt}' with creative elements."
        }
    
    def _analyze_logical_reasoning(self, prompt: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze logical reasoning tasks."""
        return {
            "analysis_type": "logical",
            "logic_steps": self._identify_logic_steps(prompt),
            "reasoning_pattern": self._determine_reasoning_pattern(prompt),
            "confidence_factors": ["logical_consistency", "premise_validity"],
            "partial_answer": f"Applying logical reasoning to '{prompt}'."
        }
    
    def _analyze_general_query(self, prompt: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze general queries."""
        return {
            "analysis_type": "general",
            "query_intent": self._identify_intent(prompt),
            "response_strategy": "comprehensive_answer",
            "confidence_factors": ["query_clarity", "context_availability"],
            "partial_answer": f"Providing a comprehensive response to '{prompt}'."
        }
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities from text (simplified)."""
        # Simple entity extraction - in production, use NLP libraries
        words = text.split()
        entities = [word for word in words if len(word) > 3 and word.isalpha()]
        return entities[:5]  # Return top 5 entities
    
    def _extract_creative_requirements(self, prompt: str) -> List[str]:
        """Extract creative requirements from prompt."""
        requirements = []
        if "story" in prompt.lower():
            requirements.append("narrative_structure")
        if "poem" in prompt.lower():
            requirements.append("poetic_devices")
        if "code" in prompt.lower():
            requirements.append("syntax_correctness")
        return requirements
    
    def _identify_logic_steps(self, prompt: str) -> List[str]:
        """Identify logical reasoning steps."""
        return ["premise_identification", "logic_application", "conclusion_evaluation"]
    
    def _determine_reasoning_pattern(self, prompt: str) -> str:
        """Determine the reasoning pattern needed."""
        if "compare" in prompt.lower():
            return "comparative_analysis"
        elif "cause" in prompt.lower() or "effect" in prompt.lower():
            return "causal_reasoning"
        else:
            return "sequential_reasoning"
    
    def _identify_intent(self, prompt: str) -> str:
        """Identify user intent from prompt."""
        if "?" in prompt:
            return "question"
        elif any(word in prompt.lower() for word in ["help", "assist", "support"]):
            return "request_for_help"
        else:
            return "information_seeking"
    
    def _identify_corrections(self, input_data: Dict[str, Any], output: Dict[str, Any]) -> List[str]:
        """Identify potential corrections needed."""
        corrections = []
        
        # Check for missing key information
        if "partial_answer" in output and len(output["partial_answer"]) < 50:
            corrections.append("Provide more detailed information")
        
        # Check for confidence issues
        if output.get("confidence_factors", []):
            if len(output["confidence_factors"]) < 2:
                corrections.append("Add more confidence indicators")
        
        return corrections
    
    def _calculate_confidence(self, output: Dict[str, Any], corrections: List[str]) -> float:
        """Calculate confidence score for the reasoning step."""
        base_confidence = 0.8
        
        # Reduce confidence based on corrections
        for correction in corrections:
            base_confidence -= 0.1
        
        # Increase confidence based on confidence factors
        confidence_factors = output.get("confidence_factors", [])
        base_confidence += len(confidence_factors) * 0.05
        
        return max(0.0, min(1.0, base_confidence))
    
    def _generate_final_answer(self, steps: List[ReasoningStep], final_data: Dict[str, Any]) -> str:
        """Generate the final answer based on all reasoning steps."""
        if not steps:
            return "Unable to generate a response due to insufficient reasoning steps."
        
        # Use the last step's output as the primary answer
        last_step = steps[-1]
        partial_answer = last_step.output_data.get("partial_answer", "Analysis complete.")
        
        # Add context from previous steps if needed
        if len(steps) > 1 and steps[-2].output_data.get("key_entities"):
            entities = steps[-2].output_data["key_entities"]
            if entities:
                partial_answer += f" Key elements identified: {', '.join(entities)}."
        
        return partial_answer
    
    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get statistics about reasoning performance."""
        if not self.reasoning_history:
            return {
                "total_reasoning_sessions": 0,
                "avg_confidence": 0.0,
                "avg_processing_time": 0.0,
                "avg_corrections_per_session": 0.0
            }
        
        total_sessions = len(self.reasoning_history)
        avg_confidence = sum(r.confidence for r in self.reasoning_history) / total_sessions
        avg_processing_time = sum(r.processing_time for r in self.reasoning_history) / total_sessions
        avg_corrections = sum(r.corrections_made for r in self.reasoning_history) / total_sessions
        
        return {
            "total_reasoning_sessions": total_sessions,
            "avg_confidence": avg_confidence,
            "avg_processing_time": avg_processing_time,
            "avg_corrections_per_session": avg_corrections,
            "active_contexts": len(self.active_contexts)
        }