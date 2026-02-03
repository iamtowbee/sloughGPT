"""
SloughGPT Advanced Reasoning Engine
Multi-step reasoning with chain-of-thought, self-correction, and logical inference
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from sloughgpt.core.logging_system import get_logger, timer
from sloughgpt.core.performance import get_performance_optimizer

class ReasoningType(Enum):
    """Types of reasoning approaches"""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    STEP_BY_STEP = "step_by_step"
    LOGICAL_INFERENCE = "logical_inference"
    ANALOGICAL_REASONING = "analogical_reasoning"
    CAUSAL_REASONING = "causal_reasoning"
    SELF_CORRECTION = "self_correction"
    METACOGNITIVE = "metacognitive"
    TREE_OF_THOUGHTS = "tree_of_thoughts"

class ReasoningStatus(Enum):
    """Status of reasoning process"""
    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    CORRECTING = "correcting"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class ReasoningStep:
    """Individual reasoning step in the chain"""
    step_id: int
    reasoning_type: ReasoningType
    prompt: str
    context: Dict[str, Any] = field(default_factory=dict)
    input_data: Any = None
    output_data: Any = None
    confidence: float = 0.0
    tokens_used: int = 0
    duration_ms: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary"""
        return {
            "step_id": self.step_id,
            "reasoning_type": self.reasoning_type.value,
            "prompt": self.prompt,
            "context": self.context,
            "input_data": str(self.input_data) if self.input_data else None,
            "output_data": str(self.output_data) if self.output_data else None,
            "confidence": self.confidence,
            "tokens_used": self.tokens_used,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata
        }

@dataclass
class ReasoningChain:
    """Complete reasoning chain with multiple steps"""
    chain_id: str
    original_query: str
    reasoning_type: ReasoningType
    steps: List[ReasoningStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    confidence: float = 0.0
    total_tokens: int = 0
    total_duration: float = 0.0
    status: ReasoningStatus = ReasoningStatus.INITIALIZING
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_successful(self) -> bool:
        """Check if reasoning chain completed successfully"""
        return self.status == ReasoningStatus.COMPLETED and self.final_answer is not None
    
    @property
    def average_confidence(self) -> float:
        """Calculate average confidence across all steps"""
        if not self.steps:
            return 0.0
        return sum(step.confidence for step in self.steps) / len(self.steps)
    
    def add_step(self, step: ReasoningStep) -> None:
        """Add a step to the reasoning chain"""
        self.steps.append(step)
        self.total_tokens += step.tokens_used
        self.total_duration += step.duration_ms
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chain to dictionary"""
        return {
            "chain_id": self.chain_id,
            "original_query": self.original_query,
            "reasoning_type": self.reasoning_type.value,
            "steps": [step.to_dict() for step in self.steps],
            "final_answer": self.final_answer,
            "confidence": self.confidence,
            "total_tokens": self.total_tokens,
            "total_duration": self.total_duration,
            "status": self.status.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata
        }

class ReasoningStrategy(ABC):
    """Abstract base class for reasoning strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(f"reasoning_{self.__class__.__name__.lower()}")
    
    @abstractmethod
    async def plan_reasoning(self, query: str, context: Dict[str, Any]) -> List[ReasoningStep]:
        """Plan the reasoning steps"""
        pass
    
    @abstractmethod
    async def execute_step(self, step: ReasoningStep) -> ReasoningStep:
        """Execute a single reasoning step"""
        pass
    
    @abstractmethod
    async def evaluate_chain(self, chain: ReasoningChain) -> bool:
        """Evaluate the quality and correctness of the reasoning chain"""
        pass

class ChainOfThoughtStrategy(ReasoningStrategy):
    """Chain-of-Thought reasoning strategy"""
    
    async def plan_reasoning(self, query: str, context: Dict[str, Any]) -> List[ReasoningStep]:
        """Plan CoT steps: decompose problem into logical steps"""
        self.logger.info(f"Planning CoT reasoning for query: {query[:100]}...")
        
        steps = []
        
        # Step 1: Understand the problem
        steps.append(ReasoningStep(
            step_id=1,
            reasoning_type=ReasoningType.CHAIN_OF_THOUGHT,
            prompt=f"Let me think step by step to solve: {query}\n\nFirst, what is the core problem here?",
            context=context,
            metadata={"phase": "understanding"}
        ))
        
        # Step 2: Break down the problem
        steps.append(ReasoningStep(
            step_id=2,
            reasoning_type=ReasoningType.CHAIN_OF_THOUGHT,
            prompt="Now, let me break this down into smaller, manageable pieces:",
            context=context,
            metadata={"phase": "decomposition"}
        ))
        
        # Step 3: Analyze each piece
        steps.append(ReasoningStep(
            step_id=3,
            reasoning_type=ReasoningType.CHAIN_OF_THOUGHT,
            prompt="Let me analyze each piece systematically:",
            context=context,
            metadata={"phase": "analysis"}
        ))
        
        # Step 4: Synthesize solution
        steps.append(ReasoningStep(
            step_id=4,
            reasoning_type=ReasoningType.CHAIN_OF_THOUGHT,
            prompt="Based on my analysis, let me synthesize a complete solution:",
            context=context,
            metadata={"phase": "synthesis"}
        ))
        
        # Step 5: Verify solution
        steps.append(ReasoningStep(
            step_id=5,
            reasoning_type=ReasoningType.CHAIN_OF_THOUGHT,
            prompt="Finally, let me verify that my solution is correct and complete:",
            context=context,
            metadata={"phase": "verification"}
        ))
        
        return steps
    
    async def execute_step(self, step: ReasoningStep) -> ReasoningStep:
        """Execute a CoT reasoning step"""
        start_time = time.time()
        
        try:
            # Mock model execution - in real implementation, this would call the LLM
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Generate mock response based on step phase
            phase = step.metadata.get("phase", "general")
            responses = {
                "understanding": "The problem involves [analysis of core requirements and constraints].",
                "decomposition": "The problem can be broken down into: 1) [component 1], 2) [component 2], 3) [component 3].",
                "analysis": "Analyzing each component: [detailed analysis with logical connections].",
                "synthesis": "Synthesizing the solution: [comprehensive answer addressing all components].",
                "verification": "Verification: [check against original problem, confirming completeness and correctness]."
            }
            
            response = responses.get(phase, "Proceeding with logical reasoning.")
            
            step.output_data = response
            step.confidence = 0.85 + (step.step_id * 0.02)  # Increasing confidence
            step.success = True
            step.tokens_used = 150 + (step.step_id * 25)
            step.duration_ms = (time.time() - start_time) * 1000
            
            self.logger.debug(f"CoT step {step.step_id} completed successfully")
            
        except Exception as e:
            step.success = False
            step.error_message = str(e)
            step.duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"CoT step {step.step_id} failed: {e}")
        
        return step
    
    async def evaluate_chain(self, chain: ReasoningChain) -> bool:
        """Evaluate CoT chain quality"""
        self.logger.info("Evaluating CoT reasoning chain quality")
        
        # Check if all steps completed successfully
        all_steps_successful = all(step.success for step in chain.steps)
        
        # Check confidence threshold
        min_confidence = self.config.get("min_confidence", 0.8)
        avg_confidence = chain.average_confidence
        
        # Check reasoning coherence (simplified)
        coherence_score = self._calculate_coherence(chain.steps)
        
        evaluation = (
            all_steps_successful and
            avg_confidence >= min_confidence and
            coherence_score >= 0.7
        )
        
        self.logger.info(f"CoT evaluation: successful={evaluation}, confidence={avg_confidence:.2f}, coherence={coherence_score:.2f}")
        
        return evaluation
    
    def _calculate_coherence(self, steps: List[ReasoningStep]) -> float:
        """Calculate coherence score for reasoning steps"""
        if len(steps) < 2:
            return 1.0
        
        # Simplified coherence calculation based on step progression
        coherence_scores = []
        
        for i in range(1, len(steps)):
            # Check if steps follow logical progression
            current_phase = steps[i].metadata.get("phase", "")
            previous_phase = steps[i-1].metadata.get("phase", "")
            
            # Define expected phase transitions
            valid_transitions = {
                "understanding": ["decomposition"],
                "decomposition": ["analysis"], 
                "analysis": ["synthesis"],
                "synthesis": ["verification"],
                "verification": []
            }
            
            if current_phase in valid_transitions.get(previous_phase, []):
                coherence_scores.append(1.0)
            else:
                coherence_scores.append(0.5)  # Partial coherence for unexpected transitions
        
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0

class LogicalInferenceStrategy(ReasoningStrategy):
    """Logical inference reasoning strategy"""
    
    async def plan_reasoning(self, query: str, context: Dict[str, Any]) -> List[ReasoningStep]:
        """Plan logical inference steps"""
        self.logger.info(f"Planning logical inference for: {query[:100]}...")
        
        steps = []
        
        # Step 1: Identify premises
        steps.append(ReasoningStep(
            step_id=1,
            reasoning_type=ReasoningType.LOGICAL_INFERENCE,
            prompt=f"Analyze the following query and identify key premises and facts: {query}",
            context=context,
            metadata={"phase": "premise_identification"}
        ))
        
        # Step 2: Apply logical rules
        steps.append(ReasoningStep(
            step_id=2,
            reasoning_type=ReasoningType.LOGICAL_INFERENCE,
            prompt="Apply relevant logical rules (deduction, induction, abduction) to these premises:",
            context=context,
            metadata={"phase": "rule_application"}
        ))
        
        # Step 3: Derive conclusions
        steps.append(ReasoningStep(
            step_id=3,
            reasoning_type=ReasoningType.LOGICAL_INFERENCE,
            prompt="Based on the premises and logical rules, derive valid conclusions:",
            context=context,
            metadata={"phase": "conclusion_derivation"}
        ))
        
        # Step 4: Check for fallacies
        steps.append(ReasoningStep(
            step_id=4,
            reasoning_type=ReasoningType.LOGICAL_INFERENCE,
            prompt="Check the reasoning for logical fallacies or invalid inferences:",
            context=context,
            metadata={"phase": "fallacy_detection"}
        ))
        
        return steps
    
    async def execute_step(self, step: ReasoningStep) -> ReasoningStep:
        """Execute logical inference step"""
        start_time = time.time()
        
        try:
            await asyncio.sleep(0.08)  # Simulate processing
            
            phase = step.metadata.get("phase", "general")
            responses = {
                "premise_identification": "Premises identified: [list of factual statements and assumptions].",
                "rule_application": "Applying logical rules: [specific rules and their applications].",
                "conclusion_derivation": "Logical conclusions: [derived inferences with justification].",
                "fallacy_detection": "Fallacy check: [analysis for potential logical errors]."
            }
            
            response = responses.get(phase, "Logical reasoning in progress.")
            
            step.output_data = response
            step.confidence = 0.88 + (step.step_id * 0.03)
            step.success = True
            step.tokens_used = 120 + (step.step_id * 30)
            step.duration_ms = (time.time() - start_time) * 1000
            
        except Exception as e:
            step.success = False
            step.error_message = str(e)
            step.duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Logical inference step {step.step_id} failed: {e}")
        
        return step
    
    async def evaluate_chain(self, chain: ReasoningChain) -> bool:
        """Evaluate logical inference chain"""
        self.logger.info("Evaluating logical inference chain")
        
        # Check logical validity
        all_steps_successful = all(step.success for step in chain.steps)
        avg_confidence = chain.average_confidence
        min_confidence = self.config.get("min_confidence", 0.85)
        
        # Check for contradictions
        contradiction_score = self._check_contradictions(chain.steps)
        
        evaluation = (
            all_steps_successful and
            avg_confidence >= min_confidence and
            contradiction_score <= 0.2
        )
        
        self.logger.info(f"Logical inference evaluation: {evaluation}, confidence={avg_confidence:.2f}, contradiction_score={contradiction_score:.2f}")
        
        return evaluation
    
    def _check_contradictions(self, steps: List[ReasoningStep]) -> float:
        """Check for logical contradictions between steps"""
        # Simplified contradiction detection
        contradictions = 0
        
        for i in range(len(steps)):
            for j in range(i + 1, len(steps)):
                # In a real implementation, this would analyze actual content
                # For now, simulate low contradiction rate
                if steps[i].success and steps[j].success:
                    # Random small chance of contradiction for demo
                    import random
                    if random.random() < 0.1:
                        contradictions += 1
        
        return contradictions / (len(steps) * (len(steps) - 1) / 2) if len(steps) > 1 else 0.0

class AdvancedReasoningEngine:
    """Main reasoning engine with multiple strategies and capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("advanced_reasoning_engine")
        self.optimizer = get_performance_optimizer()
        
        # Initialize reasoning strategies
        self.strategies = {
            ReasoningType.CHAIN_OF_THOUGHT: ChainOfThoughtStrategy(config),
            ReasoningType.LOGICAL_INFERENCE: LogicalInferenceStrategy(config),
            # Additional strategies can be added here
        }
        
        # Cache for reasoning chains
        self._cache = {}
        
        # Performance tracking
        self._performance_stats = {
            "total_reasoning_requests": 0,
            "successful_reasoning": 0,
            "average_duration": 0.0,
            "average_tokens": 0
        }
    
    async def reason(self, query: str, reasoning_type: ReasoningType = ReasoningType.CHAIN_OF_THOUGHT,
                   context: Optional[Dict[str, Any]] = None) -> ReasoningChain:
        """Perform advanced reasoning on the given query"""
        
        with timer("reasoning_request"):
            self.logger.info(f"Starting advanced reasoning: {query[:100]}...")
            
            # Create reasoning chain
            chain = ReasoningChain(
                chain_id=f"chain_{int(time.time() * 1000)}",
                original_query=query,
                reasoning_type=reasoning_type,
                context=context or {}
            )
            
            chain.status = ReasoningStatus.PLANNING
            
            try:
                # Get appropriate strategy
                strategy = self.strategies.get(reasoning_type)
                if not strategy:
                    raise ValueError(f"Unsupported reasoning type: {reasoning_type}")
                
                # Plan reasoning steps
                steps = await strategy.plan_reasoning(query, context or {})
                self.logger.info(f"Planned {len(steps)} reasoning steps")
                
                chain.status = ReasoningStatus.EXECUTING
                
                # Execute reasoning steps
                for step in steps:
                    executed_step = await strategy.execute_step(step)
                    chain.add_step(executed_step)
                    
                    # Early termination if step fails
                    if not executed_step.success:
                        self.logger.warning(f"Reasoning step {executed_step.step_id} failed: {executed_step.error_message}")
                        break
                
                chain.status = ReasoningStatus.EVALUATING
                
                # Evaluate reasoning chain
                chain.success = await strategy.evaluate_chain(chain)
                chain.confidence = chain.average_confidence
                
                if chain.success:
                    # Synthesize final answer from last successful step
                    successful_steps = [s for s in chain.steps if s.success]
                    if successful_steps:
                        chain.final_answer = successful_steps[-1].output_data
                        chain.status = ReasoningStatus.COMPLETED
                else:
                    chain.status = ReasoningStatus.FAILED
                
                chain.completed_at = time.time()
                
                # Update performance stats
                self._update_performance_stats(chain)
                
                self.logger.info(f"Reasoning completed: status={chain.status.value}, confidence={chain.confidence:.2f}")
                
            except Exception as e:
                chain.status = ReasoningStatus.FAILED
                self.logger.error(f"Reasoning failed: {e}")
            
            return chain
    
    async def multi_strategy_reasoning(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, ReasoningChain]:
        """Perform reasoning using multiple strategies and select the best"""
        
        self.logger.info(f"Starting multi-strategy reasoning for: {query[:100]}...")
        
        strategies_to_try = [
            ReasoningType.CHAIN_OF_THOUGHT,
            ReasoningType.LOGICAL_INFERENCE,
            # Add more strategies as implemented
        ]
        
        # Run all strategies concurrently
        tasks = []
        for strategy_type in strategies_to_try:
            task = self.reason(query, strategy_type, context)
            tasks.append((strategy_type.value, task))
        
        results = {}
        for strategy_name, task in tasks:
            try:
                chain = await asyncio.wait_for(task, timeout=60.0)
                results[strategy_name] = chain
            except asyncio.TimeoutError:
                self.logger.warning(f"Strategy {strategy_name} timed out")
                # Create a timeout chain
                timeout_chain = ReasoningChain(
                    chain_id=f"timeout_{strategy_name}_{int(time.time() * 1000)}",
                    original_query=query,
                    reasoning_type=ReasoningType[strategy_name.upper()],
                    context=context or {}
                )
                timeout_chain.status = ReasoningStatus.TIMEOUT
                results[strategy_name] = timeout_chain
        
        # Select best result based on confidence and success
        best_strategy = None
        best_score = -1
        
        for strategy_name, chain in results.items():
            if chain.is_successful:
                score = chain.confidence * (1.0 - (chain.total_duration / 10000.0))  # Balance confidence and speed
                if score > best_score:
                    best_score = score
                    best_strategy = strategy_name
        
        if best_strategy:
            self.logger.info(f"Best strategy selected: {best_strategy} with score: {best_score:.3f}")
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get reasoning performance statistics"""
        return self._performance_stats.copy()
    
    def _update_performance_stats(self, chain: ReasoningChain) -> None:
        """Update performance statistics"""
        self._performance_stats["total_reasoning_requests"] += 1
        
        if chain.is_successful:
            self._performance_stats["successful_reasoning"] += 1
        
        # Update averages
        total_requests = self._performance_stats["total_reasoning_requests"]
        current_avg_duration = self._performance_stats["average_duration"]
        current_avg_tokens = self._performance_stats["average_tokens"]
        
        self._performance_stats["average_duration"] = (
            (current_avg_duration * (total_requests - 1) + chain.total_duration) / total_requests
        )
        self._performance_stats["average_tokens"] = (
            (current_avg_tokens * (total_requests - 1) + chain.total_tokens) / total_requests
        )
    
    async def self_correction_reasoning(self, query: str, context: Optional[Dict[str, Any]] = None,
                                    max_iterations: int = 3) -> ReasoningChain:
        """Perform reasoning with self-correction capabilities"""
        
        self.logger.info(f"Starting self-correction reasoning: {query[:100]}...")
        
        current_context = context or {}
        previous_answer = None
        
        for iteration in range(max_iterations):
            # Perform reasoning
            chain = await self.reason(query, ReasoningType.SELF_CORRECTION, current_context)
            
            # Check if we need correction
            needs_correction = await self._evaluate_need_for_correction(chain, previous_answer)
            
            if not needs_correction:
                self.logger.info(f"Self-correction completed after {iteration + 1} iterations")
                chain.metadata["correction_iterations"] = iteration + 1
                break
            
            # Update context for next iteration
            current_context = {
                **current_context,
                "previous_answer": chain.final_answer,
                "iteration": iteration + 1,
                "correction_needed": True
            }
            previous_answer = chain.final_answer
            
            self.logger.info(f"Self-correction iteration {iteration + 1} completed")
        
        return chain
    
    async def _evaluate_need_for_correction(self, chain: ReasoningChain, previous_answer: Optional[str]) -> bool:
        """Evaluate if reasoning needs correction"""
        
        if not chain.is_successful:
            return True
        
        if not previous_answer:
            return False
        
        # Check for significant changes
        similarity_score = self._calculate_answer_similarity(chain.final_answer, previous_answer)
        
        # If similarity is too high, might be stuck in loop
        if similarity_score > 0.9:
            return False
        
        # If confidence is low, might need correction
        if chain.confidence < 0.7:
            return True
        
        return False
    
    def _calculate_answer_similarity(self, answer1: str, answer2: str) -> float:
        """Calculate similarity between two answers"""
        # Simplified similarity calculation
        if not answer1 or not answer2:
            return 0.0
        
        # Word overlap similarity
        words1 = set(answer1.lower().split())
        words2 = set(answer2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

# Global reasoning engine instance
_global_reasoning_engine: Optional[AdvancedReasoningEngine] = None

def get_reasoning_engine(config: Optional[Dict[str, Any]] = None) -> AdvancedReasoningEngine:
    """Get or create global reasoning engine"""
    global _global_reasoning_engine
    if _global_reasoning_engine is None:
        _global_reasoning_engine = AdvancedReasoningEngine(config or {})
    return _global_reasoning_engine

# Decorators for easy use
def advanced_reasoning(reasoning_type: ReasoningType = ReasoningType.CHAIN_OF_THOUGHT):
    """Decorator for advanced reasoning"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            engine = get_reasoning_engine()
            query = kwargs.get("query", str(args[0]) if args else "")
            
            chain = await engine.reason(query, reasoning_type, kwargs.get("context"))
            
            if chain.is_successful:
                return chain.final_answer
            else:
                raise Exception(f"Reasoning failed: {chain.status}")
        
        return wrapper
    return decorator

def multi_strategy_reasoning():
    """Decorator for multi-strategy reasoning"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            engine = get_reasoning_engine()
            query = kwargs.get("query", str(args[0]) if args else "")
            context = kwargs.get("context")
            
            results = await engine.multi_strategy_reasoning(query, context)
            
            # Return best result
            successful_chains = [chain for chain in results.values() if chain.is_successful]
            
            if successful_chains:
                best_chain = max(successful_chains, key=lambda c: c.confidence)
                return {
                    "answer": best_chain.final_answer,
                    "strategy": best_chain.reasoning_type.value,
                    "confidence": best_chain.confidence,
                    "all_results": results
                }
            else:
                raise Exception("All reasoning strategies failed")
        
        return wrapper
    return decorator