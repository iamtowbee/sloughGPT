#!/usr/bin/env python3
"""
SloughGPT Multi-Agent System
Specialist agents with coordination for complex problem solving
"""

import time
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from pathlib import Path
import threading
import queue
import statistics

class AgentSpecialty(Enum):
    """Types of agent specializations"""
    RESEARCH = "research"           # Deep research and information gathering
    CODING = "coding"             # Software development and programming
    CREATIVE = "creative"           # Creative writing, brainstorming, art
    ANALYTICAL = "analytical"       # Data analysis, logic, reasoning
    MATHEMATICAL = "mathematical"   # Math, statistics, calculations
    LINGUISTIC = "linguistic"     # Language translation, grammar, writing
    HISTORICAL = "historical"       # History, timeline analysis
    SCIENTIFIC = "scientific"       # Scientific methodology, experiments

class TaskComplexity(Enum):
    """Complexity levels for tasks"""
    SIMPLE = 1        # Single agent can handle
    MODERATE = 2      # May need coordination
    COMPLEX = 3        # Definitely needs multi-agent
    VERY_COMPLEX = 4   # Requires full coordination

class AgentStatus(Enum):
    """Agent status states"""
    IDLE = "idle"
    THINKING = "thinking"
    WORKING = "working"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class AgentTask:
    """Task assigned to an agent"""
    id: str
    agent_id: str
    specialty: AgentSpecialty
    task_description: str
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    status: AgentStatus = AgentStatus.IDLE
    created_at: float = None
    started_at: float = None
    completed_at: float = None
    error_message: Optional[str] = None
    confidence: float = 0.5
    requires_coordination: bool = False
    related_tasks: List[str] = None

@dataclass
class Collaboration:
    """Information about agent collaboration"""
    agent_id: str
    specialty: AgentSpecialty
    contribution: str
    confidence: float
    response_time: float
    timestamp: float

class SloughGPTAgent:
    """Individual specialized agent"""
    
    def __init__(self, agent_id: str, specialty: AgentSpecialty):
        self.agent_id = agent_id
        self.specialty = specialty
        self.status = AgentStatus.IDLE
        self.current_task = None
        self.knowledge_base = {}  # Specialty-specific knowledge
        self.performance_history = []
        self.logger = logging.getLogger(f"agent_{agent_id}")
        
        # Initialize specialty-specific capabilities
        self._init_specialty_capabilities()
    
    def _init_specialty_capabilities(self):
        """Initialize capabilities based on specialty"""
        if self.specialty == AgentSpecialty.RESEARCH:
            self.capabilities = [
                "deep_information_retrieval",
                "source_verification",
                "cross_reference_analysis",
                "research_synthesis"
            ]
        elif self.specialty == AgentSpecialty.CODING:
            self.capabilities = [
                "code_generation",
                "debugging",
                "code_review",
                "optimization",
                "testing"
            ]
        elif self.specialty == AgentSpecialty.CREATIVE:
            self.capabilities = [
                "creative_writing",
                "brainstorming",
                "idea_generation",
                "artistic_creation"
            ]
        elif self.specialty == AgentSpecialty.ANALYTICAL:
            self.capabilities = [
                "data_analysis",
                "logical_reasoning",
                "problem_breakdown",
                "pattern_recognition"
            ]
        elif self.specialty == AgentSpecialty.MATHEMATICAL:
            self.capabilities = [
                "calculations",
                "statistical_analysis",
                "modeling",
                "optimization"
            ]
        elif self.specialty == AgentSpecialty.LINGUISTIC:
            self.capabilities = [
                "translation",
                "grammar_analysis",
                "style_analysis",
                "text_generation"
            ]
        elif self.specialty == AgentSpecialty.SCIENTIFIC:
            self.capabilities = [
                "experimental_design",
                "hypothesis_testing",
                "data_interpretation",
                "methodology_application"
            ]
        else:  # HISTORICAL
            self.capabilities = [
                "timeline_analysis",
                "context_understanding",
                "pattern_identification",
                "historical_synthesis"
            ]
    
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process assigned task"""
        self.status = AgentStatus.THINKING
        self.current_task = task
        task.started_at = time.time()
        
        self.logger.info(f"Agent {self.agent_id} ({self.specialty.value}) processing task: {task.task_description}")
        
        try:
            # Simulate processing time based on complexity
            processing_time = self._estimate_processing_time(task)
            await asyncio.sleep(processing_time)
            
            self.status = AgentStatus.WORKING
            
            # Generate response based on specialty
            result = await self._generate_specialized_response(task)
            
            self.status = AgentStatus.COMPLETED
            task.completed_at = time.time()
            task.output_data = result
            task.status = AgentStatus.COMPLETED
            task.confidence = result.get("confidence", 0.5)
            task.response_time = task.completed_at - task.started_at
            
            # Update performance history
            self.performance_history.append({
                "task_id": task.id,
                "success": True,
                "response_time": task.response_time,
                "confidence": task.confidence,
                "timestamp": task.completed_at
            })
            
            self.logger.info(f"Task {task.id} completed by agent {self.agent_id}")
            
            return result
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            task.status = AgentStatus.ERROR
            task.error_message = str(e)
            task.completed_at = time.time()
            
            self.logger.error(f"Task {task.id} failed for agent {self.agent_id}: {e}")
            
            return {
                "error": str(e),
                "agent_id": self.agent_id,
                "specialty": self.specialty.value,
                "confidence": 0.0
            }
    
    def _estimate_processing_time(self, task: AgentTask) -> float:
        """Estimate processing time based on task and specialty"""
        base_time = 1.0
        
        # Adjust based on task complexity (extract from description)
        complexity_keywords = {
            "simple": 0.5,
            "quick": 0.3,
            "basic": 0.4,
            "complex": 2.0,
            "detailed": 1.5,
            "comprehensive": 2.5,
            "analysis": 1.8,
            "research": 2.2,
            "creative": 1.6,
            "mathematical": 2.0
        }
        
        task_desc = task.task_description.lower()
        for keyword, multiplier in complexity_keywords.items():
            if keyword in task_desc:
                base_time *= multiplier
                break
        
        # Specialty-based adjustments
        specialty_multipliers = {
            AgentSpecialty.RESEARCH: 1.5,
            AgentSpecialty.CODING: 1.3,
            AgentSpecialty.ANALYTICAL: 1.4,
            AgentSpecialty.MATHEMATICAL: 1.6,
            AgentSpecialty.SCIENTIFIC: 1.8,
            AgentSpecialty.CREATIVE: 1.2,
            AgentSpecialty.LINGUISTIC: 1.1,
            AgentSpecialty.HISTORICAL: 1.3
        }
        
        base_time *= specialty_multipliers.get(self.specialty, 1.0)
        
        # Add some randomness to simulate real processing
        import random
        return base_time * (0.8 + random.random() * 0.4)
    
    async def _generate_specialized_response(self, task: AgentTask) -> Dict[str, Any]:
        """Generate response based on agent specialty"""
        
        if self.specialty == AgentSpecialty.RESEARCH:
            return await self._research_response(task)
        elif self.specialty == AgentSpecialty.CODING:
            return await self._coding_response(task)
        elif self.specialty == AgentSpecialty.CREATIVE:
            return await self._creative_response(task)
        elif self.specialty == AgentSpecialty.ANALYTICAL:
            return await self._analytical_response(task)
        elif self.specialty == AgentSpecialty.MATHEMATICAL:
            return await self._mathematical_response(task)
        elif self.specialty == AgentSpecialty.LINGUISTIC:
            return await self._linguistic_response(task)
        elif self.specialty == AgentSpecialty.SCIENTIFIC:
            return await self._scientific_response(task)
        else:  # HISTORICAL
            return await self._historical_response(task)
    
    async def _research_response(self, task: AgentTask) -> Dict[str, Any]:
        """Generate research-oriented response"""
        await asyncio.sleep(0.5)  # Simulate research time
        
        return {
            "agent_id": self.agent_id,
            "specialty": self.specialty.value,
            "response_type": "research",
            "content": f"Based on comprehensive research and multiple sources, I can provide detailed information about {task.input_data.get('topic', task.task_description)}. Key findings include: [simulated research points], methodology: [research approach], and sources: [credible sources].",
            "confidence": 0.85,
            "sources": ["Academic Journal A", "Industry Report B", "Expert Analysis C"],
            "methodology": "cross-reference analysis",
            "verification_status": "verified"
        }
    
    async def _coding_response(self, task: AgentTask) -> Dict[str, Any]:
        """Generate coding-oriented response"""
        await asyncio.sleep(0.3)  # Simulate coding time
        
        return {
            "agent_id": self.agent_id,
            "specialty": self.specialty.value,
            "response_type": "coding",
            "content": f"I can help with {task.input_data.get('programming_task', task.task_description)}. Here's a solution: [code solution with best practices], optimized for performance and readability.",
            "confidence": 0.9,
            "code_language": task.input_data.get("language", "python"),
            "best_practices": ["error handling", "documentation", "testing"],
            "optimization_applied": True
        }
    
    async def _creative_response(self, task: AgentTask) -> Dict[str, Any]:
        """Generate creative-oriented response"""
        await asyncio.sleep(0.4)  # Simulate creative thinking
        
        return {
            "agent_id": self.agent_id,
            "specialty": self.specialty.value,
            "response_type": "creative",
            "content": f"Let me brainstorm creative approaches to {task.input_data.get('creative_challenge', task.task_description)}. Here are several innovative ideas: [creative concepts with unique perspectives].",
            "confidence": 0.75,
            "creativity_score": 0.85,
            "novelty_factor": 0.9,
            "inspiration_sources": ["cross_domain_analogy", "lateral_thinking", "pattern_innovation"]
        }
    
    async def _analytical_response(self, task: AgentTask) -> Dict[str, Any]:
        """Generate analytical-oriented response"""
        await asyncio.sleep(0.6)  # Simulate analysis time
        
        return {
            "agent_id": self.agent_id,
            "specialty": self.specialty.value,
            "response_type": "analytical",
            "content": f"Analyzing {task.input_data.get('analysis_topic', task.task_description)} systematically: Breakdown: [structured analysis], Patterns: [identified patterns], Implications: [logical conclusions].",
            "confidence": 0.8,
            "analysis_method": "systematic_decomposition",
            "logical_soundness": 0.9,
            "data_driven": True
        }
    
    async def _mathematical_response(self, task: AgentTask) -> Dict[str, Any]:
        """Generate mathematical-oriented response"""
        await asyncio.sleep(0.7)  # Simulate calculation time
        
        return {
            "agent_id": self.agent_id,
            "specialty": self.specialty.value,
            "response_type": "mathematical",
            "content": f"Mathematical analysis of {task.input_data.get('math_problem', task.task_description)}: Solution: [detailed solution], Steps: [step-by-step derivation], Verification: [proof/check].",
            "confidence": 0.95,
            "calculation_method": "analytical_derivation",
            "precision_level": "high",
            "verification_included": True
        }
    
    async def _linguistic_response(self, task: AgentTask) -> Dict[str, Any]:
        """Generate linguistic-oriented response"""
        await asyncio.sleep(0.4)  # Simulate linguistic processing
        
        return {
            "agent_id": self.agent_id,
            "specialty": self.specialty.value,
            "response_type": "linguistic",
            "content": f"Linguistic analysis for {task.input_data.get('text_task', task.task_description)}: Analysis: [linguistic features], Grammar: [grammar check], Style: [style analysis].",
            "confidence": 0.85,
            "language_detected": task.input_data.get("language", "english"),
            "grammar_score": 0.9,
            "semantic_analysis": True
        }
    
    async def _scientific_response(self, task: AgentTask) -> Dict[str, Any]:
        """Generate scientific-oriented response"""
        await asyncio.sleep(0.8)  # Simulate scientific analysis
        
        return {
            "agent_id": self.agent_id,
            "specialty": self.specialty.value,
            "response_type": "scientific",
            "content": f"Scientific approach to {task.input_data.get('scientific_question', task.task_description)}: Hypothesis: [testable hypothesis], Method: [scientific methodology], Results: [expected outcomes], Validation: [peer review process].",
            "confidence": 0.9,
            "scientific_method": "empirical_analysis",
            "hypothesis_testable": True,
            "peer_review_ready": True
        }
    
    async def _historical_response(self, task: AgentTask) -> Dict[str, Any]:
        """Generate historical-oriented response"""
        await asyncio.sleep(0.5)  # Simulate historical analysis
        
        return {
            "agent_id": self.agent_id,
            "specialty": self.specialty.value,
            "response_type": "historical",
            "content": f"Historical analysis of {task.input_data.get('historical_topic', task.task_description)}: Timeline: [chronological events], Context: [historical context], Patterns: [historical patterns], Significance: [historical importance].",
            "confidence": 0.8,
            "time_period": task.input_data.get("era", "modern"),
            "primary_sources": ["historical_documents", "archaeological_evidence"],
            "contextual_analysis": True
        }

class SloughGPTMultiAgentCoordinator:
    """Coordinates multiple specialized agents for complex tasks"""
    
    def __init__(self, db_path: str = "slo_multi_agent.db"):
        self.db_path = db_path
        self.agents = {}
        self.active_tasks = {}
        self.task_queue = asyncio.Queue()
        self.logger = logging.getLogger(__name__)
        self._init_database()
        self._init_agents()
    
    def _init_database(self):
        """Initialize database for multi-agent coordination"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_tasks (
                id TEXT PRIMARY KEY,
                agent_id TEXT,
                specialty TEXT,
                task_description TEXT,
                input_data TEXT,
                output_data TEXT,
                status TEXT,
                created_at REAL,
                started_at REAL,
                completed_at REAL,
                error_message TEXT,
                confidence REAL,
                requires_coordination BOOLEAN
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS collaborations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                agent_id TEXT,
                specialty TEXT,
                contribution TEXT,
                confidence REAL,
                response_time REAL,
                timestamp REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _init_agents(self):
        """Initialize specialized agents"""
        specialties = [
            AgentSpecialty.RESEARCH,
            AgentSpecialty.CODING,
            AgentSpecialty.CREATIVE,
            AgentSpecialty.ANALYTICAL,
            AgentSpecialty.MATHEMATICAL,
            AgentSpecialty.LINGUISTIC,
            AgentSpecialty.HISTORICAL,
            AgentSpecialty.SCIENTIFIC
        ]
        
        for specialty in specialties:
            agent_id = f"agent_{specialty.value}"
            agent = SloughGPTAgent(agent_id, specialty)
            self.agents[agent_id] = agent
            self.logger.info(f"Initialized agent: {agent_id} ({specialty.value})")
    
    async def process_complex_query(self, query: str, complexity: TaskComplexity = TaskComplexity.COMPLEX) -> Dict[str, Any]:
        """Process a complex query using appropriate agents"""
        task_id = f"task_{int(time.time())}"
        
        self.logger.info(f"Processing complex query: {query} (complexity: {complexity.value})")
        
        # Decompose task based on complexity and content
        subtasks = self._decompose_query(query, complexity)
        
        # Assign subtasks to appropriate agents
        assigned_tasks = []
        for i, subtask in enumerate(subtasks):
            agent_id = self._select_best_agent(subtask)
            
            task = AgentTask(
                id=f"{task_id}_{i}",
                agent_id=agent_id,
                specialty=self.agents[agent_id].specialty,
                task_description=subtask["description"],
                input_data=subtask["data"],
                created_at=time.time(),
                requires_coordination=len(subtasks) > 1,
                related_tasks=[f"{task_id}_{j}" for j in range(len(subtasks)) if j != i]
            )
            
            assigned_tasks.append(task)
            self.active_tasks[task.id] = task
        
        # Execute tasks in parallel
        results = await self._execute_parallel_tasks(assigned_tasks)
        
        # Synthesize final response
        final_response = await self._synthesize_results(query, results)
        
        # Save collaboration data
        await self._save_collaboration(task_id, results)
        
        return final_response
    
    def _decompose_query(self, query: str, complexity: TaskComplexity) -> List[Dict[str, Any]]:
        """Decompose complex query into subtasks"""
        subtasks = []
        
        query_lower = query.lower()
        
        if complexity == TaskComplexity.SIMPLE:
            # Single agent can handle
            subtasks.append({
                "description": query,
                "data": {"query": query, "original_complexity": complexity.value},
                "priority": 1
            })
        
        elif complexity == TaskComplexity.MODERATE:
            # May need 2-3 agents
            if "research" in query_lower or "analyze" in query_lower:
                subtasks.extend([
                    {
                        "description": f"Research: {query}",
                        "data": {"topic": query, "depth": "comprehensive"},
                        "priority": 1
                    },
                    {
                        "description": f"Analyze research findings for: {query}",
                        "data": {"analysis_type": "comprehensive"},
                        "priority": 2
                    }
                ])
            else:
                subtasks.append({
                    "description": f"Process: {query}",
                    "data": {"query": query, "approach": "systematic"},
                    "priority": 1
                })
        
        elif complexity == TaskComplexity.COMPLEX:
            # Definitely needs multi-agent
            if "code" in query_lower or "program" in query_lower:
                subtasks.extend([
                    {
                        "description": f"Requirements analysis for: {query}",
                        "data": {"analysis_type": "requirements"},
                        "priority": 1
                    },
                    {
                        "description": f"Code implementation for: {query}",
                        "data": {"programming_task": query},
                        "priority": 2
                    },
                    {
                        "description": f"Code review and testing for: {query}",
                        "data": {"review_type": "comprehensive"},
                        "priority": 3
                    }
                ])
            else:
                subtasks.extend([
                    {
                        "description": f"Research phase for: {query}",
                        "data": {"topic": query, "phase": "initial"},
                        "priority": 1
                    },
                    {
                        "description": f"Analysis phase for: {query}",
                        "data": {"analysis_type": "deep"},
                        "priority": 2
                    },
                    {
                        "description": f"Synthesis phase for: {query}",
                        "data": {"synthesis_type": "comprehensive"},
                        "priority": 3
                    }
                ])
        
        else:  # VERY_COMPLEX
            # Full coordination required
            subtasks.extend([
                {
                    "description": f"Preparation and planning for: {query}",
                    "data": {"phase": "planning", "complexity": complexity.value},
                    "priority": 1
                },
                {
                    "description": f"Research and data gathering for: {query}",
                    "data": {"phase": "research", "depth": "exhaustive"},
                    "priority": 2
                },
                {
                    "description": f"Analysis and breakdown for: {query}",
                    "data": {"phase": "analysis", "method": "multi_perspective"},
                    "priority": 3
                },
                {
                    "description": f"Creative exploration for: {query}",
                    "data": {"phase": "creative", "approach": "innovative"},
                    "priority": 4
                },
                {
                    "description": f"Synthesis and integration for: {query}",
                    "data": {"phase": "synthesis", "method": "comprehensive"},
                    "priority": 5
                }
            ])
        
        return subtasks
    
    def _select_best_agent(self, subtask: Dict[str, Any]) -> str:
        """Select the best agent for a subtask"""
        description = subtask["description"].lower()
        data = subtask["data"]
        
        # Simple keyword-based agent selection
        agent_selection_rules = {
            "research": "agent_research",
            "analyze": "agent_analytical", 
            "code": "agent_coding",
            "program": "agent_coding",
            "creative": "agent_creative",
            "math": "agent_mathematical",
            "linguistic": "agent_linguistic",
            "historical": "agent_historical",
            "scientific": "agent_scientific"
        }
        
        for keyword, agent_id in agent_selection_rules.items():
            if keyword in description:
                return agent_id
        
        # Default to analytical agent
        return "agent_analytical"
    
    async def _execute_parallel_tasks(self, tasks: List[AgentTask]) -> List[Dict[str, Any]]:
        """Execute tasks in parallel using available agents"""
        self.logger.info(f"Executing {len(tasks)} tasks in parallel")
        
        # Create concurrent tasks
        concurrent_tasks = []
        for task in tasks:
            if task.agent_id in self.agents:
                agent = self.agents[task.agent_id]
                concurrent_tasks.append(agent.process_task(task))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "error": str(result),
                    "task_id": tasks[i].id,
                    "agent_id": tasks[i].agent_id
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _synthesize_results(self, original_query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize results from multiple agents into coherent response"""
        self.logger.info(f"Synthesizing {len(results)} results into final response")
        
        # Filter out errors
        valid_results = [r for r in results if "error" not in r]
        
        if not valid_results:
            return {
                "error": "All agent tasks failed",
                "original_query": original_query,
                "synthesis_method": "none",
                "confidence": 0.0
            }
        
        # Calculate overall confidence
        confidences = [r.get("confidence", 0.5) for r in valid_results]
        overall_confidence = statistics.mean(confidences)
        
        # Create synthesis based on result types
        response_types = [r.get("response_type", "unknown") for r in valid_results]
        unique_types = list(set(response_types))
        
        synthesis = {
            "original_query": original_query,
            "synthesis_method": "multi_agent_coordination",
            "overall_confidence": overall_confidence,
            "agent_contributions": valid_results,
            "contribution_count": len(valid_results),
            "specialties_involved": unique_types,
            "coordination_success": True
        }
        
        # Generate final content based on combination
        if len(unique_types) == 1:
            # Single specialty, just combine
            synthesis["final_content"] = self._combine_single_specialty(valid_results)
        else:
            # Multiple specialties, need coordination
            synthesis["final_content"] = await self._coordinate_multi_specialty(valid_results, original_query)
        
        return synthesis
    
    def _combine_single_specialty(self, results: List[Dict[str, Any]]) -> str:
        """Combine results from same specialty"""
        contents = [r.get("content", "") for r in results]
        return f"Synthesized analysis from {len(results)} {results[0].get('specialty', 'specialist')} agents:\n\n" + "\n\n".join(contents)
    
    async def _coordinate_multi_specialty(self, results: List[Dict[str, Any]], original_query: str) -> str:
        """Coordinate results from multiple different specialties"""
        await asyncio.sleep(0.5)  # Simulate coordination time
        
        # Group by specialty
        specialty_groups = {}
        for result in results:
            specialty = result.get("specialty", "unknown")
            if specialty not in specialty_groups:
                specialty_groups[specialty] = []
            specialty_groups[specialty].append(result)
        
        # Create coordinated response
        coordinated_content = f"Multi-specialist analysis for: {original_query}\n\n"
        
        for specialty, group_results in specialty_groups.items():
            coordinated_content += f"=== {specialty.upper()} PERSPECTIVE ===\n"
            for result in group_results:
                content = result.get("content", "")
                confidence = result.get("confidence", 0.5)
                coordinated_content += f"• {content} (Confidence: {confidence:.2f})\n"
            coordinated_content += "\n"
        
        coordinated_content += "=== COORDINATED SYNTHESIS ===\n"
        coordinated_content += "Integrating all perspectives, the comprehensive analysis suggests: [Integrated conclusion drawing from all specialist inputs]."
        
        return coordinated_content
    
    async def _save_collaboration(self, task_id: str, results: List[Dict[str, Any]]):
        """Save collaboration data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for result in results:
            if "error" not in result:
                cursor.execute("""
                    INSERT INTO collaborations 
                    (task_id, agent_id, specialty, contribution, confidence, response_time, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    task_id,
                    result.get("agent_id"),
                    result.get("specialty"),
                    result.get("content", "")[:500],  # Truncate long content
                    result.get("confidence", 0.5),
                    result.get("response_time", 0.0),
                    time.time()
                ))
        
        conn.commit()
        conn.close()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current multi-agent system status"""
        return {
            "system_status": "active",
            "total_agents": len(self.agents),
            "agent_specialties": [agent.specialty.value for agent in self.agents.values()],
            "active_tasks": len([t for t in self.active_tasks.values() if t.status in [AgentStatus.WORKING, AgentStatus.THINKING]]),
            "total_tasks_processed": len(self.active_tasks),
            "agent_capabilities": {agent_id: agent.capabilities for agent_id, agent in self.agents.items()}
        }

if __name__ == "__main__":
    # Example usage
    coordinator = SloughGPTMultiAgentCoordinator()
    
    async def test_complex_query():
        # Test complex query
        query = "Create a comprehensive research report on renewable energy technologies, including economic analysis and future projections"
        
        result = await coordinator.process_complex_query(query, TaskComplexity.VERY_COMPLEX)
        
        print("Multi-Agent Results:")
        print(f"Overall Confidence: {result.get('overall_confidence', 0):.2f}")
        print(f"Specialties Involved: {', '.join(result.get('specialties_involved', []))}")
        print(f"Contributions: {result.get('contribution_count', 0)}")
        print(f"\nFinal Content:\n{result.get('final_content', 'No content generated')}")
        
        # Get system status
        status = coordinator.get_system_status()
        print(f"\nSystem Status: {json.dumps(status, indent=2)}")
    
    # Run test
    asyncio.run(test_complex_query())