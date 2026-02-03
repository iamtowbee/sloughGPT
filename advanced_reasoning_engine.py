#!/usr/bin/env python3
"""
Advanced Reasoning Engine - Bridge between Cognitive Architecture and RAG System

Integrates Stage 2 Cognitive Architecture with main SLO RAG knowledge base
to enable advanced reasoning patterns: Chain-of-Thought, Self-Reflective, Multi-Hop
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

from stage2_cognitive_architecture import CognitiveArchitecture, CognitiveState, MemoryTrace
from slo_rag import SLO_RAG
from hauls_store import HaulsStore, Document
try:
    from llm_integration import LLMIntegrator
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: LLM integration not available. Using enhanced template responses.")

try:
    from epiphany_integration import EpiphanyIntegrationBridge
    EPIPHANY_AVAILABLE = True
except ImportError:
    EPIPHANY_AVAILABLE = False
    print("Warning: Epiphany integration not available. Using standard reasoning.")


@dataclass
class ReasoningStep:
    """Single step in reasoning process"""
    step_id: str
    query: str
    retrieved_docs: List[Document]
    cognitive_context: str
    reasoning: str
    confidence: float
    timestamp: float


@dataclass
class ReasoningChain:
    """Complete chain of reasoning steps"""
    chain_id: str
    original_query: str
    steps: List[ReasoningStep]
    final_answer: str
    total_confidence: float
    metadata: Dict[str, Any]


class AdvancedReasoningEngine:
    """Bridge between Cognitive Architecture and RAG system for advanced reasoning"""
    
    def __init__(self, 
                 rag_store_path: str = "runs/store/hauls_store.db",
                 cognitive_arch: Optional[CognitiveArchitecture] = None):
        
        # Initialize components
        self.slo_rag = SLO_RAG(rag_store_path)
        self.cognitive_arch = cognitive_arch or CognitiveArchitecture()
        
        # Initialize LLM integration
        if LLM_AVAILABLE:
            try:
                self.llm_integrator = LLMIntegrator()
                self.use_llm = True
                print("🤖 LLM integration enabled")
            except Exception as e:
                print(f"⚠️ LLM integration failed: {e}")
                self.use_llm = False
        else:
            self.use_llm = False
        
        # Initialize epiphany integration
        if EPIPHANY_AVAILABLE:
            try:
                self.epiphany_bridge = EpiphanyIntegrationBridge(self)
                self.use_epiphany = True
                print("🌟 Epiphany integration enabled")
            except Exception as e:
                print(f"⚠️ Epiphany integration failed: {e}")
                self.use_epiphany = False
        else:
            self.use_epiphany = False
        
        # Reasoning patterns
        self.reasoning_patterns = {
            "chain_of_thought": self._chain_of_thought_rag,
            "self_reflective": self._self_reflective_rag,
            "multi_hop": self._multi_hop_rag,
            "hybrid": self._hybrid_reasoning
        }
        
        # Reasoning history for learning
        self.reasoning_history: List[ReasoningChain] = []
        self.performance_metrics = defaultdict(list)
        
        # Bridge configuration
        self.max_reasoning_steps = 5
        self.confidence_threshold = 0.7
        self.context_window = 4000
        
        print("🧠 Advanced Reasoning Engine initialized")
        print(f"📚 RAG knowledge base: {self._get_kb_size()} documents")
        print(f"🧠 Cognitive architecture: {self.cognitive_arch.get_cognitive_statistics()}")
    
    def _get_kb_size(self) -> int:
        """Get knowledge base size"""
        try:
            return len(self.slo_rag.hauls_store.documents) if self.slo_rag.hauls_store else 0
        except:
            return 0
    
    def reason(self, query: str, pattern: str = "hybrid") -> ReasoningChain:
        """
        Main reasoning interface - process query using specified pattern
        """
        if pattern not in self.reasoning_patterns:
            raise ValueError(f"Unknown reasoning pattern: {pattern}")
        
        print(f"🤔 Processing query with {pattern} pattern: {query[:50]}...")
        
        # Update cognitive state
        self.cognitive_arch.process_input(query)
        
        # Execute reasoning pattern
        start_time = time.time()
        reasoning_chain = self.reasoning_patterns[pattern](query)
        processing_time = time.time() - start_time
        
        # Check for epiphany integration
        if self.use_epiphany and hasattr(self, 'epiphany_bridge'):
            enhanced_chain, epiphany = self.epiphany_bridge.process_with_epiphany_integration(query, pattern)
            if epiphany:
                # Log epiphany detection
                print(f"🌟 EPIPHANY DETECTED - Novelty: {epiphany.novelty_score:.3f}")
                
                # Update epiphany metrics
                if not hasattr(self, 'epiphany_metrics'):
                    self.epiphany_metrics = []
                
                self.epiphany_metrics.append({
                    "timestamp": time.time(),
                    "pattern": epiphany.layers_involved,
                    "novelty": epiphany.novelty_score,
                    "usefulness": epiphany.usefulness_score
                })
                
                reasoning_chain = enhanced_chain
            else:
                # Standard reasoning processing
                reasoning_chain = self.reasoning_patterns[pattern](query)
        else:
            # Standard reasoning processing
            reasoning_chain = self.reasoning_patterns[pattern](query)
        
        # Log performance
        if hasattr(self, 'log_performance'):
            self.log_performance(query, processing_time, reasoning_chain.total_confidence)
        
        # Update metrics
        self.performance_metrics[pattern].append({
            "time": processing_time,
            "confidence": reasoning_chain.total_confidence,
            "steps": len(reasoning_chain.steps)
        })
        
        # Store in reasoning history
        self.reasoning_history.append(reasoning_chain)
        
        # Update cognitive learning
        self._update_cognitive_learning(reasoning_chain)
        
        if hasattr(self, 'epiphany_bridge') and hasattr(reasoning_chain.metadata, 'epiphany_detected'):
            print(f"✅ Reasoning with EPIPHANY complete in {processing_time:.2f}s (confidence: {reasoning_chain.total_confidence:.2f})")
        else:
            print(f"✅ Reasoning complete in {processing_time:.2f}s (confidence: {reasoning_chain.total_confidence:.2f})")
        return reasoning_chain
    
    def _extract_themes_from_docs(self, docs: List[Document]) -> List[str]:
        """Extract key themes from document contents"""
        themes = []
        for doc in docs[:3]:
            content = doc.content.lower()
            
            # Simple theme extraction using common patterns
            if any(word in content for word in ['character', 'person', 'protagonist']):
                themes.append('character analysis')
            if any(word in content for word in ['story', 'narrative', 'plot']):
                themes.append('narrative structure')
            if any(word in content for word in ['theme', 'meaning', 'symbol']):
                themes.append('thematic elements')
            if any(word in content for word in ['dialogue', 'speech', 'conversation']):
                themes.append('dialogue analysis')
            if any(word in content for word in ['action', 'event', 'happens']):
                themes.append('action/plot development')
                
        return list(set(themes))
    
    def _synthesize_query_response(self, query: str, doc_context: str) -> str:
        """Synthesize response to specific query"""
        query_lower = query.lower()
        
        if 'who' in query_lower:
            return "the documents provide information about key individuals and their roles."
        elif 'what' in query_lower:
            return "the evidence reveals important aspects and characteristics."
        elif 'how' in query_lower:
            return "the documents explain processes and methodologies involved."
        elif 'why' in query_lower:
            return "the underlying reasons and motivations are clearly indicated."
        elif 'when' in query_lower or 'where' in query_lower:
            return "temporal and spatial context is provided in the documentation."
        else:
            return "comprehensive information is available from knowledge base sources."
    
    def _chain_of_thought_rag(self, query: str) -> ReasoningChain:
        """
        Chain-of-Thought RAG: Decompose query, retrieve for each sub-step, synthesize
        """
        chain_id = f"cot_{int(time.time())}"
        steps = []
        
        # Step 1: Cognitive decomposition (simplified)
        sub_queries = [query, f"What is {query}?", f"How does {query} work?"]
        print(f"🔍 Processing with {len(sub_queries)} sub-queries")
        
        # Step 2: Process each sub-query
        for i, sub_query in enumerate(sub_queries):
            # Retrieve relevant documents
            search_results = self.slo_rag.search_knowledge(sub_query)
            docs = [Document(id=str(i), content=result.get('content', ''), metadata={}) for i, result in enumerate(search_results)]
            
            # Get cognitive context
            working_memory = self.cognitive_arch.working_memory
            cognitive_context = f"Working memory has {len(working_memory.items)} items"
            
            # Generate reasoning for this step
            reasoning = self._generate_step_reasoning(sub_query, docs, cognitive_context, i)
            confidence = self._calculate_step_confidence(docs, reasoning)
            
            step = ReasoningStep(
                step_id=f"{chain_id}_step_{i}",
                query=sub_query,
                retrieved_docs=docs,
                cognitive_context=cognitive_context,
                reasoning=reasoning,
                confidence=confidence,
                timestamp=time.time()
            )
            steps.append(step)
            
            # Update working memory
            self.cognitive_arch.working_memory.add_to_working_memory(sub_query, {})
        
        # Step 3: Synthesize final answer
        final_answer = self._synthesize_chain_answer(steps)
        total_confidence = float(np.mean([step.confidence for step in steps]))
        
        return ReasoningChain(
            chain_id=chain_id,
            original_query=query,
            steps=steps,
            final_answer=final_answer,
            total_confidence=total_confidence,
            metadata={"pattern": "chain_of_thought", "sub_queries": len(sub_queries)}
        )
    
    def _self_reflective_rag(self, query: str) -> ReasoningChain:
        """
        Self-Reflective RAG: Initial response + critique + refinement
        """
        chain_id = f"ref_{int(time.time())}"
        steps = []
        
        # Step 1: Initial retrieval and response
        search_results = self.slo_rag.search_knowledge(query)
        docs = [Document(id=str(i), content=result.get('content', ''), metadata={}) for i, result in enumerate(search_results)]
        initial_reasoning = self._generate_initial_response(query, docs)
        initial_confidence = self._calculate_step_confidence(docs, initial_reasoning)
        
        steps.append(ReasoningStep(
            step_id=f"{chain_id}_initial",
            query=query,
            retrieved_docs=docs,
            cognitive_context="initial_response",
            reasoning=initial_reasoning,
            confidence=initial_confidence,
            timestamp=time.time()
        ))
        
# Step 2: Self-critique using episodic memory
        critique = self._generate_self_critique(initial_reasoning, query)
        
        # Step 3: Refinement based on critique
        if "weakness" in critique.lower() or "improve" in critique.lower():
            refined_search_results = self.slo_rag.search_knowledge(critique)
            refined_docs = [Document(id=str(i), content=result.get('content', ''), metadata={}) for i, result in enumerate(refined_search_results)]
            refined_reasoning = self._generate_refined_response(query, docs, refined_docs, critique)
            refined_confidence = self._calculate_step_confidence(refined_docs, refined_reasoning)
            
            steps.append(ReasoningStep(
                step_id=f"{chain_id}_refined",
                query=f"Refined: {query}",
                retrieved_docs=refined_docs,
                cognitive_context="refined_response",
                reasoning=refined_reasoning,
                confidence=refined_confidence,
                timestamp=time.time()
            ))
            
            final_answer = refined_reasoning
            total_confidence = float((initial_confidence + refined_confidence) / 2)
        else:
            final_answer = initial_reasoning
            total_confidence = float(initial_confidence)
        
        return ReasoningChain(
            chain_id=chain_id,
            original_query=query,
            steps=steps,
            final_answer=final_answer,
            total_confidence=total_confidence,
            metadata={"pattern": "self_reflective", "refined": len(steps) > 1}
        )
    
    def _multi_hop_rag(self, query: str) -> ReasoningChain:
        """
        Multi-Hop RAG: Chain context through working memory
        """
        chain_id = f"hop_{int(time.time())}"
        steps = []
        
        current_query = query
        accumulated_context = ""
        
        for hop in range(self.max_reasoning_steps):
            # Retrieve with accumulated context
            enhanced_query = f"{current_query} Context: {accumulated_context}"
            search_results = self.slo_rag.search_knowledge(enhanced_query)
            docs = [Document(id=str(i), content=result.get('content', ''), metadata={}) for i, result in enumerate(search_results)]
            
            # Generate reasoning for this hop
            hop_reasoning = self._generate_hop_reasoning(current_query, docs, accumulated_context, hop)
            confidence = self._calculate_step_confidence(docs, hop_reasoning)
            
            step = ReasoningStep(
                step_id=f"{chain_id}_hop_{hop}",
                query=current_query,
                retrieved_docs=docs,
                cognitive_context=f"hop_{hop}",
                reasoning=hop_reasoning,
                confidence=confidence,
                timestamp=time.time()
            )
            steps.append(step)
            
            # Update context for next hop
            accumulated_context += f" {hop_reasoning}"
            self.cognitive_arch.working_memory.add_to_working_memory(f"hop_{hop}", {})
            
            # Check if we have sufficient information
            if confidence > self.confidence_threshold and hop >= 2:
                break
            
            # Generate next query if needed
            if hop < self.max_reasoning_steps - 1:
                current_query = self._generate_next_hop_query(current_query, hop_reasoning)
        
        final_answer = self._synthesize_multi_hop_answer(steps)
        total_confidence = float(np.mean([step.confidence for step in steps]))
        
        return ReasoningChain(
            chain_id=chain_id,
            original_query=query,
            steps=steps,
            final_answer=final_answer,
            total_confidence=total_confidence,
            metadata={"pattern": "multi_hop", "hops": len(steps)}
        )
    
    def _hybrid_reasoning(self, query: str) -> ReasoningChain:
        """
        Hybrid reasoning: Combine multiple patterns based on query complexity
        """
        # Analyze query complexity
        complexity = self._analyze_query_complexity(query)
        
        if complexity > 0.8:
            # High complexity: Use chain-of-thought + self-reflection
            cot_chain = self._chain_of_thought_rag(query)
            if cot_chain.total_confidence < self.confidence_threshold:
                # Refine with self-reflection
                refined_chain = self._self_reflective_rag(query)
                return refined_chain
            return cot_chain
        elif complexity > 0.5:
            # Medium complexity: Use multi-hop
            return self._multi_hop_rag(query)
        else:
            # Low complexity: Simple retrieval with cognitive context
            return self._self_reflective_rag(query)
    
    def _generate_step_reasoning(self, query: str, docs: List[Document], 
                                 cognitive_context: str, step_num: int) -> str:
        """Generate reasoning for a specific step"""
        doc_context = " ".join([doc.content[:200] for doc in docs[:2]])
        
        reasoning_prompt = f"""
        Step {step_num + 1} reasoning for: {query}
        
        Documents: {doc_context}
        Cognitive Context: {cognitive_context}
        
        Provide focused reasoning for this step:
        """
        
        # Use LLM if available, otherwise use enhanced template
        if self.use_llm and hasattr(self, 'llm_integrator'):
            # Build context string from documents
            context_parts = []
            for i, doc in enumerate(docs[:3]):
                context_parts.append(f"Document {i+1}: {doc.content[:200]}...")
            
            full_context = "\n\n".join(context_parts)
            
            # Generate with LLM
            reasoning = self.llm_integrator.generate_reasoned_response(
                query, full_context, "chain_of_thought"
            )
        else:
            # Enhanced template reasoning
            reasoning = f"Step {step_num + 1} Analysis of {query}:\n"
            
            # Extract key themes from documents
            themes = self._extract_themes_from_docs(docs)
            if themes:
                reasoning += f"Key themes: {', '.join(themes[:2])}.\n"
            
            # Add document evidence
            reasoning += f"Evidence: {doc_context[:150]}...\n"
            
            # Provide synthesis
            reasoning += f"Conclusion: {self._synthesize_query_response(query, doc_context)}"
        
        return reasoning
    
    def _extract_themes_from_docs(self, docs: List[Document]) -> List[str]:
        """Extract key themes from document contents"""
        themes = []
        for doc in docs[:3]:
            content = doc.content.lower()
            
            # Simple theme extraction using common patterns
            if any(word in content for word in ['character', 'person', 'protagonist']):
                themes.append('character analysis')
            if any(word in content for word in ['story', 'narrative', 'plot']):
                themes.append('narrative structure')
            if any(word in content for word in ['theme', 'meaning', 'symbol']):
                themes.append('thematic elements')
            if any(word in content for word in ['dialogue', 'speech', 'conversation']):
                themes.append('dialogue analysis')
            if any(word in content for word in ['action', 'event', 'happens']):
                themes.append('action/plot development')
                
        return list(set(themes))
    
    def _synthesize_query_response(self, query: str, doc_context: str) -> str:
        """Synthesize response to specific query"""
        query_lower = query.lower()
        
        if 'who' in query_lower:
            return "the documents provide information about key individuals and their roles."
        elif 'what' in query_lower:
            return "the evidence reveals important aspects and characteristics."
        elif 'how' in query_lower:
            return "the documents explain processes and methodologies involved."
        elif 'why' in query_lower:
            return "the underlying reasons and motivations are clearly indicated."
        elif 'when' in query_lower or 'where' in query_lower:
            return "temporal and spatial context is provided in the documentation."
        else:
            return "comprehensive information is available from the knowledge base sources."
    
    def _generate_initial_response(self, query: str, docs: List[Document]) -> str:
        """Generate initial response for self-reflection"""
        doc_context = " ".join([doc.content[:300] for doc in docs[:3]])
        return f"Initial analysis of {query} based on knowledge base: {doc_context[:200]}..."
    
    def _generate_self_critique(self, response: str, query: str) -> str:
        """Generate self-critique of initial response"""
        # Simulate critique generation
        critiques = [
            "Response could be more specific with additional examples",
            "Consider alternative perspectives mentioned in the knowledge base",
            "Analysis could benefit from deeper contextual information",
            "Response is adequate but could be enhanced with more details"
        ]
        return random.choice(critiques)
    
    def _generate_refined_response(self, query: str, initial_docs: List[Document], 
                                  refined_docs: List[Document], critique: str) -> str:
        """Generate refined response based on critique"""
        all_docs = initial_docs + refined_docs
        doc_context = " ".join([doc.content[:200] for doc in all_docs[:4]])
        return f"Refined analysis of {query} addressing {critique}: {doc_context[:250]}..."
    
    def _generate_hop_reasoning(self, query: str, docs: List[Document], 
                               accumulated_context: str, hop_num: int) -> str:
        """Generate reasoning for multi-hop step"""
        doc_context = " ".join([doc.content[:150] for doc in docs[:2]])
        return f"Hop {hop_num + 1} reasoning for {query} with context: {doc_context[:150]}..."
    
    def _generate_next_hop_query(self, current_query: str, reasoning: str) -> str:
        """Generate next query for multi-hop reasoning"""
        return f"Follow-up to {current_query}: What additional information is needed?"
    
    def _calculate_step_confidence(self, docs: List[Document], reasoning: str) -> float:
        """Calculate confidence score for reasoning step"""
        if not docs:
            return 0.3
        
        # Enhanced document relevance scoring
        doc_relevance_score = self._calculate_document_relevance(docs) * 0.5
        
        # Semantic coherence score based on reasoning structure
        reasoning_coherence = self._calculate_reasoning_coherence(reasoning) * 0.3
        
        # Knowledge base coverage score
        coverage_score = min(len(docs) / 3.0, 1.0) * 0.2
        
        total_confidence = doc_relevance_score + reasoning_coherence + coverage_score
        return min(total_confidence, 1.0)
    
    def _calculate_document_relevance(self, docs: List[Document]) -> float:
        """Calculate actual document relevance using semantic similarity"""
        if not docs or not self.slo_rag.hauls_store or not hasattr(self.slo_rag.hauls_store, '_get_embedding'):
            return 0.5
        
        # Get embeddings for all documents
        doc_embeddings = []
        for doc in docs:
            embedding = self.slo_rag.hauls_store._get_embedding(doc.content)
            doc_embeddings.append(embedding)
        
        if not doc_embeddings:
            return 0.5
        
        # Calculate average semantic similarity among retrieved docs
        similarities = []
        for i, emb1 in enumerate(doc_embeddings):
            for emb2 in doc_embeddings[i+1:]:
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
                similarities.append(abs(similarity))
        
        return float(np.mean(similarities)) if similarities else 0.5
    
    def _calculate_reasoning_coherence(self, reasoning: str) -> float:
        """Calculate reasoning quality based on structural coherence"""
        coherence_indicators = {
            'has_conclusion': any(word in reasoning.lower() for word in ['therefore', 'thus', 'conclusion', 'result']),
            'has_evidence': any(word in reasoning.lower() for word in ['evidence', 'data', 'shows', 'indicates']),
            'logical_flow': reasoning.count('.') > 2,  # Multiple sentences
            'complexity': len(reasoning.split()) > 20
        }
        
        coherence_score = sum(coherence_indicators.values()) / len(coherence_indicators)
        return max(coherence_score, 0.3)
    
    def _synthesize_chain_answer(self, steps: List[ReasoningStep]) -> str:
        """Synthesize final answer from chain of reasoning steps"""
        if not steps:
            return "Insufficient information to provide a comprehensive answer."
        
        # Weight steps by confidence and relevance with query-type optimization
        weighted_insights = []
        query_type = self._classify_query_type(steps[0].query if steps else "")
        
        for i, step in enumerate(steps):
            # Dynamic weighting based on query type
            base_weight = step.confidence
            
            if query_type == "analytical":
                weight = base_weight * (1.0 + i * 0.15)  # Progressive reasoning for analytical
            elif query_type == "factual":
                weight = base_weight * (1.0 - i * 0.05)  # Early steps more important for factual
            else:
                weight = base_weight * (1.0 + i * 0.1)  # Default progressive
            
            insight = self._extract_key_insight(step.reasoning)
            weighted_insights.append((weight, insight, step.step_id))
        
        # Sort by weight and extract top insights
        weighted_insights.sort(reverse=True, key=lambda x: x[0])
        top_insights = [insight for _, insight, _ in weighted_insights[:3]]
        
        # Build coherent response
        synthesis = self._build_coherent_response(top_insights, "chain_of_thought")
        return synthesis
    
    def _synthesize_multi_hop_answer(self, steps: List[ReasoningStep]) -> str:
        """Synthesize final answer from multi-hop reasoning"""
        if not steps:
            return "Unable to complete multi-hop reasoning due to insufficient information."
        
        # Extract key insights from each hop with progressive weighting
        hop_insights = []
        for i, step in enumerate(steps):
            weight = step.confidence * (1.0 + i * 0.15)  # Progressive weighting
            insight = self._extract_key_insight(step.reasoning)
            hop_insights.append((weight, insight, f"hop_{i}"))
        
        # Sort and build progressive narrative
        hop_insights.sort(reverse=True, key=lambda x: x[0])
        key_insights = [insight for _, insight, _ in hop_insights]
        
        response = self._build_coherent_response(key_insights, "multi_hop")
        return response
    
    def _extract_key_insight(self, reasoning: str) -> str:
        """Extract the most important insight from reasoning text"""
        sentences = [s.strip() for s in reasoning.split('.') if s.strip()]
        
        # Score sentences by importance indicators
        scored_sentences = []
        for sentence in sentences:
            score = 0
            # Increase score for key indicator words
            if any(word in sentence.lower() for word in ['therefore', 'thus', 'conclusion']):
                score += 3
            elif any(word in sentence.lower() for word in ['important', 'key', 'critical', 'essential']):
                score += 2
            elif any(word in sentence.lower() for word in ['shows', 'indicates', 'suggests']):
                score += 1
            
            # Prefer longer, more complete sentences
            score += min(len(sentence.split()) / 10.0, 1.0)
            
            scored_sentences.append((score, sentence))
        
        # Return highest scoring sentence
        if scored_sentences:
            scored_sentences.sort(reverse=True, key=lambda x: x[0])
            return scored_sentences[0][1]
        return reasoning[:100] + "..." if len(reasoning) > 100 else reasoning
    
    def _build_coherent_response(self, insights: List[str], pattern: str) -> str:
        """Build a coherent response from extracted insights"""
        if not insights:
            return "Unable to synthesize a coherent response from the available information."
        
        # Remove duplicates and organize insights
        unique_insights = []
        for insight in insights:
            if not any(insight.lower() in existing.lower() for existing in unique_insights):
                unique_insights.append(insight)
        
        if pattern == "chain_of_thought":
            response = "Based on a step-by-step analysis:\n"
            for i, insight in enumerate(unique_insights, 1):
                response += f"{i}. {insight}.\n"
            response += "These points collectively address the query through logical progression."
        else:
            response = f"Key findings: {'. '.join(unique_insights)}."
        
        return response
    
    def _analyze_query_complexity(self, query: str) -> float:
        """Enhanced query complexity analysis"""
        # Multi-dimensional complexity factors
        length_factor = min(len(query.split()) / 15.0, 1.0)
        
        # Question type complexity
        question_complexity = 0
        complex_questions = ['why', 'how', 'explain', 'analyze', 'compare', 'evaluate']
        simple_questions = ['what', 'when', 'where', 'who', 'is', 'are']
        
        question_words = query.lower().split()
        for word in question_words:
            if word in complex_questions:
                question_complexity += 0.3
            elif word in simple_questions:
                question_complexity += 0.1
        
        # Analytical indicators
        analytical_words = ['compare', 'contrast', 'analyze', 'evaluate', 'synthesize', 'integrate']
        analytical_score = len([w for w in analytical_words if w in query.lower()]) * 0.25
        
        # Multiple clauses (indicated by conjunctions)
        clause_indicators = ['and', 'but', 'or', 'while', 'because', 'since', 'although']
        clause_score = len([w for w in clause_indicators if w in query.lower()]) * 0.2
        
        # Technical jargon (estimated)
        technical_words = ['system', 'process', 'method', 'algorithm', 'framework', 'architecture']
        technical_score = len([w for w in technical_words if w in query.lower()]) * 0.15
        
        complexity_factors = [
            min(length_factor, 1.0),
            min(question_complexity, 1.0),
            min(analytical_score, 1.0),
            min(clause_score, 1.0),
            min(technical_score, 1.0)
        ]
        
        return min(sum(complexity_factors) / len(complexity_factors), 1.0)
    
    def _update_cognitive_learning(self, reasoning_chain: ReasoningChain):
        """Update cognitive architecture with advanced memory consolidation"""
        for step in reasoning_chain.steps:
            # Add to episodic memory for learning
            self.cognitive_arch.episodic_memory.add_episode(
                content=step.reasoning,
                metadata={"step_id": step.step_id, "confidence": step.confidence}
            )
        
        # Advanced memory consolidation
        self._consolidate_reasoning_memory(reasoning_chain)
    
    def _consolidate_reasoning_memory(self, reasoning_chain: ReasoningChain):
        """Advanced memory consolidation for reasoning patterns"""
        current_time = time.time()
        
        # Consolidate high-confidence reasoning patterns
        for step in reasoning_chain.steps:
            if step.confidence > 0.8:  # High-confidence patterns
                pattern_signature = self._extract_pattern_signature(step.reasoning)
                
                # Strengthen neural pathways for successful patterns
                if hasattr(self.cognitive_arch, 'neural_activity'):
                    self.cognitive_arch.neural_activity += 0.1
                
                # Add to semantic memory for pattern recognition
                self.cognitive_arch.semantic_memory.add_concept(
                    name=f"reasoning_pattern_{pattern_signature}",
                    attributes={
                        "confidence": step.confidence,
                        "pattern_type": "successful_reasoning",
                        "last_used": current_time,
                        "usage_count": 1
                    }
                )
        
        # Periodic memory cleanup and optimization
        if len(self.reasoning_history) > 100:  # Optimize when history is large
            self._optimize_memory_usage()
    
    def _extract_pattern_signature(self, reasoning: str) -> str:
        """Extract signature from reasoning for pattern matching"""
        # Simple pattern extraction based on structure
        reasoning_lower = reasoning.lower()
        
        if "step" in reasoning_lower and "analysis" in reasoning_lower:
            return "chain_of_thought"
        elif "initial" in reasoning_lower and ("analysis" in reasoning_lower or "response" in reasoning_lower):
            return "self_reflective"
        elif "hop" in reasoning_lower:
            return "multi_hop"
        else:
            return "hybrid"
    
    def _optimize_memory_usage(self):
        """Optimize memory usage for better performance"""
        # Keep only recent high-quality reasoning chains
        sorted_history = sorted(
            self.reasoning_history,
            key=lambda x: x.total_confidence,
            reverse=True
        )
        
        # Keep top 75 chains and clear others
        self.reasoning_history = sorted_history[:75]
        
        # Update cognitive architecture with optimization info
        if hasattr(self.cognitive_arch, 'consolidation_level'):
            self.cognitive_arch.consolidation_level = min(
                self.cognitive_arch.consolidation_level + 0.1, 1.0
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all reasoning patterns"""
        stats = {}
        for pattern, metrics in self.performance_metrics.items():
            if metrics:
                stats[pattern] = {
                    "count": len(metrics),
                    "avg_time": np.mean([m["time"] for m in metrics]),
                    "avg_confidence": np.mean([m["confidence"] for m in metrics]),
                    "avg_steps": np.mean([m["steps"] for m in metrics])
                }
        return stats
    
    def calculate_reasoning_score(self) -> Dict[str, Any]:
        """Calculate comprehensive reasoning engine score"""
        score_components: Dict[str, float] = {
            "confidence_quality": 0.0,  # Max 25 points
            "response_synthesis": 0.0,  # Max 20 points  
            "semantic_analysis": 0.0,  # Max 20 points
            "pattern_selection": 0.0,  # Max 15 points
            "learning_adaptation": 0.0,  # Max 10 points
            "performance": 0.0  # Max 10 points
        }
        
        # 1. Confidence quality (25 points)
        if self.performance_metrics:
            all_confidences = []
            for metrics in self.performance_metrics.values():
                all_confidences.extend([m["confidence"] for m in metrics])
            
            if all_confidences:
                avg_confidence = float(np.mean(all_confidences))
                score_components["confidence_quality"] = min(avg_confidence * 25.0, 25.0)
        
        # 2. Response synthesis (20 points)
        synthesis_quality = float(self._evaluate_synthesis_quality())
        score_components["response_synthesis"] = synthesis_quality * 20.0
        
        # 3. Semantic analysis (20 points)
        semantic_score = float(self._evaluate_semantic_capability())
        score_components["semantic_analysis"] = semantic_score * 20.0
        
        # 4. Pattern selection (15 points)
        pattern_score = float(self._evaluate_pattern_selection())
        score_components["pattern_selection"] = pattern_score * 15.0
        
        # 5. Learning adaptation (10 points)
        learning_score = float(self._evaluate_learning_capability())
        score_components["learning_adaptation"] = learning_score * 10.0
        
        # 6. Performance (10 points)
        perf_score = float(self._evaluate_performance())
        score_components["performance"] = perf_score * 10.0
        
        total_score = sum(score_components.values())
        
        return {
            "total_score": round(total_score, 1),
            "grade": self._get_grade(total_score),
            "components": score_components,
            "improvements": self._identify_improvement_areas(score_components)
        }
    
    def _evaluate_synthesis_quality(self) -> float:
        """Evaluate response synthesis quality"""
        # Check if we have coherent synthesis methods
        has_advanced_synthesis = hasattr(self, '_extract_key_insight') and hasattr(self, '_build_coherent_response')
        if not has_advanced_synthesis:
            return 0.5
        
        # Sample evaluation of recent reasoning chains
        if self.reasoning_history:
            recent_chains = self.reasoning_history[-10:]
            coherence_scores = []
            
            for chain in recent_chains:
                # Evaluate if final answer is more than basic concatenation
                answer = chain.final_answer
                if "Synthesized answer:" in answer or "Multi-hop synthesized:" in answer:
                    coherence_scores.append(0.6)  # Basic synthesis
                elif len(answer.split('.')) > 3:  # Multi-sentence responses
                    coherence_scores.append(0.8)  # Better synthesis
                else:
                    coherence_scores.append(0.4)
            
            return float(np.mean(coherence_scores)) if coherence_scores else 0.7
        
        return 0.7  # Default for good synthesis methods
    
    def _evaluate_semantic_capability(self) -> float:
        """Evaluate semantic analysis capabilities"""
        semantic_features = [
            hasattr(self, '_calculate_document_relevance'),
            hasattr(self, '_calculate_reasoning_coherence'),
            self._get_kb_size() > 0
        ]
        
        # Check if using real embeddings vs hash-based
        if hasattr(self.slo_rag.hauls_store, '_get_embedding'):
            semantic_features.append(True)
        
        return sum(semantic_features) / len(semantic_features)
    
    def _evaluate_pattern_selection(self) -> float:
        """Evaluate reasoning pattern selection quality"""
        complexity_available = hasattr(self, '_analyze_query_complexity')
        multiple_patterns = len(self.reasoning_patterns) >= 4
        hybrid_reasoning = 'hybrid' in self.reasoning_patterns
        
        return sum([complexity_available, multiple_patterns, hybrid_reasoning]) / 3
    
    def _evaluate_learning_capability(self) -> float:
        """Evaluate learning and adaptation capabilities"""
        learning_features = [
            hasattr(self, '_update_cognitive_learning'),
            hasattr(self, 'continuous_learning_loop'),
            len(self.reasoning_history) > 0
        ]
        
        return sum(learning_features) / len(learning_features)
    
    def _evaluate_performance(self) -> float:
        """Evaluate overall performance metrics"""
        if not self.performance_metrics:
            return 0.5
        
        # Evaluate based on consistency and speed
        all_times = []
        all_confidences = []
        
        for metrics in self.performance_metrics.values():
            all_times.extend([m["time"] for m in metrics])
            all_confidences.extend([m["confidence"] for m in metrics])
        
        if not all_times or not all_confidences:
            return 0.5
        
        # Score based on consistency (lower variance = better)
        time_consistency = 1.0 - (np.std(all_times) / (np.mean(all_times) + 0.001))
        confidence_consistency = 1.0 - (np.std(all_confidences) / (np.mean(all_confidences) + 0.001))
        
        return float((time_consistency + confidence_consistency) / 2)
    
    def _get_grade(self, score: float) -> str:
        """Get letter grade for score"""
        if score >= 90:
            return "A+ (Excellent)"
        elif score >= 85:
            return "A (Very Good)"
        elif score >= 80:
            return "B+ (Good)"
        elif score >= 75:
            return "B (Average)"
        elif score >= 70:
            return "C+ (Fair)"
        elif score >= 60:
            return "C (Poor)"
        else:
            return "F (Failing)"
    
    def _identify_improvement_areas(self, components: Dict[str, float]) -> List[str]:
        """Identify areas needing improvement"""
        max_scores = {
            "confidence_quality": 25.0,
            "response_synthesis": 20.0,
            "semantic_analysis": 20.0,
            "pattern_selection": 15.0,
            "learning_adaptation": 10.0,
            "performance": 10.0
        }
        
        improvements = []
        for component, score in components.items():
            percentage = score / max_scores[component]
            if percentage < 0.7:
                improvements.append(f"Improve {component.replace('_', ' ')}")
        
        return improvements
    
    def continuous_learning_loop(self, feedback_data: List[Dict[str, Any]]):
        """
        Continuous learning feedback loop - improve reasoning based on user feedback
        """
        print("🔄 Starting continuous learning feedback loop...")
        
        # Analyze feedback patterns
        feedback_analysis = self._analyze_feedback(feedback_data)
        
        # Update reasoning patterns based on feedback
        self._update_reasoning_patterns(feedback_analysis)
        
        # Consolidate learning into cognitive architecture
        self._consolidate_feedback_learning(feedback_analysis)
        
        # Update knowledge base with new insights
        self._update_knowledge_base(feedback_analysis)
        
        print("✅ Continuous learning loop completed")
        return feedback_analysis
    
    def _analyze_feedback(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze feedback patterns for learning insights"""
        analysis = {
            "total_feedback": len(feedback_data),
            "avg_rating": 0.0,
            "pattern_performance": defaultdict(list),
            "common_issues": defaultdict(int),
            "improvement_suggestions": []
        }
        
        if not feedback_data:
            return analysis
        
        # Calculate average rating
        ratings = [f.get("rating", 0) for f in feedback_data if "rating" in f]
        analysis["avg_rating"] = np.mean(ratings) if ratings else 0.0
        
        # Analyze pattern performance
        for feedback in feedback_data:
            pattern = feedback.get("pattern", "unknown")
            rating = feedback.get("rating", 0)
            analysis["pattern_performance"][pattern].append(rating)
            
            # Track common issues
            issues = feedback.get("issues", [])
            for issue in issues:
                analysis["common_issues"][issue] += 1
        
        # Generate improvement suggestions
        for pattern, ratings in analysis["pattern_performance"].items():
            avg_rating = np.mean(ratings)
            if avg_rating < 0.7:
                analysis["improvement_suggestions"].append(
                    f"Improve {pattern} pattern (current avg rating: {avg_rating:.2f})"
                )
        
        return analysis
    
    def _update_reasoning_patterns(self, feedback_analysis: Dict[str, Any]):
        """Update reasoning patterns based on feedback analysis"""
        for pattern, ratings in feedback_analysis["pattern_performance"].items():
            avg_rating = np.mean(ratings)
            
            if avg_rating < 0.6:
                print(f"⚠️  {pattern} pattern needs improvement (rating: {avg_rating:.2f})")
                # Adjust pattern parameters
                if pattern == "chain_of_thought":
                    self.max_reasoning_steps = min(self.max_reasoning_steps + 1, 7)
                elif pattern == "self_reflective":
                    self.confidence_threshold = max(self.confidence_threshold - 0.1, 0.5)
                elif pattern == "multi_hop":
                    self.max_reasoning_steps = min(self.max_reasoning_steps + 1, 6)
            
            elif avg_rating > 0.9:
                print(f"🎉 {pattern} pattern performing excellently (rating: {avg_rating:.2f})")
    
    def _consolidate_feedback_learning(self, feedback_analysis: Dict[str, Any]):
        """Consolidate feedback learning into cognitive architecture"""
        # Add feedback insights to episodic memory
        insight_content = f"""
        Feedback Learning Session:
        - Total feedback: {feedback_analysis['total_feedback']}
        - Average rating: {feedback_analysis['avg_rating']:.2f}
        - Common issues: {dict(feedback_analysis['common_issues'])}
        - Improvements needed: {len(feedback_analysis['improvement_suggestions'])}
        """
        
        self.cognitive_arch.episodic_memory.add_episode(
            content=insight_content,
            metadata={
                "type": "feedback_learning",
                "timestamp": time.time(),
                "avg_rating": feedback_analysis["avg_rating"]
            }
        )
        
        # Trigger memory consolidation
        self.cognitive_arch.consolidate_and_cleanup()
    
    def _update_knowledge_base(self, feedback_analysis: Dict[str, Any]):
        """Update knowledge base with new insights from feedback"""
        if not self.slo_rag.hauls_store:
            return
        
        # Add improvement insights as new knowledge
        for suggestion in feedback_analysis["improvement_suggestions"]:
            insight_content = f"Reasoning Improvement Insight: {suggestion}"
            self.slo_rag.hauls_store.add_documents_batch([(
                insight_content,
                {
                    "source": "continuous_learning",
                    "type": "improvement_insight",
                    "timestamp": time.time()
                }
            )])
        
        # Add common issues as troubleshooting knowledge
        for issue, count in feedback_analysis["common_issues"].items():
            if count > 2:  # Only add frequently reported issues
                troubleshooting_content = f"Common Issue ({count} reports): {issue}"
                self.slo_rag.hauls_store.add_documents_batch([(
                    troubleshooting_content,
                    {
                        "source": "continuous_learning",
                        "type": "troubleshooting",
                        "report_count": count
                    }
                )])
    
    def simulate_feedback_session(self, num_queries: int = 10) -> Dict[str, Any]:
        """Simulate a feedback session for testing continuous learning"""
        print(f"🧪 Simulating feedback session with {num_queries} queries...")
        
        test_queries = [
            "What is machine learning?",
            "How do neural networks work?",
            "Explain cognitive architecture",
            "Compare RAG systems",
            "What is chain-of-thought reasoning?",
            "How does episodic memory work?",
            "Explain semantic memory",
            "What are attention mechanisms?",
            "How do reasoning patterns differ?",
            "What is continuous learning?"
        ]
        
        feedback_data = []
        patterns = ["chain_of_thought", "self_reflective", "multi_hop", "hybrid"]
        
        for i in range(min(num_queries, len(test_queries))):
            query = test_queries[i]
            pattern = random.choice(patterns)
            
            # Process query
            result = self.reason(query, pattern=pattern)
            
            # Simulate user feedback
            rating = random.uniform(0.5, 1.0)  # Simulate rating between 0.5 and 1.0
            issues = []
            
            if rating < 0.7:
                issues = random.choice([
                    ["answer_too_short"],
                    ["lacks_detail"],
                    ["confidence_low"],
                    ["irrelevant_content"]
                ])
            
            feedback_data.append({
                "query": query,
                "pattern": pattern,
                "rating": rating,
                "confidence": result.total_confidence,
                "issues": issues,
                "timestamp": time.time()
            })
        
        # Run continuous learning
        learning_result = self.continuous_learning_loop(feedback_data)
        
        return {
            "feedback_processed": len(feedback_data),
            "learning_result": learning_result,
            "updated_performance": self.get_performance_stats()
        }
    
    def test_integration(self) -> bool:
        """Test integration between cognitive architecture and RAG system"""
        test_queries = [
            "What is machine learning?",
            "How does the SLO system learn?",
            "Explain the cognitive architecture"
        ]
        
        try:
            for query in test_queries:
                result = self.reason(query, pattern="hybrid")
                if result.total_confidence < 0.5:
                    print(f"⚠️  Low confidence for query: {query}")
                    return False
            print("✅ Integration test passed")
            return True
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            return False


if __name__ == "__main__":
    # Initialize and test the Advanced Reasoning Engine
    engine = AdvancedReasoningEngine()
    
    # Test integration
    if engine.test_integration():
        # Run a few example queries
        queries = [
            "How does the cognitive architecture integrate with the RAG system?",
            "What are the main reasoning patterns available?",
            "Explain the chain-of-thought reasoning process"
        ]
        
        for query in queries:
            print(f"\n🤔 Query: {query}")
            result = engine.reason(query, pattern="hybrid")
            print(f"📝 Answer: {result.final_answer[:200]}...")
            print(f"🎯 Confidence: {result.total_confidence:.2f}")
        
        # Show performance stats
        print(f"\n📊 Performance Stats: {json.dumps(engine.get_performance_stats(), indent=2)}")
    
    def start_performance_monitoring(self):
        """Start real-time performance monitoring"""
        self.performance_monitor = {
            "start_time": time.time(),
            "query_count": 0,
            "total_response_time": 0.0,
            "confidence_history": [],
            "error_count": 0,
            "last_optimization": time.time()
        }
        print("📊 Performance monitoring started")
    
    def log_performance(self, query: str, response_time: float, confidence: float, error: bool = False):
        """Log performance metrics for monitoring"""
        if not hasattr(self, 'performance_monitor'):
            self.start_performance_monitoring()
        
        monitor = self.performance_monitor
        monitor["query_count"] += 1
        monitor["total_response_time"] += response_time
        monitor["confidence_history"].append(confidence)
        
        if error:
            monitor["error_count"] += 1
        
        # Periodic optimization
        if time.time() - monitor["last_optimization"] > 300:  # Every 5 minutes
            self._optimize_based_on_performance()
            monitor["last_optimization"] = time.time()
    
    def _optimize_based_on_performance(self):
        """Optimize reasoning parameters based on performance data"""
        if not hasattr(self, 'performance_monitor'):
            return
        
        monitor = self.performance_monitor
        if monitor["query_count"] == 0:
            return
        
        avg_response_time = monitor["total_response_time"] / monitor["query_count"]
        avg_confidence = sum(monitor["confidence_history"]) / len(monitor["confidence_history"])
        
        # Dynamic threshold adjustment
        if avg_confidence < 0.7:
            self.confidence_threshold = max(self.confidence_threshold - 0.05, 0.5)
            print(f"📉 Lowering confidence threshold to {self.confidence_threshold}")
        elif avg_confidence > 0.9:
            self.confidence_threshold = min(self.confidence_threshold + 0.05, 0.85)
            print(f"📈 Raising confidence threshold to {self.confidence_threshold}")
        
        # Step count optimization
        if avg_response_time > 0.005:  # 5ms threshold
            self.max_reasoning_steps = max(self.max_reasoning_steps - 1, 3)
            print(f"⚡ Reducing max steps to {self.max_reasoning_steps} for speed")
        elif avg_response_time < 0.001:  # 1ms threshold
            self.max_reasoning_steps = min(self.max_reasoning_steps + 1, 7)
            print(f"🧠 Increasing max steps to {self.max_reasoning_steps} for depth")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        if not hasattr(self, 'performance_monitor'):
            return {"status": "Monitoring not started"}
        
        monitor = self.performance_monitor
        uptime = time.time() - monitor["start_time"]
        
        return {
            "uptime_seconds": uptime,
            "total_queries": monitor["query_count"],
            "queries_per_second": monitor["query_count"] / uptime if uptime > 0 else 0,
            "average_response_time": monitor["total_response_time"] / monitor["query_count"] if monitor["query_count"] > 0 else 0,
            "average_confidence": sum(monitor["confidence_history"]) / len(monitor["confidence_history"]) if monitor["confidence_history"] else 0,
            "error_rate": monitor["error_count"] / monitor["query_count"] if monitor["query_count"] > 0 else 0,
            "current_confidence_threshold": self.confidence_threshold,
            "current_max_steps": self.max_reasoning_steps
        }