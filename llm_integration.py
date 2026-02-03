#!/usr/bin/env python3
"""
LLM Integration Module for Advanced Reasoning Engine

Integrates with various LLM providers for enhanced response generation
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
import requests
from pathlib import Path

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if LLM provider is available"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        if not self.is_available():
            return "OpenAI API key not available"
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"API Error: {response.status_code}"
                
        except Exception as e:
            return f"LLM Error: {str(e)[:100]}"

class LocalProvider(LLMProvider):
    """Local/ollama provider"""
    
    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = f"{base_url}/api/generate"
    
    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url.replace('/api/generate', '/api/tags')}", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        if not self.is_available():
            return "Local LLM not available - ensure ollama is running"
        
        try:
            data = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": False
            }
            
            response = requests.post(self.base_url, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response generated")
            else:
                return f"Local LLM Error: {response.status_code}"
                
        except Exception as e:
            return f"Local LLM Error: {str(e)[:100]}"

class MockLLMProvider(LLMProvider):
    """Mock provider for testing and fallback"""
    
    def __init__(self):
        self.templates = {
            "analytical": "Based on the provided context and logical analysis, {query} can be understood through multiple perspectives. The evidence suggests {insight}.",
            "factual": "According to the available information, {query} is addressed by {insight}. The documents clearly indicate {evidence}.",
            "general": "The information provided reveals important insights about {query}. {insight} emerges from the context."
        }
    
    def is_available(self) -> bool:
        return True
    
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        # Extract query type from prompt
        if "analyze" in prompt.lower() or "why" in prompt.lower() or "how" in prompt.lower():
            template_type = "analytical"
        elif "what" in prompt.lower() or "who" in prompt.lower() or "when" in prompt.lower():
            template_type = "factual"
        else:
            template_type = "general"
        
        template = self.templates[template_type]
        insight = "key patterns and relationships" if template_type == "analytical" else "specific details and facts"
        evidence = "clear supporting documentation" if template_type == "factual" else "relevant contextual information"
        
        return template.format(
            query=prompt[:50] + "...",
            insight=insight,
            evidence=evidence
        )

class LLMIntegrator:
    """Main LLM integration coordinator"""
    
    def __init__(self, preferred_provider: str = "auto"):
        self.preferred_provider = preferred_provider
        self.providers = []
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available LLM providers"""
        # Try OpenAI first
        openai_provider = OpenAIProvider()
        if openai_provider.is_available():
            self.providers.append(("openai", openai_provider))
            print("✅ OpenAI provider available")
        
        # Try local ollama
        local_provider = LocalProvider()
        if local_provider.is_available():
            self.providers.append(("local", local_provider))
            print("✅ Local LLM provider available")
        
        # Always have mock fallback
        mock_provider = MockLLMProvider()
        self.providers.append(("mock", mock_provider))
        print("✅ Mock LLM provider available (fallback)")
        
        if not self.providers:
            raise ValueError("No LLM providers available")
        
        # Select preferred provider
        self.current_provider = self._select_provider()
        print(f"🧠 Using LLM provider: {self.current_provider[0]}")
    
    def _select_provider(self):
        """Select the best available provider"""
        if self.preferred_provider == "auto":
            # Prefer OpenAI > Local > Mock
            for name, provider in self.providers:
                if provider.is_available():
                    return (name, provider)
        
        # Try specific provider
        for name, provider in self.providers:
            if name == self.preferred_provider and provider.is_available():
                return (name, provider)
        
        # Fallback to first available
        return self.providers[0]
    
    def generate_reasoned_response(self, query: str, context: str, reasoning_type: str = "analytical") -> str:
        """Generate response with LLM integration"""
        
        # Build prompt for LLM
        prompt = self._build_reasoning_prompt(query, context, reasoning_type)
        
        # Generate response
        provider_name, provider = self.current_provider
        start_time = time.time()
        
        response = provider.generate_response(prompt, max_tokens=500)
        
        generation_time = time.time() - start_time
        
        # Log generation info
        print(f"🤖 {provider_name.title()} response generated in {generation_time:.3f}s")
        
        return response
    
    def _build_reasoning_prompt(self, query: str, context: str, reasoning_type: str) -> str:
        """Build specialized prompt for reasoning type"""
        
        base_prompt = f"""
You are an advanced reasoning assistant integrating knowledge from a comprehensive database.

QUERY: {query}

CONTEXT FROM KNOWLEDGE BASE:
{context[:1000]}...

REASONING TYPE: {reasoning_type}

Instructions:
- Analyze the provided context thoroughly
- Provide a comprehensive response based on the information
- Use logical reasoning and connect concepts
- If information is insufficient, acknowledge limitations
- Keep response focused and relevant to the query

RESPONSE:
"""
        
        # Specialized instructions for different reasoning types
        if reasoning_type == "chain_of_thought":
            base_prompt += """
Provide a step-by-step analysis:
1. Identify key aspects of the query
2. Extract relevant information from context
3. Build logical connections
4. Conclude with synthesis
"""
        elif reasoning_type == "self_reflective":
            base_prompt += """
Provide initial analysis, then reflect:
1. Initial assessment
2. Identify potential limitations
3. Refine with deeper insights
4. Final balanced perspective
"""
        elif reasoning_type == "multi_hop":
            base_prompt += """
Chain information through multiple connections:
1. Extract initial facts
2. Connect to related concepts  
3. Build upon connections
4. Synthesize comprehensive view
"""
        
        return base_prompt
    
    def switch_provider(self, provider_name: str):
        """Switch to different LLM provider"""
        for name, provider in self.providers:
            if name == provider_name and provider.is_available():
                self.current_provider = (name, provider)
                print(f"🔄 Switched to {name.title()} provider")
                return True
        
        print(f"❌ Provider {provider_name} not available")
        return False
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers"""
        status = {
            "current": self.current_provider[0],
            "available": []
        }
        
        for name, provider in self.providers:
            if provider.is_available():
                status["available"].append(name)
        
        return status

# Usage example
if __name__ == "__main__":
    print("🧠 Testing LLM Integration...")
    
    integrator = LLMIntegrator()
    
    # Test reasoning generation
    query = "How does Shakespeare develop Hamlet's character throughout the play?"
    context = "Hamlet is the Prince of Denmark who faces internal conflict and philosophical questions..."
    
    response = integrator.generate_reasoned_response(query, context, "analytical")
    print(f"🤖 LLM Response: {response[:200]}...")
    
    # Check provider status
    status = integrator.get_provider_status()
    print(f"📊 Provider Status: {status}")