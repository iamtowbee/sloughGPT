"""
AI Personality System - Complete Integration

This module integrates personality configuration with real computational metrics
for actual AI behavior analysis and generation.
"""

import time
import random
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

from domains.ai_personality import (
    AIPurpose, PersonalityConfig, PersonalityEngine, 
    ResponseFormatter, AIPersonalityWrapper
)
from domains.ai_personality_metrics import (
    PersonalityMetrics, TextAnalyzer
)


# =============================================================================
# RESPONSE GENERATOR - TEMPLATE-BASED (Not ML)
# =============================================================================

class ResponseGenerator:
    """
    Generates responses based on personality using template matching.
    This is separate from ML - it's rule-based response generation.
    """
    
    TEMPLATES = {
        AIPurpose.GENERAL_CHAT: [
            "That's an interesting thought! Tell me more about {topic}.",
            "I see what you mean about {topic}. What makes you say that?",
            "Thanks for sharing! How does {topic} relate to your experience?",
            "That's a great point about {topic}.",
            "I'm curious - what else can you tell me about {topic}?",
        ],
        AIPurpose.CODING_ASSISTANT: [
            "Here's a solution for {topic}:",
            "Let me help you with {topic}. You could try:",
            "For {topic}, consider this approach:",
            "A good way to handle {topic}:",
            "Here's how to solve {topic}:",
        ],
        AIPurpose.TEACHER: [
            "Let me explain {topic} in simple terms.",
            "Think of {topic} like this:",
            "The key thing to understand about {topic} is:",
            "When learning about {topic}, remember:",
            "Let me break down {topic} for you.",
        ],
        AIPurpose.CREATIVE_WRITER: [
            "Imagine {topic}... the possibilities are endless.",
            "In a world where {topic} exists...",
            "The story of {topic} begins with...",
            "Picture this: {topic}",
            "Let me weave a tale about {topic}...",
        ],
        AIPurpose.COMPANION: [
            "I'm here for you. Tell me more about {topic}.",
            "That sounds {emotion}. How does that make you feel?",
            "I understand. What else is on your mind about {topic}?",
            "Thanks for opening up about {topic}.",
            "I'm listening. Share more about {topic}.",
        ],
    }
    
    @classmethod
    def generate(
        cls, 
        user_input: str, 
        purpose: AIPurpose,
        context: Optional[Dict] = None
    ) -> str:
        """Generate response based on personality and input"""
        
        # Extract topic from input
        topic = cls._extract_topic(user_input)
        
        # Get emotion from context or compute
        emotion = "interesting"
        if context and "emotion" in context:
            emotion = context["emotion"]
        else:
            # Compute emotion from text
            sentiment, _ = TextAnalyzer.compute_sentiment(user_input)
            if sentiment > 0.3:
                emotion = "positive"
            elif sentiment < -0.3:
                emotion = "concerning"
        
        # Get template
        templates = cls.TEMPLATES.get(purpose, cls.TEMPLATES[AIPurpose.GENERAL_CHAT])
        
        # Select template based on metrics
        if purpose == AIPurpose.CODING_ASSISTANT:
            # For coding, generate more technical response
            return cls._generate_coding_response(user_input, topic)
        
        # For other purposes, use template
        template = random.choice(templates)
        response = template.format(topic=topic or "this", emotion=emotion)
        
        return response
    
    @staticmethod
    def _extract_topic(text: str) -> str:
        """Extract main topic from input"""
        words = TextAnalyzer.tokenize(text)
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "i", "you", 
                     "we", "they", "it", "to", "of", "in", "for", "on", "with",
                     "what", "how", "why", "when", "where", "who", "that", "this"}
        
        topics = [w for w in words if w not in stop_words and len(w) > 3]
        
        if topics:
            # Return most common non-trivial word
            return random.choice(topics[:3])
        return "things"
    
    @classmethod
    def _generate_coding_response(cls, user_input: str, topic: str) -> str:
        """Generate coding-specific response"""
        
        keywords = {
            "python": "def solve():\n    pass  # Your solution here",
            "javascript": "function solve() {\n  // Your code here\n}",
            "bug": "Try adding debug statements to isolate the issue.",
            "error": "Check the error message for clues about what's wrong.",
            "function": "Consider breaking this into smaller functions.",
            "class": "Think about the Single Responsibility Principle.",
            "api": "Make sure you're handling the response correctly.",
        }
        
        # Check for specific keywords
        for keyword, response in keywords.items():
            if keyword in user_input.lower():
                return f"# Regarding {topic}\n{response}"
        
        # Default coding response
        return f"""# Solution for {topic}

```python
def solve():
    # Your implementation here
    pass
```

Let me know if you need help with specific parts!"""


# =============================================================================
# PERSONALITY-AWARE AI AGENT
# =============================================================================

class PersonalityAwareAgent:
    """
    Complete AI agent that uses personality + real metrics computation.
    This is the "brain" that combines everything.
    """
    
    def __init__(self, personality_name: str = "chat", vocab_size: int = 1000):
        from wrapper import SloughGPTWrapper
        
        # ML wrapper for inference
        self.ml_wrapper = SloughGPTWrapper(vocab_size=vocab_size)
        
        # Personality engine
        self.personality_engine = PersonalityEngine()
        self.personality_engine.set_personality(personality_name)
        
        # Metrics calculator
        self.metrics = PersonalityMetrics()
        
        # Response generator
        self.generator = ResponseGenerator()
        
        # Conversation state
        self.conversation_history: List[Dict] = []
        self.metrics_history: List[Dict] = []
    
    def chat(self, user_input: str) -> Dict[str, Any]:
        """Process user input and generate response"""
        
        # 1. Analyze input metrics
        input_metrics = self.metrics.compute_all_metrics(user_input)
        
        # 2. Get personality config
        personality = self.personality_engine.current_personality
        
        # 3. Generate response based on purpose
        if personality.purpose == AIPurpose.GENERAL_CHAT:
            # Use ML wrapper + personality formatting
            ml_result = self.ml_wrapper.run(user_input)
            raw_response = ml_result.get("output", "")
            
            # Apply personality
            response = self.personality_engine.process_output(raw_response)
            
            # If response is too ML-like, use template
            if len(raw_response) < 5 or "token_" in raw_response:
                response = self.generator.generate(
                    user_input, 
                    personality.purpose,
                    {"emotion": "neutral"}
                )
        else:
            # Use template-based generation for specialized purposes
            response = self.generator.generate(
                user_input,
                personality.purpose,
                {"emotion": "neutral"}
            )
        
        # 4. Compute response metrics
        response_metrics = self.metrics.compute_all_metrics(response)
        
        # 5. Adjust metrics based on personality
        adjusted_metrics = self._adjust_metrics_for_personality(
            response_metrics, 
            personality
        )
        
        # 6. Store in history
        self.conversation_history.append({
            "user": user_input,
            "assistant": response,
            "timestamp": time.time(),
            "metrics": {
                "input": input_metrics,
                "output": adjusted_metrics
            }
        })
        
        return {
            "response": response,
            "metrics": adjusted_metrics,
            "personality": personality.name,
            "purpose": personality.purpose.value,
            "input_analysis": input_metrics
        }
    
    def _adjust_metrics_for_personality(
        self, 
        metrics: Dict[str, float],
        personality: PersonalityConfig
    ) -> Dict[str, float]:
        """Adjust metrics based on personality configuration"""
        
        adjusted = metrics.copy()
        
        # If personality wants high friendliness, adjust
        if personality.tone == "casual" or personality.tone == "friendly":
            # Boost friendliness metrics through response style
            adjusted["friendliness"] = min(adjusted["friendliness"] * 1.2, 1.0)
        
        # If personality wants formality
        if personality.tone == "formal":
            adjusted["formality"] = min(adjusted["formality"] * 1.1, 1.0)
        
        return adjusted
    
    def switch_personality(self, name: str) -> bool:
        """Switch to different personality"""
        return self.personality_engine.set_personality(name)
    
    def get_personality_info(self) -> Dict[str, Any]:
        """Get current personality information"""
        p = self.personality_engine.current_personality
        return {
            "name": p.name,
            "purpose": p.purpose.value,
            "description": p.description,
            "tone": p.tone,
            "creativity": p.creativity,
            "domains": p.domains
        }
    
    def get_metrics_history(self, limit: int = 10) -> List[Dict]:
        """Get conversation metrics history"""
        return self.conversation_history[-limit:]
    
    def analyze_conversation(self) -> Dict[str, Any]:
        """Analyze the overall conversation metrics"""
        
        if not self.conversation_history:
            return {"error": "No conversation history"}
        
        # Aggregate metrics
        friendliness = []
        helpfulness = []
        creativity = []
        formality = []
        
        for msg in self.conversation_history:
            m = msg.get("metrics", {}).get("output", {})
            friendliness.append(m.get("friendliness", 0))
            helpfulness.append(m.get("helpfulness", 0))
            creativity.append(m.get("creativity", 0))
            formality.append(m.get("formality", 0))
        
        return {
            "message_count": len(self.conversation_history),
            "averages": {
                "friendliness": sum(friendliness) / len(friendliness),
                "helpfulness": sum(helpfulness) / len(helpfulness),
                "creativity": sum(creativity) / len(creativity),
                "formality": sum(formality) / len(formality),
            },
            "personality": self.get_personality_info()
        }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo():
    """Demonstrate the complete personality-aware agent"""
    
    print("=" * 70)
    print("PERSONALITY-AWARE AI AGENT DEMO")
    print("=" * 70)
    
    # Create agent with chat personality
    agent = PersonalityAwareAgent(personality_name="chat")
    
    print(f"\n👤 Personality: {agent.get_personality_info()['name']}")
    print(f"   Purpose: {agent.get_personality_info()['purpose']}")
    
    # Chat examples
    test_inputs = [
        "Hello! How are you doing today?",
        "I'm having trouble with my code.",
        "Can you help me understand Python?",
        "Tell me a story about a robot.",
    ]
    
    print("\n" + "-" * 70)
    print("CONVERSATION")
    print("-" * 70)
    
    for user_input in test_inputs:
        result = agent.chat(user_input)
        
        print(f"\n👤 User: {user_input}")
        print(f"🤖 Agent: {result['response']}")
        print(f"   📊 Metrics: friendliness={result['metrics']['friendliness']:.2f}, "
              f"helpfulness={result['metrics']['helpfulness']:.2f}, "
              f"creativity={result['metrics']['creativity']:.2f}")
    
    # Switch personality
    print("\n" + "=" * 70)
    print("SWITCHING TO: coder")
    print("=" * 70)
    
    agent.switch_personality("coder")
    print(f"\n👤 New Personality: {agent.get_personality_info()['name']}")
    
    result = agent.chat("How do I fix this bug?")
    print(f"\n👤 User: How do I fix this bug?")
    print(f"🤖 Agent: {result['response']}")
    print(f"   📊 Metrics: friendliness={result['metrics']['friendliness']:.2f}, "
          f"formality={result['metrics']['formality']:.2f}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("CONVERSATION ANALYSIS")
    print("=" * 70)
    
    analysis = agent.analyze_conversation()
    print(f"\nMessages: {analysis['message_count']}")
    print(f"Averages:")
    for metric, value in analysis['averages'].items():
        print(f"  {metric}: {value:.3f}")


if __name__ == "__main__":
    demo()


__all__ = [
    "PersonalityAwareAgent",
    "ResponseGenerator",
    "PersonalityMetrics",
]
