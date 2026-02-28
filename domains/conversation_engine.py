"""
AI Conversation Engine - Real ML Model Integration

This integrates the personality system with real ML inference
and applies actual computational metrics.
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wrapper import SloughGPTWrapper
from domains.ai_personality import PersonalityEngine, PersonalityConfig, AIPurpose
from domains.ai_personality_metrics import PersonalityMetrics, TextAnalyzer


@dataclass
class ConversationMessage:
    """A single conversation message"""
    role: str  # user, assistant, system
    content: str
    timestamp: float = field(default_factory=time.time)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Conversation context for the AI"""
    messages: List[ConversationMessage] = field(default_factory=list)
    user_name: str = "User"
    session_id: str = ""
    
    def add_message(self, role: str, content: str):
        """Add a message and compute its metrics"""
        metrics = PersonalityMetrics.compute_all_metrics(content)
        msg = ConversationMessage(role=role, content=content, metrics=metrics)
        self.messages.append(msg)
        
    def get_recent(self, n: int = 5) -> List[ConversationMessage]:
        """Get last n messages"""
        return self.messages[-n:]


class ConversationEngine:
    """
    Real conversation engine that:
    - Uses ML wrapper for inference
    - Applies personality configuration
    - Computes real metrics on responses
    - Tracks conversation state
    """
    
    def __init__(
        self,
        wrapper: Optional[SloughGPTWrapper] = None,
        personality_name: str = "chat",
        track_metrics: bool = True
    ):
        # ML Model (the "brain")
        self.wrapper = wrapper or SloughGPTWrapper(vocab_size=1000)
        
        # Personality system
        self.personality = PersonalityEngine()
        self.personality.set_personality(personality_name)
        
        # Metrics tracking
        self.track_metrics = track_metrics
        
        # Conversation state
        self.context = ConversationContext()
        self.response_count = 0
        
        # Metrics history
        self.metrics_history: List[Dict[str, float]] = []
    
    def chat(self, user_input: str) -> Dict[str, Any]:
        """
        Main chat method - processes input and returns response
        with real computed metrics.
        """
        # 1. Add user message to context
        self.context.add_message("user", user_input)
        
        # 2. Get personality config
        personality = self.personality.current_personality
        
        # 3. Prepare prompt with personality
        system_prompt = personality.system_prompt
        prompt = f"{system_prompt}\n\nUser: {user_input}\nAssistant:"
        
        # 4. Get raw response from ML model
        raw_response = self.wrapper.run(prompt)
        raw_text = raw_response.get("output", "I'm not sure how to respond to that.")
        
        # 5. Apply personality-based formatting
        formatted_text = self._apply_personality(raw_text, personality)
        
        # 6. Compute REAL metrics on the response
        response_metrics = PersonalityMetrics.compute_all_metrics(formatted_text)
        
        # 7. Add assistant message with metrics
        self.context.add_message("assistant", formatted_text)
        
        # 8. Track metrics over time
        self.metrics_history.append(response_metrics)
        self.response_count += 1
        
        # 9. Build comprehensive response
        result = {
            "response": formatted_text,
            "personality": personality.name,
            "metrics": response_metrics,
            "context": {
                "message_count": len(self.context.messages),
                "session_id": self.context.session_id
            }
        }
        
        # 10. Add aggregate metrics if enough data
        if len(self.metrics_history) >= 3:
            result["aggregate_metrics"] = self._compute_aggregate_metrics()
        
        return result
    
    def _apply_personality(self, text: str, personality: PersonalityConfig) -> str:
        """Apply personality-based transformations"""
        
        # Tone adjustments based on config
        if personality.tone == "casual":
            # Replace formal words with casual
            replacements = {
                "Certainly": "Sure",
                "However": "But",
                "Therefore": "So",
                "Furthermore": "Also",
                "Unfortunately": "Aw",
                "I understand": "I get it",
                "I would": "I'll",
                "Please": "",
                "Thank you": "Thanks",
            }
            for formal, casual in replacements.items():
                text = text.replace(formal, casual)
        
        elif personality.tone == "formal":
            # Make more formal
            replacements = {
                "Hey": "Hello",
                "Hi": "Greetings",
                "Sure": "Certainly",
                "But": "However",
                "So": "Therefore",
            }
            for casual, formal in replacements.items():
                text = text.replace(casual, formal)
        
        # Add emojis if configured
        if personality.use_emoji:
            if "?" in text and not any(e in text for e in ["😊", "🙂", "🤔"]):
                text += " 😊"
            if "!" in text and not any(e in text for e in ["👍", "✨", "💡"]):
                text += " ✨"
        
        # Adjust response length
        if personality.response_length == "short":
            sentences = text.split('.')
            if len(sentences) > 2:
                text = '. '.join(sentences[:2]) + '.'
        elif personality.response_length == "long":
            # Expand short responses
            if len(text.split()) < 20:
                text += " Is there anything else I can help you with?"
        
        return text
    
    def _compute_aggregate_metrics(self) -> Dict[str, float]:
        """Compute aggregate metrics over conversation"""
        if not self.metrics_history:
            return {}
        
        # Average each metric
        num_responses = len(self.metrics_history)
        
        averages = {}
        for key in ["friendliness", "helpfulness", "creativity", "formality"]:
            values = [m[key] for m in self.metrics_history if key in m]
            if values:
                averages[f"avg_{key}"] = sum(values) / len(values)
                averages[f"min_{key}"] = min(values)
                averages[f"max_{key}"] = max(values)
        
        return averages
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        if not self.metrics_history:
            return {"status": "No responses yet"}
        
        latest = self.metrics_history[-1]
        aggregate = self._compute_aggregate_metrics() if len(self.metrics_history) >= 3 else {}
        
        return {
            "response_count": self.response_count,
            "current_personality": self.personality.current_personality.name,
            "latest_metrics": latest,
            "aggregate_metrics": aggregate,
            "metrics_trend": self._compute_trend()
        }
    
    def _compute_trend(self) -> Dict[str, str]:
        """Compute trend direction for each metric"""
        if len(self.metrics_history) < 3:
            return {}
        
        trends = {}
        for key in ["friendliness", "helpfulness", "creativity", "formality"]:
            values = [m[key] for m in self.metrics_history[-5:] if key in m]
            if len(values) >= 3:
                first_half = sum(values[:len(values)//2]) / (len(values)//2)
                second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
                
                if second_half > first_half + 0.05:
                    trends[key] = "increasing"
                elif second_half < first_half - 0.05:
                    trends[key] = "decreasing"
                else:
                    trends[key] = "stable"
        
        return trends
    
    def switch_personality(self, name: str) -> bool:
        """Switch to a different personality"""
        success = self.personality.set_personality(name)
        if success:
            # Add system message about personality change
            self.context.add_message(
                "system",
                f"Personality changed to {self.personality.current_personality.name}"
            )
        return success
    
    def export_conversation(self) -> Dict[str, Any]:
        """Export full conversation with metrics"""
        return {
            "session_id": self.context.session_id,
            "personality": self.personality.current_personality.name,
            "message_count": len(self.context.messages),
            "metrics_summary": self.get_metrics_summary(),
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "metrics": m.metrics,
                    "timestamp": m.timestamp
                }
                for m in self.context.messages
            ]
        }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo():
    """Demonstrate the conversation engine"""
    
    print("=" * 60)
    print("CONVERSATION ENGINE DEMO")
    print("=" * 60)
    
    # Initialize engine
    engine = ConversationEngine(personality_name="chat")
    
    # Test different personalities
    test_inputs = [
        "Hello! How are you?",
        "I need help with something.",
        "Can you explain what AI is?",
        "That's really interesting!",
        "Thanks for your help!"
    ]
    
    for user_input in test_inputs:
        print(f"\n{'='*50}")
        print(f"User: {user_input}")
        
        result = engine.chat(user_input)
        
        print(f"Assistant: {result['response']}")
        print(f"Personality: {result['personality']}")
        print(f"Metrics: ", end="")
        for key, val in result['metrics'].items():
            print(f"{key}={val:.2f} ", end="")
        print()
    
    # Show metrics summary
    print(f"\n{'='*60}")
    print("METRICS SUMMARY")
    print("=" * 60)
    
    summary = engine.get_metrics_summary()
    print(f"Response count: {summary['response_count']}")
    print(f"Current personality: {summary['current_personality']}")
    print(f"Latest metrics: {summary['latest_metrics']}")
    
    if 'aggregate_metrics' in summary:
        print(f"Aggregate: {summary['aggregate_metrics']}")
    
    if summary.get('metrics_trend'):
        print(f"Trends: {summary['metrics_trend']}")


if __name__ == "__main__":
    demo()


__all__ = [
    "ConversationMessage",
    "ConversationContext", 
    "ConversationEngine",
]
