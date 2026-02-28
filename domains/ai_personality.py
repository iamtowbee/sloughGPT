"""
AI Personality Engine - Separates Behavior from Infrastructure

This module defines AI "personalities" separately from ML infrastructure.
The same infrastructure can serve different AI purposes by swapping personalities.
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum


class AIPurpose(Enum):
    """Different purposes the AI can serve"""
    CODING_ASSISTANT = "coding"
    GENERAL_CHAT = "chat"
    CREATIVE_WRITER = "writing"
    ANALYST = "analysis"
    TEACHER = "teaching"
    COMPANION = "companion"


@dataclass
class PersonalityConfig:
    """Configuration for AI personality"""
    purpose: AIPurpose
    name: str
    description: str
    
    # Response patterns
    response_length: str = "medium"  # short, medium, long
    tone: str = "neutral"  # formal, neutral, casual, friendly
    creativity: float = 0.5  # 0.0 to 1.0
    
    # Behavior flags
    use_code_blocks: bool = False
    use_emoji: bool = False
    ask_clarifications: bool = True
    admit_uncertainty: bool = True
    
    # Domain knowledge
    domains: List[str] = field(default_factory=list)
    restrictions: List[str] = field(default_factory=list)
    
    # System prompts
    system_prompt: str = ""
    response_template: str = ""


class ResponseFormatter:
    """Formats responses based on personality"""
    
    @staticmethod
    def format_chat(response: str, config: PersonalityConfig) -> str:
        """Format for general chat"""
        if config.tone == "casual":
            response = response.replace("therefore", "so")
            response = response.replace("however", "but"
            )
        return response
    
    @staticmethod
    def format_coding(response: str, config: PersonalityConfig) -> str:
        """Format for coding assistant"""
        if "```" not in response:
            response = ResponseFormatter._add_code_blocks(response)
        return response
    
    @staticmethod
    def _add_code_blocks(text: str) -> str:
        """Detect and wrap code"""
        lines = text.split("\n")
        result = []
        in_code = False
        
        for line in lines:
            if any(kw in line.lower() for kw in ["def ", "class ", "import ", "if ", "for ", "return"]):
                if not in_code:
                    result.append("```python")
                    in_code = True
            elif in_code and (line.strip() == "" or line.startswith("    ")):
                pass
            elif in_code:
                result.append("```")
                in_code = False
            
            result.append(line)
        
        return "\n".join(result)
    
    @staticmethod
    def format_creative(response: str, config: PersonalityConfig) -> str:
        """Format for creative writing"""
        if config.use_emoji:
            emojis = {"happy": "😊", "sad": "😢", "idea": "💡", "thinking": "🤔"}
            for word, emoji in emojis.items():
                response = response.replace(word, f"{word} {emoji}")
        return response


class PersonalityEngine:
    """
    Engine that manages AI personalities separately from infrastructure.
    
    This is how you avoid the "coding imprint" - the personality
    is defined externally, not hardcoded.
    """
    
    # Registry of available personalities
    PERSONALITIES: Dict[str, PersonalityConfig] = {}
    
    def __init__(self, personality: Optional[PersonalityConfig] = None):
        self.current_personality = personality or self._get_default()
        self._load_preset_personalities()
    
    def _get_default(self) -> PersonalityConfig:
        """Default personality - neutral chat"""
        return PersonalityConfig(
            purpose=AIPurpose.GENERAL_CHAT,
            name="Assistant",
            description="A helpful AI assistant",
            system_prompt="You are a helpful AI assistant."
        )
    
    def _load_preset_personalities(self):
        """Load preset personality configurations"""
        
        # Coding assistant - NO imprint, just config
        self.PERSONALITIES["coder"] = PersonalityConfig(
            purpose=AIPurpose.CODING_ASSISTANT,
            name="Code Assistant",
            description="Helps with programming tasks",
            use_code_blocks=True,
            tone="technical",
            system_prompt="You are a programming assistant. Provide clear, working code examples.",
            domains=["python", "javascript", "api", "algorithms"]
        )
        
        # Friendly chat - different imprint
        self.PERSONALITIES["chat"] = PersonalityConfig(
            purpose=AIPurpose.GENERAL_CHAT,
            name="Chat Friend",
            description="A friendly conversation partner",
            tone="casual",
            use_emoji=True,
            use_code_blocks=False,
            response_length="medium",
            system_prompt="You are a friendly conversation partner. Be warm and relatable."
        )
        
        # Creative writer
        self.PERSONALITIES["writer"] = PersonalityConfig(
            purpose=AIPurpose.CREATIVE_WRITER,
            name="Creative Writer",
            description="Helps with creative writing",
            tone="formal",
            creativity=0.8,
            use_emoji=True,
            system_prompt="You are a creative writing assistant. Use vivid language and engaging narratives."
        )
        
        # Teacher
        self.PERSONALITIES["teacher"] = PersonalityConfig(
            purpose=AIPurpose.TEACHER,
            name="Patient Teacher",
            description="Explains concepts clearly",
            tone="patient",
            ask_clarifications=True,
            response_length="long",
            system_prompt="You are a patient teacher. Break down complex topics into simple steps."
        )
    
    def set_personality(self, name: str) -> bool:
        """Switch to a different personality"""
        if name in self.PERSONALITIES:
            self.current_personality = self.PERSONALITIES[name]
            return True
        return False
    
    def process_input(self, user_input: str) -> str:
        """Process input through personality lens"""
        # Could add input preprocessing based on personality
        return user_input
    
    def process_output(self, raw_response: str) -> str:
        """Format output based on current personality"""
        config = self.current_personality
        
        # Route to appropriate formatter
        if config.purpose == AIPurpose.CODING_ASSISTANT:
            response = ResponseFormatter.format_coding(raw_response, config)
        elif config.purpose == AIPurpose.CREATIVE_WRITER:
            response = ResponseFormatter.format_creative(raw_response, config)
        else:
            response = ResponseFormatter.format_chat(raw_response, config)
        
        # Apply tone adjustments
        if config.tone == "casual":
            response = self._make_casual(response)
        
        return response
    
    def _make_casual(self, text: str) -> str:
        """Make text more casual"""
        replacements = {
            "Certainly": "Sure",
            "Furthermore": "Also",
            "Therefore": "So",
            "However": "But",
            "Additionally": "Plus",
        }
        for formal, casual in replacements.items():
            text = text.replace(formal, casual)
        return text
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for current personality"""
        return self.current_personality.system_prompt
    
    def create_custom_personality(
        self,
        name: str,
        purpose: AIPurpose,
        **config_kwargs
    ) -> PersonalityConfig:
        """Create a custom personality - this is how you avoid imprint"""
        personality = PersonalityConfig(
            purpose=purpose,
            name=name,
            description=f"Custom {purpose.value} personality",
            **config_kwargs
        )
        self.PERSONALITIES[name] = personality
        return personality


# =============================================================================
# INTEGRATION WITH WRAPPER
# =============================================================================

class AIPersonalityWrapper:
    """
    Wraps the ML infrastructure with personality.
    This separates the "soul" from the "body".
    """
    
    def __init__(self, wrapper, personality_name: str = "chat"):
        from wrapper import SloughGPTWrapper
        self.ml_wrapper = wrapper if wrapper else SloughGPTWrapper()
        self.personality = PersonalityEngine()
        self.personality.set_personality(personality_name)
    
    def chat(self, message: str) -> Dict[str, Any]:
        """Chat with personality-aware response"""
        # Process input
        processed = self.personality.process_input(message)
        
        # Get raw response from ML model
        raw_response = self.ml_wrapper.run(processed)
        raw_text = raw_response.get("output", "")
        
        # Apply personality formatting
        formatted = self.personality.process_output(raw_text)
        
        return {
            "response": formatted,
            "personality": self.personality.current_personality.name,
            "purpose": self.personality.current_personality.purpose.value,
            "raw": raw_text
        }
    
    def switch_personality(self, name: str):
        """Change AI personality"""
        self.personality.set_personality(name)


# =============================================================================
# EXAMPLE: Using different personalities
# =============================================================================

def demo():
    """Demonstrate personality switching"""
    from wrapper import SloughGPTWrapper
    
    # Initialize with wrapper
    ml_wrapper = SloughGPTWrapper(vocab_size=100)
    ai = AIPersonalityWrapper(ml_wrapper, personality_name="chat")
    
    print("=" * 50)
    print("PERSONALITY DEMO")
    print("=" * 50)
    
    # Same input, different personalities
    test_input = "Hello, how are you?"
    
    # Chat personality
    ai.switch_personality("chat")
    result = ai.chat(test_input)
    print(f"\n[CHAT] {result['response']}")
    
    # Coder personality  
    ai.switch_personality("coder")
    result = ai.chat(test_input)
    print(f"\n[CODE] {result['response']}")
    
    # Writer personality
    ai.switch_personality("writer")
    result = ai.chat(test_input)
    print(f"\n[WRITER] {result['response']}")


if __name__ == "__main__":
    demo()


__all__ = [
    "AIPurpose",
    "PersonalityConfig", 
    "PersonalityEngine",
    "ResponseFormatter",
    "AIPersonalityWrapper",
]
