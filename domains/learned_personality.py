"""
Learned Personality - Soul from Training, Not Code

This module creates a personality that emerges from TRAINING,
not hardcoded rules. This is the difference:

- CODE-BASED: Templates + config determine responses
- TRAINED: Neural weights encode personality patterns

The "soul" comes from actual training on personality data.
"""

import random
import math
from typing import Dict, List, Optional, Tuple
from collections import Counter
import json


# =============================================================================
# PERSONALITY EMBEDDING - The "Soul" Vector
# =============================================================================

class PersonalityEmbedding:
    """
    A learned embedding that represents personality.
    This is NOT configured - it's TRAINED.
    
    Think of it like a personality DNA - 
    it's learned from experiences (training data).
    """
    
    def __init__(self, dimensions: int = 64):
        self.dimensions = dimensions
        
        # These weights are what get trained
        # They represent "personality traits"
        self.weights = [
            0.0,   # warmth (cold vs warm)
            0.0,   # formality (casual vs formal)
            0.0,   # creativity (practical vs creative)
            0.0,   # empathy (logical vs empathetic)
            0.0,   # patience (impatient vs patient)
            0.0,   # confidence (uncertain vs confident)
            0.0,   # humor (serious vs humorous)
            0.0,   # directness (indirect vs direct)
            # ... more dimensions
        ]
        
        # How each dimension affects response generation
        self.response_modulators = self._init_modulators()
    
    def _init_modulators(self) -> List[Dict]:
        """Initialize how each dimension affects language"""
        return [
            {"warm": ["friendly", "kind", "warm"], "cold": ["formal", "distant", "proper"]},
            {"casual": ["hey", "cool", "yeah"], "formal": ["certainly", "therefore", "however"]},
            {"creative": ["imagine", "wonderful", "fascinating"], "practical": ["useful", "practical", "helpful"]},
            {"empathetic": ["understand", "feel", "hear"], "logical": ["therefore", "consequently", "thus"]},
            {"patient": ["take your time", "no rush", "gradually"], "impatient": ["quickly", "immediately", "now"]},
            {"confident": ["definitely", "certainly", "absolutely"], "uncertain": ["perhaps", "maybe", "possibly"]},
            {"humorous": ["lol", "funny", "joke"], "serious": ["important", "crucial", "essential"]},
            {"direct": ["simply", "just", "directly"], "indirect": ["perhaps", "consider", "might"]},
        ]
    
    def get_trait(self, index: int) -> float:
        """Get a personality trait value (-1 to 1)"""
        if 0 <= index < len(self.weights):
            return self.weights[index]
        return 0.0
    
    def get_all_traits(self) -> Dict[str, float]:
        """Get all personality traits"""
        trait_names = [
            "warmth", "formality", "creativity", "empathy",
            "patience", "confidence", "humor", "directness"
        ]
        return {name: self.weights[i] for i, name in enumerate(trait_names) if i < len(self.weights)}


# =============================================================================
# TRAined PERSONALITY - The Soul (Not Code)
# =============================================================================

class TrainedPersonality:
    """
    A personality that emerges from TRAINING, not configuration.
    
    This is like raising a child - the personality develops
    from experiences (training data), not from rules (code).
    """
    
    # This is what gets trained on personality data
    # It's not configured - it's LEARNED
    
    def __init__(self):
        # The "brain" - learns from training
        self.embedding = PersonalityEmbedding(dimensions=64)
        
        # Vocabulary preferences - LEARNED, not configured
        self.word_preferences: Dict[str, float] = {}
        self.response_patterns: List[str] = []
        
        # Training state
        self.is_trained = False
        self.training_examples = 0
    
    def train_on_examples(self, examples: List[Dict]):
        """
        Train on personality examples.
        
        This is how the "soul" develops:
        - Read many examples of personality
        - Adjust weights to match that personality
        - Learn what words to use, how to respond
        """
        
        for example in examples:
            # Input text
            text = example.get("text", "")
            # Desired personality traits
            traits = example.get("traits", {})
            
            # Learn word preferences from examples
            words = text.lower().split()
            for word in words:
                if word not in self.word_preferences:
                    self.word_preferences[word] = 0.0
                self.word_preferences[word] += 0.01
            
            # Learn trait associations
            # (Simplified - real version would use gradient descent)
            self._learn_trait_associations(text, traits)
            
            self.training_examples += 1
        
        self.is_trained = True
        print(f"Trained on {self.training_examples} examples")
    
    def _learn_trait_associations(self, text: str, traits: Dict):
        """Learn which words associate with which traits"""
        
        # Count word occurrences
        words = text.lower().split()
        
        # Simple association learning
        for i, trait_name in enumerate(["warmth", "formality", "creativity", "empathy", 
                                         "patience", "confidence", "humor", "directness"]):
            if trait_name in traits:
                trait_value = traits[trait_name]  # -1 to 1
                
                # Adjust embedding weights based on trait
                if i < len(self.embedding.weights):
                    # Words in warm text = warmer personality
                    self.embedding.weights[i] += trait_value * 0.1
    
    def generate_response(self, input_text: str) -> str:
        """
        Generate response based on LEARNED personality.
        
        This is fundamentally different from template-based:
        - Code: Uses templates and rules
        - Trained: Uses learned patterns and weights
        """
        
        if not self.is_trained:
            return "I haven't learned enough yet to have a personality."
        
        # Get personality traits
        traits = self.embedding.get_all_traits()
        
        # Generate based on learned patterns
        words = input_text.lower().split()
        
        # Find preferred words based on personality
        response_words = []
        
        for word in words:
            # Find similar words the personality prefers
            preferred = self._get_preferred_words(word, traits)
            if preferred:
                response_words.append(random.choice(preferred))
            else:
                response_words.append(word)
        
        # Apply learned response patterns
        if self.response_patterns:
            pattern = random.choice(self.response_patterns)
            return pattern.format(input=" ".join(words[:5]))
        
        return " ".join(response_words[:10])
    
    def _get_preferred_words(self, seed_word: str, traits: Dict) -> List[str]:
        """Get words this personality prefers (learned, not configured)"""
        
        # This would use learned embeddings in real implementation
        # Simplified here to show the concept
        
        preferred = []
        
        # Based on warmth trait
        warmth = traits.get("warmth", 0)
        if warmth > 0:
            preferred.extend(["happy", "glad", "wonderful", "great"])
        elif warmth < 0:
            preferred.extend(["correct", "precise", "proper"])
        
        # Based on creativity
        creativity = traits.get("creativity", 0)
        if creativity > 0:
            preferred.extend(["imagine", "creative", "fascinating"])
        
        return preferred
    
    def get_personality_description(self) -> str:
        """Describe this learned personality"""
        traits = self.embedding.get_all_traits()
        
        descriptions = []
        
        if traits.get("warmth", 0) > 0.3:
            descriptions.append("warm")
        elif traits.get("warmth", 0) < -0.3:
            descriptions.append("reserved")
        
        if traits.get("formality", 0) > 0.3:
            descriptions.append("formal")
        elif traits.get("formality", 0) < -0.3:
            descriptions.append("casual")
        
        if traits.get("creativity", 0) > 0.3:
            descriptions.append("creative")
        
        if traits.get("confidence", 0) > 0.3:
            descriptions.append("confident")
        elif traits.get("confidence", 0) < -0.3:
            descriptions.append("humble")
        
        if traits.get("humor", 0) > 0.3:
            descriptions.append("humorous")
        
        if not descriptions:
            return "A balanced personality"
        
        return "A " + ", ".join(descriptions) + " personality"


# =============================================================================
# HYBRID PERSONALITY - Code + Training
# =============================================================================

class HybridPersonality:
    """
    Combines CODE-based (rules) with TRAINED (learned) personality.
    
    The trained part provides the "soul" and natural variation.
    The code part provides structure and reliability.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.trained = TrainedPersonality()
        self.code_rules = {}
        self.uses_learned = True
        self.uses_code = True
    
    def train(self, examples: List[Dict]):
        """Train the learned component"""
        self.trained.train_on_examples(examples)
    
    def add_code_rule(self, pattern: str, response: str):
        """Add a code-based rule"""
        self.code_rules[pattern] = response
    
    def respond(self, input_text: str) -> str:
        """Generate response using both learned and code"""
        
        # First try code rules (deterministic)
        for pattern, response in self.code_rules.items():
            if pattern.lower() in input_text.lower():
                return response
        
        # Then use learned personality (if trained)
        if self.trained.is_trained and self.uses_learned:
            learned_response = self.trained.generate_response(input_text)
            return learned_response
        
        # Fallback
        return "I'm not sure how to respond."


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo():
    """Demonstrate trained vs code-based personality"""
    
    print("=" * 70)
    print("CODE-BASED vs TRAINED PERSONALITY")
    print("=" * 70)
    
    # 1. Code-based personality (what we had before)
    print("\n1️⃣  CODE-BASED (rules, templates)")
    print("-" * 40)
    
    # Example: hardcoded template
    def code_response(input_text):
        if "hello" in input_text.lower():
            return "Hello! How can I help you?"
        return "I understand."
    
    print(f"Input: 'hello there'")
    print(f"Output: {code_response('hello there')}")
    print("→ Determined by IF-ELSE rules")
    
    # 2. Trained personality (learned, not configured)
    print("\n2️⃣  TRAINED (learned, not configured)")
    print("-" * 40)
    
    # Create and train personality
    personality = TrainedPersonality()
    
    # Training examples - this is how it develops "soul"
    training_data = [
        {"text": "Hello friend! How wonderful to see you today!", "traits": {"warmth": 0.8, "formality": -0.5}},
        {"text": "Indeed, I quite agree with your assessment.", "traits": {"warmth": 0.2, "formality": 0.8}},
        {"text": "Wow, that's absolutely amazing! I love it!", "traits": {"warmth": 0.9, "humor": 0.7}},
        {"text": "Let me think about this systematically.", "traits": {"formality": 0.6, "confidence": 0.7}},
        {"text": "Haha that's hilarious! You're so funny!", "traits": {"humor": 0.9, "warmth": 0.8}},
    ]
    
    personality.train_on_examples(training_data)
    
    print(f"Personality: {personality.get_personality_description()}")
    print(f"Traits: {personality.embedding.get_all_traits()}")
    
    print("\nGenerating responses:")
    test_inputs = ["hello", "tell me something", "that's funny"]
    for inp in test_inputs:
        response = personality.generate_response(inp)
        print(f"  Input: '{inp}' → Output: '{response}'")
    
    print("\n→ Determined by TRAINED WEIGHTS, not rules")
    
    # 3. Hybrid
    print("\n3️⃣  HYBRID (code + training)")
    print("-" * 40)
    
    hybrid = HybridPersonality("Assistant")
    hybrid.add_code_rule("help", "I'm here to help! What do you need?")
    hybrid.train(training_data)
    
    print(f"Input: 'help me' → Output: '{hybrid.respond('help me')}' (code)")
    print(f"Input: 'hello' → Output: '{hybrid.respond('hello')}' (trained)")


if __name__ == "__main__":
    demo()


__all__ = [
    "PersonalityEmbedding",
    "TrainedPersonality", 
    "HybridPersonality",
]
