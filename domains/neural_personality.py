"""
Neural Personality - Simplified Working Version

Real backpropagation training for personality.
"""

import random
import math
from typing import Dict, List


class SimpleNeuralNetwork:
    """Simplified neural network that actually works"""
    
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 16):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        # Xavier initialization
        self.W1 = [[random.gauss(0, 0.5) for _ in range(hidden_size)] for _ in range(input_size)]
        self.b1 = [0.0] * hidden_size
        self.W2 = [[random.gauss(0, 0.5) for _ in range(output_size)] for _ in range(hidden_size)]
        self.b2 = [0.0] * output_size
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def sigmoid_deriv(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def relu(self, x):
        return max(0, x)
    
    def relu_deriv(self, x):
        return 1.0 if x > 0 else 0.0
    
    def forward(self, inputs: List[float]) -> List[float]:
        # Hidden layer
        self.hidden = []
        for j in range(self.hidden_size):
            val = self.b1[j]
            for i in range(self.input_size):
                val += inputs[i] * self.W1[i][j]
            self.hidden.append(self.relu(val))
        
        # Output layer
        outputs = []
        for j in range(self.output_size):
            val = self.b2[j]
            for i in range(self.hidden_size):
                val += self.hidden[i] * self.W2[i][j]
            outputs.append(self.sigmoid(val))
        
        return outputs
    
    def train(self, inputs: List[float], targets: List[float], lr: float = 0.1):
        # Forward pass
        outputs = self.forward(inputs)
        
        # Output error
        output_errors = [outputs[i] - targets[i] for i in range(self.output_size)]
        
        # Hidden error
        hidden_errors = []
        for i in range(self.hidden_size):
            err = sum(output_errors[j] * self.W2[i][j] for j in range(self.output_size))
            hidden_errors.append(err * self.relu_deriv(self.hidden[i]))
        
        # Update W2 and b2
        for i in range(self.hidden_size):
            for j in range(self.output_size):
                self.W2[i][j] -= lr * output_errors[j] * self.hidden[i]
        for j in range(self.output_size):
            self.b2[j] -= lr * output_errors[j]
        
        # Update W1 and b1
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.W1[i][j] -= lr * hidden_errors[j] * inputs[i]
        for j in range(self.hidden_size):
            self.b1[j] -= lr * hidden_errors[j]


class NeuralPersonality:
    """Neural personality that learns from training"""
    
    TRAIT_NAMES = ["warmth", "formality", "creativity", "empathy", 
                   "patience", "confidence", "humor", "directness"]
    
    def __init__(self, vocab_size: int = 100):
        self.vocab = {}
        self.embeddings = []
        
        # Simple embedding: each word gets a random vector
        for i in range(vocab_size):
            self.embeddings.append([random.gauss(0, 0.1) for _ in range(8)])
        
        # Neural network: 8 input (avg embedding) -> 16 hidden -> 8 output (traits)
        self.network = SimpleNeuralNetwork(input_size=8, output_size=8, hidden_size=16)
        
        self.trained = False
    
    def _tokenize(self, text: str) -> List[int]:
        words = text.lower().split()
        indices = []
        for w in words:
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab) % len(self.embeddings)
            indices.append(self.vocab[w])
        return indices
    
    def _get_embedding(self, idx: int) -> List[float]:
        if 0 <= idx < len(self.embeddings):
            return self.embeddings[idx]
        return [0] * 8
    
    def _mean_pooling(self, indices: List[int]) -> List[float]:
        if not indices:
            return [0] * 8
        pool = [0] * 8
        for idx in indices:
            emb = self._get_embedding(idx)
            for i in range(8):
                pool[i] += emb[i]
        return [v / len(indices) for v in pool]
    
    def train(self, examples: List[Dict], epochs: int = 100, lr: float = 0.1):
        print(f"Training neural personality on {len(examples)} examples...")
        
        for epoch in range(epochs):
            total_loss = 0
            
            for ex in examples:
                tokens = self._tokenize(ex.get("text", ""))
                inputs = self._mean_pooling(tokens)
                
                # Target traits
                raw = ex.get("traits", {})
                targets = [(raw.get(name, 0) + 1) / 2 for name in self.TRAIT_NAMES]
                
                # Forward + backward
                self.network.train(inputs, targets, lr)
                
                # Loss
                out = self.network.forward(inputs)
                loss = sum((out[i] - targets[i])**2 for i in range(8))
                total_loss += loss
            
            if epoch % 25 == 0:
                print(f"  Epoch {epoch}: loss = {total_loss/len(examples):.4f}")
        
        self.trained = True
        print("Training complete!")
    
    def predict_traits(self, text: str) -> Dict[str, float]:
        if not self.trained:
            return {"error": "Not trained"}
        
        tokens = self._tokenize(text)
        inputs = self._mean_pooling(tokens)
        outputs = self.network.forward(inputs)
        
        return {name: (outputs[i] - 0.5) * 2 for i, name in enumerate(self.TRAIT_NAMES)}
    
    def generate_response(self, text: str) -> str:
        traits = self.predict_traits(text)
        
        warmth = traits.get("warmth", 0)
        formality = traits.get("formality", 0)
        
        words = []
        
        if warmth > 0.3:
            words.extend(["wonderful", "great", "happy"])
        elif warmth < -0.3:
            words.extend(["interesting", "noted", "understood"])
        else:
            words.extend(["good", "nice"])
        
        if formality > 0.3:
            words.append("certainly")
        elif formality < -0.3:
            words.append("yeah")
        
        return " ".join(words[:5])


def demo():
    print("=" * 60)
    print("NEURAL PERSONALITY - REAL BACKPROPAGATION")
    print("=" * 60)
    
    # Create
    np = NeuralPersonality(vocab_size=100)
    
    # Training data - this is where personality "soul" comes from
    data = [
        {"text": "Hey friend! How wonderful to see you today!", 
         "traits": {"warmth": 0.9, "formality": -0.8, "humor": 0.7}},
        {"text": "I would like to submit this proposal formally.",
         "traits": {"warmth": 0.0, "formality": 0.9, "confidence": 0.7}},
        {"text": "Imagine a world of endless possibilities!",
         "traits": {"warmth": 0.7, "creativity": 0.9, "humor": 0.5}},
        {"text": "I understand how you feel. Tell me more.",
         "traits": {"empathy": 0.9, "patience": 0.8, "warmth": 0.7}},
    ]
    
    # Train
    np.train(data, epochs=100, lr=0.1)
    
    print("\nTesting:")
    for text in ["Hello friend!", "I have a formal request.", "Tell me something creative!"]:
        traits = np.predict_traits(text)
        resp = np.generate_response(text)
        print(f"\n'{text}'")
        print(f"  Traits: {dict((k,f'{v:.2f}') for k,v in traits.items() if abs(v)>0.1)}")
        print(f"  Response: {resp}")


if __name__ == "__main__":
    demo()


__all__ = ["NeuralPersonality", "SimpleNeuralNetwork"]
