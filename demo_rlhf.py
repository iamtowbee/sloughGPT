#!/usr/bin/env python3
"""
RLHF Fine-tuning Demo for SloughGPT

This demonstrates how to fine-tune a model using Reinforcement Learning from Human Feedback.

Usage:
    python3 demo_rlhf.py
"""

import random
from dataclasses import dataclass
from typing import List, Tuple

# Disable CUDA to avoid blocking
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''


@dataclass
class PreferencePair:
    """A preference pair for RLHF training."""
    prompt: str
    chosen_response: str
    rejected_response: str


def create_sample_preferences() -> List[PreferencePair]:
    """Create sample preference data for demonstration."""
    return [
        PreferencePair(
            prompt="What is Python?",
            chosen_response="Python is a high-level programming language created by Guido van Rossum in 1991. It's known for its simple syntax and is widely used in web development, data science, and AI.",
            rejected_response="Python is a snake. It crawls on the ground."
        ),
        PreferencePair(
            prompt="Explain machine learning",
            chosen_response="Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses algorithms to identify patterns and make decisions.",
            rejected_response="ML is when machines learn things. Like robots."
        ),
        PreferencePair(
            prompt="What is RAG?",
            chosen_response="RAG (Retrieval-Augmented Generation) combines a language model with a retrieval system. It fetches relevant documents and uses them as context for generating responses, reducing hallucinations.",
            rejected_response="RAG is a farm animal. It gives milk."
        ),
        PreferencePair(
            prompt="Explain transformers in AI",
            chosen_response="Transformers are neural network architectures introduced in 2017. They use self-attention mechanisms to process input sequences in parallel, enabling efficient handling of long-range dependencies.",
            rejected_response="Transformers are toys that turn into robots."
        ),
        PreferencePair(
            prompt="What is catastrophic forgetting?",
            chosen_response="Catastrophic forgetting is when a neural network forgets previously learned information when learning new tasks. It's a major challenge in continual learning.",
            rejected_response="It's when you forget things as you get older."
        ),
    ]


class RewardModel:
    """Simple reward model that scores responses."""
    
    def __init__(self):
        self.weights = {}
        self.bias = 0.0
    
    def score(self, response: str) -> float:
        """Score a response based on quality heuristics."""
        score = 0.5  # Base score
        
        # Length bonus (not too short, not too long)
        word_count = len(response.split())
        if 20 <= word_count <= 100:
            score += 0.15
        
        # Contains technical terms
        technical_terms = [
            'programming', 'learning', 'model', 'neural', 'data', 
            'algorithm', 'language', 'computer', 'training', 'network'
        ]
        for term in technical_terms:
            if term in response.lower():
                score += 0.05
        
        # Negative indicators
        negative_terms = ['snake', 'farm', 'milk', 'toy', 'robot']
        for term in negative_terms:
            if term in response.lower():
                score -= 0.15
        
        return max(0.0, min(1.0, score))
    
    def update(self, chosen_score: float, rejected_score: float, lr: float = 0.1):
        """Update model based on preference (simulated gradient update)."""
        # If chosen < rejected, increase chosen, decrease rejected
        if chosen_score < rejected_score:
            diff = rejected_score - chosen_score
            # Simulate learning
            print(f"    Learning: adjusting weights (diff={diff:.3f})")


class RLHFDemo:
    """Demonstrates RLHF training process."""
    
    def __init__(self):
        self.preferences = create_sample_preferences()
        self.reward_model = RewardModel()
        self.training_history = []
    
    def train_step(self) -> Tuple[float, float]:
        """Perform one RLHF training step."""
        # Sample a preference pair
        pref = random.choice(self.preferences)
        
        # Compute rewards for both responses
        chosen_score = self.reward_model.score(pref.chosen_response)
        rejected_score = self.reward_model.score(pref.rejected_response)
        
        # Update model if needed
        self.reward_model.update(chosen_score, rejected_score)
        
        return chosen_score, rejected_score
    
    def run(self, num_steps: int = 20):
        """Run RLHF demo."""
        print("=" * 60)
        print("RLHF FINE-TUNING DEMO")
        print("=" * 60)
        
        print("\nInitial Reward Scores:")
        print("-" * 40)
        for pref in self.preferences:
            chosen = self.reward_model.score(pref.chosen_response)
            rejected = self.reward_model.score(pref.rejected_response)
            print(f"Prompt: {pref.prompt[:40]}...")
            print(f"  Chosen: {chosen:.3f}, Rejected: {rejected:.3f}")
        
        print("\n" + "=" * 60)
        print("TRAINING REWARD MODEL")
        print("=" * 60)
        
        print("\nTraining for {} steps...".format(num_steps))
        print("-" * 40)
        
        for step in range(num_steps):
            chosen_r, rejected_r = self.train_step()
            self.training_history.append((chosen_r, rejected_r))
            
            if (step + 1) % 5 == 0:
                avg_chosen = sum(h[0] for h in self.training_history[-5:]) / 5
                avg_rejected = sum(h[1] for h in self.training_history[-5:]) / 5
                margin = avg_chosen - avg_rejected
                print(f"Step {step + 1:2d}: Avg Chosen={avg_chosen:.3f}, Avg Rejected={avg_rejected:.3f}, Margin={margin:+.3f}")
        
        print("\n" + "=" * 60)
        print("EVALUATION")
        print("=" * 60)
        
        # Evaluate on all preferences
        print("\nFinal Preference Scores:")
        all_correct = 0
        for pref in self.preferences:
            chosen_r = self.reward_model.score(pref.chosen_response)
            rejected_r = self.reward_model.score(pref.rejected_response)
            
            correct = chosen_r > rejected_r
            if correct:
                all_correct += 1
            
            print(f"\nPrompt: {pref.prompt[:45]}...")
            status = "✓" if correct else "✗"
            print(f"  Chosen: {chosen_r:.3f} {status}")
            print(f"  Rejected: {rejected_r:.3f}")
        
        accuracy = all_correct / len(self.preferences) * 100
        print("\n" + "=" * 60)
        print(f"PREFERENCE ACCURACY: {accuracy:.0f}% ({all_correct}/{len(self.preferences)})")
        print("=" * 60)
        
        print("""
In production, RLHF would:
1. Collect human preference data (e.g., "which response is better?")
2. Train a reward model on the preference data
3. Use PPO (Proximal Policy Optimization) to fine-tune the LLM
4. Apply KL divergence penalty to stay close to the original model
5. Use GAE (Generalized Advantage Estimation) for stable training

To use the production RLHF:
    from domains.training.rlhf import PPOTrainer, RLHFConfig
    
    config = RLHFConfig(
        ppo_epochs=4,
        clip_epsilon=0.2,
        gamma=1.0,
        lam=0.95,
    )
    
    trainer = PPOTrainer(model, config)
    trainer.train(preference_dataset)
""")


if __name__ == "__main__":
    demo = RLHFDemo()
    demo.run(num_steps=20)
