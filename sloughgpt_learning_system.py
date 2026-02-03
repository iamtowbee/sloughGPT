#!/usr/bin/env python3
"""
SloughGPT Learning System
Continuous Learning and Parameter Adjustment
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import logging
from enum import Enum

from sloughgpt_neural_network import SloughGPT, ModelConfig

class LearningType(Enum):
    """Types of learning strategies"""
    REINFORCEMENT = "reinforcement"     # Learn from feedback
    FINE_TUNING = "fine_tuning"         # Gradient-based updates
    BEHAVIOR_CLONING = "behavior_cloning"  # Learn from examples
    SELF_SUPERVISED = "self_supervised"  # Learn from own predictions

@dataclass
class Experience:
    """Single learning experience"""
    prompt: str
    response: str
    feedback: Dict[str, Any]
    timestamp: float
    confidence: float
    reward: float
    thinking_mode: str

@dataclass
class LearningMetrics:
    """Learning performance metrics"""
    total_experiences: int
    average_reward: float
    improvement_rate: float
    confidence_trend: float
    success_rate: float
    last_update: float

class AdaptiveLearningRate:
    """Dynamic learning rate adjustment"""
    
    def __init__(self, initial_lr: float = 1e-4):
        self.current_lr = initial_lr
        self.initial_lr = initial_lr
        self.performance_history = deque(maxlen=100)
        self.adjustment_factor = 0.1
        
    def update(self, performance_score: float) -> float:
        """Update learning rate based on performance"""
        self.performance_history.append(performance_score)
        
        if len(self.performance_history) < 10:
            return self.current_lr
        
        recent_avg = np.mean(list(self.performance_history)[-10:])
        
        if recent_avg > 0.8:  # Performing well, increase lr
            self.current_lr *= (1 + self.adjustment_factor)
        elif recent_avg < 0.4:  # Poor performance, decrease lr
            self.current_lr *= (1 - self.adjustment_factor)
        
        # Keep learning rate in reasonable bounds
        self.current_lr = max(1e-6, min(1e-2, self.current_lr))
        return self.current_lr
    
    def get_current_lr(self) -> float:
        return self.current_lr

class SloughGPLearningSystem:
    """Main learning system for SloughGPT"""
    
    def __init__(self, model: SloughGPT, config: ModelConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        
        # Learning components
        self.experience_buffer = deque(maxlen=10000)
        self.learning_metrics = LearningMetrics(0, 0.0, 0.0, 0.0, 0.0, time.time())
        
        # Adaptive learning components
        self.lr_scheduler = AdaptiveLearningRate()
        self.cognitive_parameters = {
            'creative_confidence': 0.7,
            'analytical_confidence': 0.8,
            'reasoning_depth': 3,
            'creativity_threshold': 0.6,
            'exploration_rate': 0.1
        }
        
        # Optimization setup
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.lr_scheduler.get_current_lr(),
            weight_decay=0.01
        )
        
        # Learning statistics
        self.thinking_performance = defaultdict(list)
        self.pattern_memory = {}
        self.success_patterns = []
        
        self.logger = logging.getLogger(__name__)
    
    def learn_from_interaction(self, prompt: str, response: str, 
                           feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Main learning method - learn from each interaction"""
        
        # Create experience
        experience = Experience(
            prompt=prompt,
            response=response,
            feedback=feedback,
            timestamp=time.time(),
            confidence=feedback.get('confidence', 0.5),
            reward=self._calculate_reward(feedback),
            thinking_mode=feedback.get('thinking_mode', 'analytical')
        )
        
        # Store experience
        self.experience_buffer.append(experience)
        
        # Update learning metrics
        self._update_learning_metrics(experience)
        
        # Learning strategies
        learning_updates = {}
        
        # 1. Reinforcement learning from feedback
        if feedback.get('rating', 0) > 0.7:
            rl_update = self._reinforce_successful_patterns(experience)
            learning_updates['reinforcement'] = rl_update
        
        # 2. Parameter adjustment based on feedback
        param_update = self._adjust_model_parameters(experience)
        learning_updates['parameter_adjustment'] = param_update
        
        # 3. Cognitive parameter learning
        cognitive_update = self._adjust_cognitive_parameters(experience)
        learning_updates['cognitive_learning'] = cognitive_update
        
        # 4. Pattern learning
        pattern_update = self._learn_patterns(experience)
        learning_updates['pattern_learning'] = pattern_update
        
        # 5. Model fine-tuning (if enough data)
        if len(self.experience_buffer) >= 32:
            ft_update = self._fine_tune_model()
            learning_updates['fine_tuning'] = ft_update
        
        return learning_updates
    
    def _calculate_reward(self, feedback: Dict[str, Any]) -> float:
        """Calculate reward signal from feedback"""
        base_reward = feedback.get('rating', 0.5) * 2.0 - 1.0  # Convert to [-1, 1]
        
        # Bonus factors
        if feedback.get('novelty_score', 0) > 0.8:
            base_reward += 0.2
        
        if feedback.get('helpfulness', 0) > 0.8:
            base_reward += 0.2
        
        if feedback.get('accuracy', 0) > 0.9:
            base_reward += 0.3
        
        # Penalty factors
        if feedback.get('coherence', 1.0) < 0.5:
            base_reward -= 0.3
        
        return max(-1.0, min(1.0, base_reward))
    
    def _reinforce_successful_patterns(self, experience: Experience) -> Dict[str, Any]:
        """Reinforce successful response patterns"""
        
        if experience.reward > 0.5:  # Successful interaction
            # Store successful pattern
            pattern_key = f"{experience.thinking_mode}_{hash(experience.prompt[:50]) % 1000}"
            self.pattern_memory[pattern_key] = {
                'response_pattern': experience.response[:200],
                'reward': experience.reward,
                'timestamp': experience.timestamp,
                'confidence': experience.confidence
            }
            
            self.success_patterns.append({
                'mode': experience.thinking_mode,
                'prompt_type': self._classify_prompt(experience.prompt),
                'success_score': experience.reward
            })
            
            return {
                'action': 'pattern_reinforced',
                'pattern_key': pattern_key,
                'reward': experience.reward
            }
        
        return {'action': 'no_reinforcement', 'reason': 'low_reward'}
    
    def _adjust_model_parameters(self, experience: Experience) -> Dict[str, Any]:
        """Adjust model parameters based on experience"""
        
        if abs(experience.reward) < 0.1:  # Neutral experience
            return {'action': 'no_adjustment', 'reason': 'neutral_reward'}
        
        # Update learning rate based on recent performance
        current_lr = self.lr_scheduler.update(self.learning_metrics.average_reward)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # Small gradient-based adjustment
        if experience.reward > 0.3:
            # Positive adjustment - encourage similar responses
            adjustment_factor = 0.01 * experience.reward
        else:
            # Negative adjustment - discourage similar responses
            adjustment_factor = 0.01 * experience.reward
        
        # Apply adjustment to attention weights (focusing mechanism)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'attention' in name and param.requires_grad:
                    param.data += adjustment_factor * torch.randn_like(param.data) * 0.001
        
        return {
            'action': 'parameter_adjusted',
            'adjustment_factor': adjustment_factor,
            'new_learning_rate': current_lr
        }
    
    def _adjust_cognitive_parameters(self, experience: Experience) -> Dict[str, Any]:
        """Learn cognitive parameters from experience"""
        
        updates = {}
        
        # Learn which thinking modes work best
        self.thinking_performance[experience.thinking_mode].append(experience.reward)
        
        if len(self.thinking_performance[experience.thinking_mode]) >= 5:
            avg_performance = np.mean(self.thinking_performance[experience.thinking_mode])
            
            if avg_performance > 0.7:
                # Boost confidence for successful mode
                param_name = f"{experience.thinking_mode}_confidence"
                self.cognitive_parameters[param_name] *= 1.1
                updates[param_name] = 'increased'
            elif avg_performance < 0.3:
                # Decrease confidence for struggling mode
                param_name = f"{experience.thinking_mode}_confidence"
                self.cognitive_parameters[param_name] *= 0.9
                updates[param_name] = 'decreased'
        
        # Learn reasoning depth preference
        if experience.reward > 0.6:
            self.cognitive_parameters['reasoning_depth'] = min(
                5, self.cognitive_parameters['reasoning_depth'] + 0.1
            )
            updates['reasoning_depth'] = 'increased'
        elif experience.reward < 0.3:
            self.cognitive_parameters['reasoning_depth'] = max(
                1, self.cognitive_parameters['reasoning_depth'] - 0.1
            )
            updates['reasoning_depth'] = 'decreased'
        
        # Learn creativity threshold
        if experience.thinking_mode == 'creative' and experience.reward > 0.7:
            self.cognitive_parameters['creativity_threshold'] *= 0.95  # Lower threshold for more creativity
            updates['creativity_threshold'] = 'lowered'
        elif experience.reward < 0.4:
            self.cognitive_parameters['creativity_threshold'] *= 1.05  # Raise threshold for less randomness
            updates['creativity_threshold'] = 'raised'
        
        return updates
    
    def _learn_patterns(self, experience: Experience) -> Dict[str, Any]:
        """Learn interaction patterns"""
        
        # Classify prompt type
        prompt_type = self._classify_prompt(experience.prompt)
        
        # Store pattern success
        pattern_key = f"{prompt_type}_{experience.thinking_mode}"
        
        if pattern_key not in self.pattern_memory:
            self.pattern_memory[pattern_key] = []
        
        self.pattern_memory[pattern_key].append({
            'reward': experience.reward,
            'timestamp': experience.timestamp,
            'confidence': experience.confidence
        })
        
        # Keep only recent patterns
        if len(self.pattern_memory[pattern_key]) > 100:
            self.pattern_memory[pattern_key] = self.pattern_memory[pattern_key][-50:]
        
        return {
            'action': 'pattern_learned',
            'pattern_key': pattern_key,
            'pattern_count': len(self.pattern_memory[pattern_key])
        }
    
    def _fine_tune_model(self) -> Dict[str, Any]:
        """Fine-tune model on recent experiences"""
        
        if len(self.experience_buffer) < 32:
            return {'action': 'insufficient_data'}
        
        # Sample recent experiences for training
        recent_experiences = list(self.experience_buffer)[-32:]
        
        # Create mini-batch training
        self.model.train()
        total_loss = 0
        
        for experience in recent_experiences:
            if experience.reward > 0.5:  # Only train on positive examples
                try:
                    # Tokenize (simplified - would need proper tokenizer)
                    prompt_tokens = self._simple_tokenize(experience.prompt)[:50]
                    target_tokens = self._simple_tokenize(experience.response)[:50]
                    
                    if len(prompt_tokens) < 2 or len(target_tokens) < 1:
                        continue
                    
                    input_ids = torch.tensor([prompt_tokens], device=self.device)
                    targets = torch.tensor([target_tokens], device=self.device)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    logits = self.model(input_ids)
                    
                    # Calculate loss
                    loss = nn.CrossEntropyLoss()(
                        logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                        targets[:, 1:].contiguous().view(-1)
                    )
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    self.optimizer.step()
                    total_loss += loss.item()
                    
                except Exception as e:
                    self.logger.warning(f"Training error: {e}")
                    continue
        
        avg_loss = total_loss / max(1, len(recent_experiences))
        
        self.model.eval()
        
        return {
            'action': 'fine_tuned',
            'experiences_used': len(recent_experiences),
            'average_loss': avg_loss,
            'learning_rate': self.lr_scheduler.get_current_lr()
        }
    
    def _classify_prompt(self, prompt: str) -> str:
        """Classify prompt type for pattern learning"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['what is', 'define', 'explain']):
            return 'definition'
        elif any(word in prompt_lower for word in ['how to', 'create', 'design']):
            return 'creation'
        elif any(word in prompt_lower for word in ['why', 'cause', 'reason']):
            return 'analysis'
        elif any(word in prompt_lower for word in ['compare', 'difference', 'versus']):
            return 'comparison'
        else:
            return 'general'
    
    def _simple_tokenize(self, text: str) -> List[int]:
        """Simple tokenization for demonstration"""
        # In real implementation, would use proper tokenizer
        return [hash(word) % 1000 for word in text.split()]
    
    def _update_learning_metrics(self, experience: Experience):
        """Update learning performance metrics"""
        
        self.learning_metrics.total_experiences += 1
        
        # Update average reward
        if self.learning_metrics.total_experiences == 1:
            self.learning_metrics.average_reward = experience.reward
        else:
            alpha = 0.1  # Learning rate for metrics
            self.learning_metrics.average_reward = (
                alpha * experience.reward + 
                (1 - alpha) * self.learning_metrics.average_reward
            )
        
        # Update success rate
        success = 1 if experience.reward > 0.5 else 0
        if self.learning_metrics.total_experiences == 1:
            self.learning_metrics.success_rate = success
        else:
            self.learning_metrics.success_rate = (
                alpha * success + 
                (1 - alpha) * self.learning_metrics.success_rate
            )
        
        # Update improvement rate (trend)
        if len(self.experience_buffer) >= 10:
            recent_rewards = [exp.reward for exp in list(self.experience_buffer)[-10:]]
            older_rewards = [exp.reward for exp in list(self.experience_buffer)[-20:-10]]
            
            if older_rewards:
                recent_avg = np.mean(recent_rewards)
                older_avg = np.mean(older_rewards)
                self.learning_metrics.improvement_rate = recent_avg - older_avg
        
        self.learning_metrics.last_update = time.time()
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get comprehensive learning status"""
        
        return {
            'learning_metrics': asdict(self.learning_metrics),
            'cognitive_parameters': self.cognitive_parameters,
            'learning_rate': self.lr_scheduler.get_current_lr(),
            'experience_buffer_size': len(self.experience_buffer),
            'pattern_memory_size': len(self.pattern_memory),
            'thinking_performance': {
                mode: {
                    'count': len(performance),
                    'average': np.mean(performance) if performance else 0,
                    'trend': 'improving' if len(performance) > 5 and np.mean(performance[-5:]) > np.mean(performance[-10:-5]) else 'stable'
                }
                for mode, performance in self.thinking_performance.items()
            }
        }
    
    def save_learning_state(self, filepath: str):
        """Save learning state"""
        state = {
            'learning_metrics': asdict(self.learning_metrics),
            'cognitive_parameters': self.cognitive_parameters,
            'pattern_memory': {k: v for k, v in self.pattern_memory.items() if isinstance(v, (int, float, str)) or len(v) < 100},
            'success_patterns': self.success_patterns[-100:],  # Last 100 patterns
            'thinking_performance': {k: v[-50:] for k, v in self.thinking_performance.items()},  # Last 50 per mode
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)

# Test learning system
def test_learning_system():
    """Test the learning system"""
    print("🧠 SloughGPT Learning System")
    print("=" * 50)
    
    # Create model and learning system
    config = ModelConfig(vocab_size=1000, d_model=256, n_heads=8, n_layers=4)
    model = SloughGPT(config)
    learning_system = SloughGPLearningSystem(model, config)
    
    print("✅ Learning system initialized")
    print(f"✅ Initial cognitive parameters: {learning_system.cognitive_parameters}")
    
    # Simulate learning interactions
    print("\n🎓 Simulating Learning Interactions")
    print("-" * 40)
    
    test_interactions = [
        {
            'prompt': 'What is artificial intelligence?',
            'response': 'AI is the simulation of human intelligence by machines.',
            'feedback': {'rating': 0.8, 'accuracy': 0.9, 'helpfulness': 0.7, 'thinking_mode': 'analytical'}
        },
        {
            'prompt': 'Create a poem about nature',
            'response': 'In forests deep where shadows play, the ancient trees do sway and sway...',
            'feedback': {'rating': 0.9, 'novelty_score': 0.8, 'creativity': 0.9, 'thinking_mode': 'creative'}
        },
        {
            'prompt': 'Explain quantum computing',
            'response': 'Quantum computing uses quantum bits to perform calculations...',
            'feedback': {'rating': 0.6, 'accuracy': 0.7, 'clarity': 0.5, 'thinking_mode': 'analytical'}
        }
    ]
    
    for i, interaction in enumerate(test_interactions, 1):
        print(f"\n📝 Interaction {i}")
        print(f"   Prompt: {interaction['prompt'][:50]}...")
        print(f"   Feedback Rating: {interaction['feedback']['rating']}")
        
        updates = learning_system.learn_from_interaction(
            interaction['prompt'],
            interaction['response'], 
            interaction['feedback']
        )
        
        print(f"   Learning Updates: {list(updates.keys())}")
        print(f"   Cognitive Parameters: {learning_system.cognitive_parameters}")
    
    # Get final status
    status = learning_system.get_learning_status()
    
    print(f"\n📊 Final Learning Status")
    print("=" * 30)
    print(f"Total Experiences: {status['learning_metrics']['total_experiences']}")
    print(f"Average Reward: {status['learning_metrics']['average_reward']:.3f}")
    print(f"Success Rate: {status['learning_metrics']['success_rate']:.3f}")
    print(f"Improvement Rate: {status['learning_metrics']['improvement_rate']:.3f}")
    print(f"Current Learning Rate: {status['learning_rate']:.6f}")
    
    print(f"\n🧠 Cognitive Parameters:")
    for param, value in status['cognitive_parameters'].items():
        print(f"   {param}: {value:.3f}")
    
    print(f"\n🎯 Thinking Performance:")
    for mode, perf in status['thinking_performance'].items():
        print(f"   {mode}: {perf['average']:.3f} avg, {perf['trend']} trend")
    
    # Save learning state
    learning_system.save_learning_state("sloughgpt_learning_state.json")
    print(f"\n💾 Learning state saved")
    
    print(f"\n🎉 Learning System Test Complete!")

if __name__ == "__main__":
    test_learning_system()