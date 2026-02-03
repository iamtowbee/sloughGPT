#!/usr/bin/env python3
"""
SloughGPT Advanced Learning System - Spaced Repetition Scheduler
Implements intelligent review scheduling based on performance and forgetting curves
"""

import time
import json
import math
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from pathlib import Path

class Difficulty(Enum):
    VERY_EASY = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    VERY_HARD = 5

class MemoryStrength(Enum):
    WEAK = 1
    LEARNING = 2
    YOUNG = 3
    MATURE = 4
    STRONG = 5

@dataclass
class LearningItem:
    """Represents a single learning item in the spaced repetition system"""
    id: str
    content: str
    concept: str
    difficulty: Difficulty
    created_at: float
    last_reviewed: float
    next_review: float
    review_count: int
    success_count: int
    failure_count: int
    memory_strength: MemoryStrength
    ease_factor: float
    interval: float  # days until next review
    
    def __post_init__(self):
        if self.ease_factor == 0:
            self.ease_factor = 2.5  # Default ease factor
        if self.interval == 0:
            self.interval = 1.0  # Start with 1 day interval

class SpacedRepetitionScheduler:
    """Advanced Spaced Repetition Learning System for SloughGPT"""
    
    def __init__(self, db_path: str = "slo_learning.db"):
        self.db_path = db_path
        self.conn = None
        self.items: Dict[str, LearningItem] = {}
        self._init_database()
        self._load_items()
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_items (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                concept TEXT NOT NULL,
                difficulty INTEGER NOT NULL,
                created_at REAL NOT NULL,
                last_reviewed REAL NOT NULL,
                next_review REAL NOT NULL,
                review_count INTEGER NOT NULL,
                success_count INTEGER NOT NULL,
                failure_count INTEGER NOT NULL,
                memory_strength INTEGER NOT NULL,
                ease_factor REAL NOT NULL,
                interval REAL NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS review_sessions (
                session_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                total_items INTEGER NOT NULL,
                successful_reviews INTEGER NOT NULL,
                failed_reviews INTEGER NOT NULL,
                avg_response_time REAL
            )
        ''')
        
        self.conn.commit()
    
    def _load_items(self):
        """Load all learning items from database"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM learning_items")
        rows = cursor.fetchall()
        
        for row in rows:
            self.items[row[0]] = LearningItem(
                id=row[0],
                content=row[1],
                concept=row[2],
                difficulty=Difficulty(row[3]),
                created_at=row[4],
                last_reviewed=row[5],
                next_review=row[6],
                review_count=row[7],
                success_count=row[8],
                failure_count=row[9],
                memory_strength=MemoryStrength(row[10]),
                ease_factor=row[11],
                interval=row[12]
            )
    
    def add_learning_item(self, item_id: str, content: str, concept: str, 
                        difficulty: Difficulty) -> bool:
        """Add new learning item to the system"""
        if item_id in self.items:
            return False
        
        current_time = time.time()
        item = LearningItem(
            id=item_id,
            content=content,
            concept=concept,
            difficulty=difficulty,
            created_at=current_time,
            last_reviewed=current_time,
            next_review=current_time + (24 * 3600),  # 1 day from now
            review_count=0,
            success_count=0,
            failure_count=0,
            memory_strength=MemoryStrength.WEAK,
            ease_factor=2.5,
            interval=1.0
        )
        
        self.items[item_id] = item
        self._save_item(item)
        return True
    
    def _save_item(self, item: LearningItem):
        """Save learning item to database"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO learning_items 
            (id, content, concept, difficulty, created_at, last_reviewed, 
             next_review, review_count, success_count, failure_count, 
             memory_strength, ease_factor, interval)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            item.id, item.content, item.concept, item.difficulty.value,
            item.created_at, item.last_reviewed, item.next_review,
            item.review_count, item.success_count, item.failure_count,
            item.memory_strength.value, item.ease_factor, item.interval
        ))
        self.conn.commit()
    
    def calculate_forgetting_curve(self, item: LearningItem) -> float:
        """Calculate forgetting curve based on memory strength and time"""
        time_elapsed = (time.time() - item.last_reviewed) / (24 * 3600)  # days
        
        # Ebbinghaus forgetting curve: R = e^(-t/S)
        # where t is time and S is memory strength
        memory_strength_value = item.memory_strength.value
        retention_rate = math.exp(-time_elapsed / memory_strength_value)
        
        return retention_rate
    
    def calculate_next_interval(self, item: LearningItem, performance: float, 
                            response_time: float) -> float:
        """SM-2 algorithm for calculating next review interval"""
        
        # Performance factor (0.0 to 1.0)
        if performance >= 0.8:  # Good performance
            item.ease_factor = max(1.3, item.ease_factor + 0.1 - (1 - performance) * (2 - response_time * 0.1))
        else:  # Poor performance
            item.ease_factor = max(1.3, item.ease_factor - 0.2 + (1 - performance) * 2)
        
        # Calculate next interval based on SM-2
        if performance < 0.3:  # Failed completely
            next_interval = 1.0
        elif item.review_count == 0:
            next_interval = 1.0
        elif item.review_count == 1:
            next_interval = 6.0
        else:
            # SM-2 formula: I(n) = I(n-1) * EF
            next_interval = item.interval * item.ease_factor
        
        # Apply difficulty factor
        difficulty_modifier = 1.0 + (item.difficulty.value - 3) * 0.1
        next_interval *= difficulty_modifier
        
        # Cap maximum interval
        max_interval = 365.0  # 1 year
        next_interval = min(next_interval, max_interval)
        
        item.interval = next_interval
        return next_interval
    
    def update_memory_strength(self, item: LearningItem, performance: float):
        """Update memory strength based on performance"""
        if performance >= 0.9:
            # Excellent performance - strengthen memory
            if item.memory_strength == MemoryStrength.WEAK:
                item.memory_strength = MemoryStrength.LEARNING
            elif item.memory_strength == MemoryStrength.LEARNING:
                item.memory_strength = MemoryStrength.YOUNG
            elif item.memory_strength == MemoryStrength.YOUNG:
                item.memory_strength = MemoryStrength.MATURE
            elif item.memory_strength == MemoryStrength.MATURE:
                item.memory_strength = MemoryStrength.STRONG
        elif performance >= 0.6:
            # Good performance - gradual strengthening
            strength_values = list(MemoryStrength)
            current_index = strength_values.index(item.memory_strength)
            if current_index < len(strength_values) - 1:
                item.memory_strength = strength_values[current_index + 1]
        else:
            # Poor performance - weaken memory
            strength_values = list(MemoryStrength)
            current_index = strength_values.index(item.memory_strength)
            if current_index > 0:
                item.memory_strength = strength_values[current_index - 1]
    
    def review_item(self, item_id: str, performance: float, 
                  response_time: float) -> Dict[str, any]:
        """Process a review of a learning item"""
        if item_id not in self.items:
            return {"success": False, "error": "Item not found"}
        
        item = self.items[item_id]
        current_time = time.time()
        
        # Update statistics
        item.review_count += 1
        if performance >= 0.6:
            item.success_count += 1
        else:
            item.failure_count += 1
        
        # Calculate next interval and update memory strength
        next_interval = self.calculate_next_interval(item, performance, response_time)
        self.update_memory_strength(item, performance)
        
        # Update timestamps
        item.last_reviewed = current_time
        item.next_review = current_time + (next_interval * 24 * 3600)
        
        # Save to database
        self._save_item(item)
        
        # Calculate forgetting probability
        forgetting_probability = 1.0 - self.calculate_forgetting_curve(item)
        
        return {
            "success": True,
            "item_id": item_id,
            "performance": performance,
            "response_time": response_time,
            "next_interval_days": next_interval,
            "next_review_date": datetime.fromtimestamp(item.next_review).isoformat(),
            "memory_strength": item.memory_strength.name,
            "ease_factor": item.ease_factor,
            "forgetting_probability": forgetting_probability,
            "review_count": item.review_count,
            "success_rate": item.success_count / item.review_count
        }
    
    def get_due_reviews(self, limit: int = 50) -> List[LearningItem]:
        """Get items that are due for review"""
        current_time = time.time()
        due_items = []
        
        for item in self.items.values():
            if current_time >= item.next_review:
                due_items.append(item)
        
        # Sort by priority (difficulty and memory strength)
        due_items.sort(key=lambda x: (
            x.difficulty.value, 
            x.memory_strength.value
        ))
        
        return due_items[:limit]
    
    def get_review_schedule(self, days_ahead: int = 7) -> Dict[str, List[LearningItem]]:
        """Get review schedule for next N days"""
        current_time = time.time()
        schedule = {}
        
        for day_offset in range(days_ahead + 1):
            target_date = current_time + (day_offset * 24 * 3600)
            date_key = (datetime.now() + timedelta(days=day_offset)).strftime("%Y-%m-%d")
            
            items_for_day = []
            for item in self.items.values():
                if item.next_review <= target_date < (target_date + 24 * 3600):
                    items_for_day.append(item)
            
            if items_for_day:
                # Sort by priority
                items_for_day.sort(key=lambda x: (
                    x.difficulty.value,
                    x.memory_strength.value
                ))
                schedule[date_key] = items_for_day
        
        return schedule
    
    def get_learning_statistics(self) -> Dict[str, any]:
        """Get comprehensive learning statistics"""
        if not self.items:
            return {"total_items": 0}
        
        current_time = time.time()
        total_items = len(self.items)
        due_now = sum(1 for item in self.items.values() if current_time >= item.next_review)
        
        # Memory strength distribution
        strength_counts = {}
        for strength in MemoryStrength:
            strength_counts[strength.name] = sum(1 for item in self.items.values() 
                                            if item.memory_strength == strength)
        
        # Performance statistics
        total_reviews = sum(item.review_count for item in self.items.values())
        total_successes = sum(item.success_count for item in self.items.values())
        overall_success_rate = total_successes / total_reviews if total_reviews > 0 else 0
        
        # Difficulty distribution
        difficulty_counts = {}
        for difficulty in Difficulty:
            difficulty_counts[difficulty.name] = sum(1 for item in self.items.values() 
                                              if item.difficulty == difficulty)
        
        # Intervals distribution
        intervals = [item.interval for item in self.items.values()]
        avg_interval = statistics.mean(intervals) if intervals else 0
        
        # Streak calculation
        recent_reviews = [item for item in self.items.values() 
                          if (current_time - item.last_reviewed) < (7 * 24 * 3600)]
        
        return {
            "total_items": total_items,
            "due_now": due_now,
            "reviews_completed": total_reviews,
            "overall_success_rate": overall_success_rate,
            "memory_strength_distribution": strength_counts,
            "difficulty_distribution": difficulty_counts,
            "average_interval_days": avg_interval,
            "recent_activity_7_days": len(recent_reviews),
            "learning_items": [asdict(item) for item in list(self.items.values())[:10]]  # Last 10 items
        }
    
    def optimize_review_order(self, items: List[LearningItem]) -> List[LearningItem]:
        """Optimize review order using advanced algorithms"""
        # Multiple factors for optimization:
        # 1. Memory strength (weaker first)
        # 2. Difficulty (easier first to build momentum)
        # 3. Interleaving (mix concepts)
        # 4. Fatigue management (avoid too many hard items in a row)
        
        def priority_score(item: LearningItem) -> float:
            # Lower score = higher priority
            strength_penalty = item.memory_strength.value * 2
            difficulty_penalty = item.difficulty.value * 1.5
            urgency_bonus = max(0, (time.time() - item.next_review) / (24 * 3600)) * 5
            
            return strength_penalty + difficulty_penalty - urgency_bonus
        
        # Sort by optimized priority
        optimized_items = sorted(items, key=priority_score)
        
        # Apply interleaving - don't put same concept items together
        final_order = []
        recent_concepts = []
        
        for item in optimized_items:
            # Skip if same concept was recently reviewed (within last 3 items)
            if item.concept not in recent_concepts[-3:]:
                final_order.append(item)
                recent_concepts.append(item.concept)
            else:
                # Put it at the end
                final_order.append(item)
        
        return final_order
    
    def export_learning_data(self) -> Dict[str, any]:
        """Export all learning data for backup or analysis"""
        return {
            "export_timestamp": datetime.now().isoformat(),
            "total_items": len(self.items),
            "statistics": self.get_learning_statistics(),
            "items": [asdict(item) for item in self.items.values()],
            "review_schedule": self.get_review_schedule(30),
            "system_info": {
                "algorithm": "SM-2 with modifications",
                "forgetting_curve": "Ebbinghaus",
                "database_path": self.db_path
            }
        }
    
    def import_learning_data(self, data: Dict[str, any]) -> bool:
        """Import learning data from backup"""
        try:
            for item_data in data.get("items", []):
                item = LearningItem(**item_data)
                self.items[item.id] = item
                self._save_item(item)
            return True
        except Exception as e:
            print(f"Import error: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

# Advanced Learning Algorithms
class AdaptiveLearningEngine:
    """Advanced learning engine that adapts to individual learning patterns"""
    
    def __init__(self, scheduler: SpacedRepetitionScheduler):
        self.scheduler = scheduler
        self.learning_patterns = {}
        self.performance_history = {}
    
    def analyze_learning_patterns(self, user_id: str):
        """Analyze individual learning patterns and adapt"""
        user_items = [item for item in self.scheduler.items.values() 
                      if item.id.startswith(f"{user_id}_")]
        
        if len(user_items) < 10:
            return None  # Not enough data
        
        # Calculate optimal review times
        review_times = []
        performance_by_time = {}
        
        for item in user_items:
            hour_of_review = datetime.fromtimestamp(item.last_reviewed).hour
            review_times.append(hour_of_review)
        
        # Find best performance times
        for hour in range(24):
            hour_performance = []
            for item in user_items:
                if datetime.fromtimestamp(item.last_reviewed).hour == hour:
                    success_rate = item.success_count / item.review_count
                    hour_performance.append(success_rate)
            
            if hour_performance:
                performance_by_time[hour] = statistics.mean(hour_performance)
        
        return {
            "optimal_review_hours": sorted(performance_by_time.keys(), 
                                        key=lambda x: performance_by_time[x], 
                                        reverse=True)[:3],
            "performance_by_hour": performance_by_time,
            "learning_velocity": self._calculate_learning_velocity(user_items)
        }
    
    def _calculate_learning_velocity(self, items: List[LearningItem]) -> float:
        """Calculate how quickly user is learning new items"""
        if not items:
            return 0
        
        # Calculate items learned per week
        recent_items = [item for item in items 
                        if (time.time() - item.created_at) < (30 * 24 * 3600)]
        return len(recent_items) / 4.0  # 4 weeks in a month
    
    def generate_personalized_schedule(self, user_id: str, 
                                 daily_time_budget: int = 30) -> Dict[str, any]:
        """Generate personalized review schedule based on patterns"""
        patterns = self.analyze_learning_patterns(user_id)
        if not patterns:
            return None
        
        due_items = self.scheduler.get_due_reviews()
        if not due_items:
            return {"message": "No reviews due!"}
        
        # Estimate review time based on difficulty and performance history
        estimated_time = 0
        scheduled_items = []
        
        for item in due_items:
            # Estimate time based on difficulty and past performance
            difficulty_time = {
                Difficulty.VERY_EASY: 30,
                Difficulty.EASY: 45,
                Difficulty.MEDIUM: 60,
                Difficulty.HARD: 90,
                Difficulty.VERY_HARD: 120
            }
            
            performance_factor = 1.0
            if item.review_count > 0:
                success_rate = item.success_count / item.review_count
                performance_factor = 2.0 - success_rate  # Slower for poor performance
            
            estimated_time_per_item = difficulty_time[item.difficulty] * performance_factor
            
            if estimated_time + estimated_time_per_item <= daily_time_budget * 60:
                scheduled_items.append(item)
                estimated_time += estimated_time_per_item
        
        return {
            "scheduled_items": len(scheduled_items),
            "estimated_time_minutes": estimated_time,
            "time_utilization": estimated_time / (daily_time_budget * 60),
            "optimal_hours": patterns["optimal_review_hours"],
            "items": [asdict(item) for item in scheduled_items]
        }

# Example usage and testing
def main():
    """Main function to demonstrate the advanced learning system"""
    print("🧠 SloughGPT Advanced Learning System - Spaced Repetition")
    print("=" * 60)
    
    # Initialize the system
    scheduler = SpacedRepetitionScheduler("slo_learning.db")
    adaptive_engine = AdaptiveLearningEngine(scheduler)
    
    # Add sample learning items
    sample_items = [
        ("concept_1", "Python list comprehension is a concise way to create lists", "python_basics", Difficulty.EASY),
        ("concept_2", "Decorators modify function behavior without changing function code", "python_advanced", Difficulty.MEDIUM),
        ("concept_3", "Machine learning requires feature engineering and model training", "ml_basics", Difficulty.HARD),
        ("concept_4", "FastAPI is a modern Python web framework", "web_development", Difficulty.EASY),
        ("concept_5", "Vector embeddings represent semantic meaning in high-dimensional space", "nlp_concepts", Difficulty.VERY_HARD)
    ]
    
    print("📚 Adding sample learning items...")
    for item_id, content, concept, difficulty in sample_items:
        success = scheduler.add_learning_item(f"user1_{item_id}", content, concept, difficulty)
        if success:
            print(f"✅ Added: {concept}")
        else:
            print(f"❌ Failed to add: {concept}")
    
    # Simulate some reviews
    print("\n🔄 Simulating learning reviews...")
    for i, item_id in enumerate(["user1_concept_1", "user1_concept_2", "user1_concept_3"]):
        performance = 0.6 + (i * 0.15)  # Improving performance
        response_time = 30 + (i * 10)  # Variable response time
        
        result = scheduler.review_item(item_id, performance, response_time)
        print(f"📝 Review {item_id}: Performance {performance:.1f}, Next review in {result['next_interval_days']:.1f} days")
    
    # Get statistics
    print("\n📊 Learning Statistics:")
    stats = scheduler.get_learning_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Get due reviews
    print("\n📅 Upcoming Reviews:")
    due_items = scheduler.get_due_reviews(5)
    for i, item in enumerate(due_items):
        print(f"   {i+1}. {item.concept} (Difficulty: {item.difficulty.name}, Strength: {item.memory_strength.name})")
    
    # Analyze learning patterns
    print("\n🔍 Analyzing Learning Patterns:")
    patterns = adaptive_engine.analyze_learning_patterns("user1")
    if patterns:
        print(f"   Optimal review hours: {patterns['optimal_review_hours']}")
        print(f"   Learning velocity: {patterns['learning_velocity']:.1f} items/month")
    
    # Generate personalized schedule
    print("\n🗓️ Personalized Schedule:")
    schedule = adaptive_engine.generate_personalized_schedule("user1", 30)
    if schedule:
        print(f"   {schedule['scheduled_items']} items scheduled")
        print(f"   Estimated time: {schedule['estimated_time_minutes']} minutes")
    
    print("\n🎯 Advanced Learning System initialized successfully!")
    
    # Export data for backup
    export_data = scheduler.export_learning_data()
    with open("learning_backup.json", "w") as f:
        json.dump(export_data, f, indent=2)
    print("💾 Learning data exported to learning_backup.json")
    
    scheduler.close()
    return True

if __name__ == "__main__":
    success = main()
    print(f"\n{'✅ Success!' if success else '❌ Failed!'}")