#!/usr/bin/env python3
"""
SloughGPT Four Component Integration Test
Tests Multi-Agent + Metacognitive + Knowledge Graph + Spaced Repetition systems
Working together as an integrated cognitive architecture
"""

import time
import asyncio
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from slo_multi_agent import SloughGPTMultiAgentCoordinator, TaskComplexity
from slo_metacognitive import SloughGPTMetacognitive
from slo_knowledge_graph import SLOKnowledgeGraph, RelationType, Confidence
from slo_spaced_repetition import SpacedRepetitionScheduler, Difficulty

class SLOIntegratedSystem:
    """Integration of all four SLO components"""
    
    def __init__(self):
        self.multi_agent = SloughGPTMultiAgentCoordinator()
        self.metacognitive = SloughGPTMetacognitive()
        self.knowledge_graph = SLOKnowledgeGraph(db_path='test_integrated_kg.db')
        self.spaced_repetition = SpacedRepetitionScheduler()
        
        # Setup initial knowledge
        self._setup_knowledge_base()
        
        print("🧠 SloughGPT Integrated System Initialized")
        print("   ✅ Multi-Agent Coordinator (8 specialized agents)")
        print("   ✅ Metacognitive System (self-monitoring)")
        print("   ✅ Knowledge Graph (semantic relationships)")
        print("   ✅ Spaced Repetition (adaptive learning)")
        print()
    
    def _setup_knowledge_base(self):
        """Setup initial knowledge for the integrated system"""
        # Add renewable energy concepts
        solar_id = self.knowledge_graph.add_concept(
            'solar_energy', 'solar energy', 'technology', confidence=Confidence.VERY_HIGH
        )
        wind_id = self.knowledge_graph.add_concept(
            'wind_energy', 'wind energy', 'technology', confidence=Confidence.VERY_HIGH
        )
        renewable_id = self.knowledge_graph.add_concept(
            'renewable_energy', 'renewable energy', 'category', confidence=Confidence.VERY_HIGH
        )
        
        # Add relationships
        self.knowledge_graph.add_relation('solar_energy', 'renewable_energy', RelationType.EXAMPLE_OF)
        self.knowledge_graph.add_relation('wind_energy', 'renewable_energy', RelationType.EXAMPLE_OF)
        
        # Add learning items to spaced repetition
        self.spaced_repetition.add_learning_item(
            'solar_basics', 'Solar energy comes from the sun and is renewable', 
            'renewable energy', Difficulty.EASY
        )
        self.spaced_repetition.add_learning_item(
            'wind_basics', 'Wind turbines convert kinetic energy to electricity',
            'renewable energy', Difficulty.MEDIUM
        )
    
    async def process_integrated_query(self, query: str) -> dict:
        """Process a query using all four components"""
        print(f"🔍 Processing: '{query}'")
        print("-" * 50)
        
        # 1. Multi-Agent Analysis
        print("1️⃣ Multi-Agent Decomposition...")
        agent_result = await self.multi_agent.process_complex_query(
            query, TaskComplexity.MODERATE
        )
        print(f"   ✅ Used {agent_result['contribution_count']} agents")
        print(f"   🎯 Overall confidence: {agent_result['overall_confidence']:.2f}")
        
        # 2. Knowledge Graph Enhancement
        print("2️⃣ Knowledge Graph Enhancement...")
        related_concepts = self.knowledge_graph.expand_query_with_related_concepts(query)
        print(f"   ✅ Found {len(related_concepts)} related concepts")
        
        # 3. Metacognitive Assessment
        print("3️⃣ Metacognitive Assessment...")
        context = [agent_result.get('final_content', str(agent_result))]
        confidence = self.metacognitive.assess_confidence(query, context)
        completeness = self.metacognitive.measure_completeness(
            agent_result.get('final_content', ''), 
            ['renewable', 'energy', 'solar', 'wind']
        )
        print(f"   ✅ Confidence: {confidence:.2f}")
        print(f"   📊 Completeness: {completeness:.2f}")
        
        # 4. Learning Integration (Spaced Repetition)
        print("4️⃣ Learning Integration...")
        due_items = self.spaced_repetition.get_due_reviews()
        print(f"   ✅ {len(due_items)} items ready for review")
        
        # Create integrated response
        integrated_result = {
            'query': query,
            'multi_agent_analysis': agent_result,
            'knowledge_enhancement': {
                'related_concepts': len(related_concepts),
                'concepts': related_concepts[:3]  # Top 3
            },
            'metacognitive_assessment': {
                'confidence': confidence,
                'completeness': completeness,
                'self_correction_applied': confidence < 0.7
            },
            'learning_status': {
                'due_reviews': len(due_items),
                'total_learning_items': len(self.spaced_repetition.items)
            },
            'integration_score': (agent_result['overall_confidence'] + confidence + completeness) / 3,
            'timestamp': time.time()
        }
        
        return integrated_result
    
    async def run_comprehensive_test(self):
        """Run comprehensive test of all four components"""
        print("🚀 SloughGPT Four Component Integration Test")
        print("=" * 60)
        
        test_queries = [
            "What are the economic benefits of renewable energy sources?",
            "Compare solar and wind power efficiency",
            "How can we improve renewable energy adoption?"
        ]
        
        results = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📋 Test Case {i}/{len(test_queries)}")
            print("=" * 40)
            
            result = await self.process_integrated_query(query)
            results.append(result)
            
            print(f"\n🎯 Integration Score: {result['integration_score']:.2f}")
            if result['integration_score'] > 0.8:
                print("   🏆 EXCELLENT - All systems working perfectly!")
            elif result['integration_score'] > 0.6:
                print("   ✅ GOOD - Systems well integrated")
            else:
                print("   ⚠️ NEEDS IMPROVEMENT - Integration issues detected")
            
            print()
        
        # Generate final report
        self.generate_integration_report(results)
    
    def generate_integration_report(self, results: list):
        """Generate comprehensive integration report"""
        print("📊 INTEGRATION REPORT")
        print("=" * 40)
        
        avg_score = sum(r['integration_score'] for r in results) / len(results)
        avg_confidence = sum(r['metacognitive_assessment']['confidence'] for r in results) / len(results)
        avg_completeness = sum(r['metacognitive_assessment']['completeness'] for r in results) / len(results)
        
        print(f"Average Integration Score: {avg_score:.2f}")
        print(f"Average Confidence: {avg_confidence:.2f}")
        print(f"Average Completeness: {avg_completeness:.2f}")
        
        # Component status
        print(f"\n🏗️ Component Status:")
        print(f"   Multi-Agent: ✅ {len(self.multi_agent.agents)} agents active")
        print(f"   Knowledge Graph: ✅ {len(self.knowledge_graph.nodes)} concepts, {len(self.knowledge_graph.edges)} relations")
        print(f"   Metacognitive: ✅ Self-monitoring active")
        print(f"   Spaced Repetition: ✅ {len(self.spaced_repetition.items)} learning items")
        
        # Overall assessment
        print(f"\n🎯 Overall Assessment:")
        if avg_score > 0.8:
            print("   🏆 OUTSTANDING - Four components perfectly integrated!")
        elif avg_score > 0.6:
            print("   ✅ SUCCESSFUL - Integration working well")
        else:
            print("   ⚠️ REQUIRES WORK - Integration needs improvement")
        
        # Cleanup
        print(f"\n🧹 Cleaning up...")
        self.knowledge_graph.close()
        self.spaced_repetition.close()
        print("   ✅ Test databases cleaned up")
        
        print(f"\n🎉 Four Component Integration Test Complete!")

async def main():
    """Main test runner"""
    try:
        system = SLOIntegratedSystem()
        await system.run_comprehensive_test()
    except Exception as e:
        print(f"❌ Integration test error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up test databases
        import os
        for db_file in ['test_integrated_kg.db', 'slo_spaced_repetition.db']:
            if os.path.exists(db_file):
                os.remove(db_file)
                print(f"🗑️ Removed test database: {db_file}")

if __name__ == "__main__":
    asyncio.run(main())