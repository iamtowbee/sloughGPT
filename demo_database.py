#!/usr/bin/env python3
"""
SloughGPT Database Demo
Demonstrates the new ORM database functionality
"""

import sys
import os

# Add sloughgpt to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sloughgpt import (
    DatabaseManager, 
    LearningExperience, 
    KnowledgeNode, 
    KnowledgeEdge,
    get_db_session,
    db_health_check,
    get_db_stats
)

def demo_database_functionality():
    """Demonstrate key database features"""
    print("🗄️  SloughGPT Database Demo")
    print("=" * 50)
    
    # Initialize database
    print("\n1. Initializing Database...")
    db = DatabaseManager("sqlite:///demo_sloughgpt.db", echo=False)
    db.initialize()
    print("   ✅ Database initialized")
    
    # Health check
    print("\n2. Performing Health Check...")
    health = db.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Tables: {list(health['tables'].keys())}")
    
    # Add some test data
    print("\n3. Adding Test Data...")
    
    with db.get_session() as session:
        # Add learning experience
        experience = LearningExperience(
            prompt="What is machine learning?",
            response="Machine learning is a subset of artificial intelligence...",
            rating=4.5,
            session_id=None,  # Don't set session to avoid relationship issues for now
            metadata_json={"source": "demo", "category": "education"}
        )
        session.add(experience)
        
        # Add knowledge nodes
        node1 = KnowledgeNode(
            node_id="ml_concept",
            content="Machine Learning concept definition",
            node_type="concept",
            importance=5.0,
            metadata_json={"domain": "ai", "complexity": "beginner"}
        )
        session.add(node1)
        
        node2 = KnowledgeNode(
            node_id="ai_concept", 
            content="Artificial Intelligence concept definition",
            node_type="concept",
            importance=4.8,
            metadata_json={"domain": "ai", "complexity": "beginner"}
        )
        session.add(node2)
        
        # Add knowledge edge
        edge = KnowledgeEdge(
            edge_id="ml_to_ai",
            source_node_id="ml_concept",
            target_node_id="ai_concept",
            relationship_type="is_subset_of",
            strength=3.0,
            metadata_json={"confidence": 0.9}
        )
        session.add(edge)
        
        print("   ✅ Test data added")
    
    # Query data back
    print("\n4. Querying Data...")
    
    with db.get_session() as session:
        # Get learning experiences
        experiences = session.query(LearningExperience).all()
        print(f"   Learning Experiences: {len(experiences)}")
        for exp in experiences:
            print(f"     • {exp.prompt[:50]}... (Rating: {exp.rating})")
        
        # Get knowledge nodes
        nodes = session.query(KnowledgeNode).all()
        print(f"   Knowledge Nodes: {len(nodes)}")
        for node in nodes:
            print(f"     • {node.node_id}: {node.content[:30]}...")
        
        # Get knowledge edges
        edges = session.query(KnowledgeEdge).all()
        print(f"   Knowledge Edges: {len(edges)}")
        for edge in edges:
            print(f"     • {edge.source_node_id} -> {edge.target_node_id} ({edge.relationship_type})")
    
    # Get database statistics
    print("\n5. Database Statistics...")
    stats = db.get_statistics()
    print(f"   Database URL: {stats['database_url']}")
    print("   Table counts:")
    for table, info in stats['tables'].items():
        count = info.get('count', 0)
        print(f"     • {table}: {count} records")
    
    # Test cleanup
    print("\n6. Cleaning Up...")
    cleanup_stats = db.cleanup_expired_data(days_old=-1)  # Clean all old data
    print(f"   Cleaned records: {cleanup_stats}")
    
    # Final health check
    print("\n7. Final Health Check...")
    final_health = db.health_check()
    print(f"   Final Status: {final_health['status']}")
    
    # Close database
    db.close()
    print("\n   ✅ Database connection closed")
    
    # Clean up demo file
    if os.path.exists("demo_sloughgpt.db"):
        os.remove("demo_sloughgpt.db")
        print("   🗑️  Demo database file removed")

if __name__ == "__main__":
    try:
        demo_database_functionality()
        print("\n🎉 Database demo completed successfully!")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()