#!/usr/bin/env python3
"""
SloughGPT Knowledge Graph Integration
Advanced knowledge representation and semantic search capabilities
"""

import sqlite3
import json
import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, asdict
from enum import Enum
import networkx as nx
import numpy as np
from collections import defaultdict, Counter

class RelationType(Enum):
    """Types of relationships between concepts"""
    IS_A = "is_a"           # "Car is_a Vehicle"
    PART_OF = "part_of"      # "Engine part_of Car"
    RELATED_TO = "related_to"  # "Engine related_to Fuel"
    CAUSES = "causes"        # "Inflammation causes Pain"
    ENABLES = "enables"       # "Electricity enables Lights"
    REQUIRES = "requires"      # "Programming requires Logic"
    SIMILAR_TO = "similar_to"  # "Cat similar_to Dog"
    OPPOSITE_OF = "opposite_of" # "Hot opposite_of Cold"
    EXAMPLE_OF = "example_of"  # "Poodle example_of Dog"

class Confidence(Enum):
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9

@dataclass
class KnowledgeNode:
    """Represents a concept/entity in the knowledge graph"""
    id: str
    text: str
    concept_type: str
    embedding: Optional[List[float]] = None
    confidence: Confidence = Confidence.MEDIUM
    created_at: float = None
    last_accessed: float = None
    access_count: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.last_accessed is None:
            self.last_accessed = time.time()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class KnowledgeEdge:
    """Represents a relationship between two concepts"""
    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 1.0
    confidence: Confidence = Confidence.MEDIUM
    created_at: float = None
    verified: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.metadata is None:
            self.metadata = {}

class SLOKnowledgeGraph:
    """Advanced Knowledge Graph for SloughGPT with semantic capabilities"""
    
    def __init__(self, db_path: str = "slo_knowledge.db"):
        self.db_path = db_path
        self.conn = None
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[Tuple[str, str, RelationType], KnowledgeEdge] = {}
        self.embedding_cache: Dict[str, List[float]] = {}
        self._init_database()
        self._load_knowledge()
    
    def _init_database(self):
        """Initialize SQLite database for persistent knowledge storage"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_nodes (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                concept_type TEXT NOT NULL,
                embedding BLOB,
                confidence REAL NOT NULL,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER NOT NULL,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                weight REAL NOT NULL,
                confidence REAL NOT NULL,
                created_at REAL NOT NULL,
                verified BOOLEAN NOT NULL,
                metadata TEXT,
                PRIMARY KEY (source_id, target_id, relation_type)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_updates (
                update_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                update_type TEXT NOT NULL,
                description TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_nodes_type ON knowledge_nodes(concept_type)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_edges_source ON knowledge_edges(source_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_edges_target ON knowledge_edges(target_id)
        ''')
        
        self.conn.commit()
    
    def _load_knowledge(self):
        """Load knowledge graph from database"""
        cursor = self.conn.cursor()
        
        # Load nodes
        cursor.execute("SELECT * FROM knowledge_nodes")
        rows = cursor.fetchall()
        for row in rows:
            embedding = None
            if row[3]:  # embedding BLOB is not None
                import pickle
                embedding = pickle.loads(row[3])
            
            metadata = {}
            if row[8]:
                metadata = json.loads(row[8])
            
            node = KnowledgeNode(
                id=row[0],
                text=row[1],
                concept_type=row[2],
                embedding=embedding,
                confidence=Confidence(row[4]),
                created_at=row[5],
                last_accessed=row[6],
                access_count=row[7],
                metadata=metadata
            )
            
            self.nodes[node.id] = node
            self.graph.add_node(node.id, **asdict(node))
        
        # Load edges
        cursor.execute("SELECT * FROM knowledge_edges")
        rows = cursor.fetchall()
        for row in rows:
            metadata = {}
            if row[9]:
                metadata = json.loads(row[9])
            
            edge = KnowledgeEdge(
                source_id=row[0],
                target_id=row[1],
                relation_type=RelationType(row[2]),
                weight=row[3],
                confidence=Confidence(row[4]),
                created_at=row[5],
                verified=bool(row[6]),
                metadata=metadata
            )
            
            edge_key = (edge.source_id, edge.target_id, edge.relation_type)
            self.edges[edge_key] = edge
            self.graph.add_edge(
                edge.source_id, 
                edge.target_id,
                relation_type=edge.relation_type.value,
                weight=edge.weight,
                confidence=edge.confidence.value,
                **asdict(edge)
            )
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text (simplified for demo)"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Simplified embedding generation (in production, use actual embedding model)
        words = text.lower().split()
        embedding = []
        
        # Create basic semantic features
        for i in range(128):  # 128-dimensional embedding
            if i < len(words):
                # Hash-based feature extraction
                word_hash = sum(ord(c) for c in words[i]) % 100
                embedding.append(word_hash / 100.0)
            else:
                embedding.append(0.0)
        
        self.embedding_cache[text] = embedding
        return embedding
    
    def add_concept(self, concept_id: str, text: str, concept_type: str,
                   confidence: Confidence = Confidence.MEDIUM,
                   metadata: Dict[str, Any] = None) -> bool:
        """Add new concept to knowledge graph"""
        if concept_id in self.nodes:
            return False
        
        embedding = self.generate_embedding(text)
        
        node = KnowledgeNode(
            id=concept_id,
            text=text,
            concept_type=concept_type,
            embedding=embedding,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        # Add to memory structures
        self.nodes[concept_id] = node
        self.graph.add_node(concept_id, **asdict(node))
        
        # Save to database
        cursor = self.conn.cursor()
        import pickle
        cursor.execute('''
            INSERT INTO knowledge_nodes 
            (id, text, concept_type, embedding, confidence, created_at, last_accessed, access_count, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            concept_id, text, concept_type, pickle.dumps(embedding),
            confidence.value, node.created_at, node.last_accessed, 0,
            json.dumps(metadata or {})
        ))
        
        self.conn.commit()
        self._log_update("add_concept", f"Added concept: {concept_id}")
        return True
    
    def add_relation(self, source_id: str, target_id: str, relation_type: RelationType,
                   weight: float = 1.0, confidence: Confidence = Confidence.MEDIUM,
                   verified: bool = False, metadata: Dict[str, Any] = None) -> bool:
        """Add relationship between concepts"""
        if source_id not in self.nodes or target_id not in self.nodes:
            return False
        
        edge = KnowledgeEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            confidence=confidence,
            verified=verified,
            metadata=metadata or {}
        )
        
        # Add to memory structures
        edge_key = (source_id, target_id, relation_type)
        self.edges[edge_key] = edge
        self.graph.add_edge(source_id, target_id, **asdict(edge))
        
        # Save to database
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO knowledge_edges 
            (source_id, target_id, relation_type, weight, confidence, created_at, verified, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            source_id, target_id, relation_type.value, weight,
            confidence.value, edge.created_at, verified,
            json.dumps(metadata or {})
        ))
        
        self.conn.commit()
        self._log_update("add_relation", f"Added relation: {source_id} -> {target_id} ({relation_type.value})")
        return True
    
    def find_similar_concepts(self, query: str, top_k: int = 10, 
                             similarity_threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find concepts similar to query using embedding similarity"""
        if not self.nodes:
            return []
        
        query_embedding = self.generate_embedding(query)
        similarities = []
        
        for concept_id, node in self.nodes.items():
            if not node.embedding:
                continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, node.embedding)
            
            if similarity >= similarity_threshold:
                similarities.append((concept_id, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def expand_query_with_related_concepts(self, query: str, max_depth: int = 2, 
                                        max_expansions: int = 20) -> List[str]:
        """Expand query using related concepts from knowledge graph"""
        similar_concepts = self.find_similar_concepts(query, top_k=max_expansions)
        
        expanded_concepts = set()
        expansion_queue = [(concept_id, 0) for concept_id, _ in similar_concepts]
        
        while expansion_queue and len(expanded_concepts) < max_expansions:
            concept_id, depth = expansion_queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            if concept_id in expanded_concepts:
                continue
            
            expanded_concepts.add(concept_id)
            
            # Find related concepts through outgoing edges
            for edge_key, edge in self.edges.items():
                if edge.source_id == concept_id and edge.confidence.value >= 0.5:
                    related_id = edge.target_id
                    if related_id not in expanded_concepts:
                        expansion_queue.append((related_id, depth + 1))
        
        return list(expanded_concepts)
    
    def get_concept_path(self, source_id: str, target_id: str) -> List[str]:
        """Find shortest path between two concepts"""
        if source_id not in self.nodes or target_id not in self.nodes:
            return []
        
        try:
            path = nx.shortest_path(self.graph, source_id, target_id, weight='weight')
            return path
        except nx.NetworkXNoPath:
            return []
    
    def find_concept_clusters(self, min_cluster_size: int = 3) -> List[List[str]]:
        """Find clusters of related concepts using community detection"""
        if not self.graph.nodes():
            return []
        
        # Use connected components for basic clustering
        undirected_graph = self.graph.to_undirected()
        communities = list(nx.connected_components(undirected_graph))
        
        # Filter small communities
        clusters = [list(community) for community in communities 
                     if len(community) >= min_cluster_size]
        
        return clusters
    
    def get_knowledge_summary(self, concept_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get comprehensive summary of a concept and its relationships"""
        if concept_id not in self.nodes:
            return {"error": "Concept not found"}
        
        concept = self.nodes[concept_id]
        
        # Update access statistics
        concept.last_accessed = time.time()
        concept.access_count += 1
        
        # Get relationships
        incoming_edges = []
        outgoing_edges = []
        
        for edge_key, edge in self.edges.items():
            if edge.target_id == concept_id:
                incoming_edges.append(edge)
            elif edge.source_id == concept_id:
                outgoing_edges.append(edge)
        
        # Get related concepts
        related_concepts = set()
        for edge in incoming_edges + outgoing_edges:
            related_concepts.add(edge.source_id)
            related_concepts.add(edge.target_id)
        related_concepts.discard(concept_id)
        
        # Get concept cluster
        cluster = []
        for cluster_group in self.find_concept_clusters():
            if concept_id in cluster_group:
                cluster = cluster_group
                break
        
        return {
            "concept_id": concept_id,
            "text": concept.text,
            "concept_type": concept.concept_type,
            "confidence": concept.confidence.value,
            "access_count": concept.access_count,
            "incoming_relations": [
                {
                    "source_id": edge.source_id,
                    "relation_type": edge.relation_type.value,
                    "weight": edge.weight,
                    "confidence": edge.confidence.value
                }
                for edge in incoming_edges
            ],
            "outgoing_relations": [
                {
                    "target_id": edge.target_id,
                    "relation_type": edge.relation_type.value,
                    "weight": edge.weight,
                    "confidence": edge.confidence.value
                }
                for edge in outgoing_edges
            ],
            "related_concepts": list(related_concepts),
            "cluster": cluster,
            "concept_connections": len(incoming_edges) + len(outgoing_edges)
        }
    
    def semantic_search(self, query: str, max_results: int = 20,
                     include_related: bool = True) -> List[Dict[str, Any]]:
        """Perform semantic search with concept expansion"""
        # Find directly similar concepts
        similar_concepts = self.find_similar_concepts(query, top_k=max_results)
        
        # Expand query with related concepts if requested
        if include_related:
            expanded_concept_ids = self.expand_query_with_related_concepts(query)
            similar_concepts.extend([(cid, 0.8) for cid in expanded_concept_ids 
                                  if cid not in [sc[0] for sc in similar_concepts]])
        
        # Remove duplicates and sort
        seen_concepts = set()
        unique_results = []
        for concept_id, similarity in similar_concepts:
            if concept_id not in seen_concepts:
                seen_concepts.add(concept_id)
                if concept_id in self.nodes:
                    node = self.nodes[concept_id]
                    unique_results.append({
                        "concept_id": concept_id,
                        "text": node.text,
                        "concept_type": node.concept_type,
                        "similarity": similarity,
                        "confidence": node.confidence.value,
                        "access_count": node.access_count
                    })
        
        # Sort by similarity and return top results
        unique_results.sort(key=lambda x: x["similarity"], reverse=True)
        return unique_results[:max_results]
    
    def get_concept_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph"""
        total_nodes = len(self.nodes)
        total_edges = len(self.edges)
        
        if total_nodes == 0:
            return {"total_concepts": 0}
        
        # Concept type distribution
        type_counts = Counter(node.concept_type for node in self.nodes.values())
        
        # Relation type distribution
        relation_counts = Counter(edge.relation_type for edge in self.edges.values())
        
        # Confidence distribution
        confidence_counts = Counter(node.confidence for node in self.nodes.values())
        
        # Most accessed concepts
        most_accessed = sorted(self.nodes.values(), 
                             key=lambda x: x.access_count, reverse=True)[:10]
        
        # Network statistics
        if self.graph.nodes():
            density = nx.density(self.graph)
            avg_clustering = nx.average_clustering(self.graph.to_undirected())
            
            if nx.is_connected(self.graph.to_undirected()):
                avg_path_length = nx.average_shortest_path_length(self.graph.to_undirected())
            else:
                avg_path_length = None
        else:
            density = 0
            avg_clustering = 0
            avg_path_length = None
        
        return {
            "total_concepts": total_nodes,
            "total_relations": total_edges,
            "network_density": density,
            "average_clustering": avg_clustering,
            "average_path_length": avg_path_length,
            "concept_types": dict(type_counts),
            "relation_types": {rt.value: count for rt, count in relation_counts.items()},
            "confidence_distribution": {c.value: count for c, count in confidence_counts.items()},
            "most_accessed_concepts": [
                {
                    "id": node.id,
                    "text": node.text,
                    "access_count": node.access_count
                }
                for node in most_accessed
            ]
        }
    
    def export_knowledge_graph(self) -> Dict[str, Any]:
        """Export complete knowledge graph for backup"""
        return {
            "export_timestamp": datetime.now().isoformat(),
            "total_concepts": len(self.nodes),
            "total_relations": len(self.edges),
            "concepts": [asdict(node) for node in self.nodes.values()],
            "relations": [asdict(edge) for edge in self.edges.values()],
            "statistics": self.get_concept_statistics(),
            "graph_info": {
                "nodes": len(self.graph.nodes()),
                "edges": len(self.graph.edges()),
                "directed": True,
                "multigraph": False
            }
        }
    
    def _log_update(self, update_type: str, description: str):
        """Log knowledge graph updates"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO knowledge_updates (update_id, timestamp, update_type, description)
            VALUES (?, ?, ?, ?)
        ''', (
            f"update_{int(time.time() * 1000)}",  # Unique ID based on timestamp
            time.time(),
            update_type,
            description
        ))
        self.conn.commit()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

# Demo and testing
def main():
    """Demonstrate SloughGPT Knowledge Graph System"""
    print("🧠 SloughGPT Knowledge Graph Integration")
    print("=" * 60)
    
    # Initialize knowledge graph
    kg = SLOKnowledgeGraph("slo_knowledge.db")
    
    # Add sample concepts
    concepts = [
        ("programming", "Writing computer code to solve problems", "skill", Confidence.HIGH),
        ("python", "A high-level programming language", "language", Confidence.VERY_HIGH),
        ("machine_learning", "Teaching computers to learn from data", "field", Confidence.HIGH),
        ("algorithm", "Step-by-step procedure for solving problems", "concept", Confidence.HIGH),
        ("data_science", "Extracting insights from data", "field", Confidence.MEDIUM),
        ("neural_network", "Computing system inspired by biological brains", "technique", Confidence.HIGH),
        ("artificial_intelligence", "Machines that can think like humans", "field", Confidence.VERY_HIGH)
    ]
    
    print("📚 Adding concepts to knowledge graph...")
    for concept_id, text, concept_type, confidence in concepts:
        success = kg.add_concept(concept_id, text, concept_type, confidence)
        if success:
            print(f"✅ Added: {concept_id} - {text}")
    
    # Add relationships
    relations = [
        ("python", "programming", RelationType.IS_A, 1.0, Confidence.VERY_HIGH),
        ("machine_learning", "programming", RelationType.REQUIRES, 0.8, Confidence.HIGH),
        ("algorithm", "programming", RelationType.REQUIRES, 0.9, Confidence.HIGH),
        ("neural_network", "machine_learning", RelationType.PART_OF, 0.8, Confidence.HIGH),
        ("data_science", "programming", RelationType.REQUIRES, 0.7, Confidence.MEDIUM),
        ("artificial_intelligence", "machine_learning", RelationType.RELATED_TO, 0.9, Confidence.HIGH)
    ]
    
    print("\n🔗 Adding relationships...")
    for source, target, relation_type, weight, confidence in relations:
        success = kg.add_relation(source, target, relation_type, weight, confidence)
        if success:
            print(f"✅ Relation: {source} -> {target} ({relation_type.value})")
    
    # Semantic search demo
    print("\n🔍 Semantic Search Demo:")
    search_queries = ["python programming", "machine learning", "AI algorithms"]
    
    for query in search_queries:
        print(f"\n📝 Searching for: '{query}'")
        results = kg.semantic_search(query, max_results=5)
        
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result['text']} (Similarity: {result['similarity']:.3f})")
    
    # Concept expansion demo
    print("\n🌐 Concept Expansion Demo:")
    expansion_query = "programming"
    expanded = kg.expand_query_with_related_concepts(expansion_query, max_expansions=10)
    print(f"Original query: '{expansion_query}'")
    print(f"Expanded with {len(expanded)} related concepts")
    
    # Path finding demo
    print("\n🛤️ Knowledge Path Demo:")
    path = kg.get_concept_path("algorithm", "artificial_intelligence")
    if path:
        print("Path from 'algorithm' to 'artificial_intelligence':")
        print(" → ".join(path))
    else:
        print("No path found between 'algorithm' and 'artificial_intelligence'")
    
    # Concept summary demo
    print("\n📊 Concept Summary Demo:")
    summary = kg.get_knowledge_summary("python")
    if "error" not in summary:
        print(f"Concept: {summary['text']}")
        print(f"Type: {summary['concept_type']}")
        print(f"Access count: {summary['access_count']}")
        print(f"Connections: {summary['concept_connections']}")
        print(f"Related concepts: {len(summary['related_concepts'])}")
    
    # Statistics
    print("\n📈 Knowledge Graph Statistics:")
    stats = kg.get_concept_statistics()
    for key, value in stats.items():
        if key not in ["concept_types", "relation_types", "confidence_distribution", "most_accessed_concepts"]:
            print(f"   {key}: {value}")
    
    # Export knowledge
    print("\n💾 Exporting knowledge graph...")
    export_data = kg.export_knowledge_graph()
    with open("knowledge_graph_export.json", "w") as f:
        json.dump(export_data, f, indent=2, default=str)
    print("✅ Knowledge exported to knowledge_graph_export.json")
    
    print("\n🎯 Knowledge Graph System initialized successfully!")
    
    # Test concept clustering
    print("\n🔗 Concept Clustering:")
    clusters = kg.find_concept_clusters(min_cluster_size=2)
    for i, cluster in enumerate(clusters, 1):
        print(f"   Cluster {i}: {len(cluster)} concepts")
        for concept_id in cluster:
            if concept_id in kg.nodes:
                print(f"      - {kg.nodes[concept_id].text}")
    
    kg.close()
    return True

if __name__ == "__main__":
    success = main()
    print(f"\n{'✅ Success!' if success else '❌ Failed!'}")