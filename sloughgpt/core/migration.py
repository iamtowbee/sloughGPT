"""
SloughGPT Database Migration Utilities
Tools to migrate from raw SQLite to proper ORM structure
"""

import os
import sqlite3
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .db_manager import get_database_manager
from .database import LearningExperience, LearningSession, KnowledgeNode, KnowledgeEdge, CognitiveState
from .exceptions import DatabaseError, create_error

logger = logging.getLogger(__name__)

class DatabaseMigrator:
    """Handles migration from raw SQLite to ORM structure"""
    
    def __init__(self, legacy_db_path: str, new_db_manager=None):
        """
        Initialize migrator
        
        Args:
            legacy_db_path: Path to existing SQLite database
            new_db_manager: DatabaseManager instance (creates new one if None)
        """
        self.legacy_db_path = legacy_db_path
        self.new_db_manager = new_db_manager or get_database_manager()
        
        if not os.path.exists(legacy_db_path):
            raise create_error(
                DatabaseError,
                f"Legacy database not found: {legacy_db_path}",
                None
            )
    
    def analyze_legacy_database(self) -> Dict[str, Any]:
        """Analyze the legacy database structure and content"""
        analysis = {
            "database_path": self.legacy_db_path,
            "tables": {},
            "total_records": 0
        }
        
        try:
            conn = sqlite3.connect(self.legacy_db_path)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                
                # Get table schema
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [row[1] for row in cursor.fetchall()]
                
                # Get sample data
                sample_data = []
                if count > 0:
                    cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                    sample_rows = cursor.fetchall()
                    sample_data = [dict(zip(columns, row)) for row in sample_rows]
                
                analysis["tables"][table] = {
                    "record_count": count,
                    "columns": columns,
                    "sample_data": sample_data
                }
                
                analysis["total_records"] += count
            
            conn.close()
            logger.info(f"Legacy database analysis complete: {analysis['total_records']} total records")
            
        except sqlite3.Error as e:
            raise create_error(
                DatabaseError,
                f"Failed to analyze legacy database: {str(e)}",
                None,
                cause=e,
                context={"database_path": self.legacy_db_path}
            )
        
        return analysis
    
    def migrate_learning_data(self) -> Dict[str, int]:
        """Migrate learning experiences from legacy database"""
        migration_stats = {"experiences": 0, "sessions": 0}
        
        try:
            conn = sqlite3.connect(self.legacy_db_path)
            cursor = conn.cursor()
            
            # Check if learning tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('learning_experiences', 'experiences');")
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            if not existing_tables:
                logger.info("No learning tables found in legacy database")
                return migration_stats
            
            with self.new_db_manager.get_session() as session:
                for table_name in existing_tables:
                    cursor.execute(f"SELECT * FROM {table_name}")
                    columns = [description[0] for description in cursor.description]
                    rows = cursor.fetchall()
                    
                    for row in rows:
                        data = dict(zip(columns, row))
                        
                        # Map legacy data to new schema
                        if table_name == 'learning_experiences' or table_name == 'experiences':
                            experience = LearningExperience(
                                id=data.get('id'),
                                prompt=data.get('prompt', ''),
                                response=data.get('response', ''),
                                rating=float(data.get('rating', 0.0)),
                                timestamp=self._parse_timestamp(data.get('timestamp')),
                                metadata_json=data.get('metadata') or {},
                                model_version=data.get('model_version', '1.0'),
                                session_id=data.get('session_id')
                            )
                            session.add(experience)
                            migration_stats["experiences"] += 1
                
                session.commit()
            
            conn.close()
            logger.info(f"Learning data migration complete: {migration_stats}")
            
        except Exception as e:
            raise create_error(
                DatabaseError,
                f"Learning data migration failed: {str(e)}",
                None,
                cause=e
            )
        
        return migration_stats
    
    def migrate_knowledge_data(self) -> Dict[str, int]:
        """Migrate knowledge graph data from legacy database"""
        migration_stats = {"nodes": 0, "edges": 0}
        
        try:
            conn = sqlite3.connect(self.legacy_db_path)
            cursor = conn.cursor()
            
            # Check knowledge tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('knowledge_nodes', 'knowledge_edges');")
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            if not existing_tables:
                logger.info("No knowledge tables found in legacy database")
                return migration_stats
            
            with self.new_db_manager.get_session() as session:
                # Migrate nodes first
                if 'knowledge_nodes' in existing_tables:
                    cursor.execute("SELECT * FROM knowledge_nodes")
                    columns = [description[0] for description in cursor.description]
                    rows = cursor.fetchall()
                    
                    for row in rows:
                        data = dict(zip(columns, row))
                        
                        node = KnowledgeNode(
                            node_id=data.get('node_id', f"node_{data.get('id')}"),
                            content=data.get('content', ''),
                            node_type=data.get('node_type', 'concept'),
                            importance=float(data.get('importance', 1.0)),
                            last_accessed=self._parse_timestamp(data.get('last_accessed')),
                            access_count=data.get('access_count', 0),
                            embedding=data.get('embedding'),
                            metadata_json=data.get('metadata') or {}
                        )
                        session.add(node)
                        migration_stats["nodes"] += 1
                
                # Migrate edges
                if 'knowledge_edges' in existing_tables:
                    cursor.execute("SELECT * FROM knowledge_edges")
                    columns = [description[0] for description in cursor.description]
                    rows = cursor.fetchall()
                    
                    for row in rows:
                        data = dict(zip(columns, row))
                        
                        edge = KnowledgeEdge(
                            edge_id=data.get('edge_id', f"edge_{data.get('id')}"),
                            source_node_id=data.get('source_node_id', ''),
                            target_node_id=data.get('target_node_id', ''),
                            relationship_type=data.get('relationship_type', 'related_to'),
                            strength=float(data.get('strength', 1.0)),
                            last_activation=self._parse_timestamp(data.get('last_activation')),
                            activation_count=data.get('activation_count', 0),
                            metadata_json=data.get('metadata') or {}
                        )
                        session.add(edge)
                        migration_stats["edges"] += 1
                
                session.commit()
            
            conn.close()
            logger.info(f"Knowledge data migration complete: {migration_stats}")
            
        except Exception as e:
            raise create_error(
                DatabaseError,
                f"Knowledge data migration failed: {str(e)}",
                None,
                cause=e
            )
        
        return migration_stats
    
    def migrate_cognitive_data(self) -> Dict[str, int]:
        """Migrate cognitive state data from legacy database"""
        migration_stats = {"states": 0}
        
        try:
            conn = sqlite3.connect(self.legacy_db_path)
            cursor = conn.cursor()
            
            # Check cognitive tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('cognitive_states', 'working_memory');")
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            if not existing_tables:
                logger.info("No cognitive tables found in legacy database")
                return migration_stats
            
            with self.new_db_manager.get_session() as session:
                for table_name in existing_tables:
                    cursor.execute(f"SELECT * FROM {table_name}")
                    columns = [description[0] for description in cursor.description]
                    rows = cursor.fetchall()
                    
                    for row in rows:
                        data = dict(zip(columns, row))
                        
                        state = CognitiveState(
                            state_id=data.get('state_id', f"state_{data.get('id')}"),
                            state_type=data.get('state_type', 'working_memory'),
                            content=data.get('content', ''),
                            priority=float(data.get('priority', 1.0)),
                            creation_time=self._parse_timestamp(data.get('creation_time')),
                            last_accessed=self._parse_timestamp(data.get('last_accessed')),
                            access_count=data.get('access_count', 0),
                            expiration_time=self._parse_timestamp(data.get('expiration_time')),
                            metadata_json=data.get('metadata') or {}
                        )
                        session.add(state)
                        migration_stats["states"] += 1
                
                session.commit()
            
            conn.close()
            logger.info(f"Cognitive data migration complete: {migration_stats}")
            
        except Exception as e:
            raise create_error(
                DatabaseError,
                f"Cognitive data migration failed: {str(e)}",
                None,
                cause=e
            )
        
        return migration_stats
    
    def run_full_migration(self) -> Dict[str, Any]:
        """Run complete migration from legacy database"""
        migration_report = {
            "legacy_database": self.legacy_db_path,
            "migration_time": datetime.utcnow().isoformat(),
            "steps_completed": [],
            "migrated_records": {},
            "errors": []
        }
        
        try:
            logger.info("Starting full database migration...")
            
            # Step 1: Analyze legacy database
            analysis = self.analyze_legacy_database()
            migration_report["steps_completed"].append("legacy_analysis")
            migration_report["legacy_analysis"] = analysis
            
            # Step 2: Migrate learning data
            learning_stats = self.migrate_learning_data()
            migration_report["steps_completed"].append("learning_migration")
            migration_report["migrated_records"]["learning"] = learning_stats
            
            # Step 3: Migrate knowledge data
            knowledge_stats = self.migrate_knowledge_data()
            migration_report["steps_completed"].append("knowledge_migration")
            migration_report["migrated_records"]["knowledge"] = knowledge_stats
            
            # Step 4: Migrate cognitive data
            cognitive_stats = self.migrate_cognitive_data()
            migration_report["steps_completed"].append("cognitive_migration")
            migration_report["migrated_records"]["cognitive"] = cognitive_stats
            
            # Step 5: Verify migration
            verification = self.verify_migration()
            migration_report["steps_completed"].append("verification")
            migration_report["verification"] = verification
            
            logger.info("Database migration completed successfully")
            
        except Exception as e:
            error_msg = f"Migration failed: {str(e)}"
            logger.error(error_msg)
            migration_report["errors"].append(error_msg)
            
        return migration_report
    
    def verify_migration(self) -> Dict[str, Any]:
        """Verify that migration was successful"""
        verification = {
            "status": "verified",
            "checks": {},
            "issues": []
        }
        
        try:
            # Get new database statistics
            new_stats = self.new_db_manager.get_statistics()
            
            # Check each table has expected data
            expected_tables = ['learning_experiences', 'knowledge_nodes', 'knowledge_edges', 'cognitive_states']
            
            for table in expected_tables:
                if table in new_stats["tables"]:
                    count = new_stats["tables"][table]["count"]
                    verification["checks"][table] = {
                        "record_count": count,
                        "status": "ok" if count > 0 else "empty"
                    }
                    
                    if count == 0:
                        verification["issues"].append(f"Table {table} is empty after migration")
                else:
                    verification["checks"][table] = {"status": "missing"}
                    verification["issues"].append(f"Table {table} missing from new database")
            
            if verification["issues"]:
                verification["status"] = "issues_found"
            
        except Exception as e:
            verification["status"] = "verification_failed"
            verification["issues"].append(f"Verification error: {str(e)}")
        
        return verification
    
    def _parse_timestamp(self, timestamp) -> Optional[datetime]:
        """Parse timestamp from various legacy formats"""
        if timestamp is None:
            return None
        
        if isinstance(timestamp, datetime):
            return timestamp
        
        if isinstance(timestamp, str):
            try:
                # Try ISO format first
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                try:
                    # Try common formats
                    return datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    logger.warning(f"Could not parse timestamp: {timestamp}")
                    return None
        
        if isinstance(timestamp, (int, float)):
            # Assume Unix timestamp
            return datetime.fromtimestamp(timestamp)
        
        return None

def migrate_legacy_database(legacy_db_path: str, backup: bool = True) -> Dict[str, Any]:
    """
    Convenience function to migrate a legacy database
    
    Args:
        legacy_db_path: Path to legacy SQLite database
        backup: Whether to backup the legacy database first
        
    Returns:
        Migration report
    """
    if backup and os.path.exists(legacy_db_path):
        backup_path = f"{legacy_db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        import shutil
        shutil.copy2(legacy_db_path, backup_path)
        logger.info(f"Legacy database backed up to: {backup_path}")
    
    migrator = DatabaseMigrator(legacy_db_path)
    return migrator.run_full_migration()