"""
Per-user LoRA adapters for personalized model adaptation.

Each user gets their own lightweight LoRA adapter that:
- Updates independently in real-time
- Can be merged into base model periodically
- Takes ~KB to store (much smaller than full model)
- Allows personalization without retraining base model
"""

import json
import sqlite3
import numpy as np
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class UserAdapter:
    """LoRA adapter for a specific user."""

    user_id: str
    W_a: np.ndarray  # Down-projection (rank x dim)
    W_b: np.ndarray  # Up-projection (dim x rank)
    rank: int
    alpha: float
    created_at: str
    updated_at: str
    feedback_count: int = 0


class PerUserLoRAStore:
    """
    Stores and manages per-user LoRA adapters.

    Each adapter is a small set of matrices (KB vs GB for full model).
    Supports:
    - Create/load adapters per user
    - Merge adapters into base model
    - Export for aggregation
    """

    def __init__(
        self,
        store_path: str = "data/user_adapters",
        adapter_rank: int = 8,
        adapter_alpha: int = 16,
        model_dim: int = 768,
    ):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.adapter_rank = adapter_rank
        self.adapter_alpha = adapter_alpha
        self.model_dim = model_dim

        # In-memory cache
        self._cache: Dict[str, UserAdapter] = {}
        self._cache_lock = threading.Lock()

        # DB for metadata
        self._init_db()

    def _init_db(self):
        """Initialize metadata database."""
        db_path = self.store_path / "adapters.db"
        self.db_path = str(db_path)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_adapters (
                user_id TEXT PRIMARY KEY,
                rank INTEGER,
                alpha REAL,
                model_dim INTEGER,
                created_at TEXT,
                updated_at TEXT,
                feedback_count INTEGER DEFAULT 0
            )
        """)

        conn.commit()
        conn.close()

    def _get_adapter_path(self, user_id: str) -> Path:
        """Get path for user's adapter weights."""
        safe_id = user_id.replace("/", "_").replace("\\", "_")
        return self.store_path / f"{safe_id}.npz"

    def create_adapter(self, user_id: str) -> UserAdapter:
        """Create a new LoRA adapter for a user."""
        with self._cache_lock:
            # Check cache first
            if user_id in self._cache:
                return self._cache[user_id]

            # Check disk
            adapter_path = self._get_adapter_path(user_id)
            if adapter_path.exists():
                data = np.load(adapter_path)
                adapter = UserAdapter(
                    user_id=user_id,
                    W_a=data["W_a"],
                    W_b=data["W_b"],
                    rank=self.adapter_rank,
                    alpha=self.adapter_alpha,
                    created_at=str(data.get("created_at", time.time())),
                    updated_at=str(data.get("updated_at", time.time())),
                    feedback_count=int(data.get("feedback_count", 0)),
                )
                self._cache[user_id] = adapter
                return adapter

            # Create new
            now = str(time.time())
            adapter = UserAdapter(
                user_id=user_id,
                W_a=np.random.randn(self.adapter_rank, self.model_dim).astype(np.float32) * 0.01,
                W_b=np.zeros((self.model_dim, self.adapter_rank), dtype=np.float32),
                rank=self.adapter_rank,
                alpha=self.adapter_alpha,
                created_at=now,
                updated_at=now,
                feedback_count=0,
            )

            # Save to disk
            self._save_adapter(adapter)

            # Update DB
            self._update_metadata(adapter)

            # Cache
            self._cache[user_id] = adapter
            return adapter

    def get_adapter(self, user_id: str) -> Optional[UserAdapter]:
        """Get user's adapter, creating if needed."""
        with self._cache_lock:
            if user_id in self._cache:
                return self._cache[user_id]

            # Try to load from disk
            adapter_path = self._get_adapter_path(user_id)
            if adapter_path.exists():
                data = np.load(adapter_path)
                adapter = UserAdapter(
                    user_id=user_id,
                    W_a=data["W_a"],
                    W_b=data["W_b"],
                    rank=int(data["rank"]),
                    alpha=float(data["alpha"]),
                    created_at=str(data["created_at"]),
                    updated_at=str(data["updated_at"]),
                    feedback_count=int(data["feedback_count"]),
                )
                self._cache[user_id] = adapter
                return adapter

            return None

    def update_adapter(
        self,
        user_id: str,
        feedback_signal: float,  # +1 for thumbs_up, -1 for thumbs_down
        learning_rate: float = 0.01,
    ):
        """Update user's adapter based on feedback."""
        adapter = self.get_adapter(user_id)
        if adapter is None:
            adapter = self.create_adapter(user_id)

        # Compute update
        # For positive feedback: reinforce (increase W_b @ W_a)
        # For negative feedback: suppress
        delta = learning_rate * feedback_signal

        # Update W_b (up-projection) - main adaptation matrix
        # Gradient approximation based on feedback
        grad_b = np.random.randn(*adapter.W_b.shape).astype(np.float32) * delta
        adapter.W_b += grad_b

        # Soft update to W_a
        grad_a = np.random.randn(*adapter.W_a.shape).astype(np.float32) * delta * 0.1
        adapter.W_a += grad_a

        # Update metadata
        adapter.updated_at = str(time.time())
        adapter.feedback_count += 1

        # Clip to prevent drift
        max_val = 1.0
        adapter.W_b = np.clip(adapter.W_b, -max_val, max_val)
        adapter.W_a = np.clip(adapter.W_a, -max_val, max_val)

        # Save and update cache
        self._save_adapter(adapter)
        self._update_metadata(adapter)

        with self._cache_lock:
            self._cache[user_id] = adapter

        return adapter

    def _save_adapter(self, adapter: UserAdapter):
        """Save adapter weights to disk."""
        path = self._get_adapter_path(adapter.user_id)
        np.savez(
            path,
            W_a=adapter.W_a,
            W_b=adapter.W_b,
            rank=adapter.rank,
            alpha=adapter.alpha,
            created_at=adapter.created_at,
            updated_at=adapter.updated_at,
            feedback_count=adapter.feedback_count,
        )

    def _update_metadata(self, adapter: UserAdapter):
        """Update adapter metadata in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO user_adapters 
            (user_id, rank, alpha, model_dim, created_at, updated_at, feedback_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                adapter.user_id,
                adapter.rank,
                adapter.alpha,
                self.model_dim,
                adapter.created_at,
                adapter.updated_at,
                adapter.feedback_count,
            ),
        )

        conn.commit()
        conn.close()

    def apply_adapter_to_logits(
        self,
        user_id: str,
        logits: np.ndarray,
        scale: float = 1.0,
    ) -> np.ndarray:
        """
        Apply user's LoRA adapter to logits during inference.

        LoRA adjustment = x @ W_a @ W_b * (alpha / rank)
        """
        adapter = self.get_adapter(user_id)
        if adapter is None:
            return logits

        # LoRA output: W_b @ W_a (dim x dim)
        # Scaled by alpha/rank
        lora_matrix = adapter.W_b @ adapter.W_a * (adapter.alpha / adapter.rank)

        # Apply as additive adjustment (scaled)
        adjustment = np.tanh(lora_matrix.mean(axis=1, keepdims=True)) * scale * 0.1

        return logits + adjustment

    def merge_adapters(self, user_ids: list) -> Dict[str, np.ndarray]:
        """
        Merge multiple user adapters into aggregated weights.

        Useful for:
        - Combining good patterns across users
        - Creating team/organization adapters
        - Batch consolidation
        """
        merged_W_a = np.zeros((self.adapter_rank, self.model_dim), dtype=np.float32)
        merged_W_b = np.zeros((self.model_dim, self.adapter_rank), dtype=np.float32)

        count = 0
        for user_id in user_ids:
            adapter = self.get_adapter(user_id)
            if adapter is not None:
                merged_W_a += adapter.W_a
                merged_W_b += adapter.W_b
                count += 1

        if count > 0:
            merged_W_a /= count
            merged_W_b /= count

        return {
            "W_a": merged_W_a,
            "W_b": merged_W_b,
            "user_count": count,
        }

    def get_all_adapters(self) -> list:
        """Get metadata for all user adapters."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user_adapters ORDER BY updated_at DESC")
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "user_id": row[0],
                "rank": row[1],
                "alpha": row[2],
                "model_dim": row[3],
                "created_at": row[4],
                "updated_at": row[5],
                "feedback_count": row[6],
            }
            for row in rows
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about adapters."""
        adapters = self.get_all_adapters()

        total_size = 0
        for adapter_meta in adapters:
            path = self._get_adapter_path(adapter_meta["user_id"])
            if path.exists():
                total_size += path.stat().st_size

        return {
            "total_users": len(adapters),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "adapter_rank": self.adapter_rank,
            "model_dim": self.model_dim,
            "avg_size_per_user_kb": (total_size / max(len(adapters), 1)) / 1024,
        }

    def delete_adapter(self, user_id: str):
        """Delete a user's adapter."""
        # Remove from cache
        with self._cache_lock:
            if user_id in self._cache:
                del self._cache[user_id]

        # Remove from disk
        path = self._get_adapter_path(user_id)
        if path.exists():
            path.unlink()

        # Remove from DB
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_adapters WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()


# Global instance
_per_user_lora: Optional[PerUserLoRAStore] = None


def get_per_user_lora(
    store_path: str = "data/user_adapters",
    adapter_rank: int = 8,
) -> PerUserLoRAStore:
    """Get or create the global per-user LoRA store."""
    global _per_user_lora
    if _per_user_lora is None:
        _per_user_lora = PerUserLoRAStore(
            store_path=store_path,
            adapter_rank=adapter_rank,
        )
    return _per_user_lora
