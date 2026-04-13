"""
Automated Feedback Workflow Manager.

Orchestrates the complete feedback → training pipeline:
- Records feedback and updates adapters automatically
- Scheduled aggregation and pruning
- Periodic training data export
- Health monitoring and stats
"""

import threading
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .database import FeedbackDB, get_feedback_db
from .meta_weights import MetaWeightManager, get_meta_weight_manager
from .online_train import OnlineLoRAUpdater, get_online_lora_updater
from .per_user_lora import PerUserLoRAStore, get_per_user_lora
from .training import FeedbackTrainer


@dataclass
class WorkflowConfig:
    """Configuration for automated feedback workflow."""

    aggregate_interval_minutes: int = 60
    prune_interval_minutes: int = 120
    export_interval_hours: int = 24
    health_check_interval_seconds: int = 30

    auto_aggregate_threshold: int = 50
    auto_prune_threshold: int = 100
    min_feedback_for_aggregation: int = 3

    export_format: str = "dpo"
    export_path: str = "data/training_exports"


class FeedbackWorkflowManager:
    """
    Manages the complete automated feedback workflow.

    Runs scheduled tasks for:
    - Periodic aggregation of user adapters
    - Pruning of low-quality adapters
    - Exporting training data
    - Health monitoring
    """

    def __init__(
        self,
        config: WorkflowConfig = None,
        feedback_db: FeedbackDB = None,
        meta_manager: MetaWeightManager = None,
        lora_store: PerUserLoRAStore = None,
        lora_updater: OnlineLoRAUpdater = None,
    ):
        self.config = config or WorkflowConfig()

        self.db = feedback_db or get_feedback_db()
        self.meta_manager = meta_manager or get_meta_weight_manager()

        if lora_store:
            self.lora_store = lora_store
        else:
            self.lora_store = get_per_user_lora()
            self.lora_store.auto_aggregate_threshold = self.config.auto_aggregate_threshold
            self.lora_store.auto_prune_threshold = self.config.auto_prune_threshold
            self.lora_store.min_feedback_for_aggregation = self.config.min_feedback_for_aggregation

        self.lora_updater = lora_updater or get_online_lora_updater()

        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self._health_thread: Optional[threading.Thread] = None
        self._last_aggregate_time: float = 0
        self._last_prune_time: float = 0
        self._last_export_time: float = 0
        self._last_health_check: float = 0

        self._stats = {
            "workflow_runs": 0,
            "aggregations_performed": 0,
            "prunes_performed": 0,
            "exports_performed": 0,
            "feedback_recorded": 0,
            "start_time": None,
        }

    def record_feedback(
        self,
        user_message: str,
        assistant_response: str,
        rating: str,
        conversation_id: str = None,
        quality_score: float = None,
        user_id: str = "default",
    ) -> str:
        """
        Record feedback and trigger automatic updates.

        This is the main entry point - records feedback and
        automatically updates all learning systems.
        """
        feedback_id = self.meta_manager.record_feedback(
            user_message=user_message,
            assistant_response=assistant_response,
            rating=rating,
            conversation_id=conversation_id,
            quality_score=quality_score,
            user_id=user_id,
        )

        self.lora_updater.add_feedback(
            prompt=user_message,
            response=assistant_response,
            rating=rating,
            quality_score=quality_score,
        )

        self.lora_store.update_adapter(
            user_id=user_id,
            feedback_signal=1.0 if rating == "thumbs_up" else -1.0,
        )

        self._stats["feedback_recorded"] += 1

        return feedback_id

    def run_scheduled_tasks(self):
        """Run any scheduled tasks that are due."""
        now = time.time()

        if now - self._last_aggregate_time > self.config.aggregate_interval_minutes * 60:
            self._do_aggregate()
            self._last_aggregate_time = now

        if now - self._last_prune_time > self.config.prune_interval_minutes * 60:
            self._do_prune()
            self._last_prune_time = now

        if now - self._last_export_time > self.config.export_interval_hours * 3600:
            self._do_export()
            self._last_export_time = now

    def _do_aggregate(self):
        """Perform aggregation task."""
        try:
            quality_adapters = self.lora_store.get_quality_adapters(
                min_feedback_count=self.config.min_feedback_for_aggregation
            )
            if len(quality_adapters) >= self.config.auto_aggregate_threshold:
                result = self.lora_store.aggregate_best_adapters(
                    top_k=min(20, len(quality_adapters)),
                    min_feedback_count=self.config.min_feedback_for_aggregation,
                    output_name=f"scheduled_{int(time.time())}",
                )
                if "error" not in result:
                    self._stats["aggregations_performed"] += 1
                    print(f"[Workflow] Aggregated {result.get('user_count', 0)} adapters")
        except Exception as e:
            print(f"[Workflow] Aggregate error: {e}")

    def _do_prune(self):
        """Perform pruning task."""
        try:
            deleted = self.lora_store.prune_low_quality(
                min_feedback_count=1,
                max_age_days=7,
            )
            if deleted:
                self._stats["prunes_performed"] += 1
                print(f"[Workflow] Pruned {len(deleted)} adapters")
        except Exception as e:
            print(f"[Workflow] Prune error: {e}")

    def _do_export(self):
        """Perform training data export task."""
        try:
            from pathlib import Path

            export_path = Path(self.config.export_path)
            export_path.mkdir(parents=True, exist_ok=True)

            timestamp = int(time.time())
            filepath = export_path / f"feedback_export_{timestamp}.jsonl"

            self.db.export_feedback_jsonl(str(filepath))
            self._stats["exports_performed"] += 1
            print(f"[Workflow] Exported training data to {filepath}")
        except Exception as e:
            print(f"[Workflow] Export error: {e}")

    def _health_check(self):
        """Perform health check and run scheduled tasks."""
        try:
            self.run_scheduled_tasks()

            self._stats["workflow_runs"] += 1
            self._last_health_check = time.time()
        except Exception as e:
            print(f"[Workflow] Health check error: {e}")

    def start(self):
        """Start the automated workflow in background threads."""
        if self._running:
            return

        self._running = True
        self._stats["start_time"] = time.time()

        def scheduler_loop():
            while self._running:
                self._health_check()
                time.sleep(self.config.health_check_interval_seconds)

        self._scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self._scheduler_thread.start()

        print("[Workflow] Started automated feedback workflow")

    def stop(self):
        """Stop the automated workflow."""
        self._running = False
        print("[Workflow] Stopped automated feedback workflow")

    def get_status(self) -> Dict[str, Any]:
        """Get current workflow status and statistics."""
        return {
            "running": self._running,
            "stats": self._stats.copy(),
            "config": {
                "aggregate_interval_minutes": self.config.aggregate_interval_minutes,
                "prune_interval_minutes": self.config.prune_interval_minutes,
                "export_interval_hours": self.config.export_interval_hours,
                "health_check_interval_seconds": self.config.health_check_interval_seconds,
            },
            "last_runs": {
                "aggregate": self._last_aggregate_time,
                "prune": self._last_prune_time,
                "export": self._last_export_time,
                "health_check": self._last_health_check,
            },
            "systems": {
                "feedback_db": self.db.get_stats() if hasattr(self.db, "get_stats") else {},
                "meta_weights": self.meta_manager.get_stats()
                if hasattr(self.meta_manager, "get_stats")
                else {},
                "lora_store": self.lora_store.get_stats(),
                "lora_updater": self.lora_updater.get_stats()
                if hasattr(self.lora_updater, "get_stats")
                else {},
            },
        }

    def trigger_aggregate(self) -> Dict[str, Any]:
        """Manually trigger aggregation."""
        self._do_aggregate()
        return {"status": "aggregated", "timestamp": time.time()}

    def trigger_prune(self) -> Dict[str, Any]:
        """Manually trigger pruning."""
        self._do_prune()
        return {"status": "pruned", "timestamp": time.time()}

    def trigger_export(self) -> Dict[str, Any]:
        """Manually trigger export."""
        self._do_export()
        return {"status": "exported", "timestamp": time.time()}


_workflow_manager: Optional[FeedbackWorkflowManager] = None


def get_feedback_workflow(
    config: WorkflowConfig = None,
) -> FeedbackWorkflowManager:
    """Get or create the global feedback workflow manager."""
    global _workflow_manager
    if _workflow_manager is None:
        _workflow_manager = FeedbackWorkflowManager(config=config)
    return _workflow_manager
