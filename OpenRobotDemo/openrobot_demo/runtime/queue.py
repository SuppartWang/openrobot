"""RobotQueue: serializes robot task execution with retry and persistence hooks."""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from openrobot_demo.persistence.db import EpisodeRecorder, RobotDatabase

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class _QueuedTask:
    episode_id: str
    instruction: str
    task_fn: Callable[[str, Optional[EpisodeRecorder]], Any]
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    result: Any = field(default=None)
    error: Optional[str] = field(default=None)


class RobotQueue:
    """
    Serializes robot task execution to avoid resource conflicts.

    Features:
    - Single background worker thread processes tasks in FIFO order.
    - Per-task retry with exponential backoff.
    - Automatic EpisodeRecorder injection so tasks can persist steps/states.
    """

    def __init__(
        self,
        db: Optional[RobotDatabase] = None,
        max_retries: int = 3,
        base_retry_delay: float = 1.0,
    ):
        self.db = db
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        self._tasks: Dict[str, _QueuedTask] = {}
        self._queue: List[str] = []
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown = False
        self._cv = threading.Condition(self._lock)

    def enqueue(
        self,
        instruction: str,
        task_fn: Callable[[str, Optional[EpisodeRecorder]], Any],
        episode_id: Optional[str] = None,
    ) -> str:
        """Add a task to the queue. Returns the episode_id."""
        eid = episode_id or str(uuid.uuid4())

        if self.db:
            self.db.create_episode(instruction, eid)

        task = _QueuedTask(episode_id=eid, instruction=instruction, task_fn=task_fn)
        with self._lock:
            self._tasks[eid] = task
            self._queue.append(eid)
            self._cv.notify()

        logger.info("[RobotQueue] Enqueued task %s: %s", eid, instruction)
        self._ensure_worker()
        return eid

    def get_status(self, episode_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            task = self._tasks.get(episode_id)
            if not task:
                return None
            return {
                "episode_id": task.episode_id,
                "instruction": task.instruction,
                "status": task.status.value,
                "retry_count": task.retry_count,
                "result": task.result,
                "error": task.error,
            }

    def list_tasks(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [self.get_status(eid) for eid in self._queue]

    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """Signal the worker to shut down after finishing current tasks."""
        self._shutdown = True
        with self._lock:
            self._cv.notify()
        if wait and self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker_thread is None or not self._worker_thread.is_alive():
                self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
                self._worker_thread.start()

    def _worker_loop(self) -> None:
        logger.info("[RobotQueue] Worker started")
        while not self._shutdown:
            with self._lock:
                while not self._queue and not self._shutdown:
                    self._cv.wait(timeout=0.5)
                if not self._queue:
                    continue
                episode_id = self._queue.pop(0)
                task = self._tasks[episode_id]
                task.status = TaskStatus.RUNNING

            if self.db:
                self.db.update_episode_status(episode_id, "running")

            logger.info("[RobotQueue] Starting task %s", episode_id)
            recorder = EpisodeRecorder(self.db, episode_id) if self.db else None

            try:
                result = task.task_fn(task.instruction, recorder) if recorder else task.task_fn(task.instruction, None)
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.error = None
                if self.db:
                    self.db.update_episode_status(episode_id, "completed")
                logger.info("[RobotQueue] Task %s completed", episode_id)
            except Exception as exc:
                logger.exception("[RobotQueue] Task %s failed", episode_id)
                task.error = str(exc)
                if task.retry_count < self.max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.RETRYING
                    delay = self.base_retry_delay * (2 ** (task.retry_count - 1))
                    logger.info(
                        "[RobotQueue] Retrying task %s in %.1fs (attempt %d/%d)",
                        episode_id,
                        delay,
                        task.retry_count,
                        self.max_retries,
                    )
                    time.sleep(delay)
                    with self._lock:
                        self._queue.insert(0, episode_id)
                        self._cv.notify()
                else:
                    task.status = TaskStatus.FAILED
                    if self.db:
                        self.db.update_episode_status(episode_id, "failed")
                    logger.error(
                        "[RobotQueue] Task %s exhausted retries and failed", episode_id
                    )

        logger.info("[RobotQueue] Worker stopped")
