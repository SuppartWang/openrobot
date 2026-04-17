"""SQLite persistence for robot episodes, steps, and state snapshots."""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "openrobot.db"

_SQL_CREATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    instruction TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    step_id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL,
    step_idx INTEGER NOT NULL,
    thought TEXT,
    action_json TEXT,
    state_summary TEXT,
    wait_time_s REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS step_results (
    result_id TEXT PRIMARY KEY,
    step_id TEXT NOT NULL,
    skill_name TEXT,
    success INTEGER,
    message TEXT,
    result_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (step_id) REFERENCES steps(step_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS state_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL,
    step_idx INTEGER NOT NULL,
    context_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_steps_episode ON steps(episode_id);
CREATE INDEX IF NOT EXISTS idx_steps_idx ON steps(episode_id, step_idx);
CREATE INDEX IF NOT EXISTS idx_results_step ON step_results(step_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_episode ON state_snapshots(episode_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_idx ON state_snapshots(episode_id, step_idx);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RobotDatabase:
    """Thread-safe SQLite wrapper for robot execution history."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.executescript(_SQL_CREATE_SCHEMA)
            conn.commit()

    def create_episode(self, instruction: str, episode_id: Optional[str] = None) -> str:
        eid = episode_id or str(uuid.uuid4())
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO episodes (episode_id, instruction, status, created_at) VALUES (?, ?, ?, ?)",
                (eid, instruction, "pending", _now()),
            )
            conn.commit()
        logger.info("[DB] Created episode %s for instruction: %s", eid, instruction)
        return eid

    def update_episode_status(self, episode_id: str, status: str) -> None:
        completed_at = _now() if status in ("completed", "failed") else None
        with self._lock:
            conn = self._get_conn()
            if completed_at:
                conn.execute(
                    "UPDATE episodes SET status = ?, completed_at = ? WHERE episode_id = ?",
                    (status, completed_at, episode_id),
                )
            else:
                conn.execute(
                    "UPDATE episodes SET status = ? WHERE episode_id = ?",
                    (status, episode_id),
                )
            conn.commit()

    def record_step(
        self,
        episode_id: str,
        step_idx: int,
        thought: str = "",
        action: Optional[Dict[str, Any]] = None,
        state_summary: str = "",
        wait_time_s: float = 0.0,
    ) -> str:
        step_id = str(uuid.uuid4())
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO steps (step_id, episode_id, step_idx, thought, action_json, state_summary, wait_time_s, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    step_id,
                    episode_id,
                    step_idx,
                    thought,
                    json.dumps(_sanitize_for_json(action), ensure_ascii=False) if action else None,
                    state_summary,
                    wait_time_s,
                    _now(),
                ),
            )
            conn.commit()
        return step_id

    def record_step_result(
        self,
        step_id: str,
        skill_name: str,
        success: bool,
        message: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> str:
        result_id = str(uuid.uuid4())
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO step_results (result_id, step_id, skill_name, success, message, result_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    result_id,
                    step_id,
                    skill_name,
                    1 if success else 0,
                    message,
                    json.dumps(_sanitize_for_json(result), ensure_ascii=False) if result else None,
                    _now(),
                ),
            )
            conn.commit()
        return result_id

    def record_state_snapshot(self, episode_id: str, step_idx: int, context: Dict[str, Any]) -> str:
        snapshot_id = str(uuid.uuid4())
        # Sanitize context: numpy arrays and other non-serializable objects
        safe_context = _sanitize_for_json(context)
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO state_snapshots (snapshot_id, episode_id, step_idx, context_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (snapshot_id, episode_id, step_idx, json.dumps(safe_context, ensure_ascii=False), _now()),
            )
            conn.commit()
        return snapshot_id

    def get_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._get_conn()
            row = conn.execute("SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)).fetchone()
            return dict(row) if row else None

    def list_episodes(self, limit: int = 100) -> list:
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM episodes ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_episode_steps(self, episode_id: str) -> list:
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                """
                SELECT s.*, r.skill_name, r.success, r.message, r.result_json
                FROM steps s
                LEFT JOIN step_results r ON s.step_id = r.step_id
                WHERE s.episode_id = ?
                ORDER BY s.step_idx
                """,
                (episode_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_latest_state_snapshot(self, episode_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT * FROM state_snapshots WHERE episode_id = ? ORDER BY step_idx DESC LIMIT 1",
                (episode_id,),
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            d["context"] = json.loads(d["context_json"])
            return d

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None


class EpisodeRecorder:
    """Convenience helper to record an episode's execution history."""

    def __init__(self, db: RobotDatabase, episode_id: str):
        self.db = db
        self.episode_id = episode_id
        self._step_ids: Dict[int, str] = {}

    def record_step(
        self,
        step_idx: int,
        thought: str = "",
        action: Optional[Dict[str, Any]] = None,
        state_summary: str = "",
        wait_time_s: float = 0.0,
    ) -> str:
        step_id = self.db.record_step(
            self.episode_id, step_idx, thought, action, state_summary, wait_time_s
        )
        self._step_ids[step_idx] = step_id
        return step_id

    def record_step_result(
        self,
        step_idx: int,
        skill_name: str,
        success: bool,
        message: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> str:
        step_id = self._step_ids.get(step_idx)
        if not step_id:
            step_id = self.db.record_step(self.episode_id, step_idx)
            self._step_ids[step_idx] = step_id
        return self.db.record_step_result(step_id, skill_name, success, message, result)

    def record_state_snapshot(self, step_idx: int, context: Dict[str, Any]) -> str:
        return self.db.record_state_snapshot(self.episode_id, step_idx, context)

    def finish(self, status: str = "completed") -> None:
        self.db.update_episode_status(self.episode_id, status)


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable objects to safe types."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, bytes):
        return "<bytes>"
    return obj


# Singleton-like default instance for convenience
_default_db: Optional[RobotDatabase] = None


def init_database(db_path: Optional[Path] = None) -> RobotDatabase:
    global _default_db
    _default_db = RobotDatabase(db_path)
    return _default_db


def get_db() -> RobotDatabase:
    global _default_db
    if _default_db is None:
        _default_db = init_database()
    return _default_db
