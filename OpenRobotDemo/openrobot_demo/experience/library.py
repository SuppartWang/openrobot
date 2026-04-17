"""ExperienceLibrary: SQLite-backed persistent storage for robot experiences."""

import json
import logging
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from openrobot_demo.experience.schema import Experience

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "experience.db"

_SQL_CREATE = """
CREATE TABLE IF NOT EXISTS experiences (
    experience_id TEXT PRIMARY KEY,
    data_json TEXT NOT NULL,
    task_intent TEXT,
    target_object_type TEXT,
    action_type TEXT,
    gripper_config TEXT,
    arm_count INTEGER,
    success INTEGER,
    use_count INTEGER DEFAULT 0,
    created_at REAL,
    last_used REAL
);
CREATE INDEX IF NOT EXISTS idx_task ON experiences(task_intent);
CREATE INDEX IF NOT EXISTS idx_object ON experiences(target_object_type);
CREATE INDEX IF NOT EXISTS idx_action ON experiences(action_type);
CREATE INDEX IF NOT EXISTS idx_success ON experiences(success);
"""


class ExperienceLibrary:
    """Thread-safe SQLite library for experience CRUD."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        return self._conn

    def _init_db(self):
        with self._lock:
            self._get_conn().executescript(_SQL_CREATE)
            self._get_conn().commit()

    def add(self, exp: Experience) -> str:
        """Add a new experience. Returns its id."""
        d = exp.to_dict()
        with self._lock:
            self._get_conn().execute(
                """
                INSERT OR REPLACE INTO experiences
                (experience_id, data_json, task_intent, target_object_type,
                 action_type, gripper_config, arm_count, success, use_count,
                 created_at, last_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exp.experience_id,
                    json.dumps(d, ensure_ascii=False),
                    exp.task_intent,
                    exp.target_object_type,
                    exp.action_type,
                    exp.gripper_config.value,
                    exp.arm_count,
                    1 if exp.success else 0,
                    exp.use_count,
                    exp.created_at,
                    exp.last_used,
                ),
            )
            self._get_conn().commit()
        logger.info("[ExperienceLibrary] Added %s (%s | %s)", exp.experience_id, exp.task_intent, exp.action_type)
        return exp.experience_id

    def get(self, experience_id: str) -> Optional[Experience]:
        with self._lock:
            row = (
                self._get_conn()
                .execute("SELECT data_json FROM experiences WHERE experience_id = ?", (experience_id,))
                .fetchone()
            )
        if not row:
            return None
        return Experience.from_dict(json.loads(row[0]))

    def list_all(self, limit: int = 1000) -> List[Experience]:
        with self._lock:
            rows = (
                self._get_conn()
                .execute("SELECT data_json FROM experiences ORDER BY created_at DESC LIMIT ?", (limit,))
                .fetchall()
            )
        return [Experience.from_dict(json.loads(r[0])) for r in rows]

    def query(
        self,
        task_intent: Optional[str] = None,
        target_object_type: Optional[str] = None,
        action_type: Optional[str] = None,
        gripper_config: Optional[str] = None,
        success_only: bool = True,
        limit: int = 20,
    ) -> List[Experience]:
        """Structured query with optional filters."""
        conditions: List[str] = []
        params: List = []
        if task_intent:
            conditions.append("task_intent LIKE ?")
            params.append(f"%{task_intent}%")
        if target_object_type:
            conditions.append("target_object_type LIKE ?")
            params.append(f"%{target_object_type}%")
        if action_type:
            conditions.append("action_type = ?")
            params.append(action_type)
        if gripper_config:
            conditions.append("gripper_config = ?")
            params.append(gripper_config)
        if success_only:
            conditions.append("success = 1")

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT data_json FROM experiences WHERE {where} ORDER BY use_count DESC, created_at DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            rows = self._get_conn().execute(sql, params).fetchall()
        return [Experience.from_dict(json.loads(r[0])) for r in rows]

    def increment_use(self, experience_id: str):
        """Bump use_count and update last_used timestamp."""
        with self._lock:
            self._get_conn().execute(
                """
                UPDATE experiences
                SET use_count = use_count + 1, last_used = ?
                WHERE experience_id = ?
                """,
                (time.time(), experience_id),
            )
            self._get_conn().commit()

    def close(self):
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None


import time
