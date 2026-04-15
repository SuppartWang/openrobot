"""Layer 5: Task priority scheduler."""

from typing import List, Dict, Any


class TaskScheduler:
    """Maintains a priority queue of pending tasks."""

    def __init__(self):
        self._tasks: List[Dict[str, Any]] = []

    def add(self, task_id: str, description: str, priority: float = 1.0):
        self._tasks.append({
            "task_id": task_id,
            "description": description,
            "priority": priority,
        })
        self._tasks.sort(key=lambda t: t["priority"], reverse=True)

    def next_task(self) -> Dict[str, Any]:
        if not self._tasks:
            raise IndexError("No tasks in scheduler")
        return self._tasks.pop(0)

    def peek(self) -> Dict[str, Any]:
        if not self._tasks:
            raise IndexError("No tasks in scheduler")
        return self._tasks[0]

    def list_tasks(self) -> List[Dict[str, Any]]:
        return self._tasks.copy()
