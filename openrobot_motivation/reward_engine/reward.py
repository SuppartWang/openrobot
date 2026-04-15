"""Layer 5: Motivation and reward computation."""

from typing import Dict, Any


class RewardEngine:
    """Computes intrinsic and extrinsic rewards for task execution."""

    def __init__(self):
        self.task_rewards: Dict[str, float] = {}

    def set_task_reward(self, task_name: str, reward: float):
        self.task_rewards[task_name] = reward

    def compute(self, task_name: str, success: bool, metadata: Dict[str, Any] = None) -> float:
        base = self.task_rewards.get(task_name, 0.0)
        if success:
            return base + 1.0
        # Small penalty for failure, modulated by effort (e.g., steps taken)
        effort = metadata.get("effort", 0.0) if metadata else 0.0
        return -0.1 - 0.001 * effort
