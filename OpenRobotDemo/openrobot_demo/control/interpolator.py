"""Joint-space trajectory interpolation."""

import numpy as np
from typing import List


class JointSpaceInterpolator:
    def __init__(self, num_steps: int = 50):
        self.num_steps = num_steps

    def plan(self, current: np.ndarray, target: np.ndarray) -> List[np.ndarray]:
        trajectory = []
        for alpha in np.linspace(0, 1, self.num_steps + 1):
            q = current + alpha * (target - current)
            trajectory.append(q.copy())
        return trajectory
