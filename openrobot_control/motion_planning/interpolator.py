"""Layer 3: Simple motion planning via joint-space interpolation."""

import numpy as np
from typing import List
from openrobot_msgs import ActionCmd


class JointSpaceInterpolator:
    """
    Generates a smooth joint-space trajectory from current positions to target positions.
    For MVP, uses linear interpolation; orientation/EE planning is future work.
    """

    def __init__(self, num_steps: int = 50):
        self.num_steps = num_steps

    def plan(self, current: np.ndarray, target: np.ndarray) -> List[ActionCmd]:
        trajectory = []
        for alpha in np.linspace(0, 1, self.num_steps + 1):
            q = current + alpha * (target - current)
            trajectory.append(ActionCmd(type="joint_position", values=q))
        return trajectory


class GripperTrajectory:
    """Simple open/close gripper trajectory."""

    def __init__(self, num_steps: int = 20):
        self.num_steps = num_steps

    def plan(self, open_width: float) -> List[ActionCmd]:
        # Two finger joints: assume symmetric control
        target = np.array([open_width, open_width])
        return [ActionCmd(type="gripper", values=target) for _ in range(self.num_steps)]
