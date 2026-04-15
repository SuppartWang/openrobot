"""Execution adapter for MuJoCo simulation."""

import numpy as np
import mujoco
from openrobot_msgs import ActionCmd


class MujocoExecutor:
    """Executes ActionCmd on a MuJoCo simulation instance."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

    def apply(self, cmd: ActionCmd):
        if cmd.type == "joint_position":
            self.data.ctrl[:] = cmd.values[: self.model.nu]
        elif cmd.type == "ee_pose":
            # Simplified: for MVP, ee_pose commands require IK which is not yet implemented.
            # We keep the current control as a no-op placeholder.
            pass
        elif cmd.type == "gripper":
            # Assume last 2 actuators are fingers
            self.data.ctrl[-2:] = cmd.values[:2]
        elif cmd.type == "stop":
            self.data.ctrl[:] = 0.0
        else:
            raise ValueError(f"Unknown action type: {cmd.type}")

    def step(self, n_steps: int = 1):
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)
