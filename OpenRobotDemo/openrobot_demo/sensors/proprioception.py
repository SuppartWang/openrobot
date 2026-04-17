"""Proprioception sensor: robot's own joint and end-effector state."""

import time
from typing import Optional, List

import numpy as np

from .base import SensorChannel, PerceptionData


class ProprioceptionSensor(SensorChannel):
    """Read robot joint positions, velocities, and end-effector pose."""

    name = "proprioception"

    def __init__(
        self,
        source_id: str = "franka_arm",
        arm_adapter=None,
        kinematics_solver=None,
    ):
        self.source_id = source_id
        self._arm = arm_adapter
        self._kin = kinematics_solver

    def is_available(self) -> bool:
        return self._arm is not None

    def capture(self) -> PerceptionData:
        if not self.is_available():
            raise RuntimeError("ProprioceptionSensor: arm adapter not available")

        joint_positions = np.array(self._arm.get_pos(), dtype=float)
        joint_velocities = getattr(self._arm, "get_vel", lambda: np.zeros_like(joint_positions))()
        if hasattr(self._arm, "_data"):
            joint_velocities = np.array(self._arm._data.qvel[: len(joint_positions)], dtype=float)

        ee_pose = None
        if self._kin is not None:
            ee_pose = self._kin.forward_quat(joint_positions.tolist())

        payload = {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "end_effector_pose": ee_pose,
        }

        return PerceptionData(
            modality="proprioception",
            source_id=self.source_id,
            timestamp=time.time(),
            payload=payload,
            spatial_ref="base_frame",
            metadata={"dof": len(joint_positions)},
        )
