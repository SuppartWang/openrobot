"""Odometry sensor: wheel encoders + pose estimation.

Real-world: rotary encoders on drive wheels, optical flow, visual odometry.
"""

import time
from typing import Optional

import numpy as np

from .base import SensorChannel, PerceptionData


class OdometrySensor(SensorChannel):
    """Wheel odometry and pose estimation."""

    name = "odometry"

    def __init__(
        self,
        source_id: str = "odom_default",
        wheel_radius: float = 0.05,
        wheel_base: float = 0.3,
        # Mock
        mock_pose: Optional[np.ndarray] = None,
        mock_twist: Optional[np.ndarray] = None,
    ):
        self.source_id = source_id
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self._mock_pose = mock_pose if mock_pose is not None else np.array([0.0, 0.0, 0.0])  # x, y, theta
        self._mock_twist = mock_twist if mock_twist is not None else np.array([0.0, 0.0])  # vx, wz

    def is_available(self) -> bool:
        return True

    def capture(self) -> PerceptionData:
        # In a real implementation, this would read encoder ticks and integrate
        pose = self._mock_pose.copy()
        twist = self._mock_twist.copy()

        return PerceptionData(
            modality="odometry",
            source_id=self.source_id,
            timestamp=time.time(),
            payload={
                "pose": pose.tolist(),  # [x, y, theta] in meters / radians
                "twist": twist.tolist(),  # [linear_x, angular_z]
                "wheel_radius": self.wheel_radius,
                "wheel_base": self.wheel_base,
            },
            spatial_ref="odom_frame",
            metadata={"mock": True},
        )
