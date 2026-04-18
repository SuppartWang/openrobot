"""IMU sensor: accelerometer + gyroscope + optional magnetometer.

Real-world: Bosch BNO055, MPU9250, XSens MTi, etc.
Simulation: read from MuJoCo accelerometer/gyro sites.
"""

import time
from typing import Optional

import numpy as np

from .base import SensorChannel, PerceptionData


class IMUSensor(SensorChannel):
    """Inertial Measurement Unit sensor."""

    name = "imu"

    def __init__(
        self,
        source_id: str = "imu_body",
        mujoco_model=None,
        mujoco_data=None,
        body_name: Optional[str] = None,
        # Mock defaults
        mock_acc: Optional[np.ndarray] = None,
        mock_gyro: Optional[np.ndarray] = None,
        mock_quat: Optional[np.ndarray] = None,
    ):
        self.source_id = source_id
        self._model = mujoco_model
        self._data = mujoco_data
        self._body_name = body_name
        self._mock_acc = mock_acc if mock_acc is not None else np.array([0.0, 0.0, 9.81])
        self._mock_gyro = mock_gyro if mock_gyro is not None else np.array([0.0, 0.0, 0.0])
        self._mock_quat = mock_quat if mock_quat is not None else np.array([1.0, 0.0, 0.0, 0.0])

    def is_available(self) -> bool:
        # Always available in mock mode; requires MuJoCo in sim mode
        return True

    def capture(self) -> PerceptionData:
        if self._model is not None and self._data is not None and self._body_name is not None:
            import mujoco
            body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, self._body_name)
            if body_id >= 0:
                # MuJoCo body accelerometer and gyro are available via sensor API
                # For now, approximate from body velocity derivatives
                acc = self._data.cacc[body_id * 6 : body_id * 6 + 3].copy()
                gyro = self._data.cvel[body_id * 6 : body_id * 6 + 3].copy()
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, self._data.xmat[body_id].reshape(3, 3).flatten())
                # MuJoCo quat is [w, x, y, z]
                quat = np.array([quat[1], quat[2], quat[3], quat[0]])
            else:
                acc, gyro, quat = self._mock_acc, self._mock_gyro, self._mock_quat
        else:
            acc, gyro, quat = self._mock_acc, self._mock_gyro, self._mock_quat

        return PerceptionData(
            modality="imu",
            source_id=self.source_id,
            timestamp=time.time(),
            payload={
                "acceleration": acc.tolist(),  # m/s^2
                "angular_velocity": gyro.tolist(),  # rad/s
                "orientation_quat": quat.tolist(),  # [x, y, z, w]
            },
            spatial_ref="imu_frame",
            metadata={"has_magnetometer": False},
        )
