"""Wrench sensor: 6-DOF force/torque sensor.

Real-world: ATI Nano, Robotiq FT-300, OnRobot HEX, etc.
Simulation: aggregate contact forces at the sensor site.
"""

import time
from typing import Optional, List

import numpy as np

from .base import SensorChannel, PerceptionData


class WrenchSensor(SensorChannel):
    """6-DOF force/torque sensor."""

    name = "wrench"

    def __init__(
        self,
        source_id: str = "wrist_ft",
        mujoco_model=None,
        mujoco_data=None,
        site_name: Optional[str] = None,
        body_names: Optional[List[str]] = None,
        # Mock defaults
        mock_force: Optional[np.ndarray] = None,
        mock_torque: Optional[np.ndarray] = None,
    ):
        self.source_id = source_id
        self._model = mujoco_model
        self._data = mujoco_data
        self._site_name = site_name
        self._body_names = body_names or []
        self._mock_force = mock_force if mock_force is not None else np.zeros(3)
        self._mock_torque = mock_torque if mock_torque is not None else np.zeros(3)

    def is_available(self) -> bool:
        return True

    def capture(self) -> PerceptionData:
        force = self._mock_force.copy()
        torque = self._mock_torque.copy()

        if self._model is not None and self._data is not None:
            # Try to read from MuJoCo sensor sites if available
            if self._site_name:
                import mujoco
                site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, self._site_name)
                if site_id >= 0 and hasattr(self._data, "sensordata"):
                    # Approximate: read sensor data if an F/T sensor is configured
                    pass  # Requires explicit sensor config in XML

            # Fallback: sum contact forces on tracked bodies
            if self._body_names:
                import mujoco
                total_force = np.zeros(3)
                total_torque = np.zeros(3)
                for body_name in self._body_names:
                    body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                    if body_id < 0:
                        continue
                    for i in range(self._data.ncon):
                        contact = self._data.contact[i]
                        geom1_body = self._model.geom_bodyid[contact.geom1]
                        geom2_body = self._model.geom_bodyid[contact.geom2]
                        if geom1_body == body_id or geom2_body == body_id:
                            cforce = np.zeros(6)
                            mujoco.mj_contactForce(self._model, self._data, i, cforce)
                            total_force += cforce[:3]
                            # Torque approximation about body CoM
                            contact_pos = contact.pos.copy()
                            body_pos = self._data.xpos[body_id].copy()
                            r = contact_pos - body_pos
                            total_torque += np.cross(r, cforce[:3]) + cforce[3:]
                force = total_force
                torque = total_torque

        return PerceptionData(
            modality="wrench",
            source_id=self.source_id,
            timestamp=time.time(),
            payload={
                "force": force.tolist(),    # N [fx, fy, fz]
                "torque": torque.tolist(),  # Nm [tx, ty, tz]
            },
            spatial_ref="sensor_frame",
            metadata={"units": "SI"},
        )
