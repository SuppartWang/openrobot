"""Tactile sensor: contact forces from MuJoCo simulation."""

import time
from typing import List, Optional

import numpy as np

from .base import SensorChannel, PerceptionData


class TactileSensor(SensorChannel):
    """
    Approximate tactile sensing using MuJoCo contact forces.

    For each tracked body name, sum the contact normal forces acting on it.
    """

    name = "tactile"

    def __init__(
        self,
        source_id: str = "gripper",
        body_names: Optional[List[str]] = None,
        mujoco_model=None,
        mujoco_data=None,
    ):
        self.source_id = source_id
        self.body_names = body_names or ["gripper_base", "left_finger", "right_finger"]
        self._model = mujoco_model
        self._data = mujoco_data

    def is_available(self) -> bool:
        return self._model is not None and self._data is not None

    def capture(self) -> PerceptionData:
        if not self.is_available():
            raise RuntimeError("TactileSensor: MuJoCo model/data not available")

        import mujoco

        body_forces = {}
        total_force = 0.0

        for body_name in self.body_names:
            body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                body_forces[body_name] = 0.0
                continue

            # Sum contact forces where either geom1 or geom2 belongs to this body
            force_sum = 0.0
            for i in range(self._data.ncon):
                contact = self._data.contact[i]
                geom1_body = self._model.geom_bodyid[contact.geom1]
                geom2_body = self._model.geom_bodyid[contact.geom2]

                if geom1_body == body_id or geom2_body == body_id:
                    # Extract contact force in contact frame
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self._model, self._data, i, force)
                    normal_force = force[0]  # normal component
                    force_sum += abs(normal_force)

            body_forces[body_name] = force_sum
            total_force += force_sum

        # Simple binary contact detection based on force threshold
        contact_detected = total_force > 0.1

        return PerceptionData(
            modality="tactile",
            source_id=self.source_id,
            timestamp=time.time(),
            payload={
                "body_forces": body_forces,
                "total_force": total_force,
                "in_contact": contact_detected,
            },
            spatial_ref="base_frame",
            metadata={"tracked_bodies": self.body_names},
        )
