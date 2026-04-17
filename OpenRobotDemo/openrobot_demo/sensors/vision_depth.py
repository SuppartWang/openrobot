"""Depth vision sensor for MuJoCo simulation."""

import time
from typing import Optional

import numpy as np

from .base import SensorChannel, PerceptionData


class VisionDepthSensor(SensorChannel):
    """Capture depth frames from a MuJoCo camera."""

    name = "vision_depth"

    def __init__(
        self,
        source_id: str = "wrist_cam",
        width: int = 640,
        height: int = 480,
        mujoco_model=None,
        mujoco_data=None,
    ):
        self.source_id = source_id
        self.width = width
        self.height = height
        self._model = mujoco_model
        self._data = mujoco_data
        self._renderer = None
        if self._model is not None:
            try:
                import mujoco

                self._renderer = mujoco.Renderer(self._model, height=height, width=width)
            except Exception:
                pass

    def is_available(self) -> bool:
        return self._model is not None and self._data is not None

    def capture(self) -> PerceptionData:
        if not self.is_available():
            raise RuntimeError("VisionDepthSensor: MuJoCo model/data not available")

        import mujoco

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=self.height, width=self.width)

        self._renderer.update_scene(self._data, camera=self.source_id)
        # MuJoCo 3.x requires enabling depth rendering explicitly
        self._renderer.enable_depth_rendering()
        depth = self._renderer.render()
        self._renderer.disable_depth_rendering()

        return PerceptionData(
            modality="depth",
            source_id=self.source_id,
            timestamp=time.time(),
            payload=depth,
            spatial_ref=f"{self.source_id}_frame",
            metadata={"width": self.width, "height": self.height, "units": "meters"},
        )

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
