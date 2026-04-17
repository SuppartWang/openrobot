"""Point-cloud sensor: converts RGB-D into a point cloud."""

import time
from typing import Optional

import numpy as np

from .base import SensorChannel, PerceptionData


class PointCloudSensor(SensorChannel):
    """
    Generate a point cloud from RGB-D data.

    This MVP implementation uses numpy for depth-to-point-cloud conversion.
    If Open3D is available, future versions can optionally return o3d geometry.
    """

    name = "pointcloud"

    def __init__(
        self,
        source_id: str = "wrist_cam",
        width: int = 640,
        height: int = 480,
        fovy: float = 45.0,
        max_depth: float = 2.0,
        mujoco_model=None,
        mujoco_data=None,
    ):
        self.source_id = source_id
        self.width = width
        self.height = height
        self.fovy = np.deg2rad(fovy)
        self.max_depth = max_depth
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
            raise RuntimeError("PointCloudSensor: MuJoCo model/data not available")

        import mujoco

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=self.height, width=self.width)

        self._renderer.update_scene(self._data, camera=self.source_id)
        rgb = self._renderer.render()
        self._renderer.enable_depth_rendering()
        depth = self._renderer.render()
        self._renderer.disable_depth_rendering()

        points, colors = self._depth_to_pointcloud(depth, rgb)

        return PerceptionData(
            modality="pointcloud",
            source_id=self.source_id,
            timestamp=time.time(),
            payload={"points": points, "colors": colors},
            spatial_ref=f"{self.source_id}_frame",
            metadata={
                "width": self.width,
                "height": self.height,
                "num_points": len(points),
                "max_depth": self.max_depth,
            },
        )

    def _depth_to_pointcloud(self, depth: np.ndarray, rgb: np.ndarray):
        """Simple pinhole depth-to-point-cloud conversion."""
        h, w = depth.shape
        f = h / (2.0 * np.tan(self.fovy / 2.0))

        # Pixel coordinates
        u = np.arange(w)
        v = np.arange(h)
        u, v = np.meshgrid(u, v)

        z = depth
        x = (u - w / 2) * z / f
        y = (v - h / 2) * z / f

        # Stack and filter by max_depth and valid depth
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        colors = rgb.reshape(-1, 3) if rgb is not None else np.zeros((points.shape[0], 3))

        valid = (z.reshape(-1) > 0) & (z.reshape(-1) < self.max_depth)
        points = points[valid]
        colors = colors[valid]

        return points, colors

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
