"""LiDAR sensor: 2D or 3D laser rangefinder.

Real-world: Hokuyo UST, Velodyne VLP-16, Livox, Ouster, etc.
Simulation: ray-casting via MuJoCo or synthetic data.
"""

import time
import logging
from typing import Optional

import numpy as np

from .base import SensorChannel, PerceptionData

logger = logging.getLogger(__name__)


try:
    import rplidar
    _RPLIDAR_AVAILABLE = True
except Exception:
    _RPLIDAR_AVAILABLE = False


class LidarSensor(SensorChannel):
    """2D/3D LiDAR sensor."""

    name = "lidar"

    def __init__(
        self,
        source_id: str = "lidar_default",
        lidar_type: str = "2d",  # "2d" or "3d"
        num_beams: int = 360,
        max_range: float = 12.0,
        port: Optional[str] = None,
        # Mock
        mock_mode: bool = False,
    ):
        self.source_id = source_id
        self.lidar_type = lidar_type
        self.num_beams = num_beams
        self.max_range = max_range
        self.port = port
        self._mock_mode = mock_mode or not _RPLIDAR_AVAILABLE
        self._driver = None

        if not self._mock_mode and _RPLIDAR_AVAILABLE and port:
            try:
                self._driver = rplidar.RPLidar(port)
                logger.info("[LidarSensor] RPLidar connected on %s", port)
            except Exception as exc:
                logger.warning("[LidarSensor] Failed to connect RPLidar: %s", exc)
                self._mock_mode = True

    def is_available(self) -> bool:
        return True  # Mock always available

    def capture(self) -> PerceptionData:
        if self._mock_mode:
            # Synthetic scan: a circle with a wall at some angles
            angles = np.linspace(0, 2 * np.pi, self.num_beams, endpoint=False)
            ranges = np.full(self.num_beams, self.max_range, dtype=np.float32)
            # Simulate a wall at 90 degrees
            wall_start = int(self.num_beams * 0.2)
            wall_end = int(self.num_beams * 0.3)
            ranges[wall_start:wall_end] = 2.5
            # Add some noise
            ranges += np.random.normal(0, 0.02, self.num_beams)
            ranges = np.clip(ranges, 0.05, self.max_range)

            return PerceptionData(
                modality="lidar",
                source_id=self.source_id,
                timestamp=time.time(),
                payload={
                    "angles": angles.tolist(),
                    "ranges": ranges.tolist(),
                },
                spatial_ref="lidar_frame",
                metadata={
                    "type": self.lidar_type,
                    "num_beams": self.num_beams,
                    "max_range": self.max_range,
                    "mock": True,
                },
            )

        # Real RPLidar capture
        try:
            scan = next(self._driver.iter_scans())
            angles = []
            ranges_list = []
            for _, angle, distance in scan:
                angles.append(np.radians(angle))
                ranges_list.append(distance / 1000.0)  # mm to m
            return PerceptionData(
                modality="lidar",
                source_id=self.source_id,
                timestamp=time.time(),
                payload={
                    "angles": angles,
                    "ranges": ranges_list,
                },
                spatial_ref="lidar_frame",
                metadata={
                    "type": self.lidar_type,
                    "num_beams": len(scan),
                    "mock": False,
                },
            )
        except Exception as exc:
            logger.error("[LidarSensor] Capture failed: %s", exc)
            raise RuntimeError(f"LiDAR capture failed: {exc}")

    def close(self):
        if self._driver:
            try:
                self._driver.stop()
                self._driver.disconnect()
            except Exception as exc:
                logger.warning("[LidarSensor] Error closing: %s", exc)
