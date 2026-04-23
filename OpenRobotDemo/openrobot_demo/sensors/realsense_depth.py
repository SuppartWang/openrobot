"""RealSense depth sensor for OpenRobotDemo — uses shared device pool."""

import logging
import time
from typing import Optional

import numpy as np

from .base import SensorChannel, PerceptionData
from .realsense_shared import RealSenseDevicePool

logger = logging.getLogger(__name__)

try:
    import pyrealsense2 as rs
    _RS_AVAILABLE = True
except Exception as exc:
    _RS_AVAILABLE = False
    logger.debug("[RealSenseDepthSensor] pyrealsense2 not available: %s", exc)


class RealSenseDepthSensor(SensorChannel):
    """Capture aligned depth frames from a shared RealSense pipeline."""

    name = "realsense_depth"

    def __init__(
        self,
        source_id: str = "rs_d435i_depth",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial: Optional[str] = None,
    ):
        self.source_id = source_id
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self._started = False

        if _RS_AVAILABLE and serial:
            try:
                RealSenseDevicePool.get_device(serial, width, height, fps)
                self._started = True
                logger.info("[RealSenseDepthSensor] Shared pipeline attached (%s)", serial)
            except Exception as exc:
                logger.warning("[RealSenseDepthSensor] Failed to attach shared pipeline: %s", exc)
                self._started = False

    def is_available(self) -> bool:
        return _RS_AVAILABLE and self._started

    def capture(self) -> PerceptionData:
        if not self.is_available():
            raise RuntimeError("RealSenseDepthSensor: camera not available")

        _, depth_frame = RealSenseDevicePool.capture_frames(self.serial, apply_filters=True)
        if not depth_frame:
            raise RuntimeError("RealSenseDepthSensor: no depth frame returned")

        depth_image = np.asanyarray(depth_frame.get_data())
        # Convert uint16 mm -> float meters
        depth_meters = depth_image.astype(np.float32) * 0.001

        return PerceptionData(
            modality="depth",
            source_id=self.source_id,
            timestamp=time.time(),
            payload=depth_meters,
            spatial_ref=f"{self.source_id}_frame",
            metadata={
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "units": "meters",
            },
        )

    def get_intrinsics(self):
        """Return color camera intrinsics if available."""
        if not self.is_available():
            return None
        return RealSenseDevicePool.get_intrinsics(self.serial)

    def close(self) -> None:
        if self.serial:
            RealSenseDevicePool.release_device(self.serial)
            logger.info("[RealSenseDepthSensor] Released shared pipeline")
        self._started = False
