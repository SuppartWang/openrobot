"""RealSense RGB sensor for OpenRobotDemo — uses shared device pool."""

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
    logger.debug("[RealSenseRGBSensor] pyrealsense2 not available: %s", exc)


class RealSenseRGBSensor(SensorChannel):
    """Capture aligned RGB frames from a shared RealSense pipeline."""

    name = "realsense_rgb"

    def __init__(
        self,
        source_id: str = "rs_d435i_rgb",
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
                logger.info("[RealSenseRGBSensor] Shared pipeline attached (%s)", serial)
            except Exception as exc:
                logger.warning("[RealSenseRGBSensor] Failed to attach shared pipeline: %s", exc)
                self._started = False

    def is_available(self) -> bool:
        return _RS_AVAILABLE and self._started

    def capture(self) -> PerceptionData:
        if not self.is_available():
            raise RuntimeError("RealSenseRGBSensor: camera not available")

        color_frame, _ = RealSenseDevicePool.capture_frames(self.serial, apply_filters=False)
        if not color_frame:
            raise RuntimeError("RealSenseRGBSensor: no color frame returned")

        rgb_image = np.asanyarray(color_frame.get_data())
        # BGR -> RGB for consistency
        rgb_image = np.ascontiguousarray(rgb_image[:, :, ::-1])

        return PerceptionData(
            modality="rgb",
            source_id=self.source_id,
            timestamp=time.time(),
            payload=rgb_image,
            spatial_ref=f"{self.source_id}_frame",
            metadata={"width": self.width, "height": self.height, "fps": self.fps},
        )

    def close(self) -> None:
        if self.serial:
            RealSenseDevicePool.release_device(self.serial)
            logger.info("[RealSenseRGBSensor] Released shared pipeline")
        self._started = False
