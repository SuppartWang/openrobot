"""RealSense RGB sensor for OpenRobotDemo."""

import logging
import time
from typing import Optional

import numpy as np

from .base import SensorChannel, PerceptionData

logger = logging.getLogger(__name__)


try:
    import pyrealsense2 as rs

    _RS_AVAILABLE = True
except Exception as exc:
    _RS_AVAILABLE = False
    logger.debug("[RealSenseRGBSensor] pyrealsense2 not available: %s", exc)


class RealSenseRGBSensor(SensorChannel):
    """Capture aligned RGB frames from an Intel RealSense camera."""

    name = "realsense_rgb"

    def __init__(
        self,
        source_id: str = "rs_d435i_rgb",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.source_id = source_id
        self.width = width
        self.height = height
        self.fps = fps
        self._pipeline: Optional["rs.pipeline"] = None
        self._config: Optional["rs.config"] = None
        self._align: Optional["rs.align"] = None
        self._started = False

        if _RS_AVAILABLE:
            try:
                self._pipeline = rs.pipeline()
                self._config = rs.config()
                self._config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                self._config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
                profile = self._pipeline.start(self._config)
                self._align = rs.align(rs.stream.color)
                self._started = True
                logger.info("[RealSenseRGBSensor] Camera started (%dx%d@%d)", width, height, fps)
            except Exception as exc:
                logger.warning("[RealSenseRGBSensor] Failed to start camera: %s", exc)
                self._started = False

    def is_available(self) -> bool:
        return _RS_AVAILABLE and self._started

    def capture(self) -> PerceptionData:
        if not self.is_available():
            raise RuntimeError("RealSenseRGBSensor: camera not available")

        frames = self._pipeline.wait_for_frames(timeout_ms=5000)
        aligned_frames = self._align.process(frames)
        color_frame = aligned_frames.get_color_frame()
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
        if self._pipeline and self._started:
            try:
                self._pipeline.stop()
                logger.info("[RealSenseRGBSensor] Camera stopped")
            except Exception as exc:
                logger.warning("[RealSenseRGBSensor] Error stopping camera: %s", exc)
        self._started = False
