"""RealSense depth sensor for OpenRobotDemo."""

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
    logger.debug("[RealSenseDepthSensor] pyrealsense2 not available: %s", exc)


class RealSenseDepthSensor(SensorChannel):
    """Capture aligned depth frames from an Intel RealSense camera."""

    name = "realsense_depth"

    def __init__(
        self,
        source_id: str = "rs_d435i_depth",
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
                self._config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
                self._config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                profile = self._pipeline.start(self._config)
                self._align = rs.align(rs.stream.color)
                self._started = True
                logger.info("[RealSenseDepthSensor] Camera started (%dx%d@%d)", width, height, fps)
            except Exception as exc:
                logger.warning("[RealSenseDepthSensor] Failed to start camera: %s", exc)
                self._started = False

    def is_available(self) -> bool:
        return _RS_AVAILABLE and self._started

    def capture(self) -> PerceptionData:
        if not self.is_available():
            raise RuntimeError("RealSenseDepthSensor: camera not available")

        frames = self._pipeline.wait_for_frames(timeout_ms=5000)
        aligned_frames = self._align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
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
        try:
            profile = self._pipeline.get_active_profile()
            return profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        except Exception:
            return None

    def close(self) -> None:
        if self._pipeline and self._started:
            try:
                self._pipeline.stop()
                logger.info("[RealSenseDepthSensor] Camera stopped")
            except Exception as exc:
                logger.warning("[RealSenseDepthSensor] Error stopping camera: %s", exc)
        self._started = False
