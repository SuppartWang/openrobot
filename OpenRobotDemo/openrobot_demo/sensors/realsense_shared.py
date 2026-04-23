"""Shared RealSense device pool for multi-sensor concurrent access.

OpenRobotDemo's RealSenseRGBSensor and RealSenseDepthSensor both need
to consume frames from the same physical camera.  This module provides a
thread-safe singleton pool so that multiple SensorChannel instances can
share a single rs.pipeline() per serial number.

The pool also holds the post-processing filter chain (spatial, temporal,
hole-filling) so that every consumer gets filtered depth without
re-creating filters.
"""

import logging
import threading
from typing import Optional, Tuple, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pyrealsense2 as rs
    _RS_AVAILABLE = True
except Exception as exc:
    _RS_AVAILABLE = False
    logger.debug("[RealSenseDevicePool] pyrealsense2 not available: %s", exc)


class RealSenseDevicePool:
    """Thread-safe pool of shared RealSense pipelines keyed by serial number."""

    _lock = threading.Lock()
    _devices: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get_device(cls, serial: str, width: int = 640, height: int = 480, fps: int = 30):
        """Get or create a shared pipeline for the given device serial."""
        if not _RS_AVAILABLE:
            raise RuntimeError("pyrealsense2 is not available")

        with cls._lock:
            if serial not in cls._devices:
                logger.info("[RealSenseDevicePool] Creating shared pipeline for %s", serial)
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
                profile = pipeline.start(config)
                align = rs.align(rs.stream.color)

                # Post-processing filters (same as OpenRobotDemo depth pipeline)
                spatial = rs.spatial_filter()
                temporal = rs.temporal_filter()
                hole = rs.hole_filling_filter()

                cls._devices[serial] = {
                    "pipeline": pipeline,
                    "profile": profile,
                    "align": align,
                    "spatial": spatial,
                    "temporal": temporal,
                    "hole": hole,
                    "ref_count": 0,
                    "config": {"width": width, "height": height, "fps": fps},
                }

            cls._devices[serial]["ref_count"] += 1
            logger.info(
                "[RealSenseDevicePool] %s ref_count -> %d",
                serial,
                cls._devices[serial]["ref_count"],
            )
            return cls._devices[serial]

    @classmethod
    def release_device(cls, serial: str):
        """Release a reference to a shared pipeline.  Stops it when ref_count hits 0."""
        with cls._lock:
            if serial not in cls._devices:
                return
            cls._devices[serial]["ref_count"] -= 1
            new_count = cls._devices[serial]["ref_count"]
            logger.info("[RealSenseDevicePool] %s ref_count -> %d", serial, new_count)
            if new_count <= 0:
                try:
                    cls._devices[serial]["pipeline"].stop()
                    logger.info("[RealSenseDevicePool] %s pipeline stopped", serial)
                except Exception as exc:
                    logger.warning("[RealSenseDevicePool] Error stopping %s: %s", serial, exc)
                del cls._devices[serial]

    @classmethod
    def capture_frames(
        cls, serial: str, timeout_ms: int = 5000, apply_filters: bool = True
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """Capture aligned color + depth frames from the shared pipeline.

        Returns:
            (color_frame, depth_frame)  where depth has filters applied if requested.
        """
        device = cls._devices.get(serial)
        if device is None:
            raise RuntimeError(f"No shared pipeline for serial {serial}")

        frames = device["pipeline"].wait_for_frames(timeout_ms=timeout_ms)
        aligned = device["align"].process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if apply_filters and depth_frame:
            depth_frame = device["spatial"].process(depth_frame)
            depth_frame = device["temporal"].process(depth_frame)
            depth_frame = device["hole"].process(depth_frame)

        return color_frame, depth_frame

    @classmethod
    def get_intrinsics(cls, serial: str):
        """Return color camera intrinsics for the given serial."""
        device = cls._devices.get(serial)
        if device is None:
            return None
        try:
            return (
                device["profile"]
                .get_stream(rs.stream.color)
                .as_video_stream_profile()
                .get_intrinsics()
            )
        except Exception:
            return None
