"""Skill 1: CameraCapture"""

import time
import logging
from typing import Any, Dict
from openrobot_demo.skills.base import SkillInterface, SkillSchema, ParamSchema, ResultSchema
from openrobot_demo.perception.camera_driver import CameraDriver

logger = logging.getLogger(__name__)


class CameraCapture(SkillInterface):
    def __init__(self, camera_type: str = "usb", device_id: int = 0,
                 width: int = 640, height: int = 480, fps: int = 30,
                 serial: str = None):
        self._driver = CameraDriver(camera_type, device_id, width, height, fps, serial=serial)
        self._connected = False

    @property
    def name(self) -> str:
        return "camera_capture"

    @property
    def schema(self) -> SkillSchema:
        return SkillSchema(
            description="Capture an RGB image (and optionally depth) from the camera.",
            parameters=[
                ParamSchema(
                    name="return_depth",
                    type="bool",
                    description="Whether to also capture a depth frame.",
                    required=False,
                    default=True,
                ),
            ],
            returns=[
                ResultSchema(name="success", type="bool", description="Whether capture succeeded."),
                ResultSchema(name="message", type="str", description="Status message."),
                ResultSchema(name="rgb_frame", type="ndarray", description="RGB image array (H, W, 3)."),
                ResultSchema(name="depth_frame", type="ndarray", description="Depth image array (H, W) if return_depth=True."),
                ResultSchema(name="timestamp", type="float", description="Capture timestamp."),
            ],
            dependencies=["camera"],
        )

    def execute(self, return_depth: bool = True, **kwargs) -> Dict[str, Any]:
        if not self._connected:
            ok = self._driver.connect()
            if not ok:
                return {"success": False, "message": "Failed to connect camera."}
            self._connected = True

        rgb, depth = self._driver.read()
        if rgb is None:
            return {"success": False, "message": "Failed to capture frame from camera."}

        result = {
            "success": True,
            "message": "Frame captured.",
            "rgb_frame": rgb,
            "timestamp": time.time(),
        }
        if return_depth and depth is not None:
            result["depth_frame"] = depth
        return result

    def disconnect(self):
        self._driver.disconnect()
        self._connected = False
