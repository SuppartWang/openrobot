"""Camera driver abstraction supporting USB (OpenCV), Intel RealSense, and MuJoCo renderer."""

import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class CameraDriver:
    """Unified camera interface."""

    def __init__(self, camera_type: str = "usb", device_id: int = 0,
                 width: int = 640, height: int = 480, fps: int = 30,
                 mujoco_model=None, mujoco_data=None, mujoco_camera_name: str = "wrist_cam"):
        self.camera_type = camera_type.lower()
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self._cap = None
        self._rs_pipeline = None
        self._rs_config = None
        self._rs_align = None
        self._fallback_synthetic = False

        # MuJoCo-specific
        self._mujoco_model = mujoco_model
        self._mujoco_data = mujoco_data
        self._mujoco_camera_name = mujoco_camera_name
        self._mujoco_renderer = None

    def connect(self) -> bool:
        if self.camera_type == "usb":
            try:
                import cv2
                self._cap = cv2.VideoCapture(self.device_id)
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self._cap.set(cv2.CAP_PROP_FPS, self.fps)
                if not self._cap.isOpened():
                    logger.warning(f"[CameraDriver] Failed to open USB camera {self.device_id}. Falling back to synthetic frames.")
                    self._cap = None
                    self._fallback_synthetic = True
                    return True
                logger.info(f"[CameraDriver] USB camera {self.device_id} connected.")
                return True
            except Exception as e:
                logger.warning(f"[CameraDriver] USB camera error: {e}. Falling back to synthetic frames.")
                self._cap = None
                self._fallback_synthetic = True
                return True

        elif self.camera_type == "realsense":
            try:
                import pyrealsense2 as rs
                self._rs_pipeline = rs.pipeline()
                self._rs_config = rs.config()
                self._rs_config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
                self._rs_config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
                self._rs_pipeline.start(self._rs_config)
                self._rs_align = rs.align(rs.stream.color)
                logger.info("[CameraDriver] RealSense connected.")
                return True
            except Exception as e:
                logger.error(f"[CameraDriver] RealSense error: {e}")
                return False

        elif self.camera_type == "mujoco":
            if self._mujoco_model is None:
                logger.error("[CameraDriver] MuJoCo model not provided.")
                return False
            try:
                import mujoco
                self._mujoco_renderer = mujoco.Renderer(self._mujoco_model, height=self.height, width=self.width)
                logger.info(f"[CameraDriver] MuJoCo renderer connected for camera '{self._mujoco_camera_name}'.")
                return True
            except Exception as e:
                logger.error(f"[CameraDriver] MuJoCo renderer error: {e}")
                return False

        else:
            logger.error(f"[CameraDriver] Unknown camera type: {self.camera_type}")
            return False

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (rgb_frame, depth_frame). Depth may be None for USB/MuJoCo RGB-only."""
        if self.camera_type == "usb":
            if self._fallback_synthetic:
                # Generate a synthetic test pattern when no real camera is available
                import numpy as np
                rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                # Draw a simple colored rectangle to simulate an object
                rgb[self.height//3:2*self.height//3, self.width//3:2*self.width//3] = [200, 100, 50]
                # Synthetic depth: 500 mm in center region, 1000 mm elsewhere
                depth = np.ones((self.height, self.width), dtype=np.uint16) * 1000
                depth[self.height//3:2*self.height//3, self.width//3:2*self.width//3] = 500
                return rgb, depth
            if self._cap is None:
                return None, None
            import cv2
            ret, frame = self._cap.read()
            if not ret:
                return None, None
            # Convert BGR -> RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return rgb, None

        elif self.camera_type == "realsense":
            if self._rs_pipeline is None:
                return None, None
            import pyrealsense2 as rs
            import cv2
            frames = self._rs_pipeline.wait_for_frames()
            aligned = self._rs_align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                return None, None
            rgb = np.asanyarray(color_frame.get_data())
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            depth = np.asanyarray(depth_frame.get_data())
            return rgb, depth

        elif self.camera_type == "mujoco":
            if self._mujoco_renderer is None or self._mujoco_data is None:
                return None, None
            self._mujoco_renderer.update_scene(self._mujoco_data, camera=self._mujoco_camera_name)
            rgb = self._mujoco_renderer.render()
            # No native depth from basic renderer; caller can compute depth from scene geometry
            return rgb, None

        return None, None

    def disconnect(self):
        if self.camera_type == "usb" and self._cap is not None:
            self._cap.release()
            self._cap = None
        elif self.camera_type == "realsense" and self._rs_pipeline is not None:
            self._rs_pipeline.stop()
            self._rs_pipeline = None
        elif self.camera_type == "mujoco" and self._mujoco_renderer is not None:
            self._mujoco_renderer.close()
            self._mujoco_renderer = None
        logger.info("[CameraDriver] Disconnected.")
