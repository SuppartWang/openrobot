"""Skill 3: Vision3DEstimator"""

import os
import re
import json
import base64
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R

from dotenv import load_dotenv

# Load .env from OpenRobotDemo root
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from openrobot_demo.skills.base import SkillInterface, SkillSchema, ParamSchema, ResultSchema

logger = logging.getLogger(__name__)


class Vision3DEstimator(SkillInterface):
    """Uses a VLM (e.g. qwen-vl-max) to detect objects and reproject to 3D."""

    def __init__(self, model: str = "qwen-vl-max-latest",
                 api_key: Optional[str] = None,
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.model = model
        self.api_key = api_key or os.getenv("KIMI_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self._client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception as e:
                logger.warning(f"Failed to init OpenAI client: {e}")
        else:
            logger.warning("No API key provided for Vision3DEstimator.")

    @property
    def name(self) -> str:
        return "vision_3d_estimator"

    @property
    def schema(self) -> SkillSchema:
        return SkillSchema(
            description="Detect a target object in an RGB image and estimate its 3D position using depth and camera intrinsics.",
            parameters=[
                ParamSchema(name="rgb_frame", type="ndarray", description="RGB image array (H, W, 3).", required=True),
                ParamSchema(name="target_name", type="str", description="Name of the object to detect, e.g. 'cube', '筒状布料'.", required=True),
                ParamSchema(name="depth_frame", type="ndarray", description="Depth image array (H, W) in mm.", required=False, default=None),
                ParamSchema(name="camera_intrinsics", type="dict", description="Camera intrinsics dict with fx, fy, ppx, ppy.", required=False, default=None),
                ParamSchema(name="hand_eye_calib", type="dict", description="Hand-eye calibration dict with rotation_matrix and translation_vector.", required=False, default=None),
                ParamSchema(name="end_effector_pose", type="list", description="Current end-effector pose for hand-eye transform.", required=False, default=None),
                ParamSchema(name="ground_truth_depth_mm", type="float", description="Optional ground-truth depth in mm for testing.", required=False, default=None),
                ParamSchema(name="detect_mode", type="str", description='Detection mode: "center" (default), "opening_left", "opening_right", "bottom_edge".', required=False, default="center"),
                ParamSchema(name="opening_offset_m", type="float", description="X-axis offset for opening detection (default 0.2m).", required=False, default=0.2),
            ],
            returns=[
                ResultSchema(name="success", type="bool", description="Whether detection succeeded."),
                ResultSchema(name="message", type="str", description="Status message."),
                ResultSchema(name="pixel_bbox", type="list", description="Bounding box [x1, y1, x2, y2] in pixel coordinates."),
                ResultSchema(name="pixel_center", type="list", description="Center point [u, v] in pixel coordinates."),
                ResultSchema(name="camera_3d", type="list", description="3D position [x, y, z] in camera frame (meters)."),
                ResultSchema(name="base_3d", type="list", description="3D position [x, y, z] in robot base frame (meters)."),
            ],
            dependencies=["camera", "vlm_api"],
        )

    def execute(self,
                rgb_frame: np.ndarray,
                target_name: str,
                depth_frame: Optional[np.ndarray] = None,
                camera_intrinsics: Optional[Dict[str, float]] = None,
                hand_eye_calib: Optional[Dict[str, Any]] = None,
                end_effector_pose: Optional[list] = None,
                ground_truth_depth_mm: Optional[float] = None,
                detect_mode: str = "center",
                opening_offset_m: float = 0.2,
                **kwargs) -> Dict[str, Any]:
        intrinsics = camera_intrinsics or {"fx": 600.0, "fy": 600.0, "ppx": 320.0, "ppy": 240.0}

        # Step 1: VLM detection (with color-based fallback)
        if self._client is None:
            logger.warning("VLM client not initialized. Using color detection fallback.")
            bbox, center = self._mock_detect(rgb_frame, detect_mode)
        else:
            bbox, center = self._detect_with_vlm(rgb_frame, target_name, detect_mode)
            if bbox is None:
                logger.warning("VLM detection failed. Falling back to color detection.")
                bbox, center = self._mock_detect(rgb_frame, detect_mode)

        if bbox is None:
            return {"success": False, "message": f"Could not detect '{target_name}' (mode={detect_mode})."}

        u, v = int(center[0]), int(center[1])

        # Apply opening offset in camera X direction (pixel x)
        if detect_mode in ("opening_left", "opening_right"):
            # Convert offset in meters to pixel offset using approximate focal length
            fx = (camera_intrinsics or {"fx": 600.0})["fx"]
            # Need depth to convert meters to pixels
            temp_z = None
            if depth_frame is not None:
                h, w = depth_frame.shape
                if 0 <= v < h and 0 <= u < w:
                    patch = depth_frame[max(0, v - 2):min(h, v + 3), max(0, u - 2):min(w, u + 3)]
                    temp_z = float(np.median(patch)) * 0.001  # mm -> m
            if temp_z is None and ground_truth_depth_mm is not None:
                temp_z = ground_truth_depth_mm * 0.001
            if temp_z is not None and temp_z > 0:
                pixel_offset = int(opening_offset_m * fx / temp_z)
                sign = -1 if detect_mode == "opening_left" else 1
                u += sign * pixel_offset
                u = max(0, min(u, rgb_frame.shape[1] - 1))
                logger.info("[Vision3DEstimator] Applied opening offset: %+.3fm -> %+d px (z=%.3fm)",
                            sign * opening_offset_m, sign * pixel_offset, temp_z)

        # Step 2: Get depth at center
        camera_3d = None
        z_m = None
        if depth_frame is not None:
            h, w = depth_frame.shape
            if 0 <= v < h and 0 <= u < w:
                # median filter over a small patch for robustness
                patch = depth_frame[max(0, v - 2):min(h, v + 3), max(0, u - 2):min(w, u + 3)]
                z_raw = float(np.median(patch))
                z_mm = z_raw  # assume depth_frame already in mm if from RealSense raw
                z_m = z_mm * 0.001
            else:
                logger.warning(f"Center ({u},{v}) out of depth bounds.")
        elif ground_truth_depth_mm is not None:
            z_m = ground_truth_depth_mm * 0.001
            logger.info(f"Using ground-truth depth: {z_m:.3f} m")
        else:
            logger.warning("No depth_frame or GT depth provided; skipping 3D reprojection.")

        if z_m is not None:
            fx, fy, ppx, ppy = intrinsics["fx"], intrinsics["fy"], intrinsics["ppx"], intrinsics["ppy"]
            x_m = z_m * (u - ppx) / fx
            y_m = z_m * (v - ppy) / fy
            camera_3d = [x_m, y_m, z_m]

        # Step 3: Hand-eye transform to base frame
        base_3d = None
        if camera_3d is not None and hand_eye_calib is not None and end_effector_pose is not None:
            try:
                base_3d = self._cam_to_base(camera_3d, hand_eye_calib, end_effector_pose)
            except Exception as e:
                logger.warning(f"Hand-eye transform failed: {e}")

        return {
            "success": True,
            "message": f"Detected '{target_name}' at pixel ({u},{v}).",
            "pixel_bbox": bbox,
            "pixel_center": [u, v],
            "camera_3d": camera_3d,
            "base_3d": base_3d,
        }

    def _mock_detect(self, rgb_frame: np.ndarray, detect_mode: str = "center"):
        """Fallback detection for testing without VLM API.
        Tries color-based object detection (yellow) using OpenCV;
        falls back to image center if no blob is found.
        """
        try:
            import cv2
            hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([40, 255, 255])
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                cx, cy = x + w / 2.0, y + h / 2.0
                # Adjust center based on detect_mode
                if detect_mode == "bottom_edge":
                    cy = y + h * 0.9  # near bottom
                elif detect_mode == "opening_left":
                    cx = x + w * 0.2  # near left edge
                elif detect_mode == "opening_right":
                    cx = x + w * 0.8  # near right edge
                return (x, y, x + w, y + h), (cx, cy)
        except Exception as e:
            logger.warning(f"Color detection failed: {e}")

        # Fallback: center of image with mode offset
        h, w = rgb_frame.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        if detect_mode == "bottom_edge":
            cy = h * 0.85
        elif detect_mode == "opening_left":
            cx = w * 0.3
        elif detect_mode == "opening_right":
            cx = w * 0.7
        x1, y1 = int(w * 0.3), int(h * 0.3)
        x2, y2 = int(w * 0.7), int(h * 0.7)
        return (x1, y1, x2, y2), (cx, cy)

    def _detect_with_vlm(self, rgb_frame: np.ndarray, target_name: str, detect_mode: str = "center"):
        """Call VLM and parse bounding box. Returns (bbox, center) or (None, None)."""
        import cv2
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
        b64_image = base64.b64encode(buf.tobytes()).decode("utf-8")

        # Build prompt based on detect_mode
        if detect_mode == "bottom_edge":
            target_desc = f"{target_name}的最下沿中心点"
            extra_instr = (
                "请在图片中找到该物体的最下方边缘，返回最下沿中心点的坐标。"
                "如果物体是弧形，返回弧形最下端的中心位置。"
            )
        elif detect_mode in ("opening_left", "opening_right"):
            side = "左侧" if detect_mode == "opening_left" else "右侧"
            target_desc = f"{target_name}的{side}开口处"
            extra_instr = (
                f"请在图片中找到该物体的{side}开口/边缘处，"
                f"返回{side}开口中心点的坐标。"
            )
        else:
            target_desc = target_name
            extra_instr = "请在图片中找到该目标并返回边界框中心点坐标。"

        system_prompt = (
            "你是一个专业的机器人视觉检测助手。任务是：\n"
            "1. 分析图片中的物体；\n"
            f"2. {extra_instr}\n"
            "3. 返回该目标的边界框左上角和右下角坐标；\n"
            "坐标系：图片左上角为原点(0,0)，向右为x轴正方向，向下为y轴正方向\n"
            "必须严格按纯文本JSON返回：{\"x1\": 200, \"y1\": 150, \"x2\": 500, \"y2\": 350}\n"
            "未找到返回：{\"x1\": 0, \"y1\": 0, \"x2\": 0, \"y2\": 0}\n"
            "不要输出任何多余文字。"
        )
        user_prompt = f'请在图片中找到"{target_desc}"并返回边界框坐标。'

        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                ],
                temperature=0.1,
                max_tokens=100,
            )
            text = completion.choices[0].message.content.strip()
            json_match = re.search(r'\{[^}]*"x1"[^}]*"y1"[^}]*"x2"[^}]*"y2"[^}]*\}', text)
            if not json_match:
                return None, None
            coords = json.loads(json_match.group())
            x1, y1, x2, y2 = int(coords["x1"]), int(coords["y1"]), int(coords["x2"]), int(coords["y2"])
            if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                return None, None
            return (x1, y1, x2, y2), ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        except Exception as e:
            logger.error(f"VLM call failed: {e}")
            return None, None

    def _cam_to_base(self, camera_3d: list, hand_eye_calib: dict, ee_pose: list) -> list:
        """Transform point from camera frame to robot base frame."""
        obj_cam = np.array(camera_3d)
        rot_mat = np.array(hand_eye_calib["rotation_matrix"])
        trans_vec = np.array(hand_eye_calib["translation_vector"])

        T_cam2end = np.eye(4)
        T_cam2end[:3, :3] = rot_mat
        T_cam2end[:3, 3] = trans_vec

        pos = np.array(ee_pose[:3])
        quat = ee_pose[3:7]
        ori = R.from_quat(quat).as_matrix()

        T_base2end = np.eye(4)
        T_base2end[:3, :3] = ori
        T_base2end[:3, 3] = pos

        obj_homo = np.append(obj_cam, 1.0)
        obj_base_homo = T_base2end @ T_cam2end @ obj_homo
        return obj_base_homo[:3].tolist()
