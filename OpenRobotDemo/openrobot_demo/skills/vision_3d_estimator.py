"""Skill 3: Vision3DEstimator"""

import os
import re
import json
import base64
import logging
from typing import Any, Dict, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R

from openrobot_demo.skills.base import SkillInterface

logger = logging.getLogger(__name__)


class Vision3DEstimator(SkillInterface):
    """Uses a VLM (e.g. qwen-vl-max) to detect objects and reproject to 3D."""

    def __init__(self, model: str = "qwen-vl-max-latest",
                 api_key: Optional[str] = None,
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.model = model
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
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

    def execute(self,
                rgb_frame: np.ndarray,
                target_name: str,
                depth_frame: Optional[np.ndarray] = None,
                camera_intrinsics: Optional[Dict[str, float]] = None,
                hand_eye_calib: Optional[Dict[str, Any]] = None,
                end_effector_pose: Optional[list] = None,
                **kwargs) -> Dict[str, Any]:
        intrinsics = camera_intrinsics or {"fx": 600.0, "fy": 600.0, "ppx": 320.0, "ppy": 240.0}

        # Step 1: VLM detection (with mock fallback)
        if self._client is None:
            logger.warning("VLM client not initialized. Using mock detection fallback.")
            bbox, center = self._mock_detect(rgb_frame)
        else:
            bbox, center = self._detect_with_vlm(rgb_frame, target_name)

        if bbox is None:
            return {"success": False, "message": f"Could not detect '{target_name}'."}

        u, v = int(center[0]), int(center[1])

        # Step 2: Get depth at center
        camera_3d = None
        if depth_frame is not None:
            h, w = depth_frame.shape
            if 0 <= v < h and 0 <= u < w:
                # median filter over a small patch for robustness
                patch = depth_frame[max(0, v - 2):min(h, v + 3), max(0, u - 2):min(w, u + 3)]
                z_raw = float(np.median(patch))
                z_mm = z_raw  # assume depth_frame already in mm if from RealSense raw
                z_m = z_mm * 0.001
                fx, fy, ppx, ppy = intrinsics["fx"], intrinsics["fy"], intrinsics["ppx"], intrinsics["ppy"]
                x_m = z_m * (u - ppx) / fx
                y_m = z_m * (v - ppy) / fy
                camera_3d = [x_m, y_m, z_m]
            else:
                logger.warning(f"Center ({u},{v}) out of depth bounds.")
        else:
            logger.warning("No depth_frame provided; skipping 3D reprojection.")

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

    def _mock_detect(self, rgb_frame: np.ndarray):
        """Fallback detection for testing without VLM API.
        Returns a bounding box around the center of the image.
        """
        h, w = rgb_frame.shape[:2]
        x1, y1 = w // 3, h // 3
        x2, y2 = 2 * w // 3, 2 * h // 3
        return (x1, y1, x2, y2), ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _detect_with_vlm(self, rgb_frame: np.ndarray, target_name: str):
        """Call VLM and parse bounding box. Returns (bbox, center) or (None, None)."""
        import cv2
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
        b64_image = base64.b64encode(buf.tobytes()).decode("utf-8")

        system_prompt = (
            "你是一个专业的目标检测助手。任务是：\n"
            "1. 分析图片中的物体；\n"
            "2. 找到用户指定的目标；\n"
            "3. 返回该物体的边界框左上角和右下角坐标；\n"
            "坐标系：图片左上角为原点(0,0)，向右为x轴正方向，向下为y轴正方向\n"
            "必须严格按纯文本JSON返回：{\"x1\": 200, \"y1\": 150, \"x2\": 500, \"y2\": 350}\n"
            "未找到返回：{\"x1\": 0, \"y1\": 0, \"x2\": 0, \"y2\": 0}\n"
            "不要输出任何多余文字。"
        )
        user_prompt = f'请在图片中找到"{target_name}"并返回边界框坐标。'

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
