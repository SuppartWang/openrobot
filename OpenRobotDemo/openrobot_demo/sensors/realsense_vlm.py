"""RealSense + VLM sensor: detects a target in RGB and returns its 3D position via depth."""

import base64
import json
import logging
import os
import re
import time
from typing import Optional

import cv2
import numpy as np

from .base import SensorChannel, PerceptionData
from .realsense_shared import RealSenseDevicePool

logger = logging.getLogger(__name__)

try:
    import pyrealsense2 as rs

    _RS_AVAILABLE = True
except Exception as exc:
    _RS_AVAILABLE = False
    logger.debug("[RealSenseVLMSensor] pyrealsense2 not available: %s", exc)


try:
    from openai import OpenAI

    _OPENAI_AVAILABLE = True
except Exception as exc:
    _OPENAI_AVAILABLE = False
    logger.debug("[RealSenseVLMSensor] openai not available: %s", exc)


class RealSenseVLMSensor(SensorChannel):
    """
    Capture RGB-D from RealSense, use a VLM (qwen-vl-max-latest) to detect a target,
    and deproject the 2D center into a 3D point in the camera frame.
    """

    name = "realsense_vlm"

    def __init__(
        self,
        source_id: str = "rs_d435i_vlm",
        target_name: str = "目标物体",
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen-vl-max-latest",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        detect_width: int = 640,
    ):
        self.source_id = source_id
        self.target_name = target_name
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        self.base_url = base_url
        self.model = model
        self.width = width
        self.height = height
        self.fps = fps
        self.detect_width = detect_width

        self._client = None
        if _OPENAI_AVAILABLE and self.api_key:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            logger.warning("[RealSenseVLMSensor] OpenAI client not available or API key missing.")

        self.serial: Optional[str] = None
        self._started = False

    def attach(self, serial: str, width: int = 640, height: int = 480, fps: int = 30):
        """Attach to a shared RealSense pipeline by serial number."""
        if _RS_AVAILABLE:
            try:
                RealSenseDevicePool.get_device(serial, width, height, fps)
                self.serial = serial
                self._started = True
                logger.info("[RealSenseVLMSensor] Attached to shared pipeline (%s)", serial)
            except Exception as exc:
                logger.warning("[RealSenseVLMSensor] Failed to attach shared pipeline: %s", exc)
                self._started = False

    def is_available(self) -> bool:
        return _RS_AVAILABLE and self._started and self._client is not None

    def capture(self) -> PerceptionData:
        if not self.is_available():
            raise RuntimeError("RealSenseVLMSensor: camera or VLM client not available")

        color_frame, depth_frame = RealSenseDevicePool.capture_frames(self.serial, apply_filters=True)
        if not depth_frame or not color_frame:
            raise RuntimeError("RealSenseVLMSensor: failed to get aligned frames")

        color_image = np.asanyarray(color_frame.get_data())
        intrinsics = RealSenseDevicePool.get_intrinsics(self.serial)

        # --- VLM detection ---
        h, w, _ = color_image.shape
        scale_ratio = self.detect_width / w
        detect_w = self.detect_width
        detect_h = int(h * scale_ratio)
        resized = cv2.resize(color_image, (detect_w, detect_h))

        _, buffer = cv2.imencode(".jpg", resized)
        base64_image = base64.b64encode(buffer).decode("utf-8")

        prompt_text = (
            f"你是一个精确的机器视觉定位系统。\n"
            f"请在宽{detect_w}，高{detect_h}的图片中，精确找到“{self.target_name}”的中心点。\n"
            f"要求：\n"
            f"1. 必须精准定位该物体中心位置的坐标。\n"
            f"2. 绝对禁止输出任何额外解释文字！\n"
            f"3. 格式只能是包含两个数字的纯JSON数组：[中心X, 中心Y]\n"
            f"如果找不到目标，请返回 [0, 0]"
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ],
                temperature=0.1,
            )
            result_text = response.choices[0].message.content.strip()
            logger.debug("[RealSenseVLMSensor] VLM raw response: %s", result_text)
        except Exception as exc:
            logger.error("[RealSenseVLMSensor] VLM API call failed: %s", exc)
            raise RuntimeError(f"VLM API call failed: {exc}")

        # Parse center point [cx, cy]
        clean_text = re.sub(r'[a-zA-Z"\'{}]+', "", result_text)
        nums = re.findall(r"\d+", clean_text)
        if len(nums) >= 2:
            cx = int(nums[0])
            cy = int(nums[1])
        else:
            logger.warning("[RealSenseVLMSensor] Could not parse center point from VLM response")
            return PerceptionData(
                modality="vlm_detection",
                source_id=self.source_id,
                timestamp=time.time(),
                payload={"pixel": None, "point_3d_cam": None, "target_name": self.target_name},
                spatial_ref=f"{self.source_id}_frame",
                confidence=0.0,
                metadata={"reason": "parse_failure"},
            )

        if cx == 0 and cy == 0:
            logger.info("[RealSenseVLMSensor] VLM reported target not found")
            return PerceptionData(
                modality="vlm_detection",
                source_id=self.source_id,
                timestamp=time.time(),
                payload={"pixel": [0, 0], "point_3d_cam": None, "target_name": self.target_name},
                spatial_ref=f"{self.source_id}_frame",
                confidence=0.0,
                metadata={"reason": "target_not_found"},
            )

        # Map back to original resolution
        real_cx = int(cx / scale_ratio)
        real_cy = int(cy / scale_ratio)

        # Get depth and deproject
        distance_m = depth_frame.get_distance(real_cx, real_cy)
        if distance_m <= 0:
            logger.warning("[RealSenseVLMSensor] No valid depth at (%d, %d)", real_cx, real_cy)
            return PerceptionData(
                modality="vlm_detection",
                source_id=self.source_id,
                timestamp=time.time(),
                payload={"pixel": [real_cx, real_cy], "point_3d_cam": None, "target_name": self.target_name},
                spatial_ref=f"{self.source_id}_frame",
                confidence=0.0,
                metadata={"reason": "no_depth"},
            )

        point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [real_cx, real_cy], distance_m)
        logger.info(
            "[RealSenseVLMSensor] Detected '%s' at pixel (%d, %d), depth=%.3fm, 3D=%s",
            self.target_name,
            real_cx,
            real_cy,
            distance_m,
            point_3d,
        )

        return PerceptionData(
            modality="vlm_detection",
            source_id=self.source_id,
            timestamp=time.time(),
            payload={
                "pixel": [real_cx, real_cy],
                "point_3d_cam": point_3d,
                "distance_m": distance_m,
                "target_name": self.target_name,
            },
            spatial_ref=f"{self.source_id}_frame",
            confidence=1.0,
            metadata={"detect_width": detect_w, "detect_height": detect_h},
        )

    def close(self) -> None:
        if self.serial:
            RealSenseDevicePool.release_device(self.serial)
            logger.info("[RealSenseVLMSensor] Released shared pipeline")
        self._started = False
