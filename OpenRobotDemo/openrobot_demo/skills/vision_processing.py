"""Vision processing skills: object detection, feature extraction, tracking.

These skills operate on image data and produce semantic/structural outputs.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from openrobot_demo.skills.base import SkillInterface, SkillSchema, ParamSchema, ResultSchema

logger = logging.getLogger(__name__)


class ColorDetectorSkill(SkillInterface):
    """Detect objects by color using HSV thresholding."""

    name = "color_detector"

    @property
    def schema(self) -> SkillSchema:
        return SkillSchema(
            description="Detect colored objects in an RGB image using HSV thresholding (fallback when VLM is unavailable).",
            parameters=[
                ParamSchema(name="rgb_frame", type="ndarray", description="RGB image array (H, W, 3).", required=True),
                ParamSchema(name="color", type="str", description="Color to detect: 'red', 'green', 'blue', 'yellow', 'white', 'black'.", required=True),
                ParamSchema(name="min_area", type="int", description="Minimum contour area in pixels.", required=False, default=100),
            ],
            returns=[
                ResultSchema(name="success", type="bool", description="Whether detection succeeded."),
                ResultSchema(name="message", type="str", description="Status message."),
                ResultSchema(name="detections", type="list", description="List of detections, each with {'bbox': [x1,y1,x2,y2], 'center': [cx,cy], 'area': int}."),
                ResultSchema(name="num_detections", type="int", description="Number of objects detected."),
            ],
            dependencies=["camera"],
        )

    def execute(self, rgb_frame, color: str, min_area: int = 100, **kwargs) -> Dict[str, Any]:
        try:
            import cv2
            img = np.array(rgb_frame, dtype=np.uint8)
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            color_ranges = {
                "red": [(np.array([0, 100, 100]), np.array([10, 255, 255])),
                        (np.array([160, 100, 100]), np.array([180, 255, 255]))],
                "green": [(np.array([35, 100, 100]), np.array([85, 255, 255]))],
                "blue": [(np.array([90, 100, 100]), np.array([130, 255, 255]))],
                "yellow": [(np.array([20, 100, 100]), np.array([40, 255, 255]))],
                "white": [(np.array([0, 0, 200]), np.array([180, 30, 255]))],
                "black": [(np.array([0, 0, 0]), np.array([180, 255, 50]))],
            }

            ranges = color_ranges.get(color.lower())
            if not ranges:
                return {"success": False, "message": f"Unknown color: {color}"}

            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, lower, upper)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detections = []
            for c in contours:
                area = cv2.contourArea(c)
                if area < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                detections.append({
                    "bbox": [x, y, x + w, y + h],
                    "center": [x + w / 2, y + h / 2],
                    "area": int(area),
                })

            return {
                "success": True,
                "message": f"Detected {len(detections)} {color} object(s).",
                "detections": detections,
                "num_detections": len(detections),
            }
        except Exception as exc:
            logger.exception("[ColorDetectorSkill] Failed")
            return {"success": False, "message": str(exc)}


class FeatureExtractorSkill(SkillInterface):
    """Extract visual features from an image (e.g. ORB keypoints)."""

    name = "feature_extractor"

    @property
    def schema(self) -> SkillSchema:
        return SkillSchema(
            description="Extract visual features (ORB keypoints and descriptors) from an RGB image for tracking or matching.",
            parameters=[
                ParamSchema(name="rgb_frame", type="ndarray", description="RGB image array (H, W, 3).", required=True),
                ParamSchema(name="max_features", type="int", description="Maximum number of features to extract.", required=False, default=500),
            ],
            returns=[
                ResultSchema(name="success", type="bool", description="Whether extraction succeeded."),
                ResultSchema(name="message", type="str", description="Status message."),
                ResultSchema(name="keypoints", type="list", description="List of keypoint coordinates [[x,y], ...]."),
                ResultSchema(name="num_keypoints", type="int", description="Number of keypoints extracted."),
            ],
            dependencies=["camera"],
        )

    def execute(self, rgb_frame, max_features: int = 500, **kwargs) -> Dict[str, Any]:
        try:
            import cv2
            img = np.array(rgb_frame, dtype=np.uint8)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            orb = cv2.ORB_create(nfeatures=max_features)
            kp, desc = orb.detectAndCompute(gray, None)

            keypoints = [[float(k.pt[0]), float(k.pt[1])] for k in kp] if kp else []

            return {
                "success": True,
                "message": f"Extracted {len(keypoints)} ORB features.",
                "keypoints": keypoints,
                "num_keypoints": len(keypoints),
            }
        except Exception as exc:
            logger.exception("[FeatureExtractorSkill] Failed")
            return {"success": False, "message": str(exc)}
