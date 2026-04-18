"""VLM Perception Cognition Engine.

This module uses a Vision-Language Model (VLM) as a SIGNAL PROCESSOR, not a
决策器 (decision maker). It receives raw visual input (RGB, optionally depth)
and outputs structured cognitive parameters about the environment:

- Scene description (natural language)
- Object list with semantic attributes (type, color, estimated position, state)
- Spatial relations between objects
- Anomaly detection (unexpected objects, hazards)
- Affordance predictions (what actions are possible on each object)

All outputs are wrapped as PerceptionData(modality="vlm_cognition") and fed
into the WorldModel via WorldModel.ingest().

Design principle: VLM sees → understands → describes. It does NOT decide.
The BDI Agent decides what to do with these descriptions.
"""

import base64
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from openrobot_demo.sensors.base import SensorChannel, PerceptionData

logger = logging.getLogger(__name__)


try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception as exc:
    _OPENAI_AVAILABLE = False
    logger.debug("[VLMCognitionEngine] openai not available: %s", exc)


# ------------------------------------------------------------------
# Structured output schemas for VLM prompts
# ------------------------------------------------------------------
_VLM_COGNITION_SYSTEM_PROMPT = """你是一个精确的机器人视觉认知系统。

你的任务：分析输入图像，输出结构化的环境认知参数。你只负责"看见并描述"，不做任何决策。

输出必须严格按以下 JSON 格式，不要 markdown，不要解释：

{
  "scene_description": "一句话描述整个场景",
  "objects": [
    {
      "id": "obj_0",
      "type": "cube|cylinder|sphere|cloth|plate|...",
      "color": "red|green|blue|yellow|white|silver|...",
      "estimated_position": [x, y, z],
      "state": "stable|unstable|moving|deformed|...",
      "confidence": 0.0~1.0
    }
  ],
  "spatial_relations": [
    {"subject": "obj_0", "relation": "on|left_of|right_of|above|under|near", "object": "obj_1"}
  ],
  "anomalies": [
    {"description": "描述异常", "severity": "low|medium|high"}
  ],
  "affordances": [
    {"object_id": "obj_0", "possible_actions": ["grasp", "push", "lift", ...]}
  ]
}

位置坐标 [x, y, z] 是相对于机器人基座的大致估计（米）。
如果无法估计深度，将 z 设为 0.0 并在 state 中注明 "depth_unknown"。
如果图像中没有明显物体，objects 为空数组 []。
"""


class VLMCognitionEngine:
    """Process visual input through a VLM and emit structured cognition."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen-vl-max-latest",
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        self.base_url = base_url
        self.model = model
        self._client = None
        if _OPENAI_AVAILABLE and self.api_key:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            logger.warning("[VLMCognitionEngine] VLM client not available or API key missing.")

    def is_available(self) -> bool:
        return _OPENAI_AVAILABLE and self._client is not None

    def analyze(self, rgb_frame: np.ndarray, target_query: Optional[str] = None) -> Dict[str, Any]:
        """Analyze an RGB image and return structured cognition.

        Args:
            rgb_frame: RGB image array (H, W, 3)
            target_query: Optional specific query, e.g. "检测支撑板上方是否有障碍物"

        Returns:
            Structured cognition dict (see _VLM_COGNITION_SYSTEM_PROMPT)
        """
        if not self.is_available():
            return self._mock_analyze(rgb_frame, target_query)

        try:
            _, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
            b64_image = base64.b64encode(buf.tobytes()).decode("utf-8")

            user_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                },
            ]
            if target_query:
                user_content.append({"type": "text", "text": f"特殊关注：{target_query}\n\n请输出JSON格式的环境认知。"})
            else:
                user_content.append({"type": "text", "text": "请输出JSON格式的环境认知。"})

            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _VLM_COGNITION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            text = response.choices[0].message.content.strip()
            logger.debug("[VLMCognitionEngine] Raw VLM response: %s", text[:500])
            return self._parse_response(text)

        except Exception as exc:
            logger.error("[VLMCognitionEngine] VLM analysis failed: %s", exc)
            return self._mock_analyze(rgb_frame, target_query)

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse VLM text response into structured dict."""
        # Strip markdown fences
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from mixed text
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    data = self._create_fallback(text)
            else:
                data = self._create_fallback(text)

        # Ensure required keys
        for key in ["scene_description", "objects", "spatial_relations", "anomalies", "affordances"]:
            if key not in data:
                data[key] = [] if key != "scene_description" else ""

        return data

    def _create_fallback(self, text: str) -> Dict[str, Any]:
        """Create a minimal fallback when JSON parsing fails."""
        return {
            "scene_description": text[:200] if text else "解析失败",
            "objects": [],
            "spatial_relations": [],
            "anomalies": [{"description": "VLM输出解析失败", "severity": "low"}],
            "affordances": [],
            "raw_vlm_response": text,
            "parse_error": True,
        }

    def _mock_analyze(self, rgb_frame: np.ndarray, target_query: Optional[str] = None) -> Dict[str, Any]:
        """Mock cognition when VLM is unavailable."""
        h, w = rgb_frame.shape[:2]
        return {
            "scene_description": "模拟场景：工作台上有一个测试物体（VLM未接入）。",
            "objects": [
                {
                    "id": "obj_0",
                    "type": "unknown",
                    "color": "unknown",
                    "estimated_position": [0.30, 0.0, 0.05],
                    "state": "mock_cognition",
                    "confidence": 0.5,
                }
            ],
            "spatial_relations": [],
            "anomalies": [],
            "affordances": [
                {"object_id": "obj_0", "possible_actions": ["grasp"]}
            ],
            "mock": True,
        }


# ------------------------------------------------------------------
# SensorChannel wrapper for continuous cognition
# ------------------------------------------------------------------
class VLMCognitionSensor(SensorChannel):
    """Sensor that continuously runs VLM cognition on camera frames."""

    name = "vlm_cognition"

    def __init__(
        self,
        source_id: str = "vlm_cognition",
        camera_source_id: str = "rs_d435i_rgb",
        query: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen-vl-max-latest",
    ):
        self.source_id = source_id
        self.camera_source_id = camera_source_id
        self.query = query
        self._engine = VLMCognitionEngine(api_key=api_key, base_url=base_url, model=model)
        self._last_rgb: Optional[np.ndarray] = None

    def is_available(self) -> bool:
        return self._engine.is_available()

    def set_frame(self, rgb_frame: np.ndarray):
        """Feed a frame from an external camera source."""
        self._last_rgb = rgb_frame

    def capture(self) -> PerceptionData:
        if self._last_rgb is None:
            raise RuntimeError("VLMCognitionSensor: no frame available. Call set_frame() first.")

        cognition = self._engine.analyze(self._last_rgb, target_query=self.query)

        return PerceptionData(
            modality="vlm_cognition",
            source_id=self.source_id,
            timestamp=time.time(),
            payload=cognition,
            spatial_ref="base_frame",
            confidence=1.0,
            metadata={
                "query": self.query,
                "mock": cognition.get("mock", False),
                "parse_error": cognition.get("parse_error", False),
            },
        )


# ------------------------------------------------------------------
# Skill wrapper for on-demand cognition
# ------------------------------------------------------------------
class VLMCognitionSkill:
    """Skill that calls VLM cognition on demand (not a SensorChannel, just a helper).

    Can be registered as a SkillInterface if needed, but typically invoked
    directly by the BDI Agent when it needs specific visual understanding.
    """

    @staticmethod
    def query(rgb_frame: np.ndarray, question: str,
              api_key: Optional[str] = None,
              base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
              model: str = "qwen-vl-max-latest") -> Dict[str, Any]:
        """Ask a specific visual question and get structured answer."""
        engine = VLMCognitionEngine(api_key=api_key, base_url=base_url, model=model)
        return engine.analyze(rgb_frame, target_query=question)
