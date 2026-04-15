"""LLM-based task planner with ReAct-style skill orchestration."""

import os
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are the high-level planner for an embodied robot assistant.
The robot has the following skills:

1. camera_capture: captures RGB (and optional depth) from the camera.
   Args: return_depth (bool)

2. arm_state_reader: reads the robot arm's current joint positions and end-effector pose.
   Args: fields (list of strings, e.g., ["pos"])

3. vision_3d_estimator: detects an object in the image and computes its 3D pose.
   Args: rgb_frame, target_name (str), depth_frame (optional), camera_intrinsics (optional)

4. grasp_point_predictor: given an object's 3D pose, predicts a grasp pose and pre-grasp approach pose.
   Args: object_pose_base (list), object_type (str)

5. arm_motion_executor: moves the arm or controls the gripper.
   Args: command_type ("joint" | "cartesian" | "gripper"), target_values (list), speed (float)

Your job is to break the user's instruction into a sequence of skill calls.
You must output a JSON object with exactly these fields:
- thought: string, your reasoning
- plan: list of objects, each with {skill: string, args: object}

Example:
User: "Pick up the red box"
Output:
{
  "thought": "I need to see the scene, find the red box, compute its 3D pose, predict a grasp point, move to pre-grasp, move to grasp, close gripper, and lift up.",
  "plan": [
    {"skill": "camera_capture", "args": {"return_depth": true}},
    {"skill": "vision_3d_estimator", "args": {"target_name": "red box"}},
    {"skill": "grasp_point_predictor", "args": {"object_type": "box"}},
    {"skill": "arm_motion_executor", "args": {"command_type": "cartesian", "target_values": [PRE_GRASP_POSE], "speed": 0.8}},
    {"skill": "arm_motion_executor", "args": {"command_type": "cartesian", "target_values": [GRASP_POSE], "speed": 0.5}},
    {"skill": "arm_motion_executor", "args": {"command_type": "gripper", "target_values": [0.0, 0.5]}},
    {"skill": "arm_motion_executor", "args": {"command_type": "cartesian", "target_values": [LIFT_POSE], "speed": 0.5}}
  ]
}

Rules:
- Only use the skills listed above.
- Do not include any markdown code fences.
- Return pure JSON only.
"""


class LLMPlanner:
    def __init__(self, model: str = "gpt-4o-mini",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self._client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception as e:
                logger.warning(f"LLMPlanner client init failed: {e}")

    def plan(self, instruction: str) -> List[Dict[str, Any]]:
        if self._client is None:
            logger.warning("No LLM API key available. Using mock planner.")
            return self._mock_plan(instruction)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"User instruction: {instruction}"},
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            text = response.choices[0].message.content.strip()
            # Strip fences if present
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            data = json.loads(text)
            return data.get("plan", [])
        except Exception as e:
            logger.error(f"LLM planning failed: {e}")
            return self._mock_plan(instruction)

    def _mock_plan(self, instruction: str) -> List[Dict[str, Any]]:
        """Deterministic fallback plan for common pick-and-place tasks."""
        instr = instruction.lower()
        if any(k in instr for k in ["pick", "grab", "抓", "拿"]):
            return [
                {"skill": "camera_capture", "args": {"return_depth": True}},
                {"skill": "vision_3d_estimator", "args": {"rgb_frame": "rgb_frame", "depth_frame": "depth_frame", "target_name": "target object", "end_effector_pose": "end_effector_pose"}},
                {"skill": "grasp_point_predictor", "args": {"object_pose_base": "object_pose_base", "object_type": "box"}},
                {"skill": "arm_motion_executor", "args": {"command_type": "cartesian", "target_values": "PRE_GRASP", "speed": 0.8}},
                {"skill": "arm_motion_executor", "args": {"command_type": "cartesian", "target_values": "GRASP", "speed": 0.5}},
                {"skill": "arm_motion_executor", "args": {"command_type": "gripper", "target_values": [0.0, 0.5]}},
                {"skill": "arm_motion_executor", "args": {"command_type": "cartesian", "target_values": "LIFT", "speed": 0.5}},
            ]
        elif any(k in instr for k in ["place", "放", "放置"]):
            return [
                {"skill": "arm_motion_executor", "args": {"command_type": "cartesian", "target_values": "PLACE", "speed": 0.5}},
                {"skill": "arm_motion_executor", "args": {"command_type": "gripper", "target_values": [1.0, 0.5]}},
                {"skill": "arm_motion_executor", "args": {"command_type": "cartesian", "target_values": "RETREAT", "speed": 0.8}},
            ]
        else:
            return [
                {"skill": "camera_capture", "args": {"return_depth": True}},
                {"skill": "arm_state_reader", "args": {"fields": ["pos"]}},
            ]
