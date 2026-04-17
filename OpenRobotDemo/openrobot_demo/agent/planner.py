"""LLM-based task planner with ReAct-style skill orchestration."""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

# Load .env from OpenRobotDemo root
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是一个快速的具身机器人任务规划器。
可用技能：
- camera_capture(return_depth:bool) 拍摄RGB（可选深度）图像
- arm_state_reader(fields:list) 读取机械臂关节位置与末端位姿
- vision_3d_estimator(rgb_frame,target_name:str) 检测目标并估计3D位姿
- grasp_point_predictor(object_pose_base:list,object_type:str) 预测抓取位姿与预抓取位姿
- arm_motion_executor(command_type:str,target_values,speed:float) 控制机械臂或夹爪运动

arm_motion_executor 的 cartesian 命令可使用占位符：PRE_GRASP（预抓取）、GRASP（抓取）、LIFT（提升）、PLACE（放置）、RETREAT（撤退）。

要求：
- thought 必须用中文写，不超过一句话
- 只输出纯 JSON，不要 markdown，不要多余文字
- 格式：{"thought":"...","plan":[{"skill":"...","args":{}}]}
"""

REACT_SYSTEM_PROMPT = """你是一个快速的具身机器人任务规划器。每次只能调用一个技能。
可用技能：
- camera_capture(return_depth:bool) 拍摄RGB（可选深度）图像
- arm_state_reader(fields:list) 读取机械臂关节位置与末端位姿
- vision_3d_estimator(rgb_frame,target_name:str) 检测目标并估计3D位姿
- grasp_point_predictor(object_pose_base:list,object_type:str) 预测抓取位姿与预抓取位姿
- arm_motion_executor(command_type:str,target_values,speed:float) 控制机械臂或夹爪运动

arm_motion_executor 的 command_type 只能是 "joint"、"cartesian"、"gripper" 三者之一。
cartesian 命令的 target_values 可使用占位符：PRE_GRASP（预抓取）、GRASP（抓取）、LIFT（提升）、PLACE（放置）、RETREAT（撤退）。

规则：
- thought 必须用中文写，不超过一句话
- 复用上下文中已有的信息，除非必要不要重复拍照
- 只输出纯 JSON，不要 markdown，不要多余文字
- 格式：{"thought":"...","action":"skill_call","skill":"...","args":{}} 或 {"thought":"任务完成","action":"finish","result":"..."}
"""


class LLMPlanner:
    def __init__(self, model: str = "kimi-latest",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("KIMI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or "https://api.kimi.com/coding/v1"
        self._client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    default_headers={"User-Agent": "KimiCLI/1.30.0"}
                )
            except Exception as e:
                logger.warning(f"LLMPlanner client init failed: {e}")

    # ------------------------------------------------------------------
    # Legacy static-plan API
    # ------------------------------------------------------------------
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
                temperature=0.1,
                max_tokens=512,
                timeout=15,
            )
            text = self._strip_fences(response.choices[0].message.content.strip())
            data = json.loads(text)
            return data.get("plan", [])
        except Exception as e:
            logger.error(f"LLM planning failed: {e}")
            return self._mock_plan(instruction)

    # ------------------------------------------------------------------
    # ReAct iterative API
    # ------------------------------------------------------------------
    def start_task(self, instruction: str):
        """Reset conversation state for a new task."""
        self._instruction = instruction
        self._messages = [{"role": "system", "content": REACT_SYSTEM_PROMPT}]
        self._mock_plan_steps = self._mock_plan(instruction)
        self._mock_idx = 0
        self._turn = 0

    def next_action(self, state_summary: str = "") -> Dict[str, Any]:
        """Ask the LLM for the next single action given current state."""
        if self._client is None:
            return self._next_mock_action()

        if self._turn == 0:
            content = f"Task: {self._instruction}\n\nWhat is your first action? Output pure JSON only."
        else:
            content = (
                f"Current state:\n{state_summary}\n\n"
                "What is your next action? Output pure JSON only."
            )

        self._messages.append({"role": "user", "content": content})
        self._turn += 1

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=self._messages,
                temperature=0.1,
                max_tokens=512,
                timeout=12,
            )
            text = self._strip_fences(response.choices[0].message.content.strip())
            if not text:
                logger.warning("LLM returned empty response.")
                return self._next_mock_action()
            data = json.loads(text)
            self._messages.append({"role": "assistant", "content": json.dumps(data)})
            return data
        except Exception as e:
            logger.error(f"LLM ReAct call failed: {e}")
            return self._next_mock_action()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _next_mock_action(self) -> Dict[str, Any]:
        if self._mock_idx < len(self._mock_plan_steps):
            step = self._mock_plan_steps[self._mock_idx]
            self._mock_idx += 1
            return {"thought": "Using mock fallback.", "action": "skill_call", **step}
        return {"thought": "Mock plan finished.", "action": "finish", "result": "Done."}

    @staticmethod
    def _strip_fences(text: str) -> str:
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def _mock_plan(self, instruction: str) -> List[Dict[str, Any]]:
        """Deterministic fallback plan for common tasks."""
        instr = instruction.lower()

        # Fabric manipulation demo (3-day task)
        if any(k in instr for k in ["布料", "套", "支撑板", "fabric", "cloth", "tube"]):
            return [
                {"skill": "camera_capture", "args": {"return_depth": True}},
                {"skill": "vision_3d_estimator", "args": {"rgb_frame": "rgb_frame", "target_name": "筒状布料", "end_effector_pose": "end_effector_pose"}},
                {"skill": "fabric_manipulation", "args": {"operation": "pinch_edge", "fabric_center": "object_pose_base", "fabric_diameter_m": 0.08}},
                {"skill": "fabric_manipulation", "args": {"operation": "lift", "height_m": 0.10}},
                {"skill": "vision_3d_estimator", "args": {"rgb_frame": "rgb_frame", "target_name": "铝合金支撑板", "end_effector_pose": "end_effector_pose"}},
                {"skill": "fabric_manipulation", "args": {"operation": "insert", "plate_center": "plate_pose_base", "plate_height_m": 0.05, "insert_depth_m": 0.06}},
                {"skill": "fabric_manipulation", "args": {"operation": "hold_wait", "wait_seconds": 5.0}},
                {"skill": "fabric_manipulation", "args": {"operation": "withdraw", "lift_height_m": 0.10}},
            ]

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
