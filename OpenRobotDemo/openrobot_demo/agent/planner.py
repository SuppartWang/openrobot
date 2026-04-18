"""LLM-based task planner with ReAct-style skill orchestration.

Enhancements over v1:
- Experience-aware planning: retrieves relevant experiences and injects them into prompts
- Schema-aware skill descriptions: auto-generates tool descriptions from SkillInterface schemas
- Dynamic prompt assembly via PromptEngine
- Few-shot example injection from skill schemas
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

# Load .env from OpenRobotDemo root
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# PromptEngine: dynamic prompt assembly
# ------------------------------------------------------------------
class PromptEngine:
    """Assemble LLM prompts dynamically from components."""

    SYSTEM_HEADER = """你是一个快速的具身机器人任务规划器。你的目标是将用户指令分解为一系列可执行的机器人技能调用。

规则：
1. thought 必须用中文写，不超过一句话。
2. 每次只能调用一个技能。
3. 复用上下文中已有的信息，除非必要不要重复拍照。
4. 如果之前有类似任务的经验，优先参考经验中的参数和建议。
5. 只输出纯 JSON，不要 markdown，不要多余文字。

输出格式：
{"thought":"...","action":"skill_call","skill":"...","args":{}}
或 {"thought":"任务完成","action":"finish","result":"..."}
"""

    def __init__(self):
        self._skill_descriptions: str = ""
        self._experience_summary: str = ""
        self._few_shots: List[Dict[str, Any]] = []

    def set_skill_descriptions(self, text: str):
        self._skill_descriptions = text

    def set_experience_summary(self, text: str):
        self._experience_summary = text

    def set_few_shots(self, examples: List[Dict[str, Any]]):
        self._few_shots = examples

    def build_system_prompt(self) -> str:
        parts = [self.SYSTEM_HEADER]
        if self._skill_descriptions:
            parts.append("\n可用技能：\n" + self._skill_descriptions)
        if self._experience_summary:
            parts.append("\n相关经验：\n" + self._experience_summary)
        if self._few_shots:
            parts.append("\n示例：")
            for ex in self._few_shots:
                parts.append(json.dumps(ex, ensure_ascii=False))
        return "\n".join(parts)

    def build_user_prompt(self, instruction: str, state_summary: str = "", turn: int = 0) -> str:
        if turn == 0:
            content = f"任务指令：{instruction}\n\n"
        else:
            content = f"当前状态：\n{state_summary}\n\n"

        content += "请输出你的下一步动作（纯 JSON）："
        return content


# ------------------------------------------------------------------
# LLMPlanner
# ------------------------------------------------------------------
class LLMPlanner:
    def __init__(
        self,
        model: str = "kimi-latest",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        experience_retriever=None,
        skill_router=None,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("KIMI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or "https://api.kimi.com/coding/v1"
        self._client = None
        self._experience_retriever = experience_retriever
        self._skill_router = skill_router
        self._prompt_engine = PromptEngine()

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
    # Setup: skill descriptions + experience injection
    # ------------------------------------------------------------------
    def _setup_prompts(self, instruction: str):
        """Build prompts with skill schemas and relevant experiences."""
        # Skill descriptions from router
        if self._skill_router is not None:
            skill_text = self._skill_router.get_skill_schemas_text()
            self._prompt_engine.set_skill_descriptions(skill_text)

        # Experience injection
        exp_text = self._retrieve_experiences(instruction)
        if exp_text:
            self._prompt_engine.set_experience_summary(exp_text)

        # Few-shot examples from skill schemas
        few_shots = self._collect_few_shots()
        if few_shots:
            self._prompt_engine.set_few_shots(few_shots)

    def _retrieve_experiences(self, instruction: str) -> str:
        """Retrieve relevant experiences and format as natural language."""
        if self._experience_retriever is None:
            return ""

        # Try to infer action types from instruction keywords
        action_hints = []
        keywords = {
            "pinch": "pinch", "夹": "pinch", "捏": "pinch",
            "lift": "lift", "提": "lift", "举": "lift",
            "insert": "insert", "套": "insert", "插入": "insert",
            "withdraw": "withdraw", "取": "withdraw", "拔": "withdraw",
            "grasp": "grasp", "抓": "grasp", "拿": "grasp",
        }
        instr_lower = instruction.lower()
        for kw, action in keywords.items():
            if kw in instr_lower:
                action_hints.append(action)

        if not action_hints:
            action_hints = ["grasp", "lift", "place"]

        lines = []
        for action_type in action_hints:
            exps = self._experience_retriever.retrieve(
                task_intent=instruction,
                target_object_type="",
                action_type=action_type,
                top_k=2,
            )
            for exp in exps:
                lines.append(f"- 动作类型: {exp.action_type}")
                lines.append(f"  目标: {exp.task_intent}")
                lines.append(f"  参数: pre_contact_offset={exp.pre_contact_offset}, approach_angle={exp.approach_angle_deg}°")
                lines.append(f"  速度: max_velocity={exp.max_velocity_m_s}m/s, compliance={exp.compliance_stiffness}N/m")
                if exp.human_feedback:
                    lines.append(f"  经验建议: {exp.human_feedback}")
                lines.append("")

        return "\n".join(lines) if lines else ""

    def _collect_few_shots(self) -> List[Dict[str, Any]]:
        """Collect few-shot examples from registered skill schemas."""
        if self._skill_router is None:
            return []

        shots = []
        for name, skill in self._skill_router._skills.items():
            for ex in skill.schema.examples:
                shots.append({
                    "skill": name,
                    "input": ex.get("input", {}),
                    "output": ex.get("output", {}),
                })
        return shots[:5]  # Limit to avoid token overflow

    # ------------------------------------------------------------------
    # ReAct iterative API
    # ------------------------------------------------------------------
    def start_task(self, instruction: str):
        """Reset conversation state for a new task."""
        self._instruction = instruction
        self._setup_prompts(instruction)
        self._system_prompt = self._prompt_engine.build_system_prompt()
        self._messages = [{"role": "system", "content": self._system_prompt}]
        self._mock_plan_steps = self._mock_plan(instruction)
        self._mock_idx = 0
        self._turn = 0

    def next_action(self, state_summary: str = "") -> Dict[str, Any]:
        """Ask the LLM for the next single action given current state."""
        if self._client is None:
            return self._next_mock_action()

        content = self._prompt_engine.build_user_prompt(
            self._instruction, state_summary, self._turn
        )
        self._messages.append({"role": "user", "content": content})
        self._turn += 1

        # Use function calling if skill router provides tool descriptions
        tools = None
        if self._skill_router is not None:
            tools = self._skill_router.get_tool_descriptions()

        try:
            kwargs = {
                "model": self.model,
                "messages": self._messages,
                "temperature": 0.1,
                "max_tokens": 512,
                "timeout": 12,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = self._client.chat.completions.create(**kwargs)
            msg = response.choices[0].message

            # Handle tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tc = msg.tool_calls[0]
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                data = {
                    "thought": f"调用技能 {tc.function.name}",
                    "action": "skill_call",
                    "skill": tc.function.name,
                    "args": args,
                }
                self._messages.append({"role": "assistant", "content": json.dumps(data)})
                return data

            # Handle plain text JSON response
            text = self._strip_fences(msg.content.strip()) if msg.content else ""
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
    # Legacy static-plan API
    # ------------------------------------------------------------------
    def plan(self, instruction: str) -> List[Dict[str, Any]]:
        if self._client is None:
            logger.warning("No LLM API key available. Using mock planner.")
            return self._mock_plan(instruction)

        self._setup_prompts(instruction)
        system_prompt = self._prompt_engine.build_system_prompt()

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User instruction: {instruction}\n\n请输出完整的执行计划（纯 JSON）。"},
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
