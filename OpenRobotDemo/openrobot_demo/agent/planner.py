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
from typing import List, Dict, Any, Optional, Callable

from dotenv import load_dotenv

# Load .env from OpenRobotDemo root
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# PromptEngine: dynamic prompt assembly
# ------------------------------------------------------------------
class PromptEngine:
    """Assemble LLM prompts dynamically from components.

    Design inspired by:
    - Inner Monologue (Huang et al., 2023): closed-loop feedback after each step
    - MLDT (Wu et al., 2024): multi-level task decomposition
    - SayCan (Brohan et al., 2023): affordance-grounded planning
    """

    SYSTEM_HEADER = """你是 OpenRobot，一台配备双臂 YHRG S1 机械臂、平行夹爪和 RealSense 视觉系统的具身智能机器人。

你的核心能力：
1. 理解用户高层指令 → 分解为子目标 → 选择并调用机器人技能
2. 每步执行后评估结果，根据环境反馈调整计划（闭环重规划）
3. 识别任务进展，判断何时完成、何时需要重试或改变策略

可用技能类型：
- camera_capture: 拍摄 RGB/深度图像，获取环境信息
- arm_state_reader: 读取机械臂关节位置、末端位姿
- vision_3d_estimator: 从图像中估计目标物体的 3D 位姿
- grasp_point_predictor: 预测最佳抓取点
- arm_motion_executor: 执行关节/笛卡尔空间运动
- fabric_manipulation: 布料操作（捏合、提起、套入、保持、取下）
- vla_policy_executor: 端到端视觉-语言-动作策略

规划规则（基于 Inner Monologue 闭环反馈）：
1. 任务分解：先将复杂指令分解为 2-4 个子目标，按依赖顺序执行
2. 每步思考：用中文一句话说明当前状态和下一步意图
3. 环境感知优先：执行动作前，如果环境状态不确定，先拍照确认
4. 闭环检查：每步执行后，系统会反馈执行结果。你必须据此判断：
   - 成功 → 继续下一步
   - 失败/不确定 → 重新感知环境或调整策略
   - 场景变化 → 重新分解任务
5. 安全约束：
   - 运动速度不超过 0.8
   - 关节运动范围限制在 [-170°, 170°]（基座）等安全范围内
   - 碰撞检测启用，避免自碰撞
6. 只输出纯 JSON，不要 markdown，不要多余文字

输出格式（ReAct 风格，每轮一步）：
{"thought":"...","action":"skill_call","skill":"...","args":{}}
或 {"thought":"...","action":"replan","reason":"..."}
或 {"thought":"任务全部完成","action":"finish","result":"..."}
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

    def build_user_prompt(
        self,
        instruction: str,
        state_summary: str = "",
        turn: int = 0,
        feedback_history: Optional[List[Dict[str, Any]]] = None,
        task_progress: Optional[str] = None,
    ) -> str:
        """Build user prompt with Inner Monologue style closed-loop feedback.

        Args:
            instruction: Original user instruction
            state_summary: Current robot/world state
            turn: Conversation turn number
            feedback_history: List of {step, skill, result, observation} dicts
            task_progress: Textual description of task progress
        """
        parts = []

        if turn == 0:
            parts.append(f"【任务指令】{instruction}")
            parts.append("\n【任务分解】请先将此任务分解为 2-4 个按依赖顺序排列的子目标，然后从第一个子目标开始逐步执行。")
        else:
            parts.append(f"【任务指令】{instruction}")

        if task_progress:
            parts.append(f"\n【任务进展】{task_progress}")

        if feedback_history:
            parts.append("\n【执行历史 / 内心独白】")
            for i, fb in enumerate(feedback_history[-8:], 1):  # Last 8 steps
                skill = fb.get("skill", "?")
                result = "成功" if fb.get("success") else "失败"
                msg = fb.get("message", "")
                obs = fb.get("observation", "")
                parts.append(f"  步骤{i}: [{skill}] → {result} | {msg}")
                if obs:
                    parts.append(f"    观察: {obs}")

        if state_summary:
            parts.append(f"\n【当前状态】\n{state_summary}")

        parts.append("\n【思考与行动】基于上述历史和当前状态，请输出下一步（纯 JSON）：")
        return "\n".join(parts)


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
        self._observers: List[Callable] = []

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
    def add_observer(self, observer: Callable):
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: Callable):
        if observer in self._observers:
            self._observers.remove(observer)

    def _notify(self, event_type: str, data: Dict[str, Any]):
        for obs in self._observers:
            try:
                obs(event_type, data)
            except Exception as exc:
                logger.debug("Planner observer notify failed: %s", exc)

    def start_task(self, instruction: str):
        """Reset conversation state for a new task."""
        self._instruction = instruction
        self._setup_prompts(instruction)
        self._system_prompt = self._prompt_engine.build_system_prompt()
        self._messages = [{"role": "system", "content": self._system_prompt}]
        self._mock_plan_steps = self._mock_plan(instruction)
        self._mock_idx = 0
        self._turn = 0
        self._notify("llm_start", {"instruction": instruction})

    def next_action(
        self,
        state_summary: str = "",
        feedback_history: Optional[List[Dict[str, Any]]] = None,
        task_progress: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ask the LLM for the next single action given current state.

        Implements Inner Monologue style closed-loop planning:
        - feedback_history: execution results from previous steps
        - task_progress: textual description of overall task progress
        """
        if self._client is None:
            return self._next_mock_action()

        content = self._prompt_engine.build_user_prompt(
            self._instruction,
            state_summary=state_summary,
            turn=self._turn,
            feedback_history=feedback_history,
            task_progress=task_progress,
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
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # Try to extract JSON from malformed response
                data = self._extract_json_from_text(text)
                if data is None:
                    logger.warning("LLM returned non-JSON response: %s", text[:200])
                    return self._next_mock_action()

            self._messages.append({"role": "assistant", "content": json.dumps(data, ensure_ascii=False)})
            self._notify("llm_react", {
                "turn": self._turn,
                "thought": data.get("thought", ""),
                "action": data.get("action", ""),
                "skill": data.get("skill", ""),
                "args": data.get("args", {}),
                "raw": data,
            })
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
            plan = data.get("plan", [])
            self._notify("llm_plan", {"instruction": instruction, "plan": plan})
            return plan
        except Exception as e:
            logger.error(f"LLM planning failed: {e}")
            plan = self._mock_plan(instruction)
            self._notify("llm_plan", {"instruction": instruction, "plan": plan, "fallback": True})
            return plan

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
        """Try to extract a JSON object from text that may contain extra content."""
        # Find the first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            return None

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

        # Sub-goal specific plans for fabric manipulation
        if any(k in instr for k in ["拍摄", "rgb", "相机", "拍照", "capture"]):
            return [{"skill": "camera_capture", "args": {"return_depth": True}}]

        if any(k in instr for k in ["检测", "定位", "3d", "vision", "estimat"]):
            target = "筒状布料" if "布料" in instr else "目标物体"
            return [
                {"skill": "camera_capture", "args": {"return_depth": True}},
                {"skill": "vision_3d_estimator", "args": {"rgb_frame": "rgb_frame", "target_name": target, "end_effector_pose": "end_effector_pose"}},
            ]

        if any(k in instr for k in ["捏合", "pinch", "抓取", "夹取"]):
            return [
                {"skill": "fabric_manipulation", "args": {"operation": "pinch_edge", "fabric_center": "object_pose_base", "fabric_diameter_m": 0.08}},
            ]

        if any(k in instr for k in ["提起", "lift", "举高", "提升"]):
            return [
                {"skill": "fabric_manipulation", "args": {"operation": "lift", "height_m": 0.10}},
            ]

        if any(k in instr for k in ["套入", "insert", "下降", "套上", "覆盖", "上方", "对准", "移动"]):
            return [
                {"skill": "fabric_manipulation", "args": {"operation": "insert", "plate_center": "plate_pose_base", "plate_height_m": 0.05, "insert_depth_m": 0.06}},
            ]

        if any(k in instr for k in ["保持", "hold", "等待", "wait"]):
            return [
                {"skill": "fabric_manipulation", "args": {"operation": "hold_wait", "wait_seconds": 5.0}},
            ]

        if any(k in instr for k in ["取下", "withdraw", "取出", "释放", "放开"]):
            return [
                {"skill": "fabric_manipulation", "args": {"operation": "withdraw", "lift_height_m": 0.10}},
            ]

        # Full fabric manipulation demo (3-day task)
        if any(k in instr for k in ["布料", "套", "支撑板", "fabric", "cloth", "tube"]):
            return [
                {"skill": "camera_capture", "args": {"return_depth": True}},
                {"skill": "vision_3d_estimator", "args": {"rgb_frame": "rgb_frame", "target_name": "筒状布料", "end_effector_pose": "end_effector_pose"}},
                {"skill": "fabric_manipulation", "args": {"operation": "pinch_edge", "fabric_center": "object_pose_base", "fabric_diameter_m": 0.08}},
                {"skill": "fabric_manipulation", "args": {"operation": "lift", "height_m": 0.10}},
                {"skill": "vision_3d_estimator", "args": {"rgb_frame": "rgb_frame", "target_name": "水平支撑板", "end_effector_pose": "end_effector_pose"}},
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
