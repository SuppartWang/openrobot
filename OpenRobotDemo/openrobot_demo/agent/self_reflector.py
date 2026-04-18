"""SelfReflector: analyzes execution failures and proposes recovery strategies.

When a skill execution fails, the SelfReflector:
1. Analyzes the failure context (what failed, why, what was the state)
2. Queries relevant experiences for similar failures
3. Decides whether to: retry with adjusted params / replan / give up
4. Optionally uses LLM for deeper causal analysis
"""

import json
import logging
from typing import Any, Dict, List, Optional

from openrobot_demo.agent.bdi import Reflection

logger = logging.getLogger(__name__)


_SELF_REFLECTION_SYSTEM_PROMPT = """你是一个机器人执行反思专家。

你的任务：分析一次机器人技能执行失败的原因，并给出改进策略。

分析维度：
1. 失败类型：传感器错误？运动执行错误？规划错误？环境变化？
2. 根因：是参数问题（如速度太快）？是环境条件不满足（如物体被遮挡）？是硬件限制（如 IK 无解）？
3. 可恢复性：这个失败是否可以通过调整参数重试？是否需要重新感知？是否需要完全重新规划？

输出必须严格按以下 JSON 格式：
{
  "analysis": "用中文写失败原因分析，不超过两句话",
  "failure_type": "sensor_error|motion_error|planning_error|environment_change|unknown",
  "should_retry": true|false,
  "adjusted_params": {"参数名": "新值", ...},
  "should_replan": true|false,
  "new_strategy": "如果需要重新规划，新的策略是什么（一句话）",
  "should_give_up": false
}

注意：should_give_up 几乎永远应该是 false。机器人应该不断尝试直到成功或用户干预。
"""


class SelfReflector:
    """Reflect on execution failures and propose recovery actions."""

    def __init__(self, client=None, model: str = "kimi-latest",
                 experience_retriever=None):
        self._client = client
        self._model = model
        self._experience_retriever = experience_retriever

    def reflect(
        self,
        skill_name: str,
        skill_args: Dict[str, Any],
        failure_message: str,
        state_summary: str,
        execution_history: List[Dict[str, Any]],
    ) -> Reflection:
        """Analyze a failure and return a Reflection with recovery strategy."""

        # 1. Try experience-based reflection first
        exp_reflection = self._reflect_from_experience(skill_name, failure_message)
        if exp_reflection is not None:
            logger.info("[SelfReflector] Experience-based recovery: %s", exp_reflection.analysis)
            return exp_reflection

        # 2. Try LLM-based reflection
        if self._client is not None:
            try:
                return self._reflect_with_llm(skill_name, skill_args, failure_message, state_summary, execution_history)
            except Exception as exc:
                logger.error("[SelfReflector] LLM reflection failed: %s", exc)

        # 3. Fallback: rule-based reflection
        return self._reflect_rule_based(skill_name, failure_message)

    def _reflect_from_experience(self, skill_name: str, failure_message: str) -> Optional[Reflection]:
        """Check experience library for similar failures with recovery strategies."""
        if self._experience_retriever is None:
            return None

        # Query for experiences with the same skill that had human_feedback about failures
        experiences = self._experience_retriever.retrieve(
            task_intent="",
            target_object_type="",
            action_type=skill_name,
            top_k=3,
        )

        for exp in experiences:
            feedback = (exp.human_feedback or "").lower()
            # If human feedback mentions failure recovery keywords
            if any(kw in feedback for kw in ["失败", "错误", "重试", "回退", "阻力", "失败", "error", "retry", "back off"]):
                # Extract suggested params from the experience
                adjusted = {}
                if "速度" in feedback or "speed" in feedback:
                    adjusted["speed"] = exp.max_velocity_m_s * 0.5
                if "力" in feedback or "force" in feedback:
                    adjusted["contact_force_threshold_n"] = exp.contact_force_threshold_n * 0.8

                return Reflection(
                    analysis=f"根据经验 '{exp.experience_id[:8]}' 的建议: {exp.human_feedback}",
                    should_retry=True,
                    adjusted_params=adjusted,
                    should_replan=False,
                )

        return None

    def _reflect_with_llm(
        self,
        skill_name: str,
        skill_args: Dict[str, Any],
        failure_message: str,
        state_summary: str,
        execution_history: List[Dict[str, Any]],
    ) -> Reflection:
        """Use LLM for deep causal analysis of failure."""

        trace_text = json.dumps(execution_history[-3:], ensure_ascii=False) if execution_history else "无历史记录"

        user_prompt = (
            f"失败技能：{skill_name}\n"
            f"技能参数：{json.dumps(skill_args, ensure_ascii=False)}\n"
            f"错误信息：{failure_message}\n"
            f"当前状态：{state_summary}\n"
            f"最近执行历史：{trace_text}\n\n"
            f"请分析失败原因并给出改进策略（纯 JSON）："
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SELF_REFLECTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=512,
            timeout=15,
        )
        text = response.choices[0].message.content.strip()
        text = self._strip_fences(text)
        data = json.loads(text)

        return Reflection(
            analysis=data.get("analysis", ""),
            should_retry=data.get("should_retry", False),
            adjusted_params=data.get("adjusted_params", {}),
            should_replan=data.get("should_replan", False),
            new_strategy=data.get("new_strategy", ""),
            should_give_up=data.get("should_give_up", False),
        )

    def _reflect_rule_based(self, skill_name: str, failure_message: str) -> Reflection:
        """Simple rule-based fallback for common failure patterns."""
        msg_lower = failure_message.lower()

        # IK failures → replan with different approach
        if any(k in msg_lower for k in ["ik failed", "unreachable", "inverse kinematics", "ik 失败", "不可达"]):
            return Reflection(
                analysis="逆运动学求解失败，目标位姿不可达。",
                should_retry=False,
                should_replan=True,
                new_strategy="尝试不同的接近角度或使用关节空间路径",
            )

        # Safety check failures → retry with adjusted target
        if any(k in msg_lower for k in ["safety", "out of limits", "workspace", "安全", "限位", "工作空间"]):
            return Reflection(
                analysis="目标超出安全范围。",
                should_retry=True,
                adjusted_params={"target_offset": [0, 0, 0.02]},  # move up 2cm
                should_replan=False,
            )

        # Gripper / contact failures → retry with more force or different grasp
        if any(k in msg_lower for k in ["gripper", "grasp", "slip", "夹爪", "抓取", "滑动"]):
            return Reflection(
                analysis="抓取失败，可能需要更大的夹持力或不同的抓取点。",
                should_retry=True,
                adjusted_params={"force": 0.8},
                should_replan=False,
            )

        # Vision failures → re-sense
        if any(k in msg_lower for k in ["detect", "not found", "vision", "检测", "找不到"]):
            return Reflection(
                analysis="视觉检测失败，可能需要重新拍摄或调整相机角度。",
                should_retry=True,
                adjusted_params={"return_depth": True},
                should_replan=False,
            )

        # Default: retry once with same params
        return Reflection(
            analysis=f"执行失败: {failure_message}。尝试重试。",
            should_retry=True,
            should_replan=False,
        )

    @staticmethod
    def _strip_fences(text: str) -> str:
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
