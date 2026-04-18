"""TaskDecomposer: LLM-based hierarchical goal decomposition.

Takes a natural language instruction and produces a tree of Goals
that the BDI Agent can execute.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from openrobot_demo.agent.bdi import Goal, GoalStatus

logger = logging.getLogger(__name__)


_TASK_DECOMPOSITION_SYSTEM_PROMPT = """你是一个机器人任务分解专家。

你的任务：将用户的自然语言指令分解为一个层次化的目标树。

每个目标必须包含以下字段：
- description: 自然语言描述（必须具体、可执行）
- goal_id: 唯一标识符（简短，如 "g1", "g1_1"）
- preconditions: 执行前必须满足的条件列表（字符串数组）
- completion_criteria: 如何判断目标已完成（字符串数组）
- required_skills: 可能需要的技能名称列表
- estimated_steps: 估计需要的步骤数（整数）
- priority: 优先级（0-10，越大越优先）
- sub_goals: 子目标列表（嵌套结构，叶节点为空数组）

分解原则：
1. 顶层目标对应用户指令的整体意图
2. 中层目标对应主要阶段（如：感知→规划→执行→验证）
3. 叶节点目标对应可直接调用技能的原子操作
4. 尽量保持每个目标的 estimated_steps <= 10
5. 如果某个步骤有明确的先后顺序依赖，用 preconditions 说明

只输出纯 JSON，不要 markdown，不要解释文字。
输出格式必须是一个目标对象（顶层），内部嵌套 sub_goals。
"""


class TaskDecomposer:
    """Decompose a high-level instruction into a Goal tree using LLM."""

    def __init__(self, client=None, model: str = "kimi-latest"):
        self._client = client
        self._model = model

    def decompose(self, instruction: str, available_skills: Optional[List[str]] = None) -> Goal:
        """Decompose instruction into a Goal tree.

        If LLM is unavailable, falls back to rule-based decomposition
        for common task patterns.
        """
        if self._client is not None:
            try:
                return self._decompose_with_llm(instruction, available_skills)
            except Exception as exc:
                logger.error("[TaskDecomposer] LLM decomposition failed: %s", exc)

        logger.warning("[TaskDecomposer] Using rule-based fallback decomposition.")
        return self._decompose_rule_based(instruction)

    def _decompose_with_llm(self, instruction: str, available_skills: Optional[List[str]]) -> Goal:
        skill_text = f"\n可用技能：{', '.join(available_skills)}" if available_skills else ""

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _TASK_DECOMPOSITION_SYSTEM_PROMPT},
                {"role": "user", "content": f"用户指令：{instruction}{skill_text}\n\n请输出目标树（纯 JSON）："},
            ],
            temperature=0.2,
            max_tokens=2048,
            timeout=20,
        )
        text = response.choices[0].message.content.strip()
        text = self._strip_fences(text)
        data = json.loads(text)
        return self._dict_to_goal(data)

    def _decompose_rule_based(self, instruction: str) -> Goal:
        """Rule-based fallback for common tasks."""
        instr_lower = instruction.lower()

        # Fabric manipulation pattern
        if any(k in instr_lower for k in ["布料", "套", "支撑板", "fabric", "cloth", "tube"]):
            return Goal(
                description=instruction,
                sub_goals=[
                    Goal(
                        description="感知环境：定位布料和支撑板",
                        goal_id="g1",
                        preconditions=["相机可用"],
                        completion_criteria=["布料位置已知", "支撑板位置已知"],
                        required_skills=["camera_capture", "vision_3d_estimator", "vlm_cognition"],
                        estimated_steps=3,
                        priority=10,
                        sub_goals=[
                            Goal(description="拍摄RGB图像", goal_id="g1_1", required_skills=["camera_capture"], estimated_steps=1),
                            Goal(description="检测布料3D位置", goal_id="g1_2", preconditions=["RGB图像已获取"], required_skills=["vision_3d_estimator"], estimated_steps=1),
                            Goal(description="检测支撑板3D位置", goal_id="g1_3", preconditions=["RGB图像已获取"], required_skills=["vision_3d_estimator"], estimated_steps=1),
                        ],
                    ),
                    Goal(
                        description="抓取并提起布料",
                        goal_id="g2",
                        preconditions=["布料位置已知", "双臂已使能"],
                        completion_criteria=["布料被双臂提起，高度>10cm"],
                        required_skills=["fabric_manipulation"],
                        estimated_steps=3,
                        priority=9,
                        sub_goals=[
                            Goal(description="双臂捏合布料边缘", goal_id="g2_1", required_skills=["fabric_manipulation"], estimated_steps=1),
                            Goal(description="提起布料", goal_id="g2_2", preconditions=["布料已被捏合"], required_skills=["fabric_manipulation"], estimated_steps=1),
                        ],
                    ),
                    Goal(
                        description="将布料套入支撑板",
                        goal_id="g3",
                        preconditions=["布料已被提起", "支撑板位置已知"],
                        completion_criteria=["布料套入支撑板深度>5cm"],
                        required_skills=["fabric_manipulation"],
                        estimated_steps=2,
                        priority=9,
                        sub_goals=[
                            Goal(description="移动布料到支撑板上方", goal_id="g3_1", required_skills=["fabric_manipulation"], estimated_steps=1),
                            Goal(description="下降套入支撑板", goal_id="g3_2", preconditions=["布料在支撑板上方"], required_skills=["fabric_manipulation"], estimated_steps=1),
                        ],
                    ),
                    Goal(
                        description="保持并等待检测",
                        goal_id="g4",
                        preconditions=["布料已套入支撑板"],
                        completion_criteria=["等待时间已结束"],
                        required_skills=["fabric_manipulation"],
                        estimated_steps=1,
                        priority=8,
                    ),
                    Goal(
                        description="取下布料",
                        goal_id="g5",
                        preconditions=["等待已结束"],
                        completion_criteria=["布料已取下并释放"],
                        required_skills=["fabric_manipulation"],
                        estimated_steps=2,
                        priority=9,
                        sub_goals=[
                            Goal(description="提起布料脱离支撑板", goal_id="g5_1", required_skills=["fabric_manipulation"], estimated_steps=1),
                            Goal(description="释放布料", goal_id="g5_2", preconditions=["布料已提起"], required_skills=["fabric_manipulation"], estimated_steps=1),
                        ],
                    ),
                ],
            )

        # Generic pick-and-place
        if any(k in instr_lower for k in ["pick", "grab", "抓", "拿", "捡起"]):
            return Goal(
                description=instruction,
                sub_goals=[
                    Goal(description="感知：定位目标物体", goal_id="g1", required_skills=["camera_capture", "vision_3d_estimator"], estimated_steps=2),
                    Goal(description="规划：计算抓取位姿", goal_id="g2", preconditions=["物体位置已知"], required_skills=["grasp_point_predictor"], estimated_steps=1),
                    Goal(description="执行：移动到预抓取位姿", goal_id="g3", preconditions=["抓取位姿已知"], required_skills=["arm_motion_executor"], estimated_steps=1),
                    Goal(description="执行：抓取物体", goal_id="g4", preconditions=["已到达预抓取位姿"], required_skills=["arm_motion_executor"], estimated_steps=1),
                    Goal(description="执行：提起物体", goal_id="g5", preconditions=["物体已被抓取"], required_skills=["arm_motion_executor"], estimated_steps=1),
                ],
            )

        # Generic place
        if any(k in instr_lower for k in ["place", "放", "放置"]):
            return Goal(
                description=instruction,
                sub_goals=[
                    Goal(description="感知：定位放置位置", goal_id="g1", required_skills=["camera_capture", "vision_3d_estimator"], estimated_steps=2),
                    Goal(description="执行：移动到放置位置", goal_id="g2", preconditions=["放置位置已知"], required_skills=["arm_motion_executor"], estimated_steps=1),
                    Goal(description="执行：释放物体", goal_id="g3", preconditions=["已到达放置位置"], required_skills=["arm_motion_executor"], estimated_steps=1),
                    Goal(description="执行：撤退", goal_id="g4", preconditions=["物体已释放"], required_skills=["arm_motion_executor"], estimated_steps=1),
                ],
            )

        # Default: single generic goal
        return Goal(description=instruction, goal_id="g0", required_skills=["camera_capture"], estimated_steps=5)

    @staticmethod
    def _dict_to_goal(data: Dict[str, Any]) -> Goal:
        """Recursively convert a dict to a Goal tree."""
        sub_goals = [TaskDecomposer._dict_to_goal(sg) for sg in data.get("sub_goals", [])]
        return Goal(
            description=data.get("description", ""),
            goal_id=data.get("goal_id", "unknown"),
            preconditions=data.get("preconditions", []),
            completion_criteria=data.get("completion_criteria", []),
            required_skills=data.get("required_skills", []),
            estimated_steps=data.get("estimated_steps", 1),
            priority=data.get("priority", 0),
            sub_goals=sub_goals,
            metadata={"llm_decomposed": True},
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
