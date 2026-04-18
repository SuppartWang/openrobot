"""Agent layer for OpenRobotDemo.

Provides BDI-style autonomous agents with goal decomposition,
intent management, and self-reflection.
"""

from openrobot_demo.agent.planner import LLMPlanner, PromptEngine
from openrobot_demo.agent.skill_router import SkillRouter
from openrobot_demo.agent.bdi import (
    Belief,
    BeliefSet,
    Desire,
    Goal,
    GoalStatus,
    Intent,
    IntentStatus,
    Reflection,
)
from openrobot_demo.agent.decomposer import TaskDecomposer
from openrobot_demo.agent.self_reflector import SelfReflector
from openrobot_demo.agent.agent import BDIAgent

__all__ = [
    "LLMPlanner",
    "PromptEngine",
    "SkillRouter",
    # BDI
    "Belief",
    "BeliefSet",
    "Desire",
    "Goal",
    "GoalStatus",
    "Intent",
    "IntentStatus",
    "Reflection",
    # Components
    "TaskDecomposer",
    "SelfReflector",
    "BDIAgent",
]
