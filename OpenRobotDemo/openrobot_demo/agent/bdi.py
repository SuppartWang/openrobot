"""BDI core data models: Belief, Desire, Intent, Goal.

This module defines the fundamental data structures for a BDI-style agent.
No LLM calls here — pure data models and state machines.
"""

from __future__ import annotations

import uuid
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class GoalStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"  # Preconditions not met


class IntentStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class Goal:
    """A hierarchical goal with sub-goals and completion criteria."""

    description: str
    goal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: Optional[str] = None
    sub_goals: List[Goal] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    completion_criteria: List[str] = field(default_factory=list)
    required_skills: List[str] = field(default_factory=list)
    estimated_steps: int = 1
    priority: int = 0  # Higher = more important
    status: GoalStatus = GoalStatus.PENDING
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Execution tracking
    current_step: int = 0
    total_steps_executed: int = 0
    failure_count: int = 0
    max_retries: int = 2

    def is_leaf(self) -> bool:
        """Return True if this goal has no sub-goals."""
        return len(self.sub_goals) == 0

    def is_complete(self) -> bool:
        """A goal is complete if it's marked completed, or all sub-goals are complete."""
        if self.status == GoalStatus.COMPLETED:
            return True
        if self.sub_goals:
            return all(g.is_complete() for g in self.sub_goals)
        return False

    def is_failed(self) -> bool:
        if self.status == GoalStatus.FAILED:
            return True
        if self.sub_goals:
            # A goal fails if any sub-goal fails and cannot be retried
            return any(g.is_failed() for g in self.sub_goals)
        return self.failure_count > self.max_retries

    def get_active_subgoal(self) -> Optional[Goal]:
        """Return the first pending or active sub-goal."""
        for g in self.sub_goals:
            if g.status in (GoalStatus.PENDING, GoalStatus.ACTIVE, GoalStatus.BLOCKED):
                return g
        return None

    def all_subgoals_complete(self) -> bool:
        return all(g.is_complete() for g in self.sub_goals)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority,
            "is_leaf": self.is_leaf(),
            "is_complete": self.is_complete(),
            "is_failed": self.is_failed(),
            "current_step": self.current_step,
            "failure_count": self.failure_count,
            "sub_goals": [g.to_dict() for g in self.sub_goals],
        }


@dataclass
class Belief:
    """A single belief about the world (fact + confidence)."""

    subject: str      # e.g. "obj_0"
    predicate: str    # e.g. "position"
    value: Any        # e.g. [0.3, 0.0, 0.02]
    confidence: float = 1.0
    source: str = ""  # e.g. "vlm_cognition", "proprioception"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
        }


class BeliefSet:
    """The agent's collection of beliefs about the world."""

    def __init__(self):
        self._beliefs: Dict[str, List[Belief]] = {}  # subject -> [Belief, ...]

    def add(self, belief: Belief):
        if belief.subject not in self._beliefs:
            self._beliefs[belief.subject] = []
        self._beliefs[belief.subject].append(belief)

    def get(self, subject: str, predicate: str) -> Optional[Belief]:
        """Get the most recent belief for a subject+predicate."""
        beliefs = self._beliefs.get(subject, [])
        matches = [b for b in beliefs if b.predicate == predicate]
        if not matches:
            return None
        return max(matches, key=lambda b: b.timestamp)

    def query(self, subject: Optional[str] = None, predicate: Optional[str] = None) -> List[Belief]:
        """Query beliefs with optional filters."""
        results = []
        for subj, beliefs in self._beliefs.items():
            if subject is not None and subj != subject:
                continue
            for b in beliefs:
                if predicate is not None and b.predicate != predicate:
                    continue
                results.append(b)
        return results

    def update_from_world_model(self, world_model) -> None:
        """Sync beliefs from the WorldModel."""
        # Robot state beliefs
        rs = world_model.robot_state
        if rs.end_effector_pose is not None:
            self.add(Belief("robot", "end_effector_pose", rs.end_effector_pose, source="proprioception"))
        if rs.gripper_width is not None:
            self.add(Belief("robot", "gripper_width", rs.gripper_width, source="proprioception"))

        # Object beliefs
        for obj_id, obj in world_model.objects.items():
            self.add(Belief(obj_id, "type", obj.object_type, confidence=obj.confidence, source="world_model"))
            if obj.position:
                self.add(Belief(obj_id, "position", obj.position, confidence=obj.confidence, source="world_model"))
            if obj.color:
                self.add(Belief(obj_id, "color", obj.color, confidence=obj.confidence, source="world_model"))
            for rel_key, rel_val in obj.relations.items():
                self.add(Belief(obj_id, f"relation_{rel_key}", rel_val, confidence=obj.confidence, source="world_model"))

    def get_summary(self) -> str:
        """Return a concise text summary for LLM prompts."""
        lines = ["Agent Beliefs:"]
        for subject, beliefs in sorted(self._beliefs.items()):
            if not beliefs:
                continue
            latest = max(beliefs, key=lambda b: b.timestamp)
            lines.append(f"  {subject}: {latest.predicate}={latest.value} (conf={latest.confidence:.2f})")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            subject: [b.to_dict() for b in beliefs]
            for subject, beliefs in self._beliefs.items()
        }


@dataclass
class Desire:
    """A candidate goal the agent wishes to achieve."""

    goal: Goal
    feasibility: float = 1.0  # 0.0~1.0, estimated by desire filter
    reason: str = ""          # Why this desire was formed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal.to_dict(),
            "feasibility": self.feasibility,
            "reason": self.reason,
        }


@dataclass
class Intent:
    """An active plan: a goal + the skill sequence to achieve it."""

    goal_id: str
    plan_steps: List[Dict[str, Any]] = field(default_factory=list)
    status: IntentStatus = IntentStatus.PENDING
    current_step_idx: int = 0
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_complete(self) -> bool:
        return self.status == IntentStatus.COMPLETED

    def current_step(self) -> Optional[Dict[str, Any]]:
        if 0 <= self.current_step_idx < len(self.plan_steps):
            return self.plan_steps[self.current_step_idx]
        return None

    def advance(self):
        self.current_step_idx += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "status": self.status.value,
            "current_step": self.current_step_idx,
            "total_steps": len(self.plan_steps),
            "plan_steps": self.plan_steps,
        }


@dataclass
class Reflection:
    """Result of self-reflection after a failure."""

    analysis: str = ""
    should_retry: bool = False
    adjusted_params: Dict[str, Any] = field(default_factory=dict)
    should_replan: bool = False
    new_strategy: str = ""
    should_give_up: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis": self.analysis,
            "should_retry": self.should_retry,
            "adjusted_params": self.adjusted_params,
            "should_replan": self.should_replan,
            "new_strategy": self.new_strategy,
            "should_give_up": self.should_give_up,
        }
