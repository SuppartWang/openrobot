"""BDIAgent: Belief-Desire-Intent autonomous robot agent.

This is the top-level orchestrator that:
1. Receives user instructions
2. Decomposes them into goal hierarchies (via TaskDecomposer)
3. Maintains beliefs synced from WorldModel
4. Manages an intent stack (active plans)
5. Delegates step-level planning to LLMPlanner
6. Reflects on failures and self-corrects (via SelfReflector)

The Agent does NOT replace LLMPlanner — it wraps it. The Agent handles
macro-level reasoning (goals, decomposition, recovery), while the Planner
handles micro-level reasoning (next skill call given current state).
"""

import logging
import time
from typing import Any, Dict, List, Optional

from openrobot_demo.agent.bdi import BeliefSet, Desire, Goal, GoalStatus, Intent, IntentStatus, Reflection
from openrobot_demo.agent.decomposer import TaskDecomposer
from openrobot_demo.agent.planner import LLMPlanner
from openrobot_demo.agent.self_reflector import SelfReflector
from openrobot_demo.agent.skill_router import SkillRouter
from openrobot_demo.runtime.harness import HarnessRunner
from openrobot_demo.world_model import WorldModel

logger = logging.getLogger(__name__)


class BDIAgent:
    """Autonomous BDI agent for robot task execution."""

    def __init__(
        self,
        planner: LLMPlanner,
        skill_router: SkillRouter,
        world_model: WorldModel,
        decomposer: Optional[TaskDecomposer] = None,
        reflector: Optional[SelfReflector] = None,
        harness_runner: Optional[HarnessRunner] = None,
        max_steps_per_intent: int = 20,
        max_total_steps: int = 100,
    ):
        self.planner = planner
        self.skill_router = skill_router
        self.world_model = world_model
        self.decomposer = decomposer or TaskDecomposer(client=getattr(planner, "_client", None))
        self.reflector = reflector or SelfReflector(client=getattr(planner, "_client", None))
        self.harness = harness_runner or HarnessRunner(skill_router)

        self.beliefs = BeliefSet()
        self.goal_tree: Optional[Goal] = None
        self.intent_stack: List[Intent] = []
        self.current_intent: Optional[Intent] = None

        self.max_steps_per_intent = max_steps_per_intent
        self.max_total_steps = max_total_steps
        self._total_steps = 0
        self._instruction = ""

    # ------------------------------------------------------------------
    # Main execution entry
    # ------------------------------------------------------------------
    def execute(self, instruction: str, sensors: Optional[List] = None) -> Dict[str, Any]:
        """Execute a user instruction autonomously.

        Returns a summary dict with success status, execution trace, etc.
        """
        self._instruction = instruction
        logger.info("[BDIAgent] Starting execution of: %s", instruction)
        start_time = time.time()

        # 1. Decompose into goal tree
        available_skills = self.skill_router.list_skills()
        self.goal_tree = self.decomposer.decompose(instruction, available_skills)
        logger.info("[BDIAgent] Decomposed into goal tree with %d top-level goals", len(self.goal_tree.sub_goals))

        # 2. Initialize planner for this task
        self.planner.start_task(instruction)

        # 3. Main BDI loop
        finished = False
        try:
            for step in range(self.max_total_steps):
                self._total_steps = step

                # Perceive
                if sensors:
                    self._perceive(sensors)

                # Update beliefs
                self.beliefs.update_from_world_model(self.world_model)

                # Check if all goals complete
                if self.goal_tree.is_complete():
                    logger.info("[BDIAgent] All goals completed after %d steps.", step)
                    finished = True
                    break

                # Check if goal tree failed irrecoverably
                if self.goal_tree.is_failed():
                    logger.error("[BDIAgent] Goal tree failed irrecoverably after %d steps.", step)
                    break

                # Manage intent
                if self.current_intent is None or self.current_intent.is_complete():
                    self._select_next_intent()
                    if self.current_intent is None:
                        logger.info("[BDIAgent] No more active intents. Waiting...")
                        time.sleep(0.5)
                        continue

                # Execute one step of current intent
                result = self._execute_intent_step()

                if not result.get("success", False):
                    # Failure → reflect and recover
                    reflection = self._reflect_on_failure(result)
                    recovery = self._apply_reflection(reflection)
                    if not recovery:
                        logger.error("[BDIAgent] Recovery failed. Stopping.")
                        break

        except KeyboardInterrupt:
            logger.info("[BDIAgent] Interrupted by user.")
        except Exception as exc:
            logger.exception("[BDIAgent] Unexpected error")

        elapsed = time.time() - start_time
        summary = self._build_summary(finished, elapsed)
        logger.info("[BDIAgent] Execution finished: success=%s, time=%.1fs, steps=%d",
                    summary["success"], elapsed, self._total_steps)
        return summary

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------
    def _perceive(self, sensors: List):
        """Capture from all sensors and feed into WorldModel."""
        for sensor in sensors:
            if sensor.is_available():
                try:
                    reading = sensor.capture()
                    self.world_model.ingest(reading)
                except Exception as exc:
                    logger.debug("Sensor %s capture failed: %s", sensor.source_id, exc)

    # ------------------------------------------------------------------
    # Intent management
    # ------------------------------------------------------------------
    def _select_next_intent(self):
        """Find the next active leaf goal and create an intent for it."""
        if self.goal_tree is None:
            return

        leaf = self._find_next_active_leaf(self.goal_tree)
        if leaf is None:
            self.current_intent = None
            return

        leaf.status = GoalStatus.ACTIVE
        logger.info("[BDIAgent] Forming intent for goal '%s': %s", leaf.goal_id, leaf.description)

        # Build state summary for planner
        state_summary = self._build_agent_state_summary(leaf)

        # Ask planner for a plan (list of skill calls)
        plan = self.planner.plan(leaf.description)
        if not plan:
            logger.warning("[BDIAgent] Planner returned empty plan for goal '%s'", leaf.goal_id)
            plan = []

        self.current_intent = Intent(
            goal_id=leaf.goal_id,
            plan_steps=plan,
            status=IntentStatus.ACTIVE,
        )

    def _find_next_active_leaf(self, goal: Goal) -> Optional[Goal]:
        """DFS to find the first pending or blocked leaf goal."""
        if goal.status in (GoalStatus.PENDING, GoalStatus.BLOCKED):
            if goal.is_leaf():
                return goal
            for sub in goal.sub_goals:
                leaf = self._find_next_active_leaf(sub)
                if leaf is not None:
                    return leaf
        elif goal.status == GoalStatus.ACTIVE:
            if goal.is_leaf():
                return goal
            for sub in goal.sub_goals:
                leaf = self._find_next_active_leaf(sub)
                if leaf is not None:
                    return leaf
        return None

    # ------------------------------------------------------------------
    # Intent execution
    # ------------------------------------------------------------------
    def _execute_intent_step(self) -> Dict[str, Any]:
        """Execute one step of the current intent."""
        if self.current_intent is None:
            return {"success": False, "message": "No active intent"}

        step = self.current_intent.current_step()
        if step is None:
            self.current_intent.status = IntentStatus.COMPLETED
            # Mark corresponding goal as completed
            self._mark_goal_completed(self.current_intent.goal_id)
            return {"success": True, "message": "Intent completed"}

        skill_name = step.get("skill")
        args = step.get("args", {})

        logger.info("[BDIAgent] Step %d/%d: skill=%s",
                    self.current_intent.current_step_idx + 1,
                    len(self.current_intent.plan_steps),
                    skill_name)

        result = self.skill_router.execute_single(skill_name, args)

        self.current_intent.execution_history.append({
            "step_idx": self.current_intent.current_step_idx,
            "skill": skill_name,
            "args": args,
            "result": result,
            "timestamp": time.time(),
        })

        if result.get("success", False):
            self.current_intent.advance()
        else:
            self.current_intent.status = IntentStatus.FAILED
            # Mark corresponding goal
            self._increment_goal_failure(self.current_intent.goal_id)

        return result

    # ------------------------------------------------------------------
    # Reflection and recovery
    # ------------------------------------------------------------------
    def _reflect_on_failure(self, result: Dict[str, Any]) -> Reflection:
        """Invoke SelfReflector to analyze the failure."""
        last_step = self.current_intent.execution_history[-1] if self.current_intent else {}
        skill_name = last_step.get("skill", "unknown")
        args = last_step.get("args", {})
        failure_msg = result.get("message", "Unknown failure")
        state_summary = self._build_agent_state_summary()
        history = [h for h in (self.current_intent.execution_history if self.current_intent else [])]

        return self.reflector.reflect(
            skill_name=skill_name,
            skill_args=args,
            failure_message=failure_msg,
            state_summary=state_summary,
            execution_history=history,
        )

    def _apply_reflection(self, reflection: Reflection) -> bool:
        """Apply recovery strategy from reflection. Return True if recovery succeeded."""
        logger.info("[BDIAgent] Reflection: %s", reflection.analysis)

        if reflection.should_give_up:
            logger.error("[BDIAgent] Reflection says give up.")
            return False

        if reflection.should_retry and self.current_intent is not None:
            # Retry the current intent with adjusted parameters
            logger.info("[BDIAgent] Retrying intent with adjusted params: %s", reflection.adjusted_params)
            self.current_intent.status = IntentStatus.RETRYING
            self.current_intent.current_step_idx = 0  # Restart from beginning
            # Inject adjusted params into plan steps (heuristic)
            for step in self.current_intent.plan_steps:
                step_args = step.get("args", {})
                for key, val in reflection.adjusted_params.items():
                    if key in step_args:
                        step_args[key] = val
            return True

        if reflection.should_replan and self.current_intent is not None:
            # Re-plan the current goal
            logger.info("[BDIAgent] Replanning current goal.")
            goal = self._find_goal_by_id(self.current_intent.goal_id)
            if goal:
                goal.status = GoalStatus.PENDING
                # Get a new plan from planner
                new_plan = self.planner.plan(goal.description)
                self.current_intent.plan_steps = new_plan
                self.current_intent.current_step_idx = 0
                self.current_intent.status = IntentStatus.ACTIVE
                return True

        return False

    # ------------------------------------------------------------------
    # Goal tree helpers
    # ------------------------------------------------------------------
    def _mark_goal_completed(self, goal_id: str):
        goal = self._find_goal_by_id(goal_id)
        if goal:
            goal.status = GoalStatus.COMPLETED
            logger.info("[BDIAgent] Goal '%s' marked as completed.", goal_id)
            # If parent goal's sub-goals are all complete, mark parent too
            self._propagate_completion(self.goal_tree)

    def _increment_goal_failure(self, goal_id: str):
        goal = self._find_goal_by_id(goal_id)
        if goal:
            goal.failure_count += 1
            logger.warning("[BDIAgent] Goal '%s' failure count: %d/%d",
                           goal_id, goal.failure_count, goal.max_retries)
            if goal.failure_count > goal.max_retries:
                goal.status = GoalStatus.FAILED
                logger.error("[BDIAgent] Goal '%s' marked as FAILED (exceeded max retries).", goal_id)

    def _find_goal_by_id(self, goal_id: str) -> Optional[Goal]:
        return self._find_goal_recursive(self.goal_tree, goal_id)

    def _find_goal_recursive(self, goal: Optional[Goal], goal_id: str) -> Optional[Goal]:
        if goal is None:
            return None
        if goal.goal_id == goal_id:
            return goal
        for sub in goal.sub_goals:
            found = self._find_goal_recursive(sub, goal_id)
            if found:
                return found
        return None

    def _propagate_completion(self, goal: Goal):
        if goal.sub_goals and all(g.is_complete() for g in goal.sub_goals):
            goal.status = GoalStatus.COMPLETED
            logger.info("[BDIAgent] Goal '%s' auto-completed (all sub-goals done).", goal.goal_id)

    # ------------------------------------------------------------------
    # State summary for planner
    # ------------------------------------------------------------------
    def _build_agent_state_summary(self, active_goal: Optional[Goal] = None) -> str:
        """Build a comprehensive state summary for the LLM planner."""
        lines = []

        if active_goal:
            lines.append(f"当前目标: {active_goal.description} (ID: {active_goal.goal_id})")
            if active_goal.preconditions:
                lines.append(f"前置条件: {', '.join(active_goal.preconditions)}")
            if active_goal.completion_criteria:
                lines.append(f"完成标准: {', '.join(active_goal.completion_criteria)}")

        lines.append(self.world_model.build_state_summary())
        lines.append(self.beliefs.get_summary())

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def _build_summary(self, finished: bool, elapsed: float) -> Dict[str, Any]:
        return {
            "success": finished,
            "instruction": self._instruction,
            "elapsed_time_s": round(elapsed, 2),
            "total_steps": self._total_steps,
            "goal_tree": self.goal_tree.to_dict() if self.goal_tree else None,
            "current_intent": self.current_intent.to_dict() if self.current_intent else None,
            "beliefs": self.beliefs.to_dict(),
        }
