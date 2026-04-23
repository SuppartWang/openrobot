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
        self._observers: List[Any] = []
        self._last_perception: Dict[str, Any] = {}
        self._execution_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Main execution entry
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Observer / callback interface for dashboard / external UIs
    # ------------------------------------------------------------------
    def add_observer(self, observer):
        """Add an observer callable.  observer(event_type, data_dict)"""
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer):
        if observer in self._observers:
            self._observers.remove(observer)

    def _notify(self, event_type: str, data: Dict[str, Any]):
        for obs in self._observers:
            try:
                obs(event_type, data)
            except Exception as exc:
                logger.debug("Observer notify failed: %s", exc)

    def execute(self, instruction: str, sensors: Optional[List] = None) -> Dict[str, Any]:
        """Execute a user instruction autonomously.

        Returns a summary dict with success status, execution trace, etc.
        """
        self._instruction = instruction
        self._execution_log = []
        logger.info("[BDIAgent] Starting execution of: %s", instruction)
        start_time = time.time()
        self._notify("task_start", {"instruction": instruction})

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
                    self._notify("perception", self._last_perception)

                # Update beliefs
                self.beliefs.update_from_world_model(self.world_model)
                self._notify("beliefs", {"beliefs": self.beliefs.to_dict()})

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
                    self._notify("intent", {
                        "current_intent": self.current_intent.to_dict() if self.current_intent else None,
                        "goal_tree": self.goal_tree.to_dict() if self.goal_tree else None,
                    })
                    if self.current_intent is None:
                        logger.info("[BDIAgent] No more active intents. Waiting...")
                        time.sleep(0.5)
                        continue

                # Execute one step of current intent
                result = self._execute_intent_step()
                self._notify("step_result", result)

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
        self._notify("task_end", summary)
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
                    # Cache latest perception for dashboard
                    if hasattr(reading, 'modality') and hasattr(reading, 'payload'):
                        src = getattr(reading, 'source_id', sensor.source_id)
                        self._last_perception[src] = {
                            "modality": reading.modality,
                            "payload": reading.payload,
                            "timestamp": getattr(reading, 'timestamp', time.time()),
                        }
                except Exception as exc:
                    logger.debug("Sensor %s capture failed: %s", sensor.source_id, exc)

    # ------------------------------------------------------------------
    # Intent management
    # ------------------------------------------------------------------
    def _select_next_intent(self):
        """Find the next active leaf goal and create an intent for it.

        Uses MLDT-style multi-level decomposition: the planner receives the
        high-level leaf goal and produces an initial plan.  The plan is then
        executed step-by-step with closed-loop feedback (Inner Monologue).
        """
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

        # Ask planner for an initial plan (MLDT: goal -> sub-goals -> actions)
        plan = self.planner.plan(leaf.description)
        if not plan:
            logger.warning("[BDIAgent] Planner returned empty plan for goal '%s', will use reactive mode", leaf.goal_id)
            plan = []

        self.current_intent = Intent(
            goal_id=leaf.goal_id,
            plan_steps=plan,
            status=IntentStatus.ACTIVE,
        )
        self._notify("intent", {
            "current_intent": self.current_intent.to_dict(),
            "goal_tree": self.goal_tree.to_dict() if self.goal_tree else None,
        })

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
        """Execute one step of the current intent with closed-loop feedback.

        If the pre-generated plan is exhausted or a step fails, the agent
        asks the LLM planner for the next action reactively (Inner Monologue).
        """
        if self.current_intent is None:
            return {"success": False, "message": "No active intent"}

        step = self.current_intent.current_step()

        # If plan exhausted or empty, ask LLM reactively for next step
        if step is None:
            if self.current_intent.plan_steps:
                # Pre-generated plan finished
                self.current_intent.status = IntentStatus.COMPLETED
                self._mark_goal_completed(self.current_intent.goal_id)
                return {"success": True, "message": "Intent completed"}
            # Reactive mode: ask LLM for next action
            step = self._reactive_plan_next_step()
            if step is None:
                self.current_intent.status = IntentStatus.FAILED
                self._increment_goal_failure(self.current_intent.goal_id)
                return {"success": False, "message": "Reactive planning returned no action"}
            # Append reactive step to plan
            self.current_intent.plan_steps.append(step)

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
            # Check if overall task progress suggests we should replan
            progress = self._check_task_progress()
            if progress.get("should_replan", False):
                logger.info("[BDIAgent] Task progress indicates replanning needed: %s", progress.get("reason", ""))
                new_step = self._reactive_plan_next_step()
                if new_step:
                    self.current_intent.plan_steps.append(new_step)
        else:
            self.current_intent.status = IntentStatus.FAILED
            self._increment_goal_failure(self.current_intent.goal_id)

        return result

    def _reactive_plan_next_step(self) -> Optional[Dict[str, Any]]:
        """Ask the LLM planner for the next action reactively (Inner Monologue mode).

        Builds feedback history from execution history and calls planner.next_action().
        """
        if self.current_intent is None:
            return None

        leaf = self._find_goal_by_id(self.current_intent.goal_id)
        state_summary = self._build_agent_state_summary(leaf)
        feedback = self._build_feedback_history()
        progress = self._format_task_progress()

        try:
            action = self.planner.next_action(
                state_summary=state_summary,
                feedback_history=feedback,
                task_progress=progress,
            )
            if action.get("action") == "finish":
                return None  # Task complete
            if action.get("action") == "skill_call":
                return {
                    "skill": action.get("skill", ""),
                    "args": action.get("args", {}),
                }
            if action.get("action") == "replan":
                # LLM explicitly asked to replan
                logger.info("[BDIAgent] LLM requested replan: %s", action.get("reason", ""))
                return self._reactive_plan_next_step()
        except Exception as exc:
            logger.warning("[BDIAgent] Reactive planning failed: %s", exc)

        return None

    def _build_feedback_history(self) -> List[Dict[str, Any]]:
        """Format execution history into Inner Monologue feedback entries."""
        if self.current_intent is None:
            return []
        feedback = []
        for h in self.current_intent.execution_history:
            result = h.get("result", {})
            entry = {
                "skill": h.get("skill", "?"),
                "success": result.get("success", False),
                "message": result.get("message", "")[:120],
                "observation": self._format_step_observation(h),
            }
            feedback.append(entry)
        return feedback

    def _format_step_observation(self, history_entry: Dict[str, Any]) -> str:
        """Generate a natural language observation of the step's effect."""
        skill = history_entry.get("skill", "")
        result = history_entry.get("result", {})
        if not result.get("success"):
            return "执行失败，需要调整策略或重新感知环境。"
        if skill == "camera_capture":
            return "已获取最新环境图像。"
        if skill == "vision_3d_estimator":
            return "已更新目标物体位置估计。"
        if skill == "arm_motion_executor":
            return "机械臂已完成指定运动。"
        if skill == "fabric_manipulation":
            return "布料操作步骤已完成。"
        return "步骤执行成功。"

    def _check_task_progress(self) -> Dict[str, Any]:
        """Analyze task progress and decide if replanning is needed.

        Inspired by Inner Monologue's Scene feedback.
        """
        if self.current_intent is None or not self.current_intent.execution_history:
            return {"should_replan": False}

        last = self.current_intent.execution_history[-1]
        result = last.get("result", {})

        # If last action failed, definitely replan
        if not result.get("success", False):
            return {"should_replan": True, "reason": "上一步执行失败，需要重新规划"}

        # If we've been on the same intent for too many steps without progress
        if len(self.current_intent.execution_history) > self.max_steps_per_intent:
            return {"should_replan": True, "reason": "当前意图执行步数过多，可能需要调整策略"}

        # Check if goal conditions are satisfied based on execution history patterns
        goal = self._find_goal_by_id(self.current_intent.goal_id)
        if goal and goal.completion_criteria:
            # Simple heuristic: if we've executed enough diverse skills, consider progress good
            skills_used = {h.get("skill", "") for h in self.current_intent.execution_history}
            if len(skills_used) >= len(goal.completion_criteria):
                return {"should_replan": False, "progress": "good"}

        return {"should_replan": False}

    def _format_task_progress(self) -> str:
        """Generate a textual summary of overall task progress for the LLM."""
        if self.goal_tree is None:
            return "尚未开始任务。"

        total = len(self.goal_tree.sub_goals)
        completed = sum(1 for g in self.goal_tree.sub_goals if g.is_complete())
        failed = sum(1 for g in self.goal_tree.sub_goals if g.is_failed())
        active = self.current_intent.goal_id if self.current_intent else "无"

        lines = [f"总体进度: {completed}/{total} 个子目标已完成"]
        if failed > 0:
            lines.append(f"失败子目标: {failed} 个")
        lines.append(f"当前活跃目标: {active}")
        if self.current_intent:
            lines.append(f"当前意图执行步数: {self.current_intent.current_step_idx}/{len(self.current_intent.plan_steps)}")
        return "; ".join(lines)

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
