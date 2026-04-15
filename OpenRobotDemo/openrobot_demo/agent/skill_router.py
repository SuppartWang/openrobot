"""SkillRouter: executes a planned sequence of skills with context substitution."""

import logging
from typing import Any, Dict, List
from openrobot_demo.skills.base import SkillInterface

logger = logging.getLogger(__name__)


class SkillRouter:
    def __init__(self):
        self._skills: Dict[str, SkillInterface] = {}
        self._context: Dict[str, Any] = {}

    def register(self, skill: SkillInterface):
        self._skills[skill.name] = skill
        logger.info(f"[SkillRouter] Registered skill: {skill.name}")

    def execute_plan(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a list of {skill, args} steps sequentially."""
        results = []
        for idx, step in enumerate(plan):
            skill_name = step.get("skill")
            raw_args = step.get("args", {})
            logger.info(f"[SkillRouter] Step {idx + 1}/{len(plan)}: {skill_name}")

            if skill_name not in self._skills:
                msg = f"Unknown skill: {skill_name}"
                logger.error(msg)
                results.append({"step": idx, "skill": skill_name, "success": False, "message": msg})
                return {"success": False, "message": msg, "results": results}

            # Resolve placeholders in args using context
            args = self._resolve_args(raw_args)

            # Special handling for arm_motion_executor placeholders
            if skill_name == "arm_motion_executor":
                args = self._resolve_motion_args(args)

            skill = self._skills[skill_name]
            try:
                result = skill.execute(**args)
            except Exception as e:
                logger.exception(f"Skill {skill_name} execution failed")
                result = {"success": False, "message": str(e)}

            results.append({"step": idx, "skill": skill_name, "result": result})

            if not result.get("success", False):
                msg = f"Step {idx} ({skill_name}) failed: {result.get('message')}"
                logger.error(msg)
                return {"success": False, "message": msg, "results": results}

            # Update context with key outputs for downstream skills
            self._update_context(skill_name, result)

        return {"success": True, "message": "Plan executed successfully.", "results": results}

    def _resolve_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively substitute string placeholders with context values."""
        def _resolve(v):
            if isinstance(v, str) and v in self._context:
                return self._context[v]
            if isinstance(v, dict):
                return {kk: _resolve(vv) for kk, vv in v.items()}
            if isinstance(v, list):
                return [_resolve(vv) for vv in v]
            return v

        resolved = {}
        for k, v in args.items():
            resolved[k] = _resolve(v)
        return resolved

    def _resolve_motion_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Replace special motion placeholders (PRE_GRASP, GRASP, LIFT, PLACE, RETREAT)."""
        tv = args.get("target_values")
        if not isinstance(tv, str):
            return args

        placeholder = tv.upper()
        if placeholder == "PRE_GRASP":
            pose = self._context.get("pre_grasp_pose")
        elif placeholder == "GRASP":
            pose = self._context.get("grasp_pose")
        elif placeholder == "LIFT":
            pose = self._context.get("grasp_pose", [0, 0, 0, 0, 0, 0, 1]).copy()
            if isinstance(pose, list):
                pose[2] += 0.1  # lift 10cm
        elif placeholder == "PLACE":
            # Simple place location: current grasp pose but shifted in XY
            pose = self._context.get("grasp_pose", [0, 0, 0, 0, 0, 0, 1]).copy()
            if isinstance(pose, list):
                pose[1] += 0.15  # place to the left 15cm
        elif placeholder == "RETREAT":
            pose = self._context.get("grasp_pose", [0, 0, 0, 0, 0, 0, 1]).copy()
            if isinstance(pose, list):
                pose[2] += 0.1
        else:
            pose = None

        if pose is not None:
            args = dict(args)
            args["target_values"] = pose
        return args

    def _update_context(self, skill_name: str, result: Dict[str, Any]):
        if skill_name == "camera_capture":
            self._context["rgb_frame"] = result.get("rgb_frame")
            self._context["depth_frame"] = result.get("depth_frame")
        elif skill_name == "arm_state_reader":
            self._context["joint_positions"] = result.get("joint_positions")
            self._context["end_effector_pose"] = result.get("end_effector_pose")
        elif skill_name == "vision_3d_estimator":
            self._context["pixel_bbox"] = result.get("pixel_bbox")
            self._context["pixel_center"] = result.get("pixel_center")
            self._context["camera_3d"] = result.get("camera_3d")
            self._context["base_3d"] = result.get("base_3d")
            # Fallback: if base_3d unavailable, use camera_3d; if neither, use a default test pose
            obj_pose = result.get("base_3d") or result.get("camera_3d") or [0.3, 0.0, 0.2]
            # Append identity quaternion if pose is only 3D position
            if len(obj_pose) == 3:
                obj_pose = obj_pose + [0.0, 0.0, 0.0, 1.0]
            self._context["object_pose_base"] = obj_pose
        elif skill_name == "grasp_point_predictor":
            self._context["grasp_pose"] = result.get("grasp_pose")
            self._context["pre_grasp_pose"] = result.get("pre_grasp_pose")
            self._context["approach_vector"] = result.get("approach_vector")
            self._context["gripper_width"] = result.get("gripper_width")
