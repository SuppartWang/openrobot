"""ExperienceRecorder: create new experiences from execution traces."""

import logging
from typing import Any, Dict, Optional

from openrobot_demo.experience.library import ExperienceLibrary
from openrobot_demo.experience.schema import Experience, GripperConfig, DualArmPattern

logger = logging.getLogger(__name__)


class ExperienceRecorder:
    """
    Record execution outcomes as new experiences.

    Supports three sources:
    - human_demo:   operator manually demonstrates (teaching mode)
    - autonomous_trial: robot tries, succeeds/fails, auto-records
    - vla_inference:  VLA model outputs a successful trajectory
    """

    def __init__(self, library: ExperienceLibrary):
        self.library = library

    def record_from_params(
        self,
        task_intent: str,
        target_object_type: str,
        action_type: str,
        params: Dict[str, Any],
        success: bool = True,
        source: str = "autonomous_trial",
        human_feedback: Optional[str] = None,
    ) -> str:
        """
        Record an experience from a flat parameter dict.

        Example params:
        {
            "gripper_aperture_m": 0.0,
            "pre_contact_offset": [0.0, 0.0, 0.05],
            "approach_angle_deg": 90.0,
            "dual_arm_pinch_distance_m": 0.15,
            ...
        }
        """
        exp = Experience(
            task_intent=task_intent,
            target_object_type=target_object_type,
            action_type=action_type,
            source=source,
            success=success,
            human_feedback=human_feedback,
        )
        # Overlay params onto the experience
        for key, value in params.items():
            if hasattr(exp, key):
                setattr(exp, key, value)
            else:
                logger.debug("[ExperienceRecorder] Unknown param key '%s', skipping", key)

        eid = self.library.add(exp)
        logger.info("[ExperienceRecorder] Recorded %s from %s (success=%s)", eid, source, success)
        return eid

    def record_from_execution(
        self,
        experience: Experience,
        actual_execution_time_s: float,
        actual_final_error_m: float,
        tactile_feedback: Optional[Dict[str, Any]] = None,
        success: bool = True,
    ) -> str:
        """Update an experience template with actual execution outcome and save."""
        exp = Experience(
            task_intent=experience.task_intent,
            target_object_type=experience.target_object_type,
            action_type=experience.action_type,
            gripper_config=experience.gripper_config,
            arm_count=experience.arm_count,
            dual_arm_pattern=experience.dual_arm_pattern,
            pre_contact_offset=experience.pre_contact_offset,
            approach_angle_deg=experience.approach_angle_deg,
            gripper_aperture_m=experience.gripper_aperture_m,
            contact_force_threshold_n=experience.contact_force_threshold_n,
            dual_arm_pinch_distance_m=experience.dual_arm_pinch_distance_m,
            dual_arm_sync_tolerance_m=experience.dual_arm_sync_tolerance_m,
            left_grasp_relative=experience.left_grasp_relative,
            right_grasp_relative=experience.right_grasp_relative,
            trajectory_type=experience.trajectory_type,
            waypoint_count=experience.waypoint_count,
            step_time_s=experience.step_time_s,
            max_velocity_m_s=experience.max_velocity_m_s,
            compliance_stiffness=experience.compliance_stiffness,
            execution_time_s=actual_execution_time_s,
            final_error_m=actual_final_error_m,
            tactile_feedback=tactile_feedback,
            success=success,
            source="autonomous_trial",
        )
        eid = self.library.add(exp)
        logger.info("[ExperienceRecorder] Execution recorded %s (success=%s, err=%.4fm)", eid, success, actual_final_error_m)
        return eid
