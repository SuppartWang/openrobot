"""FabricManipulationSkill: parameterized fabric operations using experience-guided control."""

import logging
import time
from typing import Any, Dict, List, Optional

from openrobot_demo.skills.base import SkillInterface
from openrobot_demo.dual_arm.controller import DualArmController, ArmSide
from openrobot_demo.experience.library import ExperienceLibrary
from openrobot_demo.experience.retriever import ExperienceRetriever
from openrobot_demo.experience.schema import Experience, GripperConfig, DualArmPattern
from openrobot_demo.world_model import WorldModel

logger = logging.getLogger(__name__)


class FabricManipulationSkill(SkillInterface):
    """
    Skill for manipulating cylindrical fabric with a dual-arm, 2-finger gripper setup.

    Core operations (parameterized by Experience):
    - pinch_edge:   two arms pinch opposite edges of the fabric tube
    - lift:         raise the fabric vertically
    - insert:       lower the fabric over a support plate
    - hold_wait:    maintain hold while external process runs
    - withdraw:     lift and remove fabric from the support plate
    """

    name = "fabric_manipulation"

    def __init__(
        self,
        dual_arm: Optional[DualArmController] = None,
        experience_library: Optional[ExperienceLibrary] = None,
        world_model: Optional[WorldModel] = None,
    ):
        self.dual_arm = dual_arm
        self.exp_retriever = ExperienceRetriever(experience_library) if experience_library else None
        self.world_model = world_model

    # ------------------------------------------------------------------
    # SkillInterface entry
    # ------------------------------------------------------------------
    def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a fabric manipulation operation.

        Args:
            operation: one of "pinch_edge", "lift", "insert", "hold_wait", "withdraw"
            **kwargs:  operation-specific parameters (overrides experience defaults)

        Returns:
            {"success": bool, "message": str, ...}
        """
        op_map = {
            "pinch_edge": self._op_pinch_edge,
            "lift": self._op_lift,
            "insert": self._op_insert,
            "hold_wait": self._op_hold_wait,
            "withdraw": self._op_withdraw,
        }
        if operation not in op_map:
            return {"success": False, "message": f"Unknown fabric operation: {operation}"}
        return op_map[operation](**kwargs)

    # ------------------------------------------------------------------
    # Experience-guided parameter loading
    # ------------------------------------------------------------------
    def _get_exp(self, action_type: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Load experience parameters, overridden by runtime kwargs."""
        params: Dict[str, Any] = {}
        if self.exp_retriever:
            exp = self.exp_retriever.retrieve_best(
                task_intent="提起筒状布料并套入支撑板",
                target_object_type="筒状布料",
                action_type=action_type,
                gripper_config=GripperConfig.PARALLEL_2_FINGER,
                arm_count=2,
            )
            if exp:
                params = {
                    "pre_contact_offset": exp.pre_contact_offset,
                    "approach_angle_deg": exp.approach_angle_deg,
                    "gripper_aperture_m": exp.gripper_aperture_m,
                    "contact_force_threshold_n": exp.contact_force_threshold_n,
                    "dual_arm_pinch_distance_m": exp.dual_arm_pinch_distance_m,
                    "dual_arm_sync_tolerance_m": exp.dual_arm_sync_tolerance_m,
                    "left_grasp_relative": exp.left_grasp_relative,
                    "right_grasp_relative": exp.right_grasp_relative,
                    "waypoint_count": exp.waypoint_count,
                    "step_time_s": exp.step_time_s,
                    "max_velocity_m_s": exp.max_velocity_m_s,
                }
                logger.info("[FabricSkill] Loaded experience %s for '%s'", exp.experience_id, action_type)
            else:
                logger.warning("[FabricSkill] No experience found for '%s', using defaults", action_type)
        # Runtime overrides take highest priority
        params.update(overrides)
        return params

    # ------------------------------------------------------------------
    # Operation implementations
    # ------------------------------------------------------------------
    def _op_pinch_edge(
        self,
        fabric_center: List[float],
        fabric_diameter_m: float = 0.08,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Pinch opposite edges of a cylindrical fabric tube.

        Args:
            fabric_center: [x, y, z] of the tube center in base frame
            fabric_diameter_m: tube outer diameter
        """
        if self.dual_arm is None:
            return {"success": False, "message": "No dual_arm controller attached"}

        params = self._get_exp("pinch", kwargs)
        pinch_dist = params.get("dual_arm_pinch_distance_m", fabric_diameter_m)
        offset_z = params.get("pre_contact_offset", [0.0, 0.0, 0.05])[2]
        sync_tol = params.get("dual_arm_sync_tolerance_m", 0.002)

        # Compute left/right grasp targets in base frame
        cx, cy, cz = fabric_center
        half_dist = pinch_dist / 2.0

        left_target = [cx, cy - half_dist, cz]   # grasp left edge
        right_target = [cx, cy + half_dist, cz]  # grasp right edge

        # Pre-grasp poses (hover above)
        left_pre = [cx, cy - half_dist, cz + offset_z]
        right_pre = [cx, cy + half_dist, cz + offset_z]

        try:
            logger.info("[FabricSkill] pinch_edge: center=%s, diameter=%.3fm", fabric_center, fabric_diameter_m)
            self.dual_arm.dual_move_cartesian(left_pre, right_pre, duration=1.0, sync_tolerance_m=sync_tol)
            self.dual_arm.dual_move_cartesian(left_target, right_target, duration=0.5, sync_tolerance_m=sync_tol)
            # Close grippers
            self.dual_arm.left_arm.control_gripper(0.0, force=0.5)
            self.dual_arm.right_arm.control_gripper(0.0, force=0.5)
            time.sleep(0.3)

            # Update world model
            if self.world_model:
                from openrobot_demo.world_model.model import ObjectDesc
                self.world_model.add_or_update_object(
                    ObjectDesc(
                        object_id="fabric_tube",
                        object_type="筒状布料",
                        position=fabric_center,
                        size=f"直径{fabric_diameter_m*100:.1f}cm",
                        relations={"grasped_by": "dual_arm", "grasp_pattern": "opposite_edge"},
                    )
                )

            return {"success": True, "message": f"Pinch complete. Grasp at y-offset ±{half_dist:.3f}m"}
        except Exception as exc:
            logger.exception("[FabricSkill] pinch_edge failed")
            return {"success": False, "message": str(exc)}

    def _op_lift(self, height_m: float = 0.10, **kwargs) -> Dict[str, Any]:
        """Lift the fabric vertically."""
        if self.dual_arm is None:
            return {"success": False, "message": "No dual_arm controller attached"}

        params = self._get_exp("lift", kwargs)
        sync_tol = params.get("dual_arm_sync_tolerance_m", 0.002)
        duration = max(0.5, height_m / params.get("max_velocity_m_s", 0.1))

        try:
            self.dual_arm.dual_lift(height_m, duration)
            return {"success": True, "message": f"Lifted {height_m:.3f}m"}
        except Exception as exc:
            logger.exception("[FabricSkill] lift failed")
            return {"success": False, "message": str(exc)}

    def _op_insert(
        self,
        plate_center: List[float],
        plate_height_m: float = 0.05,
        insert_depth_m: float = 0.06,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Lower the fabric over a support plate.

        Args:
            plate_center: [x, y, z] of the plate top surface center
            plate_height_m: height of the support plate
            insert_depth_m: how deep to lower beyond the plate top
        """
        if self.dual_arm is None:
            return {"success": False, "message": "No dual_arm controller attached"}

        params = self._get_exp("insert", kwargs)
        sync_tol = params.get("dual_arm_sync_tolerance_m", 0.002)

        # Compute target: plate top + insert_depth below
        target_z = plate_center[2] - insert_depth_m
        left_pose = self.dual_arm.get_ee_pose(ArmSide.LEFT)
        right_pose = self.dual_arm.get_ee_pose(ArmSide.RIGHT)

        left_target = [left_pose[0], left_pose[1], target_z]
        right_target = [right_pose[0], right_pose[1], target_z]

        duration = 1.0
        try:
            self.dual_arm.dual_move_cartesian(left_target, right_target, duration, sync_tol)
            return {"success": True, "message": f"Inserted to z={target_z:.3f}m (depth={insert_depth_m:.3f}m)"}
        except Exception as exc:
            logger.exception("[FabricSkill] insert failed")
            return {"success": False, "message": str(exc)}

    def _op_hold_wait(self, wait_seconds: float = 5.0, **kwargs) -> Dict[str, Any]:
        """Maintain grasp while waiting for external process (e.g. defect detection)."""
        if self.dual_arm is None:
            return {"success": False, "message": "No dual_arm controller attached"}

        logger.info("[FabricSkill] Holding fabric for %.1fs (defect detection)...", wait_seconds)
        time.sleep(wait_seconds)
        return {"success": True, "message": f"Held for {wait_seconds:.1f}s"}

    def _op_withdraw(self, lift_height_m: float = 0.10, **kwargs) -> Dict[str, Any]:
        """
        Withdraw fabric from the support plate:
        1. Lift slightly
        2. Release grippers
        3. Move arms to retreat poses
        """
        if self.dual_arm is None:
            return {"success": False, "message": "No dual_arm controller attached"}

        params = self._get_exp("withdraw", kwargs)
        sync_tol = params.get("dual_arm_sync_tolerance_m", 0.002)

        try:
            # Step 1: lift
            self.dual_arm.dual_lift(lift_height_m, duration=0.8)
            # Step 2: release (right first, then left — mimics human coordination)
            self.dual_arm.dual_release(gripper_open_pos=1.0)
            # Step 3: retreat upward
            self.dual_arm.dual_lift(0.05, duration=0.5)

            if self.world_model:
                obj = self.world_model.get_object("fabric_tube")
                if obj:
                    obj.relations["grasped_by"] = "none"
                    self.world_model.add_or_update_object(obj)

            return {"success": True, "message": "Withdraw complete. Fabric released."}
        except Exception as exc:
            logger.exception("[FabricSkill] withdraw failed")
            return {"success": False, "message": str(exc)}
