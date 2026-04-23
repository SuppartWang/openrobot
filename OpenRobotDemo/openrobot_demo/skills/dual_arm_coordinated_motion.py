"""DualArmCoordinatedMotionSkill: Synchronized dual-arm Cartesian motion.

Wraps DualArmController.dual_move_cartesian() with:
  - absolute target pose mode
  - relative offset mode (from current pose + delta)
  - single-arm or dual-arm execution
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from openrobot_demo.dual_arm.controller import ArmSide
from openrobot_demo.skills.base import SkillInterface, SkillSchema, ParamSchema, ResultSchema

logger = logging.getLogger(__name__)


class DualArmCoordinatedMotionSkill(SkillInterface):
    """Execute synchronized Cartesian motion on one or both arms.

    In relative mode the skill reads the current end-effector pose,
    adds the requested offset, and moves both arms synchronously.
    In absolute mode it moves directly to the given poses.
    """

    name = "dual_arm_coordinated_motion"

    def __init__(self, dual_arm=None):
        self._dual_arm = dual_arm

    @property
    def schema(self) -> SkillSchema:
        return SkillSchema(
            description=(
                "Synchronized Cartesian motion for dual arms. "
                "Supports absolute target poses or relative offsets."
            ),
            parameters=[
                ParamSchema(
                    name="command_type",
                    type="str",
                    description='"absolute" (go to pose) or "relative" (offset from current).',
                    required=True,
                ),
                ParamSchema(
                    name="side",
                    type="str",
                    description='"left", "right", or "both".',
                    required=False,
                    default="both",
                ),
                # Absolute mode
                ParamSchema(
                    name="left_target",
                    type="list",
                    description="Left arm target pose [x,y,z,qx,qy,qz,qw] or [x,y,z].",
                    required=False,
                    default=None,
                ),
                ParamSchema(
                    name="right_target",
                    type="list",
                    description="Right arm target pose [x,y,z,qx,qy,qz,qw] or [x,y,z].",
                    required=False,
                    default=None,
                ),
                # Relative mode offsets
                ParamSchema(
                    name="x_offset",
                    type="float",
                    description="X offset in meters.",
                    required=False,
                    default=0.0,
                ),
                ParamSchema(
                    name="y_offset",
                    type="float",
                    description="Y offset in meters.",
                    required=False,
                    default=0.0,
                ),
                ParamSchema(
                    name="z_offset",
                    type="float",
                    description="Z offset in meters.",
                    required=False,
                    default=0.0,
                ),
                # Common
                ParamSchema(
                    name="duration",
                    type="float",
                    description="Motion duration in seconds.",
                    required=False,
                    default=2.0,
                ),
                ParamSchema(
                    name="sync_tolerance_m",
                    type="float",
                    description="Sync tolerance in meters.",
                    required=False,
                    default=0.002,
                ),
            ],
            returns=[
                ResultSchema(name="success", type="bool", description="Motion succeeded."),
                ResultSchema(name="message", type="str", description="Status message."),
                ResultSchema(name="left_start", type="list", description="Left arm start pose."),
                ResultSchema(name="left_end", type="list", description="Left arm target pose."),
                ResultSchema(name="right_start", type="list", description="Right arm start pose."),
                ResultSchema(name="right_end", type="list", description="Right arm target pose."),
            ],
            dependencies=["arm"],
        )

    def execute(
        self,
        command_type: str,
        side: str = "both",
        left_target: Optional[List[float]] = None,
        right_target: Optional[List[float]] = None,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        z_offset: float = 0.0,
        duration: float = 2.0,
        sync_tolerance_m: float = 0.002,
        **kwargs,
    ) -> Dict[str, Any]:
        if self._dual_arm is None:
            logger.info("[DualArmCoordinatedMotion] Mock mode: %s motion by %s", command_type, side)
            return {
                "success": True,
                "message": f"Mock {command_type} motion on {side} side.",
                "left_start": None, "left_end": left_target,
                "right_start": None, "right_end": right_target,
            }

        side = side.lower().strip()
        command_type = command_type.lower().strip()

        # Determine which arms to move
        move_left = side in ("left", "both")
        move_right = side in ("right", "both")

        # Get current poses
        left_start = None
        right_start = None
        if move_left:
            try:
                left_start = self._dual_arm.get_ee_pose(ArmSide.LEFT)
            except Exception as exc:
                logger.warning("[DualArmCoordinatedMotion] Failed to get left EE pose: %s", exc)
        if move_right:
            try:
                right_start = self._dual_arm.get_ee_pose(ArmSide.RIGHT)
            except Exception as exc:
                logger.warning("[DualArmCoordinatedMotion] Failed to get right EE pose: %s", exc)

        # Compute target poses
        if command_type == "absolute":
            left_end = self._normalize_pose(left_target) if move_left else None
            right_end = self._normalize_pose(right_target) if move_right else None
        elif command_type == "relative":
            left_end = self._apply_offset(left_start, x_offset, y_offset, z_offset) if move_left else None
            right_end = self._apply_offset(right_start, x_offset, y_offset, z_offset) if move_right else None
        else:
            return {"success": False, "message": f"Unknown command_type: {command_type}"}

        # Execute
        try:
            if move_left and move_right:
                # Dual-arm synchronized
                if left_end is None or right_end is None:
                    return {"success": False, "message": "Missing target pose for dual-arm motion."}
                self._dual_arm.dual_move_cartesian(
                    left_target=left_end,
                    right_target=right_end,
                    duration=duration,
                    sync_tolerance_m=sync_tolerance_m,
                )
            elif move_left:
                # Left only
                if left_end is None:
                    return {"success": False, "message": "Missing left_target."}
                self._dual_arm.move_cartesian(ArmSide.LEFT, left_end, duration=duration)
            elif move_right:
                # Right only
                if right_end is None:
                    return {"success": False, "message": "Missing right_target."}
                self._dual_arm.move_cartesian(ArmSide.RIGHT, right_end, duration=duration)
            else:
                return {"success": False, "message": f"Invalid side: {side}"}

            return {
                "success": True,
                "message": f"{command_type.capitalize()} motion completed on {side} side.",
                "left_start": left_start,
                "left_end": left_end,
                "right_start": right_start,
                "right_end": right_end,
            }

        except Exception as exc:
            logger.exception("[DualArmCoordinatedMotion] Motion failed")
            return {
                "success": False,
                "message": f"Motion failed: {exc}",
                "left_start": left_start, "left_end": left_end,
                "right_start": right_start, "right_end": right_end,
            }

    @staticmethod
    def _normalize_pose(pose: Optional[List[float]]) -> Optional[List[float]]:
        """Convert 3-DOF position to 7-DOF pose if needed."""
        if pose is None:
            return None
        if len(pose) == 3:
            return list(pose) + [0.0, 0.0, 0.0, 1.0]
        return list(pose)

    @staticmethod
    def _apply_offset(
        pose: Optional[List[float]],
        dx: float, dy: float, dz: float,
    ) -> Optional[List[float]]:
        """Add translation offset to a pose (position part only)."""
        if pose is None:
            return None
        out = list(pose)
        out[0] += dx
        out[1] += dy
        out[2] += dz
        return out
