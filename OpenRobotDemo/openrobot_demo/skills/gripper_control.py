"""GripperControlSkill: Control the parallel gripper on each arm.

The YHRG S1 end-effector is a 2-finger parallel gripper.
Position ranges:
    0.0  = fully closed (pinching)
    1.0  = fully open
    ~0.5 = half open (for tube insertion)
"""

import logging
from typing import Any, Dict, Optional

from openrobot_demo.skills.base import SkillInterface, SkillSchema, ParamSchema, ResultSchema

logger = logging.getLogger(__name__)


class GripperControlSkill(SkillInterface):
    """Open/close the gripper on left, right, or both arms."""

    name = "gripper_control"

    def __init__(self, dual_arm=None):
        self._dual_arm = dual_arm

    @property
    def schema(self) -> SkillSchema:
        return SkillSchema(
            description=(
                "Control the parallel 2-finger gripper. "
                "position=0.0 closes the gripper, position=1.0 opens it fully."
            ),
            parameters=[
                ParamSchema(
                    name="side",
                    type="str",
                    description='Which arm: "left", "right", or "both".',
                    required=True,
                ),
                ParamSchema(
                    name="position",
                    type="float",
                    description="Gripper aperture: 0.0=closed, 1.0=fully open.",
                    required=True,
                ),
                ParamSchema(
                    name="force",
                    type="float",
                    description="Closing force (0.0~1.0). Default 0.5.",
                    required=False,
                    default=0.5,
                ),
            ],
            returns=[
                ResultSchema(name="success", type="bool", description="Command accepted."),
                ResultSchema(name="message", type="str", description="Status message."),
                ResultSchema(name="side", type="str", description="Which arm was controlled."),
                ResultSchema(name="position", type="float", description="Target position sent."),
            ],
            dependencies=["arm"],
        )

    def execute(
        self,
        side: str,
        position: float,
        force: float = 0.5,
        **kwargs,
    ) -> Dict[str, Any]:
        side = side.lower().strip()
        position = float(position)
        force = float(force)

        # Clamp values
        position = max(0.0, min(1.0, position))
        force = max(0.0, min(1.0, force))

        results = []
        arms = []
        if side in ("left", "both"):
            arms.append("left")
        if side in ("right", "both"):
            arms.append("right")

        if not arms:
            return {"success": False, "message": f"Invalid side '{side}'. Use left/right/both.", "side": side, "position": position}

        for arm_side in arms:
            try:
                if self._dual_arm is not None:
                    arm = self._dual_arm.left_arm if arm_side == "left" else self._dual_arm.right_arm
                    arm.control_gripper(position, force=force)
                    logger.info("[GripperControlSkill] %s gripper -> pos=%.2f force=%.2f", arm_side, position, force)
                    results.append(f"{arm_side}=OK")
                else:
                    logger.info("[GripperControlSkill] Mock: %s gripper -> pos=%.2f force=%.2f", arm_side, position, force)
                    results.append(f"{arm_side}=mock")
            except Exception as exc:
                logger.exception("[GripperControlSkill] %s gripper failed", arm_side)
                results.append(f"{arm_side}=FAIL:{exc}")

        success = all("FAIL" not in r for r in results)
        return {
            "success": success,
            "message": f"Gripper control: {', '.join(results)}",
            "side": side,
            "position": position,
            "force": force,
        }
