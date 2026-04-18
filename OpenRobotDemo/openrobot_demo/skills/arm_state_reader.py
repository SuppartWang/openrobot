"""Skill 2: ArmStateReader"""

import logging
from typing import Any, Dict, List
from openrobot_demo.skills.base import SkillInterface, SkillSchema, ParamSchema, ResultSchema
from openrobot_demo.hardware.yhrg_adapter import S1_arm, S1_slover, control_mode

logger = logging.getLogger(__name__)


class ArmStateReader(SkillInterface):
    def __init__(self, mode: control_mode = control_mode.only_sim,
                 dev: str = "/dev/ttyUSB0", end_effector: str = "gripper",
                 check_collision: bool = True,
                 external_arm=None, external_solver=None):
        if external_arm is not None:
            self._arm = external_arm
            self._solver = external_solver
        else:
            self._arm: S1_arm = S1_arm(mode=mode, dev=dev, end_effector=end_effector,
                                       check_collision=check_collision)
            self._solver = S1_slover([0.0, 0.0, 0.0])

    @property
    def name(self) -> str:
        return "arm_state_reader"

    @property
    def schema(self) -> SkillSchema:
        return SkillSchema(
            description="Read the current state of the robot arm (joints, velocities, torques, temperatures, end-effector pose).",
            parameters=[
                ParamSchema(
                    name="fields",
                    type="list",
                    description="List of fields to read. Options: 'pos', 'vel', 'tau', 'temp'. Default reads all.",
                    required=False,
                    default=None,
                    example=["pos", "vel"],
                ),
            ],
            returns=[
                ResultSchema(name="success", type="bool", description="Whether read succeeded."),
                ResultSchema(name="message", type="str", description="Status message."),
                ResultSchema(name="joint_positions", type="list", description="Current joint positions (radians)."),
                ResultSchema(name="joint_velocities", type="list", description="Current joint velocities."),
                ResultSchema(name="joint_torques", type="list", description="Current joint torques."),
                ResultSchema(name="temperatures", type="list", description="Motor temperatures (°C)."),
                ResultSchema(name="end_effector_pose", type="list", description="EE pose [x, y, z, qx, qy, qz, qw]."),
            ],
            dependencies=["arm"],
        )

    def execute(self, fields: List[str] = None, **kwargs) -> Dict[str, Any]:
        fields = fields or ["pos", "vel", "tau", "temp"]
        result = {"success": True, "message": "Arm state read."}

        if "pos" in fields:
            pos = self._arm.get_pos()
            result["joint_positions"] = pos
            # Forward kinematics -> end-effector pose
            try:
                ee_pose = self._solver.forward_quat(pos)
                result["end_effector_pose"] = ee_pose
            except Exception as e:
                logger.warning(f"FK failed: {e}")
                result["end_effector_pose"] = None

        if "vel" in fields:
            result["joint_velocities"] = self._arm.get_vel()
        if "tau" in fields:
            result["joint_torques"] = self._arm.get_tau()
        if "temp" in fields:
            result["temperatures"] = self._arm.get_temp()

        return result

    def enable(self):
        return self._arm.enable()

    def disable(self):
        return self._arm.disable()
