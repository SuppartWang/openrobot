"""Skill 2: ArmStateReader"""

import logging
from typing import Any, Dict, List
from openrobot_demo.skills.base import SkillInterface
from openrobot_demo.hardware.yhrg_adapter import S1_arm, S1_slover, control_mode

logger = logging.getLogger(__name__)


class ArmStateReader(SkillInterface):
    def __init__(self, mode: control_mode = control_mode.only_sim,
                 dev: str = "/dev/ttyUSB0", end_effector: str = "gripper",
                 check_collision: bool = True):
        self._arm: S1_arm = S1_arm(mode=mode, dev=dev, end_effector=end_effector,
                                   check_collision=check_collision)
        self._solver = S1_slover([0.0, 0.0, 0.0])

    @property
    def name(self) -> str:
        return "arm_state_reader"

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
