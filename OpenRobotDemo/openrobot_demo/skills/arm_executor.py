"""Skill 6: ArmMotionExecutor"""

import time
import logging
from typing import Any, Dict, List
import numpy as np

from openrobot_demo.skills.base import SkillInterface
from openrobot_demo.hardware.yhrg_adapter import S1_arm, S1_slover, control_mode
from openrobot_demo.control.safety_gateway import SafetyGateway
from openrobot_demo.control.interpolator import JointSpaceInterpolator

logger = logging.getLogger(__name__)


class ArmMotionExecutor(SkillInterface):
    def __init__(self,
                 mode: control_mode = control_mode.only_sim,
                 dev: str = "/dev/ttyUSB0",
                 end_effector: str = "gripper",
                 check_collision: bool = True,
                 external_arm=None, external_solver=None):
        if external_arm is not None:
            self._arm = external_arm
            self._solver = external_solver
        else:
            self._arm = S1_arm(mode=mode, dev=dev, end_effector=end_effector,
                               check_collision=check_collision)
            self._solver = S1_slover([0.0, 0.0, 0.0])
        self._safety = SafetyGateway()
        self._interpolator = JointSpaceInterpolator(num_steps=30)
        self._enabled = False

    @property
    def name(self) -> str:
        return "arm_motion_executor"

    def enable(self):
        if not self._enabled:
            self._arm.enable()
            self._enabled = True

    def disable(self):
        self._arm.disable()
        self._enabled = False

    def execute(self,
                command_type: str,
                target_values: List[float],
                speed: float = 1.0,
                use_interpolation: bool = True,
                **kwargs) -> Dict[str, Any]:
        self.enable()
        start_t = time.time()

        if command_type == "joint":
            current = np.array(self._arm.get_pos())
            target = np.array(target_values[:7])
            ok, safe_target, reason = self._safety.check_joint_command(target.tolist(), current.tolist())
            if not ok:
                return {"success": False, "message": f"Safety check failed: {reason}"}

            if use_interpolation:
                traj = self._interpolator.plan(current, np.array(safe_target))
                for q in traj:
                    self._arm.joint_control(q.tolist())
                    time.sleep(0.02 / max(0.1, speed))
            else:
                self._arm.joint_control(safe_target)
                time.sleep(0.5)

            actual = self._arm.get_pos()
            return {
                "success": True,
                "message": "Joint motion executed.",
                "actual_reached_pos": actual,
                "execution_time_ms": int((time.time() - start_t) * 1000),
            }

        elif command_type == "cartesian":
            pose = target_values[:7]  # [x, y, z, qx, qy, qz, qw]
            xyz = pose[:3]
            ok, reason = self._safety.check_cartesian_target(xyz)
            if not ok:
                return {"success": False, "message": f"Safety check failed: {reason}"}

            current_joints = self._arm.get_pos()
            joints = self._solver.inverse_quat(pose, current_joints)
            if joints is None:
                return {"success": False, "message": "IK failed: target unreachable."}

            # Re-run as joint command
            return self.execute(command_type="joint", target_values=joints,
                                speed=speed, use_interpolation=use_interpolation)

        elif command_type == "gripper":
            pos = float(np.clip(target_values[0], 0.0, 2.0))
            force = float(target_values[1]) if len(target_values) > 1 else 0.5
            self._arm.control_gripper(pos, force)
            time.sleep(0.3)
            return {
                "success": True,
                "message": f"Gripper set to {pos}.",
                "execution_time_ms": int((time.time() - start_t) * 1000),
            }

        else:
            return {"success": False, "message": f"Unknown command_type: {command_type}"}
