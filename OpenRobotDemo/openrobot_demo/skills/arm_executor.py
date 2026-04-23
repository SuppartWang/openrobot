"""Skill 6: ArmMotionExecutor"""

import time
import logging
from typing import Any, Dict, List
import numpy as np

from openrobot_demo.skills.base import SkillInterface, SkillSchema, ParamSchema, ResultSchema
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

    @property
    def schema(self) -> SkillSchema:
        return SkillSchema(
            description="Execute arm motion commands: joint position, cartesian pose, or gripper control.",
            parameters=[
                ParamSchema(
                    name="command_type",
                    type="str",
                    description="Type of motion command. Must be one of: 'joint', 'cartesian', 'gripper'.",
                    required=True,
                    example="cartesian",
                ),
                ParamSchema(
                    name="target_values",
                    type="list",
                    description="Target values. For 'joint': list of n_dof angles. For 'cartesian': [x,y,z,qx,qy,qz,qw]. For 'gripper': [position, force].",
                    required=True,
                    example=[0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0],
                ),
                ParamSchema(
                    name="speed",
                    type="float",
                    description="Motion speed factor (0.0 - 2.0). 1.0 is normal speed.",
                    required=False,
                    default=1.0,
                ),
                ParamSchema(
                    name="use_interpolation",
                    type="bool",
                    description="Whether to use smooth joint-space interpolation.",
                    required=False,
                    default=True,
                ),
            ],
            returns=[
                ResultSchema(name="success", type="bool", description="Whether motion succeeded."),
                ResultSchema(name="message", type="str", description="Status message."),
                ResultSchema(name="actual_reached_pos", type="list", description="Final joint positions after motion."),
                ResultSchema(name="execution_time_ms", type="int", description="Motion execution time in milliseconds."),
            ],
            dependencies=["arm"],
            preconditions=["arm must be enabled"],
            postconditions=["arm has reached (or attempted to reach) the target configuration"],
            examples=[
                {
                    "input": {"command_type": "gripper", "target_values": [0.0, 0.5]},
                    "output": {"success": True, "message": "Gripper set to 0.0."},
                },
                {
                    "input": {"command_type": "cartesian", "target_values": [0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], "speed": 0.5},
                    "output": {"success": True, "message": "Joint motion executed."},
                },
            ],
        )

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
            import math
            import time as _time
            n = len(target_values)
            if n == 3:
                # 3-DOF: default forward-facing orientation (arm zero pose is forward)
                target_7dof = list(target_values[:3]) + [0.0, 0.0, 0.0, 1.0]
            elif n == 6:
                # 6-DOF Euler -> convert to quaternion for interpolation
                from scipy.spatial.transform import Rotation as R
                rot = R.from_euler("xyz", target_values[3:6], degrees=False).as_quat()
                target_7dof = list(target_values[:3]) + list(rot)
            elif n == 7:
                target_7dof = target_values[:7]
            else:
                return {"success": False, "message": f"Invalid cartesian target length: {n} (expected 3, 6, or 7)"}

            xyz = target_7dof[:3]
            ok, reason = self._safety.check_cartesian_target(xyz)
            if not ok:
                return {"success": False, "message": f"Safety check failed: {reason}"}

            # Cartesian-space interpolation (same as SDK move_to_pose)
            current_joints = self._arm.get_pos()
            current_tcp = self._solver.forward_quat(current_joints)

            duration = 2.0 / max(0.1, speed)
            steps = max(1, int(duration / 0.02))
            for i in range(1, steps + 1):
                alpha = i / steps
                pos = [(1 - alpha) * current_tcp[j] + alpha * target_7dof[j] for j in range(3)]
                quat = [(1 - alpha) * current_tcp[3 + j] + alpha * target_7dof[3 + j] for j in range(4)]
                norm = math.sqrt(sum(q * q for q in quat))
                quat = [q / norm for q in quat]
                interp_pose = pos + quat
                if n == 6:
                    joints = self._solver.inverse_eular(target_values[:6], current_joints)
                else:
                    joints = self._solver.inverse_quat(interp_pose, current_joints)
                if joints is None:
                    return {"success": False, "message": f"IK failed at step {i}/{steps}"}
                self._arm.joint_control(joints[:6])
                _time.sleep(0.02)

            return {
                "success": True,
                "message": "Cartesian motion executed.",
                "execution_time_ms": int((_time.time() - start_t) * 1000),
            }

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
