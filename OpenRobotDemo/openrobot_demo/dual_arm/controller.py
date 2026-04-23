"""DualArmController: synchronized control of two YHRG S1 arms."""

import logging
import math
import time
from enum import Enum
from typing import List, Optional

import numpy as np

from openrobot_demo.hardware.yhrg_adapter import YHRGAdapter, YHRGKinematics

logger = logging.getLogger(__name__)


class ArmSide(Enum):
    LEFT = "left"
    RIGHT = "right"


class DualArmController:
    """
    Coordinated dual-arm controller for YHRG S1 robots.

    Each arm is 6-DOF (S1 spec); controller accepts 7-DOF inputs for
    Franka compatibility but internally truncates to 6.
    """

    def __init__(
        self,
        left_dev: str = "/dev/ttyUSB0",
        right_dev: str = "/dev/ttyUSB1",
        mode: str = "mock",  # "mock" or "real"
        end_effector: str = "gripper",
    ):
        self.mode = mode
        self.left_arm = YHRGAdapter(mode=mode, dev=left_dev, end_effector=end_effector)
        self.right_arm = YHRGAdapter(mode=mode, dev=right_dev, end_effector=end_effector)
        self.left_kin = YHRGKinematics(end_effector_offset=[0.0, 0.0, 0.0])
        self.right_kin = YHRGKinematics(end_effector_offset=[0.0, 0.0, 0.0])

        self._enabled = False
        logger.info("[DualArmController] Initialized (%s mode)", mode)

    def enable(self):
        self.left_arm.enable()
        self.right_arm.enable()
        self._enabled = True
        logger.info("[DualArmController] Both arms enabled")

    def disable(self):
        self.left_arm.disable()
        self.right_arm.disable()
        self._enabled = False
        logger.info("[DualArmController] Both arms disabled")

    def get_pos(self, side: ArmSide) -> List[float]:
        arm = self.left_arm if side == ArmSide.LEFT else self.right_arm
        return arm.get_pos()

    def get_ee_pose(self, side: ArmSide) -> List[float]:
        """Return current end-effector pose [x,y,z,qx,qy,qz,qw]."""
        kin = self.left_kin if side == ArmSide.LEFT else self.right_kin
        joints = self.get_pos(side)[:6]
        return kin.forward_quat(joints)

    # ------------------------------------------------------------------
    # Single-arm motion primitives
    # ------------------------------------------------------------------
    def move_joint(self, side: ArmSide, target_joints: List[float], duration: float = 1.0):
        """Move one arm to target joint positions with linear interpolation."""
        arm = self.left_arm if side == ArmSide.LEFT else self.right_arm
        current = arm.get_pos()[:6]
        target = target_joints[:6]
        steps = max(1, int(duration / 0.02))
        for i in range(1, steps + 1):
            alpha = i / steps
            interp = [(1 - alpha) * c + alpha * t for c, t in zip(current, target)]
            arm.joint_control(interp)
            time.sleep(0.02)

    def _cartesian_interpolate(self, side: ArmSide, target_pose_7dof: List[float],
                                duration: float, q0: List[float]):
        """Cartesian-space linear interpolation (position + quaternion).
        Mimics SDK SingleArmController.move_to_pose().
        """
        kin = self.left_kin if side == ArmSide.LEFT else self.right_kin
        arm = self.left_arm if side == ArmSide.LEFT else self.right_arm
        current_tcp = kin.forward_quat(q0)

        steps = max(1, int(duration / 0.02))
        for i in range(1, steps + 1):
            alpha = i / steps
            # Position linear interpolation
            pos = [(1 - alpha) * current_tcp[j] + alpha * target_pose_7dof[j] for j in range(3)]
            # Quaternion linear interpolation + normalize (same as SDK)
            quat = [(1 - alpha) * current_tcp[3 + j] + alpha * target_pose_7dof[3 + j] for j in range(4)]
            norm = math.sqrt(sum(q * q for q in quat))
            quat = [q / norm for q in quat]

            interp_pose = pos + quat
            q_target = kin.inverse_quat(interp_pose, q0)
            if q_target is None:
                raise RuntimeError(
                    f"[DualArmController] IK failed at step {i}/{steps} for {side.value} arm"
                )
            arm.joint_control(q_target[:6])
            time.sleep(0.02)

    def move_cartesian(
        self,
        side: ArmSide,
        target_pose: List[float],
        duration: float = 1.0,
        current_joints: Optional[List[float]] = None,
    ):
        """Move one arm to target Cartesian pose via Cartesian-space interpolation + IK.

        Accepts:
          - 3-DOF position [x, y, z]  -> default forward-facing orientation (identity)
          - 6-DOF Euler  [x, y, z, rx, ry, rz]  -> uses inverse_eular
          - 7-DOF Quaternion [x, y, z, qx, qy, qz, qw]  -> uses inverse_quat
        """
        arm = self.left_arm if side == ArmSide.LEFT else self.right_arm
        q0 = current_joints if current_joints is not None else arm.get_pos()[:6]

        # Parse target to 7-DOF quaternion for Cartesian interpolation
        target_7dof = None
        if len(target_pose) == 3:
            # 3-DOF: default forward-facing orientation (arm zero pose is forward)
            target_7dof = list(target_pose) + [0.0, 0.0, 0.0, 1.0]
        elif len(target_pose) == 6:
            # 6-DOF Euler -> convert to quaternion
            from scipy.spatial.transform import Rotation as R
            rot = R.from_euler("xyz", target_pose[3:6], degrees=False).as_quat()
            target_7dof = list(target_pose[:3]) + list(rot)
        elif len(target_pose) == 7:
            target_7dof = target_pose[:7]
        else:
            raise ValueError(f"[DualArmController] Invalid target_pose length: {len(target_pose)} (expected 3, 6, or 7)")

        self._cartesian_interpolate(side, target_7dof, duration, q0)

    # ------------------------------------------------------------------
    # Dual-arm synchronized motion
    # ------------------------------------------------------------------
    def dual_move_cartesian(
        self,
        left_target: List[float],
        right_target: List[float],
        duration: float = 1.0,
        sync_tolerance_m: float = 0.002,
    ):
        """
        Move both arms to their respective Cartesian targets synchronously.

        Uses Cartesian-space interpolation (same as SDK) with lockstep execution.
        Accepts 3-DOF, 6-DOF Euler, or 7-DOF Quaternion for each arm.
        """
        def _to_7dof(target):
            if len(target) == 3:
                return list(target) + [0.0, 0.0, 0.0, 1.0]
            elif len(target) == 6:
                from scipy.spatial.transform import Rotation as R
                rot = R.from_euler("xyz", target[3:6], degrees=False).as_quat()
                return list(target[:3]) + list(rot)
            elif len(target) == 7:
                return target[:7]
            else:
                raise ValueError(f"Invalid target length: {len(target)} (expected 3, 6, or 7)")

        left_q0 = self.left_arm.get_pos()[:6]
        right_q0 = self.right_arm.get_pos()[:6]
        left_target_7dof = _to_7dof(left_target)
        right_target_7dof = _to_7dof(right_target)

        left_tcp = self.left_kin.forward_quat(left_q0)
        right_tcp = self.right_kin.forward_quat(right_q0)

        steps = max(1, int(duration / 0.02))
        logger.info(
            "[DualArmController] Dual move: %d steps over %.2fs (sync_tol=%.3fm)",
            steps,
            duration,
            sync_tolerance_m,
        )

        for i in range(1, steps + 1):
            alpha = i / steps
            # Left arm interpolation
            l_pos = [(1 - alpha) * left_tcp[j] + alpha * left_target_7dof[j] for j in range(3)]
            l_quat = [(1 - alpha) * left_tcp[3 + j] + alpha * left_target_7dof[3 + j] for j in range(4)]
            l_norm = math.sqrt(sum(q * q for q in l_quat))
            l_quat = [q / l_norm for q in l_quat]
            l_qt = self.left_kin.inverse_quat(l_pos + l_quat, left_q0)
            if l_qt is None:
                raise RuntimeError(f"[DualArmController] Left arm IK failed at step {i}/{steps}")

            # Right arm interpolation
            r_pos = [(1 - alpha) * right_tcp[j] + alpha * right_target_7dof[j] for j in range(3)]
            r_quat = [(1 - alpha) * right_tcp[3 + j] + alpha * right_target_7dof[3 + j] for j in range(4)]
            r_norm = math.sqrt(sum(q * q for q in r_quat))
            r_quat = [q / r_norm for q in r_quat]
            r_qt = self.right_kin.inverse_quat(r_pos + r_quat, right_q0)
            if r_qt is None:
                raise RuntimeError(f"[DualArmController] Right arm IK failed at step {i}/{steps}")

            self.left_arm.joint_control(l_qt[:6])
            self.right_arm.joint_control(r_qt[:6])
            time.sleep(0.02)

    def dual_grasp(
        self,
        left_target: List[float],
        right_target: List[float],
        gripper_close_pos: float = 0.0,
        approach_duration: float = 1.0,
        grasp_duration: float = 0.5,
    ):
        """
        Synchronized dual-arm grasp sequence.
        If 3-DOF targets are given, they are converted to downward-facing 7-DOF poses
        (grasping typically requires downward orientation).
        """
        # Ensure downward orientation for grasping if 3-DOF is given
        ORIENTATION_DOWNWARD = [-0.7071, 0.0, 0.0, 0.7071]
        if len(left_target) == 3:
            left_target = list(left_target) + ORIENTATION_DOWNWARD
        if len(right_target) == 3:
            right_target = list(right_target) + ORIENTATION_DOWNWARD

        # Pre-grasp: offset Z by +5cm
        left_pre = left_target.copy()
        left_pre[2] += 0.05
        right_pre = right_target.copy()
        right_pre[2] += 0.05

        logger.info("[DualArmController] Step 1/3: approach pre-grasp")
        self.dual_move_cartesian(left_pre, right_pre, duration=approach_duration)

        logger.info("[DualArmController] Step 2/3: descend to grasp")
        self.dual_move_cartesian(left_target, right_target, duration=grasp_duration)

        logger.info("[DualArmController] Step 3/3: close grippers")
        self.left_arm.control_gripper(gripper_close_pos, force=0.5)
        self.right_arm.control_gripper(gripper_close_pos, force=0.5)
        time.sleep(0.3)

    def dual_lift(self, height_m: float, duration: float = 1.0):
        """Lift both arms vertically by height_m while maintaining relative pose."""
        left_pose = self.get_ee_pose(ArmSide.LEFT)
        right_pose = self.get_ee_pose(ArmSide.RIGHT)
        left_target = left_pose.copy()
        right_target = right_pose.copy()
        left_target[2] += height_m
        right_target[2] += height_m
        self.dual_move_cartesian(left_target, right_target, duration)

    def dual_release(self, gripper_open_pos: float = 1.0):
        """Open both grippers. Optional sequence: right first, then left."""
        self.right_arm.control_gripper(gripper_open_pos, force=0.5)
        time.sleep(0.2)
        self.left_arm.control_gripper(gripper_open_pos, force=0.5)
        time.sleep(0.2)
        logger.info("[DualArmController] Grippers released")

    def close(self):
        self.disable()
        self.left_arm.close()
        self.right_arm.close()
