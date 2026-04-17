"""DualArmController: synchronized control of two YHRG S1 arms."""

import logging
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
        joints = self.get_pos(side)
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

    def move_cartesian(
        self,
        side: ArmSide,
        target_pose: List[float],
        duration: float = 1.0,
        current_joints: Optional[List[float]] = None,
    ):
        """Move one arm to target Cartesian pose via IK + joint interpolation."""
        kin = self.left_kin if side == ArmSide.LEFT else self.right_kin
        arm = self.left_arm if side == ArmSide.LEFT else self.right_arm
        q0 = current_joints if current_joints is not None else arm.get_pos()
        q_target = kin.inverse_quat(target_pose, q0)
        if q_target is None:
            raise RuntimeError(f"[DualArmController] IK failed for {side.value} arm to {target_pose}")
        self.move_joint(side, q_target, duration)

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

        Uses leader-follower interpolation: both arms step in lockstep.
        """
        left_q0 = self.left_arm.get_pos()
        right_q0 = self.right_arm.get_pos()
        left_qt = self.left_kin.inverse_quat(left_target, left_q0)
        right_qt = self.right_kin.inverse_quat(right_target, right_q0)

        if left_qt is None:
            raise RuntimeError("[DualArmController] Left arm IK failed")
        if right_qt is None:
            raise RuntimeError("[DualArmController] Right arm IK failed")

        left_q0_6 = left_q0[:6]
        right_q0_6 = right_q0[:6]
        left_qt_6 = left_qt[:6]
        right_qt_6 = right_qt[:6]

        steps = max(1, int(duration / 0.02))
        logger.info(
            "[DualArmController] Dual move: %d steps over %.2fs (sync_tol=%.3fm)",
            steps,
            duration,
            sync_tolerance_m,
        )

        for i in range(1, steps + 1):
            alpha = i / steps
            left_interp = [(1 - alpha) * c + alpha * t for c, t in zip(left_q0_6, left_qt_6)]
            right_interp = [(1 - alpha) * c + alpha * t for c, t in zip(right_q0_6, right_qt_6)]
            self.left_arm.joint_control(left_interp)
            self.right_arm.joint_control(right_interp)
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
        Synchronized dual-arm grasp sequence:
        1. Move both arms to pre-grasp poses (above targets)
        2. Descend to grasp poses
        3. Close grippers
        """
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
