"""YHRG S1 SDK adapter with graceful mock fallback for development environments.

When S1_SDK is not available (e.g., macOS or missing compiled extensions),
this module provides a compatible mock implementation based on the public
S1_SDK_V2 API documented in the official readme.

This adapter implements ManipulatorInterface for unified robot abstraction.
"""

from __future__ import annotations

import math
import logging
from typing import List, Optional
from enum import IntEnum
import numpy as np
from scipy.spatial.transform import Rotation as R

from openrobot_demo.hardware.manipulator_interface import ManipulatorInterface

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try importing the real SDK; otherwise fall back to mock
# ---------------------------------------------------------------------------
try:
    from S1_SDK import S1_arm as _RealS1Arm
    from S1_SDK import S1_slover as _RealS1Slover
    from S1_SDK import control_mode as _RealControlMode
    from S1_SDK import Arm_Search as _RealArmSearch

    _SDK_AVAILABLE = True
    logger.info("[YHRGAdapter] S1_SDK loaded successfully.")
except Exception as e:
    _SDK_AVAILABLE = False
    logger.warning(f"[YHRGAdapter] S1_SDK not available ({e}). Using MOCK mode.")


class control_mode(IntEnum):
    only_sim = 0
    only_real = 1
    real_control_sim = 2


# ---------------------------------------------------------------------------
# Mock Kinematics (S1_slover compatible) -- 6-DOF
# ---------------------------------------------------------------------------
class _MockS1Slover:
    """Mock kinematics solver for YHRG S1 (6-DOF arm).

    The real S1_slover operates on 6 joint angles.
    We accept 7 joints for backward compatibility but warn and truncate.
    """

    def __init__(self, end_effector_offset: List[float] = None):
        self.offset = np.array(end_effector_offset or [0.0, 0.0, 0.0])
        # Rough link lengths for a small desktop manipulator (meters) -- 6 DOF
        self.link_lengths = np.array([0.0, 0.15, 0.12, 0.10, 0.08, 0.05])
        self.joint_axes = [
            np.array([0, 0, 1]),   # J1: base yaw
            np.array([0, 1, 0]),   # J2: shoulder pitch
            np.array([0, 0, 1]),   # J3: elbow yaw
            np.array([0, 1, 0]),   # J4: elbow pitch
            np.array([0, 0, 1]),   # J5: wrist yaw
            np.array([0, 1, 0]),   # J6: wrist pitch
        ]

    def _ensure_6dof(self, joints: List[float]) -> List[float]:
        if len(joints) > 6:
            logger.warning(
                "[MockS1Slover] Received %d joints, truncating to 6-DOF for S1 compatibility.",
                len(joints),
            )
            return joints[:6]
        if len(joints) < 6:
            return list(joints) + [0.0] * (6 - len(joints))
        return joints

    def _fk_matrix(self, joints: List[float]) -> np.ndarray:
        """Return 4x4 homogeneous transform of end-effector."""
        joints = self._ensure_6dof(joints)
        T = np.eye(4)
        for i, theta in enumerate(joints):
            axis = self.joint_axes[i]
            trans = np.array([0.0, 0.0, self.link_lengths[i]])
            if i == 0:
                trans = np.array([0.0, 0.0, 0.08])  # base height
            rotmat = R.from_rotvec(axis * theta).as_matrix()
            Ti = np.eye(4)
            Ti[:3, :3] = rotmat
            Ti[:3, 3] = trans
            T = T @ Ti
        T[:3, 3] += T[:3, :3] @ self.offset
        return T

    def forward_quat(self, joints: List[float]) -> List[float]:
        T = self._fk_matrix(joints)
        pos = T[:3, 3]
        quat = R.from_matrix(T[:3, :3]).as_quat()  # scipy: x,y,z,w
        return [float(pos[0]), float(pos[1]), float(pos[2]),
                float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]

    def forward_eular(self, joints: List[float]) -> List[float]:
        T = self._fk_matrix(joints)
        pos = T[:3, 3]
        euler = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=False)
        return [float(pos[0]), float(pos[1]), float(pos[2]),
                float(euler[0]), float(euler[1]), float(euler[2])]

    def inverse_quat(self, target_pose: List[float], joint_positions: List[float] = None) -> Optional[List[float]]:
        target = np.array(target_pose)
        target_pos = target[:3]
        target_rot = R.from_quat(target[3:7]).as_matrix()
        q = np.array(joint_positions if joint_positions is not None else [0.0] * 6)
        q = np.array(self._ensure_6dof(q.tolist()))
        for _ in range(50):
            T = self._fk_matrix(q.tolist())
            pos = T[:3, 3]
            rot = T[:3, :3]
            pos_err = target_pos - pos
            rot_err_mat = target_rot @ rot.T
            rot_err_vec = R.from_matrix(rot_err_mat).as_rotvec()
            err = np.concatenate([pos_err, rot_err_vec])
            if np.linalg.norm(err) < 1e-3:
                break
            J = self._compute_jacobian(q)
            dq = J.T @ np.linalg.solve(J @ J.T + 0.01 * np.eye(6), err)
            q = q + 0.3 * dq
            q = np.clip(q, [-2.967, 0.0, 0.0, -1.571, -1.571, -1.571],
                        [2.967, 3.142, 2.967, 1.518, 1.571, 1.571])
        return q.tolist()

    def inverse_eular(self, target_pose: List[float], joint_positions: List[float] = None) -> Optional[List[float]]:
        target = np.array(target_pose)
        pos = target[:3]
        euler = target[3:6]
        rot = R.from_euler("xyz", euler, degrees=False).as_quat()
        return self.inverse_quat([pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]], joint_positions)

    def _compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        delta = 1e-4
        J = np.zeros((6, 6))
        T0 = self._fk_matrix(q.tolist())
        pos0 = T0[:3, 3]
        rot0 = T0[:3, :3]
        for i in range(6):
            qd = q.copy()
            qd[i] += delta
            Td = self._fk_matrix(qd.tolist())
            posd = Td[:3, 3]
            rotd = Td[:3, :3]
            J[:3, i] = (posd - pos0) / delta
            dR = rotd @ rot0.T
            J[3:, i] = R.from_matrix(dR).as_rotvec() / delta
        return J


# ---------------------------------------------------------------------------
# Mock Arm (S1_arm compatible)
# ---------------------------------------------------------------------------
class _MockS1Arm:
    """Mock implementation of S1_arm for development without real hardware."""

    # 6-DOF joint limits (radians)
    _JOINT_LIMITS = [
        (-2.967, 2.967),
        (0.0, 3.142),
        (0.0, 2.967),
        (-1.571, 1.518),
        (-1.571, 1.571),
        (-1.571, 1.571),
    ]
    # 7th value in get_pos() is gripper aperture (0=closed ~ 2=open)
    _GRIPPER_LIMIT = (0.0, 2.0)

    def __init__(self, mode: control_mode, dev: str = "/dev/ttyUSB0",
                 end_effector: str = "None", check_collision: bool = True,
                 arm_version: str = "V2"):
        self.mode = mode
        self.dev = dev
        self.end_effector = end_effector
        self.check_collision = check_collision
        self.arm_version = arm_version
        self._enabled = False
        self._pos = [0.0] * 7   # [j1..j6, gripper_pos]
        self._vel = [0.0] * 7
        self._tau = [0.0] * 7
        self._temp = [25.0] * 7
        self._solver = _MockS1Slover([0.0, 0.0, 0.0])
        logger.info(f"[MockS1Arm] Initialized in mode={mode.name}, end_effector={end_effector}")

    def enable(self):
        self._enabled = True
        logger.info("[MockS1Arm] Enabled motors.")
        return True

    def disable(self):
        self._enabled = False
        logger.info("[MockS1Arm] Disabled motors.")
        return True

    def joint_control(self, pos: List[float]) -> bool:
        if not self._enabled and self.mode == control_mode.only_real:
            logger.warning("[MockS1Arm] joint_control ignored: motors not enabled.")
            return False
        # Real S1 joint_control expects 6 DOFs
        if len(pos) >= 6:
            controlled = pos[:6]
        else:
            controlled = list(pos) + [0.0] * (6 - len(pos))
        clamped = [clamp(controlled[i], self._JOINT_LIMITS[i]) for i in range(6)]
        self._pos[:6] = clamped
        return True

    def joint_control_mit(self, pos: List[float]) -> bool:
        return self.joint_control(pos)

    def control_gripper(self, pos: float, force: float = 0.5):
        self._pos[6] = float(max(self._GRIPPER_LIMIT[0], min(self._GRIPPER_LIMIT[1], pos)))
        logger.debug(f"[MockS1Arm] Gripper set to {self._pos[6]}")

    def get_pos(self) -> List[float]:
        return self._pos.copy()

    def get_vel(self) -> List[float]:
        return self._vel.copy()

    def get_tau(self) -> List[float]:
        return self._tau.copy()

    def get_temp(self) -> List[float]:
        return self._temp.copy()

    def set_zero_position(self):
        self._pos = [0.0] * 7
        self._pos[6] = 0.0  # gripper closed at zero position
        logger.info("[MockS1Arm] Zero position set.")

    def set_end_zero_position(self):
        logger.info("[MockS1Arm] End zero position set.")

    def gravity(self, return_tau: bool = False):
        if return_tau:
            return [0.0] * 7
        logger.info("[MockS1Arm] Gravity compensation active (mock).")

    def check_collision(self, qpos: List[float]) -> bool:
        return False

    def close(self):
        self.disable()


def clamp(value, checker):
    lo, hi = checker
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _mock_arm_search(bus: str, end_effector: str = "None", arm_version: str = "V2") -> bool:
    logger.info(f"[MockArmSearch] Would search {bus} (mock returns True).")
    return True


# ---------------------------------------------------------------------------
# Public exports (match S1_SDK API)
# ---------------------------------------------------------------------------
if _SDK_AVAILABLE:
    S1_arm = _RealS1Arm
    S1_slover = _RealS1Slover
    Arm_Search = _RealArmSearch
else:
    S1_arm = _MockS1Arm
    S1_slover = _MockS1Slover
    Arm_Search = _mock_arm_search


# ---------------------------------------------------------------------------
# Unified Adapter classes (compatible with OpenRobotDemo skill interfaces)
# ---------------------------------------------------------------------------
class YHRGAdapter(ManipulatorInterface):
    """
    Unified adapter for YHRG S1 arm, implementing ManipulatorInterface.
    Supports both real hardware and mock modes.
    """

    def __init__(
        self,
        mode: str = "mock",
        dev: str = "/dev/ttyUSB0",
        end_effector: str = "gripper",
    ):
        self._mode = mode
        self._dev = dev
        self._end_effector = end_effector
        self._enabled = False

        if self._mode == "real":
            if not _SDK_AVAILABLE:
                logger.warning(
                    "[YHRGAdapter] Real mode requested but S1_SDK is not available. "
                    "Falling back to mock mode."
                )
                self._mode = "mock"
            else:
                self._arm = _RealS1Arm(
                    mode=_RealControlMode.only_real,
                    dev=dev,
                    end_effector=end_effector,
                )
                logger.info("[YHRGAdapter] Real S1 arm initialized on %s", dev)

        if self._mode == "mock":
            self._arm = _MockS1Arm(
                mode=control_mode.only_real,
                dev=dev,
                end_effector=end_effector,
            )
            logger.info("[YHRGAdapter] Mock S1 arm initialized")

    def enable(self):
        self._enabled = True
        return self._arm.enable()

    def disable(self):
        self._enabled = False
        return self._arm.disable()

    # ------------------------------------------------------------------
    # RobotInterface properties
    # ------------------------------------------------------------------
    @property
    def robot_id(self) -> str:
        return f"yhrg_s1_{self._dev.replace('/', '_')}"

    @property
    def dof(self) -> int:
        return 6  # S1 is 6-DOF arm

    # ------------------------------------------------------------------
    # Legacy S1-style API (preserved for backward compatibility)
    # ------------------------------------------------------------------
    def get_pos(self) -> List[float]:
        return self._arm.get_pos()

    def get_vel(self) -> List[float]:
        return self._arm.get_vel()

    def get_tau(self) -> List[float]:
        return self._arm.get_tau()

    def get_temp(self) -> List[float]:
        return self._arm.get_temp()

    def joint_control(self, pos: List[float]) -> bool:
        # S1 joint_control expects 6 DOFs; truncate if 7 are given
        pos_6 = pos[:6] if len(pos) >= 6 else list(pos) + [0.0] * (6 - len(pos))
        return self._arm.joint_control(pos_6)

    def joint_control_mit(self, pos: List[float]) -> bool:
        return self._arm.joint_control_mit(pos)

    def set_zero_position(self):
        return self._arm.set_zero_position()

    def set_end_zero_position(self):
        return self._arm.set_end_zero_position()

    def gravity(self, return_tau: bool = False):
        return self._arm.gravity(return_tau=return_tau)

    def check_collision(self, qpos: List[float]) -> bool:
        return self._arm.check_collision(qpos)

    # ------------------------------------------------------------------
    # ManipulatorInterface implementation
    # ------------------------------------------------------------------
    def forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        """Use YHRGKinematics for FK."""
        # Lazy-init kinematics if not already present
        if not hasattr(self, "_kin"):
            self._kin = YHRGKinematics(end_effector_offset=[0.0, 0.0, 0.0])
        pose = self._kin.forward_quat(joint_positions.tolist())
        return np.array(pose, dtype=np.float32)

    def inverse_kinematics(
        self,
        target_pose: np.ndarray,
        current_joints: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        if not hasattr(self, "_kin"):
            self._kin = YHRGKinematics(end_effector_offset=[0.0, 0.0, 0.0])
        q0 = current_joints.tolist() if current_joints is not None else None
        result = self._kin.inverse_quat(target_pose.tolist(), q0)
        return np.array(result, dtype=np.float32) if result is not None else None

    def get_joint_positions(self) -> np.ndarray:
        return np.array(self._arm.get_pos()[:6], dtype=np.float32)

    def get_joint_velocities(self) -> np.ndarray:
        return np.array(self._arm.get_vel()[:6], dtype=np.float32)

    def get_joint_torques(self) -> np.ndarray:
        return np.array(self._arm.get_tau()[:6], dtype=np.float32)

    def get_end_effector_pose(self) -> np.ndarray:
        return self.forward_kinematics(self.get_joint_positions())

    def get_gripper_width(self) -> float:
        return float(self._arm.get_pos()[6])

    def set_joint_positions(self, positions: np.ndarray, **kwargs) -> bool:
        pos_list = positions.tolist() if hasattr(positions, "tolist") else list(positions)
        return self.joint_control(pos_list)

    def set_cartesian_pose(self, pose: np.ndarray, **kwargs) -> bool:
        q = self.inverse_kinematics(pose, current_joints=self.get_joint_positions())
        if q is None:
            logger.error("[YHRGAdapter] IK failed for cartesian pose %s", pose)
            return False
        return self.set_joint_positions(q, **kwargs)

    def control_gripper(self, position: float, force: Optional[float] = None) -> bool:
        self._arm.control_gripper(position, force=force or 0.5)
        return True

    def reset(self) -> bool:
        self._arm.set_zero_position()
        return True

    def is_ready(self) -> bool:
        return self._enabled

    def close(self) -> None:
        return self._arm.close()


class YHRGKinematics:
    """
    Unified kinematics solver for YHRG S1, compatible with FrankaMujocoKinematics interface.
    """

    def __init__(self, end_effector_offset: List[float] = None):
        self.offset = end_effector_offset or [0.0, 0.0, 0.0]
        if _SDK_AVAILABLE:
            self._solver = _RealS1Slover(self.offset)
        else:
            self._solver = _MockS1Slover(self.offset)

    def _ensure_6dof(self, joints: List[float]) -> List[float]:
        if len(joints) > 6:
            logger.warning(
                "[YHRGKinematics] Received %d joints, truncating to 6-DOF for S1 compatibility.",
                len(joints),
            )
            return joints[:6]
        if len(joints) < 6:
            return list(joints) + [0.0] * (6 - len(joints))
        return joints

    def forward_quat(self, joints: List[float]) -> List[float]:
        joints_6 = self._ensure_6dof(joints)
        return self._solver.forward_quat(joints_6)

    def forward_eular(self, joints: List[float]) -> List[float]:
        joints_6 = self._ensure_6dof(joints)
        return self._solver.forward_eular(joints_6)

    def inverse_quat(self, target_pose: List[float], joint_positions: List[float] = None) -> Optional[List[float]]:
        if joint_positions is not None:
            joint_positions = self._ensure_6dof(joint_positions)
        result = self._solver.inverse_quat(target_pose, joint_positions)
        # Return 7 values for backward compatibility with 7-DOF-centric code
        if result is not None:
            return list(result) + [0.0]
        return None

    def inverse_eular(self, target_pose: List[float], joint_positions: List[float] = None) -> Optional[List[float]]:
        if joint_positions is not None:
            joint_positions = self._ensure_6dof(joint_positions)
        result = self._solver.inverse_eular(target_pose, joint_positions)
        if result is not None:
            return list(result) + [0.0]
        return None


__all__ = [
    "S1_arm",
    "S1_slover",
    "control_mode",
    "Arm_Search",
    "YHRGAdapter",
    "YHRGKinematics",
]
