"""YHRG S1 SDK adapter with graceful mock fallback for development environments.

When S1_SDK is not available (e.g., macOS or missing compiled extensions),
this module provides a compatible mock implementation based on the public
S1_SDK_V2 API documented in the official readme.
"""

from __future__ import annotations

import math
import logging
from typing import List, Optional
from enum import IntEnum
import numpy as np
from scipy.spatial.transform import Rotation as R

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
# Mock Kinematics (S1_slover compatible)
# ---------------------------------------------------------------------------
class _MockS1Slover:
    """Mock kinematics solver for YHRG S1 (7-DOF arm).

    Uses a simplified kinematic model good enough for MVP testing.
    """

    def __init__(self, end_effector_offset: List[float] = None):
        self.offset = np.array(end_effector_offset or [0.0, 0.0, 0.0])
        # Rough link lengths for a small desktop manipulator (meters)
        self.link_lengths = np.array([0.0, 0.15, 0.12, 0.10, 0.08, 0.05, 0.05])
        self.joint_axes = [
            np.array([0, 0, 1]),   # J1: base yaw
            np.array([0, 1, 0]),   # J2: shoulder pitch
            np.array([0, 0, 1]),   # J3: elbow yaw
            np.array([0, 1, 0]),   # J4: elbow pitch
            np.array([0, 0, 1]),   # J5: wrist yaw
            np.array([0, 1, 0]),   # J6: wrist pitch
            np.array([0, 0, 1]),   # J7: wrist roll
        ]

    def _fk_matrix(self, joints: List[float]) -> np.ndarray:
        """Return 4x4 homogeneous transform of end-effector."""
        T = np.eye(4)
        for i, theta in enumerate(joints):
            axis = self.joint_axes[i]
            trans = np.array([0.0, 0.0, self.link_lengths[i]])
            if i == 0:
                trans = np.array([0.0, 0.0, 0.08])  # base height
            # Rotation around axis by theta
            rotmat = R.from_rotvec(axis * theta).as_matrix()
            Ti = np.eye(4)
            Ti[:3, :3] = rotmat
            Ti[:3, 3] = trans
            T = T @ Ti
        # Apply end-effector offset
        T[:3, 3] += T[:3, :3] @ self.offset
        return T

    def forward_quat(self, joints: List[float]) -> List[float]:
        T = self._fk_matrix(joints)
        pos = T[:3, 3]
        quat = R.from_matrix(T[:3, :3]).as_quat()  # scipy: x,y,z,w
        # Return [x,y,z,qx,qy,qz,qw]
        return [float(pos[0]), float(pos[1]), float(pos[2]),
                float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]

    def forward_eular(self, joints: List[float]) -> List[float]:
        T = self._fk_matrix(joints)
        pos = T[:3, 3]
        euler = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=False)
        return [float(pos[0]), float(pos[1]), float(pos[2]),
                float(euler[0]), float(euler[1]), float(euler[2])]

    def inverse_quat(self, target_pose: List[float], joint_positions: List[float] = None) -> Optional[List[float]]:
        """Simple numerical IK using pseudo-inverse Jacobian."""
        target = np.array(target_pose)
        target_pos = target[:3]
        target_rot = R.from_quat(target[3:7]).as_matrix()
        # Initial guess
        q = np.array(joint_positions if joint_positions is not None else [0.0] * 7)
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
            # Damped least squares
            dq = J.T @ np.linalg.solve(J @ J.T + 0.01 * np.eye(6), err)
            q = q + 0.3 * dq
            # Clamp
            q = np.clip(q, [-2.967, 0.0, 0.0, -1.571, -1.571, -1.571, -1.745],
                        [2.967, 3.142, 2.967, 1.518, 1.571, 1.571, 1.745])
        return q.tolist()

    def inverse_eular(self, target_pose: List[float], joint_positions: List[float] = None) -> Optional[List[float]]:
        target = np.array(target_pose)
        pos = target[:3]
        euler = target[3:6]
        rot = R.from_euler("xyz", euler, degrees=False).as_quat()
        return self.inverse_quat([pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]], joint_positions)

    def _compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        delta = 1e-4
        J = np.zeros((6, 7))
        T0 = self._fk_matrix(q.tolist())
        pos0 = T0[:3, 3]
        rot0 = T0[:3, :3]
        for i in range(7):
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

    _JOINT_LIMITS = [
        (-2.967, 2.967),
        (0.0, 3.142),
        (0.0, 2.967),
        (-1.571, 1.518),
        (-1.571, 1.571),
        (-1.571, 1.571),
        (-1.745, 1.745),
    ]

    def __init__(self, mode: control_mode, dev: str = "/dev/ttyUSB0",
                 end_effector: str = "None", check_collision: bool = True,
                 arm_version: str = "V2"):
        self.mode = mode
        self.dev = dev
        self.end_effector = end_effector
        self.check_collision = check_collision
        self.arm_version = arm_version
        self._enabled = False
        self._pos = [0.0] * 7
        self._vel = [0.0] * 7
        self._tau = [0.0] * 7
        self._temp = [25.0] * 7
        self._gripper_pos = 0.0
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
        if len(pos) < 7:
            pos = list(pos) + [0.0] * (7 - len(pos))
        clamped = [clamp(pos[i], self._JOINT_LIMITS[i]) for i in range(7)]
        self._pos = clamped
        return True

    def joint_control_mit(self, pos: List[float]) -> bool:
        return self.joint_control(pos)

    def control_gripper(self, pos: float, force: float = 0.5):
        self._gripper_pos = float(max(0.0, min(2.0, pos)))
        logger.debug(f"[MockS1Arm] Gripper set to {self._gripper_pos}")

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
        logger.info("[MockS1Arm] Zero position set.")

    def set_end_zero_position(self):
        logger.info("[MockS1Arm] End zero position set.")

    def gravity(self, return_tau: bool = False):
        if return_tau:
            return [0.0] * 7
        logger.info("[MockS1Arm] Gravity compensation active (mock).")

    def check_collision(self, qpos: List[float]) -> bool:
        # Mock: no collision
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

__all__ = ["S1_arm", "S1_slover", "control_mode", "Arm_Search"]
