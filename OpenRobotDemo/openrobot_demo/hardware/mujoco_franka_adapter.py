"""MuJoCo-backed adapter for a simplified Franka arm, compatible with S1-style API.

This allows OpenRobotDemo skills to control the MuJoCo Franka model without
modifying the skill code.

Implements ManipulatorInterface for unified robot abstraction.
"""

import logging
from typing import List, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R

from openrobot_demo.hardware.manipulator_interface import ManipulatorInterface

logger = logging.getLogger(__name__)


class FrankaMujocoAdapter(ManipulatorInterface):
    """Wraps a MuJoCo model/data pair to control the Franka arm."""

    def __init__(self, model, data, end_effector: str = "gripper"):
        import mujoco
        self._model = model
        self._data = data
        self._end_effector = end_effector
        self._enabled = False
        self._joint_names = [f"joint{i}" for i in range(1, 8)]
        self._joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self._joint_names
        ]
        # qpos addresses for the 7 arm joints
        self._qpos_adr = [model.jnt_qposadr[jid] for jid in self._joint_ids]
        # actuators for the 7 arm joints + 2 finger joints
        self._nu_arm = 7
        self._nu_fingers = 2
        self._solver = FrankaMujocoKinematics(model, data, end_effector_offset=[0.0, 0.0, 0.0])

    # ------------------------------------------------------------------
    # RobotInterface properties
    # ------------------------------------------------------------------
    @property
    def robot_id(self) -> str:
        return "franka_mujoco"

    @property
    def dof(self) -> int:
        return 7

    # ------------------------------------------------------------------
    # Legacy API (preserved for backward compatibility)
    # ------------------------------------------------------------------
    def enable(self):
        self._enabled = True
        return True

    def disable(self):
        self._enabled = False
        return True

    def get_pos(self) -> List[float]:
        return [float(self._data.qpos[adr]) for adr in self._qpos_adr]

    def get_vel(self) -> List[float]:
        return [float(self._data.qvel[self._model.jnt_dofadr[jid]]) for jid in self._joint_ids]

    def get_tau(self) -> List[float]:
        return [float(self._data.actuator_force[i]) for i in range(self._nu_arm)]

    def get_temp(self) -> List[float]:
        return [0.0] * self._nu_arm

    def joint_control(self, pos: List[float]) -> bool:
        if len(pos) < self._nu_arm:
            pos = list(pos) + [0.0] * (self._nu_arm - len(pos))
        for i in range(self._nu_arm):
            lo, hi = self._model.actuator_ctrlrange[i]
            pos[i] = float(np.clip(pos[i], lo, hi))
        self._data.ctrl[:self._nu_arm] = pos[:self._nu_arm]
        return True

    def joint_control_mit(self, pos: List[float]) -> bool:
        return self.joint_control(pos)

    def set_zero_position(self):
        for adr in self._qpos_adr:
            self._data.qpos[adr] = 0.0
        self._data.ctrl[:self._nu_arm] = 0.0

    def set_end_zero_position(self):
        pass

    def gravity(self, return_tau: bool = False):
        if return_tau:
            return [0.0] * self._nu_arm

    def check_collision(self, qpos: List[float]) -> bool:
        return False

    # ------------------------------------------------------------------
    # ManipulatorInterface implementation
    # ------------------------------------------------------------------
    def forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        return np.array(self._solver.forward_quat(joint_positions.tolist()), dtype=np.float32)

    def inverse_kinematics(
        self,
        target_pose: np.ndarray,
        current_joints: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        q0 = current_joints.tolist() if current_joints is not None else None
        result = self._solver.inverse_quat(target_pose.tolist(), q0)
        return np.array(result, dtype=np.float32) if result is not None else None

    def get_joint_positions(self) -> np.ndarray:
        return np.array(self.get_pos(), dtype=np.float32)

    def get_joint_velocities(self) -> np.ndarray:
        return np.array(self.get_vel(), dtype=np.float32)

    def get_joint_torques(self) -> np.ndarray:
        return np.array(self.get_tau(), dtype=np.float32)

    def get_end_effector_pose(self) -> np.ndarray:
        return self.forward_kinematics(self.get_joint_positions())

    def get_gripper_width(self) -> float:
        # Franka fingers: actuator 7 and 8 control widths; approximate
        f1 = float(self._data.ctrl[self._nu_arm])
        f2 = float(self._data.ctrl[self._nu_arm + 1])
        return (f1 + f2) / 2.0

    def set_joint_positions(self, positions: np.ndarray, **kwargs) -> bool:
        pos_list = positions.tolist() if hasattr(positions, "tolist") else list(positions)
        return self.joint_control(pos_list)

    def set_cartesian_pose(self, pose: np.ndarray, **kwargs) -> bool:
        q = self.inverse_kinematics(pose, current_joints=self.get_joint_positions())
        if q is None:
            logger.error("[FrankaMujocoAdapter] IK failed for cartesian pose %s", pose)
            return False
        return self.set_joint_positions(q, **kwargs)

    def control_gripper(self, position: float, force: Optional[float] = None) -> bool:
        g = float(np.clip(position, 0.0, 0.04))
        self._data.ctrl[self._nu_arm] = g
        self._data.ctrl[self._nu_arm + 1] = g
        return True

    def reset(self) -> bool:
        self.set_zero_position()
        return True

    def is_ready(self) -> bool:
        return self._enabled

    def close(self) -> None:
        self.disable()


class FrankaMujocoKinematics:
    """Kinematics for the MuJoCo Franka model using numerical IK."""

    def __init__(self, model, data, end_effector_offset=None):
        self.model = model
        self.data = data
        self.offset = np.array(end_effector_offset or [0.0, 0.0, 0.0])
        self._body_name = "gripper_base"
        import mujoco
        self._body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self._body_name)
        self._joint_names = [f"joint{i}" for i in range(1, 8)]
        self._joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self._joint_names
        ]
        self._qpos_adr = [model.jnt_qposadr[jid] for jid in self._joint_ids]

    def forward_quat(self, joints: List[float]) -> List[float]:
        # Set joints, run forward kinematics, read body pose
        for adr, val in zip(self._qpos_adr, joints):
            self.data.qpos[adr] = val
        import mujoco
        mujoco.mj_forward(self.model, self.data)
        xpos = self.data.xpos[self._body_id].copy()
        # MuJoCo xmat is flattened row-major 9
        xmat = self.data.xmat[self._body_id].reshape(3, 3).copy()
        # Apply offset along local z (gripper approach)
        xpos += xmat @ self.offset
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, xmat.flatten())
        # MuJoCo gives [w, x, y, z]; convert to [x, y, z, w] for scipy compatibility
        return [float(xpos[0]), float(xpos[1]), float(xpos[2]),
                float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])]

    def forward_eular(self, joints: List[float]) -> List[float]:
        pose_quat = self.forward_quat(joints)
        euler = R.from_quat(pose_quat[3:7]).as_euler("xyz", degrees=False)
        return pose_quat[:3] + euler.tolist()

    def inverse_quat(self, target_pose: List[float], joint_positions: List[float] = None) -> Optional[List[float]]:
        target = np.array(target_pose)
        target_pos = target[:3]
        target_rot = R.from_quat(target[3:7]).as_matrix()
        q = np.array(joint_positions if joint_positions is not None else self._get_current_joints())

        for _ in range(100):
            pose = self.forward_quat(q.tolist())
            pos = np.array(pose[:3])
            rot = R.from_quat(pose[3:7]).as_matrix()
            pos_err = target_pos - pos
            rot_err_mat = target_rot @ rot.T
            rot_err_vec = R.from_matrix(rot_err_mat).as_rotvec()
            err = np.concatenate([pos_err, rot_err_vec])
            if np.linalg.norm(err) < 1e-3:
                break
            J = self._compute_jacobian(q)
            dq = J.T @ np.linalg.solve(J @ J.T + 0.01 * np.eye(6), err)
            q = q + 0.5 * dq
            # Clip to joint limits
            for i in range(7):
                lo, hi = self.model.jnt_range[self._joint_ids[i]]
                q[i] = np.clip(q[i], lo, hi)

        # Final forward check
        final_pose = self.forward_quat(q.tolist())
        final_pos = np.array(final_pose[:3])
        if np.linalg.norm(final_pos - target_pos) > 0.02:
            logger.warning("IK did not converge to target position.")
            return None
        return q.tolist()

    def inverse_eular(self, target_pose: List[float], joint_positions: List[float] = None) -> Optional[List[float]]:
        target = np.array(target_pose)
        pos = target[:3]
        euler = target[3:6]
        rot = R.from_euler("xyz", euler, degrees=False).as_quat()
        return self.inverse_quat([pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]], joint_positions)

    def _get_current_joints(self) -> List[float]:
        return [float(self.data.qpos[adr]) for adr in self._qpos_adr]

    def _compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        delta = 1e-4
        J = np.zeros((6, 7))
        T0 = self.forward_quat(q.tolist())
        pos0 = np.array(T0[:3])
        rot0 = R.from_quat(T0[3:7]).as_matrix()
        for i in range(7):
            qd = q.copy()
            qd[i] += delta
            Td = self.forward_quat(qd.tolist())
            posd = np.array(Td[:3])
            rotd = R.from_quat(Td[3:7]).as_matrix()
            J[:3, i] = (posd - pos0) / delta
            dR = rotd @ rot0.T
            J[3:, i] = R.from_matrix(dR).as_rotvec() / delta
        return J
