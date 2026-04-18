"""ManipulatorInterface: extension of RobotInterface for articulated arms.

Covers serial arms (6-DOF, 7-DOF), SCARA, Delta, collaborative arms,
and any end-effector that moves in Cartesian space.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import List, Optional

import numpy as np

from .robot_interface import Action, Observation, RobotInterface, Space


class ManipulatorInterface(RobotInterface):
    """Interface for manipulator arms with forward/inverse kinematics.

    In addition to the generic RobotInterface, manipulators provide:
    - Forward / inverse kinematics
    - Joint-level and Cartesian-level commands
    - End-effector state queries
    - Gripper control
    """

    @property
    def robot_type(self) -> str:
        return "manipulator"

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------
    @abstractmethod
    def forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        """Return end-effector pose [x, y, z, qx, qy, qz, qw].

        Args:
            joint_positions: shape (n_dof,)

        Returns:
            pose: shape (7,)  [position (3) + quaternion (4)]
        """
        ...

    @abstractmethod
    def inverse_kinematics(
        self,
        target_pose: np.ndarray,
        current_joints: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Solve IK for a target Cartesian pose.

        Args:
            target_pose: shape (7,) [x, y, z, qx, qy, qz, qw]
            current_joints: seed for IK solver, shape (n_dof,)

        Returns:
            joint_positions: shape (n_dof,) or None if unreachable.
        """
        ...

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------
    @abstractmethod
    def get_joint_positions(self) -> np.ndarray:
        """Return current joint positions, shape (n_dof,)."""
        ...

    @abstractmethod
    def get_joint_velocities(self) -> np.ndarray:
        """Return current joint velocities, shape (n_dof,)."""
        ...

    @abstractmethod
    def get_joint_torques(self) -> np.ndarray:
        """Return current joint torques, shape (n_dof,)."""
        ...

    @abstractmethod
    def get_end_effector_pose(self) -> np.ndarray:
        """Return current EE pose [x, y, z, qx, qy, qz, qw], shape (7,)."""
        ...

    @abstractmethod
    def get_gripper_width(self) -> float:
        """Return current gripper aperture (0=closed, >0=open)."""
        ...

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------
    @abstractmethod
    def set_joint_positions(self, positions: np.ndarray, **kwargs) -> bool:
        """Command joints to target positions.

        Args:
            positions: shape (n_dof,)
            kwargs: implementation-specific (speed, acceleration, etc.)
        """
        ...

    @abstractmethod
    def set_cartesian_pose(self, pose: np.ndarray, **kwargs) -> bool:
        """Command end-effector to target pose via IK.

        Args:
            pose: shape (7,) [x, y, z, qx, qy, qz, qw]
            kwargs: implementation-specific
        """
        ...

    @abstractmethod
    def control_gripper(self, position: float, force: Optional[float] = None) -> bool:
        """Command gripper aperture.

        Args:
            position: target aperture (0=closed, implementation-defined max=fully open)
            force: optional force/effort limit
        """
        ...

    # ------------------------------------------------------------------
    # Generic RobotInterface bridge
    # ------------------------------------------------------------------
    def command(self, action: Action) -> bool:
        """Dispatch generic Action to manipulator-specific commands.

        Supported action_types:
            - "joint_position":  values shape (n_dof,)
            - "cartesian_pose":  values shape (7,)
            - "gripper":         values shape (1,) or scalar
            - "joint_velocity":  values shape (n_dof,)
            - "cartesian_twist": values shape (6,) [vx, vy, vz, wx, wy, wz]
        """
        atype = action.action_type
        vals = np.atleast_1d(action.values)

        if atype == "joint_position":
            return self.set_joint_positions(vals, **action.metadata)

        if atype == "cartesian_pose":
            return self.set_cartesian_pose(vals, **action.metadata)

        if atype == "gripper":
            pos = float(vals.flat[0])
            force = action.metadata.get("force")
            return self.control_gripper(pos, force=force)

        if atype == "joint_velocity":
            # Default: treat as position delta over dt
            dt = action.metadata.get("dt", 0.02)
            current = self.get_joint_positions()
            target = current + vals * dt
            return self.set_joint_positions(target, **action.metadata)

        if atype == "cartesian_twist":
            # Default: integrate twist over dt
            dt = action.metadata.get("dt", 0.02)
            from scipy.spatial.transform import Rotation as R
            current_pose = self.get_end_effector_pose()
            pos = np.array(current_pose[:3])
            quat = np.array(current_pose[3:7])
            # Linear
            vel = vals[:3]
            new_pos = pos + vel * dt
            # Angular
            omega = vals[3:6]
            delta_rot = R.from_rotvec(omega * dt)
            new_rot = delta_rot * R.from_quat(quat)
            new_quat = new_rot.as_quat()
            new_pose = np.concatenate([new_pos, new_quat])
            return self.set_cartesian_pose(new_pose, **action.metadata)

        raise ValueError(f"Unsupported action_type '{atype}' for manipulator {self.robot_id}")

    def observe(self) -> Observation:
        """Build a manipulator-specific observation."""
        obs = Observation()
        obs.proprioception = {
            "joint_positions": self.get_joint_positions(),
            "joint_velocities": self.get_joint_velocities(),
            "joint_torques": self.get_joint_torques(),
            "end_effector_pose": self.get_end_effector_pose(),
            "gripper_width": self.get_gripper_width(),
        }
        return obs

    # ------------------------------------------------------------------
    # Default action_space / observation_space
    # ------------------------------------------------------------------
    @property
    def action_space(self):
        n = self.dof
        return {
            "joint_position": Space(
                name="joint_position",
                shape=(n,),
                dtype="float32",
                description=f"Target joint positions for {n}-DOF arm",
            ),
            "cartesian_pose": Space(
                name="cartesian_pose",
                shape=(7,),
                dtype="float32",
                description="Target end-effector pose [x,y,z,qx,qy,qz,qw]",
            ),
            "gripper": Space(
                name="gripper",
                shape=(1,),
                dtype="float32",
                description="Gripper aperture (0=closed)",
            ),
        }

    @property
    def observation_space(self):
        n = self.dof
        return {
            "joint_positions": Space(
                name="joint_positions", shape=(n,), dtype="float32"
            ),
            "joint_velocities": Space(
                name="joint_velocities", shape=(n,), dtype="float32"
            ),
            "joint_torques": Space(
                name="joint_torques", shape=(n,), dtype="float32"
            ),
            "end_effector_pose": Space(
                name="end_effector_pose", shape=(7,), dtype="float32"
            ),
            "gripper_width": Space(
                name="gripper_width", shape=(1,), dtype="float32"
            ),
        }
