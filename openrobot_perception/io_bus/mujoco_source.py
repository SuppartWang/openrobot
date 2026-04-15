"""Sensor source adapter for MuJoCo simulation."""

import numpy as np
import mujoco


class MujocoSensorSource:
    """Wraps a MuJoCo model/data pair to provide sensor readings for PerceptionBus."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, camera_name: str = "wrist_cam"):
        self.model = model
        self.data = data
        self.camera_name = camera_name
        self._renderer = mujoco.Renderer(model, height=240, width=320)

    def read_rgb(self) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=self.camera_name)
        return self._renderer.render()

    def read_proprioception(self) -> dict:
        # Map actuated joints (first model.nu correspond to arm joints, excluding freejoint)
        # For simplicity, we read all joint positions/velocities from qpos/qvel.
        # We'll assume the robot arm joints are contiguous after any freejoints.
        joint_positions = self.data.qpos.copy()
        joint_velocities = self.data.qvel.copy()

        # Compute end-effector pose (gripper_base body)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
        xpos = self.data.xpos[body_id].copy()
        xmat = self.data.xmat[body_id].reshape(3, 3).copy()
        # Convert rotation matrix to quaternion (w, x, y, z)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, xmat.flatten())
        ee_pose = np.concatenate([xpos, quat])

        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "ee_pose": ee_pose,
            "timestamp": self.data.time,
        }
