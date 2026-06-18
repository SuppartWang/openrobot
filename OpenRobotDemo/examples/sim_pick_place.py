"""
Minimal MuJoCo pick-and-place example (no LLM/VLM required).

This is the recommended entry point for new users who want to verify the
simulation stack before connecting real hardware or API keys.

Usage:
    cd OpenRobotDemo
    python examples/sim_pick_place.py

The robot will:
    1. Reset to the home pose
    2. Move above the yellow cube
    3. Descend to grasp height
    4. Close the gripper
    5. Lift the cube
    6. Move to a place position
    7. Open the gripper
    8. Retreat
"""

import os
import sys
import time
from pathlib import Path

_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(os.path.dirname(_project_root), "openrobot_core"))

import mujoco
import numpy as np
from PIL import Image

from openrobot_demo.hardware.mujoco_franka_adapter import FrankaMujocoKinematics


XML_PATH = os.path.abspath(
    os.path.join(_project_root, "..", "sim", "mujoco", "franka_rgb_scene.xml")
)

# Cube ground-truth position (matches the XML keyframe)
CUBE_POS = np.array([0.55, 0.15, 0.45])
CUBE_SIZE = 0.03

# Motion parameters
PRE_GRASP_OFFSET = np.array([0.0, 0.0, 0.10])
GRASP_OFFSET = np.array([0.0, 0.0, CUBE_SIZE])
PLACE_POS = np.array([0.45, -0.10, 0.45])
LIFT_HEIGHT = 0.06

# Default downward-facing orientation (quaternion [x, y, z, w])
# Rotates +90 deg around the Y axis so the gripper points down.
DOWN_QUAT = np.array([0.0, 0.70710678, 0.0, 0.70710678])


def load_model():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    return model, data


def get_arm_joints(model, data):
    joint_names = [f"joint{i}" for i in range(1, 8)]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    qpos_adr = [model.jnt_qposadr[jid] for jid in joint_ids]
    return qpos_adr


def set_arm_joints(model, data, qpos_adr, joints):
    for adr, val in zip(qpos_adr, joints):
        data.qpos[adr] = val
    mujoco.mj_forward(model, data)


def set_fingers(model, data, left, right):
    """Set finger joint positions directly (0=closed, 0.04=open)."""
    j1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
    j2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
    data.qpos[model.jnt_qposadr[j1]] = left
    data.qpos[model.jnt_qposadr[j2]] = right
    mujoco.mj_forward(model, data)


def capture_frame(model, data, camera="top"):
    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data, camera=camera)
    frame = renderer.render()
    renderer.close()
    return frame


def save_frames(frames, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(os.path.join(out_dir, f"pick_place_{i:02d}.png"))
    print(f"🖼️ Saved {len(frames)} frames to {out_dir}")


def interpolate_joints(model, data, qpos_adr, start, end, steps, frames, camera="top"):
    """Joint-space interpolation with frame capture at start, middle and end."""
    start = np.array(start)
    end = np.array(end)
    for i in range(steps + 1):
        alpha = i / steps
        joints = start * (1 - alpha) + end * alpha
        set_arm_joints(model, data, qpos_adr, joints.tolist())
        if i in {0, steps // 2, steps}:
            frames.append(capture_frame(model, data, camera=camera))
        time.sleep(0.005)


def move_to_pose(model, data, qpos_adr, solver, target_pose, frames, camera="top", speed=1.0):
    """Move the arm to a target Cartesian pose using IK."""
    current_joints = [data.qpos[adr] for adr in qpos_adr]
    joints = None
    for attempt in range(5):
        joints = solver.inverse_quat(target_pose, current_joints)
        if joints is not None:
            break
    if joints is None:
        print(f"    ⚠️ IK failed for pose {target_pose[:3]}")
        return False

    steps = max(20, int(np.linalg.norm(np.array(joints) - np.array(current_joints)) * 100 / max(0.1, speed)))
    interpolate_joints(model, data, qpos_adr, current_joints, joints, steps, frames, camera=camera)
    return True


def run(out_dir: str = None) -> bool:
    print("=" * 60)
    print(" OpenRobotDemo — Minimal MuJoCo Pick & Place")
    print("=" * 60 + "\n")
    if out_dir is None:
        out_dir = os.path.join(_project_root, "data", "episodes", "sim_pick_place")

    model, data = load_model()
    qpos_adr = get_arm_joints(model, data)
    solver = FrankaMujocoKinematics(model, data, end_effector_offset=[0.0, 0.0, 0.0])

    frames = []
    frames.append(capture_frame(model, data))

    # 1. Pre-grasp
    pre_grasp = np.concatenate([CUBE_POS + PRE_GRASP_OFFSET, DOWN_QUAT])
    print("[1/6] Moving to pre-grasp...")
    if not move_to_pose(model, data, qpos_adr, solver, pre_grasp.tolist(), frames):
        return False

    # 2. Grasp
    grasp = np.concatenate([CUBE_POS + GRASP_OFFSET, DOWN_QUAT])
    print("[2/6] Moving to grasp...")
    if not move_to_pose(model, data, qpos_adr, solver, grasp.tolist(), frames):
        return False

    # 3. Close gripper
    print("[3/6] Closing gripper...")
    set_fingers(model, data, 0.0, 0.0)
    frames.append(capture_frame(model, data))

    # 4. Lift
    lift = np.concatenate([CUBE_POS + GRASP_OFFSET + np.array([0, 0, LIFT_HEIGHT]), DOWN_QUAT])
    print("[4/6] Lifting...")
    if not move_to_pose(model, data, qpos_adr, solver, lift.tolist(), frames):
        return False

    # 5. Move to place
    place_above = np.concatenate([PLACE_POS + GRASP_OFFSET + np.array([0, 0, LIFT_HEIGHT]), DOWN_QUAT])
    print("[5/6] Moving to place...")
    if not move_to_pose(model, data, qpos_adr, solver, place_above.tolist(), frames):
        return False

    place_down = np.concatenate([PLACE_POS + GRASP_OFFSET, DOWN_QUAT])
    if not move_to_pose(model, data, qpos_adr, solver, place_down.tolist(), frames):
        return False

    # 6. Open gripper
    print("[6/6] Opening gripper...")
    set_fingers(model, data, 0.04, 0.04)
    frames.append(capture_frame(model, data))

    # Drop cube visually at place position
    # The cube uses the first (unnamed) free joint; its qpos starts at 0.
    new_pos = PLACE_POS + np.array([0, 0, CUBE_SIZE])
    data.qpos[0] = new_pos[0]
    data.qpos[1] = new_pos[1]
    data.qpos[2] = new_pos[2]
    data.qpos[3] = 1.0
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = 0.0
    mujoco.mj_forward(model, data)
    frames.append(capture_frame(model, data))

    # Retreat
    retreat = np.concatenate([PLACE_POS + GRASP_OFFSET + np.array([0, 0, LIFT_HEIGHT]), DOWN_QUAT])
    move_to_pose(model, data, qpos_adr, solver, retreat.tolist(), frames)

    save_frames(frames, out_dir)

    final_ee = solver.forward_quat([data.qpos[adr] for adr in qpos_adr])
    print(f"\n📊 Final EE position: {final_ee[:3]}")
    print(f"📊 Target place position: {PLACE_POS}")
    print("\n✅ Pick-and-place demo completed.")
    return True


if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
