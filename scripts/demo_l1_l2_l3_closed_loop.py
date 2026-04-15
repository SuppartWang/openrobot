"""
Integration demo: L1 Monitor + L2 PerceptionBus + L3 MujocoExecutor closed loop.
Reads RGB and proprioception, executes a simple sinusoidal joint trajectory.
"""

import os
import sys
import time
import numpy as np
from PIL import Image

# Ensure project packages are importable
_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "openrobot_core"))

import mujoco
from openrobot_core.openrobot_monitor.monitor import RobotMonitor
from openrobot_perception.io_bus.interface import RGBCamera, ProprioceptionSensor
from openrobot_perception.io_bus.bus import PerceptionBus
from openrobot_control.execution.mujoco_executor import MujocoExecutor
from openrobot_perception.io_bus.mujoco_source import MujocoSensorSource


def main(duration_sec: float = 2.0):
    xml_path = os.path.join(os.path.dirname(__file__), "..", "sim", "mujoco", "franka_rgb_scene.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Reset to home keyframe
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)

    # L1: Monitor
    monitor = RobotMonitor()
    monitor.register_node("perception_bus")
    monitor.register_node("motion_executor")

    # L2: Perception Bus
    sensor_source = MujocoSensorSource(model, data, camera_name="wrist_cam")
    bus = PerceptionBus()
    bus.attach(RGBCamera(sensor_source))
    bus.attach(ProprioceptionSensor(sensor_source))

    # L3: Executor
    executor = MujocoExecutor(model, data)

    # Run closed-loop
    step = 0
    render_every = 50
    images = []

    print("[openrobot] Starting L1+L2+L3 closed-loop demo...")
    while data.time < duration_sec:
        # L2: Poll sensors
        perception = bus.poll()
        monitor.heartbeat("perception_bus", metadata={"timestamp": perception.timestamp})

        # Simple trajectory: sinusoidal around home position
        # We read the home qpos for the robot arm (skip freejoint of cube)
        # For this simplified model, freejoint starts at qpos[0], arm joints start at qpos[7].
        # We'll command the first model.nu actuators.
        home_ctrl = np.array([0.0, 1.57, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
        target = home_ctrl + np.sin(data.time * 2.0 + np.arange(model.nu) * 0.5) * 0.15
        for i in range(model.nu):
            lo, hi = model.actuator_ctrlrange[i]
            target[i] = np.clip(target[i], lo, hi)

        from openrobot_msgs import ActionCmd
        cmd = ActionCmd(type="joint_position", values=target)
        executor.apply(cmd)
        monitor.heartbeat("motion_executor", metadata={"ctrl_norm": float(np.linalg.norm(target))})

        # Step simulation
        executor.step()
        step += 1

        if step % render_every == 0:
            sensor_source._renderer.update_scene(data, camera="wrist_cam")
            img = sensor_source._renderer.render()
            images.append(img)
            print(f"  t={data.time:.3f}s | ee_z={perception.proprioception.ee_pose[2]:.3f}")

    print(f"[openrobot] Demo finished: {step} steps")
    print(f"[openrobot] System health: {monitor.check_health()}")

    # Save the last frame
    if images:
        out_path = os.path.join(os.path.dirname(__file__), "..", "sim", "mujoco", "demo_closed_loop_last_frame.png")
        Image.fromarray(images[-1]).save(out_path)
        print(f"[openrobot] Saved last frame to {out_path}")

    bus.disconnect_all()


if __name__ == "__main__":
    main(duration_sec=2.0)
