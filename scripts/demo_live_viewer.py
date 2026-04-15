"""
Real-time MuJoCo viewer demo for openrobot MVP.
Opens an interactive GLFW window showing the Franka arm and cube.
The arm will follow a slow sinusoidal trajectory while the viewer runs.
Press ESC or close the window to exit early.
"""

import os
import sys
import time
import numpy as np

_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "openrobot_core"))

import mujoco
import mujoco.viewer


def main(duration_sec: float = 5.0):
    xml_path = os.path.join(_project_root, "sim", "mujoco", "franka_rgb_scene.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)

    home_ctrl = data.ctrl.copy()
    print(f"[openrobot] Launching live viewer for {duration_sec}s...")
    print("[openrobot] Close the viewer window or press ESC to exit early.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running() and (time.time() - start) < duration_sec:
            step_start = time.time()

            # Simple sinusoidal trajectory
            t = data.time
            target = home_ctrl + np.sin(t * 1.5 + np.arange(model.nu) * 0.5) * 0.15
            for i in range(model.nu):
                lo, hi = model.actuator_ctrlrange[i]
                target[i] = np.clip(target[i], lo, hi)
            data.ctrl[:] = target

            mujoco.mj_step(model, data)
            viewer.sync()

            # Cap at roughly 60 Hz real-time
            elapsed = time.time() - step_start
            sleep_for = max(0.0, model.opt.timestep - elapsed)
            if sleep_for > 0:
                time.sleep(sleep_for)

    print("[openrobot] Viewer closed.")


if __name__ == "__main__":
    try:
        main(duration_sec=5.0)
    except Exception as e:
        print(f"[openrobot] Viewer failed to open: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
