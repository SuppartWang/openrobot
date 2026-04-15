"""
MuJoCo simulation demo for openrobot MVP.
Loads a simplified Franka-like arm with an RGB wrist camera and runs a random-control loop.
"""

import os
import time
import numpy as np

# Optional: use matplotlib to save the rendered frame
try:
    from PIL import Image
except ImportError:
    Image = None


def run_demo(duration_sec: float = 3.0, render_every: int = 50):
    import mujoco
    import mujoco.viewer

    xml_path = os.path.join(os.path.dirname(__file__), "franka_rgb_scene.xml")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"MJCF not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Reset to 'home' keyframe if available
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print(f"[openrobot] Reset to keyframe 'home' (id={key_id})")
    else:
        mujoco.mj_resetData(model, data)

    # Camera configuration
    renderer = mujoco.Renderer(model, height=240, width=320)
    camera_name = "wrist_cam"
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id == -1:
        raise RuntimeError(f"Camera '{camera_name}' not found in model.")

    # Number of actuators
    nu = model.nu

    print(f"[openrobot] Simulation started: dt={model.opt.timestep}s, actuators={nu}")
    print(f"[openrobot] Camera '{camera_name}' id={cam_id}")

    step = 0
    start = time.time()
    images = []

    while data.time < duration_sec:
        # Random control: small sinusoidal signals around neutral
        target = np.sin(data.time * 2.0 + np.arange(nu) * 0.5) * 0.3
        # Clip to ctrlrange
        for i in range(nu):
            lo, hi = model.actuator_ctrlrange[i]
            target[i] = np.clip(target[i], lo, hi)
        data.ctrl[:] = target

        mujoco.mj_step(model, data)

        if step % render_every == 0:
            renderer.update_scene(data, camera=camera_name)
            rgb = renderer.render()
            images.append(rgb)
            print(f"  t={data.time:.3f}s | rendered frame {len(images)}")

        step += 1

    elapsed = time.time() - start
    print(f"[openrobot] Simulation finished: {step} steps in {elapsed:.2f}s ({step/elapsed:.1f} Hz)")

    # Save the last frame
    if images and Image is not None:
        out_path = os.path.join(os.path.dirname(__file__), "demo_output_rgb.png")
        img = Image.fromarray(images[-1])
        img.save(out_path)
        print(f"[openrobot] Saved last RGB frame to {out_path}")
    elif images:
        out_path = os.path.join(os.path.dirname(__file__), "demo_output_rgb.npy")
        np.save(out_path, images[-1])
        print(f"[openrobot] Saved last RGB frame to {out_path}")

    return images


if __name__ == "__main__":
    run_demo(duration_sec=2.0, render_every=25)
