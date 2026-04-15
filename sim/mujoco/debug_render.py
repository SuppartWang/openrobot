import os
import mujoco
import numpy as np
from PIL import Image

xml_path = os.path.join(os.path.dirname(__file__), "franka_rgb_scene.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, key_id)
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=240, width=320)

camera_name = "wrist_cam"
renderer.update_scene(data, camera=camera_name)
img = renderer.render()
out_path = os.path.join(os.path.dirname(__file__), "debug_render_home.png")
Image.fromarray(img).save(out_path)
print(f"Saved render from keyframe to {out_path}")

# Also render from a world-aligned camera above the table for reference
renderer.update_scene(data, camera="top")
img_top = renderer.render()
out_path_top = os.path.join(os.path.dirname(__file__), "debug_render_top.png")
Image.fromarray(img_top).save(out_path_top)
print(f"Saved top-down render to {out_path_top}")
