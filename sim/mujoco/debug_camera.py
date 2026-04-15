import mujoco
import os

xml_path = os.path.join(os.path.dirname(__file__), "franka_rgb_scene.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if key_id >= 0:
    mujoco.mj_resetDataKeyframe(model, data, key_id)
else:
    mujoco.mj_resetData(model, data)

mujoco.mj_forward(model, data)

cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
print(f"Camera id: {cam_id}")
print(f"Camera pos (world): {data.cam_xpos[cam_id]}")
print(f"Camera xmat (world):\n{data.cam_xmat[cam_id].reshape(3,3)}")

# Also print end-effector (gripper_base) position and orientation
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
print(f"\nGripper base pos: {data.xpos[body_id]}")
print(f"Gripper base xmat:\n{data.xmat[body_id].reshape(3,3)}")

# Print all joint positions
print("\nJoint positions (qpos):")
for i in range(model.nq):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"  {name}: {data.qpos[i]:.4f}")
