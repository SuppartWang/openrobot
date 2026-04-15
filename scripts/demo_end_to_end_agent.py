"""
End-to-end demo: L4 LLM Agent -> L3 Motion Planning -> L2 Perception -> L1 Monitor -> MuJoCo.
"""

import os
import sys
import numpy as np
from PIL import Image

_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "openrobot_core"))

import mujoco
from openrobot_core.openrobot_monitor.monitor import RobotMonitor
from openrobot_perception.io_bus.interface import RGBCamera, ProprioceptionSensor
from openrobot_perception.io_bus.bus import PerceptionBus
from openrobot_perception.io_bus.mujoco_source import MujocoSensorSource
from openrobot_control.execution.mujoco_executor import MujocoExecutor
from openrobot_control.motion_planning.interpolator import JointSpaceInterpolator
from openrobot_cognition.agent.llm_agent import LLMAgent
from openrobot_cognition.spatial.scene_graph import SceneGraph
from openrobot_msgs import ActionCmd


def mock_plan(instruction: str):
    """Fallback deterministic plan when LLM is unavailable."""
    return [
        {"action": "move_to", "args": [0.45, 0.1, 0.55]},
        {"action": "grasp", "args": ["cube"]},
        {"action": "move_to", "args": [0.45, -0.1, 0.55]},
        {"action": "release", "args": []},
    ]


def main():
    xml_path = os.path.join(_project_root, "sim", "mujoco", "franka_rgb_scene.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)

    # L1 Monitor
    monitor = RobotMonitor()
    monitor.register_node("agent")
    monitor.register_node("planner")
    monitor.register_node("executor")

    # L2 Perception
    sensor_source = MujocoSensorSource(model, data, camera_name="wrist_cam")
    bus = PerceptionBus()
    bus.attach(RGBCamera(sensor_source))
    bus.attach(ProprioceptionSensor(sensor_source))

    # L3 Executor & Planner
    executor = MujocoExecutor(model, data)
    interpolator = JointSpaceInterpolator(num_steps=30)

    # L4 Agent & Spatial Cognition
    agent = None
    try:
        if os.getenv("OPENAI_API_KEY"):
            agent = LLMAgent()
            print("[openrobot] LLM Agent initialized.")
        else:
            print("[openrobot] OPENAI_API_KEY not set, using mock agent.")
    except Exception as e:
        print(f"[openrobot] LLM Agent failed to init: {e}, using mock agent.")

    scene = SceneGraph()
    scene.register_object("cube", position=np.array([0.5, 0.1, 0.45]), obj_type="cube", color="yellow")
    scene.register_object("table", position=np.array([0.4, 0.0, 0.4]), obj_type="table", color="brown")

    instruction = "Pick up the yellow cube and place it on the left side of the table."
    print(f"[openrobot] Instruction: {instruction}")

    # Planning
    if agent:
        context = scene.to_context_string()
        try:
            plan = agent.plan(instruction, scene_context=context)
        except Exception as e:
            print(f"[openrobot] LLM planning failed: {e}, falling back to mock plan.")
            plan = mock_plan(instruction)
    else:
        plan = mock_plan(instruction)

    print(f"[openrobot] Plan: {plan}")
    monitor.heartbeat("agent", metadata={"plan_steps": len(plan)})

    # Execute plan step by step
    images = []
    home_q = data.ctrl.copy()
    for step_idx, step in enumerate(plan):
        action = step.get("action", "unknown")
        args = step.get("args", [])
        print(f"[openrobot] Executing step {step_idx + 1}/{len(plan)}: {action}({args})")

        # Simple action interpreter
        if action == "move_to":
            target_pos = np.array(args[:3])
            # For MVP, we do a naive heuristic: modulate joint2 and joint4 to move EE near target.
            # A proper IK is future work.
            current = data.qpos[7 : 7 + model.nu].copy()
            # Heuristic: shift joint1 (yaw) toward target y, joint2 (shoulder) toward target x/z
            heuristic = current.copy()
            heuristic[0] = np.arctan2(target_pos[1], target_pos[0])  # yaw
            heuristic[1] = 1.0 + (0.55 - target_pos[2]) * 1.5  # shoulder elevation
            heuristic[3] = 0.3 + (target_pos[0] - 0.3) * 0.5  # elbow
            # Clip
            for i in range(model.nu):
                lo, hi = model.actuator_ctrlrange[i]
                heuristic[i] = np.clip(heuristic[i], lo, hi)
            traj = interpolator.plan(current, heuristic)
            for cmd in traj:
                executor.apply(cmd)
                executor.step()
        elif action == "grasp":
            # Close gripper
            cmd = ActionCmd(type="gripper", values=np.array([0.0, 0.0]))
            executor.apply(cmd)
            for _ in range(40):
                executor.step()
        elif action == "release":
            # Open gripper
            cmd = ActionCmd(type="gripper", values=np.array([0.04, 0.04]))
            executor.apply(cmd)
            for _ in range(40):
                executor.step()
        else:
            print(f"  [skip] Unsupported action: {action}")

        # Capture observation
        perception = bus.poll()
        monitor.heartbeat("executor", metadata={"step": step_idx, "action": action})
        sensor_source._renderer.update_scene(data, camera="wrist_cam")
        images.append(sensor_source._renderer.render())

    # Save final frame
    if images:
        out_path = os.path.join(_project_root, "sim", "mujoco", "demo_agent_last_frame.png")
        Image.fromarray(images[-1]).save(out_path)
        print(f"[openrobot] Saved final agent frame to {out_path}")

    print(f"[openrobot] System health: {monitor.check_health()}")
    bus.disconnect_all()


if __name__ == "__main__":
    main()
