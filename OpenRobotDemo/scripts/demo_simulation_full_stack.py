"""
Full-stack simulation demo for OpenRobotDemo.

Integrates MuJoCo physics with OpenRobotDemo skills to perform a real
pick-and-place task in simulation using a ReAct planning loop.
"""

import os
import sys
import time
import threading
import yaml
from pathlib import Path
import numpy as np

from dotenv import load_dotenv

_project_root = os.path.join(os.path.dirname(__file__), "..")
# Load .env from OpenRobotDemo root
load_dotenv(Path(_project_root) / ".env")

sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(os.path.dirname(_project_root), "openrobot_core"))

import mujoco
import mujoco.viewer

from openrobot_demo.agent.planner import LLMPlanner
from openrobot_demo.agent.skill_router import SkillRouter
from openrobot_demo.skills.camera_capture import CameraCapture
from openrobot_demo.skills.arm_state_reader import ArmStateReader
from openrobot_demo.skills.vision_3d_estimator import Vision3DEstimator
from openrobot_demo.skills.grasp_predictor import GraspPointPredictor
from openrobot_demo.skills.arm_executor import ArmMotionExecutor
from openrobot_demo.hardware.mujoco_franka_adapter import FrankaMujocoAdapter, FrankaMujocoKinematics
from openrobot_demo.perception.camera_driver import CameraDriver
from openrobot_demo.persistence.db import EpisodeRecorder
from openrobot_demo.sensors import (
    VisionRGBSensor,
    VisionDepthSensor,
    PointCloudSensor,
    ProprioceptionSensor,
    TactileSensor,
)
from openrobot_demo.world_model import WorldModel, ObjectDesc


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_ground_truth_depth(model, data, cam_id, body_name="cube") -> float:
    """Compute GT depth (mm) of body center along camera Z axis."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    cam_pos = data.cam_xpos[cam_id]
    cam_xmat = data.cam_xmat[cam_id].reshape(3, 3)
    body_pos = data.xpos[body_id]
    delta = body_pos - cam_pos
    z_cam = float(cam_xmat[:, 2].dot(delta))
    return abs(z_cam) * 1000.0


def compute_hand_eye_from_mujoco(model, data, cam_name="wrist_cam", end_body="gripper_base"):
    """Compute hand-eye calibration (T_cam2end) from current MuJoCo state."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_body)

    cam_pos = data.cam_xpos[cam_id]
    cam_xmat = data.cam_xmat[cam_id].reshape(3, 3)

    end_pos = data.xpos[body_id]
    end_xmat = data.xmat[body_id].reshape(3, 3)

    R_cam2end = end_xmat.T @ cam_xmat
    t_cam2end = end_xmat.T @ (cam_pos - end_pos)

    return {
        "rotation_matrix": R_cam2end.tolist(),
        "translation_vector": t_cam2end.tolist(),
    }


def _capture_all_sensors(sensors):
    """Capture from all available sensors and return list of PerceptionData."""
    readings = []
    for sensor in sensors:
        if sensor.is_available():
            try:
                readings.append(sensor.capture())
            except Exception as exc:
                print(f"    ⚠️ Sensor {sensor.name}/{sensor.source_id} capture failed: {exc}")
    return readings


def run_simulation(recorder: EpisodeRecorder = None, instruction: str = "捡起黄色方块，而后扔向远处."):
    """
    Run the full-stack MuJoCo simulation demo.

    Args:
        recorder: Optional EpisodeRecorder to persist steps and state snapshots.
        instruction: Natural-language task instruction.
    """
    xml_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "sim", "mujoco", "franka_rgb_scene.xml"
    )
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    cam_name = "wrist_cam"
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)

    arm_adapter = FrankaMujocoAdapter(model, data, end_effector="gripper")
    arm_kinematics = FrankaMujocoKinematics(model, data, end_effector_offset=[0.0, 0.0, 0.0])
    arm_adapter.enable()

    cam_cfg = load_config(os.path.join(_project_root, "configs", "camera_config.yaml"))
    intrinsics = cam_cfg.get("intrinsics", {})

    # Create camera skill with MuJoCo driver wired to our model/data
    camera_skill = CameraCapture(camera_type="mujoco", width=640, height=480)
    camera_skill._driver = CameraDriver(
        camera_type="mujoco", width=640, height=480,
        mujoco_model=model, mujoco_data=data, mujoco_camera_name=cam_name
    )

    arm_reader = ArmStateReader(external_arm=arm_adapter, external_solver=arm_kinematics)
    vision_estimator = Vision3DEstimator()
    grasp_predictor = GraspPointPredictor()
    arm_executor = ArmMotionExecutor(external_arm=arm_adapter, external_solver=arm_kinematics)
    arm_executor.enable()

    router = SkillRouter()
    router.register(camera_skill)
    router.register(arm_reader)
    router.register(vision_estimator)
    router.register(grasp_predictor)
    router.register(arm_executor)

    # ------------------------------------------------------------------
    # Phase 2 MVP: Sensor registry + World Model
    # ------------------------------------------------------------------
    world_model = WorldModel()
    world_model.add_surface("table")
    world_model.spatial_memory.workspace_bounds = {
        "x": [0.1, 0.6],
        "y": [-0.4, 0.4],
        "z": [0.0, 0.5],
    }

    sensors = [
        VisionRGBSensor(source_id=cam_name, width=640, height=480, mujoco_model=model, mujoco_data=data),
        VisionDepthSensor(source_id=cam_name, width=640, height=480, mujoco_model=model, mujoco_data=data),
        PointCloudSensor(source_id=cam_name, width=640, height=480, fovy=45.0, max_depth=2.0, mujoco_model=model, mujoco_data=data),
        ProprioceptionSensor(source_id="franka_arm", arm_adapter=arm_adapter, kinematics_solver=arm_kinematics),
        TactileSensor(source_id="gripper", body_names=["gripper_base", "left_finger", "right_finger"], mujoco_model=model, mujoco_data=data),
    ]

    # Seed world model with initial proprioception
    for reading in _capture_all_sensors(sensors):
        world_model.ingest(reading)

    print(f"\n🤖 User instruction: {instruction}")

    planner = LLMPlanner()
    planner.start_task(instruction)

    # Compute ground-truth hand-eye calibration from MuJoCo
    hand_eye_calib = compute_hand_eye_from_mujoco(model, data, cam_name, "gripper_base")

    print("\n▶️ Executing ReAct loop in MuJoCo simulation...")
    motion_skills = {"arm_motion_executor"}
    images = []
    max_steps = 15

    # Try to launch the interactive viewer; fall back to headless if unavailable
    try:
        viewer = mujoco.viewer.launch_passive(model, data)
        print("   (Close the viewer window or press ESC to exit)")
        use_viewer = True
    except RuntimeError as e:
        print(f"   [Viewer unavailable: {e}]")
        print("   Running in headless mode...")
        viewer = None
        use_viewer = False

    finished = False
    try:
        for step_idx in range(max_steps):
            if use_viewer and not viewer.is_running():
                print("\n👋 Viewer closed by user.")
                break

            # ------------------------------------------------------------
            # Step start: capture all sensors into world model
            # ------------------------------------------------------------
            for reading in _capture_all_sensors(sensors):
                world_model.ingest(reading)

            # Refresh router context from world model for backward compatibility
            router._context["end_effector_pose"] = world_model.robot_state.end_effector_pose
            router._context["joint_positions"] = world_model.robot_state.joint_positions

            state_summary = world_model.build_state_summary()
            print(f"\n  Step {step_idx + 1}/{max_steps}")
            print("    ⏳ Waiting for Kimi to plan next step...")
            start_wait = time.time()

            # Run LLM planning in a background thread so the viewer stays responsive
            plan_result = {"action": None}
            def _call_planner():
                plan_result["action"] = planner.next_action(state_summary)
            plan_thread = threading.Thread(target=_call_planner, daemon=True)
            plan_thread.start()

            deadline = time.time() + 12
            while plan_thread.is_alive() and time.time() < deadline:
                if use_viewer and viewer.is_running():
                    viewer.sync()
                time.sleep(0.016)  # ~60 Hz render loop

            if plan_thread.is_alive():
                print("    ⏰ LLM planning timed out (>12s), using mock fallback.")
                action = planner._next_mock_action()
            else:
                action = plan_result["action"] if plan_result["action"] is not None else planner._next_mock_action()

            wait_time = time.time() - start_wait

            # Persist planner step
            if recorder:
                recorder.record_step(
                    step_idx=step_idx,
                    thought=action.get("thought", ""),
                    action={
                        "action": action.get("action"),
                        "skill": action.get("skill"),
                        "args": action.get("args"),
                    },
                    state_summary=state_summary,
                    wait_time_s=round(wait_time, 3),
                )

            if action.get("action") == "finish":
                print(f"\n✅ Task finished: {action.get('thought', '')} (planning took {wait_time:.2f}s)")
                finished = True
                break

            skill_name = action.get("skill")
            raw_args = action.get("args", {})
            print(f"    💭 {action.get('thought', '')} (planning took {wait_time:.2f}s)")

            args = router._resolve_args(raw_args)
            if skill_name == "arm_motion_executor":
                args = router._resolve_motion_args(args)
                # Validate to prevent crashes from malformed LLM outputs
                cmd_type = args.get("command_type")
                if cmd_type not in {"joint", "cartesian", "gripper"}:
                    print(f"    ⚠️ Invalid command_type '{cmd_type}' from LLM, skipping step.")
                    if recorder:
                        recorder.record_step_result(step_idx, skill_name or "unknown", False, f"Invalid command_type '{cmd_type}'", {})
                    continue
                if cmd_type == "cartesian":
                    tv = args.get("target_values", [])
                    if not isinstance(tv, list) or len(tv) != 7:
                        print(f"    ⚠️ Invalid cartesian target {tv}, skipping step.")
                        if recorder:
                            recorder.record_step_result(step_idx, skill_name, False, f"Invalid cartesian target {tv}", {})
                        continue
                if cmd_type == "gripper":
                    tv = args.get("target_values", [])
                    if isinstance(tv, (int, float)):
                        args["target_values"] = [float(tv), 0.5]
                    elif not isinstance(tv, list) or len(tv) == 0:
                        print(f"    ⚠️ Invalid gripper target {tv}, skipping step.")
                        if recorder:
                            recorder.record_step_result(step_idx, skill_name, False, f"Invalid gripper target {tv}", {})
                        continue

            if skill_name == "vision_3d_estimator":
                # Overwrite LLM-provided values with actual context to avoid type mismatches
                args["rgb_frame"] = router._context.get("rgb_frame")
                args["depth_frame"] = router._context.get("depth_frame")
                args["end_effector_pose"] = router._context.get("end_effector_pose")
                gt_depth_mm = compute_ground_truth_depth(model, data, cam_id, "cube")
                args["ground_truth_depth_mm"] = gt_depth_mm
                args.setdefault("camera_intrinsics", intrinsics)
                args.setdefault("hand_eye_calib", hand_eye_calib)
            elif skill_name == "grasp_point_predictor":
                args.setdefault("object_type", "box")

            if skill_name in motion_skills:
                import openrobot_demo.skills.arm_executor as arm_exec_mod
                old_sleep = arm_exec_mod.time.sleep

                def make_sim_sleep(m, d, v):
                    def sim_sleep(seconds):
                        steps = max(1, int(round(seconds / m.opt.timestep)))
                        step_dt = seconds / steps
                        for _ in range(steps):
                            t0 = time.time()
                            mujoco.mj_step(m, d)
                            if v is not None:
                                v.sync()
                            elapsed = time.time() - t0
                            if elapsed < step_dt:
                                time.sleep(step_dt - elapsed)
                    return sim_sleep

                arm_exec_mod.time.sleep = make_sim_sleep(model, data, viewer)
                try:
                    result = arm_executor.execute(**args)
                finally:
                    arm_exec_mod.time.sleep = old_sleep
            else:
                skill = router._skills[skill_name]
                result = skill.execute(**args)

            renderer = mujoco.Renderer(model, height=240, width=320)
            renderer.update_scene(data, camera=cam_name)
            images.append(renderer.render())
            renderer.close()

            router._update_context(skill_name, result)

            # Refresh arm state in context after every motion so the LLM sees current pose
            if skill_name == "arm_motion_executor":
                current_joints = arm_adapter.get_pos()
                router._context["joint_positions"] = current_joints
                router._context["end_effector_pose"] = arm_kinematics.forward_quat(current_joints)
                # Guard against simulation instability (NaN in state)
                if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)):
                    print("    ⚠️ Simulation became unstable (NaN detected). Resetting to home.")
                    mujoco.mj_resetDataKeyframe(model, data, key_id)
                    mujoco.mj_forward(model, data)
                    router._context["joint_positions"] = arm_adapter.get_pos()
                    router._context["end_effector_pose"] = arm_kinematics.forward_quat(arm_adapter.get_pos())

            # Inject ground-truth object position to bypass vision inaccuracies in simulation
            if skill_name == "vision_3d_estimator":
                cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
                gt_cube_pos = data.xpos[cube_body_id].copy().tolist()
                router._context["object_pose_base"] = gt_cube_pos + [0.0, 0.0, 0.0, 1.0]
                # Update world model with detected object
                world_model.add_or_update_object(
                    ObjectDesc(
                        object_id="cube",
                        object_type="box",
                        position=gt_cube_pos,
                        color="yellow",
                        size="small",
                        relations={"on": "table"},
                    )
                )

            # Override grasp/pre_grasp rotation to current EE rotation for this model's IK compatibility
            if skill_name == "grasp_point_predictor":
                ee_pose = router._context.get("end_effector_pose", [0, 0, 0, 0, 0, 0, 1])
                ee_quat = ee_pose[3:7]
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
                end_xmat = data.xmat[body_id].reshape(3, 3)
                approach_dir = -end_xmat[:, 2]

                grasp_pose = router._context.get("grasp_pose", [0, 0, 0, 0, 0, 0, 1])
                grasp_pos = np.array(grasp_pose[:3])
                pre_grasp_pos = grasp_pos - approach_dir * 0.10

                router._context["grasp_pose"] = grasp_pos.tolist() + ee_quat
                router._context["pre_grasp_pose"] = pre_grasp_pos.tolist() + ee_quat
                router._context["approach_vector"] = approach_dir.tolist()

                # Update world model with grasp affordances
                obj = world_model.get_object("cube")
                if obj is not None:
                    obj.grasp_points = [grasp_pos.tolist()]
                    world_model.add_or_update_object(obj)

            if recorder:
                recorder.record_state_snapshot(step_idx, router._context)
                recorder.record_step_result(
                    step_idx=step_idx,
                    skill_name=skill_name or "unknown",
                    success=result.get("success", False),
                    message=result.get("message", ""),
                    result=result,
                )

            if not result.get("success", False):
                print(f"    ❌ Failed: {result.get('message')}")
                break
            else:
                print(f"    ✅ Success: {result.get('message')}")
        else:
            print(f"\n⚠️ Reached max steps ({max_steps}).")
    finally:
        if viewer is not None:
            viewer.close()

        # Cleanup sensors
        for sensor in sensors:
            if hasattr(sensor, "close"):
                try:
                    sensor.close()
                except Exception:
                    pass

    from PIL import Image
    out_dir = os.path.join(_project_root, "data", "episodes")
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(images):
        Image.fromarray(img).save(os.path.join(out_dir, f"sim_step_{i:02d}.png"))
    print(f"\n🖼️ Saved {len(images)} simulation frames to {out_dir}")

    final_pos = arm_adapter.get_pos()
    final_ee = arm_kinematics.forward_quat(final_pos)
    cube_pos = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")]
    print(f"\n📊 Final arm joints: {[round(p, 3) for p in final_pos]}")
    print(f"📊 Final EE pose:    {[round(p, 3) for p in final_ee[:3]]}")
    print(f"📊 Cube position:    {[round(p, 3) for p in cube_pos]}")

    camera_skill.disconnect()
    arm_executor.disable()
    arm_reader.disable()

    if recorder:
        recorder.finish(status="completed" if finished else "failed")

    return {
        "success": finished,
        "images_count": len(images),
        "final_ee": final_ee,
        "cube_pos": cube_pos.tolist(),
    }


def run_simulation_queued(instruction: str = "捡起黄色方块，而后扔向远处.", block: bool = True):
    """
    Enqueue a simulation task via RobotQueue and optionally block until done.
    """
    from openrobot_demo.persistence.db import init_database
    from openrobot_demo.runtime.queue import RobotQueue

    db = init_database()
    queue = RobotQueue(db=db, max_retries=2, base_retry_delay=2.0)

    def _task(instr: str, recorder: EpisodeRecorder):
        return run_simulation(recorder=recorder, instruction=instr)

    episode_id = queue.enqueue(instruction, _task)
    print(f"\n📋 Task enqueued: {episode_id}")

    if block:
        while True:
            status = queue.get_status(episode_id)
            if status["status"] in ("completed", "failed"):
                print(f"\n🏁 Task {episode_id} finished with status: {status['status']}")
                return status
            time.sleep(0.5)
    return episode_id


if __name__ == "__main__":
    # Default: run directly for backward compatibility
    run_simulation()
