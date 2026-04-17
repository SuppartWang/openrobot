"""
Dual-arm fabric manipulation demo for OpenRobotDemo.

Task: A 7-DOF dual-arm robot with 2-finger grippers picks up a cylindrical fabric tube,
inserts it over an aluminum support plate, waits for defect inspection,
then withdraws the fabric.

Hardware:
    - Left arm:  YHRG S1 on /dev/ttyUSB0
    - Right arm: YHRG S1 on /dev/ttyUSB1
    - Camera:    RealSense D435i (optional, mock fallback on macOS)
    - End-effector: parallel 2-finger gripper

Usage:
    # Simulation / mock mode (macOS or development)
    python scripts/demo_fabric_dual_arm.py --mode mock

    # Real hardware mode (Linux + S1_SDK + RealSense)
    python scripts/demo_fabric_dual_arm.py --mode real \
        --left-dev /dev/ttyUSB0 --right-dev /dev/ttyUSB1
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_project_root = os.path.join(os.path.dirname(__file__), "..")
load_dotenv(Path(_project_root) / ".env")
sys.path.insert(0, _project_root)

from openrobot_demo.agent.planner import LLMPlanner
from openrobot_demo.agent.skill_router import SkillRouter
from openrobot_demo.dual_arm.controller import DualArmController, ArmSide
from openrobot_demo.dual_arm.fabric_skills import FabricManipulationSkill
from openrobot_demo.experience.library import ExperienceLibrary
from openrobot_demo.experience.retriever import ExperienceRetriever
from openrobot_demo.experience.seed import seed_fabric_experiences
from openrobot_demo.persistence.db import EpisodeRecorder, init_database
from openrobot_demo.sensors import (
    ProprioceptionSensor,
    RealSenseRGBSensor,
    RealSenseDepthSensor,
    TactileSensor,
)
from openrobot_demo.skills import (
    ArmMotionExecutor,
    ArmStateReader,
    CameraCapture,
    GraspPointPredictor,
    Vision3DEstimator,
    VLAPolicyExecutor,
)
from openrobot_demo.world_model import WorldModel, ObjectDesc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def setup_experiences() -> ExperienceLibrary:
    """Initialize experience library and seed with fabric manipulation knowledge."""
    lib = ExperienceLibrary()
    existing = lib.list_all(limit=1)
    if not existing:
        seed_fabric_experiences(lib)
    else:
        print(f"[Setup] Experience library already has {len(existing)} records, skipping seed.")
    return lib


def setup_dual_arm(mode: str, left_dev: str, right_dev: str) -> DualArmController:
    """Initialize dual-arm controller."""
    ctrl = DualArmController(
        left_dev=left_dev,
        right_dev=right_dev,
        mode=mode,
        end_effector="gripper",
    )
    ctrl.enable()
    return ctrl


def setup_sensors(dual_arm: DualArmController) -> list:
    """Initialize sensor channels."""
    sensors = [
        RealSenseRGBSensor(source_id="rs_d435i_rgb"),
        RealSenseDepthSensor(source_id="rs_d435i_depth"),
        ProprioceptionSensor(source_id="left_arm", arm_adapter=dual_arm.left_arm, kinematics_solver=dual_arm.left_kin),
        ProprioceptionSensor(source_id="right_arm", arm_adapter=dual_arm.right_arm, kinematics_solver=dual_arm.right_kin),
        TactileSensor(source_id="left_gripper", body_names=["gripper_base", "left_finger"], mujoco_model=None, mujoco_data=None),
        TactileSensor(source_id="right_gripper", body_names=["gripper_base", "right_finger"], mujoco_model=None, mujoco_data=None),
    ]
    return sensors


def run_fabric_demo(
    mode: str = "mock",
    left_dev: str = "/dev/ttyUSB0",
    right_dev: str = "/dev/ttyUSB1",
    instruction: str = "将筒状布料提起，套在铝合金支撑板上，等待检测后再取下来",
    recorder: EpisodeRecorder = None,
):
    """Run the full fabric manipulation demo."""

    print("\n" + "=" * 60)
    print("  OpenRobotDemo — Dual-Arm Fabric Manipulation")
    print(f"  Mode: {mode} | Arms: {left_dev} + {right_dev}")
    print("=" * 60 + "\n")

    # 1. Experience library
    print("[1/6] Loading experience library...")
    exp_lib = setup_experiences()
    exp_list = exp_lib.list_all()
    print(f"      {len(exp_list)} experiences loaded.")
    for e in exp_list:
        print(f"      - [{e.action_type}] {e.task_intent[:30]}...")

    # 2. Dual-arm controller
    print("\n[2/6] Initializing dual-arm controller...")
    dual_arm = setup_dual_arm(mode, left_dev, right_dev)
    print(f"      Left  arm pos:  {dual_arm.get_pos(ArmSide.LEFT)[:6]}")
    print(f"      Right arm pos:  {dual_arm.get_pos(ArmSide.RIGHT)[:6]}")

    # 3. World model & sensors
    print("\n[3/6] Initializing world model & sensors...")
    world = WorldModel()
    world.add_surface("workbench")
    # Pre-register expected objects
    world.add_or_update_object(
        ObjectDesc(
            object_id="fabric_tube",
            object_type="筒状布料",
            position=[0.30, 0.0, 0.02],
            size="直径8cm",
            color="white",
            material="textile",
            relations={"on": "workbench"},
        )
    )
    world.add_or_update_object(
        ObjectDesc(
            object_id="support_plate",
            object_type="铝合金支撑板",
            position=[0.30, 0.0, 0.0],
            size="高度5cm",
            color="silver",
            material="aluminum",
            relations={"on": "workbench", "under": "fabric_tube"},
        )
    )
    sensors = setup_sensors(dual_arm)
    for s in sensors:
        print(f"      Sensor {s.name}/{s.source_id}: available={s.is_available()}")

    # 4. Skills
    print("\n[4/6] Registering skills...")
    router = SkillRouter()
    fabric_skill = FabricManipulationSkill(
        dual_arm=dual_arm,
        experience_library=exp_lib,
        world_model=world,
    )
    # Register legacy skills for camera/vision fallback
    camera_skill = CameraCapture(camera_type="mock")
    arm_reader = ArmStateReader()
    vision_estimator = Vision3DEstimator()
    grasp_predictor = GraspPointPredictor()
    arm_executor = ArmMotionExecutor()
    vla_executor = VLAPolicyExecutor(external_arm=dual_arm.left_arm)

    router.register(camera_skill)
    router.register(arm_reader)
    router.register(vision_estimator)
    router.register(grasp_predictor)
    router.register(arm_executor)
    router.register(fabric_skill)
    router.register(vla_executor)

    # 5. Planner
    print("\n[5/6] Starting ReAct planner...")
    planner = LLMPlanner()
    planner.start_task(instruction)
    print(f"      Instruction: {instruction}")

    # 6. ReAct loop
    print("\n[6/6] Executing ReAct loop...\n")
    max_steps = 20
    step_idx = 0
    finished = False

    try:
        for step_idx in range(max_steps):
            # Update world model from sensors
            for sensor in sensors:
                if sensor.is_available():
                    try:
                        reading = sensor.capture()
                        world.ingest(reading)
                    except Exception as exc:
                        logger.debug("Sensor %s capture failed: %s", sensor.source_id, exc)

            state_summary = world.build_state_summary()
            print(f"\n  Step {step_idx + 1}/{max_steps}")
            print(f"    📊 World state:\n{state_summary[:300]}...")
            print("    ⏳ Planner thinking...")

            action = planner.next_action(state_summary)
            wait_time = 0.1  # mock timing

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
                print(f"\n    ✅ Task finished: {action.get('thought', '')}")
                finished = True
                break

            skill_name = action.get("skill")
            raw_args = action.get("args", {})
            print(f"    💭 {action.get('thought', '')}")
            print(f"    🔧 Skill: {skill_name}")

            # Resolve args from context
            args = router._resolve_args(raw_args)
            if skill_name == "fabric_manipulation":
                # Fabric skill handles its own experience loading internally
                result = fabric_skill.execute(**args)
            elif skill_name in router._skills:
                skill = router._skills[skill_name]
                result = skill.execute(**args)
            else:
                print(f"    ⚠️ Unknown skill: {skill_name}")
                result = {"success": False, "message": f"Unknown skill: {skill_name}"}

            if recorder:
                recorder.record_step_result(
                    step_idx=step_idx,
                    skill_name=skill_name or "unknown",
                    success=result.get("success", False),
                    message=result.get("message", ""),
                    result=result,
                )
                recorder.record_state_snapshot(step_idx, world.to_dict())

            if result.get("success", False):
                print(f"    ✅ {result.get('message', 'OK')}")
            else:
                print(f"    ❌ {result.get('message', 'Failed')}")
                # On failure, could trigger experience auto-refinement here
                break

            # Refresh context
            router._update_context(skill_name, result)

        else:
            print(f"\n⚠️ Reached max steps ({max_steps}).")

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user.")
    finally:
        print("\n[Cleanup] Disabling arms and releasing resources...")
        dual_arm.disable()
        for sensor in sensors:
            if hasattr(sensor, "close"):
                try:
                    sensor.close()
                except Exception:
                    pass
        exp_lib.close()
        if recorder:
            recorder.finish(status="completed" if finished else "failed")

    # Print final world model state
    print("\n📋 Final World Model:")
    for obj_id, obj in world.objects.items():
        print(f"   {obj_id}: pos={obj.position}, relations={obj.relations}")

    return {"success": finished, "steps": step_idx + 1}


def main():
    parser = argparse.ArgumentParser(description="Dual-Arm Fabric Manipulation Demo")
    parser.add_argument("--mode", choices=["mock", "real"], default="mock", help="Execution mode")
    parser.add_argument("--left-dev", default="/dev/ttyUSB0", help="Left arm serial device")
    parser.add_argument("--right-dev", default="/dev/ttyUSB1", help="Right arm serial device")
    parser.add_argument(
        "--instruction",
        default="将筒状布料提起，套在铝合金支撑板上，等待检测后再取下来",
        help="Task instruction",
    )
    parser.add_argument("--persist", action="store_true", help="Persist episode to SQLite")
    args = parser.parse_args()

    recorder = None
    if args.persist:
        db = init_database()
        eid = db.create_episode(args.instruction)
        recorder = EpisodeRecorder(db, eid)
        print(f"[Persist] Episode {eid} created in database.")

    result = run_fabric_demo(
        mode=args.mode,
        left_dev=args.left_dev,
        right_dev=args.right_dev,
        instruction=args.instruction,
        recorder=recorder,
    )

    print(f"\n🏁 Demo result: success={result['success']}, steps={result['steps']}")


if __name__ == "__main__":
    main()
