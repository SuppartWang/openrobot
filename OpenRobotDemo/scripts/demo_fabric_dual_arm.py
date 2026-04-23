"""
Dual-arm fabric manipulation demo for OpenRobotDemo.

Task: A 7-DOF dual-arm robot with 2-finger grippers picks up a cylindrical fabric tube,
inserts it over an aluminum support plate, waits for defect inspection,
then withdraws the fabric.

Hardware:
    - Left arm:  YHRG S1 on /dev/left_leader  -> ttyUSB0
    - Right arm: YHRG S1 on /dev/right_follower -> ttyUSB1
    - Camera:    RealSense D435i (main_camera, serial=135122077817)
    - End-effector: parallel 2-finger gripper

Usage:
    # Simulation / mock mode (macOS or development)
    python scripts/demo_fabric_dual_arm.py --mode mock

    # Real hardware mode (Linux + S1_SDK + RealSense)
    python scripts/demo_fabric_dual_arm.py --mode real \
        --left-dev /dev/left_leader --right-dev /dev/right_follower \
        --camera-serial 135122077817
"""

import os
import argparse
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_project_root = os.path.join(os.path.dirname(__file__), "..")
load_dotenv(Path(_project_root) / ".env")
sys.path.insert(0, _project_root)

# Parse mode early so we can decide whether to force mock / remove SDK path.
# (argparse will re-parse later; this is just for environment setup.)
_mode = "mock"
for i, arg in enumerate(sys.argv):
    if arg == "--mode" and i + 1 < len(sys.argv):
        _mode = sys.argv[i + 1]
        break

if _mode == "mock":
    # Force mock mode to avoid segfaults from the real S1_SDK launching
    # multiple MuJoCo viewer instances in the background.
    os.environ["OPENROBOT_FORCE_MOCK"] = "1"
    # Prevent the real S1_SDK from shadowing our mock implementation
    _sdk_path = os.path.join(os.path.dirname(_project_root), "YHRG_control", "S1_SDK_V2", "src")
    if _sdk_path in sys.path:
        sys.path.remove(_sdk_path)
else:
    # Real mode: ensure S1_SDK is on sys.path so YHRGAdapter can import it.
    _sdk_path = os.path.join(os.path.dirname(_project_root), "YHRG_control", "S1_SDK_V2", "src")
    if _sdk_path not in sys.path:
        sys.path.insert(0, _sdk_path)
    os.environ.pop("OPENROBOT_FORCE_MOCK", None)

from openrobot_demo.agent import BDIAgent, LLMPlanner, SkillRouter
from openrobot_demo.dual_arm.controller import DualArmController, ArmSide
from openrobot_demo.dual_arm.fabric_skills import FabricManipulationSkill
from openrobot_demo.experience.library import ExperienceLibrary
from openrobot_demo.experience.retriever import ExperienceRetriever
from openrobot_demo.experience.seed import seed_fabric_experiences
from openrobot_demo.persistence.db import EpisodeRecorder, init_database
from openrobot_demo.perception.vlm_cognition import VLMCognitionSensor
from openrobot_demo.sensors import (
    ProprioceptionSensor,
    RealSenseRGBSensor,
    RealSenseDepthSensor,
    TactileSensor,
    IMUSensor,
    WrenchSensor,
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
        end_effector="None",
    )
    ctrl.enable()
    return ctrl


def setup_sensors(dual_arm: DualArmController, camera_serial: str) -> list:
    """Initialize sensor channels.  RGB and Depth share one RealSense pipeline."""
    sensors = [
        RealSenseRGBSensor(source_id="rs_d435i_rgb", serial=camera_serial),
        RealSenseDepthSensor(source_id="rs_d435i_depth", serial=camera_serial),
        ProprioceptionSensor(source_id="left_arm", arm_adapter=dual_arm.left_arm, kinematics_solver=dual_arm.left_kin),
        ProprioceptionSensor(source_id="right_arm", arm_adapter=dual_arm.right_arm, kinematics_solver=dual_arm.right_kin),
        TactileSensor(source_id="left_gripper", body_names=["gripper_base", "left_finger"], mujoco_model=None, mujoco_data=None),
        TactileSensor(source_id="right_gripper", body_names=["gripper_base", "right_finger"], mujoco_model=None, mujoco_data=None),
    ]
    return sensors


def run_fabric_demo(
    mode: str = "mock",
    left_dev: str = "/dev/left_leader",
    right_dev: str = "/dev/right_follower",
    camera_serial: str = "135122077817",
    instruction: str = "将筒状布料提起，套在铝合金支撑板上，等待检测后再取下来",
    recorder: EpisodeRecorder = None,
):
    """Run the full fabric manipulation demo."""

    print("\n" + "=" * 60)
    print("  OpenRobotDemo — Dual-Arm Fabric Manipulation")
    print(f"  Mode: {mode} | Arms: {left_dev} + {right_dev}")
    print(f"  Camera: {camera_serial}")
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
    sensors = setup_sensors(dual_arm, camera_serial)
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
    # Camera skill uses realsense driver for real hardware
    camera_skill = CameraCapture(
        camera_type="realsense" if mode == "real" else "usb",
        device_id=0,
        width=640,
        height=480,
        serial=camera_serial if mode == "real" else None,
    )
    arm_reader = ArmStateReader(external_arm=dual_arm.left_arm)
    vision_estimator = Vision3DEstimator()
    grasp_predictor = GraspPointPredictor()
    arm_executor = ArmMotionExecutor(external_arm=dual_arm.left_arm)
    vla_executor = VLAPolicyExecutor(external_arm=dual_arm.left_arm)

    router.register(camera_skill)
    router.register(arm_reader)
    router.register(vision_estimator)
    router.register(grasp_predictor)
    router.register(arm_executor)
    router.register(fabric_skill)
    router.register(vla_executor)

    # 5. VLM Cognition Sensor (feeds from RGB sensor)
    vlm_sensor = VLMCognitionSensor(
        source_id="vlm_cognition",
        camera_source_id="rs_d435i_rgb",
    )
    sensors.append(vlm_sensor)

    # 6. BDI Agent (replaces manual ReAct loop)
    print("\n[5/6] Initializing BDI Agent...")
    exp_retriever = ExperienceRetriever(exp_lib)
    planner = LLMPlanner(
        experience_retriever=exp_retriever,
        skill_router=router,
        api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-max",
    )
    agent = BDIAgent(
        planner=planner,
        skill_router=router,
        world_model=world,
        max_total_steps=30,
    )
    print(f"      Instruction: {instruction}")
    print(f"      Skills: {router.list_skills()}")
    print(f"      Tool descriptions auto-generated from schemas.")

    # 7. Execute via BDI Agent
    print("\n[6/6] BDI Agent executing...\n")
    try:
        summary = agent.execute(instruction, sensors=sensors)
        finished = summary["success"]
        print(f"\n{'='*60}")
        print(f"  BDI Agent finished: success={finished}")
        print(f"  Steps: {summary['total_steps']}")
        print(f"  Time: {summary['elapsed_time_s']:.1f}s")
        if summary.get("goal_tree"):
            print(f"  Goal tree depth: {len(summary['goal_tree'].get('sub_goals', []))}")
        print(f"{'='*60}")
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user.")
        finished = False
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

    return {"success": finished, "steps": agent._total_steps}


def main():
    parser = argparse.ArgumentParser(description="Dual-Arm Fabric Manipulation Demo")
    parser.add_argument("--mode", choices=["mock", "real"], default="mock", help="Execution mode")
    parser.add_argument("--left-dev", default="/dev/left_follower", help="Left arm serial device")
    parser.add_argument("--right-dev", default="/dev/right_follower", help="Right arm serial device")
    parser.add_argument("--camera-serial", default="135122077817", help="RealSense camera serial number")
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
        camera_serial=args.camera_serial,
        instruction=args.instruction,
        recorder=recorder,
    )

    print(f"\n🏁 Demo result: success={result['success']}, steps={result['steps']}")


if __name__ == "__main__":
    main()
