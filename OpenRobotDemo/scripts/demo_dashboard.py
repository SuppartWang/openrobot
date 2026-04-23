"""
OpenRobotDemo — 三栏式可视化控制台启动入口

Usage:
    # Real hardware mode with GUI
    python scripts/demo_dashboard.py --mode real \
        --left-dev /dev/left_follower --right-dev /dev/right_follower \
        --camera-serial 135122077817

    # Mock mode (no hardware required)
    python scripts/demo_dashboard.py --mode mock
"""

import os
import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

_project_root = os.path.join(os.path.dirname(__file__), "..")
load_dotenv(Path(_project_root) / ".env")
sys.path.insert(0, _project_root)

# Parse mode early so we can decide whether to force mock / remove SDK path.
_mode = "mock"
for i, arg in enumerate(sys.argv):
    if arg == "--mode" and i + 1 < len(sys.argv):
        _mode = sys.argv[i + 1]
        break

if _mode == "mock":
    os.environ["OPENROBOT_FORCE_MOCK"] = "1"
    _sdk_path = os.path.join(os.path.dirname(_project_root), "YHRG_control", "S1_SDK_V2", "src")
    if _sdk_path in sys.path:
        sys.path.remove(_sdk_path)
else:
    _sdk_path = os.path.join(os.path.dirname(_project_root), "YHRG_control", "S1_SDK_V2", "src")
    if _sdk_path not in sys.path:
        sys.path.insert(0, _sdk_path)
    os.environ.pop("OPENROBOT_FORCE_MOCK", None)

from openrobot_demo.agent import BDIAgent, LLMPlanner
from openrobot_demo.dual_arm.controller import DualArmController, ArmSide
from openrobot_demo.dual_arm.fabric_skills import FabricManipulationSkill
from openrobot_demo.experience.library import ExperienceLibrary
from openrobot_demo.experience.retriever import ExperienceRetriever
from openrobot_demo.experience.seed import seed_fabric_experiences
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
from openrobot_demo.ui.dashboard import RobotDashboard
from openrobot_demo.world_model import WorldModel, ObjectDesc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def setup_experiences() -> ExperienceLibrary:
    lib = ExperienceLibrary()
    existing = lib.list_all(limit=1)
    if not existing:
        seed_fabric_experiences(lib)
    else:
        print(f"[Setup] Experience library already has {len(existing)} records, skipping seed.")
    return lib


def setup_dual_arm(mode: str, left_dev: str, right_dev: str) -> DualArmController:
    ctrl = DualArmController(
        left_dev=left_dev,
        right_dev=right_dev,
        mode=mode,
        end_effector="None",
    )
    ctrl.enable()
    return ctrl


def setup_sensors(dual_arm: DualArmController, camera_serial: str) -> list:
    sensors = [
        RealSenseRGBSensor(source_id="rs_d435i_rgb", serial=camera_serial),
        RealSenseDepthSensor(source_id="rs_d435i_depth", serial=camera_serial),
        ProprioceptionSensor(source_id="left_arm", arm_adapter=dual_arm.left_arm, kinematics_solver=dual_arm.left_kin),
        ProprioceptionSensor(source_id="right_arm", arm_adapter=dual_arm.right_arm, kinematics_solver=dual_arm.right_kin),
        TactileSensor(source_id="left_gripper", body_names=["gripper_base", "left_finger"], mujoco_model=None, mujoco_data=None),
        TactileSensor(source_id="right_gripper", body_names=["gripper_base", "right_finger"], mujoco_model=None, mujoco_data=None),
    ]
    return sensors


def main():
    parser = argparse.ArgumentParser(description="OpenRobotDemo 三栏可视化控制台")
    parser.add_argument("--mode", choices=["mock", "real"], default="mock", help="Execution mode")
    parser.add_argument("--left-dev", default="/dev/left_follower", help="Left arm serial device")
    parser.add_argument("--right-dev", default="/dev/right_follower", help="Right arm serial device")
    parser.add_argument("--camera-serial", default="135122077817", help="RealSense camera serial number")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  OpenRobotDemo — 三栏可视化控制台")
    print(f"  Mode: {args.mode}")
    print("=" * 60 + "\n")

    # 1. Experience library
    print("[1/5] Loading experience library...")
    exp_lib = setup_experiences()

    # 2. Dual-arm controller
    print("[2/5] Initializing dual-arm controller...")
    dual_arm = setup_dual_arm(args.mode, args.left_dev, args.right_dev)
    print(f"      Left  arm pos:  {dual_arm.get_pos(ArmSide.LEFT)[:6]}")
    print(f"      Right arm pos:  {dual_arm.get_pos(ArmSide.RIGHT)[:6]}")

    # 3. World model & sensors
    print("[3/5] Initializing world model & sensors...")
    world = WorldModel()
    world.add_surface("workbench")
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
    sensors = setup_sensors(dual_arm, args.camera_serial)
    for s in sensors:
        print(f"      Sensor {s.name}/{s.source_id}: available={s.is_available()}")

    # 4. Skills
    print("[4/5] Registering skills...")
    from openrobot_demo.agent import SkillRouter
    router = SkillRouter()
    fabric_skill = FabricManipulationSkill(
        dual_arm=dual_arm,
        experience_library=exp_lib,
        world_model=world,
    )
    camera_skill = CameraCapture(
        camera_type="realsense" if args.mode == "real" else "usb",
        device_id=0,
        width=640,
        height=480,
        serial=args.camera_serial if args.mode == "real" else None,
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

    # 5. VLM Cognition Sensor
    vlm_sensor = VLMCognitionSensor(
        source_id="vlm_cognition",
        camera_source_id="rs_d435i_rgb",
    )
    sensors.append(vlm_sensor)

    # 6. BDI Agent
    print("[5/5] Initializing BDI Agent...")
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
    print(f"      Skills: {router.list_skills()}")

    # 7. Launch dashboard
    print("\n🚀 启动可视化界面...")
    dash = RobotDashboard(
        agent=agent,
        sensors=sensors,
        dual_arm=dual_arm,
    )
    try:
        dash.run()
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


if __name__ == "__main__":
    main()
