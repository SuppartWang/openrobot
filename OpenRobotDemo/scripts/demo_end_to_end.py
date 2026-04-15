"""End-to-end demo: natural language -> Agent -> Skills -> YHRG arm execution."""

import os
import sys
import yaml

_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)

from openrobot_demo.agent.planner import LLMPlanner
from openrobot_demo.agent.skill_router import SkillRouter
from openrobot_demo.skills.camera_capture import CameraCapture
from openrobot_demo.skills.arm_state_reader import ArmStateReader
from openrobot_demo.skills.vision_3d_estimator import Vision3DEstimator
from openrobot_demo.skills.grasp_predictor import GraspPointPredictor
from openrobot_demo.skills.arm_executor import ArmMotionExecutor
from openrobot_demo.hardware.yhrg_adapter import control_mode


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cam_cfg = load_config(os.path.join(_project_root, "configs", "camera_config.yaml"))
    arm_cfg = load_config(os.path.join(_project_root, "configs", "arm_config.yaml"))

    # Initialize skills
    camera_skill = CameraCapture(
        camera_type=cam_cfg["camera"]["type"],
        device_id=cam_cfg["camera"]["device_id"],
        width=cam_cfg["camera"]["width"],
        height=cam_cfg["camera"]["height"],
    )
    arm_reader = ArmStateReader(
        mode=control_mode[arm_cfg["arm"]["mode"]],
        dev=arm_cfg["arm"]["dev"],
        end_effector=arm_cfg["arm"]["end_effector"],
        check_collision=arm_cfg["arm"]["check_collision"],
    )
    vision_estimator = Vision3DEstimator()
    grasp_predictor = GraspPointPredictor()
    arm_executor = ArmMotionExecutor(
        mode=control_mode[arm_cfg["arm"]["mode"]],
        dev=arm_cfg["arm"]["dev"],
        end_effector=arm_cfg["arm"]["end_effector"],
        check_collision=arm_cfg["arm"]["check_collision"],
    )

    # Register skills
    router = SkillRouter()
    router.register(camera_skill)
    router.register(arm_reader)
    router.register(vision_estimator)
    router.register(grasp_predictor)
    router.register(arm_executor)

    # Plan
    instruction = "Pick up the target object on the table."
    print(f"\n🤖 User instruction: {instruction}")

    planner = LLMPlanner()
    plan = planner.plan(instruction)
    print(f"\n📋 Plan generated ({len(plan)} steps):")
    for i, step in enumerate(plan, 1):
        print(f"  {i}. {step['skill']}({step.get('args', {})})")

    # Enrich plan args with camera intrinsics / hand-eye if needed
    intrinsics = cam_cfg.get("intrinsics", {})
    hand_eye = cam_cfg.get("hand_eye", {})

    for step in plan:
        if step["skill"] == "vision_3d_estimator":
            step["args"]["camera_intrinsics"] = intrinsics
            if hand_eye.get("enabled"):
                step["args"]["hand_eye_calib"] = {
                    "rotation_matrix": hand_eye["rotation_matrix"],
                    "translation_vector": hand_eye["translation_vector"],
                }
                # We will inject end_effector_pose at execution time via context
        elif step["skill"] == "grasp_point_predictor":
            if "object_type" not in step["args"]:
                step["args"]["object_type"] = "box"

    # Pre-populate arm state into router context so vision_3d_estimator can use EE pose for hand-eye
    arm_state = arm_reader.execute(fields=["pos"])
    router._context["end_effector_pose"] = arm_state.get("end_effector_pose")
    router._context["joint_positions"] = arm_state.get("joint_positions")

    # Execute
    print("\n▶️ Executing plan...")
    result = router.execute_plan(plan)

    print(f"\n✅ Result: {result['message']}")
    if not result["success"]:
        print("Detailed results:")
        for r in result.get("results", []):
            print(f"  Step {r['step']} {r['skill']}: success={r['result'].get('success')} | {r['result'].get('message')}")
    else:
        print("All steps completed successfully!")
        final_context = {k: v for k, v in router._context.items() if k not in ["rgb_frame", "depth_frame"]}
        print("Final context keys:", list(final_context.keys()))

    # Cleanup
    camera_skill.disconnect()
    arm_executor.disable()
    arm_reader.disable()


if __name__ == "__main__":
    main()
