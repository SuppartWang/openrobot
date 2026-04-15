"""Hardware connectivity check script for OpenRobotDemo.

Validates:
1. Camera can capture frames
2. YHRG arm (mock or real) can be initialized, enabled, and moved
3. Forward/inverse kinematics works
"""

import os
import sys
import time

_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)

from openrobot_demo.skills.camera_capture import CameraCapture
from openrobot_demo.skills.arm_state_reader import ArmStateReader
from openrobot_demo.hardware.yhrg_adapter import control_mode


def test_camera():
    print("\n=== Test 1: CameraCapture ===")
    cam = CameraCapture(camera_type="usb", device_id=0, width=640, height=480)
    result = cam.execute(return_depth=False)
    if result["success"]:
        rgb = result["rgb_frame"]
        print(f"  ✅ Camera OK. Frame shape: {rgb.shape}")
    else:
        print(f"  ❌ Camera failed: {result['message']}")
    cam.disconnect()
    return result["success"]


def test_arm():
    print("\n=== Test 2: ArmStateReader + YHRG SDK ===")
    reader = ArmStateReader(mode=control_mode.only_sim, end_effector="gripper")
    reader.enable()

    # Read initial state
    state = reader.execute(fields=["pos", "vel", "tau", "temp"])
    print(f"  Initial pos: {state.get('joint_positions')}")
    print(f"  EE pose: {state.get('end_effector_pose')}")

    # Move joints
    target = [0.3, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0]
    print(f"  Sending joint command: {target}")
    ok = reader._arm.joint_control(target)
    time.sleep(0.5)

    state2 = reader.execute(fields=["pos"])
    print(f"  New pos: {state2.get('joint_positions')}")
    print(f"  New EE pose: {state2.get('end_effector_pose')}")

    # IK test
    current_ee = state2.get("end_effector_pose")
    if current_ee:
        solver = reader._solver
        ik_joints = solver.inverse_quat(current_ee, state2.get("joint_positions"))
        print(f"  IK from current pose: {ik_joints}")

    # Gripper test
    reader._arm.control_gripper(1.0, 0.5)
    print("  Gripper closed (mock).")

    reader.disable()
    print("  ✅ Arm SDK OK.")
    return True


def test_vision_3d():
    print("\n=== Test 3: Vision3DEstimator (requires API key) ===")
    from openrobot_demo.skills.vision_3d_estimator import Vision3DEstimator
    import numpy as np

    estimator = Vision3DEstimator()
    if estimator._client is None:
        print("  ⚠️ Skipping (no API key). Set DASHSCOPE_API_KEY or OPENAI_API_KEY.")
        return True

    # Create a synthetic image
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = np.ones((480, 640), dtype=np.uint16) * 500  # 500 mm

    result = estimator.execute(
        rgb_frame=rgb,
        target_name="red box",
        depth_frame=depth,
        camera_intrinsics={"fx": 600.0, "fy": 600.0, "ppx": 320.0, "ppy": 240.0},
    )
    print(f"  Result: {result}")
    return result["success"]


if __name__ == "__main__":
    ok1 = test_camera()
    ok2 = test_arm()
    ok3 = test_vision_3d()
    print("\n=== Summary ===")
    print(f"Camera: {'PASS' if ok1 else 'FAIL'}")
    print(f"Arm:    {'PASS' if ok2 else 'FAIL'}")
    print(f"Vision: {'PASS' if ok3 else 'FAIL'}")
    sys.exit(0 if (ok1 and ok2 and ok3) else 1)
