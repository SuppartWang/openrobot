#!/usr/bin/env python3
"""
真机预规划安全测试脚本
========================
用法:
    python scripts/test_preplan_real.py --mode real \
        --left-dev /dev/left_follower --right-dev /dev/right_follower \
        --camera-serial 135122077817

测试菜单:
    1. 视觉检测 + 坐标转换（不动机械臂，最安全）
    2. 夹爪开合测试（先夹后开）
    3. 单臂 z 轴移动（↑/↓ 箭头，每步 2cm）
    4. 双臂 z 轴移动（↑/↓ 箭头，每步 2cm）
    5. 单臂坐标点动（输入 xyz 匀速移动）
    6. Phase 1 完整流程（需二次确认）
    7. 完整 20 步预规划（需二次确认）
    0. 退出
"""

from __future__ import annotations

import argparse
import os
import sys
import termios
import tty
import select
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# ------------------------------------------------------------------
# Environment setup (must be before importing demo_web_dashboard)
# ------------------------------------------------------------------
_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)

# Trick: make demo_web_dashboard.py think we're in real mode
old_argv = list(sys.argv)
sys.argv = ["demo_web_dashboard.py", "--mode", "real"]

from dotenv import load_dotenv
load_dotenv(Path(_project_root) / ".env")

# Import after env setup
from scripts.demo_web_dashboard import setup_hardware
from openrobot_demo.sensors.realsense_shared import RealSenseDevicePool

# Restore argv
sys.argv = old_argv


def _getch(timeout: float = 0.3) -> Optional[str]:
    """Read a single keypress from stdin without pressing Enter (Linux only).
    Supports ANSI escape sequences for arrow keys.
    Skips residual newline characters from previous input.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            if not select.select([sys.stdin], [], [], timeout)[0]:
                return None
            ch = sys.stdin.read(1)
            # Skip residual newline characters from previous input (e.g. after 'y\n')
            if ch in ('\n', '\r'):
                continue
            if ch == '\x1b':  # ESC - could be start of escape sequence
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch += sys.stdin.read(2)  # e.g. '[A' or '[B'
            return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _capture_images(sensors):
    """Capture one frame from RealSense RGB + Depth sensors."""
    rgb_sensor = next((s for s in sensors if s.source_id == "rs_d435i_rgb"), None)
    depth_sensor = next((s for s in sensors if s.source_id == "rs_d435i_depth"), None)
    rgb_frame, depth_frame = None, None
    if rgb_sensor and rgb_sensor.is_available():
        reading = rgb_sensor.capture()
        if hasattr(reading, "payload"):
            rgb_frame = reading.payload
    if depth_sensor and depth_sensor.is_available():
        reading = depth_sensor.capture()
        if hasattr(reading, "payload"):
            depth_frame = reading.payload
    return rgb_frame, depth_frame


def _exec_skill(skill, **kwargs) -> Dict[str, Any]:
    if skill is None:
        return {"success": False, "message": "Skill not available"}
    try:
        return skill.execute(**kwargs)
    except Exception as e:
        print(f"  [ERROR] Skill {getattr(skill, 'name', '?')} failed: {e}")
        return {"success": False, "message": str(e)}


def _detect_with_vlm_backup(frame, target_name, instruction, api_key):
    """VLM target detection using backup approach (center point only).
    Resizes image to 640px width before sending to VLM for reliability.
    Args:
        frame: RGB image
        target_name: display name of target
        instruction: specific detection instruction for this target
        api_key: VLM API key
    Returns:
        (cx, cy) in original image coordinates, or None.
    """
    import cv2
    from openai import OpenAI
    import base64
    import re

    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    h, w, _ = frame.shape
    scale_ratio = 640 / w
    detect_w, detect_h = 640, int(h * scale_ratio)
    resized_frame = cv2.resize(frame, (detect_w, detect_h))

    _, buffer = cv2.imencode('.jpg', resized_frame)
    base64_image = base64.b64encode(buffer).decode("utf-8")

    # Per-target customized prompt
    prompt_text = f"""你是一个精确的机器视觉定位系统。
请在宽{detect_w}，高{detect_h}的图片中，{instruction}。
要求：
1. 必须精准定位该物体指定位置的坐标。
2. 绝对禁止输出任何额外解释文字！
3. 格式只能是包含两个数字的纯JSON数组：[中心X, 中心Y]
如果找不到目标，请返回 [0, 0]"""

    try:
        response = client.chat.completions.create(
            model="qwen-vl-max-latest",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": prompt_text}
                ]
            }],
            temperature=0.1,
            max_tokens=100
        )

        result_text = response.choices[0].message.content.strip()
        print(f"  [VLM] 模型回复: {result_text}")

        clean_text = re.sub(r'[a-zA-Z"\'{}]+', '', result_text)
        nums = re.findall(r'\d+', clean_text)

        if len(nums) >= 2:
            x, y = map(int, nums[:2])
            if x == 0 and y == 0:
                print("  [VLM] 模型判定画面中不存在该物体。")
                return None
            real_x = int(x / scale_ratio)
            real_y = int(y / scale_ratio)
            return (real_x, real_y)
        else:
            print("  [VLM] 模型返回的数据里数字不足！")
            return None
    except Exception as e:
        print(f"  [ERROR] VLM 调用失败: {e}")
        return None


def test_vision_only(hw, apply_depth_filters: bool = False):
    """Test 1: Vision detection + coordinate transform using backup approach.
    Directly uses pyrealsense2 aligned frames + VLM center-point detection.
    Outputs: original image, annotated image with detection points, 3D coordinates.
    Args:
        apply_depth_filters: Enable spatial/temporal/hole-filling filters on depth.
    """
    print("\n" + "=" * 60)
    print("测试 1: 视觉检测 + 坐标转换（不动机械臂）")
    print("=" * 60)
    print(f"  深度滤波: {'开启' if apply_depth_filters else '关闭 (默认)'}")

    import cv2
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("  [ERROR] pyrealsense2 未安装")
        return

    # Get API key
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    if not api_key:
        print("  [ERROR] 未找到 API Key（DASHSCOPE_API_KEY 或 QWEN_API_KEY）")
        return

    # Find sensor serial
    sensors = hw["sensors"]
    rgb_sensor = next((s for s in sensors if s.source_id == "rs_d435i_rgb"), None)
    if rgb_sensor is None:
        print("  [ERROR] RealSense RGB 传感器未加载")
        return
    serial = rgb_sensor.serial
    if not serial:
        print("  [ERROR] 无法获取相机序列号")
        return

    # Capture aligned frames directly from shared pool
    print("\n[1/4] 正在捕获 RGB + Depth 对齐图像...")
    try:
        color_frame, depth_frame = RealSenseDevicePool.capture_frames(serial, apply_filters=apply_depth_filters)
        if not color_frame or not depth_frame:
            print("  [ERROR] 未能获取帧")
            return

        # Filters may return generic rs.frame instead of rs.depth_frame
        # Cast back to depth_frame to use get_distance()
        if hasattr(depth_frame, 'as_depth_frame'):
            depth_frame = depth_frame.as_depth_frame()

        # BGR image from RealSense
        color_image_bgr = np.asanyarray(color_frame.get_data())
        # RGB for VLM
        color_image_rgb = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)
        # Intrinsics for deprojection
        intrinsics = RealSenseDevicePool.get_intrinsics(serial)

        print(f"  ✅ 图像获取成功: {color_image_rgb.shape}")
    except Exception as e:
        print(f"  [ERROR] 捕获图像失败: {e}")
        return

    # Save original RGB image
    try:
        save_path_original = os.path.abspath("test_rgb_original.jpg")
        cv2.imwrite(save_path_original, color_image_bgr)
        print(f"  💾 原始图像已保存到: {save_path_original}")
    except Exception as e:
        print(f"  [WARN] 保存原始图像失败: {e}")

    # Save original depth map (16-bit PNG + pseudo-color visualization)
    try:
        depth_raw = np.asanyarray(depth_frame.get_data())  # uint16 in mm
        save_path_depth_raw = os.path.abspath("test_depth_raw.png")
        cv2.imwrite(save_path_depth_raw, depth_raw)
        print(f"  💾 原始深度图已保存到: {save_path_depth_raw} (uint16 mm)")

        # Pseudo-color visualization for human inspection
        depth_m = depth_raw.astype(np.float32) * 0.001  # mm -> m
        depth_vis = np.clip(depth_m / 2.0 * 255, 0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        save_path_depth_color = os.path.abspath("test_depth_color.jpg")
        cv2.imwrite(save_path_depth_color, depth_color)
        print(f"  💾 伪彩色深度图已保存到: {save_path_depth_color}")
    except Exception as e:
        print(f"  [WARN] 保存深度图失败: {e}")

    # Detection targets with customized instructions and distinct colors
    targets = [
        ("筒状布料下边沿", "找到筒状布料的下边沿中心点的坐标，不是图像的中心点。布料通常呈扁平状放置在桌面上"),
        ("筒状布料下边沿左侧顶点", "找到筒状布料的左侧开口处，返回左侧开口边缘的坐标。顶点是布料卷筒左侧的边缘"),
        ("筒状布料下边沿右侧顶点", "找到筒状布料的右侧开口处，返回右侧开口边缘的坐标。顶点是布料卷筒右侧的边缘"),
        ("悬垂布料下沿中点", "找到从画面最上方悬垂下来的半圆形的布料，返回其下沿中心点的坐标，不是图像的中心点。注意寻找布料的次低点，可以不是最低点，但是一定要在布料内部"),
    ]
    # BGR color palette for each target
    colors = [
        (0, 0, 255),      # Red
        (0, 255, 0),      # Green
        (255, 0, 0),      # Blue
        (0, 255, 255),    # Yellow
    ]

    result_frame = color_image_bgr.copy()
    detected_any = False

    for idx, (target_name, instruction) in enumerate(targets):
        color = colors[idx]
        print(f"\n[检测] {target_name}...")
        print(f"  [提示词] {instruction}")
        point = _detect_with_vlm_backup(color_image_rgb, target_name, instruction, api_key)

        if point:
            cx, cy = point
            # Validate pixel bounds
            h, w = color_image_bgr.shape[:2]
            cx = max(0, min(cx, w - 1))
            cy = max(0, min(cy, h - 1))

            # Get depth at center point
            distance_m = depth_frame.get_distance(cx, cy)

            if distance_m > 0 and intrinsics:
                point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], distance_m)
                print(f"  ✅ 2D 像素坐标: u={cx}, v={cy}")
                print(f"  ✅ 深度距离: {distance_m:.3f}m")
                print(f"  ✅ 相机坐标系 3D: X={point_3d[0]:.3f}m, Y={point_3d[1]:.3f}m, Z={point_3d[2]:.3f}m")

                # Draw colored dot (no text to avoid ??? encoding issues)
                cv2.circle(result_frame, (cx, cy), 8, color, -1)
                cv2.circle(result_frame, (cx, cy), 10, (255, 255, 255), 2)
                detected_any = True
            else:
                print(f"  ⚠️ 深度数据无效 (distance={distance_m:.3f}m)")
                cv2.circle(result_frame, (cx, cy), 8, (0, 165, 255), -1)
        else:
            print(f"  ⚠️ 未检测到 {target_name}")

    # Save annotated image
    try:
        save_path_annotated = os.path.abspath("test_rgb_annotated.jpg")
        cv2.imwrite(save_path_annotated, result_frame)
        print(f"\n💾 标注图像已保存到: {save_path_annotated}")
    except Exception as e:
        print(f"\n[WARN] 保存标注图像失败: {e}")

    if detected_any:
        print("\n✅ 视觉测试完成！请检查标注图像中的检测点和 3D 坐标。")
    else:
        print("\n⚠️ 视觉测试完成，但未检测到任何目标。请检查：")
        print("   1. 目标物体是否在相机视野内")
        print("   2. VLM API Key 是否有效")
        print("   3. 光线是否充足")


def test_gripper(hw):
    """Test 2: Close then open grippers."""
    print("\n" + "=" * 60)
    print("测试 2: 夹爪开合（先夹后开）")
    print("=" * 60)

    gripper = hw.get("skills", {}).get("gripper_control")
    if gripper is None:
        print("  [ERROR] gripper_control 未加载")
        return

    confirm = input("\n  ⚠️  即将执行夹爪动作，请确保夹爪周围安全 [y/N]: ")
    if confirm.lower() != "y":
        print("  已取消")
        return

    print("\n[1/4] 张开左臂夹爪...")
    _exec_skill(gripper, side="left", position=1.0, force=0.5)
    time.sleep(1)

    print("[2/4] 张开右臂夹爪...")
    _exec_skill(gripper, side="right", position=1.0, force=0.5)
    time.sleep(1)

    print("[3/4] 闭合双臂夹爪...")
    _exec_skill(gripper, side="both", position=0.0, force=0.5)
    time.sleep(1)

    print("[4/4] 再次张开双臂夹爪...")
    _exec_skill(gripper, side="both", position=1.0, force=0.5)
    time.sleep(1)

    print("\n✅ 夹爪测试完成")


def test_single_arm_sequential(hw):
    """Test 3: Control left and right arms sequentially (raise/lower each arm)."""
    print("\n" + "=" * 60)
    print("测试 3: 单臂依次抬起/落下")
    print("=" * 60)

    skills = hw.get("skills", {})
    dual_motion = skills.get("dual_arm_coordinated_motion")
    if dual_motion is None:
        print("  [ERROR] dual_arm_coordinated_motion 未加载")
        return

    # Show current poses
    dual_arm = hw["dual_arm"]
    try:
        lp = dual_arm.get_ee_pose("left")
        rp = dual_arm.get_ee_pose("right")
        print(f"\n  当前左臂 z: {lp[2]:.3f}  右臂 z: {rp[2]:.3f}")
    except Exception as e:
        print(f"  [WARN] 获取当前位姿失败: {e}")

    confirm = input("\n  ⚠️  即将执行单臂依次抬起/落下，请确认双臂周围安全 [y/N]: ")
    if confirm.lower() != "y":
        print("  已取消")
        return

    print(f"\n{'='*50}")
    print("  单臂依次控制")
    print(f"{'='*50}")
    print("  W = 左臂抬起 2cm")
    print("  S = 左臂落下 2cm")
    print("  E = 右臂抬起 2cm")
    print("  D = 右臂落下 2cm")
    print("  其他任意键 = 双臂恢复初始位置并退出")
    print(f"{'='*50}\n")

    step_m = 0.02  # 2cm

    while True:
        ch = _getch(timeout=0.5)
        if ch is None:
            continue

        key = ch.lower()

        if key == 'w':
            print("  [W] 左臂抬起 2cm")
            r = _exec_skill(dual_motion, command_type="relative", side="left",
                            z_offset=step_m, duration=0.5)
            if not r.get("success"):
                print(f"    [FAIL] {r.get('message')}")
            time.sleep(0.5)
        elif key == 's':
            print("  [S] 左臂落下 2cm")
            r = _exec_skill(dual_motion, command_type="relative", side="left",
                            z_offset=-step_m, duration=0.5)
            if not r.get("success"):
                print(f"    [FAIL] {r.get('message')}")
            time.sleep(0.5)
        elif key == 'e':
            print("  [E] 右臂抬起 2cm")
            r = _exec_skill(dual_motion, command_type="relative", side="right",
                            z_offset=step_m, duration=0.5)
            if not r.get("success"):
                print(f"    [FAIL] {r.get('message')}")
            time.sleep(0.5)
        elif key == 'd':
            print("  [D] 右臂落下 2cm")
            r = _exec_skill(dual_motion, command_type="relative", side="right",
                            z_offset=-step_m, duration=0.5)
            if not r.get("success"):
                print(f"    [FAIL] {r.get('message')}")
            time.sleep(0.5)
        else:
            print("\n  恢复双臂初始位置...")
            left_arm = skills.get("arm_motion_executor_left")
            right_arm = skills.get("arm_motion_executor_right")
            home = [0.2, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0]
            if left_arm:
                _exec_skill(left_arm, command_type="cartesian", target_values=home, speed=0.5)
            if right_arm:
                _exec_skill(right_arm, command_type="cartesian", target_values=home, speed=0.5)
            time.sleep(2)
            print("\n✅ 单臂依次控制测试完成")
            break


def test_dual_arm_z(hw):
    """Test 4: Raise/lower both arms simultaneously with W/S keys."""
    print("\n" + "=" * 60)
    print("测试 4: 双臂同时抬起/落下")
    print("=" * 60)

    dual_motion = hw.get("skills", {}).get("dual_arm_coordinated_motion")
    if dual_motion is None:
        print("  [ERROR] dual_arm_coordinated_motion 未加载")
        return

    # Show current poses
    dual_arm = hw["dual_arm"]
    try:
        lp = dual_arm.get_ee_pose("left")
        rp = dual_arm.get_ee_pose("right")
        print(f"\n  当前左臂 z: {lp[2]:.3f}  右臂 z: {rp[2]:.3f}")
    except Exception as e:
        print(f"  [WARN] 获取当前位姿失败: {e}")

    confirm = input("\n  ⚠️  即将执行双臂同时抬起/落下，请确认双臂周围安全 [y/N]: ")
    if confirm.lower() != "y":
        print("  已取消")
        return

    print(f"\n{'='*50}")
    print("  双臂同时控制")
    print(f"{'='*50}")
    print("  W = 双臂同时抬起 2cm")
    print("  S = 双臂同时落下 2cm")
    print("  其他任意键 = 恢复初始位置并退出")
    print(f"{'='*50}\n")

    step_m = 0.02  # 2cm

    while True:
        ch = _getch(timeout=0.5)
        if ch is None:
            continue

        key = ch.lower()

        if key == 'w':
            print("  [W] 双臂同时抬起 2cm")
            r = _exec_skill(dual_motion, command_type="relative", side="both",
                            z_offset=step_m, duration=0.5)
            if not r.get("success"):
                print(f"    [FAIL] {r.get('message')}")
            time.sleep(0.5)
        elif key == 's':
            print("  [S] 双臂同时落下 2cm")
            r = _exec_skill(dual_motion, command_type="relative", side="both",
                            z_offset=-step_m, duration=0.5)
            if not r.get("success"):
                print(f"    [FAIL] {r.get('message')}")
            time.sleep(0.5)
        else:
            print("\n  恢复双臂初始位置...")
            skills = hw.get("skills", {})
            left_arm = skills.get("arm_motion_executor_left")
            right_arm = skills.get("arm_motion_executor_right")
            home = [0.2, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0]
            if left_arm:
                _exec_skill(left_arm, command_type="cartesian", target_values=home, speed=0.5)
            if right_arm:
                _exec_skill(right_arm, command_type="cartesian", target_values=home, speed=0.5)
            time.sleep(2)
            print("\n✅ 双臂同时控制测试完成")
            break


def test_goto_xyz(hw):
    """Test 5: Input x,y,z and move single arm to target position."""
    print("\n" + "=" * 60)
    print("测试 5: 单臂坐标点动（输入 xyz 匀速移动）")
    print("=" * 60)

    skills = hw.get("skills", {})
    left_arm = skills.get("arm_motion_executor_left")
    right_arm = skills.get("arm_motion_executor_right")
    dual_arm = hw["dual_arm"]

    if left_arm is None or right_arm is None:
        print("  [ERROR] 臂运动执行器未加载")
        return

    # Choose arm
    side = input("  选择手臂 [l=左臂 / r=右臂]: ").strip().lower()
    if side not in ("l", "left", "r", "right"):
        print("  已取消")
        return
    use_left = side in ("l", "left")
    arm_skill = left_arm if use_left else right_arm
    arm_name = "左臂" if use_left else "右臂"
    side_str = "left" if use_left else "right"

    # Show current pose and reachable range
    try:
        current_pose = dual_arm.get_ee_pose(side_str)
        cx, cy, cz = current_pose[0], current_pose[1], current_pose[2]
        print(f"\n  当前 {arm_name} 位姿: [{cx:.3f}, {cy:.3f}, {cz:.3f}]")
        print(f"\n  📐 建议可达范围（相对于当前位置）:")
        print(f"     x: [{cx-0.15:.3f}, {cx+0.15:.3f}]")
        print(f"     y: [{cy-0.15:.3f}, {cy+0.15:.3f}]")
        print(f"     z: [{max(0.02, cz-0.15):.3f}, {min(0.55, cz+0.15):.3f}]")
        print(f"     提示: 优先尝试 z 方向 ±5cm 的移动，最容易成功")
    except Exception as e:
        print(f"  [WARN] 获取当前位姿失败: {e}")
        print(f"\n  📐 典型可达范围（{arm_name} 基座坐标系）:")
        print(f"     x: [-0.20, 0.40]  y: [-0.20, 0.40]  z: [0.02, 0.55]")

    confirm = input(f"\n  ⚠️  即将进入 {arm_name} 坐标点动模式，请确认周围安全 [y/N]: ")
    if confirm.lower() != "y":
        print("  已取消")
        return

    while True:
        print(f"\n{'='*50}")
        print(f"  {arm_name} 坐标点动")
        print(f"{'='*50}")
        print("  输入格式: x y z（空格分隔，单位: 米）")
        print("  安全示例: 0.20 0.00 0.30（水平前伸，高度 30cm）")
        print("  安全示例: 0.15 0.10 0.25（右前方，高度 25cm）")
        print("  输入 q 退出并恢复初始位置")
        print(f"{'='*50}")

        coord = input(f"  {arm_name} 目标坐标 x y z: ").strip()
        if coord.lower() == 'q':
            break

        try:
            parts = coord.split()
            if len(parts) != 3:
                raise ValueError("需要输入 3 个数字")
            x, y, z = map(float, parts)
        except ValueError as e:
            print(f"  [ERROR] 格式错误: {e}")
            continue

        speed_str = input("  速度 0.1~1.0 [默认0.3]: ").strip()
        try:
            speed = float(speed_str) if speed_str else 0.3
        except ValueError:
            speed = 0.3

        # Build target pose: keep current orientation, only change position
        try:
            current = dual_arm.get_ee_pose(side_str)
            target = [x, y, z] + list(current[3:7])
        except Exception as e:
            print(f"  [WARN] 获取当前姿态失败: {e}，使用默认向下姿态")
            target = [x, y, z, 0.0, 0.0, 0.0, 1.0]

        duration = max(1.0, 2.0 / max(0.1, speed))
        print(f"\n  移动到: [{x:.3f}, {y:.3f}, {z:.3f}]，耗时 {duration:.1f}s...")

        r = _exec_skill(arm_skill, command_type="cartesian", target_values=target, speed=speed)
        if r.get("success"):
            print(f"  ✅ 到达目标")
        else:
            msg = r.get('message', '')
            print(f"  ⚠️ 运动失败: {msg}")
            if "IK failed" in msg or "unreachable" in msg:
                print(f"\n  💡 建议: 目标位置可能不可达。请尝试以下坐标：")
                try:
                    cx, cy, cz = dual_arm.get_ee_pose(side_str)[:3]
                    print(f"     当前位置附近: {cx:.2f} {cy:.2f} {cz:.2f}")
                    print(f"     仅 z 变化 +5cm: {cx:.2f} {cy:.2f} {cz+0.05:.2f}")
                    print(f"     仅 z 变化 -5cm: {cx:.2f} {cy:.2f} {max(0.02, cz-0.05):.2f}")
                except Exception:
                    print(f"     安全测试点: 0.20 0.00 0.30")
        time.sleep(duration + 0.5)

    # Home on exit
    print(f"\n  恢复 {arm_name} 初始位置...")
    home = [0.2, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0]
    r = _exec_skill(arm_skill, command_type="cartesian", target_values=home, speed=0.5)
    print(f"  结果: {r.get('message')}")
    time.sleep(2)
    print(f"\n✅ {arm_name} 坐标点动测试完成")


def test_full_phase1(hw, args=None):
    """Test 6: Execute full Phase 1 with extra safety prompt."""
    print("\n" + "=" * 60)
    print("测试 6: Phase 1 完整流程")
    print("=" * 60)
    print("  步骤: 视觉检测 → 移动到开口上方 → 插入 → 闭合 → 抬升 → 前伸 → 松开 → 复位")

    confirm = input("\n  ⚠️⚠️  即将执行 Phase 1 完整流程，请确认：\n"
                    "      1. 双臂周围无遮挡\n"
                    "      2. 筒状布料已放置在桌面上\n"
                    "      3. 紧急停止按钮在手边\n"
                    "      输入 'PHASE1' 确认执行: ")
    if confirm != "PHASE1":
        print("  已取消")
        return

    for i in range(5, 0, -1):
        print(f"  ⏱️  {i} 秒后开始...")
        time.sleep(1)

    skills = hw.get("skills", {})
    sensors = hw["sensors"]
    dual_arm = hw["dual_arm"]

    coord_skill = skills.get("coordinate_transform")
    left_arm = skills.get("arm_motion_executor_left")
    right_arm = skills.get("arm_motion_executor_right")
    gripper = skills.get("gripper_control")
    dual_motion = skills.get("dual_arm_coordinated_motion")

    # Default coords (fallback)
    left_opening_pos = [0.18, 0.20, 0.05]
    right_opening_pos = [-0.18, 0.20, 0.05]

    step = 0
    def _notify(msg, ok=True):
        nonlocal step
        step += 1
        print(f"  [{step:02d}] {'✅' if ok else '⚠️'} {msg}")

    # ---- Step 1: Vision detection using pyrealsense2 directly (same as Test 1) ----
    rgb_sensor = next((s for s in sensors if s.source_id == "rs_d435i_rgb"), None)
    if rgb_sensor is None:
        _notify("找不到 RGB 传感器，使用默认坐标", ok=False)
    else:
        serial = rgb_sensor.serial
        try:
            color_frame, depth_frame = RealSenseDevicePool.capture_frames(
                serial, apply_filters=args.apply_depth_filters if args else False)
            if hasattr(depth_frame, 'as_depth_frame'):
                depth_frame = depth_frame.as_depth_frame()

            color_image_bgr = np.asanyarray(color_frame.get_data())
            color_image_rgb = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)
            _notify("图像获取成功")

            intrinsics = RealSenseDevicePool.get_intrinsics(serial)
            if intrinsics is None:
                _notify("获取相机内参失败，使用默认坐标", ok=False)
            else:
                api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")

                targets_detect = [
                    ("筒状布料左侧开口",
                     "找到筒状布料的左侧开口处，返回左侧开口边缘的中心点坐标",
                     "left", 0.03),
                    ("筒状布料右侧开口",
                     "找到筒状布料的右侧开口处，返回右侧开口边缘的中心点坐标",
                     "right", 0.03),
                ]

                for target_name, instruction, arm_side, offset_z in targets_detect:
                    point = _detect_with_vlm_backup(
                        color_image_rgb, target_name, instruction, api_key)
                    if point is None:
                        _notify(f"VLM 未检测到 {target_name}，使用默认坐标", ok=False)
                        continue

                    cx, cy = point
                    distance_m = depth_frame.get_distance(cx, cy)
                    _notify(f"{target_name} 深度: {distance_m:.3f}m @ ({cx},{cy})")

                    if distance_m > 0 and intrinsics:
                        cam_3d = rs.rs2_deproject_pixel_to_point(
                            intrinsics, [cx, cy], distance_m)
                        _notify(f"  相机坐标: [{cam_3d[0]:.4f}, {cam_3d[1]:.4f}, {cam_3d[2]:.4f}]")

                        if coord_skill:
                            tf = _exec_skill(coord_skill, point_camera=cam_3d,
                                             target_frame=arm_side, offset_z=offset_z)
                            if tf.get("success"):
                                pt = tf.get("point")[:3]
                                if arm_side == "left":
                                    left_opening_pos = pt
                                else:
                                    right_opening_pos = pt
                                _notify(f"{target_name} 臂基坐标: {pt}")
                            else:
                                _notify(f"坐标转换失败: {tf.get('error')}", ok=False)
                        else:
                            _notify("坐标转换技能不可用", ok=False)
                    else:
                        _notify(f"深度无效 ({distance_m})，使用默认坐标", ok=False)

                # Save debug images
                save_dir = os.path.join(os.path.dirname(__file__), "..", "test_outputs")
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_dir, "phase1_rgb.jpg"), color_image_bgr)
                _notify("保存 phase1_rgb.jpg")

        except Exception as e:
            _notify(f"视觉检测异常: {e}，使用默认坐标", ok=False)

    _notify(f"最终左坐标: {left_opening_pos}")
    _notify(f"最终右坐标: {right_opening_pos}")

    # ---- Standard orientations ----
    # Arm zero pose is forward-facing; identity = forward
    ORIENTATION_FORWARD = [0.0, 0.0, 0.0, 1.0]
    # Rotate -90° around x-axis to face vertically down
    ORIENTATION_DOWNWARD = [-0.7071, 0.0, 0.0, 0.7071]

    # Build poses with explicit orientations
    HOME_LEFT = [0.2, 0.0, 0.25] + ORIENTATION_FORWARD
    HOME_RIGHT = [0.2, 0.0, 0.25] + ORIENTATION_FORWARD
    left_above = list(left_opening_pos) + ORIENTATION_DOWNWARD
    right_above = list(right_opening_pos) + ORIENTATION_DOWNWARD
    left_insert = [left_opening_pos[0], left_opening_pos[1], left_opening_pos[2] - 0.03] + ORIENTATION_DOWNWARD
    right_insert = [right_opening_pos[0], right_opening_pos[1], right_opening_pos[2] - 0.03] + ORIENTATION_DOWNWARD

    # ---- Move sequence ----
    if gripper:
        _exec_skill(gripper, side="both", position=1.0, force=0.5)
        _notify("双臂夹爪张开")
        time.sleep(0.5)

    if left_arm:
        r = _exec_skill(left_arm, command_type="cartesian", target_values=left_above, speed=0.5)
        _notify(f"左臂移动到 {left_above[:3]} (垂直向下)", ok=r.get("success", False))
        time.sleep(2)

    if right_arm:
        r = _exec_skill(right_arm, command_type="cartesian", target_values=right_above, speed=0.5)
        _notify(f"右臂移动到 {right_above[:3]} (垂直向下)", ok=r.get("success", False))
        time.sleep(2)

    if left_arm:
        r = _exec_skill(left_arm, command_type="cartesian", target_values=left_insert, speed=0.3)
        _notify(f"左臂插入 z={left_insert[2]:.3f}", ok=r.get("success", False))
        time.sleep(2)

    if right_arm:
        r = _exec_skill(right_arm, command_type="cartesian", target_values=right_insert, speed=0.3)
        _notify(f"右臂插入 z={right_insert[2]:.3f}", ok=r.get("success", False))
        time.sleep(2)

    if gripper:
        _exec_skill(gripper, side="both", position=0.0, force=0.8)
        _notify("夹爪闭合")
        time.sleep(1)

    # 抬升到 z=0.5m，同时切换为朝向前方
    if left_arm and right_arm:
        try:
            left_current = dual_arm.get_ee_pose("left")
            right_current = dual_arm.get_ee_pose("right")
            left_lift = [left_current[0], left_current[1], 0.5] + ORIENTATION_FORWARD
            right_lift = [right_current[0], right_current[1], 0.5] + ORIENTATION_FORWARD
            dual_arm.dual_move_cartesian(left_lift, right_lift, duration=3.0)
            _notify("双臂抬升到 z=0.5m 且末端朝前")
            time.sleep(3.5)
        except Exception as e:
            _notify(f"抬升失败: {e}", ok=False)

    if dual_motion:
        r = _exec_skill(dual_motion, command_type="relative", side="both", y_offset=-0.4, duration=2.0)
        _notify("前伸 0.4m", ok=r.get("success", False))
        time.sleep(2.5)

    if gripper:
        _exec_skill(gripper, side="both", position=1.0, force=0.5)
        _notify("夹爪松开")
        time.sleep(1)

    if left_arm:
        r = _exec_skill(left_arm, command_type="cartesian", target_values=HOME_LEFT, speed=0.5)
        _notify("左臂复位 (垂直向下)", ok=r.get("success", False))
        time.sleep(2)

    if right_arm:
        r = _exec_skill(right_arm, command_type="cartesian", target_values=HOME_RIGHT, speed=0.5)
        _notify("右臂复位 (垂直向下)", ok=r.get("success", False))
        time.sleep(2)

    print("\n✅ Phase 1 测试完成")


def test_full_preplan(hw):
    """Test 7: Run the full 20-step preplanned sequence."""
    print("\n" + "=" * 60)
    print("测试 7: 完整 20 步预规划")
    print("=" * 60)

    confirm = input("\n  ⚠️⚠️⚠️  即将执行完整 20 步预规划，请确认所有安全条件满足\n"
                    "      输入 'FULL' 确认执行: ")
    if confirm != "FULL":
        print("  已取消")
        return

    for i in range(5, 0, -1):
        print(f"  ⏱️  {i} 秒后开始...")
        time.sleep(1)

    from scripts.demo_web_dashboard import run_preplanned_demo

    class ConsoleState:
        def __init__(self):
            self.task_running = True
            self.step_count = 0
        def add_chat(self, role, msg):
            self.step_count += 1
            print(f"  [{self.step_count:02d}] {msg}")
        def add_exec(self, row):
            pass
        def update_action(self, action):
            print(f"  [ACTION] {action}")
        def update_vlm(self, img):
            pass
        def update_goal_tree(self, tree):
            pass
        def get_snapshot(self):
            return {}

    state = ConsoleState()
    run_preplanned_demo(state, hw["dual_arm"], hw["sensors"], hw.get("skills", {}), "真机测试")
    print(f"\n✅ 完整预规划执行完成，总步数: {state.step_count}")


def main():
    parser = argparse.ArgumentParser(description="真机预规划安全测试")
    parser.add_argument("--mode", choices=["mock", "real"], default="real")
    parser.add_argument("--left-dev", default="/dev/left_follower")
    parser.add_argument("--right-dev", default="/dev/right_follower")
    parser.add_argument("--camera-serial", default="135122077817")
    parser.add_argument("--tcp-offset-z", type=float, default=0.10)
    parser.add_argument("--apply-depth-filters", action="store_true",
                        help="Enable RealSense spatial/temporal/hole-filling depth filters (default: off)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  OpenRobotDemo — 真机预规划安全测试")
    print(f"  Mode: {args.mode}")
    print("=" * 60)

    if args.mode == "real":
        print("\n⚠️  真机模式 — 请确保：")
        print("   1. 双臂周围 0.5m 内无障碍物")
        print("   2. 紧急停止按钮在手边")
        print("   3. 测试人员远离机械臂运动范围")
        print("")
        confirm = input("  安全确认 [yes/no]: ")
        if confirm.lower() not in ("yes", "y"):
            print("  已取消")
            return

    print("\n[1/2] 初始化硬件...")
    hw = setup_hardware(args.mode, args.left_dev, args.right_dev, args.camera_serial,
                        end_effector="gripper", tcp_offset_z=args.tcp_offset_z)
    print("      硬件就绪")

    while True:
        print("\n" + "=" * 60)
        print("测试菜单:")
        print("  1. 视觉检测 + 坐标转换（不动机械臂）")
        print("  2. 夹爪开合测试（先夹后开）")
        print("  3. 单臂依次抬起/落下（W/S/E/D 键）")
        print("  4. 双臂同时抬起/落下（W/S 键）")
        print("  5. 单臂坐标点动（输入 xyz 匀速移动）")
        print("  6. Phase 1 完整流程（视觉→提起→前伸→释放）")
        print("  7. 完整 20 步预规划")
        print("  0. 退出")
        print("=" * 60)

        choice = input("选择测试项: ").strip()

        if choice == "1":
            test_vision_only(hw, args.apply_depth_filters)
        elif choice == "2":
            test_gripper(hw)
        elif choice == "3":
            test_single_arm_sequential(hw)
        elif choice == "4":
            test_dual_arm_z(hw)
        elif choice == "5":
            test_goto_xyz(hw)
        elif choice == "6":
            test_full_phase1(hw, args)
        elif choice == "7":
            test_full_preplan(hw)
        elif choice == "0":
            break
        else:
            print("无效选择")

    print("\n[2/2] 清理...")
    hw["dual_arm"].disable()
    for s in hw["sensors"]:
        if hasattr(s, "close"):
            try:
                s.close()
            except Exception:
                pass
    hw["exp_lib"].close()
    print("      测试结束")


if __name__ == "__main__":
    main()
