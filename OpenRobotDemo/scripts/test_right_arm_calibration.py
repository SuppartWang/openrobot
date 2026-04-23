#!/usr/bin/env python3
"""
右臂方向标定 & 末端姿态标定 & 笛卡尔键盘控制测试

参考: YHRG_control/S1_SDK_V2/examples/python/keyborad_end_effect.py

流程:
  Phase 0 — 键盘控制测试（先测试再标定）
    内部使用 quaternion + inverse_quat 做 IK，避免万向节锁导致的 J1 偏转
    每 5 秒自动打印一次关节/夹爪状态

  Phase 1 — 机械臂坐标方向标定
    gravity() 模式下手动拖动，依次记录原点 → 大致 +x → 大致 +y → 大致 +z

  Phase 2 — 末端夹爪姿态标定
    gravity() 模式下手动旋转，依次记录向前(+y) → 向右(+x) → 向上(+z)

  Phase 3 — 自由键盘控制（可选）

Usage:
    cd ~/openrobot/OpenRobotDemo
    conda run -n openrobot python scripts/test_right_arm_calibration.py --dev /dev/ttyUSB1
"""

import argparse
import json
import math
import os
import sys
import time

_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)

_sdk_path = os.path.join(os.path.dirname(_project_root), "YHRG_control", "S1_SDK_V2", "src")
if _sdk_path not in sys.path:
    sys.path.insert(0, _sdk_path)

from S1_SDK import S1_arm, control_mode, S1_slover
from scipy.spatial.transform import Rotation as R


def _input_continue(prompt: str):
    input(f"\n  ➤ {prompt} [按 Enter 确认]")


def _getch_nonblocking(timeout: float = 0.05) -> str:
    """Read a single keypress without Enter (Linux TTY), or fall back to line input."""
    import select

    # If not a real TTY, fall back to non-blocking line input
    if not sys.stdin.isatty():
        if select.select([sys.stdin], [], [], timeout)[0]:
            line = sys.stdin.readline().strip()
            if line:
                return line[0]
        return ''

    try:
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            if select.select([sys.stdin], [], [], timeout)[0]:
                ch = sys.stdin.read(1)
                if ch == '\x1b':
                    seq = sys.stdin.read(2) if select.select([sys.stdin], [], [], 0.05)[0] else ''
                    return ch + seq
                return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        # termios unavailable (e.g., non-Linux or redirected stdin)
        if select.select([sys.stdin], [], [], timeout)[0]:
            line = sys.stdin.readline().strip()
            if line:
                return line[0]
    return ''


def _clear_line():
    print("\r" + " " * 80 + "\r", end='', flush=True)


def _print_tcp_status(arm, solver):
    q = arm.get_pos()[:6]
    tcp = solver.forward_eular(q)
    print(f"\n[{time.strftime('%H:%M:%S')}] "
          f"TCP位置: [{tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f}]  "
          f"偏航角: {math.degrees(tcp[5]):.1f}°")


def _keyboard_control_loop(arm, solver, tcp0_7dof, is_test=True):
    """Shared keyboard control loop using quaternion IK to avoid J1 drift.

    Internally maintains a 7-DOF target pose (position + quaternion).
    When user presses position keys, only position changes.
    When user presses rotation keys, Euler angles are updated and converted to quaternion.
    IK uses inverse_quat with current joints as seed for stability.
    """
    label = "Phase 0: 键盘控制测试" if is_test else "Phase 3: 自由键盘控制"
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)
    print("  —— 笛卡尔控制 ——")
    print("    W/S — 前/后(-x/+x)    A/D — 左/右(-y/+y)    E/Q — 升/降(+z/-z)")
    print("  —— 末端姿态控制（绕末端坐标系）——")
    print("    8/5 — 俯仰角(pitch) +/-    4/6 — 翻滚角(roll) -/+    7/9 — 偏航角(yaw) -/+")
    print("    1/2 — 绕末端X轴旋转（精细）")
    print("  —— 其他 ——")
    print("    R — 复位到初始z+5cm    C — " + ("进入下一步" if is_test else "退出"))
    print("    (每5秒自动打印一次TCP位置与偏航角)")

    deta_pos = 0.01    # 1cm
    deta_rot = 0.05    # ~3deg

    target_pos = tcp0_7dof[:3]
    target_quat = tcp0_7dof[3:7]
    target_euler = list(R.from_quat(target_quat).as_euler("xyz"))

    print(f"\n  初始位姿: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}, "
          f"{target_euler[0]:.3f}, {target_euler[1]:.3f}, {target_euler[2]:.3f}]")
    print("  开始控制...")

    last_print = time.time()
    try:
        while True:
            ch = _getch_nonblocking(0.05)
            now = time.time()
            if now - last_print >= 5.0:
                _print_tcp_status(arm, solver)
                last_print = now

            if not ch:
                q_now = arm.get_pos()[:6]
                q_target = solver.inverse_quat(target_pos + target_quat, q_now)
                if q_target is not None:
                    arm.joint_control_mit(q_target)
                elif now - last_print >= 1.0:
                    _clear_line()
                    print("⚠️ IK 求解失败，调整目标位置或姿态后重试", end='', flush=True)
                time.sleep(0.01)
                continue

            if ch in ('c', 'C'):
                print("\n  " + ("✅ 测试完成，进入标定阶段。" if is_test else "退出。"))
                break

            elif ch in ('r', 'R'):
                target_pos = list(tcp0_7dof[:3])
                target_pos[2] += 0.05
                target_quat = list(tcp0_7dof[3:7])
                target_euler = list(R.from_quat(target_quat).as_euler("xyz"))
                print("\n  [复位到初始z+5cm]")
                continue

            elif ch in ('w', 'W'):
                target_pos[0] -= deta_pos
            elif ch in ('s', 'S'):
                target_pos[0] += deta_pos
            elif ch in ('a', 'A'):
                target_pos[1] -= deta_pos
            elif ch in ('d', 'D'):
                target_pos[1] += deta_pos
            elif ch in ('e', 'E'):
                target_pos[2] += deta_pos
            elif ch in ('q', 'Q'):
                target_pos[2] -= deta_pos

            # 4/6 — roll (X axis) -/+
            elif ch in ('4', '$'):
                target_euler[0] -= deta_rot
                target_quat = list(R.from_euler("xyz", target_euler).as_quat())
            elif ch in ('6', '^'):
                target_euler[0] += deta_rot
                target_quat = list(R.from_euler("xyz", target_euler).as_quat())
            # 8/5 — pitch (Y axis) +/-
            elif ch in ('8', '*'):
                target_euler[1] += deta_rot
                target_quat = list(R.from_euler("xyz", target_euler).as_quat())
            elif ch in ('5', '%'):
                target_euler[1] -= deta_rot
                target_quat = list(R.from_euler("xyz", target_euler).as_quat())
            # 7/9 — yaw (Z axis) -/+
            elif ch in ('7', '&'):
                target_euler[2] -= deta_rot
                target_quat = list(R.from_euler("xyz", target_euler).as_quat())
            elif ch in ('9', '('):
                target_euler[2] += deta_rot
                target_quat = list(R.from_euler("xyz", target_euler).as_quat())
            # 1/2 — fine roll adjustment
            elif ch in ('1', '!'):
                target_euler[0] += deta_rot
                target_quat = list(R.from_euler("xyz", target_euler).as_quat())
            elif ch in ('2', '@'):
                target_euler[0] -= deta_rot
                target_quat = list(R.from_euler("xyz", target_euler).as_quat())
            else:
                continue

            target_pos[2] = max(0.02, min(0.60, target_pos[2]))

            q_now = arm.get_pos()[:6]
            q_target = solver.inverse_quat(target_pos + target_quat, q_now)
            if q_target is not None:
                arm.joint_control_mit(q_target)
                _clear_line()
                print(f"目标: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}, "
                      f"{math.degrees(target_euler[0]):.1f}°, "
                      f"{math.degrees(target_euler[1]):.1f}°, "
                      f"{math.degrees(target_euler[2]):.1f}°]", end='', flush=True)
            else:
                _clear_line()
                print(f"⚠️ IK 失败", end='', flush=True)
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n  中断")


def phase1_direction_calibration(arm, solver):
    """Phase 1: Direction calibration — manually drag to +x/+y/+z directions."""
    print("\n" + "=" * 60)
    print("Phase 1: 机械臂坐标方向标定")
    print("=" * 60)
    print("  说明: 进入 gravity() 重力补偿模式后，用手轻轻拖动机械臂。")
    print("  只需将末端大致移到目标方向即可，不限制精确距离。")
    print("  " + "-" * 56)

    records = {}

    print("\n  [1/4] 请将末端放在工作空间中心 (大致位置即可)")
    _input_continue("放好后")
    arm.gravity()
    time.sleep(0.5)
    q0 = arm.get_pos()[:6]
    origin = solver.forward_eular(q0)
    records["origin"] = {"joints": q0, "tcp": origin}
    print(f"    原点 TCP: [{origin[0]:.4f}, {origin[1]:.4f}, {origin[2]:.4f}]")

    print("\n  [2/4] 请将末端向前方移动 (对应臂坐标系 -x 方向)")
    _input_continue("移动好后")
    arm.gravity()
    time.sleep(0.5)
    q_fwd = arm.get_pos()[:6]
    p_fwd = solver.forward_eular(q_fwd)
    records["forward_dir"] = {"joints": q_fwd, "tcp": p_fwd}
    v_fwd = [p_fwd[i] - origin[i] for i in range(3)]
    print(f"    前方点 TCP: [{p_fwd[0]:.4f}, {p_fwd[1]:.4f}, {p_fwd[2]:.4f}]")
    print(f"    → 方向向量: [{v_fwd[0]:.4f}, {v_fwd[1]:.4f}, {v_fwd[2]:.4f}]")

    print("\n  [3/4] 请将末端向右侧移动 (对应臂坐标系 +y 方向)")
    _input_continue("移动好后")
    arm.gravity()
    time.sleep(0.5)
    q_right = arm.get_pos()[:6]
    p_right = solver.forward_eular(q_right)
    records["right_dir"] = {"joints": q_right, "tcp": p_right}
    v_right = [p_right[i] - origin[i] for i in range(3)]
    print(f"    右侧点 TCP: [{p_right[0]:.4f}, {p_right[1]:.4f}, {p_right[2]:.4f}]")
    print(f"    → 方向向量: [{v_right[0]:.4f}, {v_right[1]:.4f}, {v_right[2]:.4f}]")

    print("\n  [4/4] 请将末端向 +z 方向移动 (向上)")
    _input_continue("移动好后")
    arm.gravity()
    time.sleep(0.5)
    qz = arm.get_pos()[:6]
    pz = solver.forward_eular(qz)
    records["plus_z"] = {"joints": qz, "tcp": pz}
    vz = [pz[i] - origin[i] for i in range(3)]
    print(f"    +z 点 TCP: [{pz[0]:.4f}, {pz[1]:.4f}, {pz[2]:.4f}]")
    print(f"    → 方向向量: [{vz[0]:.4f}, {vz[1]:.4f}, {vz[2]:.4f}]")

    print("\n  —— 方向标定结果 ——")
    print(f"    原点:       [{origin[0]:.4f}, {origin[1]:.4f}, {origin[2]:.4f}]")
    print(f"    前方(-x):   [{v_fwd[0]:.4f}, {v_fwd[1]:.4f}, {v_fwd[2]:.4f}]  (模长 {math.sqrt(v_fwd[0]**2+v_fwd[1]**2+v_fwd[2]**2):.4f})")
    print(f"    右侧(+y):   [{v_right[0]:.4f}, {v_right[1]:.4f}, {v_right[2]:.4f}]  (模长 {math.sqrt(v_right[0]**2+v_right[1]**2+v_right[2]**2):.4f})")
    print(f"    上方(+z):   [{vz[0]:.4f}, {vz[1]:.4f}, {vz[2]:.4f}]  (模长 {math.sqrt(vz[0]**2+vz[1]**2+vz[2]**2):.4f})")
    print("    提示: 以上向量应分别近似指向世界坐标的前/右/上方向。")

    return records


def phase2_orientation_calibration(arm, solver):
    """Phase 2: End-effector orientation calibration."""
    print("\n" + "=" * 60)
    print("Phase 2: 末端夹爪姿态标定")
    print("=" * 60)
    print("  说明: 进入 gravity() 模式，手动旋转末端关节，")
    print("  依次使夹爪朝向: 向前(+y) → 向右(+x) → 向上(+z)")
    print("  只需大致姿态，不要求精确对准。")
    print("  " + "-" * 56)

    records = {}

    print("\n  [1/3] 请旋转末端，使夹爪大致朝向正前方 (+y 水平向前)")
    _input_continue("调好后")
    arm.gravity()
    time.sleep(0.5)
    q_fwd = arm.get_pos()[:6]
    tcp_fwd = solver.forward_quat(q_fwd)
    records["forward"] = {"joints": q_fwd, "tcp": tcp_fwd}
    print(f"    向前姿态:")
    print(f"      位置: [{tcp_fwd[0]:.4f}, {tcp_fwd[1]:.4f}, {tcp_fwd[2]:.4f}]")
    print(f"      四元数: [{tcp_fwd[3]:.4f}, {tcp_fwd[4]:.4f}, {tcp_fwd[5]:.4f}, {tcp_fwd[6]:.4f}]")

    print("\n  [2/3] 请旋转末端，使夹爪大致朝向正右方 (+x)")
    _input_continue("调好后")
    arm.gravity()
    time.sleep(0.5)
    q_right = arm.get_pos()[:6]
    tcp_right = solver.forward_quat(q_right)
    records["right"] = {"joints": q_right, "tcp": tcp_right}
    print(f"    向右姿态:")
    print(f"      位置: [{tcp_right[0]:.4f}, {tcp_right[1]:.4f}, {tcp_right[2]:.4f}]")
    print(f"      四元数: [{tcp_right[3]:.4f}, {tcp_right[4]:.4f}, {tcp_right[5]:.4f}, {tcp_right[6]:.4f}]")

    print("\n  [3/3] 请旋转末端，使夹爪大致朝向正上方 (+z 垂直向上)")
    _input_continue("调好后")
    arm.gravity()
    time.sleep(0.5)
    q_up = arm.get_pos()[:6]
    tcp_up = solver.forward_quat(q_up)
    records["up"] = {"joints": q_up, "tcp": tcp_up}
    print(f"    向上姿态:")
    print(f"      位置: [{tcp_up[0]:.4f}, {tcp_up[1]:.4f}, {tcp_up[2]:.4f}]")
    print(f"      四元数: [{tcp_up[3]:.4f}, {tcp_up[4]:.4f}, {tcp_up[5]:.4f}, {tcp_up[6]:.4f}]")

    print("\n  —— 姿态标定结果 ——")
    print(f"    向前 quaternion: [{tcp_fwd[3]:.4f}, {tcp_fwd[4]:.4f}, {tcp_fwd[5]:.4f}, {tcp_fwd[6]:.4f}]")
    print(f"    向右 quaternion: [{tcp_right[3]:.4f}, {tcp_right[4]:.4f}, {tcp_right[5]:.4f}, {tcp_right[6]:.4f}]")
    print(f"    向上 quaternion: [{tcp_up[3]:.4f}, {tcp_up[4]:.4f}, {tcp_up[5]:.4f}, {tcp_up[6]:.4f}]")

    print("\n    工具坐标系到臂基座的旋转矩阵 (近似):")
    print("    列1(x_tool→world) ≈ 向右姿态的末端X轴方向")
    print("    列2(y_tool→world) ≈ 向前姿态的末端Y轴方向")
    print("    列3(z_tool→world) ≈ 向上姿态的末端Z轴方向")
    print("    (精确值需要通过正解的旋转矩阵提取)")

    return records


def save_results(records_dir, records_rot, path: str):
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "direction_calibration": records_dir,
        "orientation_calibration": records_rot,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n  标定结果已保存: {path}")


def main():
    parser = argparse.ArgumentParser(description="右臂方向标定 & 姿态标定 & 键盘控制")
    parser.add_argument("--dev", type=str, default="/dev/ttyUSB1",
                        help="右臂串口设备 (默认 /dev/ttyUSB1)")
    parser.add_argument("--end", type=str, default="gripper",
                        help="末端执行器类型: gripper / None / teach")
    parser.add_argument("--tcp-z", type=float, default=0.10,
                        help="TCP Z方向偏移量 (米), 默认 0.10")
    parser.add_argument("--skip-test", action="store_true",
                        help="跳过键盘测试，直接进入标定")
    parser.add_argument("--output", type=str, default="right_arm_calibration.json",
                        help="标定结果保存路径")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("右臂方向标定 & 末端姿态标定")
    print("=" * 60)
    print(f"  设备: {args.dev}")
    print(f"  末端: {args.end}")
    print(f"  TCP偏移Z: {args.tcp_z} m")

    print("\n[1/4] 初始化机械臂...")
    arm = S1_arm(
        mode=control_mode.only_real,
        dev=args.dev,
        end_effector=args.end,
        check_collision=False,
    )
    solver = S1_slover(end_offset=[0.0, 0.0, args.tcp_z])
    arm.enable()
    print("      机械臂已使能")

    q = arm.get_pos()[:6]
    tcp = solver.forward_quat(q)
    print(f"\n  当前关节角: {[f'{v:.3f}' for v in q]}")
    print(f"  当前TCP位姿: [{tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f}, "
          f"{tcp[3]:.4f}, {tcp[4]:.4f}, {tcp[5]:.4f}, {tcp[6]:.4f}]")

    if not args.skip_test:
        _keyboard_control_loop(arm, solver, tcp, is_test=True)

    records_dir = phase1_direction_calibration(arm, solver)
    records_rot = phase2_orientation_calibration(arm, solver)
    save_results(records_dir, records_rot, args.output)

    _keyboard_control_loop(arm, solver, tcp, is_test=False)

    print("\n[4/4] 关闭机械臂...")
    arm.disable()
    arm.close()
    print("      完成")


if __name__ == "__main__":
    main()
