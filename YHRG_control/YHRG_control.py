"""
YHRG S1 机械臂控制脚本

功能：
    1. 支持指定串口号（/dev/ttyUSB* 等）初始化单臂或双臂。
    2. 支持笛卡尔空间位姿控制（运动到指定空间坐标）。
    3. 内部通过 S1_slover 做逆运动学求解，并以步进插值方式平滑运动。
    4. 保留与视觉系统（step1_capture_detect）联动的旧版接口。
    5. 支持仿真模式（MuJoCo 无头仿真），便于在无实机环境调试。

参考：OpenRobotDemo/openrobot_demo/hardware/yhrg_adapter.py
"""

from __future__ import annotations

import os
import time
import argparse
import logging
from typing import List, Optional

# ---------------------------------------------------------------------------
# 外部依赖可用性检测
# ---------------------------------------------------------------------------
try:
    from S1_SDK import S1_arm, control_mode, S1_slover, Arm_Search
    _SDK_AVAILABLE = True
except Exception as _e:
    _SDK_AVAILABLE = False
    logging.warning(f"[YHRGControl] S1_SDK 未加载 ({_e})，启用回退模式。")

try:
    import mujoco
    _MUJOCO_AVAILABLE = True
except Exception:
    _MUJOCO_AVAILABLE = False


# ---------------------------------------------------------------------------
# 当 S1_SDK 不可用时，提供基础 Mock 实现
# ---------------------------------------------------------------------------
if not _SDK_AVAILABLE:
    class control_mode:
        only_sim = 0
        only_real = 1
        real_control_sim = 2

    class S1_arm:
        _JOINT_LIMITS = [
            (-2.967, 2.967), (0.0, 3.142), (0.0, 2.967),
            (-1.571, 1.518), (-1.571, 1.571), (-1.571, 1.571),
        ]

        def __init__(self, mode, dev="/dev/ttyUSB0", end_effector="None",
                     check_collision=True, arm_version="V2"):
            self.dev = dev
            self._pos = [0.0] * 7
            self._enabled = False

        def enable(self):
            self._enabled = True
            return True

        def disable(self):
            self._enabled = False
            return True

        def joint_control(self, pos: List[float]) -> bool:
            controlled = list(pos) + [0.0] * (6 - len(pos)) if len(pos) < 6 else pos[:6]
            self._pos[:6] = [max(l, min(v, h)) for v, (l, h) in zip(controlled, self._JOINT_LIMITS)]
            return True

        def joint_control_mit(self, pos: List[float]) -> bool:
            return self.joint_control(pos)

        def control_gripper(self, pos: float, force: float = 0.5):
            self._pos[6] = float(max(0.0, min(2.0, pos)))

        def get_pos(self) -> List[float]:
            return self._pos.copy()

        def get_vel(self) -> List[float]:
            return [0.0] * 7

        def get_tau(self) -> List[float]:
            return [0.0] * 7

        def get_temp(self) -> List[float]:
            return [25.0] * 7

        def set_zero_position(self):
            self._pos = [0.0] * 7

        def set_end_zero_position(self):
            pass

        def gravity(self, return_tau: bool = False):
            return [0.0] * 7 if return_tau else None

        def check_collision(self, qpos: List[float]) -> bool:
            return False

        def close(self):
            self.disable()

    class S1_slover:
        def __init__(self, end_offset: List[float] = None):
            self.offset = end_offset or [0.0, 0.0, 0.0]

        def forward_quat(self, qpos: List[float]) -> List[float]:
            return [0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]

        def forward_eular(self, qpos: List[float]) -> List[float]:
            return [0.3, 0.0, 0.2, 0.0, 0.0, 0.0]

        def inverse_quat(self, target: List[float], joints: List[float] = None) -> Optional[List[float]]:
            return [0.0] * 6

        def inverse_eular(self, target: List[float], joints: List[float] = None) -> Optional[List[float]]:
            return [0.0] * 6

    def Arm_Search(bus: str, end_effector: str = "None", arm_version: str = "V2") -> bool:
        return True


# ---------------------------------------------------------------------------
# MuJoCo 无头仿真机械臂（当真实 SDK 不可用时，提供基于物理模型的仿真）
# ---------------------------------------------------------------------------
class _HeadlessMujocoArm:
    """
    基于 MuJoCo 的无头仿真机械臂。

    不启动 GUI，仅加载 S1 的 XML 模型并执行前向计算，
    用于在无实机/无 SDK 环境下验证控制逻辑。
    """

    def __init__(self, dev: str = "/dev/ttyUSB0", end_effector: str = "None", arm_version: str = "V2"):
        self.dev = dev
        self.end_effector = end_effector
        self._enabled = False

        xml_name = "gripper_less.xml" if end_effector == "None" else "gripper.xml"
        candidates = [
            os.path.join(os.path.dirname(__file__), "S1_SDK_V2", "src", "S1_SDK", "resource", "meshes", xml_name),
            os.path.join("S1_SDK_V2", "src", "S1_SDK", "resource", "meshes", xml_name),
        ]
        xml_path = None
        for c in candidates:
            if os.path.exists(c):
                xml_path = c
                break
        if xml_path is None:
            raise FileNotFoundError(f"找不到 MuJoCo XML 文件: {xml_name}，请确认 S1_SDK_V2 路径正确。")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        print(f"[_HeadlessMujocoArm] 加载 {xml_name} | nq={self.model.nq} | dev={dev}")

    # ---------------- 生命周期 ----------------
    def enable(self):
        self._enabled = True
        return True

    def disable(self):
        self._enabled = False
        return True

    def close(self):
        self.disable()

    # ---------------- 控制 ----------------
    def joint_control(self, pos: List[float]) -> bool:
        for i in range(min(6, len(pos))):
            self.data.qpos[i] = pos[i]
        mujoco.mj_forward(self.model, self.data)
        return True

    def joint_control_mit(self, pos: List[float]) -> bool:
        return self.joint_control(pos)

    def control_gripper(self, pos: float, force: float = 0.5):
        # S1 gripper 范围 0~2.1 -> MuJoCo slide joint 范围 0~0.05
        slide = (pos / 2.1) * 0.05
        if self.model.nq > 6:
            self.data.qpos[6] = slide
        if self.model.nq > 7:
            self.data.qpos[7] = -slide
        mujoco.mj_forward(self.model, self.data)

    # ---------------- 状态读取 ----------------
    def get_pos(self) -> List[float]:
        gripper = 0.0
        if self.model.nq > 6:
            slide = self.data.qpos[6]
            gripper = (float(slide) / 0.05) * 2.1
        return [float(v) for v in self.data.qpos[:6]] + [gripper]

    def get_vel(self) -> List[float]:
        return [0.0] * 7

    def get_tau(self) -> List[float]:
        return [0.0] * 7

    def get_temp(self) -> List[float]:
        return [25.0] * 7

    def set_zero_position(self):
        self.data.qpos[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_end_zero_position(self):
        pass

    def gravity(self, return_tau: bool = False):
        return [0.0] * 7 if return_tau else None

    def check_collision(self, qpos: List[float]) -> bool:
        return False

    # ---------------- 额外工具：获取 MuJoCo 真实连杆位置 ----------------
    def get_body_position(self, body_name: str) -> List[float]:
        """获取指定 body 在世界坐标系中的位置（用于验证）。"""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return list(self.data.xpos[body_id])


# ---------------------------------------------------------------------------
# 单臂控制器
# ---------------------------------------------------------------------------
class SingleArmController:
    """
    单臂控制器，支持关节空间与笛卡尔空间运动。

    Parameters
    ----------
    dev : str
        串口设备路径，例如 ``/dev/ttyUSB0`` 或 ``COM23``。
    end_effector : str
        末端执行器类型："None" / "gripper" / "teach"。
    arm_version : str
        机械臂硬件版本，默认 "V2"。
    check_collision : bool
        是否开启碰撞检测。
    end_effector_offset : List[float] | None
        工具中心点（TCP）相对于法兰盘的偏移量 [x, y, z]（米）。
    arm_mode : str
        运行模式："real"=真实 SDK，"sim"=MuJoCo 无头仿真，"mock"=纯 Mock。
    """

    def __init__(
        self,
        dev: str,
        end_effector: str = "None",
        arm_version: str = "V2",
        check_collision: bool = True,
        end_effector_offset: Optional[List[float]] = None,
        arm_mode: str = "real",
    ):
        self.dev = dev
        self.end_effector = end_effector
        self.end_effector_offset = end_effector_offset or [0.0, 0.0, 0.0]

        # 根据 arm_mode 选择底层实现
        if arm_mode == "real":
            if _SDK_AVAILABLE:
                self.arm = S1_arm(
                    mode=control_mode.only_real,
                    dev=dev,
                    end_effector=end_effector,
                    check_collision=check_collision,
                    arm_version=arm_version,
                )
                print(f"[SingleArm] 真实硬件模式 | dev={dev}")
            else:
                print("[SingleArm] 警告: 请求 real 模式但 S1_SDK 不可用，降级到 sim/mock。")
                arm_mode = "sim" if _MUJOCO_AVAILABLE else "mock"

        if arm_mode == "sim":
            if _MUJOCO_AVAILABLE:
                self.arm = _HeadlessMujocoArm(dev=dev, end_effector=end_effector, arm_version=arm_version)
                print(f"[SingleArm] MuJoCo 无头仿真模式 | dev={dev}")
            else:
                print("[SingleArm] 警告: 请求 sim 模式但 MuJoCo 未安装，降级到 mock。")
                arm_mode = "mock"

        if arm_mode == "mock":
            self.arm = S1_arm(
                mode=control_mode.only_real,
                dev=dev,
                end_effector=end_effector,
                check_collision=check_collision,
                arm_version=arm_version,
            )
            print(f"[SingleArm] 纯 Mock 模式 | dev={dev}")

        self.solver = S1_slover(end_offset=self.end_effector_offset)
        print(f"[SingleArm] 初始化完成 | end_effector={end_effector} | "
              f"offset={self.end_effector_offset}")

    # ------------------------------ 生命周期 ------------------------------
    def enable(self):
        self.arm.enable()
        print(f"[SingleArm] {self.dev} 已使能。")

    def disable(self):
        self.arm.disable()
        print(f"[SingleArm] {self.dev} 已失能。")

    def close(self):
        self.arm.close()

    # ------------------------------ 状态读取 ------------------------------
    def get_joint_positions(self) -> List[float]:
        """获取 6 维关节角度（弧度）。"""
        return self.arm.get_pos()[:6]

    def get_joint_velocities(self) -> List[float]:
        return self.arm.get_vel()[:6]

    def get_joint_torques(self) -> List[float]:
        return self.arm.get_tau()[:6]

    def get_end_effector_pose(self, format: str = "quat") -> List[float]:
        """
        获取末端执行器位姿。

        format="quat"  -> [x, y, z, qx, qy, qz, qw]
        format="euler" -> [x, y, z, rx, ry, rz] (弧度，xyz 顺序)
        """
        q = self.get_joint_positions()
        if format == "quat":
            return self.solver.forward_quat(q)
        elif format == "euler":
            return self.solver.forward_eular(q)
        else:
            raise ValueError("format 只能是 'quat' 或 'euler'")

    # ------------------------------ 关节空间控制 ------------------------------
    def set_joint_positions(self, positions: List[float], use_mit: bool = False) -> bool:
        """直接下发关节角度（6-DOF）。"""
        pos_6 = positions[:6] if len(positions) >= 6 else list(positions) + [0.0] * (6 - len(positions))
        if use_mit:
            return self.arm.joint_control_mit(pos_6)
        return self.arm.joint_control(pos_6)

    # ------------------------------ 笛卡尔空间控制 ------------------------------
    def move_to_pose(
        self,
        target_pose: List[float],
        pose_format: str = "quat",
        steps: int = 50,
        step_time: float = 0.02,
        use_mit: bool = False,
    ) -> bool:
        """
        控制末端运动到指定空间坐标（带步进插值）。

        Parameters
        ----------
        target_pose : List[float]
            quat  -> [x, y, z, qx, qy, qz, qw]
            euler -> [x, y, z, rx, ry, rz]
        pose_format : str
            "quat" 或 "euler"。
        steps : int
            插值步数，越大运动越平滑。
        step_time : float
            每步等待时间（秒）。
        use_mit : bool
            是否使用 MIT 模式下发（响应更快、更平滑）。

        Returns
        -------
        bool
            是否成功完成运动。
        """
        current_q = self.get_joint_positions()

        if pose_format == "quat":
            if len(target_pose) != 7:
                raise ValueError("quat 格式需要 7 个值 [x, y, z, qx, qy, qz, qw]")
            current_tcp = self.solver.forward_quat(current_q)
            ik_func = self.solver.inverse_quat
        elif pose_format == "euler":
            if len(target_pose) != 6:
                raise ValueError("euler 格式需要 6 个值 [x, y, z, rx, ry, rz]")
            current_tcp = self.solver.forward_eular(current_q)
            ik_func = self.solver.inverse_eular
        else:
            raise ValueError("pose_format 只能是 'quat' 或 'euler'")

        print(f"[SingleArm] 当前末端位姿: {['%.4f' % v for v in current_tcp]}")
        print(f"[SingleArm] 目标末端位姿: {['%.4f' % v for v in target_pose]}")
        print(f"[SingleArm] 开始步进运动 (steps={steps}, step_time={step_time}s, MIT={use_mit})...")

        success = True
        for i in range(1, steps + 1):
            alpha = i / steps
            interp = [current_tcp[j] + alpha * (target_pose[j] - current_tcp[j]) for j in range(len(target_pose))]
            target_q = ik_func(interp, current_q)

            if target_q is None or len(target_q) < 6:
                print(f"[SingleArm] 警告: 第 {i}/{steps} 步逆解失败，跳过。")
                success = False
                continue

            self.set_joint_positions(target_q, use_mit=use_mit)
            time.sleep(step_time)

        print(f"[SingleArm] 运动结束 ({'成功' if success else '存在逆解失败'})。\n")
        return success

    def move_to_position(
        self,
        x: float,
        y: float,
        z: float,
        orientation: Optional[List[float]] = None,
        pose_format: str = "quat",
        **kwargs,
    ) -> bool:
        """
        便捷接口：运动到指定位置（可选姿态）。

        注意：这里的目标位置是 **工具中心点（TCP）** 的坐标。
        若初始化时传入了 end_effector_offset，求解器会自动将该偏移
        纳入逆解计算，即你只需关心工具尖端的实际空间坐标。

        orientation 为 None 时保持当前姿态不变。
        """
        if orientation is None:
            current_pose = self.get_end_effector_pose(format="quat")
            target = [x, y, z] + current_pose[3:7]
            return self.move_to_pose(target, pose_format="quat", **kwargs)
        else:
            target = [x, y, z] + list(orientation)
            return self.move_to_pose(target, pose_format=pose_format, **kwargs)

    def move_relative(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        **kwargs,
    ) -> bool:
        """基于当前末端位姿做相对偏移运动。"""
        current = self.get_end_effector_pose(format="quat")
        target = [
            current[0] + dx,
            current[1] + dy,
            current[2] + dz,
            current[3], current[4], current[5], current[6],
        ]
        return self.move_to_pose(target, pose_format="quat", **kwargs)

    def control_gripper(self, position: float, force: float = 0.5):
        self.arm.control_gripper(position, force)

    def set_zero_position(self):
        self.arm.set_zero_position()

    def gravity(self, return_tau: bool = False):
        return self.arm.gravity(return_tau=return_tau)


# ---------------------------------------------------------------------------
# 双臂控制器
# ---------------------------------------------------------------------------
class DualArmController:
    """
    双臂控制器，封装两个 SingleArmController，方便同步控制。
    """

    def __init__(
        self,
        left_dev: str,
        right_dev: str,
        left_end_effector: str = "None",
        right_end_effector: str = "None",
        left_offset: Optional[List[float]] = None,
        right_offset: Optional[List[float]] = None,
        arm_mode: str = "real",
    ):
        self.left = SingleArmController(left_dev, end_effector=left_end_effector,
                                        end_effector_offset=left_offset, arm_mode=arm_mode)
        self.right = SingleArmController(right_dev, end_effector=right_end_effector,
                                         end_effector_offset=right_offset, arm_mode=arm_mode)

    def enable(self):
        self.left.enable()
        self.right.enable()
        print("[DualArm] 双臂已使能。")

    def disable(self):
        self.left.disable()
        self.right.disable()
        print("[DualArm] 双臂已失能。")

    def close(self):
        self.left.close()
        self.right.close()

    # ---------------------- 左臂/右臂独立笛卡尔控制 ----------------------
    def move_left_to_pose(self, target_pose: List[float], **kwargs) -> bool:
        return self.left.move_to_pose(target_pose, **kwargs)

    def move_right_to_pose(self, target_pose: List[float], **kwargs) -> bool:
        return self.right.move_to_pose(target_pose, **kwargs)

    def move_left_to_position(self, x: float, y: float, z: float, **kwargs) -> bool:
        return self.left.move_to_position(x, y, z, **kwargs)

    def move_right_to_position(self, x: float, y: float, z: float, **kwargs) -> bool:
        return self.right.move_to_position(x, y, z, **kwargs)

    # -------------------------- 旧版兼容接口 --------------------------
    def move_forward(self, distance_m: float, steps: int = 50, step_time: float = 0.02):
        """
        双臂同步沿当前 X 轴方向前进指定距离（保留旧版行为）。
        """
        safe_margin = 0.05
        actual = distance_m - safe_margin if distance_m > safe_margin else 0.0
        if actual <= 0:
            print("[DualArm] 目标距离太近，不执行移动。")
            return

        print(f"[DualArm] 同步前进 {actual:.3f}m")
        left_pose = self.left.get_end_effector_pose(format="quat")
        right_pose = self.right.get_end_effector_pose(format="quat")

        left_target = left_pose.copy()
        left_target[0] += actual
        right_target = right_pose.copy()
        right_target[0] += actual

        self.left.move_to_pose(left_target, pose_format="quat", steps=steps, step_time=step_time)
        self.right.move_to_pose(right_target, pose_format="quat", steps=steps, step_time=step_time)


# ---------------------------------------------------------------------------
# 主程序入口（CLI + 视觉联动两种模式）
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="YHRG S1 机械臂笛卡尔空间运动控制脚本"
    )
    parser.add_argument("--mode", type=str, default="cli",
                        choices=["cli", "vision"],
                        help="运行模式: cli=命令行直接控制, vision=与视觉系统联动（旧版行为）")
    parser.add_argument("--dev", type=str, default="/dev/ttyUSB0",
                        help="单臂串口设备，例如 /dev/ttyUSB0 或 COM23")
    parser.add_argument("--left_dev", type=str, default="/dev/ttyUSB0",
                        help="左臂串口（双臂/vision模式）")
    parser.add_argument("--right_dev", type=str, default="/dev/ttyUSB1",
                        help="右臂串口（双臂/vision模式）")
    parser.add_argument("--arm_mode", type=str, default="real",
                        choices=["real", "sim", "mock"],
                        help="机械臂底层模式: real=真实SDK, sim=MuJoCo无头仿真, mock=纯Mock")
    parser.add_argument("--x", type=float, default=None,
                        help="目标位置 X (米)")
    parser.add_argument("--y", type=float, default=0.0,
                        help="目标位置 Y (米)")
    parser.add_argument("--z", type=float, default=None,
                        help="目标位置 Z (米)")
    parser.add_argument("--pose_format", type=str, default="quat",
                        choices=["quat", "euler"],
                        help="位姿格式")
    parser.add_argument("--orientation", type=float, nargs="+", default=None,
                        help="目标姿态（quat: qx qy qz qw | euler: rx ry rz）")
    parser.add_argument("--steps", type=int, default=50,
                        help="插值步数")
    parser.add_argument("--step_time", type=float, default=0.02,
                        help="每步时间间隔（秒）")
    parser.add_argument("--mit", action="store_true",
                        help="使用 MIT 关节控制模式")
    parser.add_argument("--end", type=str, default="None",
                        help="末端执行器类型: None / gripper / teach")
    parser.add_argument("--dual", action="store_true",
                        help="启用双臂模式（仅对 vision 模式有意义）")
    parser.add_argument("--api_key", type=str, default=None,
                        help="视觉系统 API Key（vision 模式需要）")
    parser.add_argument("--target_obj", type=str, default="目标物体",
                        help="视觉检测目标描述（vision 模式）")

    args = parser.parse_args()

    # ========================= CLI 直接控制模式 =========================
    if args.mode == "cli":
        if args.x is None or args.z is None:
            parser.error("--cli 模式下必须提供 --x 和 --z（--y 默认为 0）")

        arm = SingleArmController(dev=args.dev, end_effector=args.end, arm_mode=args.arm_mode)
        arm.enable()
        try:
            success = arm.move_to_position(
                x=args.x,
                y=args.y,
                z=args.z,
                orientation=args.orientation,
                pose_format=args.pose_format,
                steps=args.steps,
                step_time=args.step_time,
                use_mit=args.mit,
            )
            print(f"\n最终结果: 运动 {'成功' if success else '失败'}")
        except KeyboardInterrupt:
            print("\n收到中断信号，停止运动...")
        finally:
            arm.disable()
            arm.close()
        return

    # ========================= 视觉联动模式（旧版行为） =========================
    if args.mode == "vision":
        try:
            from step1_capture_detect import VisionSystem
        except ImportError as e:
            print(f"错误: 无法导入视觉模块 step1_capture_detect ({e})")
            return

        if not args.api_key:
            print("错误: vision 模式需要提供 --api_key")
            return

        vision_system = VisionSystem(api_key=args.api_key, target_obj=args.target_obj)

        if args.dual:
            arm_controller = DualArmController(
                left_dev=args.left_dev,
                right_dev=args.right_dev,
                arm_mode=args.arm_mode,
            )
        else:
            # 单臂 vision 模式默认使用 left_dev
            arm_controller = SingleArmController(dev=args.left_dev, end_effector=args.end, arm_mode=args.arm_mode)

        arm_controller.enable()

        try:
            while True:
                distance = vision_system.wait_and_detect()
                if distance == -1:
                    print("检测到退出指令。")
                    break
                elif distance is not None:
                    if args.dual:
                        arm_controller.move_forward(distance, steps=args.steps, step_time=args.step_time)
                    else:
                        arm_controller.move_relative(dx=distance, steps=args.steps, step_time=args.step_time)
                else:
                    print("未获取到有效距离，继续监测...")
        except KeyboardInterrupt:
            print("\n收到 Ctrl+C 中断信号...")
        finally:
            if args.dual:
                arm_controller.disable()
                arm_controller.close()
            else:
                arm_controller.disable()
                arm_controller.close()
            vision_system.close()
        return


if __name__ == "__main__":
    main()
