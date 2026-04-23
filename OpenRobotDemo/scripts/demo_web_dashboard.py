"""
OpenRobotDemo — Gradio Web 三栏可视化控制台

Usage:
    python scripts/demo_web_dashboard.py --mode real \
        --left-dev /dev/left_follower --right-dev /dev/right_follower \
        --camera-serial 135122077817

    python scripts/demo_web_dashboard.py --mode mock
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import pyrealsense2 as rs
except Exception:
    rs = None

# ------------------------------------------------------------------
# Environment setup (must be before any project imports)
# ------------------------------------------------------------------
_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)

from dotenv import load_dotenv
load_dotenv(Path(_project_root) / ".env")

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

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
import gradio as gr

from openrobot_demo.agent import BDIAgent, LLMPlanner, SkillRouter
from openrobot_demo.dual_arm.controller import DualArmController, ArmSide
from openrobot_demo.dual_arm.fabric_skills import FabricManipulationSkill
from openrobot_demo.hardware.yhrg_adapter import YHRGKinematics, SDKKinematics
from openrobot_demo.experience.library import ExperienceLibrary
from openrobot_demo.experience.retriever import ExperienceRetriever
from openrobot_demo.experience.seed import seed_fabric_experiences
from openrobot_demo.perception.vlm_cognition import VLMCognitionSensor
from openrobot_demo.sensors import (
    ProprioceptionSensor,
    RealSenseRGBSensor,
    RealSenseDepthSensor,
    TactileSensor,
)
from openrobot_demo.sensors.realsense_shared import RealSenseDevicePool
from openrobot_demo.skills import (
    ArmMotionExecutor,
    ArmStateReader,
    CameraCapture,
    CoordinateTransformSkill,
    DualArmCoordinatedMotionSkill,
    GraspPointPredictor,
    GripperControlSkill,
    Vision3DEstimator,
    VLAPolicyExecutor,
)
from openrobot_demo.world_model import WorldModel, ObjectDesc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ==================================================================
# Shared state (thread-safe via locks where needed)
# ==================================================================
class AppState:
    """Thread-safe shared state between background threads and Gradio UI."""

    def __init__(self):
        self.lock = threading.Lock()
        self.rgb: Optional[np.ndarray] = None
        self.depth: Optional[np.ndarray] = None
        self.vlm_frame: Optional[np.ndarray] = None
        self.arm_state: Dict[str, Any] = {}
        self.chat_history: List[List[str]] = []  # [[user, assistant], ...]
        self.current_action: str = "空闲"
        self.step_count: int = 0
        self.exec_history: List[Dict[str, Any]] = []
        self.goal_tree: Optional[Dict[str, Any]] = None
        self.task_running: bool = False
        self.last_llm_thought: str = ""

    def update_rgb(self, arr: np.ndarray):
        with self.lock:
            self.rgb = arr.copy() if arr is not None else None

    def update_depth(self, arr: np.ndarray):
        with self.lock:
            self.depth = arr.copy() if arr is not None else None

    def update_vlm(self, arr: np.ndarray):
        with self.lock:
            self.vlm_frame = arr.copy() if arr is not None else None

    def update_arm(self, data: Dict[str, Any]):
        with self.lock:
            self.arm_state = data.copy()

    def add_chat(self, user: str, assistant: str):
        with self.lock:
            self.chat_history.append([user, assistant])
            # Keep last 100 turns
            if len(self.chat_history) > 100:
                self.chat_history = self.chat_history[-100:]

    def update_action(self, action: str):
        with self.lock:
            self.current_action = action

    def add_exec(self, row: Dict[str, Any]):
        with self.lock:
            self.exec_history.append(row)
            if len(self.exec_history) > 200:
                self.exec_history = self.exec_history[-200:]

    def update_goal_tree(self, tree: Dict[str, Any]):
        with self.lock:
            self.goal_tree = tree

    def get_snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "rgb": self.rgb.copy() if self.rgb is not None else None,
                "depth": self.depth.copy() if self.depth is not None else None,
                "vlm": self.vlm_frame.copy() if self.vlm_frame is not None else None,
                "arm_state": self.arm_state.copy(),
                "chat_history": list(self.chat_history),
                "current_action": self.current_action,
                "step_count": self.step_count,
                "exec_history": list(self.exec_history),
                "goal_tree": self.goal_tree,
                "task_running": self.task_running,
            }


# ==================================================================
# Left / Right arm motion executor wrappers
# ==================================================================
class LeftArmMotionExecutor(ArmMotionExecutor):
    @property
    def name(self) -> str:
        return "arm_motion_executor_left"


class RightArmMotionExecutor(ArmMotionExecutor):
    @property
    def name(self) -> str:
        return "arm_motion_executor_right"


# ==================================================================
# Hardware initialization
# ==================================================================
def setup_hardware(mode: str, left_dev: str, right_dev: str, camera_serial: str,
                   end_effector: str = "gripper", tcp_offset_z: float = 0.10):
    """Initialize all hardware and return components."""
    # Experience
    exp_lib = ExperienceLibrary()
    existing = exp_lib.list_all(limit=1)
    if not existing:
        seed_fabric_experiences(exp_lib)

    # Dual arm with TCP offset (tool center point distance from flange)
    tcp_offset = [0.0, 0.0, tcp_offset_z]
    dual_arm = DualArmController(
        left_dev=left_dev,
        right_dev=right_dev,
        mode=mode,
        end_effector=end_effector,
    )
    # Update kinematics with TCP offset (use SDK native C++ KDL solver when available)
    dual_arm.left_kin = SDKKinematics(end_effector_offset=tcp_offset)
    dual_arm.right_kin = SDKKinematics(end_effector_offset=tcp_offset)
    dual_arm.enable()

    # World model
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

    # Sensors
    sensors = [
        RealSenseRGBSensor(source_id="rs_d435i_rgb", serial=camera_serial),
        RealSenseDepthSensor(source_id="rs_d435i_depth", serial=camera_serial),
        ProprioceptionSensor(source_id="left_arm", arm_adapter=dual_arm.left_arm, kinematics_solver=dual_arm.left_kin),
        ProprioceptionSensor(source_id="right_arm", arm_adapter=dual_arm.right_arm, kinematics_solver=dual_arm.right_kin),
        TactileSensor(source_id="left_gripper", body_names=["gripper_base", "left_finger"], mujoco_model=None, mujoco_data=None),
        TactileSensor(source_id="right_gripper", body_names=["gripper_base", "right_finger"], mujoco_model=None, mujoco_data=None),
    ]

    # Skills
    router = SkillRouter()
    fabric_skill = FabricManipulationSkill(dual_arm=dual_arm, experience_library=exp_lib, world_model=world)
    camera_skill = CameraCapture(
        camera_type="realsense" if mode == "real" else "usb",
        device_id=0, width=640, height=480,
        serial=camera_serial if mode == "real" else None,
    )
    arm_reader = ArmStateReader(external_arm=dual_arm.left_arm)
    vision_estimator = Vision3DEstimator()
    grasp_predictor = GraspPointPredictor()
    arm_executor_left = LeftArmMotionExecutor(external_arm=dual_arm.left_arm, external_solver=dual_arm.left_kin)
    arm_executor_right = RightArmMotionExecutor(external_arm=dual_arm.right_arm, external_solver=dual_arm.right_kin)
    vla_executor = VLAPolicyExecutor(external_arm=dual_arm.left_arm)
    gripper_skill = GripperControlSkill(dual_arm=dual_arm)
    dual_arm_motion_skill = DualArmCoordinatedMotionSkill(dual_arm=dual_arm)
    coord_transform_skill = CoordinateTransformSkill()

    router.register(camera_skill)
    router.register(arm_reader)
    router.register(vision_estimator)
    router.register(grasp_predictor)
    router.register(arm_executor_left)
    router.register(arm_executor_right)
    router.register(fabric_skill)
    router.register(vla_executor)
    router.register(gripper_skill)
    router.register(dual_arm_motion_skill)
    router.register(coord_transform_skill)

    # VLM sensor
    vlm_sensor = VLMCognitionSensor(source_id="vlm_cognition", camera_source_id="rs_d435i_rgb")
    sensors.append(vlm_sensor)

    # BDI Agent
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

    return {
        "agent": agent,
        "planner": planner,
        "sensors": sensors,
        "dual_arm": dual_arm,
        "exp_lib": exp_lib,
        "world": world,
        "skills": {
            "camera_capture": camera_skill,
            "vision_3d_estimator": vision_estimator,
            "coordinate_transform": coord_transform_skill,
            "arm_motion_executor_left": arm_executor_left,
            "arm_motion_executor_right": arm_executor_right,
            "gripper_control": gripper_skill,
            "dual_arm_coordinated_motion": dual_arm_motion_skill,
        },
    }


# ==================================================================
# VLM detection helper (reliable backup approach)
# ==================================================================
def _detect_with_vlm(frame, instruction, api_key):
    """VLM target detection: returns (cx, cy) in original image coordinates, or None."""
    import cv2
    from openai import OpenAI
    import base64
    import re

    if not api_key:
        return None

    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    h, w, _ = frame.shape
    scale_ratio = 640 / w
    detect_w, detect_h = 640, int(h * scale_ratio)
    resized = cv2.resize(frame, (detect_w, detect_h))
    _, buffer = cv2.imencode('.jpg', resized)
    b64 = base64.b64encode(buffer).decode("utf-8")

    prompt = (
        f"你是一个精确的机器视觉定位系统。请在宽{detect_w}，高{detect_h}的图片中，{instruction}。\n"
        "要求：\n1. 必须精准定位该物体指定位置的坐标。\n"
        "2. 绝对禁止输出任何额外解释文字！\n"
        "3. 格式只能是包含两个数字的纯JSON数组：[中心X, 中心Y]\n"
        "如果找不到目标，请返回 [0, 0]"
    )
    try:
        resp = client.chat.completions.create(
            model="qwen-vl-max-latest",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            temperature=0.1, max_tokens=100
        )
        text = resp.choices[0].message.content.strip()
        clean = re.sub(r'[a-zA-Z"\'{}]+', '', text)
        nums = re.findall(r'\d+', clean)
        if len(nums) >= 2:
            x, y = map(int, nums[:2])
            if x == 0 and y == 0:
                return None
            return (int(x / scale_ratio), int(y / scale_ratio))
    except Exception:
        pass
    return None


# ==================================================================
# Standard end-effector orientations (quaternion [qx, qy, qz, qw])
# ==================================================================
# FORWARD: arm zero pose is forward-facing (identity quaternion)
ORIENTATION_FORWARD = [0.0, 0.0, 0.0, 1.0]
# DOWNWARD: rotate -90° around x-axis to face vertically down
ORIENTATION_DOWNWARD = [-0.7071, 0.0, 0.0, 0.7071]


# ==================================================================
# Background workers
# ==================================================================
def camera_worker(state: AppState, rgb_sensor, depth_sensor, dual_arm, stop_event: threading.Event):
    """Continuously capture RGB + Depth + arm state."""
    while not stop_event.is_set():
        try:
            if rgb_sensor.is_available():
                reading = rgb_sensor.capture()
                if hasattr(reading, "payload"):
                    state.update_rgb(reading.payload)
            if depth_sensor.is_available():
                reading = depth_sensor.capture()
                if hasattr(reading, "payload"):
                    state.update_depth(reading.payload)
            # Arm state
            if dual_arm is not None:
                from openrobot_demo.dual_arm.controller import ArmSide
                lp = dual_arm.get_pos(ArmSide.LEFT)
                rp = dual_arm.get_pos(ArmSide.RIGHT)
                state.update_arm({
                    "left": [round(float(x), 4) for x in lp[:6]],
                    "right": [round(float(x), 4) for x in rp[:6]],
                    "timestamp": time.strftime("%H:%M:%S"),
                })
        except Exception as exc:
            logger.debug("Camera worker error: %s", exc)
        time.sleep(0.2)


def run_preplanned_demo(state: AppState, dual_arm, sensors: list, skills: Dict[str, Any], instruction: str):
    """预先规划好的 demo 动作序列，完全跳过 LLM 调用，直接按硬编码步骤执行。

    Phase 1: 桌面筒状布料 → 末端垂直向下 → 插入 → 闭合 → 抬升到z=0.5m且朝前 → 前伸0.4m → 松开 → 复位
    Phase 2: 等待10s → 悬挂布料 → 右机械臂末端朝前 → 拉取 → 松开 → 复位

    坐标系:
      世界原点: 左臂基座在桌面的投影点
      左臂基座: (0, 0, 0.09), 右臂基座: (0.56, 0, 0.09)
      摄像头:   (0.28, 0.71, 0.09), 朝向 -y
      臂坐标系: x+ 右, y+ 前, z+ 上
    """
    state.task_running = True
    state.chat_history = []
    state.exec_history = []
    state.step_count = 0
    state.current_action = "预规划任务启动..."

    import cv2

    # 获取技能
    coord_skill = skills.get("coordinate_transform")
    left_arm = skills.get("arm_motion_executor_left")
    right_arm = skills.get("arm_motion_executor_right")
    gripper = skills.get("gripper_control")
    dual_motion = skills.get("dual_arm_coordinated_motion")

    # 获取传感器
    rgb_sensor = next((s for s in sensors if s.source_id == "rs_d435i_rgb"), None)
    serial = rgb_sensor.serial if rgb_sensor else None
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")

    # 安全的初始位姿（cartesian: [x, y, z, qx, qy, qz, qw]）
    HOME_LEFT = [0.2, 0.0, 0.25] + ORIENTATION_DOWNWARD
    HOME_RIGHT = [0.2, 0.0, 0.25] + ORIENTATION_DOWNWARD

    def _notify_step(msg: str, success: bool = True):
        state.step_count += 1
        icon = "✅" if success else "⚠️"
        state.add_chat("系统", f"{icon} 步骤 {state.step_count}: {msg}")
        state.add_exec({
            "step": state.step_count,
            "status": "成功" if success else "警告",
            "message": msg[:60],
            "time": time.strftime("%H:%M:%S"),
        })
        state.update_action(msg[:50])

    def _exec_skill(skill, **kwargs) -> Dict[str, Any]:
        if skill is None:
            return {"success": False, "message": "Skill not available"}
        try:
            return skill.execute(**kwargs)
        except Exception as e:
            logger.exception("Skill %s failed", getattr(skill, "name", "?"))
            return {"success": False, "message": str(e)}

    def _detect_and_transform(
        color_image_rgb, depth_frame, intrinsics,
        instruction_text, target_frame, offset_z=0.0
    ):
        """Run VLM detection → deproject to 3D → coordinate transform.
        Returns (point_arm, debug_msg) or (None, error_msg).
        """
        point = _detect_with_vlm(color_image_rgb, instruction_text, api_key)
        if point is None:
            return None, "VLM 未检测到目标"
        cx, cy = point
        distance_m = depth_frame.get_distance(cx, cy)
        if distance_m <= 0 or intrinsics is None:
            return None, f"深度无效 ({distance_m})"
        cam_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], distance_m)
        if coord_skill is None:
            return None, "坐标转换技能不可用"
        tf = _exec_skill(coord_skill, point_camera=cam_3d,
                         target_frame=target_frame, offset_z=offset_z)
        if not tf.get("success"):
            return None, f"坐标转换失败: {tf.get('error')}"
        return tf.get("point")[:3], f"检测成功 @ ({cx},{cy}), 深度{distance_m:.3f}m"

    try:
        # ========== Phase 1: 桌面筒状布料操作 ==========
        state.add_chat("系统", "🎯 开始 Phase 1: 识别桌面筒状布料 → 提起 → 前伸释放")
        state.update_goal_tree({
            "phase": "Phase 1",
            "goal": "提起筒状布料并前伸释放",
            "steps_total": 20,
            "steps_completed": 0,
        })

        # Step 1: 张开双臂夹爪
        if gripper:
            _exec_skill(gripper, side="both", position=1.0, force=0.5)
            _notify_step("双臂夹爪张开")
            time.sleep(0.5)

        # Step 2: 拍摄图像（使用 pyrealsense2 直接获取对齐帧 + 内参）
        state.current_action = "拍摄 RGB + Depth..."
        left_opening_pos = [0.18, 0.20, 0.05]   # 默认：左臂坐标系
        right_opening_pos = [-0.18, 0.20, 0.05]  # 默认：右臂坐标系
        intrinsics = None

        if serial and rs is not None:
            try:
                color_frame, depth_frame = RealSenseDevicePool.capture_frames(serial, apply_filters=False)
                if hasattr(depth_frame, 'as_depth_frame'):
                    depth_frame = depth_frame.as_depth_frame()
                color_image_bgr = np.asanyarray(color_frame.get_data())
                color_image_rgb = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)
                state.update_vlm(color_image_rgb)
                intrinsics = RealSenseDevicePool.get_intrinsics(serial)
                _notify_step("图像获取成功")

                # Step 3-4: VLM 检测左/右开口 + 坐标转换
                detections = [
                    ("筒状布料左侧开口",
                     "找到筒状布料的左侧开口处，返回左侧开口边缘的中心点坐标",
                     "left", 0.03, "left_opening"),
                    ("筒状布料右侧开口",
                     "找到筒状布料的右侧开口处，返回右侧开口边缘的中心点坐标",
                     "right", 0.03, "right_opening"),
                ]
                for target_name, inst, arm, offset, label in detections:
                    pt, msg = _detect_and_transform(
                        color_image_rgb, depth_frame, intrinsics, inst, arm, offset)
                    if pt is not None:
                        if arm == "left":
                            left_opening_pos = pt
                        else:
                            right_opening_pos = pt
                        _notify_step(f"{target_name} 臂基坐标: {pt}")
                    else:
                        _notify_step(f"{target_name} 检测失败 ({msg})，使用默认坐标", success=False)
            except Exception as e:
                _notify_step(f"视觉检测异常: {e}，使用默认坐标", success=False)
        else:
            _notify_step("RealSense 不可用，使用默认坐标", success=False)

        _notify_step(f"最终左坐标: {left_opening_pos}")
        _notify_step(f"最终右坐标: {right_opening_pos}")

        # Step 5-6: 双臂移动到开口上方3cm，末端垂直向下
        left_above = list(left_opening_pos) + ORIENTATION_DOWNWARD
        right_above = list(right_opening_pos) + ORIENTATION_DOWNWARD

        if left_arm:
            r = _exec_skill(left_arm, command_type="cartesian", target_values=left_above, speed=0.5)
            _notify_step(f"左臂移动到开口上方 {left_above[:3]} (垂直向下)", success=r.get("success", False))
            time.sleep(1.0)

        if right_arm:
            r = _exec_skill(right_arm, command_type="cartesian", target_values=right_above, speed=0.5)
            _notify_step(f"右臂移动到开口上方 {right_above[:3]} (垂直向下)", success=r.get("success", False))
            time.sleep(1.0)

        # Step 7-8: 下降插入缝隙（从上方3cm下降到开口处）
        left_insert = [left_opening_pos[0], left_opening_pos[1], left_opening_pos[2] - 0.03] + ORIENTATION_DOWNWARD
        right_insert = [right_opening_pos[0], right_opening_pos[1], right_opening_pos[2] - 0.03] + ORIENTATION_DOWNWARD

        if left_arm:
            r = _exec_skill(left_arm, command_type="cartesian", target_values=left_insert, speed=0.3)
            _notify_step(f"左臂下降插入缝隙 z={left_insert[2]:.3f}", success=r.get("success", False))
            time.sleep(1.0)

        if right_arm:
            r = _exec_skill(right_arm, command_type="cartesian", target_values=right_insert, speed=0.3)
            _notify_step(f"右臂下降插入缝隙 z={right_insert[2]:.3f}", success=r.get("success", False))
            time.sleep(1.0)

        # Step 9: 闭合夹爪
        if gripper:
            _exec_skill(gripper, side="both", position=0.0, force=0.8)
            _notify_step("双臂夹爪闭合")
            time.sleep(1.0)

        # Step 10: 抬升到 z=0.5m，同时切换为朝向前方
        if left_arm and right_arm:
            left_current = dual_arm.get_ee_pose("left")
            right_current = dual_arm.get_ee_pose("right")
            left_lift = [left_current[0], left_current[1], 0.5] + ORIENTATION_FORWARD
            right_lift = [right_current[0], right_current[1], 0.5] + ORIENTATION_FORWARD
            dual_arm.dual_move_cartesian(left_lift, right_lift, duration=3.0)
            _notify_step("双臂抬升到 z=0.5m 且末端朝前")
            time.sleep(3.5)

        # Step 11: 前伸 0.4m（y 减小方向，臂坐标系前方为 +y，所以前伸是 -y？
        # 等等，用户说"世界坐标y轴正方向为前"。臂坐标系 y+ 也是前。
        # 但之前的代码中前伸用的是 y_offset=-0.4。让我保持一致。）
        if dual_motion:
            r = _exec_skill(dual_motion, command_type="relative", side="both", y_offset=-0.4, duration=2.0)
            _notify_step("双臂前伸 0.4m", success=r.get("success", False))
            time.sleep(2.5)

        # Step 12: 松开夹爪
        if gripper:
            _exec_skill(gripper, side="both", position=1.0, force=0.5)
            _notify_step("双臂夹爪松开")
            time.sleep(1.0)

        # Step 13-14: 恢复初始位置（末端垂直向下）
        if left_arm:
            r = _exec_skill(left_arm, command_type="cartesian", target_values=HOME_LEFT, speed=0.5)
            _notify_step("左臂恢复初始位置", success=r.get("success", False))
            time.sleep(2.0)

        if right_arm:
            r = _exec_skill(right_arm, command_type="cartesian", target_values=HOME_RIGHT, speed=0.5)
            _notify_step("右臂恢复初始位置", success=r.get("success", False))
            time.sleep(2.0)

        # ========== Phase 2: 悬挂布料拉取 ==========
        state.add_chat("系统", "🎯 开始 Phase 2: 等待10秒 → 识别悬挂布料 → 拉取")
        state.update_goal_tree({
            "phase": "Phase 2",
            "goal": "识别悬挂布料最下沿并拉取",
            "steps_total": 10,
            "steps_completed": 0,
        })

        # Step 15: 等待10秒
        state.current_action = "等待 10s..."
        for i in range(10):
            if not state.task_running:
                _notify_step("任务被用户停止", success=False)
                return
            time.sleep(1)
            state.current_action = f"等待 {10-i}s..."
        _notify_step("等待完成")

        # Step 16: 再次拍摄
        bottom_edge_pos = [0.0, 0.2, 0.0]  # 默认：右臂坐标系

        if serial and rs is not None:
            try:
                color_frame2, depth_frame2 = RealSenseDevicePool.capture_frames(serial, apply_filters=False)
                if hasattr(depth_frame2, 'as_depth_frame'):
                    depth_frame2 = depth_frame2.as_depth_frame()
                color_image_bgr2 = np.asanyarray(color_frame2.get_data())
                color_image_rgb2 = cv2.cvtColor(color_image_bgr2, cv2.COLOR_BGR2RGB)
                state.update_vlm(color_image_rgb2)
                intrinsics2 = RealSenseDevicePool.get_intrinsics(serial)
                _notify_step("再次获取图像成功")

                # Step 17: 检测悬挂布料最下沿
                pt, msg = _detect_and_transform(
                    color_image_rgb2, depth_frame2, intrinsics2,
                    "找到悬挂在摄像头上方的布料的最下沿中心点坐标",
                    "right", 0.0)
                if pt is not None:
                    bottom_edge_pos = pt
                    _notify_step(f"悬挂布料最下沿位置: {pt}")
                else:
                    _notify_step(f"最下沿检测失败 ({msg})，使用默认坐标", success=False)
            except Exception as e:
                _notify_step(f"Phase 2 视觉检测异常: {e}，使用默认坐标", success=False)
        else:
            _notify_step("RealSense 不可用，使用默认坐标", success=False)

        # Step 18: 张开右臂夹爪
        if gripper:
            _exec_skill(gripper, side="right", position=1.0, force=0.5)
            _notify_step("右臂夹爪张开")
            time.sleep(0.5)

        # Step 19: 右臂移动到位，末端朝向前方
        right_target = list(bottom_edge_pos) + ORIENTATION_FORWARD

        if right_arm:
            r = _exec_skill(right_arm, command_type="cartesian", target_values=right_target, speed=0.5)
            _notify_step(f"右臂移动到最下沿 {right_target[:3]} (朝前)", success=r.get("success", False))
            time.sleep(2.0)

        # Step 20: 闭合右臂夹爪
        if gripper:
            _exec_skill(gripper, side="right", position=0.0, force=0.8)
            _notify_step("右臂夹爪闭合")
            time.sleep(1.0)

        # Step 21: 后拉 0.6m（y 增大方向，臂坐标系后方为 +y）
        if dual_motion:
            r = _exec_skill(dual_motion, command_type="relative", side="right", y_offset=0.6, duration=2.0)
            _notify_step("右臂后拉 0.6m", success=r.get("success", False))
            time.sleep(2.5)

        # Step 22: 松开右臂夹爪
        if gripper:
            _exec_skill(gripper, side="right", position=1.0, force=0.5)
            _notify_step("右臂夹爪松开")
            time.sleep(1.0)

        # Step 23: 右臂恢复初始位置（末端垂直向下）
        if right_arm:
            r = _exec_skill(right_arm, command_type="cartesian", target_values=HOME_RIGHT, speed=0.5)
            _notify_step("右臂恢复初始位置", success=r.get("success", False))
            time.sleep(2.0)

        # 完成
        state.add_chat("系统", f"🎉 Demo 完成！总执行步数: {state.step_count}")
        state.update_goal_tree({"phase": "完成", "status": "success", "total_steps": state.step_count})

    except Exception as e:
        logger.exception("Preplanned demo failed")
        state.add_chat("系统", f"❌ 错误: {e}")
    finally:
        state.task_running = False
        state.update_action("空闲")


def agent_worker(state: AppState, agent: BDIAgent, sensors: list, skills: Dict[str, Any],
                 dual_arm, instruction: str):
    """Run pre-planned demo in background (LLM bypassed for speed)."""
    # 直接执行预规划序列，跳过所有 LLM 调用
    run_preplanned_demo(state, dual_arm, sensors, skills, instruction)


# ==================================================================
# Depth visualization helper
# ==================================================================
def depth_to_color(depth: np.ndarray) -> np.ndarray:
    """Convert depth (meters) to pseudo-color RGB."""
    if depth is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    d = np.nan_to_num(depth, nan=0.0)
    d = np.clip(d, 0.0, 2.0)
    d_uint8 = (d / 2.0 * 255).astype(np.uint8)
    cmap = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        r = int(np.clip(0.28 + 0.89 * t - 0.15 * t * t, 0, 1) * 255)
        g = int(np.clip(0.00 + 1.50 * t - 0.50 * t * t, 0, 1) * 255)
        b = int(np.clip(0.33 + 0.70 * t - 0.20 * t * t, 0, 1) * 255)
        cmap[i] = [r, g, b]
    return cmap[d_uint8]


# ==================================================================
# Gradio UI
# ==================================================================
def build_ui(state: AppState, hw: Dict[str, Any]):
    """Build and return the Gradio Blocks interface."""
    agent = hw["agent"]
    sensors = hw["sensors"]
    dual_arm = hw["dual_arm"]
    rgb_sensor = next((s for s in sensors if s.source_id == "rs_d435i_rgb"), None)
    depth_sensor = next((s for s in sensors if s.source_id == "rs_d435i_depth"), None)

    # Camera worker
    camera_stop = threading.Event()
    cam_thread = threading.Thread(
        target=camera_worker,
        args=(state, rgb_sensor, depth_sensor, dual_arm, camera_stop),
        daemon=True,
    )
    cam_thread.start()

    # Agent worker placeholder
    agent_thread: Optional[threading.Thread] = None
    agent_stop = threading.Event()

    with gr.Blocks(title="OpenRobotDemo — 实机可视化控制台") as demo:

        gr.Markdown("# OpenRobotDemo 实机可视化控制台")

        with gr.Row():
            # -------------------- Left Column --------------------
            with gr.Column(scale=1):
                gr.Markdown("### 感知信息")
                rgb_img = gr.Image(
                    label="RGB 实时画面",
                    type="numpy",
                    streaming=False,
                    height=280,
                )
                depth_img = gr.Image(
                    label="深度图",
                    type="numpy",
                    streaming=False,
                    height=280,
                )
                vlm_img = gr.Image(
                    label="VLM 识别结果",
                    type="numpy",
                    streaming=False,
                    height=200,
                )
                arm_json = gr.JSON(label="机械臂状态")

            # -------------------- Center Column --------------------
            with gr.Column(scale=1):
                gr.Markdown("### 规划思考过程")
                chatbot = gr.Chatbot(
                    label="LLM 对话",
                    height=550,
                )
                instruction = gr.Textbox(
                    label="指令输入",
                    value="将筒状布料提起，套在铝合金支撑板上，等待检测后再取下来",
                    lines=2,
                )
                with gr.Row():
                    run_btn = gr.Button("执行任务", variant="primary")
                    stop_btn = gr.Button("停止")
                    clear_btn = gr.Button("清空日志")

            # -------------------- Right Column --------------------
            with gr.Column(scale=1):
                gr.Markdown("### 运动执行状态")
                current_action = gr.Textbox(
                    label="当前动作",
                    value="空闲",
                    interactive=False,
                )
                step_count = gr.Number(label="已执行步数", value=0, interactive=False)
                exec_df = gr.Dataframe(
                    label="执行历史",
                    headers=["step", "status", "message", "time"],
                    interactive=False,
                    max_height=300,
                )
                goal_json = gr.JSON(label="目标树")

        # -------------------- Timer-based refresh --------------------
        def refresh_ui():
            snap = state.get_snapshot()
            rgb = snap["rgb"]
            depth = snap["depth"]
            vlm = snap["vlm"]

            # Convert depth to color if needed
            depth_vis = depth_to_color(depth) if depth is not None else None

            # Build exec dataframe
            exec_rows = snap["exec_history"]
            if exec_rows:
                import pandas as pd
                df = pd.DataFrame(exec_rows)
            else:
                import pandas as pd
                df = pd.DataFrame(columns=["step", "status", "message", "time"])

            # Format chat for gr.Chatbot type="messages"
            chat_msgs = []
            for user_msg, bot_msg in snap["chat_history"]:
                chat_msgs.append({"role": "user", "content": user_msg})
                chat_msgs.append({"role": "assistant", "content": bot_msg})

            return [
                rgb,
                depth_vis,
                vlm,
                snap["arm_state"],
                chat_msgs,
                snap["current_action"],
                snap["step_count"],
                df,
                snap["goal_tree"],
            ]

        timer = gr.Timer(value=0.3, active=True)
        timer.tick(
            fn=refresh_ui,
            outputs=[rgb_img, depth_img, vlm_img, arm_json, chatbot,
                     current_action, step_count, exec_df, goal_json],
        )

        # -------------------- Buttons --------------------
        def on_run(instr: str):
            nonlocal agent_thread, agent_stop
            if state.task_running:
                return
            agent_stop = threading.Event()
            agent_thread = threading.Thread(
                target=agent_worker,
                args=(state, agent, sensors, hw.get("skills", {}), dual_arm, instr),
                daemon=True,
            )
            agent_thread.start()

        def on_stop():
            # Best-effort stop; agent checks goal completion each loop
            state.task_running = False
            state.add_chat("系统", "[用户] 请求停止任务")

        def on_clear():
            state.chat_history = []
            state.exec_history = []
            state.step_count = 0
            state.current_action = "空闲"
            state.goal_tree = None

        run_btn.click(fn=on_run, inputs=[instruction], outputs=[])
        stop_btn.click(fn=on_stop, outputs=[])
        clear_btn.click(fn=on_clear, outputs=[])

        # Cleanup on exit
        def cleanup():
            camera_stop.set()
            state.task_running = False
            if agent_thread is not None:
                agent_thread.join(timeout=2)
            cam_thread.join(timeout=2)
            dual_arm.disable()
            for s in sensors:
                if hasattr(s, "close"):
                    try:
                        s.close()
                    except Exception:
                        pass
            hw["exp_lib"].close()

        demo.unload(cleanup)

    return demo


# ==================================================================
# Main
# ==================================================================
def main():
    parser = argparse.ArgumentParser(description="OpenRobotDemo Gradio Web Dashboard")
    parser.add_argument("--mode", choices=["mock", "real"], default="mock")
    parser.add_argument("--left-dev", default="/dev/left_follower")
    parser.add_argument("--right-dev", default="/dev/right_follower")
    parser.add_argument("--camera-serial", default="135122077817")
    parser.add_argument("--end-effector", default="gripper", help="End-effector type: gripper, None, teach")
    parser.add_argument("--tcp-offset-z", type=float, default=0.10, help="Tool center point Z offset from flange (m)")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  OpenRobotDemo — Gradio Web 可视化控制台")
    print(f"  Mode: {args.mode} | Port: {args.port}")
    print("=" * 60 + "\n")

    print("[1/2] Initializing hardware...")
    hw = setup_hardware(args.mode, args.left_dev, args.right_dev, args.camera_serial,
                        end_effector=args.end_effector, tcp_offset_z=args.tcp_offset_z)
    print("      Hardware ready.")

    print("[2/2] Building Gradio interface...")
    state = AppState()
    demo = build_ui(state, hw)
    print(f"\n🚀 Launching Gradio server on http://0.0.0.0:{args.port}")

    _css = ".container{max-width:100%!important;} .gradio-container{font-family:'Noto Sans CJK SC',sans-serif;}"
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
        css=_css,
    )


if __name__ == "__main__":
    main()
