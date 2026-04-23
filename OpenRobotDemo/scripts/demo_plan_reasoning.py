"""Demo: Show step-by-step LLM planning reasoning for a complex fabric task."""

import os
import sys

# Force mock mode to avoid S1_SDK/MuJoCo segfault during import
os.environ["OPENROBOT_FORCE_MOCK"] = "1"

_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)

# Remove S1_SDK path to prevent MuJoCo initialization
_sdk_path = os.path.join(os.path.dirname(_project_root), "YHRG_control", "S1_SDK_V2", "src")
if _sdk_path in sys.path:
    sys.path.remove(_sdk_path)

from dotenv import load_dotenv
load_dotenv(os.path.join(_project_root, ".env"))

from openrobot_demo.agent import LLMPlanner, SkillRouter, TaskDecomposer
from openrobot_demo.skills import (
    ArmMotionExecutor,
    ArmStateReader,
    CameraCapture,
    GraspPointPredictor,
    Vision3DEstimator,
)
from openrobot_demo.skills.coordinate_transform import CoordinateTransformSkill
from openrobot_demo.dual_arm.fabric_skills import FabricManipulationSkill

# Build a minimal skill router for the demo
router = SkillRouter()
router.register(CameraCapture())
router.register(ArmStateReader())
router.register(Vision3DEstimator())
router.register(GraspPointPredictor())
router.register(ArmMotionExecutor())
router.register(FabricManipulationSkill(dual_arm=None, experience_library=None, world_model=None))
router.register(CoordinateTransformSkill())

api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")

planner = LLMPlanner(
    model="qwen-max",
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    skill_router=router,
)

decomposer = TaskDecomposer(
    client=getattr(planner, "_client", None),
    model="qwen-max",
)

INSTRUCTION = (
    "识别摄像头面前桌面上的筒状布料，调用工具分别找到左右机械臂可以操作的布料开口处的空间点位，"
    "移动左右机械臂到对应点位上方3cm，此时机械臂的末端夹爪朝向垂直向下，"
    "而后依靠机械臂上的相机控制机械臂末端夹爪的下方的夹爪插入筒状布料开口处的缝隙，"
    "而后闭合夹爪，确认已经夹起布料后，双机械臂将末端移动到z轴=0.5m且末端夹爪朝向前方的状态再前伸，"
    "前伸0.4米后松开夹爪，双臂恢复初始位置。"
    "等待10秒钟后，识别悬挂在摄像头上方的布料，调用工具找到右机械臂可以操作的布料最下沿的空间点位，"
    "移动右机械臂到该点位，此时机械臂的末端夹爪朝向前方，而后闭合夹爪，水平向后拉0.6米，松开夹爪，而后右机械臂恢复初始位置。"
)

print("=" * 70)
print("指令输入")
print("=" * 70)
print(INSTRUCTION)
print()

# Step 1: Task Decomposition (MLDT-style)
print("=" * 70)
print("Step 1: 任务分解 (MLDT Multi-Level Decomposition)")
print("=" * 70)

goal_tree = decomposer.decompose(INSTRUCTION, router.list_skills())

def print_goal(g, indent=0):
    prefix = "  " * indent
    print(f"{prefix}[{g.goal_id}] {g.description}")
    print(f"{prefix}  preconditions: {g.preconditions}")
    print(f"{prefix}  completion:    {g.completion_criteria}")
    print(f"{prefix}  skills:        {g.required_skills}")
    print(f"{prefix}  est.steps:     {g.estimated_steps}")
    for sg in g.sub_goals:
        print_goal(sg, indent + 1)

print_goal(goal_tree)
print()

# Step 2: Plan generation (closed-loop ReAct simulation)
print("=" * 70)
print("Step 2: 逐步规划思考 (Inner Monologue / ReAct)")
print("=" * 70)

planner.start_task(INSTRUCTION)
feedback_history = []
state_summary = "机械臂处于初始位置，关节角度接近零位。RealSense 相机已连接。"

# Simulate the first few steps to show reasoning
for step_idx in range(8):
    action = planner.next_action(
        state_summary=state_summary,
        feedback_history=feedback_history,
        task_progress=f"总体进度: 执行中，第 {step_idx + 1} 步",
    )

    thought = action.get("thought", "")
    skill = action.get("skill", "")
    args = action.get("args", {})
    act_type = action.get("action", "")

    if act_type == "finish":
        print(f"\n  [Step {step_idx + 1}] thought: {thought}")
        print(f"            action: {act_type}")
        break

    print(f"\n  [Step {step_idx + 1}] thought: {thought}")
    print(f"            skill:  {skill}")
    print(f"            args:   {args}")

    # Simulate feedback
    feedback = {
        "skill": skill,
        "success": True,
        "message": f"{skill} 执行成功",
        "observation": "环境状态已更新",
    }
    feedback_history.append(feedback)

    # Update state summary heuristically
    if skill == "camera_capture":
        state_summary = "已获取 RGB + 深度图像。图像中可见桌面上的筒状布料（白色，直径约8cm）。"
    elif skill == "vision_3d_estimator":
        state_summary = "已估计布料开口处空间点位。左臂目标: [0.25, -0.15, 0.05], 右臂目标: [0.25, 0.15, 0.05]"
    elif skill == "arm_motion_executor":
        state_summary = "机械臂已移动到目标位置附近。末端高度约 0.08m，接近布料上方 3cm。"
    elif skill == "fabric_manipulation":
        state_summary = "夹爪已插入布料开口缝隙并闭合。力反馈正常，确认已夹持布料。"
    else:
        state_summary = "执行中，等待下一步指令。"

print("\n" + "=" * 70)
print("演示结束")
print("=" * 70)
