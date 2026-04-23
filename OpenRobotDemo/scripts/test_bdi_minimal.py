import os
import sys

os.environ["OPENROBOT_FORCE_MOCK"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openrobot_demo.agent.bdi import GoalStatus, IntentStatus
from openrobot_demo.agent.agent import BDIAgent
from openrobot_demo.agent.planner import LLMPlanner
from openrobot_demo.agent.skill_router import SkillRouter
from openrobot_demo.world_model import WorldModel
from openrobot_demo.skills import CameraCapture, ArmStateReader

router = SkillRouter()
camera_skill = CameraCapture(camera_type="usb")
arm_reader = ArmStateReader()
router.register(camera_skill)
router.register(arm_reader)

planner = LLMPlanner(skill_router=router)
agent = BDIAgent(
    planner=planner,
    skill_router=router,
    world_model=WorldModel(),
    max_total_steps=15,
)

# Inject a simple goal tree manually
from openrobot_demo.agent.bdi import Goal

g1_1 = Goal(description="拍摄RGB图像", goal_id="g1_1", required_skills=["camera_capture"], estimated_steps=1)
g1_2 = Goal(description="检测布料3D位置", goal_id="g1_2", required_skills=["vision_3d_estimator"], estimated_steps=1)

g1 = Goal(
    description="感知环境",
    goal_id="g1",
    sub_goals=[g1_1, g1_2],
    estimated_steps=2,
)

top = Goal(description="整体任务", goal_id="top", sub_goals=[g1], estimated_steps=3)
agent.goal_tree = top

print("=== Starting minimal BDI test ===")
summary = agent.execute("test", sensors=[])
print("=== Finished ===")
print("success:", summary["success"])
print("steps:", summary["total_steps"])

for g in [top, g1, g1_1, g1_2]:
    print(f"  {g.goal_id}: status={g.status.value}, is_complete={g.is_complete()}")
