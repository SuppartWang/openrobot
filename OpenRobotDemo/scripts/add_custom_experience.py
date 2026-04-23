"""
Add a custom experience for dual-arm grasping cylindrical fabric
and inserting it onto a rectangular support plate.
"""

import os
import sys

_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)

from openrobot_demo.experience.library import ExperienceLibrary
from openrobot_demo.experience.schema import Experience, GripperConfig, DualArmPattern


def add_dual_arm_fabric_to_plate_experience():
    lib = ExperienceLibrary()

    # Experience: dual-arm grasp cylindrical fabric and insert onto rectangular plate
    exp = Experience(
        task_intent="将操作区域的筒形布料夹起来套在前方的长方形支撑板上",
        target_object_type="长方形支撑板",
        target_object_tags=["rigid", "rectangular", "plate", "support"],
        gripper_config=GripperConfig.PARALLEL_2_FINGER,
        robot_dof=7,
        arm_count=2,
        action_type="insert",
        dual_arm_pattern=DualArmPattern.COMPLEMENTARY,
        pre_contact_offset=[0.0, 0.0, 0.03],
        approach_angle_deg=90.0,
        gripper_aperture_m=0.0,
        contact_force_threshold_n=1.5,
        dual_arm_pinch_distance_m=0.10,
        dual_arm_sync_tolerance_m=0.001,
        left_grasp_relative=[0.0, -0.05, 0.0],
        right_grasp_relative=[0.0, 0.05, 0.0],
        trajectory_type="cartesian",
        waypoint_count=60,
        step_time_s=0.02,
        max_velocity_m_s=0.04,
        compliance_stiffness=100.0,
        source="human_demo",
        success=True,
        human_feedback="""针对长方形支撑板的抓取套入经验：
1. 双臂需从筒形布料两侧水平捏合，捏合间距略大于布料直径(约10cm)，确保夹取稳固。
2. 提起时保持双臂高度同步，提升速度适中(0.04m/s)。
3. 套入长方形支撑板前，先让布料中心对准支撑板中心，再缓慢下降。
4. 下降过程中若感受到阻力突增，应暂停0.5秒再继续，避免布料褶皱卡死。
5. 套入深度建议为支撑板高度+8mm，确保稳固但不过度下压。""",
    )

    exp_id = lib.add(exp)
    print(f"[AddExperience] Added experience {exp_id}")
    print(f"  Task: {exp.task_intent}")
    print(f"  Action: {exp.action_type}")
    print(f"  Target: {exp.target_object_type}")

    # Also add a pinch experience for the same task
    exp_pinch = Experience(
        task_intent="将操作区域的筒形布料夹起来套在前方的长方形支撑板上",
        target_object_type="筒形布料",
        target_object_tags=["soft", "cylindrical", "flexible", "textile"],
        gripper_config=GripperConfig.PARALLEL_2_FINGER,
        robot_dof=7,
        arm_count=2,
        action_type="pinch",
        dual_arm_pattern=DualArmPattern.COMPLEMENTARY,
        pre_contact_offset=[0.0, 0.0, 0.03],
        approach_angle_deg=90.0,
        gripper_aperture_m=0.0,
        contact_force_threshold_n=0.6,
        dual_arm_pinch_distance_m=0.10,
        dual_arm_sync_tolerance_m=0.001,
        left_grasp_relative=[0.0, -0.05, 0.0],
        right_grasp_relative=[0.0, 0.05, 0.0],
        trajectory_type="cartesian",
        waypoint_count=60,
        step_time_s=0.02,
        max_velocity_m_s=0.04,
        compliance_stiffness=80.0,
        source="human_demo",
        success=True,
        human_feedback="""抓取筒形布料时，双臂需同时接触布料边缘后再闭合夹爪。
建议先以较低速度接近(pre_contact_offset=3cm)，确认双侧均接触后再闭合。
捏合间距设为10cm，适合直径8-10cm的筒形布料。""",
    )

    exp_id2 = lib.add(exp_pinch)
    print(f"[AddExperience] Added experience {exp_id2}")
    print(f"  Task: {exp_pinch.task_intent}")
    print(f"  Action: {exp_pinch.action_type}")
    print(f"  Target: {exp_pinch.target_object_type}")

    # List all experiences
    all_exps = lib.list_all()
    print(f"\n[AddExperience] Total experiences in library: {len(all_exps)}")
    lib.close()


if __name__ == "__main__":
    add_dual_arm_fabric_to_plate_experience()
