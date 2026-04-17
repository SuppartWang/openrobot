"""Pre-seed experiences for the cylindrical fabric manipulation demo.

These are human-summarized experiences that can be refined by autonomous
execution feedback after the demo.
"""

from openrobot_demo.experience.library import ExperienceLibrary
from openrobot_demo.experience.schema import Experience, GripperConfig, DualArmPattern


def seed_fabric_experiences(library: ExperienceLibrary):
    """Add pre-defined experiences for the 3-day fabric demo."""

    experiences = [
        # ------------------------------------------------------------
        # Experience 1: Dual-arm pinch of cylindrical fabric edges
        # ------------------------------------------------------------
        Experience(
            task_intent="提起筒状布料并套入支撑板",
            target_object_type="筒状布料",
            target_object_tags=["soft", "cylindrical", "flexible", "textile"],
            gripper_config=GripperConfig.PARALLEL_2_FINGER,
            robot_dof=7,
            arm_count=2,
            action_type="pinch",
            dual_arm_pattern=DualArmPattern.COMPLEMENTARY,
            pre_contact_offset=[0.0, 0.0, 0.05],
            approach_angle_deg=90.0,
            gripper_aperture_m=0.0,
            contact_force_threshold_n=0.8,
            dual_arm_pinch_distance_m=0.08,
            dual_arm_sync_tolerance_m=0.002,
            left_grasp_relative=[0.0, -0.04, 0.0],
            right_grasp_relative=[0.0, 0.04, 0.0],
            trajectory_type="cartesian",
            waypoint_count=50,
            step_time_s=0.02,
            max_velocity_m_s=0.08,
            compliance_stiffness=80.0,
            source="human_demo",
            success=True,
            human_feedback="""双臂需同时接触布料后再闭合夹爪，避免单侧先夹导致布料滑动。
建议先以低速接近，触觉反馈确认接触后再闭合。""",
        ),

        # ------------------------------------------------------------
        # Experience 2: Dual-arm synchronized lift
        # ------------------------------------------------------------
        Experience(
            task_intent="提起筒状布料并套入支撑板",
            target_object_type="筒状布料",
            target_object_tags=["soft", "cylindrical"],
            gripper_config=GripperConfig.PARALLEL_2_FINGER,
            robot_dof=7,
            arm_count=2,
            action_type="lift",
            dual_arm_pattern=DualArmPattern.COMPLEMENTARY,
            pre_contact_offset=[0.0, 0.0, 0.0],
            approach_angle_deg=90.0,
            gripper_aperture_m=0.0,
            contact_force_threshold_n=1.0,
            dual_arm_sync_tolerance_m=0.002,
            trajectory_type="cartesian",
            waypoint_count=50,
            step_time_s=0.02,
            max_velocity_m_s=0.05,
            compliance_stiffness=100.0,
            source="human_demo",
            success=True,
            human_feedback="""提升时保持双臂严格同步，高度差不要超过2mm。
筒状布料重心在中间，提升过程不要旋转或倾斜。""",
        ),

        # ------------------------------------------------------------
        # Experience 3: Insert fabric over support plate
        # ------------------------------------------------------------
        Experience(
            task_intent="提起筒状布料并套入支撑板",
            target_object_type="铝合金支撑板",
            target_object_tags=["rigid", "plate", "aluminum"],
            gripper_config=GripperConfig.PARALLEL_2_FINGER,
            robot_dof=7,
            arm_count=2,
            action_type="insert",
            dual_arm_pattern=DualArmPattern.COMPLEMENTARY,
            pre_contact_offset=[0.0, 0.0, 0.02],
            approach_angle_deg=90.0,
            gripper_aperture_m=0.0,
            contact_force_threshold_n=2.0,
            dual_arm_sync_tolerance_m=0.002,
            trajectory_type="cartesian",
            waypoint_count=30,
            step_time_s=0.02,
            max_velocity_m_s=0.03,
            compliance_stiffness=120.0,
            source="human_demo",
            success=True,
            human_feedback="""套入时速度要慢，插入深度=支撑板高度+5~10mm余量。
如果遇到阻力（力矩突增），应立即停止并回退2mm，再尝试。""",
        ),

        # ------------------------------------------------------------
        # Experience 4: Withdraw fabric from support plate
        # ------------------------------------------------------------
        Experience(
            task_intent="提起筒状布料并套入支撑板",
            target_object_type="筒状布料",
            target_object_tags=["soft", "cylindrical"],
            gripper_config=GripperConfig.PARALLEL_2_FINGER,
            robot_dof=7,
            arm_count=2,
            action_type="withdraw",
            dual_arm_pattern=DualArmPattern.COMPLEMENTARY,
            pre_contact_offset=[0.0, 0.0, 0.05],
            approach_angle_deg=90.0,
            gripper_aperture_m=0.0,
            contact_force_threshold_n=1.5,
            dual_arm_sync_tolerance_m=0.002,
            trajectory_type="cartesian",
            waypoint_count=40,
            step_time_s=0.02,
            max_velocity_m_s=0.04,
            compliance_stiffness=100.0,
            source="human_demo",
            success=True,
            human_feedback="""先松右臂再松左臂，避免布料突然掉落。
提升时如果感受到较大阻力，说明布料可能粘在支撑板上，需要小幅晃动后提取。""",
        ),

        # ------------------------------------------------------------
        # Experience 5: Single-arm edge grasp (fallback for asymmetric fabric)
        # ------------------------------------------------------------
        Experience(
            task_intent="提起筒状布料并套入支撑板",
            target_object_type="筒状布料",
            target_object_tags=["soft", "cylindrical"],
            gripper_config=GripperConfig.PARALLEL_2_FINGER,
            robot_dof=7,
            arm_count=1,
            action_type="grasp",
            pre_contact_offset=[0.0, 0.0, 0.05],
            approach_angle_deg=45.0,
            gripper_aperture_m=0.02,
            contact_force_threshold_n=0.5,
            trajectory_type="cartesian",
            waypoint_count=50,
            step_time_s=0.02,
            max_velocity_m_s=0.06,
            compliance_stiffness=60.0,
            source="human_demo",
            success=True,
            human_feedback="""单臂抓取时建议夹取布料边缘重叠处（双层），增加摩擦力。
夹爪闭合前保持轻微下压，确保布料不会滑脱。""",
        ),
    ]

    for exp in experiences:
        library.add(exp)

    print(f"[Seed] Pre-loaded {len(experiences)} fabric manipulation experiences.")
    return experiences
