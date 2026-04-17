"""Experience schema: structured, parameterizable skill primitives."""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
import time


class GripperConfig(str, Enum):
    """End-effector gripper configurations."""

    PARALLEL_2_FINGER = "parallel_2_finger"
    PARALLEL_3_FINGER = "parallel_3_finger"
    VACUUM_SUCTION = "vacuum_suction"
    CUSTOM = "custom"


class DualArmPattern(str, Enum):
    """Coordination pattern for dual-arm manipulation."""

    MIRROR = "mirror"          # 左右臂镜像对称
    COMPLEMENTARY = "complementary"  # 左右臂互补（如各抓一侧）
    INDEPENDENT = "independent"      # 左右臂独立执行不同动作
    LEADER_FOLLOWER = "leader_follower"  # 主从跟随


@dataclass
class Experience:
    """
    A parameterizable, retrievable experience record.

    Structure: Context → Parametric Policy → Outcome
    This replaces rigid lookup tables with fuzzy, adaptive skill memories.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    experience_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    # ------------------------------------------------------------------
    # Context — what situation triggers this experience?
    # ------------------------------------------------------------------
    task_intent: str = ""                    # e.g. "提起筒状布料并套入支撑板"
    target_object_type: str = ""             # e.g. "筒状布料"
    target_object_tags: List[str] = field(default_factory=list)  # e.g. ["soft", "cylindrical", "flexible"]
    gripper_config: GripperConfig = GripperConfig.PARALLEL_2_FINGER
    robot_dof: int = 7                       # 7-DOF arm
    arm_count: int = 1                       # 1 or 2

    # ------------------------------------------------------------------
    # Parametric Policy — the reusable, adjustable action recipe
    # ------------------------------------------------------------------
    action_type: str = ""                    # e.g. "grasp", "insert", "pinch", "lift", "withdraw"
    dual_arm_pattern: Optional[DualArmPattern] = None

    # Spatial parameters (unit: meters unless noted)
    pre_contact_offset: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.05])
    approach_angle_deg: float = 90.0         # 0=horizontal, 90=vertical
    gripper_aperture_m: float = 0.0          # 0=closed, >0=partial open
    contact_force_threshold_n: float = 1.0   # tactile/contact force trigger

    # Dual-arm specific parameters
    dual_arm_pinch_distance_m: Optional[float] = None   # 双臂夹取时两夹爪间距
    dual_arm_sync_tolerance_m: float = 0.002            # 双臂位置同步容差
    left_grasp_relative: Optional[List[float]] = None   # 相对物体中心的左臂抓取偏移
    right_grasp_relative: Optional[List[float]] = None  # 相对物体中心的右臂抓取偏移

    # Trajectory hints
    trajectory_type: str = "joint"           # "joint" or "cartesian"
    waypoint_count: int = 50                 # 插值步数
    step_time_s: float = 0.02                # 每步周期

    # Compliance / safety
    max_velocity_m_s: float = 0.1
    compliance_stiffness: float = 100.0      # N/m

    # ------------------------------------------------------------------
    # Outcome — what happened last time?
    # ------------------------------------------------------------------
    success: bool = True
    execution_time_s: float = 0.0
    final_error_m: float = 0.0               # 末端误差
    tactile_feedback: Optional[Dict[str, Any]] = None
    human_feedback: Optional[str] = None     # 人工改进建议

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    source: str = "human_demo"               # "human_demo", "autonomous_trial", "vla_inference"
    use_count: int = 0
    last_used: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert enums to strings for JSON serialization
        d["gripper_config"] = self.gripper_config.value
        if self.dual_arm_pattern:
            d["dual_arm_pattern"] = self.dual_arm_pattern.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experience":
        # Restore enums from strings
        if "gripper_config" in data:
            data["gripper_config"] = GripperConfig(data["gripper_config"])
        if "dual_arm_pattern" in data and data["dual_arm_pattern"]:
            data["dual_arm_pattern"] = DualArmPattern(data["dual_arm_pattern"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def context_signature(self) -> str:
        """Return a compact signature for fuzzy matching."""
        return (
            f"{self.task_intent}|"
            f"{self.target_object_type}|"
            f"{self.gripper_config.value}|"
            f"{self.arm_count}arm|"
            f"{self.action_type}"
        )
