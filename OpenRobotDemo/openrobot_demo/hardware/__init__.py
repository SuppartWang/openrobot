"""Hardware abstraction layer for OpenRobotDemo.

Provides unified robot interfaces across all kinematic families.
"""

from openrobot_demo.hardware.robot_interface import (
    RobotInterface,
    Action,
    Observation,
    Space,
)
from openrobot_demo.hardware.manipulator_interface import ManipulatorInterface
from openrobot_demo.hardware.yhrg_adapter import (
    YHRGAdapter,
    YHRGKinematics,
    S1_arm,
    S1_slover,
    control_mode,
    Arm_Search,
)
from openrobot_demo.hardware.mujoco_franka_adapter import (
    FrankaMujocoAdapter,
    FrankaMujocoKinematics,
)

__all__ = [
    # Universal interfaces
    "RobotInterface",
    "ManipulatorInterface",
    "Action",
    "Observation",
    "Space",
    # S1 hardware
    "YHRGAdapter",
    "YHRGKinematics",
    "S1_arm",
    "S1_slover",
    "control_mode",
    "Arm_Search",
    # MuJoCo Franka
    "FrankaMujocoAdapter",
    "FrankaMujocoKinematics",
]
