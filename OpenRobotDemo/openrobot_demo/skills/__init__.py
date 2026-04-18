"""Skills for OpenRobotDemo."""

from openrobot_demo.skills.base import SkillInterface, SkillSchema, ParamSchema, ResultSchema
from openrobot_demo.skills.camera_capture import CameraCapture
from openrobot_demo.skills.arm_state_reader import ArmStateReader
from openrobot_demo.skills.vision_3d_estimator import Vision3DEstimator
from openrobot_demo.skills.grasp_predictor import GraspPointPredictor
from openrobot_demo.skills.arm_executor import ArmMotionExecutor
from openrobot_demo.skills.vla_policy_executor import VLAPolicyExecutor

# Signal processing & algorithms (P1-3)
from openrobot_demo.skills.signal_processing import (
    LowPassFilterSkill,
    KalmanFilter1DSkill,
    FFTSkill,
)
from openrobot_demo.skills.pointcloud_processing import (
    RANSACPlaneSegmentationSkill,
    EuclideanClusteringSkill,
    StatisticalOutlierRemovalSkill,
)
from openrobot_demo.skills.vision_processing import (
    ColorDetectorSkill,
    FeatureExtractorSkill,
)
from openrobot_demo.skills.motion_planning import (
    StraightLinePlannerSkill,
    JointSpacePlannerSkill,
)

__all__ = [
    # Base
    "SkillInterface",
    "SkillSchema",
    "ParamSchema",
    "ResultSchema",
    # Core skills
    "CameraCapture",
    "ArmStateReader",
    "Vision3DEstimator",
    "GraspPointPredictor",
    "ArmMotionExecutor",
    "VLAPolicyExecutor",
    # Signal processing
    "LowPassFilterSkill",
    "KalmanFilter1DSkill",
    "FFTSkill",
    # Point cloud
    "RANSACPlaneSegmentationSkill",
    "EuclideanClusteringSkill",
    "StatisticalOutlierRemovalSkill",
    # Vision
    "ColorDetectorSkill",
    "FeatureExtractorSkill",
    # Motion planning
    "StraightLinePlannerSkill",
    "JointSpacePlannerSkill",
]
