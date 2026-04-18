"""Skill 4: GraspPointPredictor (rule-based MVP)."""

import logging
from typing import Any, Dict
import numpy as np
from scipy.spatial.transform import Rotation as R

from openrobot_demo.skills.base import SkillInterface, SkillSchema, ParamSchema, ResultSchema

logger = logging.getLogger(__name__)


class GraspPointPredictor(SkillInterface):
    """Predict grasp pose based on object type and 3D pose (rule-based)."""

    @property
    def name(self) -> str:
        return "grasp_point_predictor"

    @property
    def schema(self) -> SkillSchema:
        return SkillSchema(
            description="Predict a grasp pose (position + orientation) and pre-grasp approach pose for a given object type and 3D position.",
            parameters=[
                ParamSchema(name="object_pose_base", type="list", description="Object 3D pose in base frame [x, y, z, qx, qy, qz, qw] or [x, y, z].", required=True),
                ParamSchema(name="object_type", type="str", description="Object type: 'box', 'cube', 'cylinder', 'bottle', 'sphere', etc.", required=False, default="unknown"),
                ParamSchema(name="gripper_type", type="str", description="Gripper type: 'parallel', 'suction', etc.", required=False, default="parallel"),
            ],
            returns=[
                ResultSchema(name="success", type="bool", description="Whether prediction succeeded."),
                ResultSchema(name="message", type="str", description="Status message."),
                ResultSchema(name="grasp_pose", type="list", description="Grasp pose [x, y, z, qx, qy, qz, qw]."),
                ResultSchema(name="pre_grasp_pose", type="list", description="Pre-grasp approach pose [x, y, z, qx, qy, qz, qw]."),
                ResultSchema(name="approach_vector", type="list", description="Approach direction vector [dx, dy, dz]."),
                ResultSchema(name="gripper_width", type="float", description="Recommended gripper aperture (meters)."),
            ],
            dependencies=["arm"],
            preconditions=["object 3D pose must be known"],
            postconditions=["grasp pose is computed and available for arm motion executor"],
        )

    def execute(self,
                object_pose_base: list,
                object_type: str = "unknown",
                gripper_type: str = "parallel",
                **kwargs) -> Dict[str, Any]:
        obj_pos = np.array(object_pose_base[:3])
        obj_type = object_type.lower()

        # Default: top-down grasp
        approach = np.array([0.0, 0.0, -1.0])
        grasp_pos = obj_pos.copy()
        grasp_rot = R.from_euler("xyz", [0, 0, 0])
        gripper_width = 0.04

        if obj_type in ["box", "cube", "rectangular", "长方体"]:
            # Top-down vertical grasp
            grasp_pos[2] += 0.02  # slightly above center
            grasp_rot = R.from_euler("xyz", [np.pi, 0, 0])
            approach = np.array([0.0, 0.0, -1.0])
            gripper_width = 0.05

        elif obj_type in ["cylinder", "bottle", "圆柱体"]:
            # Horizontal side grasp
            grasp_pos[1] += 0.0
            grasp_rot = R.from_euler("xyz", [0, np.pi / 2, 0])
            approach = np.array([0.0, -1.0, 0.0])
            gripper_width = 0.04

        elif obj_type in ["sphere", "ball", "球体"]:
            # Top-down grasp
            grasp_pos[2] += 0.01
            grasp_rot = R.from_euler("xyz", [np.pi, 0, 0])
            approach = np.array([0.0, 0.0, -1.0])
            gripper_width = 0.03

        else:
            # Generic top-down
            grasp_pos[2] += 0.02
            grasp_rot = R.from_euler("xyz", [np.pi, 0, 0])
            logger.info(f"Unknown object type '{object_type}', using default top-down grasp.")

        # Pre-grasp (approach) position: 10cm back along approach vector
        pre_grasp_pos = grasp_pos - approach * 0.10

        quat = grasp_rot.as_quat()  # x,y,z,w
        grasp_pose = [float(grasp_pos[0]), float(grasp_pos[1]), float(grasp_pos[2]),
                      float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
        pre_grasp_pose = [float(pre_grasp_pos[0]), float(pre_grasp_pos[1]), float(pre_grasp_pos[2]),
                          float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]

        return {
            "success": True,
            "message": f"Grasp predicted for {object_type}.",
            "grasp_pose": grasp_pose,
            "pre_grasp_pose": pre_grasp_pose,
            "approach_vector": approach.tolist(),
            "gripper_width": gripper_width,
        }
