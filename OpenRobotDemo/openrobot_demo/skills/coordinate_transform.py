"""CoordinateTransformSkill: Convert 3D points from camera frame to arm base frames.

Camera setup (world frame = left arm base frame):
    Left arm base:  (0.00, 0.00, 9.0)
    Right arm base: (0.56, 0.00, 9.0)
    Camera center:  (0.28, 0.71, 9.0)
    Camera facing:  -y direction

Camera coordinate system (standard RealSense):
    z: forward (optical axis, points to world -y)
    x: right     (points to world +x)
    y: down      (points to world -z)

Rotation matrix R_cam_to_world:
    [[ 1,  0,  0],
     [ 0,  0, -1],
     [ 0, -1,  0]]
"""

import logging
from typing import Any, Dict, List

import numpy as np

from openrobot_demo.skills.base import SkillInterface, SkillSchema, ParamSchema, ResultSchema

logger = logging.getLogger(__name__)


class CoordinateTransformSkill(SkillInterface):
    """Transform 3D points between camera frame and robot arm base frames."""

    name = "coordinate_transform"

    # World frame origin = projection of left arm base on table surface
    LEFT_ARM_BASE = np.array([0.0, 0.0, 0.09])
    RIGHT_ARM_BASE = np.array([0.56, 0.0, 0.09])
    CAMERA_POS = np.array([0.28, 0.71, 0.09])

    # Camera z -> world -y, camera x -> world x, camera y -> world -z
    R_CAM_TO_WORLD = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
    ])

    @property
    def schema(self) -> SkillSchema:
        return SkillSchema(
            description=(
                "Transform a 3D point from the fixed RealSense camera frame "
                "to the left-arm or right-arm base coordinate frame. "
                "Also supports batch transformation of multiple points."
            ),
            parameters=[
                ParamSchema(
                    name="point_camera",
                    type="list",
                    description="3D point [x, y, z] in camera frame (meters).",
                    required=True,
                ),
                ParamSchema(
                    name="target_frame",
                    type="str",
                    description='Target frame: "left", "right", or "world".',
                    required=False,
                    default="left",
                ),
                ParamSchema(
                    name="offset_z",
                    type="float",
                    description="Optional Z offset to add after transform (e.g. +0.03 for 3cm above).",
                    required=False,
                    default=0.0,
                ),
                ParamSchema(
                    name="points_camera",
                    type="list",
                    description="Batch mode: list of [x, y, z] points in camera frame.",
                    required=False,
                    default=None,
                ),
            ],
            returns=[
                ResultSchema(name="success", type="bool", description="Transform succeeded."),
                ResultSchema(name="message", type="str", description="Status message."),
                ResultSchema(name="point", type="list", description="Transformed 3D point [x, y, z]."),
                ResultSchema(name="points", type="list", description="Batch result: list of transformed points."),
            ],
            dependencies=[],
        )

    def execute(
        self,
        point_camera: List[float] = None,
        target_frame: str = "left",
        offset_z: float = 0.0,
        points_camera: List[List[float]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # Batch mode
        if points_camera is not None:
            out = []
            for p in points_camera:
                res = self._transform_single(p, target_frame, offset_z)
                out.append(res)
            return {
                "success": True,
                "message": f"Transformed {len(out)} points to {target_frame} frame.",
                "points": out,
                "point": out[0] if out else None,
            }

        # Single point mode
        if point_camera is None:
            return {"success": False, "message": "No input point provided.", "point": None}

        pt = self._transform_single(point_camera, target_frame, offset_z)
        return {
            "success": True,
            "message": f"Transformed to {target_frame} frame: {pt}",
            "point": pt,
        }

    def _transform_single(self, point_camera: List[float], target_frame: str, offset_z: float) -> List[float]:
        p_cam = np.array(point_camera, dtype=float)
        p_world = self.CAMERA_POS + self.R_CAM_TO_WORLD @ p_cam

        if target_frame == "world":
            p_out = p_world
        elif target_frame == "left":
            p_out = p_world - self.LEFT_ARM_BASE
        elif target_frame == "right":
            p_out = p_world - self.RIGHT_ARM_BASE
        else:
            raise ValueError(f"Unknown target_frame: {target_frame}")

        # Apply Z offset (e.g. move 3cm above)
        if offset_z != 0.0:
            p_out = p_out.copy()
            p_out[2] += offset_z

        return [round(float(v), 4) for v in p_out.tolist()]
