"""Motion planning skills: path planning, trajectory optimization.

These skills generate collision-free paths and smooth trajectories
for robot navigation and manipulation.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from openrobot_demo.skills.base import SkillInterface, SkillSchema, ParamSchema, ResultSchema

logger = logging.getLogger(__name__)


class StraightLinePlannerSkill(SkillInterface):
    """Simple straight-line Cartesian planner (no obstacle avoidance)."""

    name = "straight_line_planner"

    @property
    def schema(self) -> SkillSchema:
        return SkillSchema(
            description="Plan a straight-line Cartesian path from current pose to target pose with linear interpolation.",
            parameters=[
                ParamSchema(name="start_pose", type="list", description="Start pose [x, y, z, qx, qy, qz, qw].", required=True),
                ParamSchema(name="goal_pose", type="list", description="Goal pose [x, y, z, qx, qy, qz, qw].", required=True),
                ParamSchema(name="num_waypoints", type="int", description="Number of waypoints including start and goal.", required=False, default=20),
            ],
            returns=[
                ResultSchema(name="success", type="bool", description="Whether planning succeeded."),
                ResultSchema(name="message", type="str", description="Status message."),
                ResultSchema(name="waypoints", type="list", description="List of pose waypoints [[x,y,z,qx,qy,qz,qw], ...]."),
                ResultSchema(name="path_length", type="float", description="Total path length in meters."),
            ],
            dependencies=["arm"],
        )

    def execute(self, start_pose: List[float], goal_pose: List[float],
                num_waypoints: int = 20, **kwargs) -> Dict[str, Any]:
        try:
            s = np.array(start_pose, dtype=np.float64)
            g = np.array(goal_pose, dtype=np.float64)

            if len(s) != 7 or len(g) != 7:
                return {"success": False, "message": "Start and goal must be 7-DOF poses [x,y,z,qx,qy,qz,qw]."}

            # Linear interpolation for position
            alphas = np.linspace(0, 1, num_waypoints)
            positions = np.array([s[:3] + a * (g[:3] - s[:3]) for a in alphas])

            # Spherical linear interpolation (slerp) for quaternion
            from scipy.spatial.transform import Rotation as R, Slerp
            key_times = [0, 1]
            key_rots = R.from_quat([s[3:7], g[3:7]])
            slerp = Slerp(key_times, key_rots)
            quats = slerp(alphas).as_quat()

            waypoints = []
            for pos, quat in zip(positions, quats):
                waypoints.append(pos.tolist() + quat.tolist())

            path_length = float(np.linalg.norm(g[:3] - s[:3]))

            return {
                "success": True,
                "message": f"Planned {len(waypoints)} waypoints, path length {path_length:.3f}m.",
                "waypoints": waypoints,
                "path_length": path_length,
            }
        except Exception as exc:
            logger.exception("[StraightLinePlannerSkill] Failed")
            return {"success": False, "message": str(exc)}


class JointSpacePlannerSkill(SkillInterface):
    """Plan a smooth joint-space trajectory between two configurations."""

    name = "joint_space_planner"

    @property
    def schema(self) -> SkillSchema:
        return SkillSchema(
            description="Plan a smooth joint-space trajectory from start to goal using cubic or linear interpolation.",
            parameters=[
                ParamSchema(name="start_joints", type="list", description="Start joint positions (radians).", required=True),
                ParamSchema(name="goal_joints", type="list", description="Goal joint positions (radians).", required=True),
                ParamSchema(name="num_waypoints", type="int", description="Number of waypoints.", required=False, default=50),
                ParamSchema(name="interpolation", type="str", description="Interpolation type: 'linear' or 'cubic'.", required=False, default="linear"),
            ],
            returns=[
                ResultSchema(name="success", type="bool", description="Whether planning succeeded."),
                ResultSchema(name="message", type="str", description="Status message."),
                ResultSchema(name="waypoints", type="list", description="List of joint waypoints."),
                ResultSchema(name="max_joint_delta", type="float", description="Max joint displacement across the path."),
            ],
            dependencies=["arm"],
        )

    def execute(self, start_joints: List[float], goal_joints: List[float],
                num_waypoints: int = 50, interpolation: str = "linear", **kwargs) -> Dict[str, Any]:
        try:
            s = np.array(start_joints, dtype=np.float64)
            g = np.array(goal_joints, dtype=np.float64)

            if len(s) != len(g):
                return {"success": False, "message": "Start and goal joint counts must match."}

            alphas = np.linspace(0, 1, num_waypoints)

            if interpolation == "linear":
                waypoints = np.array([s + a * (g - s) for a in alphas])
            elif interpolation == "cubic":
                # Cubic interpolation with zero velocity at endpoints
                waypoints = np.zeros((num_waypoints, len(s)))
                for i, a in enumerate(alphas):
                    # h(t) = 3t^2 - 2t^3
                    h = 3 * a ** 2 - 2 * a ** 3
                    waypoints[i] = s + h * (g - s)
            else:
                return {"success": False, "message": f"Unknown interpolation: {interpolation}"}

            max_delta = float(np.max(np.abs(g - s)))

            return {
                "success": True,
                "message": f"Planned {len(waypoints)} joint waypoints using {interpolation} interpolation.",
                "waypoints": waypoints.tolist(),
                "max_joint_delta": max_delta,
            }
        except Exception as exc:
            logger.exception("[JointSpacePlannerSkill] Failed")
            return {"success": False, "message": str(exc)}
