"""WorldModel: unified memory and spatial/semantic state for the robot."""

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np

from openrobot_demo.sensors.base import PerceptionData

logger = logging.getLogger(__name__)


@dataclass
class ObjectDesc:
    """Semantic description of an object in the world."""

    object_id: str
    object_type: str = "unknown"
    position: Optional[List[float]] = None  # [x, y, z] in base frame
    color: Optional[str] = None
    size: Optional[str] = None  # e.g. "5cm cube"
    material: Optional[str] = None
    grasp_points: List[List[float]] = field(default_factory=list)
    owner: Optional[str] = None
    relations: Dict[str, str] = field(default_factory=dict)  # e.g. {"on": "table", "left_of": "cup"}
    last_seen: float = field(default_factory=time.time)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RobotState:
    """Latest proprioceptive state of the robot."""

    timestamp: float = field(default_factory=time.time)
    joint_positions: Optional[List[float]] = None
    joint_velocities: Optional[List[float]] = None
    end_effector_pose: Optional[List[float]] = None  # [x, y, z, qx, qy, qz, qw]
    gripper_width: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpatialMemory:
    """Simple spatial memory: explored regions, obstacles, room layout."""

    timestamp: float = field(default_factory=time.time)
    robot_position: Optional[List[float]] = None  # [x, y, z] in world frame
    known_surfaces: List[str] = field(default_factory=list)  # e.g. ["table", "shelf"]
    obstacle_regions: List[Dict[str, Any]] = field(default_factory=list)
    workspace_bounds: Optional[Dict[str, List[float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskMemory:
    """ episodic memory of a past task. """

    episode_id: str
    instruction: str
    status: str  # completed / failed
    summary: str = ""
    final_objects_state: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    key_learnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WorldModel:
    """
    Central world model that fuses perception data into a persistent,
    queryable representation of the robot's environment and history.
    """

    def __init__(self):
        self.objects: Dict[str, ObjectDesc] = {}
        self.robot_state = RobotState()
        self.spatial_memory = SpatialMemory()
        self.task_memories: List[TaskMemory] = []
        self._raw_perception_log: List[PerceptionData] = []

    # ------------------------------------------------------------------
    # Perception ingestion
    # ------------------------------------------------------------------
    def ingest(self, data: PerceptionData) -> None:
        """Ingest a PerceptionData and update the world model accordingly."""
        self._raw_perception_log.append(data)

        if data.modality == "proprioception":
            self._update_from_proprioception(data)
        elif data.modality == "rgb":
            self._update_from_rgb(data)
        elif data.modality == "depth":
            self._update_from_depth(data)
        elif data.modality == "pointcloud":
            self._update_from_pointcloud(data)
        elif data.modality == "tactile":
            self._update_from_tactile(data)
        elif data.modality == "imu":
            self._update_from_imu(data)
        elif data.modality == "wrench":
            self._update_from_wrench(data)
        elif data.modality == "lidar":
            self._update_from_lidar(data)
        elif data.modality == "ultrasonic":
            self._update_from_ultrasonic(data)
        elif data.modality == "odometry":
            self._update_from_odometry(data)
        elif data.modality == "audio":
            self._update_from_audio(data)
        else:
            logger.debug("[WorldModel] Unhandled modality: %s", data.modality)

    def _update_from_proprioception(self, data: PerceptionData):
        payload = data.payload
        self.robot_state = RobotState(
            timestamp=data.timestamp,
            joint_positions=_tolist(payload.get("joint_positions")),
            joint_velocities=_tolist(payload.get("joint_velocities")),
            end_effector_pose=_tolist(payload.get("end_effector_pose")),
            gripper_width=payload.get("gripper_width"),
        )

    def _update_from_rgb(self, data: PerceptionData):
        # Future: run VLM/vision model here to detect objects
        logger.debug("[WorldModel] RGB frame ingested from %s", data.source_id)

    def _update_from_depth(self, data: PerceptionData):
        logger.debug("[WorldModel] Depth frame ingested from %s", data.source_id)

    def _update_from_pointcloud(self, data: PerceptionData):
        logger.debug(
            "[WorldModel] PointCloud ingested from %s (%d points)",
            data.source_id,
            data.metadata.get("num_points", 0),
        )

    def _update_from_tactile(self, data: PerceptionData):
        payload = data.payload
        in_contact = payload.get("in_contact", False)
        total_force = payload.get("total_force", 0.0)
        logger.debug(
            "[WorldModel] Tactile: contact=%s, total_force=%.3f",
            in_contact,
            total_force,
        )

    def _update_from_imu(self, data: PerceptionData):
        payload = data.payload
        acc = payload.get("acceleration")
        gyro = payload.get("angular_velocity")
        logger.debug("[WorldModel] IMU: acc=%s, gyro=%s", acc, gyro)

    def _update_from_wrench(self, data: PerceptionData):
        payload = data.payload
        force = payload.get("force")
        torque = payload.get("torque")
        logger.debug("[WorldModel] Wrench: force=%s, torque=%s", force, torque)

    def _update_from_lidar(self, data: PerceptionData):
        payload = data.payload
        num_beams = len(payload.get("ranges", []))
        logger.debug("[WorldModel] LiDAR: %d beams", num_beams)

    def _update_from_ultrasonic(self, data: PerceptionData):
        payload = data.payload
        dist = payload.get("distance_m")
        logger.debug("[WorldModel] Ultrasonic: distance=%.3fm", dist)

    def _update_from_odometry(self, data: PerceptionData):
        payload = data.payload
        pose = payload.get("pose")
        if pose is not None:
            self.spatial_memory.robot_position = pose[:3] if len(pose) >= 3 else pose
        logger.debug("[WorldModel] Odometry: pose=%s", pose)

    def _update_from_audio(self, data: PerceptionData):
        payload = data.payload
        if isinstance(payload, np.ndarray):
            logger.debug("[WorldModel] Audio: %d samples", len(payload))
        else:
            logger.debug("[WorldModel] Audio: data received")

    # ------------------------------------------------------------------
    # Object / spatial API
    # ------------------------------------------------------------------
    def add_or_update_object(self, desc: ObjectDesc):
        """Add a new object or update an existing one by object_id."""
        self.objects[desc.object_id] = desc
        logger.info("[WorldModel] Object updated: %s at %s", desc.object_id, desc.position)

    def get_object(self, object_id: str) -> Optional[ObjectDesc]:
        return self.objects.get(object_id)

    def query_nearby_objects(
        self, position: List[float], radius: float = 0.3
    ) -> List[ObjectDesc]:
        """Return objects within `radius` meters of `position`."""
        nearby = []
        pos = np.array(position[:3])
        for obj in self.objects.values():
            if obj.position is None:
                continue
            dist = np.linalg.norm(np.array(obj.position[:3]) - pos)
            if dist <= radius:
                nearby.append(obj)
        return nearby

    def remove_object(self, object_id: str):
        self.objects.pop(object_id, None)

    def set_robot_position(self, position: List[float]):
        self.spatial_memory.robot_position = _tolist(position)
        self.spatial_memory.timestamp = time.time()

    def add_surface(self, name: str):
        if name not in self.spatial_memory.known_surfaces:
            self.spatial_memory.known_surfaces.append(name)

    # ------------------------------------------------------------------
    # Task memory API
    # ------------------------------------------------------------------
    def add_task_memory(self, memory: TaskMemory):
        self.task_memories.append(memory)
        logger.info("[WorldModel] Task memory recorded: %s (%s)", memory.episode_id, memory.status)

    def get_task_memories(self, instruction_keyword: str = "", limit: int = 5) -> List[TaskMemory]:
        results = [m for m in self.task_memories if instruction_keyword.lower() in m.instruction.lower()]
        return results[-limit:]

    # ------------------------------------------------------------------
    # State summary for planner
    # ------------------------------------------------------------------
    def build_state_summary(self) -> str:
        """Build a concise state summary string for the ReAct planner."""
        lines = []

        # Robot state
        rs = self.robot_state
        if rs.end_effector_pose is not None:
            p = rs.end_effector_pose
            lines.append(f"末端位姿 xyz=[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]")
        if rs.gripper_width is not None:
            lines.append(f"夹爪开度={rs.gripper_width:.3f}m")

        # Objects
        if self.objects:
            lines.append("已知物体:")
            for obj in self.objects.values():
                pos_str = f" 位置={_fmt_xyz(obj.position)}" if obj.position else ""
                color_str = f" 颜色={obj.color}" if obj.color else ""
                rel_str = f" 关系={obj.relations}" if obj.relations else ""
                lines.append(f" - {obj.object_id} ({obj.object_type}){pos_str}{color_str}{rel_str}")
        else:
            lines.append("当前环境中没有已记录的物体。")

        # Spatial memory
        sm = self.spatial_memory
        if sm.known_surfaces:
            lines.append(f"已知平面: {', '.join(sm.known_surfaces)}")
        if sm.robot_position:
            lines.append(f"机器人当前位置: {_fmt_xyz(sm.robot_position)}")

        # Recent task memory
        recent = self.task_memories[-2:]
        if recent:
            lines.append("最近任务记忆:")
            for mem in recent:
                lines.append(f" - [{mem.status}] {mem.instruction}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full world model for debugging or persistence."""
        return {
            "robot_state": self.robot_state.to_dict(),
            "spatial_memory": self.spatial_memory.to_dict(),
            "objects": {k: v.to_dict() for k, v in self.objects.items()},
            "task_memories": [m.to_dict() for m in self.task_memories],
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _tolist(v):
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return v.tolist()
    if hasattr(v, "tolist"):
        return v.tolist()
    return list(v)


def _fmt_xyz(v) -> str:
    if v is None:
        return "None"
    return f"[{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}]"
