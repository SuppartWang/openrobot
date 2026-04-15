"""Core message types for openrobot inter-module communication."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class ProprioceptionState:
    joint_positions: np.ndarray
    joint_velocities: Optional[np.ndarray] = None
    ee_pose: Optional[np.ndarray] = None  # [x, y, z, qx, qy, qz, qw]
    timestamp: float = 0.0


@dataclass
class PerceptionMsg:
    timestamp: float
    rgb: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    proprioception: Optional[ProprioceptionState] = None
    audio: Optional[np.ndarray] = None
    touch: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitivePlan:
    task_id: str
    steps: List[Dict[str, Any]]
    reasoning: Optional[str] = None


@dataclass
class ActionCmd:
    type: str  # "joint_position", "ee_pose", "gripper", "stop"
    values: np.ndarray
    reflex_override: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
