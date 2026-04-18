"""Universal robot interface abstraction.

This module defines the base interface that ALL robot types must implement,
regardless of their kinematic structure (serial arm, mobile base, legged,
aerial, continuum, etc.).

Design principles:
1. Action/Observation are the universal lingua franca, not joint angles.
2. Each implementation defines its own action_space / observation_space.
3. Specific kinematic families extend via sub-interfaces (e.g. ManipulatorInterface).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Action:
    """Generic action container sent to a robot."""

    action_type: str
    """E.g. 'joint_position', 'cartesian_pose', 'twist', 'gripper', 'mode'."""

    values: np.ndarray
    """Raw action vector. Shape and interpretation depend on action_type."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Extra parameters, e.g. {'speed': 0.5, 'force': 1.0}."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "values": self.values.tolist() if isinstance(self.values, np.ndarray) else list(self.values),
            "metadata": self.metadata,
        }


@dataclass
class Observation:
    """Generic observation container received from a robot."""

    timestamp: float = field(default_factory=time.time)

    proprioception: Dict[str, Any] = field(default_factory=dict)
    """Self-state: joint positions, velocities, torques, temperatures, battery, etc."""

    extroception: Dict[str, Any] = field(default_factory=dict)
    """Exteroceptive sensor readings: cameras, lidar, imu, contact, etc."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Extra state: mode, errors, diagnostics."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "proprioception": _sanitize(self.proprioception),
            "extroception": _sanitize(self.extroception),
            "metadata": self.metadata,
        }


@dataclass
class Space:
    """Description of an action or observation space (gym-like)."""

    name: str
    shape: tuple
    dtype: str
    low: Optional[List[float]] = None
    high: Optional[List[float]] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": self.shape,
            "dtype": self.dtype,
            "low": self.low,
            "high": self.high,
            "description": self.description,
        }


class RobotInterface(ABC):
    """Base interface for ALL robot platforms.

    Concrete implementations: YHRGAdapter, FrankaMujocoAdapter,
    MobileBaseAdapter, QuadrupedAdapter, AerialAdapter, etc.
    """

    # ------------------------------------------------------------------
    # Identification
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def robot_type(self) -> str:
        """Return the kinematic family.

        Examples: 'manipulator', 'mobile_base', 'legged', 'aerial',
                  'aquatic', 'continuum', 'humanoid', 'end_effector'.
        """
        ...

    @property
    @abstractmethod
    def robot_id(self) -> str:
        """Unique identifier for this robot instance."""
        ...

    @property
    @abstractmethod
    def dof(self) -> int:
        """Degrees of freedom (actuated)."""
        ...

    # ------------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def action_space(self) -> Dict[str, Space]:
        """Return a dict of named action spaces this robot accepts.

        Example for a manipulator:
            {
                "joint_position": Space(shape=(6,), dtype="float32", low=[-pi,...], high=[pi,...]),
                "cartesian_pose": Space(shape=(7,), dtype="float32"),
                "gripper":        Space(shape=(1,), dtype="float32", low=[0], high=[2]),
            }
        """
        ...

    @property
    @abstractmethod
    def observation_space(self) -> Dict[str, Space]:
        """Return a dict of named observation spaces this robot provides."""
        ...

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @abstractmethod
    def enable(self) -> bool:
        """Power on / enable motors."""
        ...

    @abstractmethod
    def disable(self) -> bool:
        """Power off / disable motors."""
        ...

    @abstractmethod
    def reset(self) -> bool:
        """Reset to a known safe state."""
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True if the robot is online and ready to accept commands."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release all resources."""
        ...

    # ------------------------------------------------------------------
    # Control & Sense
    # ------------------------------------------------------------------
    @abstractmethod
    def command(self, action: Action) -> bool:
        """Send a generic action to the robot.

        The implementation maps action.action_type to the appropriate
        low-level driver call.
        """
        ...

    @abstractmethod
    def observe(self) -> Observation:
        """Return the latest observation from the robot."""
        ...

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def get_state_summary(self) -> str:
        """Return a concise human-readable state summary (for LLM prompts)."""
        obs = self.observe()
        lines = [f"Robot: {self.robot_id} (type={self.robot_type}, DOF={self.dof})"]

        # Proprioception
        prop = obs.proprioception
        if "joint_positions" in prop:
            j = prop["joint_positions"]
            lines.append(f"  Joints: {len(j)}-DOF, pos={[round(float(v), 3) for v in j]}")
        if "end_effector_pose" in prop:
            p = prop["end_effector_pose"]
            lines.append(f"  EE: xyz=[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]")
        if "battery" in prop:
            lines.append(f"  Battery: {prop['battery']}%")
        if "temperatures" in prop:
            temps = prop["temperatures"]
            lines.append(f"  Temps: max={max(temps):.1f}°C")

        # Errors
        errs = obs.metadata.get("errors", [])
        if errs:
            lines.append(f"  Errors: {errs}")

        return "\n".join(lines)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _sanitize(obj: Any) -> Any:
    """Recursively convert numpy arrays to lists for serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj
