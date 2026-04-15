"""Safety Gateway for YHRG arm motion commands."""

import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)


class SafetyGateway:
    """Intercepts and validates every motion command before execution."""

    JOINT_LIMITS = [
        (-2.967, 2.967),
        (0.0, 3.142),
        (0.0, 2.967),
        (-1.571, 1.518),
        (-1.571, 1.571),
        (-1.571, 1.571),
        (-1.745, 1.745),
    ]

    def __init__(self, max_joint_speed: float = 1.0,
                 workspace: dict = None):
        self.max_joint_speed = max_joint_speed
        self.workspace = workspace or {
            "x": (-0.2, 0.8),
            "y": (-0.6, 0.6),
            "z": (0.05, 0.8),
        }

    def check_joint_command(self, target: List[float], current: List[float] = None, dt: float = 1.0) -> tuple:
        """Returns (ok: bool, clamped_target: List[float], reason: str)."""
        if len(target) < 7:
            return False, target, f"Target joint count {len(target)} < 7"

        clamped = []
        for i, val in enumerate(target[:7]):
            lo, hi = self.JOINT_LIMITS[i]
            if val < lo or val > hi:
                logger.warning(f"Joint {i+1} out of limits: {val:.3f} not in [{lo:.3f}, {hi:.3f}]")
            clamped.append(float(np.clip(val, lo, hi)))

        if current is not None and dt > 0:
            for i in range(7):
                delta = abs(clamped[i] - current[i]) / dt
                if delta > self.max_joint_speed:
                    logger.warning(f"Joint {i+1} speed too high: {delta:.3f} rad/s")
                    # Re-clamp to max speed
                    direction = 1 if clamped[i] >= current[i] else -1
                    clamped[i] = current[i] + direction * self.max_joint_speed * dt

        return True, clamped, "ok"

    def check_cartesian_target(self, xyz: List[float]) -> tuple:
        """Returns (ok: bool, reason: str)."""
        x, y, z = xyz[:3]
        wx = self.workspace.get("x", (-float("inf"), float("inf")))
        wy = self.workspace.get("y", (-float("inf"), float("inf")))
        wz = self.workspace.get("z", (-float("inf"), float("inf")))

        if not (wx[0] <= x <= wx[1] and wy[0] <= y <= wy[1] and wz[0] <= z <= wz[1]):
            return False, f"Target ({x:.3f}, {y:.3f}, {z:.3f}) outside workspace."
        return True, "ok"
