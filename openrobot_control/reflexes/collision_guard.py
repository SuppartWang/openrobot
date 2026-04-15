"""Layer 3: Low-level reflexes for collision avoidance and emergency stop."""

from typing import Optional
import mujoco
import numpy as np
from openrobot_msgs import ActionCmd


class CollisionGuard:
    """Monitors MuJoCo contacts and triggers emergency stop if unexpected collision occurs."""

    def __init__(self, model: mujoco.MjModel, exclude_geom_names: Optional[list] = None):
        self.model = model
        self.exclude_ids = set()
        if exclude_geom_names:
            for name in exclude_geom_names:
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if gid >= 0:
                    self.exclude_ids.add(gid)

    def check(self, data: mujoco.MjData) -> bool:
        """Returns True if a collision is detected (excluding allowed pairs)."""
        for c in range(data.ncon):
            con = data.contact[c]
            if con.geom1 in self.exclude_ids or con.geom2 in self.exclude_ids:
                continue
            # Ignore contacts with zero distance (penetration) as they may be initialization artifacts
            if con.dist < -0.001:
                return True
        return False

    def emergency_stop(self) -> ActionCmd:
        return ActionCmd(type="stop", values=np.zeros(self.model.nu))
