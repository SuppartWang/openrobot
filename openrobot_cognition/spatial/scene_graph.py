"""Layer 4: Simple spatial cognition via scene graph of known objects."""

from typing import Dict, List, Any, Optional
import numpy as np


class SceneGraph:
    """Maintains a sparse semantic map of object poses in the workspace."""

    def __init__(self):
        self.objects: Dict[str, Dict[str, Any]] = {}

    def register_object(self, obj_id: str, position: np.ndarray, obj_type: str = "unknown", color: Optional[str] = None):
        self.objects[obj_id] = {
            "id": obj_id,
            "type": obj_type,
            "position": np.array(position, dtype=float),
            "color": color,
        }

    def update_pose(self, obj_id: str, position: np.ndarray):
        if obj_id in self.objects:
            self.objects[obj_id]["position"] = np.array(position, dtype=float)

    def get_object(self, obj_id: str) -> Optional[Dict[str, Any]]:
        return self.objects.get(obj_id)

    def query_spatial_relation(self, reference: str, target: str) -> Optional[str]:
        """Return a rough spatial relation string (e.g. 'left', 'right', 'above')."""
        ref = self.objects.get(reference)
        tgt = self.objects.get(target)
        if ref is None or tgt is None:
            return None
        delta = tgt["position"] - ref["position"]
        relations = []
        if delta[0] > 0.05:
            relations.append("right")
        elif delta[0] < -0.05:
            relations.append("left")
        if delta[1] > 0.05:
            relations.append("front")
        elif delta[1] < -0.05:
            relations.append("back")
        if delta[2] > 0.05:
            relations.append("above")
        elif delta[2] < -0.05:
            relations.append("below")
        return "_".join(relations) if relations else "near"

    def to_context_string(self) -> str:
        lines = []
        for obj_id, info in self.objects.items():
            pos = info["position"]
            lines.append(f'{obj_id}: {info["type"]} at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), color={info.get("color", "unknown")}')
        return "\n".join(lines)
