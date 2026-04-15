"""Layer 1: System monitoring and vital signs for the robot software stack."""

import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class NodeHeartbeat:
    node_id: str
    last_seen: float
    status: str = "ok"  # ok, warn, error, dead
    metadata: Dict[str, Any] = field(default_factory=dict)


class RobotMonitor:
    """Monitors health of all openrobot software nodes."""

    def __init__(self, heartbeat_timeout: float = 2.0):
        self.heartbeat_timeout = heartbeat_timeout
        self._nodes: Dict[str, NodeHeartbeat] = {}
        self._start_time = time.time()

    def register_node(self, node_id: str, metadata: Optional[Dict[str, Any]] = None):
        self._nodes[node_id] = NodeHeartbeat(
            node_id=node_id,
            last_seen=time.time(),
            metadata=metadata or {},
        )
        logger.info(f"[Monitor] Registered node: {node_id}")

    def heartbeat(self, node_id: str, status: str = "ok", metadata: Optional[Dict[str, Any]] = None):
        if node_id not in self._nodes:
            self.register_node(node_id, metadata)
        hb = self._nodes[node_id]
        hb.last_seen = time.time()
        hb.status = status
        if metadata:
            hb.metadata.update(metadata)

    def check_health(self) -> Dict[str, str]:
        now = time.time()
        health = {}
        for node_id, hb in self._nodes.items():
            if now - hb.last_seen > self.heartbeat_timeout:
                hb.status = "dead"
            health[node_id] = hb.status
        return health

    def get_uptime(self) -> float:
        return time.time() - self._start_time

    def is_system_healthy(self) -> bool:
        return all(s != "dead" and s != "error" for s in self.check_health().values())
