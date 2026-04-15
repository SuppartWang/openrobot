import pytest
from openrobot_core.openrobot_monitor.monitor import RobotMonitor


def test_register_and_heartbeat():
    mon = RobotMonitor(heartbeat_timeout=1.0)
    mon.register_node("test_node")
    mon.heartbeat("test_node", status="ok")
    health = mon.check_health()
    assert health["test_node"] == "ok"
    assert mon.is_system_healthy() is True


def test_dead_node():
    mon = RobotMonitor(heartbeat_timeout=0.01)
    mon.register_node("fast_node")
    mon.heartbeat("fast_node")
    import time
    time.sleep(0.02)
    health = mon.check_health()
    assert health["fast_node"] == "dead"
    assert mon.is_system_healthy() is False
