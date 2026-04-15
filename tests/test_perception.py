import numpy as np
from openrobot_msgs import PerceptionMsg, ProprioceptionState
from openrobot_perception.io_bus.bus import PerceptionBus
from openrobot_perception.io_bus.interface import SensorInterface


class DummyRGBSensor(SensorInterface):
    @property
    def name(self):
        return "rgb_camera"

    def connect(self):
        return True

    def read(self):
        return np.zeros((10, 10, 3), dtype=np.uint8)

    def disconnect(self):
        pass


class DummyProprioSensor(SensorInterface):
    @property
    def name(self):
        return "proprioception"

    def connect(self):
        return True

    def read(self):
        return {
            "joint_positions": np.array([0.1, 0.2]),
            "joint_velocities": np.array([0.0, 0.0]),
            "ee_pose": np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]),
            "timestamp": 0.0,
        }

    def disconnect(self):
        pass


def test_perception_bus_poll():
    bus = PerceptionBus()
    bus.attach(DummyRGBSensor())
    bus.attach(DummyProprioSensor())
    msg = bus.poll()
    assert isinstance(msg, PerceptionMsg)
    assert msg.rgb is not None
    assert msg.proprioception is not None
    assert msg.proprioception.ee_pose[0] == 1.0
    bus.disconnect_all()
