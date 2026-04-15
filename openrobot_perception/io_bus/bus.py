"""PerceptionBus: aggregates multi-modal sensor data into a unified PerceptionMsg."""

import time
import logging
from typing import List, Dict, Any
from openrobot_msgs import PerceptionMsg, ProprioceptionState
from .interface import SensorInterface


logger = logging.getLogger(__name__)


class PerceptionBus:
    """Central aggregator for all L2 sensor data."""

    def __init__(self):
        self._sensors: List[SensorInterface] = []

    def attach(self, sensor: SensorInterface):
        sensor.connect()
        self._sensors.append(sensor)
        logger.info(f"[PerceptionBus] Attached sensor: {sensor.name}")

    def poll(self) -> PerceptionMsg:
        timestamp = time.time()
        kwargs: Dict[str, Any] = {"timestamp": timestamp}

        for sensor in self._sensors:
            try:
                data = sensor.read()
            except Exception as e:
                logger.warning(f"[PerceptionBus] Sensor {sensor.name} read failed: {e}")
                continue

            if sensor.name == "rgb_camera":
                kwargs["rgb"] = data
            elif sensor.name == "proprioception":
                kwargs["proprioception"] = ProprioceptionState(**data)
            elif sensor.name.startswith("touch"):
                if "touch" not in kwargs:
                    kwargs["touch"] = {}
                kwargs["touch"][sensor.name] = data
            elif sensor.name == "audio":
                kwargs["audio"] = data
            else:
                if "metadata" not in kwargs:
                    kwargs["metadata"] = {}
                kwargs["metadata"][sensor.name] = data

        return PerceptionMsg(**kwargs)

    def disconnect_all(self):
        for sensor in self._sensors:
            sensor.disconnect()
        self._sensors.clear()
