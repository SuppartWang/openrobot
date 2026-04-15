"""Sensor interface abstraction for Layer 2."""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class SensorInterface(ABC):
    """Abstract base class for all sensors (vision, proprioception, audio, touch)."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def connect(self) -> bool:
        """Initialize the sensor."""
        ...

    @abstractmethod
    def read(self) -> Any:
        """Return the latest sensor reading."""
        ...

    @abstractmethod
    def disconnect(self):
        ...


class RGBCamera(SensorInterface):
    """Concrete RGB camera sensor wrapping a generic frame source."""

    def __init__(self, source: Any):
        self._source = source
        self._connected = False

    @property
    def name(self) -> str:
        return "rgb_camera"

    def connect(self) -> bool:
        self._connected = True
        return True

    def read(self) -> np.ndarray:
        if not self._connected:
            raise RuntimeError("Camera not connected")
        return self._source.read_rgb()

    def disconnect(self):
        self._connected = False


class ProprioceptionSensor(SensorInterface):
    """Concrete proprioception sensor wrapping a simulator or hardware API."""

    def __init__(self, source: Any):
        self._source = source
        self._connected = False

    @property
    def name(self) -> str:
        return "proprioception"

    def connect(self) -> bool:
        self._connected = True
        return True

    def read(self) -> dict:
        if not self._connected:
            raise RuntimeError("Proprioception sensor not connected")
        return self._source.read_proprioception()

    def disconnect(self):
        self._connected = False
