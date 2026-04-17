"""Abstract base class for sensor channels."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PerceptionData:
    """Unified perception data packet from any sensor."""

    modality: str
    source_id: str
    timestamp: float
    payload: Any
    spatial_ref: Optional[str] = None
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


class SensorChannel(ABC):
    """
    Abstract sensor channel.

    Each sensor provides a `capture()` method that returns a PerceptionData.
    Sensors are designed to be pluggable: new sensors register via SensorRegistry.
    """

    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the sensor is online and ready to capture."""
        ...

    @abstractmethod
    def capture(self) -> PerceptionData:
        """Capture one frame/sample and return it as PerceptionData."""
        ...

    def calibrate(self) -> bool:
        """Optional calibration step. Default is no-op True."""
        return True
