"""Ultrasonic rangefinder sensor.

Real-world: HC-SR04, SRF05, etc.
Simulation: approximate with ray-casting or mock data.
"""

import time
import logging
from typing import Optional

import numpy as np

from .base import SensorChannel, PerceptionData

logger = logging.getLogger(__name__)


class UltrasonicSensor(SensorChannel):
    """Ultrasonic distance sensor."""

    name = "ultrasonic"

    def __init__(
        self,
        source_id: str = "ultrasonic_default",
        max_range: float = 4.0,
        min_range: float = 0.02,
        # Mock
        mock_distance: Optional[float] = None,
    ):
        self.source_id = source_id
        self.max_range = max_range
        self.min_range = min_range
        self._mock_distance = mock_distance

    def is_available(self) -> bool:
        return True

    def capture(self) -> PerceptionData:
        if self._mock_distance is not None:
            distance = self._mock_distance
        else:
            # Random mock distance with occasional max_range (no echo)
            distance = np.random.uniform(self.min_range, self.max_range * 0.8)
            if np.random.random() < 0.05:
                distance = self.max_range  # no echo

        return PerceptionData(
            modality="ultrasonic",
            source_id=self.source_id,
            timestamp=time.time(),
            payload={"distance_m": float(distance)},
            spatial_ref="sensor_frame",
            metadata={
                "max_range": self.max_range,
                "min_range": self.min_range,
                "mock": True,
            },
        )
