"""Audio sensor: microphone input for sound detection and ASR.

Real-world: USB microphone, microphone array (ReSpeaker), contact microphone.
"""

import time
import logging
from typing import Optional

import numpy as np

from .base import SensorChannel, PerceptionData

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except Exception:
    _SD_AVAILABLE = False


class AudioSensor(SensorChannel):
    """Capture audio samples from a microphone."""

    name = "audio"

    def __init__(
        self,
        source_id: str = "mic_default",
        sample_rate: int = 16000,
        channels: int = 1,
        duration_s: float = 1.0,
        device_id: Optional[int] = None,
    ):
        self.source_id = source_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.duration_s = duration_s
        self.device_id = device_id
        self._samples = int(sample_rate * duration_s)

    def is_available(self) -> bool:
        return _SD_AVAILABLE

    def capture(self) -> PerceptionData:
        if not _SD_AVAILABLE:
            # Mock: generate synthetic audio (silence with a test tone)
            t = np.linspace(0, self.duration_s, self._samples, endpoint=False)
            audio = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
            if self.channels > 1:
                audio = np.stack([audio] * self.channels, axis=-1)
            return PerceptionData(
                modality="audio",
                source_id=self.source_id,
                timestamp=time.time(),
                payload=audio,
                spatial_ref="mic_frame",
                metadata={
                    "sample_rate": self.sample_rate,
                    "channels": self.channels,
                    "duration_s": self.duration_s,
                    "mock": True,
                },
            )

        try:
            audio = sd.rec(
                frames=self._samples,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                device=self.device_id,
            )
            sd.wait()
            return PerceptionData(
                modality="audio",
                source_id=self.source_id,
                timestamp=time.time(),
                payload=audio,
                spatial_ref="mic_frame",
                metadata={
                    "sample_rate": self.sample_rate,
                    "channels": self.channels,
                    "duration_s": self.duration_s,
                    "mock": False,
                },
            )
        except Exception as exc:
            logger.error("[AudioSensor] Capture failed: %s", exc)
            raise RuntimeError(f"Audio capture failed: {exc}")
