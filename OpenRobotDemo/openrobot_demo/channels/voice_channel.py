"""Voice input channel placeholder for OpenRobotDemo."""

import logging
from typing import Callable

from .base import Channel

logger = logging.getLogger(__name__)


class VoiceChannel(Channel):
    """
    Placeholder for voice-based instruction input.

    Future implementation should integrate:
    1. ASR (Automatic Speech Recognition) e.g. Whisper, 讯飞, Kimi STT
    2. VAD (Voice Activity Detection) to detect end of speech
    3. (Optional) TTS for spoken robot feedback
    """

    name = "voice"

    def start(self, on_message: Callable[[str], str]) -> None:
        logger.info(
            "[VoiceChannel] Placeholder started. "
            "Implement microphone + ASR loop here, then call on_message(transcript)."
        )
        print("\n🎤 Voice channel is a placeholder.")
        print("   To implement: start a microphone loop, run ASR, then call on_message(text).")

    def stop(self) -> None:
        logger.info("[VoiceChannel] Placeholder stopped.")
