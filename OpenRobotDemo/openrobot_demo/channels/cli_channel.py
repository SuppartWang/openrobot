"""CLI input channel for OpenRobotDemo."""

import logging
import threading
from typing import Callable

from .base import Channel

logger = logging.getLogger(__name__)


class CLIChannel(Channel):
    """Read instructions from standard input in a background thread."""

    name = "cli"

    def __init__(self, prompt: str = "🤖 Instruction> "):
        self.prompt = prompt
        self._running = False
        self._thread: threading.Thread | None = None
        self._on_message: Callable[[str], str] | None = None

    def start(self, on_message: Callable[[str], str]) -> None:
        self._on_message = on_message
        self._running = True
        self._thread = threading.Thread(target=self._input_loop, daemon=True)
        self._thread.start()
        logger.info("[CLIChannel] Started. Type instructions below.")

    def _input_loop(self) -> None:
        while self._running:
            try:
                instruction = input(self.prompt)
            except EOFError:
                logger.info("[CLIChannel] EOF received, stopping.")
                break
            if not instruction.strip():
                continue
            try:
                response = self._on_message(instruction.strip())
                print(f"   → {response}")
            except Exception as exc:
                logger.exception("[CLIChannel] Error handling instruction")
                print(f"   ❌ Error: {exc}")

    def stop(self) -> None:
        self._running = False
        logger.info("[CLIChannel] Stopped.")
