"""Abstract base class for input channels."""

from abc import ABC, abstractmethod
from typing import Callable


class Channel(ABC):
    """
    Abstract input channel.

    A channel is responsible for:
    1. Receiving user instructions from a specific medium (CLI, HTTP, WebSocket, voice, etc.)
    2. Calling `on_message(instruction: str) -> str` when an instruction arrives
    3. Sending the response string back to the user (if applicable)
    """

    name: str = "base"

    @abstractmethod
    def start(self, on_message: Callable[[str], str]) -> None:
        """
        Start the channel.

        Args:
            on_message: Callback that receives the instruction string and returns a response.
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the channel and release resources."""
        ...
