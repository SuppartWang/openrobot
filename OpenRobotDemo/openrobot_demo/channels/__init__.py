"""Input channels for OpenRobotDemo: CLI, HTTP, Voice, etc."""

from .base import Channel
from .registry import register_channel, create_channel, list_channels
from .cli_channel import CLIChannel
from .http_channel import HTTPChannel
from .voice_channel import VoiceChannel

__all__ = [
    "Channel",
    "register_channel",
    "create_channel",
    "list_channels",
    "CLIChannel",
    "HTTPChannel",
    "VoiceChannel",
]
