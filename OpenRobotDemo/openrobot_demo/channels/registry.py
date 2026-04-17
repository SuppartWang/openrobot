"""Channel registry for OpenRobotDemo."""

from typing import Dict, Type, Callable
from .base import Channel

_registry: Dict[str, Type[Channel]] = {}


def register_channel(name: str, cls: Type[Channel]) -> None:
    """Register a channel class under a given name."""
    _registry[name] = cls


def create_channel(name: str, **kwargs) -> Channel:
    """Instantiate a registered channel by name."""
    if name not in _registry:
        raise KeyError(f"Channel '{name}' is not registered. Available: {list(_registry.keys())}")
    return _registry[name](**kwargs)


def list_channels() -> list:
    """Return a list of registered channel names."""
    return list(_registry.keys())
