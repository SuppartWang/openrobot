"""Sensor registry for OpenRobotDemo."""

from typing import Dict, Type

from .base import SensorChannel

_registry: Dict[str, Type[SensorChannel]] = {}


def register_sensor(name: str, cls: Type[SensorChannel]) -> None:
    """Register a sensor class under a given name."""
    _registry[name] = cls


def create_sensor(name: str, **kwargs) -> SensorChannel:
    """Instantiate a registered sensor by name."""
    if name not in _registry:
        raise KeyError(
            f"Sensor '{name}' is not registered. Available: {list(_registry.keys())}"
        )
    return _registry[name](**kwargs)


def list_sensors() -> list:
    """Return a list of registered sensor names."""
    return list(_registry.keys())
