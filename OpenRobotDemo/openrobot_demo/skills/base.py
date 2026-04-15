"""Base SkillInterface for OpenRobotDemo."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class SkillInterface(ABC):
    """Abstract base class for all skills in OpenRobotDemo."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique skill identifier."""
        ...

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the skill with given arguments and return a result dict.

        Every result dict should contain at least:
        - success: bool
        - message: str
        """
        ...
