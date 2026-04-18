"""Base SkillInterface for OpenRobotDemo with schema declarations.

Every skill MUST declare:
- input_schema:  what parameters it accepts (name, type, description, required, default)
- output_schema: what fields it returns (name, type, description)
- dependencies:  what sensors / hardware it requires
- preconditions: when can this skill be called (optional)
- postconditions: how does the world change after execution (optional)

This allows the LLM planner to:
1. Auto-generate tool descriptions from code (no hard-coded prompts)
2. Validate arguments before execution
3. Understand data flow between skills
4. Discover skills dynamically without manual registration
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable


@dataclass
class ParamSchema:
    """Schema for a single input parameter."""

    name: str
    type: str  # "str", "int", "float", "bool", "list", "ndarray", "dict", "any"
    description: str
    required: bool = True
    default: Any = None
    example: Any = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "required": self.required,
        }
        if self.default is not None:
            d["default"] = self.default
        if self.example is not None:
            d["example"] = self.example
        return d


@dataclass
class ResultSchema:
    """Schema for a single output field."""

    name: str
    type: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
        }


@dataclass
class SkillSchema:
    """Complete schema description of a skill."""

    description: str = ""
    parameters: List[ParamSchema] = field(default_factory=list)
    returns: List[ResultSchema] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    """List of required sensor/hardware IDs, e.g. ['camera', 'arm']."""

    preconditions: List[str] = field(default_factory=list)
    """Natural language preconditions, e.g. ['gripper must be open']."""

    postconditions: List[str] = field(default_factory=list)
    """Natural language postconditions, e.g. ['object is grasped']."""

    examples: List[Dict[str, Any]] = field(default_factory=list)
    """Few-shot examples for the LLM: [{"input": {...}, "output": {...}}]"""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "returns": [r.to_dict() for r in self.returns],
            "dependencies": self.dependencies,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "examples": self.examples,
        }


class SkillInterface(ABC):
    """Abstract base class for all skills in OpenRobotDemo.

    Every concrete skill must:
    1. Define `name` property
    2. Define `schema` property
    3. Implement `execute(**kwargs)`
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique skill identifier."""
        ...

    @property
    def schema(self) -> SkillSchema:
        """Return the schema description of this skill.

        Override this in subclasses to declare parameters, returns, dependencies.
        Default returns an empty schema.
        """
        return SkillSchema(
            description=f"Skill '{self.name}' (no schema declared)",
        )

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the skill with given arguments and return a result dict.

        Every result dict should contain at least:
        - success: bool
        - message: str
        """
        ...

    def validate_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fill defaults for input arguments based on schema.

        Returns validated args dict, or raises ValueError on violation.
        """
        validated = {}
        schema = self.schema

        for param in schema.parameters:
            if param.name in args:
                validated[param.name] = args[param.name]
            elif param.required and param.default is None:
                raise ValueError(
                    f"Skill '{self.name}': missing required parameter '{param.name}'"
                )
            else:
                validated[param.name] = param.default

        # Warn about unknown parameters
        known = {p.name for p in schema.parameters}
        for key in args:
            if key not in known and key != "self":
                # Allow unknown kwargs but warn
                validated[key] = args[key]

        return validated

    def to_tool_description(self) -> Dict[str, Any]:
        """Generate an OpenAI-compatible function/tool description from schema.

        Returns a dict suitable for:
            client.chat.completions.create(
                model="...",
                messages=[...],
                tools=[skill.to_tool_description()],
                tool_choice="auto",
            )
        """
        schema = self.schema
        properties = {}
        required = []

        for param in schema.parameters:
            prop = {"type": _python_type_to_json_type(param.type), "description": param.description}
            if param.example is not None:
                prop["example"] = param.example
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": schema.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def get_state_change_description(self) -> str:
        """Return a natural language description of how this skill changes world state.

        Used by the planner for causal reasoning.
        """
        schema = self.schema
        lines = [f"Skill: {self.name}"]
        if schema.preconditions:
            lines.append(f"  Preconditions: {'; '.join(schema.preconditions)}")
        if schema.postconditions:
            lines.append(f"  Postconditions: {'; '.join(schema.postconditions)}")
        return "\n".join(lines)


def _python_type_to_json_type(t: str) -> str:
    """Map our internal type names to JSON Schema types."""
    mapping = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
        "ndarray": "array",
        "any": "string",  # fallback
    }
    return mapping.get(t.lower(), "string")
