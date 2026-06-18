"""Template for a custom OpenRobot Skill.

This skill does not interact with hardware; it demonstrates the required
interface and schema. Use it as a starting point for your own skills.
"""

import os
import sys

_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(os.path.dirname(_project_root), "openrobot_core"))

from openrobot_demo.skills.base import (
    SkillInterface,
    SkillSchema,
    ParamSchema,
    ResultSchema,
)


class GreetSkill(SkillInterface):
    @property
    def name(self) -> str:
        return "greet"

    @property
    def schema(self) -> SkillSchema:
        return SkillSchema(
            description="Print a greeting message.",
            parameters=[
                ParamSchema(
                    name="name",
                    type="str",
                    description="Name to greet.",
                    required=True,
                    example="OpenRobot",
                ),
            ],
            returns=[
                ResultSchema(name="success", type="bool", description="Whether the greeting succeeded."),
                ResultSchema(name="message", type="str", description="The greeting message."),
            ],
        )

    def execute(self, name: str, **kwargs):
        message = f"Hello, {name}!"
        print(message)
        return {"success": True, "message": message}


if __name__ == "__main__":
    skill = GreetSkill()
    print(skill.schema.to_dict())
    print(skill.execute(name="OpenRobot"))
