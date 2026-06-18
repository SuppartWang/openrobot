# Skill Development Guide

Skills are the basic building blocks of OpenRobot. Each Skill implements a
reusable, self-describing unit of robot behavior.

## Skill Interface

All skills inherit from `SkillInterface` and define:

- `name`: a unique string identifier.
- `schema`: a `SkillSchema` describing parameters and return values.
- `execute(**kwargs)`: the actual behavior.

## Minimal Example

Create a new file `examples/my_skill.py`:

```python
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
```

## Register and Use the Skill

```python
from openrobot_demo.agent.skill_router import SkillRouter
from examples.my_skill import GreetSkill

router = SkillRouter()
router.register(GreetSkill())

result = router.execute_skill("greet", {"name": "OpenRobot"})
print(result)
```

## Best Practices

1. **Keep skills small and focused**: one skill should do one thing well.
2. **Validate inputs**: return `{"success": False, "message": "..."}` for bad
   arguments instead of raising exceptions.
3. **Use the schema**: the schema is used by the LLM planner to generate tool
   calls. Clear descriptions improve plan quality.
4. **Handle mock mode**: skills that wrap hardware should work (possibly with
   degraded behavior) when the hardware is unavailable.
5. **Clean up resources**: implement `close()` or `disable()` if your skill
   opens cameras, serial ports, or files.

## Testing Your Skill

Add a test to `OpenRobotDemo/tests/`:

```python
from examples.my_skill import GreetSkill


def test_greet_skill():
    skill = GreetSkill()
    result = skill.execute(name="World")
    assert result["success"]
    assert "Hello, World!" in result["message"]
```

Run it with:

```bash
cd OpenRobotDemo
pytest tests/test_my_skill.py -v
```
