# OpenRobotDemo Quick Start

OpenRobotDemo is the fastest way to get started with OpenRobot. It provides a
self-contained simulation environment based on MuJoCo, so you can develop and
test skills without real robot hardware.

## What You Can Do

- Run a **minimal pick-and-place demo** in under a minute.
- Explore the **full-stack ReAct planning loop** with a simulated Franka arm.
- Develop new **Skills** and test them in simulation before deploying to real
  hardware.

## Prerequisites

- Python 3.10 or higher
- `git`
- (Optional) Docker and Docker Compose

## Installation

From the repository root, run the installer:

```bash
./install.sh
```

This will create a Python virtual environment at `.venv/` and install the
simulation dependencies.

If you also want the full perception/cognition stack, run:

```bash
INSTALL_FULL=1 ./install.sh
```

Then activate the environment:

```bash
source .venv/bin/activate
```

## Run the Minimal Demo

```bash
cd OpenRobotDemo
python examples/sim_pick_place.py
```

You should see output like:

```text
[1/6] Moving to pre-grasp...
[2/6] Moving to grasp...
[3/6] Closing gripper...
[4/6] Lifting...
[5/6] Moving to place...
[6/6] Opening gripper...
🖼️ Saved 22 frames to .../OpenRobotDemo/data/episodes/sim_pick_place
✅ Pick-and-place demo completed.
```

The generated frames are saved to `OpenRobotDemo/data/episodes/sim_pick_place/`.

## Run the Full-Stack ReAct Demo

The full-stack demo uses an LLM planner to generate a pick-and-place plan from
a natural-language instruction. It can run without API keys using a built-in
mock planner.

```bash
cd OpenRobotDemo
python scripts/demo_simulation_full_stack.py
```

To use a real LLM/VLM, configure your API keys in `OpenRobotDemo/.env`:

```bash
OPENAI_API_KEY=sk-...
# or
DASHSCOPE_API_KEY=sk-...
QWEN_API_KEY=sk-...
```

## Run with Docker

```bash
docker-compose up --build
```

This builds the `openrobot-sim` image and runs the minimal pick-and-place demo.
The generated data is mounted to `OpenRobotDemo/data/` on the host.

## Next Steps

- Read the [simulation guide](simulation.md) to understand the MuJoCo setup.
- Learn how to [write a custom Skill](skill_development.md).
- Check the [examples/](../examples/) directory for more snippets.
