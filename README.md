# openrobot MVP

A modular embodied AI robot software framework mapping human brain functions to robotic capabilities.

## Core Idea

Human brain functions → Embodied robot必备功能

1. **基础生命调控 (L1)** → System monitoring (`RobotMonitor`)
2. **感知加工 (L2)** → Multi-modal perception (`PerceptionBus`, RGB, proprioception)
3. **运控执行 (L3)** → Motion planning & execution (`JointSpaceInterpolator`, `MujocoExecutor`, `CollisionGuard`)
4. **高级认知 (L4)** → LLM Agent, RAG memory, spatial reasoning (`LLMAgent`, `RAGMemory`, `SceneGraph`)
5. **情绪动机 (L5)** → Reward engine & task scheduling (`RewardEngine`, `TaskScheduler`)

## Quick Start

The fastest way to try OpenRobot is the **OpenRobotDemo simulation stack**.
It runs entirely on your laptop with MuJoCo and does not require robot hardware
or cloud API keys.

```bash
# 1. Install dependencies (creates .venv automatically)
./install.sh

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Run the minimal pick-and-place demo
cd OpenRobotDemo
python examples/sim_pick_place.py

# 4. (Optional) Run the full-stack ReAct demo
#    Requires LLM/VLM API keys configured in OpenRobotDemo/.env
python scripts/demo_simulation_full_stack.py
```

You can also use Docker:

```bash
docker-compose up --build
```

For more details, see [OpenRobotDemo/docs/quickstart.md](OpenRobotDemo/docs/quickstart.md).

## Architecture

See [docs/architecture.md](docs/architecture.md) for detailed design.

## Project Structure

```
openrobot/
├── openrobot_core/         # L1: monitoring & message types
├── openrobot_perception/   # L2: vision, proprioception, io_bus
├── openrobot_control/      # L3: motion planning, reflexes, execution
├── openrobot_cognition/    # L4: agent, memory, spatial reasoning
├── openrobot_motivation/   # L5: reward, task scheduling
├── sim/mujoco/             # MuJoCo simulation assets & demos
├── scripts/                # Integration demos
├── tests/                  # Unit tests
└── docs/                   # Documentation
```

## Tech Stack

- **Simulation**: MuJoCo 3.x
- **Perception**: OpenCV, Transformers (VLA/CLIP ready)
- **Cognition**: OpenAI API / local LLM, ChromaDB (RAG)
- **Language**: Python 3.12+
