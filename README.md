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

```bash
# Create virtual environment (Python 3.12 recommended)
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run unit tests
PYTHONPATH=.:openrobot_core pytest tests/ -v

# Run L1+L2+L3 closed-loop demo
python scripts/demo_l1_l2_l3_closed_loop.py

# Run end-to-end agent demo (uses mock plan without OPENAI_API_KEY)
python scripts/demo_end_to_end_agent.py
```

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
