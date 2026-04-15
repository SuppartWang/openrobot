# openrobot Architecture

## Overview

openrobot is a modular embodied AI robot software framework that maps human brain functions to robotic capabilities across 5 layers.

## Layer Mapping

| Layer | Brain Function | Robot Capability | Key Modules |
|-------|---------------|------------------|-------------|
| L1 | Vital Signs | System Monitoring | `RobotMonitor` |
| L2 | Perception | Multi-modal Sensing | `PerceptionBus`, `MujocoSensorSource` |
| L3 | Motor Control | Planning & Execution | `JointSpaceInterpolator`, `MujocoExecutor`, `CollisionGuard` |
| L4 | High-level Cognition | LLM Agent, Memory, Spatial Reasoning | `LLMAgent`, `RAGMemory`, `SceneGraph` |
| L5 | Emotion & Motivation | Reward & Task Scheduling | `RewardEngine`, `TaskScheduler` |

## Message Types

- `PerceptionMsg`: unified multi-modal sensor reading
- `ActionCmd`: low-level actuator command
- `CognitivePlan`: high-level task plan from LLM Agent

## Data Flow

1. **Sense**: sensors → `PerceptionBus` → `PerceptionMsg`
2. **Decide**: `LLMAgent` + `SceneGraph` + `RAGMemory` → `CognitivePlan`
3. **Plan**: `JointSpaceInterpolator` / heuristic IK → trajectory of `ActionCmd`
4. **Act**: `MujocoExecutor` → simulation / hardware
5. **Reflect**: `CollisionGuard` + `RewardEngine` → feedback to memory & scheduler

## Simulation

MuJoCo scene: `sim/mujoco/franka_rgb_scene.xml`
- Simplified 7-DOF arm with parallel gripper
- Wrist-mounted RGB camera
- Table and target cube for manipulation tasks

## Demos

- `scripts/demo_l1_l2_l3_closed_loop.py`: L1-L3 integration with sinusoidal control
- `scripts/demo_end_to_end_agent.py`: L1-L4 end-to-end mock-agent demo

## Running Tests

```bash
source .venv/bin/activate
PYTHONPATH=.:openrobot_core pytest tests/ -v
```
