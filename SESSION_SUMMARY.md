# OpenRobotDemo Session 总结文档

> **Session 时间**: 2026-04-16 ~ 2026-04-18  
> **总提交数**: 6 次  
> **新增文件**: 21 个  
> **修改文件**: 18 个  
> **净增代码行**: +5,530 / -283  
> **最终提交**: `a337b0e`  
> **仓库**: `https://github.com/SuppartWang/openrobot`

---

## 目录

1. [原始需求](#1-原始需求)
2. [架构决策记录 ADR](#2-架构决策记录-adr)
3. [模块变更清单](#3-模块变更清单)
4. [核心接口定义](#4-核心接口定义)
5. [当前架构全景](#5-当前架构全景)
6. [待办事项与已知问题](#6-待办事项与已知问题)
7. [新设备恢复指南](#7-新设备恢复指南)

---

## 1. 原始需求

> 一个以大语言模型为智慧基座，以 harness/agent/提示词工程为规划平台，整合所有规划经验，所有可以作为真实世界信号输入的数据形式，兼容控制所有真实世界可以控制运动的机械结构，以及处理输入信号得到输出信号的算法，将信号输入和运动控制和算法以 skill 形式保存，供规划平台调用的架构。

**7 条需求维度**:

| # | 需求 | 完成度 |
|---|---|---|
| 1 | 大语言模型为智慧基座 | 85% |
| 2 | harness/agent/提示词工程为规划平台 | 85% |
| 3 | 整合所有规划经验 | 90% |
| 4 | 所有可以作为真实世界信号输入的数据形式 | 70% |
| 5 | 兼容控制所有真实世界可以控制运动的机械结构 | 55% |
| 6 | 处理输入信号得到输出信号的算法以 skill 形式保存 | 70% |
| 7 | 供规划平台调用 | 85% |
| **平均** | | **~78%** |

---

## 2. 架构决策记录 (ADR)

### ADR-001: 放弃三元组查表，采用参数化 Experience

**决策**: 经验存储不采用 `(gripper, object, action)` 三元组，而是完整的 `Experience` 对象（Context + Parametric Policy + Outcome）。

**理由**: 三元组无法表达连续参数空间、无法泛化、无法积累反馈。参数化 Experience 支持模糊匹配、人工反馈、自动优化。

**实现**: `experience/schema.py` + `experience/library.py` + `experience/retriever.py`（三级检索）。

### ADR-002: 6-DOF 关节与夹爪完全分离

**决策**: YHRG S1 的 `get_pos()` 返回 7 个值 `[j1..j6, gripper]`，但 `joint_control()` 只处理前 6 个，夹爪通过独立的 `control_gripper()` 控制。

**理由**: 避免 Franka 7-DOF 代码与 S1 6-DOF 硬件的冲突。这是多次 bug 修复后沉淀的不变式。

**实现**: `hardware/yhrg_adapter.py` 中 Mock 和 Real 模式均遵守此约定。

### ADR-003: Skill 自描述（Schema 声明）

**决策**: 每个 Skill 必须声明输入/输出 schema，Prompt 中的工具描述从代码自动生成，不手写。

**理由**: 消除 prompt 与代码的同步成本。LLM 自动理解 Skill 能力。新增 Skill 无需修改 prompt。

**实现**: `skills/base.py` 中 `SkillInterface.schema` + `to_tool_description()` + `validate_args()`。

### ADR-004: VLM 是信号处理器，不是决策器

**决策**: VLM（多模态大模型）只负责将图像转换为结构化认知参数（物体、关系、异常），不做任何动作决策。

**理由**: 分离感知与决策，使 VLM 可替换、可测试、可独立优化。决策权完全属于 BDI Agent。

**实现**: `perception/vlm_cognition.py` 输出 `PerceptionData(modality="vlm_cognition")`。

### ADR-005: BDI Agent 包装 LLMPlanner，不替代

**决策**: Agent 负责宏观目标分解和失败恢复，LLMPlanner 负责微观步骤生成。

**理由**: 保留现有 ReAct 循环投资，在其上叠加自主分解能力。两层规划避免单一层级的复杂性爆炸。

**实现**: `agent/agent.py` 中 `BDIAgent.execute()` 内部调用 `planner.plan()` 和 `skill_router.execute_single()`。

### ADR-006: 通用 RobotInterface ABC

**决策**: 所有运动结构（机械臂、底盘、无人机等）通过 `RobotInterface` 统一抽象，而不是各自独立的适配器。

**理由**: 解锁"兼容所有真实世界运动结构"的需求。新增机械结构只需实现 `command()` + `observe()`。

**实现**: `hardware/robot_interface.py`（通用）+ `hardware/manipulator_interface.py`（机械臂扩展）。

---

## 3. 模块变更清单

### 3.1 新增文件（21 个）

| # | 文件路径 | 说明 | 行数 |
|---|---|---|---|
| 1 | `hardware/robot_interface.py` | 通用机器人接口 ABC（Action/Observation/Space） | 233 |
| 2 | `hardware/manipulator_interface.py` | 机械臂扩展接口（FK/IK/关节/笛卡尔/夹爪） | 236 |
| 3 | `runtime/harness.py` | 编排引擎（Sequence/Parallel/Condition/Loop/Pipeline/TryCatch） | 493 |
| 4 | `sensors/imu.py` | IMU 传感器（加速度+陀螺仪+四元数） | 72 |
| 5 | `sensors/wrench.py` | 六维力/力矩传感器 | 90 |
| 6 | `sensors/audio.py` | 音频/麦克风传感器（sounddevice + mock） | 91 |
| 7 | `sensors/lidar.py` | 激光雷达传感器（RPLidar + mock） | 122 |
| 8 | `sensors/ultrasonic.py` | 超声波测距传感器 | 59 |
| 9 | `sensors/odometry.py` | 轮式里程计传感器 | 54 |
| 10 | `skills/signal_processing.py` | 信号处理算法 Skill（低通滤波、卡尔曼、FFT） | 160 |
| 11 | `skills/pointcloud_processing.py` | 点云处理 Skill（RANSAC、聚类、去噪） | 226 |
| 12 | `skills/vision_processing.py` | 视觉处理 Skill（颜色检测、ORB 特征） | 130 |
| 13 | `skills/motion_planning.py` | 运动规划 Skill（直线规划、关节空间规划） | 134 |
| 14 | `perception/vlm_cognition.py` | VLM 认知引擎（场景/物体/关系/异常/ affordance） | 283 |
| 15 | `agent/bdi.py` | BDI 核心模型（Goal/BeliefSet/Desire/Intent/Reflection） | 258 |
| 16 | `agent/decomposer.py` | 任务分解器（LLM + 规则 fallback） | 208 |
| 17 | `agent/self_reflector.py` | 自我反思器（经验库 + LLM 因果分析） | 210 |
| 18 | `agent/agent.py` | BDIAgent 顶层协调器 | 367 |
| 19 | `ARCHITECTURE.md` | 架构全景文档 | 513 |
| 20 | `ARCHITECTURE_GAP_ANALYSIS.md` | 架构缺口分析文档 | 268 |
| 21 | `SESSION_SUMMARY.md` | 本文件 | — |

### 3.2 修改文件（18 个）

| # | 文件路径 | 主要变更 |
|---|---|---|
| 1 | `agent/planner.py` | 集成 PromptEngine + 经验注入 + function-calling + skill_router |
| 2 | `agent/skill_router.py` | schema 校验 + 自动生成工具描述 + validate_plan |
| 3 | `agent/__init__.py` | 导出 BDI 相关类 |
| 4 | `hardware/__init__.py` | 导出 RobotInterface/ManipulatorInterface |
| 5 | `hardware/yhrg_adapter.py` | 实现 ManipulatorInterface |
| 6 | `hardware/mujoco_franka_adapter.py` | 实现 ManipulatorInterface |
| 7 | `skills/base.py` | 新增 SkillSchema/ParamSchema/ResultSchema |
| 8 | `skills/__init__.py` | 导出所有算法 Skill |
| 9 | `skills/camera_capture.py` | 添加 schema |
| 10 | `skills/arm_state_reader.py` | 添加 schema |
| 11 | `skills/vision_3d_estimator.py` | 添加 schema |
| 12 | `skills/grasp_predictor.py` | 添加 schema |
| 13 | `skills/arm_executor.py` | 添加 schema + 示例 |
| 14 | `skills/vla_policy_executor.py` | 工厂模式 + 动态导入 + schema |
| 15 | `dual_arm/fabric_skills.py` | 添加 schema |
| 16 | `sensors/__init__.py` | 导出 6 个新传感器 |
| 17 | `world_model/model.py` | 新增 7 种 modality 的 ingest 处理 |
| 18 | `scripts/demo_fabric_dual_arm.py` | 替换为 BDIAgent 执行 |

---

## 4. 核心接口定义

### 4.1 RobotInterface（通用机器人抽象）

```python
class RobotInterface(ABC):
    @property @abstractmethod def robot_type(self) -> str: ...      # "manipulator"|"mobile_base"|...
    @property @abstractmethod def robot_id(self) -> str: ...
    @property @abstractmethod def dof(self) -> int: ...
    @property @abstractmethod def action_space(self) -> Dict[str, Space]: ...
    @property @abstractmethod def observation_space(self) -> Dict[str, Space]: ...
    @abstractmethod def command(self, action: Action) -> bool: ...
    @abstractmethod def observe(self) -> Observation: ...
    @abstractmethod def enable(self) -> bool: ...
    @abstractmethod def disable(self) -> bool: ...
    @abstractmethod def reset(self) -> bool: ...
    @abstractmethod def is_ready(self) -> bool: ...
```

**已实现**: YHRGAdapter, FrankaMujocoAdapter  
**待实现**: MobileBaseAdapter, QuadrupedAdapter, AerialAdapter, ...

### 4.2 SkillInterface（带 Schema 自描述）

```python
class SkillInterface(ABC):
    @property @abstractmethod def name(self) -> str: ...
    @property def schema(self) -> SkillSchema: ...
    @abstractmethod def execute(self, **kwargs) -> Dict[str, Any]: ...
    def validate_args(self, args: Dict) -> Dict: ...
    def to_tool_description(self) -> Dict: ...  # OpenAI function schema
```

**已注册 Skill（16 个）**:
- `camera_capture`, `arm_state_reader`, `vision_3d_estimator`, `grasp_point_predictor`
- `arm_motion_executor`, `vla_policy_executor`, `fabric_manipulation`
- `low_pass_filter`, `kalman_filter_1d`, `fft_analysis`
- `ransac_plane_segmentation`, `euclidean_clustering`, `statistical_outlier_removal`
- `color_detector`, `feature_extractor`
- `straight_line_planner`, `joint_space_planner`

### 4.3 SensorChannel（可插拔感知）

```python
class SensorChannel(ABC):
    name: str
    @abstractmethod def is_available(self) -> bool: ...
    @abstractmethod def capture(self) -> PerceptionData: ...
    def calibrate(self) -> bool: ...
```

**已实现 Sensor（14 个）**:
- Vision: `VisionRGBSensor`, `VisionDepthSensor`, `PointCloudSensor`
- RealSense: `RealSenseRGBSensor`, `RealSenseDepthSensor`, `RealSenseVLMSensor`
- Proprioception/Tactile: `ProprioceptionSensor`, `TactileSensor`
- New (P1-2): `IMUSensor`, `WrenchSensor`, `AudioSensor`, `LidarSensor`, `UltrasonicSensor`, `OdometrySensor`
- Cognition: `VLMCognitionSensor`

### 4.4 BDI Agent 核心类

```python
# 数据结构
@dataclass class Goal:           description, sub_goals, preconditions, completion_criteria, status
@dataclass class Intent:         goal_id, plan_steps, current_step_idx, status
class BeliefSet:                  add/get/query beliefs, sync from WorldModel
@dataclass class Reflection:     analysis, should_retry, adjusted_params, should_replan

# 组件
class TaskDecomposer:            decompose(instruction) -> Goal tree
class SelfReflector:             reflect(failure) -> Reflection
class BDIAgent:                  execute(instruction, sensors) -> summary
```

---

## 5. 当前架构全景

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT CHANNELS                                  │
│  CLI | HTTP | Voice                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RUNTIME LAYER                                   │
│  RobotQueue (FIFO + retry)  │  HarnessRunner (Sequence/Parallel/Condition)  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AGENT LAYER (BDI)                                  │
│                                                                              │
│   BDIAgent.execute()                                                         │
│   ├── TaskDecomposer → Goal Tree                                             │
│   ├── BeliefSet ← WorldModel                                                 │
│   ├── select_next_intent() → Intent (plan_steps)                             │
│   ├── planner.plan() ← LLM + 经验 + Skill schemas                            │
│   ├── SkillRouter.execute_single()                                           │
│   └── SelfReflector.reflect() ← 失败时自动恢复                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SKILL LAYER (16)                                │
│  Perception: camera, vision_3d_estimator, color_detector, feature_extractor │
│  Motion: arm_motion_executor, straight_line_planner, joint_space_planner    │
│  Manipulation: grasp_point_predictor, fabric_manipulation                   │
│  Signal: low_pass_filter, kalman_filter_1d, fft_analysis                    │
│  PointCloud: ransac, euclidean_clustering, statistical_outlier_removal      │
│  Policy: vla_policy_executor                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PERCEPTION LAYER (15 Sensors)                        │
│  Vision: RGB, Depth, PointCloud                                              │
│  RealSense: RGB, Depth, VLM-detection                                        │
│  Proprioception/Tactile                                                      │
│  New: IMU, Wrench, Audio, LiDAR, Ultrasonic, Odometry                        │
│  Cognition: VLMCognitionSensor (VLM → structured perception)                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         WORLD MODEL (Unified Memory)                         │
│  RobotState │ ObjectDesc (+ relations, affordances) │ SpatialMemory         │
│  TaskMemory │ ingest(PerceptionData) → auto-update                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                       EXPERIENCE SYSTEM (Learning)                           │
│  Experience (Context→Policy→Outcome) → Library (SQLite)                      │
│  Retriever: exact → fuzzy → fallback                                         │
│  Recorder: human_demo | autonomous_trial | vla_inference                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HARDWARE LAYER                                       │
│  RobotInterface ABC                                                          │
│  ├── ManipulatorInterface → YHRGAdapter (S1) │ FrankaMujocoAdapter          │
│  └── (TODO) MobileBase │ Quadruped │ Aerial │ ...                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. 待办事项与已知问题

### 6.1 P2 待办（近期）

| # | 事项 | 优先级 | 说明 |
|---|---|---|---|
| 1 | **真机部署验证** | 🔴 高 | Ubuntu 22.04 + S1_SDK_V2 + pyrealsense2 下测试 `--mode real` |
| 2 | **VLA 模型接入** | 🔴 高 | 通过 `VLA_MODEL_CLASS` 或 `set_model_factory()` 接入 pi05 |
| 3 | **预置经验参数调优** | 🟡 中 | `seed.py` 中的布料直径、支撑板高度、插入深度等根据真机实测调整 |
| 4 | **多模态信号增强** | 🟡 中 | VLM 同时处理音频、力矩信号，输出跨模态认知 |
| 5 | **BDI 真机验证** | 🟡 中 | 在真实硬件上跑通完整的 BDI 分解→执行→反思循环 |

### 6.2 P3 待办（远期）

| # | 事项 | 说明 |
|---|---|---|
| 6 | 轮式底盘适配器 | `RobotInterface` 已抽象好，只需实现 |
| 7 | 四足/无人机适配器 | 同上 |
| 8 | 事件相机、热成像 | 新增 SensorChannel |
| 9 | SLAM Skill | ORB-SLAM3、LIO-SAM 接入 |
| 10 | 力控制 Skill | 导纳/阻抗控制 |
| 11 | 强化学习 Skill | PPO、SAC、Diffusion Policy |

### 6.3 已知问题

| # | 问题 | 影响 | 临时方案 |
|---|---|---|---|
| 1 | MuJoCo 长时间运行 NaN | 模拟 demo | 检测到 NaN 后自动 reset 到 home pose |
| 2 | LLM API 401 / 过期 | 离线开发 | mock fallback 自动接管 |
| 3 | pyrealsense2 macOS 不可用 | macOS 开发 | mock sensor fallback |
| 4 | VLM 输出偶发 JSON 解析失败 | 认知准确性 | 解析失败时返回 mock cognition + 记录 parse_error |
| 5 | BDI Agent 规则分解覆盖面有限 | 新任务类型 | 需要接入 LLM decomposition（需 API key） |

---

## 7. 新设备恢复指南

### 7.1 克隆仓库

```bash
git clone https://github.com/SuppartWang/openrobot.git
cd openrobot
```

### 7.2 检查提交状态

```bash
git log --oneline -6
# 应显示:
# a337b0e feat: BDI Agent + VLM Cognition Engine
# a7c9637 feat: P0+P1 architecture gap closure
# 90e43f9 docs: add architecture gap analysis
# 45d7875 docs: add comprehensive architecture overview
# 240041c feat: OpenRobotDemo v2.0
# 0d43e9f Initial commit
```

### 7.3 环境准备

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r OpenRobotDemo/requirements.txt

# 安装可选依赖（根据平台）
# macOS 开发:
pip install opencv-python numpy scipy torch transformers openai

# Linux 真机:
pip install opencv-python numpy scipy torch transformers openai pyrealsense2

# 如果需要声音传感器:
pip install sounddevice

# 如果需要 RPLidar:
pip install rplidar
```

### 7.4 配置 API Key（可选，mock 模式可跳过）

```bash
# 创建 .env 文件
cat > .env << 'EOF'
KIMI_API_KEY=your_kimi_key
DASHSCOPE_API_KEY=your_dashscope_key
QWEN_API_KEY=your_qwen_key
EOF
```

### 7.5 运行 Demo

```bash
# 模拟模式（macOS 可用）
cd OpenRobotDemo
python scripts/demo_fabric_dual_arm.py --mode mock

# 真机模式（Linux + S1_SDK + RealSense）
python scripts/demo_fabric_dual_arm.py --mode real \
    --left-dev /dev/ttyUSB0 --right-dev /dev/ttyUSB1

# MuJoCo 模拟（单臂）
python scripts/demo_simulation_full_stack.py
```

### 7.6 项目结构速览

```
openrobot/
├── ARCHITECTURE.md              # 架构全景文档
├── ARCHITECTURE_GAP_ANALYSIS.md # 缺口分析
├── SESSION_SUMMARY.md           # 本文件
├── OpenRobotDemo/
│   ├── openrobot_demo/
│   │   ├── agent/               # BDI Agent + Planner + SkillRouter
│   │   ├── channels/            # CLI/HTTP/Voice 输入
│   │   ├── control/             # SafetyGateway + Interpolator
│   │   ├── dual_arm/            # DualArmController + FabricSkill
│   │   ├── experience/          # ExperienceLibrary + Retriever + Seed
│   │   ├── hardware/            # RobotInterface + YHRGAdapter + FrankaAdapter
│   │   ├── perception/          # CameraDriver + VLMCognitionEngine
│   │   ├── persistence/         # RobotDatabase + EpisodeRecorder
│   │   ├── runtime/             # RobotQueue + HarnessRunner
│   │   ├── sensors/             # 15 个 SensorChannel
│   │   ├── skills/              # 16 个 SkillInterface
│   │   ├── world_model/         # WorldModel + ObjectDesc
│   │   └── utils/
│   ├── scripts/                 # Demo 入口
│   ├── configs/                 # 配置文件
│   ├── data/                    # SQLite 数据库 + 日志
│   └── requirements.txt
└── sim/                         # MuJoCo 模拟场景
```

---

## 附录：Git 提交完整记录

```
a337b0e  2026-04-18 20:35  feat: BDI Agent + VLM Cognition Engine
a7c9637  2026-04-18 20:19  feat: P0+P1 architecture gap closure
90e43f9  2026-04-18 18:53  docs: architecture gap analysis
45d7875  2026-04-18 16:26  docs: architecture overview
240041c  2026-04-18 01:13  feat: OpenRobotDemo v2.0
0d43e9f  2026-04-16 00:03  Initial commit
```

---

> **文档生成时间**: 2026-04-18  
> **Session ID**: `aquaman-x-23-blade`  
> **生成者**: Kimi Code CLI
