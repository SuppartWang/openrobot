# RoboClaw 双项目深度分析：Agibot 论文版 vs SJTU 社区版

## 目录
1. [项目背景与定位](#1-项目背景与定位)
2. [核心原理对比](#2-核心原理对比)
3. [功能架构拆解](#3-功能架构拆解)
4. [实现状态与代码透视](#4-实现状态与代码透视)
5. [对 openrobot MVP 的借鉴建议](#5-对-openrobot-mvp-的借鉴建议)

---

## 1. 项目背景与定位

### 1.1 RoboClaw-Robotics/RoboClaw (Agibot 论文版)

| 属性 | 内容 |
|------|------|
| **仓库** | https://github.com/RoboClaw-Robotics/RoboClaw |
| **论文** | arXiv:2603.11558 — *RoboClaw: An Agentic Framework for Scalable Long-Horizon Robotic Tasks* |
| **团队** | AgiBot (中国) + National University of Singapore + SJTU |
| **定位** | **研究型 Agentic VLA 框架**，面向长程操作任务的"单一 VLM 闭环控制器" |
| **核心问题** | 传统机器人 pipeline 将数据收集、策略学习、部署割裂，导致长程任务中多策略执行脆弱、严重依赖人工重置环境 |

**一句话概括**：用同一个 VLM Agent 贯穿机器人策略的全生命周期，让部署经验也能回流训练。

### 1.2 MINT-SJTU/RoboClaw (SJTU 社区版)

| 属性 | 内容 |
|------|------|
| **仓库** | https://github.com/MINT-SJTU/RoboClaw |
| **团队** | 上海交通大学 MINT 实验室 |
| **定位** | **工程型 Embodied AI Assistant**，强调"本体/环境/任务"三迁移 |
| **核心问题** | 当机器人本体变了、传感器变了、环境变了、任务变了，系统、技能和记忆能不能快速迁移？ |

**一句话概括**：不是"机器人版 OpenClaw"，而是一个面向任意本体、任意环境、任意任务的零代码具身智能助手。

---

## 2. 核心原理对比

### 2.1 Agibot 版：Entangled Action Pairs (EAP)

这是 Agibot 版最具学术价值的创新点。

**问题**：传统数据采集需要人类在每次 rollout 失败后手动重置环境，导致数据收集效率极低。

**EAP 原理**：
- 为每一个 manipulation skill **配对两个 prompt**：
  - **Forward Prompt**：执行目标操作（如"把润肤露放到桌子上"）
  - **Reverse Prompt**：执行逆向恢复操作（如"把润肤露放回篮子"）
- Agent 自动交替执行 Forward → Reverse，形成**自重置循环 (Self-Resetting Loop)**
- 每次循环产生的正向+反向轨迹都会被持久化到数据集 `D` 中

**价值**：
1. **最小化人工干预**：环境可以在没有人类参与的情况下持续采集数据
2. **在线策略改进 (On-policy refinement)**：采集到的数据直接用于迭代训练 VLA 策略
3. **上下文语义一致性**：采集阶段和执行阶段使用同一个 Agent 和同一套 prompt，减少了"训练-部署"之间的语义错位

### 2.2 SJTU 版：自然语言驱动的本体 onboarding + Skill 分层

SJTU 版没有提出单一的算法创新，而是在**系统架构层面**做了第一性原理思考。

**核心洞察**：
1. RoboClaw 不是"带 Agent 的机器人框架"，而是"**会控制机器人的 Agent**"
2. **LeRobot 的数据+训练管线已经是业界最优**，重写是浪费 → 直接集成作为引擎
3. **安全不能靠 prompt 提醒**，必须是架构级强制 → Safety Gateway
4. **Agent（慢脑，LLM）和 Control Runtime（快脑，确定性）必须分离**

**零代码 onboarding 原理**：
- 用户通过**对话描述硬件**（如"我有 6 个舵机，相机在手腕上"）
- Agent 自动生成 **manifest**（本体能力声明）
- 系统自动探测串口、识别协议、生成 ROS2 adapter
- 引导用户完成校准后即可使用

**Skill 三层模型**：
- **Primitive**：本体绑定的原子动作（如 `move_joint`, `open_gripper`）
- **Skill**：本体无关的可组合任务（如 `pick_and_place`, `pour_water`）
- **Policy**：学习获得的可部署策略（如 ACT checkpoint, Diffusion checkpoint）

---

## 3. 功能架构拆解

### 3.1 Agibot 版架构：Skill-Centric Agent Loop

```
┌─────────────────────────────────────────────────────────────┐
│  User Interface (TUI / GUI)                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │ 自然语言指令
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  VLM Agent (高层控制器)                                      │
│  - 任务分解 (Subtask Planning)                               │
│  - 策略选择 (Skill Selection)                                │
│  - 执行监控 (Execution Monitoring)                           │
│  - 失败恢复 (Retry / Replan / Escalate)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │ 调用 Skill
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Skill Ecosystem                                             │
│  ┌────────────────────┐  ┌──────────────────────────────┐  │
│  │ eap-data-collection│  │ long-horizon-execution       │  │
│  │ (自重置数据采集)    │  │ (多步任务执行+重试+恢复)     │  │
│  └────────────────────┘  └──────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ monitored-subtask-execution                          │  │
│  │ (MCP Server 启动 / 轮询 / 停止 / 重置)               │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │ MCP Tool Call
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  MCP Server (corobot_mcp_server / 其他工具)                  │
│  - 启动 VLA Policy Server                                    │
│  - 发送关节/末端执行器指令                                    │
│  - 读取传感器反馈                                            │
└──────────────────────┬──────────────────────────────────────┘
                       │ ROS2 / 私有协议
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Agibot G01 真机                                             │
│  (双臂 + 移动底盘 + OmniPicker 夹爪 + RGB-D 相机)            │
└─────────────────────────────────────────────────────────────┘
```

**关键设计特点**：
- **Agent 是唯一的决策中心**：没有独立的"运动规划层"或"感知层"，所有高层决策由 VLM 完成
- **Skill 即策略**：通过 `SKILL.md` 文件定义 skill 的输入、输出、工作流和边界，Agent 根据用户意图选择并调用 skill
- **数据即副产品**：每次执行都会产生 JSONL 日志，EAP skill 会主动将轨迹写入 `dataset/episodes.jsonl`

### 3.2 SJTU 版架构：三引擎 + ROS2 统一总线

```
┌─────────────────────────────────────────────────────────────────┐
│  Interface Layer                                                │
│  CLI · Web UI · Discord · Telegram · WeChat                     │
└──────────────────────────────┬──────────────────────────────────┘
                               │ 消息/回复
┌──────────────────────────────▼──────────────────────────────────┐
│  Agent Runtime (慢脑 · 智能体运行时)                             │
│  ┌──────────┐ ┌──────────────┐ ┌────────────────┐              │
│  │ Memory   │ │ Tool / Spawn │ │ Lifecycle Mgr  │              │
│  │ 对话记忆 │ │ 子 Agent 派发 │ │ 阶段状态追踪   │              │
│  └──────────┘ └──────────────┘ └────────────────┘              │
│                    ▲                                           │
│                    └──────────────────────────────────┐        │
│  ┌──────────────────────────────────────────────────┐ │        │
│  │ Agent Loop (对话理解 → 意图分析 → 工具调用)      │─┘        │
│  └──────────────────────────────────────────────────┘          │
└──────────────────────────────┬──────────────────────────────────┘
                               │ 调用 Skill
┌──────────────────────────────▼──────────────────────────────────┐
│  Skill Ecosystem                                                │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────────────┐  │
│  │  Primitive   │ │    Skill     │ │       Policy           │  │
│  │ 本体绑定     │ │ 本体无关     │ │ 学习获得               │  │
│  └──────────────┘ └──────────────┘ └────────────────────────┘  │
│         Skill Hub: 社区共享 · 上传/下载/复用 · 自动适配         │
└──────┬───────────────────┬───────────────────────┬──────────────┘
       │                   │                       │
       ▼                   ▼                       ▼
┌═══════════════╗ ┌───────────────────┐ ┌───────────────────────┐
║ Embodiment    ║ │ Learning Engine   │ │ Perception Engine     │
║ Engine        ║ │ (LeRobot 驱动)    │ │ (标准 ML 库驱动)      │
║               ║ │                   │ │                       │
║ Control       ║ │ Data Collector    │ │ Camera Manager        │
║  Dispatch     ║ │ Dataset Manager   │ │ Detection (YOLO/DINO) │
║ Embodiment    ║ │ Policy Library    │ │ VLM 场景理解          │
║  Registry     ║ │ Train + Deploy    │ │ Spatial Memory        │
║ Safety        ║ │ gRPC 推理服务     │ │ (CLIP + 向量库)       │
║  Gateway      ║ │                   │ │                       │
╚═══════════════╝ └───────────┬───────┘ └───────────┬───────────┘
          │                   │                     │
          └───────────────────┼─────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│  ROS2 Transport (统一通信层)                                     │
│  /joint_commands /joint_states /camera/* /episode /safety       │
└──────────────────────┬──────────────────────────────┬───────────┘
                       │                              │
                       ▼                              ▼
┌─ Real World ─────────┐  ◀══ sim-to-real ══▶  ┌─ Sim World ─────────┐
│  真机 (Robot+Sensor) │    同一 ROS2 接口      │  MuJoCo + Web Viewer │
└──────────────────────┘                        └─────────────────────┘
                       ▲
                       │ 注入新本体
┌──────────────────────┴──────────────────────────────────────────┐
│  Embodiment Onboarding (零代码范式)                              │
│  对话描述硬件 → Agent 生成 manifest → 自动生成 adapter → 校准   │
└──────────────────────────────────────────────────────────────────┘
```

**关键设计特点**：
- **ROS2 是唯一的通信总线**：无论是真机还是仿真，都暴露完全相同的 topic 接口，策略可以零修改迁移
- **LeRobot 作为子引擎**：不重复造轮子，RoboClaw 只负责"驾驶舱"（对话式采集、调参、部署监督）
- **Safety Gateway 是架构级强制**：每一条到达硬件的指令都必须经过安全网关检查（关节限位、力矩、碰撞检测、急停）
- **快慢脑分离**：Agent Runtime（100ms+ 决策周期）与 Control Runtime（20ms 级实时控制）明确分离

---

## 4. 实现状态与代码透视

### 4.1 Agibot 版代码现状

**已开源内容**：
- `src/agent_demo/`: Agent 层演示代码骨架，包含 agent_layer, machine_layer, session_layer, interaction_layer
- `skills/`: 3 个核心 skill 的 `SKILL.md` 定义文档
  - `eap-data-collection/`: 详细的自重置数据采集流程
  - `long-horizon-execution/`: 长程任务分解与执行流程
  - `monitored-subtask-execution/`: 底层 MCP 子任务监控流程
- `src/mcp_server_demo/`: 多个 MCP server 示例（corobot, data_analyst, gma 等）

**未完全开源**：
- VLA 模型的实际部署代码（`.whl` 包需要私信索取）
- CoRobot 底层控制器的完整实现
- 论文中的具体训练脚本和超参数

**代码风格**：
- 以 **Markdown Skill 定义 + Agent 提示工程** 为核心，业务逻辑大量通过 prompt 和 tool-calling 约束实现
- 机器层有 `dataloader_a2d.py`, `dataloader_corobot.py` 等数据加载器，用于将真机轨迹转换为训练格式

### 4.2 SJTU 版代码现状

**已开源内容**：
- `roboclaw/`: Python 主包，包含：
  - `agent/`: LLM Agent 运行时
  - `bus/`: 内部消息总线
  - `channels/`: 多入口适配（Discord/Telegram/WeChat）
  - `cli/`: 命令行交互
  - `embodied/`: 具身控制相关（executor.py 是 LeRobot CLI 的 subprocess 封装）
  - `skills/`: 技能插件系统（目前多为非具身技能：github, weather, memory, summarize, tmux）
  - `session/`, `http/`, `security/`, `i18n/`: 通用基础设施
- `bridge/`: TypeScript 桥接层（Web UI 与后端通信）
- `ui/`: Web Dashboard（2026-04-11 刚发布）
- `docs/`: 极其详细的架构对比文档（包含与 LeRobot / dimos 的全面对比）

**实现深度**：
- 目前更多是**框架骨架和文档**，具身控制的核心逻辑（如 ROS2 adapter 生成、Safety Gateway、Perception Engine）还在早期开发中
- `roboclaw/embodied/executor.py` 的实际功能：通过 `asyncio.create_subprocess_exec` 调用 LeRobot CLI 命令，支持交互式、流式、后台 detached 运行

**开发文化**：
- 社区共建（Community Co-Creation），所有方向性决策通过 Discord / GitHub Issues 公开讨论
- 工程规范严格（`CLAUDE.md` 中要求：不用 try/except 吞错误、单文件不超过 1000 行、嵌套不超过 3 层、并行 sub-agent 开发）

---

## 5. 对 openrobot MVP 的借鉴建议

基于上述分析，以下是可以直接应用到 openrobot 5 层架构中的设计启示：

### 5.1 从 Agibot 版借鉴

| 设计点 | 应用到 openrobot 的建议 |
|--------|------------------------|
| **EAP (Entangled Action Pairs)** | 在 L4/L5 中引入"正向操作 + 逆向恢复"的 skill 配对概念。对于仿真中的重复数据采集（如抓取-放置循环），可以自动生成 reverse trajectory，减少人工重置。 |
| **Skill-Centric 架构** | openrobot 的 L4 `LLMAgent` 可以模仿 Skill 调用模式：将 `move_to`, `grasp`, `place` 等封装为带有明确输入输出契约的 Skill，Agent 通过选择 Skill 来生成计划，而不是直接输出原始动作。 |
| **统一的上下文语义** | 确保数据收集阶段和部署阶段使用同一套场景描述语言（如 `SceneGraph` 的上下文字符串），避免训练-部署语义错位。 |
| **JSONL 日志即数据集** | openrobot 的每次仿真执行都应自动记录 `episodes.jsonl`，包含 `obs`, `action`, `reward`, `skill_id`，为后续策略学习做准备。 |

### 5.2 从 SJTU 版借鉴

| 设计点 | 应用到 openrobot 的建议 |
|--------|------------------------|
| **ROS2 作为统一总线** | 当前 openrobot 使用自定义 Python 对象传递消息。如果未来要支持真机，应考虑尽早引入 ROS2 topic 作为 L2/L3 之间的标准通信层，实现 sim-to-real 零修改。 |
| **Safety Gateway 架构级安全** | 在 L3 `MujocoExecutor` 和 L1 `RobotMonitor` 之间引入显式的安全网关层，对每一条 `ActionCmd` 做关节限位、速度、碰撞预检查，而不是只在仿真层依赖 MuJoCo 的物理碰撞。 |
| **LeRobot 集成** | SJTU 版的战略非常清晰：不重复造训练轮子。openrobot 在 Milestone 2/3 中可以直接将 LeRobot 的 `datasets` + `policies` 作为子模块集成，专注于自己的 Agent 层和感知层。 |
| **Skill 三层模型** | 将 openrobot 当前混在一起的 "动作执行" 明确分层：Primitive (`ActionCmd`) → Skill (`pick_cube`) → Policy (VLA checkpoint)。这能让 L4 Agent 的 planning space 更清晰。 |
| **自然语言 Onboarding** | 虽然 openrobot 当前是 MVP，但可以在 L1/L2 中预留"本体 manifest"的抽象接口，让用户未来能通过 YAML/对话描述新硬件，而不是硬编码机械臂模型。 |

### 5.3 推荐的下一步行动

1. **引入 Skill 抽象层**：在 L4 和 L3 之间增加 `SkillRegistry`，将 `demo_end_to_end_agent.py` 中的硬编码动作解释器升级为可注册的 Skill 系统。
2. **自动数据记录**：修改 `MujocoExecutor`，在每次执行时自动将 `(rgb, proprioception, action)` 三元组追加到 `data/episodes.jsonl`。
3. **安全网关**：在 `MujocoExecutor.apply()` 前加入简单的边界检查（如关节范围、最大速度限制），作为 Safety Gateway 的 MVP 版本。
4. **评估 LeRobot 集成**：测试将 LeRobot 的 `SO101` 或 `Koch` 机器人配置与 openrobot 的 MuJoCo 场景对接的可行性。

---

*分析完成于 2026-04-15。数据来源：GitHub 源码、arXiv 论文、项目官网及社区文档。*
