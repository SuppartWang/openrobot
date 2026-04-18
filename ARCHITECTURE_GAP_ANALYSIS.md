# OpenRobotDemo 架构缺口分析

> 对照用户原始需求逐条核对，标注完成度与缺失项

---

## 需求原文

> 一个以大语言模型为智慧基座，以 harness/agent/提示词工程为规划平台，整合所有规划经验，所有可以作为真实世界信号输入的数据形式，兼容控制所有真实世界可以控制运动的机械结构，以及处理输入信号得到输出信号的算法，将信号输入和运动控制和算法以 skill 形式保存，供规划平台调用的架构。

---

## 逐条核对

### 1. 大语言模型为智慧基座 ✅/🟡

| 维度 | 现状 | 评级 | 差距说明 |
|---|---|---|---|
| **LLM 作为决策核心** | `LLMPlanner` 存在 ReAct 循环 | 🟡 | API 不可用时自动 fallback 到 mock plan，LLM 不是**强制**基座 |
| **LLM 理解 sensor 数据** | `state_summary` 以中文文本形式注入 prompt | 🟡 | 原始 sensor 数据（RGB、点云、力矩）未直接送入 LLM，仅用人工总结的文本摘要 |
| **LLM 动态调整策略** | 无 | ❌ | 失败时不能根据错误类型自动切换策略（如 IK 失败时切换关节空间规划） |
| **多模态 LLM 接入** | VLM 仅在 `RealSenseVLMSensor` 和 `Vision3DEstimator` 中使用 | 🟡 | LLM 本身未接收图像输入做决策 |
| **LLM 经验学习** | 无 | ❌ | 经验库内容未注入 LLM prompt，LLM 无法从历史经验学习 |

**结论**：LLM 是"可用选项"而非"强制基座"。需要让 LLM 更深入地感知世界、利用经验、动态决策。

---

### 2. harness / agent / 提示词工程为规划平台 🟡/❌

#### 2.1 Harness 层

| 需求 | 现状 | 评级 |
|---|---|---|
| 独立的 harness/orchestrator 模块 | 不存在。只有 `SkillRouter` 做简单的顺序执行 | ❌ |
| 任务分解与编排（DAG/并行/条件） | 只有串行 plan 执行 | ❌ |
| 资源冲突管理 | `RobotQueue` 提供串行化，但无细粒度锁 | 🟡 |
| 异常恢复与重试策略 | `RobotQueue` 有指数退避，但 agent 层无策略调整 | 🟡 |

#### 2.2 Agent 层

| 需求 | 现状 | 评级 |
|---|---|---|
| 自主任务分解 | `LLMPlanner.plan()` 一次性生成计划，无动态分解 | ❌ |
| 目标管理（goal stack） | 无 | ❌ |
| 信念-愿望-意图（BDI） | 无 | ❌ |
| 自我反思（self-reflection） | 无 | ❌ |
| 工具/技能自动发现 | Skill 需手动注册，无自动发现机制 | ❌ |

#### 2.3 提示词工程

| 需求 | 现状 | 评级 |
|---|---|---|
| 动态 prompt 组装 | 2 个静态 SYSTEM_PROMPT | ❌ |
| 工具描述自动生成 | Skill 无 schema 声明，prompt 中工具描述硬编码 | ❌ |
| Few-shot 示例注入 | 无 | ❌ |
| Prompt 版本管理 | 无 | ❌ |
| Function-Calling 原生支持 | 用 JSON 解析模拟 function call，未使用 OpenAI/Claude 原生 tool_call | 🟡 |
| 提示词性能监控（token 数、延迟、成功率） | 无 | ❌ |

**结论**：规划平台目前只是 "LLM + 顺序 Skill 调用"，缺少真正的 harness（编排引擎）和 agent（自主决策体）。提示词工程几乎为空白。

---

### 3. 整合所有规划经验 🟡/❌

| 需求 | 现状 | 评级 |
|---|---|---|
| 经验结构化存储 | `ExperienceLibrary` (SQLite) + `Experience` schema ✅ | ✅ |
| 经验三级检索 | `ExperienceRetriever` (exact → fuzzy → fallback) ✅ | ✅ |
| 经验自动记录 | `ExperienceRecorder` 支持 3 种来源 | ✅ |
| **LLM 感知经验** | `LLMPlanner` 完全不知道 `ExperienceLibrary` 存在 | ❌ |
| **经验注入 Prompt** | 无 | ❌ |
| **经验自动总结** | 无（从 episodes 提取 pattern） | ❌ |
| **经验主动推荐** | 无（"根据历史经验，建议..."） | ❌ |
| **经验反馈闭环** | 执行后未自动更新 success_rate / human_feedback | 🟡 |

**结论**：经验系统的"存储和检索"已做好，但"整合到规划平台"这一步完全缺失。经验只是被 Skill 内部查询，LLM/Agent 看不到经验。

---

### 4. 所有可以作为真实世界信号输入的数据形式 🟡/❌

**已实现（9 个 sensor）：**

| Sensor | 数据形式 | 真实硬件支持 |
|---|---|---|
| VisionRGBSensor | RGB 图像 | OpenCV USB / MuJoCo renderer |
| VisionDepthSensor | 深度图 | MuJoCo renderer |
| PointCloudSensor | 点云 | MuJoCo |
| ProprioceptionSensor | 关节位置/速度/末端位姿 | YHRG S1 SDK / MuJoCo |
| TactileSensor | 接触力总和 | MuJoCo contact forces |
| RealSenseRGBSensor | RGB (D435i) | RealSense D435i |
| RealSenseDepthSensor | 深度 (D435i) | RealSense D435i |
| RealSenseVLMSensor | VLM 检测结果 + 3D 反投影 | RealSense + 阿里云 VLM |

**严重缺失的信号类型：**

| 数据形式 | 应用场景 | 架构影响 |
|---|---|---|
| **音频 / 声音** | 语音识别、声源定位、碰撞声检测 | 需新增 AudioSensor |
| **IMU / 加速度计 / 陀螺仪** | 姿态估计、振动检测 | 需新增 IMUSensor |
| **六维力/力矩传感器** | 精密装配、抛光、拖动示教 | 需新增 WrenchSensor |
| **激光雷达 (LiDAR)** | 导航、避障、场景重建 | 需新增 LidarSensor |
| **超声波传感器** | 近距离测距、液位检测 | 需新增 UltrasonicSensor |
| **编码器 / 轮式里程计** | 移动底盘定位 | 需新增 OdometrySensor |
| **GPS / RTK** | 室外定位 | 需新增 GPSSensor |
| **温度 / 湿度 / 气压** | 环境感知、农业/仓储 | 需新增 EnvironmentSensor |
| **电机电流/电压** | 健康监测、碰撞检测 | 可并入 Proprioception |
| **高分辨率触觉阵列** | 纹理识别、滑移检测 | TactileSensor 需扩展 |
| **事件相机 (Event Camera)** | 高速运动捕捉 | 需新增 EventCameraSensor |
| **红外热成像** | 热源检测、夜视 | 需新增 ThermalSensor |
| **RFID / UWB** | 物体识别、室内定位 | 需新增 RFSensor |
| **EEG / EMG** | 脑机/肌电接口 | 需新增 BiosignalSensor |

**结论**：SensorChannel 基类设计是通用的，扩展容易，但当前仅覆盖了"视觉 + 本体感 + 简单触觉"三大类。距离"所有真实世界信号"差距极大。

---

### 5. 兼容控制所有真实世界可以控制运动的机械结构 ❌

**已支持的机械结构：**

| 结构 | 实现 | 状态 |
|---|---|---|
| 6-DOF 串联机械臂 (YHRG S1) | `YHRGAdapter` | ✅ |
| 7-DOF 串联机械臂 (Franka, MuJoCo) | `FrankaMujocoAdapter` | ✅ |
| 二指平行夹爪 | `control_gripper()` | ✅ |

**严重缺失的机械结构：**

| 结构类型 | 典型代表 | 控制差异 |
|---|---|---|
| **轮式移动底盘** | TurtleBot, AGV | 差分驱动 / 全向轮，需速度指令而非关节角 |
| **四足/六足机器人** | Unitree Go2, Spot | 足端轨迹规划、CoM 平衡、步态生成 |
| **连续体/软体机械臂** | OctArm, 气压驱动 | 曲率参数空间，非刚性连杆 |
| **无人机/多旋翼** | DJI, PX4 | SE(3) 位姿控制、推力分配 |
| **水下机器人** | BlueROV | 6-DOF 流体动力学、浮力补偿 |
| **绳索驱动 (CDPR)** | 并联索驱动 | 张力分配、 workspace 分析 |
| **外骨骼** | HAL, Ekso | 人机交互力、意图识别 |
| **灵巧手** | Shadow Hand, Allegro | 20+ DOF、触觉反馈、抓取规划 |
| **SCARA / Delta** | 工业拾放 | 并联/选择性顺应结构 |
| **传送带/升降机构** | 产线设备 | 连续运动 + 离散事件混合 |

**核心问题**：当前没有通用的 `RobotInterface` ABC。`YHRGAdapter` 的接口（`joint_control`, `control_gripper`）是特定于串联机械臂的，无法直接套用到轮式底盘或无人机上。

**需要引入的抽象层：**
```python
class RobotInterface(ABC):
    # 通用控制接口，不同实现映射到各自硬件
    @abstractmethod
    def command(self, action: Action) -> bool: ...
    
    @abstractmethod  
    def observe(self) -> Observation: ...
    
    @property
    def action_space(self) -> Space: ...
    
    @property
    def observation_space(self) -> Space: ...
```

---

### 6. 处理输入信号得到输出信号的算法，以 skill 形式保存 🟡/❌

**已实现的算法 Skill：**

| Skill | 算法类型 | 评级 |
|---|---|---|
| `Vision3DEstimator` | VLM + 反投影 | ✅ |
| `GraspPointPredictor` | 规则基抓取规划 | 🟡（太简单） |
| `VLAPolicyExecutor` | VLA 端到端策略 | ❌（stub） |
| `FabricManipulationSkill` | 经验驱动的双臂操作 | 🟡（领域特定） |

**严重缺失的算法 Skill：**

| 算法类别 | 具体算法 | 为什么需要 |
|---|---|---|
| **信号处理** | 卡尔曼滤波、低通/高通滤波、FFT | 降噪、状态估计 |
| **点云处理** | RANSAC 分割、ICP 配准、DBSCAN 聚类 | 场景理解、定位 |
| **SLAM** | ORB-SLAM3, LIO-SAM, FAST-LIO | 移动机器人定位建图 |
| **运动规划** | RRT*, A*, CHOMP, TrajOpt | 避障、最优轨迹 |
| **力控制** | 导纳控制、阻抗控制、力位混合 | 接触任务 |
| **状态估计** | EKF、粒子滤波 | 多传感器融合 |
| **视觉感知** | YOLO、SAM、6D 姿态估计 | 物体检测分割 |
| **路径跟踪** | Pure Pursuit, MPC | 移动底盘导航 |
| **强化学习** | PPO, SAC | 策略学习 |
| **模仿学习** | Diffusion Policy, ACT | 从示范学习 |

**SkillInterface 的设计缺陷：**

当前接口过于简单：
```python
class SkillInterface(ABC):
    @property
    def name(self) -> str: ...
    def execute(self, **kwargs) -> Dict[str, Any]: ...
```

缺失：
- 输入/输出 **schema 声明**（让 LLM 知道这个 skill 需要什么参数、返回什么）
- **依赖声明**（需要哪些 sensor、哪些 hardware）
- **预条件/后条件**（什么状态下可以调用、调用后世界状态如何变化）
- **组合能力**（skill 的输出管道到另一个 skill 的输入）

---

### 7. 供规划平台调用 🟡

| 需求 | 现状 | 评级 |
|---|---|---|
| Skill 注册与发现 | `SkillRouter.register()` 手动注册 | 🟡 |
| Skill 串行执行 | `execute_plan()` 支持 | ✅ |
| **Skill 并行执行** | 无（如同时读多个 sensor） | ❌ |
| **Skill 条件分支** | 无（if/then/else） | ❌ |
| **Skill 循环** | 无（while/for） | ❌ |
| **Skill 管道/组合** | 无（skill A → skill B 数据流） | ❌ |
| **动态 Skill 选择** | LLM 从固定列表选 skill，无按需加载 | ❌ |
| **Skill 输入验证** | 只有运行时类型检查，无 schema 校验 | 🟡 |

**结论**：调用机制能工作，但只是"顺序调用列表"，缺少真正的编排能力。

---

## 总体评估

| 需求维度 | 完成度 | 核心差距 |
|---|---|---|
| 1. LLM 智慧基座 | 40% | 不是强制基座，未整合经验，无多模态 |
| 2. Harness/Agent/Prompt | 20% | 无编排引擎，无自主 agent，prompt 工程空白 |
| 3. 规划经验整合 | 50% | 存储检索做好，但未接入 LLM/Agent |
| 4. 所有信号输入 | 25% | 仅覆盖视觉+本体感+简单触觉 |
| 5. 所有运动结构 | 20% | 仅支持串联臂，无通用机器人抽象 |
| 6. 算法即 Skill | 30% | 算法 skill 太少，接口缺少 schema |
| 7. 规划平台调用 | 40% | 只能串行执行，无并行/条件/循环 |

**加权平均完成度：~32%**

骨架已搭好，但"血肉"差距巨大。

---

## 补全优先级建议

### P0（立即补）— 架构基石
1. **通用 RobotInterface ABC** — 统一所有运动结构的控制接口
2. **Skill Schema 声明** — 让 LLM 能自动理解每个 skill 的能力
3. **经验注入 LLM Prompt** — 让 LLM 能看到历史经验
4. **Prompt Engineering 框架** — 动态组装、版本管理、工具描述自动生成

### P1（短期）— 能力扩展
5. **新增 Signal Processing Skills** — 滤波、点云处理、状态估计
6. **新增 Sensor 类型** — IMU、力矩、音频、LiDAR（至少 stub）
7. **编排引擎（Harness）** — 支持并行、条件、循环、管道
8. **VLA 模型接入** — 补全 `VLAPolicyExecutor`

### P2（中期）— 智能化
9. **多模态 LLM 决策** — LLM 直接看图像做决策
10. **Agent 自主分解** — BDI 或类似架构
11. **自动经验总结** — 从 episodes 提取可复用 pattern
12. **动态 Skill 发现** — 根据任务自动加载所需 skill

### P3（长期）— 全覆盖
13. **更多机械结构适配器** — 轮式底盘、四足、无人机等
14. **更多传感器实现** — 事件相机、热成像、RFID 等
15. **高级算法技能** — SLAM、运动规划、RL、模仿学习
