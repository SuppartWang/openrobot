# OpenRobotDemo 架构梳理 (v2.0)

> 最后更新: 2026-04-15

## 1. 总体架构

OpenRobotDemo 是一个面向双臂具身机器人的分层控制系统，采用 **ReAct 循环 + 经验驱动 + 可插拔感知** 的架构设计。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT CHANNELS                                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                                     │
│  │ CLI     │  │ HTTP    │  │ Voice   │     (pluggable, registry-based)     │
│  │ Channel │  │ Channel │  │(placeholder)                                  │
│  └────┬────┘  └────┬────┘  └────┬────┘                                     │
└───────┼────────────┼────────────┼──────────────────────────────────────────┘
        │            │            │
        └────────────┴────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────────────────┐
│                              RUNTIME LAYER                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ RobotQueue (FIFO task queue + retry + persistence hooks)               │  │
│  │  - enqueue(instruction, task_fn) → episode_id                          │  │
│  │  - background worker thread                                           │  │
│  │  - exponential backoff retry (max 3)                                  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │ injects EpisodeRecorder
┌──────────────────────▼──────────────────────────────────────────────────────┐
│                              AGENT LAYER                                     │
│  ┌─────────────────────────┐    ┌─────────────────────────────────────────┐  │
│  │  LLMPlanner (ReAct)     │    │  SkillRouter                            │  │
│  │  ─────────────────────  │    │  ─────────────────────────────────────  │  │
│  │  • start_task()         │───→│  • register(skill: SkillInterface)      │  │
│  │  • next_action(state)   │    │  • execute_plan(plan_steps)             │  │
│  │  • _mock_plan fallback  │    │  • _resolve_args (context substitution) │  │
│  │                         │    │  • _resolve_motion_args (placeholders)  │  │
│  │  LLM API: Kimi / Qwen   │    │                                         │  │
│  └─────────────────────────┘    └─────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────────────────┐
│                              SKILL LAYER                                     │
│                                                                              │
│   SkillInterface (ABC)                                                       │
│   └── name + execute(**kwargs) → Dict{"success", "message", ...}            │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Basic Skills (single-arm, generic)                                 │   │
│   │  ─────────────────────────────────────────────────────────────────  │   │
│   │  • CameraCapture         → rgb_frame, depth_frame                  │   │
│   │  • ArmStateReader        → joint_positions, end_effector_pose      │   │
│   │  • Vision3DEstimator     → pixel_bbox, camera_3d, base_3d          │   │
│   │  • GraspPointPredictor   → grasp_pose, pre_grasp_pose (rule-based) │   │
│   │  • ArmMotionExecutor     → joint/cartesian/gripper (with safety)   │   │
│   │  • VLAPolicyExecutor     → VLA model bridge (pi05 stub)            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Composite Skills (dual-arm, domain-specific)                       │   │
│   │  ─────────────────────────────────────────────────────────────────  │   │
│   │  • FabricManipulationSkill                                          │   │
│   │    - pinch_edge(fabric_center, diameter)                            │   │
│   │    - lift(height_m)                                                 │   │
│   │    - insert(plate_center, height, depth)                            │   │
│   │    - hold_wait(seconds)                                             │   │
│   │    - withdraw(lift_height)                                          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────────────────┐
│                         PERCEPTION LAYER (Sensor Registry)                   │
│                                                                              │
│   SensorChannel (ABC) → capture() → PerceptionData                           │
│   └── name, source_id, timestamp, modality, payload, confidence             │
│                                                                              │
│   ┌─────────────────┬─────────────────┬─────────────────┐                   │
│   │ Vision Sensors  │ RealSense Sensors│ Other Sensors   │                   │
│   ├─────────────────┼─────────────────┼─────────────────┤                   │
│   │ • VisionRGB     │ • RS RGB        │ • Proprioception│                   │
│   │ • VisionDepth   │ • RS Depth      │ • Tactile       │                   │
│   │ • PointCloud    │ • RS VLM        │                 │                   │
│   └─────────────────┴─────────────────┴─────────────────┘                   │
│                                                                              │
│   RealSenseVLMSensor: RGB-D → VLM(qwen-vl-max) → pixel → deproject → 3D   │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │ PerceptionData
┌──────────────────────▼──────────────────────────────────────────────────────┐
│                         WORLD MODEL (Unified Memory)                         │
│                                                                              │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│   │ RobotState  │  │ ObjectDesc  │  │SpatialMemory│  │  TaskMemory     │   │
│   │ ─────────── │  │ ─────────── │  │ ─────────── │  │  ─────────────  │   │
│   │ joint_pos   │  │ object_type │  │ robot_pos   │  │  episode_id     │   │
│   │ joint_vel   │  │ position    │  │ surfaces    │  │  instruction    │   │
│   │ ee_pose     │  │ color/size  │  │ obstacles   │  │  status         │   │
│   │ gripper_w   │  │ material    │  │ workspace   │  │  key_learnings  │   │
│   │             │  │ grasp_points│  │ bounds      │  │                 │   │
│   │             │  │ relations   │  │             │  │                 │   │
│   │             │  │ owner       │  │             │  │                 │   │
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘   │
│                                                                              │
│   ingest(PerceptionData) → update robot_state / objects / spatial_memory    │
│   build_state_summary()  → text for ReAct planner                            │
│   to_dict()              → full serialization for persistence                │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │ queries / updates
┌──────────────────────▼──────────────────────────────────────────────────────┐
│                       EXPERIENCE SYSTEM (Learning Layer)                     │
│                                                                              │
│   Experience Schema (Context → Parametric Policy → Outcome)                  │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Context                    Policy                    Outcome       │   │
│   │  ────────                   ──────                    ───────       │   │
│   │  task_intent              action_type                 success       │   │
│   │  target_object_type       pre_contact_offset          execution_time│   │
│   │  target_object_tags       approach_angle_deg          final_error   │   │
│   │  gripper_config           gripper_aperture_m          human_feedback│   │
│   │  arm_count                dual_arm_pinch_distance_m                 │   │
│   │                           dual_arm_sync_tolerance_m                 │   │
│   │                           trajectory_type / waypoints                 │   │
│   │                           max_velocity / compliance                   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ExperienceLibrary (SQLite)  ←──CRUD──→  ExperienceRetriever (3-stage)     │
│   ├── add(exp)                              ├── Stage 1: exact match        │
│   ├── get(id)                               ├── Stage 2: keyword fuzzy      │
│   ├── query(filters)                        └── Stage 3: action fallback    │
│   └── increment_use(id)                                                     │
│                                                                              │
│   ExperienceRecorder ──creates──→ new Experience from execution trace       │
│   └── Sources: human_demo | autonomous_trial | vla_inference                │
│                                                                              │
│   seed_fabric_experiences() → 5 pre-loaded fabric manipulation records      │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │ commands
┌──────────────────────▼──────────────────────────────────────────────────────┐
│                         HARDWARE / CONTROL LAYER                             │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  DualArmController (leader-follower synchronization)                │   │
│   │  ─────────────────────────────────────────────────────────────────  │   │
│   │  left_arm: YHRGAdapter  +  left_kin: YHRGKinematics               │   │
│   │  right_arm: YHRGAdapter +  right_kin: YHRGKinematics              │   │
│   │                                                                     │   │
│   │  • dual_move_cartesian(l_target, r_target, duration, sync_tol)    │   │
│   │  • dual_grasp(l_target, r_target) → approach → descend → close    │   │
│   │  • dual_lift(height_m)                                            │   │
│   │  • dual_release(gripper_open_pos) → right first, then left        │   │
│   │  • move_joint(side, target_joints)                                │   │
│   │  • move_cartesian(side, target_pose)                              │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  YHRGAdapter / YHRGKinematics (real ↔ mock dual mode)             │   │
│   │  ─────────────────────────────────────────────────────────────────  │   │
│   │  Real: S1_SDK (S1_arm, S1_slover, control_mode, Arm_Search)       │   │
│   │  Mock: _MockS1Arm + _MockS1Slover (6-DOF FK/IK with Jacobian)     │   │
│   │                                                                     │   │
│   │  Key invariant: get_pos() returns 7 values [j1..j6, gripper]      │   │
│   │                joint_control() only updates first 6               │   │
│   │                control_gripper() updates 7th value independently  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Safety & Interpolation                                             │   │
│   │  ─────────────────────────────────────────────────────────────────  │   │
│   │  • SafetyGateway: joint limits + workspace bounds + speed limits    │   │
│   │  • JointSpaceInterpolator: linear interpolation (default 50 steps)  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 数据流全景

```
┌─────────┐    instruction     ┌──────────┐    task_fn(instruction, recorder)
│  User   │ ─────────────────→ │ RobotQueue│ ─────────────────────────────────→
│ (CLI/   │                    │  (FIFO)   │                                     
│  HTTP/  │ ←───────────────── │           │ ←─────────────────────────────────
│  Voice) │    episode_id      └──────────┘    (result / error / retry)        
└─────────┘                                                                      
                                           │                                    
                                           ▼                                    
                              ┌────────────────────────┐                       
                              │   ReAct Loop (max 20)   │                       
                              │  ─────────────────────  │                       
                              │  1. Sensor capture      │                       
                              │  2. WorldModel.ingest() │                       
                              │  3. build_state_summary │                       
                              │  4. Planner.next_action │                       
                              │  5. SkillRouter.execute │                       
                              │  6. Record step/result  │                       
                              │  7. Update context      │                       
                              └────────────────────────┘                       
                                           │                                    
                                           ▼                                    
                              ┌────────────────────────┐                       
                              │  EpisodeRecorder → SQLite                       
                              │  episodes / steps / step_results                
                              │  / state_snapshots                              
                              └────────────────────────┘                       
```

---

## 3. 模块详解

### 3.1 Input Channels (`channels/`)

| 模块 | 职责 | 状态 |
|---|---|---|
| `base.Channel` | ABC: `start(on_message)`, `stop()` | ✅ 稳定 |
| `cli_channel.CLIChannel` | 后台线程读取 stdin | ✅ 稳定 |
| `http_channel.HTTPChannel` | stdlib HTTPServer, POST /task, GET /status /health | ✅ 稳定 |
| `voice_channel.VoiceChannel` | 占位，待接入 ASR | 🚧 stub |
| `registry` | 名称→类 注册表，支持 `create_channel(name, **kwargs)` | ✅ 稳定 |

### 3.2 Runtime (`runtime/`)

| 模块 | 职责 | 关键设计 |
|---|---|---|
| `queue.RobotQueue` | 串行化任务执行 | 单后台线程 + Condition + 指数退避重试 |
| `_QueuedTask` | 任务包装 | episode_id, task_fn, status, retry_count |

### 3.3 Persistence (`persistence/`)

| 表 | 字段 | 用途 |
|---|---|---|
| `episodes` | id, instruction, status, created_at, completed_at | 任务生命周期 |
| `steps` | id, episode_id, step_idx, thought, action_json, state_summary, wait_time_s | 每步决策记录 |
| `step_results` | id, step_id, skill_name, success, message, result_json | 每步执行结果 |
| `state_snapshots` | id, episode_id, step_idx, context_json | 世界模型完整状态快照 |

**`EpisodeRecorder`**：封装了 `RobotDatabase` 的便捷写入器，自动关联 step ↔ result。

**`_sanitize_for_json()`**：递归处理 numpy ndarray、标量、bytes 等不可 JSON 序列化对象。

### 3.4 Agent (`agent/`)

#### LLMPlanner (`planner.py`)
- **双模式 API**：
  - `plan(instruction)` → 静态一次性计划（legacy）
  - `start_task()` + `next_action(state_summary)` → ReAct 迭代
- **Mock fallback**：当 LLM API 不可用时，根据指令关键词返回硬编码计划序列
  - "布料/套/支撑板" → 完整双臂布料操作 8 步计划
  - "pick/grab/抓" → 标准 pick-and-place 8 步计划
  - "place/放" → 放置 3 步计划
- **API 支持**：Kimi (api.kimi.com) / 阿里云百炼 (dashscope)

#### SkillRouter (`skill_router.py`)
- **上下文传递**：`router._context` 存储 skill 输出，下游 skill 的 args 字符串可自动替换
- **占位符解析**：`PRE_GRASP`, `GRASP`, `LIFT`, `PLACE`, `RETREAT` → 实际位姿

### 3.5 Skills (`skills/`)

| Skill | 输入 | 输出 | 依赖 |
|---|---|---|---|
| `CameraCapture` | return_depth: bool | rgb_frame, depth_frame | CameraDriver |
| `ArmStateReader` | fields: list | joint_positions, end_effector_pose, ... | S1_arm, S1_slover |
| `Vision3DEstimator` | rgb_frame, target_name, depth_frame, intrinsics | pixel_bbox, camera_3d, base_3d | VLM API / OpenCV fallback |
| `GraspPointPredictor` | object_pose_base, object_type | grasp_pose, pre_grasp_pose, approach_vector | 规则基 (box/cylinder/sphere) |
| `ArmMotionExecutor` | command_type, target_values, speed | actual_reached_pos, execution_time_ms | SafetyGateway, Interpolator |
| `VLAPolicyExecutor` | instruction, rgb_frame | actions (T×A chunk) | pi05 model (stub) |

**所有 Skill 遵循统一接口**：
```python
class SkillInterface(ABC):
    @property
    def name(self) -> str: ...
    def execute(self, **kwargs) -> Dict[str, Any]: ...  # 必须含 success, message
```

### 3.6 Dual-Arm (`dual_arm/`)

#### DualArmController
- **同步机制**：`dual_move_cartesian()` 使用 leader-follower 锁步插值
  - 计算左右臂 IK 目标关节角
  - 线性插值 (steps = duration / 0.02)
  - 每步同时发送两个臂，无显式同步检查（依赖控制器硬件定时）
- **抓取序列**：`dual_grasp()` = approach(预抓取上方5cm) → descend → close grippers
- **释放序列**：`dual_release()` = right 先开 → 0.2s → left 再开（模仿人类协调）

#### FabricManipulationSkill
- **经验驱动**：每个 operation 自动调用 `ExperienceRetriever.retrieve_best()` 加载参数
- **参数覆盖**：运行时 kwargs 优先级 > 经验库参数 > 硬编码默认值
- **世界模型更新**：pinch 后标记 fabric_tube 为 grasped，withdraw 后释放

### 3.7 Sensors (`sensors/`)

| Sensor | modality | 数据来源 | availability |
|---|---|---|---|
| `VisionRGBSensor` | rgb | OpenCV / mock | 总是可用 |
| `VisionDepthSensor` | depth | OpenCV / mock | 总是可用 |
| `PointCloudSensor` | pointcloud | 待实现 | mock |
| `ProprioceptionSensor` | proprioception | arm_adapter.get_pos() + FK | arm_adapter 非 None |
| `TactileSensor` | tactile | MuJoCo contact forces | MuJoCo model/data 非 None |
| `RealSenseRGBSensor` | rgb | pyrealsense2 pipeline | Linux/Windows + 相机在线 |
| `RealSenseDepthSensor` | depth | pyrealsense2 pipeline | Linux/Windows + 相机在线 |
| `RealSenseVLMSensor` | vlm_detection | RS RGB-D + qwen-vl-max + deproject | RS + VLM API 均可用 |

**PerceptionData 统一数据包**：
```python
@dataclass
class PerceptionData:
    modality: str          # "rgb" | "depth" | "vlm_detection" | "proprioception" | "tactile" | ...
    source_id: str
    timestamp: float
    payload: Any           # numpy array, dict, list, etc.
    spatial_ref: str       # 坐标系名称
    confidence: float
    metadata: dict
```

### 3.8 World Model (`world_model/`)

**核心设计**：不是简单的数据容器，而是 **语义融合引擎**
- `ingest(PerceptionData)` 根据 modality 分发到对应的 `_update_*` 方法
- `ObjectDesc` 支持语义关系图：`relations = {"on": "table", "left_of": "cup"}`
- `query_nearby_objects(position, radius)` 支持空间查询
- `build_state_summary()` 生成中文自然语言状态描述，供 ReAct planner 使用

### 3.9 Experience System (`experience/`)

#### 三级检索策略 (ExperienceRetriever)

```
Stage 1: 精确匹配
  query(task_intent=..., target_object_type=..., action_type=..., gripper_config=..., success_only=True)
  → 若命中，返回并按 use_count 排序

Stage 2: 关键词模糊匹配
  query(task_intent=..., action_type=..., success_only=True)
  → 放宽 object_type 约束

Stage 3: 动作类型 fallback
  query(action_type=..., success_only=True)
  → 仅按动作类型匹配（如任意 "pinch" 经验）
```

每次成功检索后自动 `increment_use()` 更新使用频率。

#### 预置经验 (seed.py)

5 条双臂布料操作经验，覆盖完整 Demo 流程：

| # | action_type | 关键参数 | human_feedback |
|---|---|---|---|
| 1 | `pinch` | pinch_dist=0.08m, sync_tol=0.002m | 双臂同时接触后闭合，低速接近 |
| 2 | `lift` | max_vel=0.05m/s, compliance=100 | 提升保持严格同步，高度差<2mm |
| 3 | `insert` | max_vel=0.03m/s, compliance=120 | 速度慢，遇阻力回退2mm再试 |
| 4 | `withdraw` | max_vel=0.04m/s | 先松右臂再松左臂，遇阻小幅晃动 |
| 5 | `grasp` (单臂 fallback) | approach=45°, aperture=0.02m | 夹取边缘重叠处增加摩擦 |

### 3.10 Hardware (`hardware/`)

#### YHRGAdapter / YHRGKinematics

**双模式架构**：
```
SDK 可用 (Linux + S1_SDK_V2)          SDK 不可用 (macOS / 开发环境)
         │                                        │
         ▼                                        ▼
   _RealS1Arm (ctypes wrapper)          _MockS1Arm (纯 Python)
   _RealS1Slover (C++ solver)           _MockS1Slover (数值 IK)
         │                                        │
         └────────────────┬───────────────────────┘
                          ▼
                    YHRGAdapter (统一接口)
                    YHRGKinematics (统一接口)
```

**MockS1Slover**：
- 6-DOF 前向运动学（DH-like 链式乘法）
- 数值逆运动学（Jacobian transpose + damping, 50 iter）
- 关节限位自动钳制

**关键不变式**（避免 Franka 7-DOF 代码与 S1 6-DOF 硬件冲突）：
- `get_pos()` 始终返回 `[j1, j2, j3, j4, j5, j6, gripper]`
- `joint_control(pos)` 只取前 6 个，第 7 个（夹爪）**绝不**在此方法中修改
- `control_gripper(pos)` 单独控制第 7 个值
- `YHRGKinematics` 收到 7 关节时 warn + 截断前 6，返回结果补 `[0.0]` 保持兼容

### 3.11 Control (`control/`)

| 模块 | 职责 |
|---|---|
| `safety_gateway.SafetyGateway` | 关节限位检查、工作空间检查、速度限制 |
| `interpolator.JointSpaceInterpolator` | 关节空间线性插值（默认 50 步） |

---

## 4. 核心执行流程

### 4.1 双臂布料操作 Demo 完整流程

```python
# scripts/demo_fabric_dual_arm.py

run_fabric_demo(mode="mock", instruction="将筒状布料提起，套在铝合金支撑板上...")
│
├─→ [1] setup_experiences()          # 加载/种子化 ExperienceLibrary
├─→ [2] setup_dual_arm()             # DualArmController.enable() 双臂上电
├─→ [3] setup_sensors()              # RS RGB/Depth + Proprioception ×2 + Tactile ×2
├─→ [4] WorldModel 预注册物体         # fabric_tube, support_plate
├─→ [5] SkillRouter 注册 7 个 skills
├─→ [6] LLMPlanner.start_task()      # 初始化 ReAct 对话状态
│
└─→ [7] ReAct Loop (max 20 steps)
    │
    ├── 每步开始:
    │   ├── 所有可用 sensor.capture() → world.ingest()
    │   └── state_summary = world.build_state_summary()
    │
    ├── planner.next_action(state_summary)
    │   ├── 有 API Key → 调用 LLM (Kimi/Qwen)
    │   └── 无 API Key → _mock_plan() 返回预定义步骤
    │
    ├── recorder.record_step()        # 写入 SQLite
    │
    ├── skill.execute(**args)
    │   └── 若是 fabric_manipulation:
    │       └── _get_exp(action_type) → ExperienceRetriever.retrieve_best()
    │           └── 三级检索 → 返回 Experience 参数
    │       └── 按参数执行 DualArmController 动作
    │
    ├── recorder.record_step_result()  # 写入 SQLite
    ├── recorder.record_state_snapshot()  # 世界模型快照
    │
    └── action == "finish" → break
```

### 4.2 HTTP 服务入口流程

```python
# scripts/serve_openrobot.py

main()
├─→ register channels (cli, http, voice)
├─→ init_database()                  # SQLite 连接
├─→ RobotQueue(db=db)               # 带持久化的任务队列
├─→ on_message(instruction):
│   └── queue.enqueue(instruction, task_fn)
│       └── task_fn = run_simulation  # 或 run_fabric_demo
└─→ channel.start(on_message)       # 阻塞监听输入
```

---

## 5. 关键设计决策

### 5.1 为什么采用 "情境→参数化策略→结果" 而非三元组查表？

| 方案 | 优点 | 缺点 |
|---|---|---|
| 三元组 `(gripper, object, action)` | 简单、O(1) 查询 | 无法表达参数差异、无法泛化、无法积累反馈 |
| **参数化 Experience** | 支持连续参数空间、模糊匹配、人工反馈、自动优化 | 实现稍复杂 |

Experience 的 `context_signature()` 提供紧凑的模糊匹配键，同时保留完整的参数细节供运行时调整。

### 5.2 为什么 Sensor 和 Skill 分离？

- **Sensor** = 原始感知（RGB 图、深度图、关节角、接触力）→ `PerceptionData`
- **Skill** = 语义操作（拍照、检测物体、移动手臂、抓取布料）→ 结构化结果

分离后：
- 同一 Sensor 可被多个 Skill 复用
- Skill 不直接依赖硬件驱动，通过 Sensor/Adapter 抽象层隔离
- 新增 Sensor 只需实现 `SensorChannel.capture()`，无需修改 Skill

### 5.3 为什么双臂同步用 "锁步插值" 而非独立控制？

- **Leader-follower 锁步**：每 20ms 同时更新两臂关节指令，确保物理时间上的同步
- **sync_tolerance_m**：虽然当前实现是定时发送而非位置反馈闭环，但参数已预留用于未来添加反馈检查
- **线性关节插值**：简单可靠，避免复杂轨迹规划的计算开销

---

## 6. 模块依赖图

```
channels/           ──depends──→  runtime/queue, persistence/db
runtime/queue       ──depends──→  persistence/db
agent/planner       ──depends──→  (external: openai SDK)
agent/skill_router  ──depends──→  skills/base
skills/*            ──depends──→  skills/base, hardware/yhrg_adapter, control/*, perception/*
dual_arm/*          ──depends──→  hardware/yhrg_adapter, experience/*, world_model/*
sensors/*           ──depends──→  sensors/base
sensors/base        ──depends──→  (none, pure ABC)
world_model/*       ──depends──→  sensors/base
experience/*        ──depends──→  experience/schema
persistence/db      ──depends──→  (stdlib only: sqlite3, threading)
hardware/yhrg_adapter ──depends──→  (optional: S1_SDK), numpy, scipy
control/*           ──depends──→  numpy
```

---

## 7. 待完善项

| 模块 | 问题 | 优先级 |
|---|---|---|
| `VLAPolicyExecutor` | `_load_model()` 和 `_infer()` 仍为 stub，pi05 未接入 | 🔴 高 |
| `RealSenseVLMSensor` | 当前用阿里云 qwen-vl-max，API Key 管理需优化 | 🟡 中 |
| `seed.py` | 布料直径、支撑板高度等参数需真机实测校准 | 🔴 高 |
| `TactileSensor` | 仅支持 MuJoCo，真机需接入力矩传感器 | 🟡 中 |
| `VoiceChannel` | 纯占位，需接入 ASR (Whisper / 讯飞) | 🟢 低 |
| `WorldModel._update_from_rgb/depth` | 目前只是 logging，未实现 VLM 物体检测 | 🟡 中 |
| `SafetyGateway` | 目前只检查单臂，双臂碰撞检测未实现 | 🟡 中 |
| MuJoCo 物理 | 长时间运行偶发 NaN / 内存溢出（旧 demo 已知问题） | 🟢 低 |
