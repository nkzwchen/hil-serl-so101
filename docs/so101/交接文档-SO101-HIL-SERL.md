# SO101 HIL-SERL 项目交接文档

> 生成时间：2026-09-27
> 用途：在新对话中无缝继续本项目的开发
> 当前阶段：**阶段 A 完成，阶段 B0 完成，下一步 B0.6（接真机跑 RLPD）**

---

## 0. 一页速览

### 项目目标
在 **SO101 机械臂**上用 **HIL-SERL（RLPD 算法）** 做真机强化学习，任务对齐仿真的
**`LeIsaac-SO101-PickOrange-v0`（把橙子放进盘子）**。

### 三条并行技术路线（本项目走第 3 条）
| 路线 | 工具 | 状态 |
|---|---|---|
| 仿真 | LeIsaac / IsaacLab | 已有（`issac_lib/`）|
| 模仿学习 | LeRobot ACT/pi0、OpenPI 远程推理 | 已有（`可执行方案.md`）|
| **真机 RL（本任务）** | **HIL-SERL / RLPD** | **进行中** |

### 已完成的能力
```
✅ SO101 能被 ROS2 遥控（阶段 A）
✅ ROS2 Humble 装进 conda（py3.12，RoboStack 方案）
✅ lerobot 0.5.1 原生安装 + torch CUDA 可用
✅ HIL-SERL 全栈可运行（jax 0.4.35 CPU + flax/optax/distrax/agentlace）
✅ Actor↔Learner 双进程握手 + 训练循环跑通（B0.2）
✅ leader 臂可被电机驱动（L1）+ 实时跟随 follower（L2）
✅ SO101Env（gym）+ 空格 toggle 接管 + 力矩管理（B0.3~B0.5）
✅ 训练专用 launch（F1 方案，不含 teleop 组件）
```

### 下一步（B0.6）
```
1. 验证相机（唯一还没实测的环节）
2. 录 20 条 demo（--demo_path 是 RLPD 硬需求）
3. 采 classifier 数据 → 训 reward classifier（准确率 >90%）
4. classifier 接进 SO101Env.compute_reward
5. 真机跑 RLPD（预期 3 小时收敛）
```

---

## 1. 硬件与环境事实

### 硬件
| 项 | 值 |
|---|---|
| 平台 | WSL2 / Ubuntu 24.04.2 / hostname `nkdevice` |
| GPU | **RTX 4060 Laptop, 8GB**，驱动 592.82 / CUDA 13.1，`nvidia-smi` 正常 |
| Leader 臂 | `/dev/ttyACM0`，CH343 SN `5AE6083913`，lerobot id=`so101_leader_main` |
| Follower 臂 | `/dev/ttyACM1`，CH343 SN `5AE6058742`，lerobot id=`so101_follower_main` |
| 相机（全局/第三视角） | `/dev/video0` → by-id `...2603031-video-index0`，**640×480** |
| 相机（腕部） | `/dev/video2` → by-id `...2602030-video-index0`，**424×240** |
| 相机接法 | USB/IP（`usbipd attach --wsl`），设备可能需重新 attach |

> ⚠️ **串口端口号会随插拔顺序漂移**。以 `control.sh` 为准：leader=ACM0、follower=ACM1。
> ⚠️ 相机 `index1` 是 UVC metadata 节点（打不开），只有 `index0` 能采集。

### conda 环境
| 环境 | Python | 用途 |
|---|---|---|
| **`hil-serl`** | **3.12.14** | **主环境**：ROS2 + lerobot + torch + jax(CPU) + serl_launcher + agentlace |
| `lerobot` | 3.12.13 | lerobot 主环境（`hil-serl` 里的 lerobot 是 editable 指向同源码）|
| `leisaac` | 3.11 | Isaac Sim 仿真（不参与本任务）|
| `lerobot_pi` | 3.11 | 树莓派相关 |

**`hil-serl` 环境关键版本**：
```
ROS2 Humble (RoboStack, conda)  ros-humble-ros2-control 2.54.0
lerobot 0.5.1 (editable)        torch 2.10.0+cu128 (CUDA 可用)
jax 0.4.35 (CPU) / jaxlib 0.4.35   flax 0.10.4 / optax 0.2.5 / distrax 0.1.5
gymnasium 0.29.1 + gym 0.26.2    agentlace 0.1.3 / serl_launcher 0.1.2
numpy 2.2.6
散装包: xacro, robot_state_publisher, usb_cam, moveit-core/msgs/ros-planning,
        control-toolbox, ament-lint-auto, ros2pkg/ros2launch/ros2run/ros2bag...
```

### 一键环境脚本
```bash
source /home/nkzwchen/pi_project/so101_env.sh
```
它做 4 件事：清 `PYTHONPATH` 污染 → 激活 hil-serl → 叠加 `ros2_ws` → 设 WSL2 DDS 优化。

---

## 2. 必须知道的坑（踩过的雷，别再踩）

### 2.1 环境类
| # | 坑 | 根因 | 现状 |
|---|---|---|---|
| 1 | `pip`/`conda`/`ros2` **全部崩溃** | `serl_launcher/setup.py` 的 `install_requires` 里有 `"typing"`（PyPI 上是个 py2 时代 backport），装出 `site-packages/typing.py` 遮蔽 stdlib | **已从 setup.py 删除** |
| 2 | `PYTHONPATH` 污染所有 Python 进程 | RoboStack 的 `ros_workspace` 钩子路径拼错（`python312` 少一个点）| activate.d + `so101_env.sh` 里都做了清理 |
| 3 | `ros2 pkg`/`ros2 launch` 不存在 | 包清单漏了 `ros2pkg`/`ros2launch`/`ros2run` 等 CLI 包 | 已补装 |
| 4 | 节点秒退 `Failed to find a free participant index` | `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` → CycloneDDS participant 耗尽 | **改用默认 `rmw_fastrtps_cpp`**，不要再设那个变量 |
| 5 | `pip install git+https://...` 失败 | `~/.bashrc` 残留 Clash 代理（`127.0.0.1:7890`），**该代理连不上 GitHub** | 临时 `git config --global --unset http.proxy` 或用 `git -c http.proxy=` |
| 6 | pip 装大包超时/断连 | `pypi.org`、`files.pythonhosted.org`、清华源**都不可达** | 已配 `~/.config/pip/pip.conf` → 阿里云源 + `retries=20` + `resume-retries=20` |
| 7 | 云端必须用 **py3.10** | 最新 jax/flax 要求 py≥3.12，但 HIL-SERL 需 jax 0.4.35；py3.12 上装不到老组合的反面 —— 云端选 py3.10 镜像最稳 | 记录在 `云端训练方案-B阶段.md` |

### 2.2 ROS2 / 桥接类
| # | 坑 | 根因 | 修法 |
|---|---|---|---|
| 8 | lerobot 导入失败 | `so101_ros2` 用旧 fork 路径 `robots.so101_follower` | 改为 `robots.so_follower`（官方 0.5.1 路径）—— 已在 `bridge/build.py`、`bridge/bridge.py` 改好 |
| 9 | 两个桥抢同一串口 | 两个 yaml 都写 `/dev/ttyACM0` | follower→ACM1（以 `control.sh` 为准）|
| 10 | 标定数据不是本机的 | 原来用仓库自带 `Gili.json` | 改用 `control.sh` 的标定 + 显式 `calibration_dir` |
| 11 | **yaml 参数被静默忽略** | 顶层键 `so101_follower_ros2_bridge` ≠ 真实节点名 `/follower/so101_follower_interface` | 改用全限定节点名 + `/**` 双保险 |
| 12 | `Failed to send commands to robot: 10` | lerobot 的 `ensure_safe_goal_position` 只接受 `float`/`dict`，`isinstance(10, float)` 为 False → `TypeError(10)` | yaml 必须写 **`10.0`**（浮点），参数声明用 double |
| 13 | leader 无法被 `sync_write` | 键名要**纯 motor 名**（`shoulder_pan`），不接受 `.pos` 后缀 | 已在 `LeaderBridge.command_callback` 里去掉后缀 |

### 2.3 算法 / 框架类
| # | 坑 | 根因 | 修法 |
|---|---|---|---|
| 14 | `import experiments.mappings` 直接失败 | 原版无条件 import 4 个 franka 实验（依赖 ROS1 `rospy`）| 改成**宽容加载**，失败的跳过并记录 |
| 15 | Learner 报 `wandb: No API key` | wandb 默认 `mode="online"` | 加 `--debug`（flag 已存在于 `train_rlpd.py:50`）|
| 16 | Learner 报 `EOFError: input()` | `--checkpoint_path` 目录存在就触发 `input("Press Enter to resume")` | checkpoint 目录与 demo 文件**分开**，且 ckpt 目录不要预先创建 |
| 17 | `--demo_path` 断言失败 | learner 路径硬要求离线数据（`train_rlpd.py:458`）| 需生成 demo pkl（`gen_dummy_demos.py` 示例）|
| 18 | `client.update()` 首次必返回 False | server 端 `last_update_id_map` 初始为空（`agentlace/trainer.py:107`）| **属正常**，数据仍会送达 |
| 19 | 误判训练卡死 | Learner 有**两个**进度条：`Filling up replay buffer`（长时间 0%）和 `learner:`（真在跑）| **看 `learner:` 那个** |

### 2.4 键盘 / 接管类（WSL2 特有）
| 后端 | 结果 |
|---|---|
| `pynput` xorg | ❌ 需要 X server，`DISPLAY=127.0.0.1:0.0` 连不上 |
| `pynput` uinput | ❌ 该版本缺 `pynput.mouse._uinput` |
| `/dev/input` | ❌ 不存在 |
| **`termios` + stdin** | ✅ **实际使用**（训练时操作者在终端前）|
| `SERL_SPACE_FILE` 文件模拟 | ✅ 自动化测试用 |

> ⚠️ `termios` 会改终端模式。如果训练被 `kill -9` 强杀，**终端可能残留 cbreak 模式** →
> 敲 `stty sane` 或 `reset` 恢复。

---

## 3. 已建立的 ROS2 话题契约（关键！）

```
┌─ 观测（env 订阅）──────────────────────────────────────────┐
│ /follower/joint_states          JointState   6 关节弧度    │
│ /follower/cam_main/image_raw    Image        640×480       │
│ /follower/cam_wrist/image_raw   Image        424×240       │
│ /leader/joint_states            JointState   人类意图信号  │
└────────────────────────────────────────────────────────────┘
┌─ 命令（env 发布）──────────────────────────────────────────┐
│ /follower/arm_controller/joint_trajectory  JointTrajectory │
│      → 只发 5 个臂关节（不含 gripper），time_from_start=3ms│
│ /follower/gripper_controller/gripper_cmd   GripperCommand  │
│      → action 接口，position + max_effort=10.0             │
│ /leader/arm_command            Float64MultiArray (弧度×6)  │
│      → 驱动 leader 跟随（L1 新增）                          │
└────────────────────────────────────────────────────────────┘
┌─ Service（env 调用）───────────────────────────────────────┐
│ /leader/set_torque     std_srvs/SetBool  ON=跟随 OFF=自由  │
│ /follower/set_torque   std_srvs/SetBool  ON=保持 OFF=松开  │
└────────────────────────────────────────────────────────────┘
```

**关节顺序（全部）**：
```
[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
```
**关节限位（取自 URDF）**：
```
shoulder_pan  ±1.920    shoulder_lift ±1.745   elbow_flex  ±1.690
wrist_flex    ±1.658    wrist_roll    -2.744~+2.841
gripper       -0.175~+1.745
```

### 控制模式设计（重要）
| 模式 | follower 力矩 | leader 力矩 | 控制源 |
|---|---|---|---|
| **A. policy** | ON（被 policy 驱动）| ON（跟随 follower）| policy action |
| **B. 人类接管** | ON（被 leader 位置驱动）| **OFF**（人类自由推）| leader 位置 |
| 调试/摆位 | OFF | OFF | 手动 |
| 训练结束 | OFF | OFF | — |

**切换机制**：空格 **toggle**（按一下进入接管、再按一下退出）。`SERL_SPACE_MODE=hold` 可切回按住模式。

---

## 4. 关键设计决策（已与使用者确认）

| 决策 | 选择 | 理由 |
|---|---|---|
| 命令路径 | **直接发 `/follower/arm_controller/joint_trajectory`** | env 必须是 follower 的**唯一控制源**；避免与 `LeaderTeleopComponent` 抢话题 |
| 训练 launch | **F1：不含 `LeaderTeleopComponent`** | 该组件也发同一个 trajectory 话题，会打架 |
| 观测源 | `/follower/joint_states`（非 raw）| 与 `io.yaml` 一致，ros2_control 广播的权威值 |
| 动作空间 | ~~**关节增量**（前 5 维），`scale=0.05 rad/步`~~ → **2026-09-29 改为 3 维关节增量**（只有 `shoulder_lift`/`elbow_flex`/`wrist_flex`），详见 **`动作空间-冻结维度.md`** | 不需要 IK；相对量分布集中；与 leader 干预信号天然同构 |
| 夹爪 | ~~**二值**（第 6 维符号决定开/合）~~ → **2026-09-29 起不在动作空间里**（任务全程夹着试管，固定不动） | 与 `GripperCloseEnv` 设计一致 |
| `RelativeFrame` | **不用** | 它变换的是笛卡尔量（tcp_pose/tcp_vel），关节增量没有坐标系 |
| FK / tcp_pose | **暂不用**，留接口 | 需引入 `placo` 可选依赖；先用 joint_pos+joint_vel |
| 干预设备 | **Leader 臂**（不用 SpaceMouse）| 已有硬件；输出与 policy 同量纲，省掉 IK 和坐标变换 |
| 干预判定 | **位置增量 + 死区**（0.05 rad）| 原版 SpaceMouse 用"非零即干预"，但 leader 是绝对角度永不为零 |
| 干预动作语义 | **`delta / action_scale`** → 与 policy 同量纲 | 否则 `info.pop("intervene_action")` 会污染 buffer |
| 空格模式 | **toggle**（按一下进入/退出）| 文档说前 1 小时干预极密集，按住会累 |
| reward | **先 `TARGET_JOINT_POSE` 占位**，最终走 **reward classifier** | 见 `HIL-SERL精密接触型插接任务.md` §3.5 |
| 云端训练 | Actor 本地 + Learner 云端 | HIL-SERL 原生架构；带宽需求极低（上行 1~3 Mbps）|

---

## 5. 任务定义（PickOrange）

**来源**：`issac_lib/leisaac/source/leisaac/leisaac/tasks/pick_orange/`

```python
task_description = "Pick three oranges and put them into the plate, then reset the arm to rest state."

# 成功判据（mdp/terminations.py: task_done）
1. orange 在 plate 的 x/y 范围内 (±0.10 m)
2. orange 低于某高度 (height_range=(-0.07, 0.07))
3. robot 回到 rest pose          ← 容易忽略
```

**子任务观测**（`mdp/observations.py: put_orange_to_plate`）：
```python
orange_in_plate   = orange 在 plate ±10cm 内
ee_near_to_orange = |ee_pos - orange_pos| < 0.05m
gripper_open      = joint_pos[-1] > 0.60
placed = 三者都满足
```

> 💡 **对真机 reward 的启示**：前两条是**视觉可判**的（第三视角相机能看到橙子和盘子），
> 所以适合训 reward classifier。`TARGET_JOINT_POSE` 只是先跑通链路的临时替代。

**Reward Classifier 规格**（`HIL-SERL精密接触型插接任务.md` §3.5）：
| 项 | 值 |
|---|---|
| 架构 | 预训练 ResNet10 + `Linear→Dropout→LayerNorm→ReLU→Linear(1)` |
| 损失 | BCE（正负比 ~1:7 → 加权 BCE 或 Focal Loss）|
| 目标 | 测试集准确率 **>90%**，特别防**假阳性** |
| 在线推理 | sigmoid **>0.8**，建议**连续 3~5 帧**都超过才触发 |
| 数据目录 | `experiments/<exp_name>/classifier_data/` |
| 权重目录 | `experiments/<exp_name>/classifier_ckpt/` |
| 输入相机 | config 的 `classifier_keys` → 应为 **`cam_main`（全局）** |

**训练时间线预期**：
| 阶段 | 时间 | 现象 |
|---|---|---|
| 训练前 | — | 采 **20 条** demo 作为 offline prior data |
| 早期干预 | ~1 小时 | 干预最密集，reward 低且波动 |
| 中期 | 1~2.5 小时 | reward 稳步上升 |
| 收敛 | ~3 小时 | 达 100% 成功率 |

---

## 6. 交付文件清单

### 6.1 环境（工作区根目录）
| 文件 | 作用 |
|---|---|
| `so101_env.sh` | **一键激活环境**（每天用这个）|
| `actor_requirements.txt` | Actor 侧依赖清单（CPU jax，含版本钉死原因）|
| `rebuild_hilserl_py312.sh` | py3.12 环境重建脚本（已实战验证）|
| `fix_hilserl_ros2.sh` | 早期修复脚本（已被重建取代，保留备查）|

### 6.2 验证 / 测试脚本（工作区根目录）
| 文件 | 验证什么 |
|---|---|
| `test_leader_bus.py` | 直接测 SO101Leader 总线写入（定位 `sync_write` 键名要求）|
| `test_leader_drive.py` | **L1**：发命令驱动 leader，校验实际位移 |
| `l1_feel_test.sh` | **L1 手感测试**（力矩开关 + 驱动位置，含完整起停与安全收尾）|
| `test_leader_follows_follower.py` | **L2**：leader 实时跟随 follower，打印跟随误差 |
| `test_so101_env.py` | **SO101Env** 端到端（读观测 / 发小动作 / 查接管）|
| `test_so101_intervention.py` | 接管状态机（hold 语义）|
| `test_space_file_fallback.py` | 文件后备路径（无 TTY 环境）|
| `test_space_toggle.py` | **toggle 模式**（按一下进入/退出）|
| `test_agentlace_link.py` | agentlace 数据通路隔离测试（排障用）|
| `b02_run.sh` | B0.2 双进程启动器（`learner`/`actor`/`stop`）|

### 6.3 我新建的核心代码
| 文件 | 内容 |
|---|---|
| `hil-serl/examples/experiments/so101/env.py` | **`SO101Env` 核心**（~640 行）：观测/命令/模式切换/力矩管理/键盘 toggle |
| `hil-serl/examples/experiments/so101/config.py` | `TrainConfig` + learner 侧 `_FakeSpacesOnlyEnv` |
| `hil-serl/examples/experiments/dummy_test/config.py` | B0.2 用的纯 numpy 假环境 |
| `hil-serl/examples/gen_dummy_demos.py` | 生成假 demo pkl |
| `so101_ros2/so101_bringup/launch/so101_rl_env.launch.py` | **F1 训练 launch**（不含 teleop 组件）|

### 6.4 我修改的既有文件
| 文件 | 改了什么 |
|---|---|
| `so101_ros2/.../bridge/bridge.py` | ① lerobot 导入路径 ② `LeaderBridge` 新增 `command_callback` + `set_torque` service + 限幅 ③ `FollowerBridge` 新增 `set_torque` ④ `max_relative_target` 改 double ⑤ 异常 traceback 补全 |
| `so101_ros2/.../bridge/build.py` | lerobot 导入路径 |
| `so101_ros2/.../config/so101_follower_params.yaml` | 端口 / 标定 / 参数键 / `max_relative_target: 10.0`(float) |
| `so101_ros2/.../config/so101_leader_params.yaml` | 端口 / id / 标定 / `command_topic` / `max_relative_target` |
| `so101_ros2/.../config/so101_usb_cam.yaml` | **修正相机错位**（cam_main=全局 640×480，cam_wrist=腕部 424×240）+ 改 by-id 稳定路径 |
| `so101_ros2/.../launch/so101_teleoperate.launch.py` | 新增 `use_cameras` 开关 |
| `hil-serl/examples/experiments/mappings.py` | 改宽容加载 + 注册 `so101`/`dummy_test` |
| `hil-serl/serl_launcher/setup.py` | **删 `typing`**（病根源头）+ `zmq`→`pyzmq` |
| `hil-serl/serl_launcher/requirements.txt` | 删 tensorflow/tf-keras，钉死 jax 等版本 |
| `hil-serl/serl_launcher/.../common/typing.py` | 删 `tf.Tensor` |
| `hil-serl/serl_launcher/.../utils/train_utils.py` | `tf.io.gfile` → 标准库 `open` |
| `hil-serl/serl_launcher/.../wrappers/video_recorder.py` | `tf.io.gfile` → `os`/`open` |
| `hil-serl/examples/train_rlpd.py` | 加了 `SERL_DEBUG_SYNC` 诊断输出（默认关闭）|

### 6.5 文档
| 文件 | 内容 |
|---|---|
| `ROS2环境搭建-RoboStack方案.md` | 阶段 A 完整记录（8 个坑 + 节点链路图）|
| `云端训练方案-B阶段.md` | 云端架构 + 带宽估算 + B0~B3 路线 + B0.1/B0.2 完成记录 |
| `HIL-SERL精密接触型插接任务.md` | 原始参考文档（用户提供，未改）|

---

## 7. 常用命令

### 7.1 环境
```bash
source /home/nkzwchen/pi_project/so101_env.sh
# 期望输出：python 3.12.14 / ROS_DISTRO humble / lerobot 0.5.1 / torch 2.10.0+cu128 True
```

### 7.2 纯遥操作（阶段 A，验证链路）
```bash
ros2 launch so101_bringup so101_teleoperate.launch.py teleop_mode:=real expert:=human
# 没接相机时加 use_cameras:=false
```

### 7.3 RL 训练环境（F1，不含 teleop 组件）
```bash
ros2 launch so101_bringup so101_rl_env.launch.py use_cameras:=false
```

### 7.4 编译
```bash
cd /home/nkzwchen/pi_project/ros2_ws
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
source install/local_setup.bash
```

### 7.5 B0.2 双进程（假环境验证）
```bash
bash /home/nkzwchen/pi_project/b02_run.sh learner
bash /home/nkzwchen/pi_project/b02_run.sh actor
bash /home/nkzwchen/pi_project/b02_run.sh stop
```

### 7.6 排障小抄
```bash
# 相机设备映射
ls -la /dev/v4l/by-id/
# 串口占用
fuser /dev/ttyACM0 /dev/ttyACM1
# ros2 pkg list 在本机会卡 → 改用文件系统
ls $CONDA_PREFIX/share/<pkg>/package.xml
# 终端被 cbreak 搞乱
stty sane
```

---

## 8. 下一步：B0.6 详细待办

### 8.1 立即要做的（按顺序）
```
[ ] 1. 验证相机：启动 so101_rl_env.launch.py use_cameras:=true
        确认：cam_main 出 640×480 全局画面，cam_wrist 出腕部画面
        确认：日志里的 [cam_main] 图像编码 = ?  （rgb8 还是 bgr8）
[ ] 2. 录 20 条 demo（Leader 臂遥操作）
        需要写：把 SO101Env 的 transition 存成 HIL-SERL pkl 格式
        （参考 gen_dummy_demos.py 的输出格式）
[ ] 3. 采 classifier 数据
        适配 record_success_fail.py 到 SO101Env（它依赖 env 接口）
        注意：`experiments/<exp>/classifier_data/` 是当前工作目录相对路径
[ ] 4. 训 classifier：train_reward_classifier.py --exp_name so101
        目标：测试集准确率 >90%
[ ] 5. 写 SO101RewardClassifier wrapper
        参考 franka_env/envs/wrappers.py 的 MultiCameraBinaryRewardClassifierWrapper
        接进 SO101Env.compute_reward
[ ] 6. 真机跑 RLPD（Actor + Learner 双进程）
```

### 8.2 需要用户确认的 3 个问题
| # | 问题 | 影响 |
|---|---|---|
| Q1 | **`TARGET_JOINT_POSE` 要不要填？** 或直接上 classifier | 不填则 reward 恒 0，RL 学不到东西 |
| Q2 | **demo 数据从哪来？** ①Leader 遥操作录 ②从已有 LeRobot dataset 转 ③先用假 demo | `--demo_path` 是硬断言 |
| Q3 | **先跑通链路还是直接上 classifier？** | 建议先跑通（可用假 demo + 关节位姿 reward）|

### 8.3 已知的未完成项
| 项 | 说明 |
|---|---|
| `SO101RewardClassifier` wrapper | 还没写（B0.6-5）|
| demo 录制脚本 | 还没写（B0.6-2）|
| `record_success_fail.py` 适配 | 还没做（它依赖 `CONFIG_MAPPING` + env 接口）|
| FK / `tcp_pose` 观测 | 留了接口但未实现（需 `placo`）|
| 云端 Learner 环境 | 未搭建（方案已写好）|
| 相机实际 encoding | **未实测**，代码已做兼容 |
| 8GB 显存约束 | RLPD 超参需压小（`batch_size=64`，云端可 256）|

---

## 9. 云端训练方案要点（`云端训练方案-B阶段.md`）

**架构**：Actor 本地（连机器人）+ Learner 云端，走 agentlace ZMQ。

**网络事实（读源码得到）**：
| 项 | 值 |
|---|---|
| 通道 | ZMQ：`TrainerServer` 5588（数据上行）+ 5589（权重下行）|
| 上行时机 | **每 episode 结束一次**（`train_rlpd.py:215` 的 `client.update()`）|
| 上行内容 | 增量 transitions（含图像）|
| 压缩 | **LZ4** |
| 下行时机 | 每 `steps_per_update=50` 步广播网络参数 |
| 上行带宽 | **1~3 Mbps**（很低，不成问题）|
| 下行带宽 | **16~48 Mbps**（比上行重，不够就调大 `steps_per_update`）|

**必做**：
- ⚠️ **必须走加密隧道**（Tailscale/WireGuard/SSH）—— ZMQ 是明文，不能暴露公网
- 云端用 **Python 3.10**（不是 3.12）
- `agentlace` 要从 git 装（不在 PyPI），且**要绕过坏的 git 代理**

---

## 10. 参考文档地图

| 文档 | 看什么 |
|---|---|
| `HIL-SERL精密接触型插接任务.md` | **HIL-SERL 原理与流程**（reward classifier 规格、干预时间线、监控指标）|
| `issac_lib/learning.md` | LeIsaac 仿真、任务清单、SO101 资产 |
| `data_info.md` | HDF5 ↔ LeRobot 数据集格式与转换 |
| `可执行方案.md` | OpenPI 远程推理方案（另一条路线）|
| `so101_ros2/README.md` | ROS2 包结构与安装 |
| `so101_ros2/docs/` | controller / hardware_interface 架构 |
| `云端训练方案-B阶段.md` | 云端训练完整方案 |
| `ROS2环境搭建-RoboStack方案.md` | 阶段 A 的坑与修复记录 |

---

## 11. 新对话开场建议

把这段贴给新对话：

```
继续 SO101 HIL-SERL 项目。请先读 /home/nkzwchen/pi_project/交接文档-SO101-HIL-SERL.md
获取完整上下文。

当前进度：阶段 A 完成、阶段 B0（B0.1~B0.5 + L1/L2）完成。
下一步：B0.6 —— 验证相机 → 录 demo → 训 reward classifier → 接进 env → 真机跑 RLPD。

关键环境：source /home/nkzwchen/pi_project/so101_env.sh
任务定义：PickOrange（把橙子放进盘子），成功判据见交接文档 §5。

我想先从 [相机验证 / demo录制 / classifier] 开始。
```

---

## 12. 安全提醒（真机操作）

1. **训练前确认机械臂周围无障碍物**，手放在急停旁
2. **模式 A 下 leader 会被电机驱动** —— 不要在此模式下硬推 leader（会和电机较劲）
3. **接管前必须先按空格**（env 会自动关掉 leader 力矩）
4. `max_relative_target=10.0`(follower) / `20.0`(leader) 是单步限幅兜底，**不要设成 0**（=不限）
5. **训练脚本被强杀时终端可能残留 cbreak** → `stty sane`
6. 脚本收尾会把力矩恢复，但**强杀时不会** → 必要时手动调 `/leader/set_torque`、`/follower/set_torque`
