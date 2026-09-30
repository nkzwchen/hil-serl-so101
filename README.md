# hil-serl-so101 — SO101 机械臂的 HIL-SERL 训练框架

> 本仓库是 [HIL-SERL](https://hil-serl.github.io/)（Berkeley）的 **SO101 移植版**，
> 用于「**试管插入试管架**」这一精密接触型插接任务。
> 上游原始 README 保留在 [`README_upstream.md`](README_upstream.md)。

---

## 1. 这套系统在做什么

用**人类示范 + 在线强化学习**（HIL-SERL / RLPD）让 SO101 五自由度机械臂学会把试管插进孔里。
人不写奖励函数、不做精确建模，而是**手扶主臂（leader）带一遍**，策略边学边被纠正。

```
┌──────────────── 云端（GPU，训练+推理）────────────────┐        ┌────────── 终端（本机，与设备交互）──────────┐
│  conda env: so101-hilserl                            │        │  conda env: hil-serl (ROS2 Humble)          │
│  train_rlpd.py --learner                             │        │  train_rlpd.py --actor                      │
│    · ReplayBuffer / demo buffer（RLPD 的 prior data）│◄──TCP──│    · SO101Env：读关节 + 相机 + 下发指令      │
│    · 策略网络 + critic，唯一持有参数的一份            │  隧道  │    · 人类接管时上传 (obs, action, reward…)   │
│    · 出 checkpoint                                    │        │    · 录制 demo / 离线标定成功帧              │
└──────────────────────────────────────────────────────┘        └────────────────────────────────────────────┘
```

**关键点：策略只在云端。** 终端不训练、不存网络，只做「采数据 / 执行动作」，
所以终端可以是算力很弱的小机器（本机就是）。
两端代码**完全相同**（同一仓库同一 `main` 分支），区别只在启动参数 `--learner` / `--actor`。

云只对外开放 SSH 端口，agentlace 的 TCP 端口不出网 —— 用 **SSH 隧道**打通（见 §4）。

---

## 2. 目录结构

```
hil-serl-so101/
├── README.md                    ← 本文档
├── README_upstream.md           ← 上游 HIL-SERL 原始 README
├── docs/so101/                  ← 本项目的中文文档（12 篇，含踩坑记录）
├── so101_tools/                 ← ★ 终端侧全部脚本（录制/标定/体检/归位/离线测试）
└── examples/
    ├── train_rlpd.py            ← 入口：--learner / --actor
    ├── gen_dummy_demos.py       ← 造假 demo（RLPD 的 --demo_path 是硬需求）
    └── experiments/
        ├── so101/               ← ★ SO101 环境：env.py / config.py / 运动学 / 起始姿态
        └── dummy_test/          ← 纯 numpy 玩具环境，**不接硬件**测双进程握手
```

`so101_tools/` 里的脚本**自带路径解析**，放在 `pi_project/` 根目录或本仓库
`so101_tools/` 下都能跑（见 `pack_so101_tools.py`）。

---

## 3. 任务与动作空间（改动作维度前必读）

| 项 | 值 |
|---|---|
| 动作维度 | **4** = `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex` |
| 冻结关节 | `wrist_roll`（管子轴对称，转了没意义）、`gripper`（全程夹着试管） |
| 动作语义 | **关节增量**：`target = 实测角 + action × ACTION_SCALE + home_offset` |
| `ACTION_SCALE` | 0.05 rad/步（10 Hz ⇒ 28.6°/s 上限） |
| 控制频率 | 10 Hz |
| 观测 | `state(1,12)` = joint_pos(6)+joint_vel(6)；`cam_main`/`cam_wrist` 各 `(1,224,224,3)` |
| `setup_mode` | `single-arm-fixed-gripper`（决定用 `SACAgent`，**不能**用 learned-gripper） |

⚠️ **三个最容易踩的坑**（详见 `docs/so101/动作空间-冻结维度.md`）：

1. **冻结维要用「实测值」下发，不能用固定目标角** —— 否则零回复力、从臂缓慢下垂。
2. 冻结之后 `setup_mode` 必须换成 `single-arm-fixed-gripper`，否则最后一个动作维会被
   当成离散夹爪头（`SACAgentHybridSingleArm`），语义全错。
3. 动作维度变了，**旧的 demo 全部作废** —— `build_demos.py` 会用真实的
   `action_space` 逐条校验并把旧数据拦下（这是设计，不是 bug）。

---

## 4. 快速开始

### 4.1 云端（训练 + 推理）

```bash
# 一次性：建环境（放到大盘，/ 只有 ~17GB）
cd /root/gpufree-data
git clone <本仓库> so101 && cd so101
bash so101_tools/setup_cloud_env.sh          # 建 conda env: so101-hilserl（jax CUDA）
```

### 4.2 终端（与设备交互）

```bash
# 依赖：ROS2 Humble（RoboStack conda 环境 hil-serl）+ ros2_ws 已 build
source so101_env.sh                          # 起 ROS + 编译好的工作空间
ros2 launch so101_bringup so101_rl_env.launch.py use_leader:=true use_cameras:=true
```

### 4.3 打通云端 ↔ 终端

```bash
# 终端侧：把云端的 agentlace 端口拉到本地（云只开 SSH）
ssh -p <port> -N -L 6000:localhost:6000 root@<cloud-ip> &
```

### 4.4 不接硬件先验证双进程（推荐第一步）

```bash
bash so101_tools/b02_run.sh learner     # 云端或本机
bash so101_tools/b02_run.sh actor       # 另一个终端
```
用的是 `examples/experiments/dummy_test/`（纯 numpy 玩具环境），
把「算法进程间通信」这个变量单独隔离出来。

---

## 5. 数据流水线：录制 → 标定 → 构建 → 训练

```
record_demo_so101.py      label_demos.py          build_demos.py        上传
  录 4 段阶段机     →     人工标「成功帧」   →    稀疏奖励 + 校验   →   --demo_path
  recordings/ep_XXXX.pkl   ep_XXXX.labels.json    demo_data/*.pkl       云端 learner
```

### 5.1 录制（`so101_tools/record_demo_so101.py`）

每段四个阶段，人在终端按单键推进：

| 阶段 | 含义 | 按键 |
|---|---|---|
| `PRECISE` | 自动回随机起点 + 主臂对齐（全自动） | — |
| `READY` | 等你就位 | `c` 开始录制 / `q` 换一个随机起点 / `x` 退出 |
| `RECORD` | **模式 B：主臂接管**，你做插入 | `d` 结束本轮 / `q` **按错了，丢弃本轮** / `x` 退出 |
| `RETURN` | 仍模式 B，把从臂带回起点附近（顺势拔出） | `c` 结束回程 / `q` 本轮标丢弃 |

**每个阶段都只有一次按键**，命令提示就印在终端上（另有 `h` 重印）。
录制**不做成功判断** —— 判断留给离线标定，这样错的判据永远不会丢好数据。

### 5.2 标定（`so101_tools/label_demos.py`）

tkinter 界面（**必须 `DISPLAY=:0`**），逐帧播放 + 放大镜，`s` 标成功帧、`f` 标失败。
标签写到 `<录制文件>.labels.json`，与数据分开存，**可反复重标**。

### 5.3 构建（`so101_tools/build_demos.py`）

- 成功帧 `k` ⇒ 稀疏奖励落在 transition **`k-1`**（`obs_k` 才是成功状态），并截断到那里
- 每条 transition 都对**真实** `observation_space`/`action_space` 逐 key 校验
- 全部失败的段可以整体上传（见 `--allow-frame0` 与 `docs/so101/录制与标定流程.md`）

### 5.4 训练

```bash
# 云端
python examples/train_rlpd.py --learner --demo_path demo_data/so101_Ndemos_*.pkl ...
# 终端
python examples/train_rlpd.py --actor   --ip <云端/隧道地址> ...
```

---

## 6. 已踩过的坑（都在 `docs/so101/` 里有完整记录）

| 现象 | 根因 | 对策 |
|---|---|---|
| 主臂一使能就"跳起来又回去" | `enable_torque()` 不写 `Goal_Position`，跳变 | 使能前先用 `Present_Position` 同步**仅力矩关闭**的电机 |
| 从臂缓慢下垂、起点歪 8.75° | 比例位置环无积分项，稳态有静差 | 归位时把收敛指令存档（`so101_home_cmd.json`），`reset()` 复用 |
| 起点有时**非常高**地收起来 | 「三维框 + IK」有 1 维零空间（转 pan + 手臂反绕到同一末端） | 随机化改成**只转 pan 的水平弧 ±1cm**，不做 IK |
| 录制数据训练侧用不了 | 裸 `SO101Env` 的观测是**嵌套**的，训练要扁平形态 | 录制走 `TrainConfig().get_environment()`（套 wrapper） |
| **81% 的步动作是 0**，而手臂其实动了 | 干预死区 0.05 rad ≈ 2.9°，比 10Hz 单步位移还大 | 录制时把死区设 0 |
| **腕部图像全是斜条纹/彩虹伪彩** | usb_cam 0.8.1 行距补齐（424×3=1272→1280），`Image.step` 却声明 1272 | 腕部相机改 **640×360**（1920 是 128 的倍数，必然对齐） |
| 录出 1 步的空段 | PRECISE 的对齐耗掉了步数预算 | RECORD 开始时清零 `_step_count` + 退段守卫 |

---

## 7. 相关文档

`docs/so101/` 下 12 篇，按需看：

- `交接文档-SO101-HIL-SERL.md` —— **新人从这里开始**
- `动作空间-冻结维度.md` —— 动作维度、冻结、起始随机化（§3.7/§3.8 是两次返工）
- `录制与标定流程.md` —— 流水线、按键、质量自检
- `安全边界-主臂.md` —— **动硬件前必读**（堵转过热、力矩跳变事故）
- `云端训练方案-B阶段.md` —— 云端/终端分工
- `ROS2环境搭建-RoboStack方案.md` —— 环境从零搭

---

## 8. 出处与许可

上游：[rail-berkeley/hil-serl](https://github.com/rail-berkeley/hil-serl)（Apache-2.0）。
本仓库保留上游全部历史与许可；SO101 相关的改动见 `git log --author=nkzwchen`。
