# 阶段 B 计划：SO101 + HIL-SERL 云端训练方案

> 前置状态：阶段 A 已完成（Leader 臂可遥控 Follower 臂）
> 本文只做方案设计，**不执行**
> 目标：把 RLPD 的 Learner 放到云端 GPU，Actor 留本地 WSL2 连机器人

---

## 0. 为什么云端训练是可行的（HIL-SERL 原生支持）

HIL-SERL 就是 **Actor-Learner 分布式架构**，而不是单机训练。源码证据：

| 角色 | 组件 | 位置要求 |
|---|---|---|
| **Actor** | `train_rlpd.py --actor` | **必须**在有机器人的机器上（本地 WSL2） |
| **Learner** | `train_rlpd.py`（无 `--actor`） | 只要 GPU，**完全不碰硬件** |

Learner 与机器人的唯一联系是网络。所以"云端训练"需要改的不是算法，而是**网络与运维**。

---

## 1. 网络层事实（读 agentlace 源码得到，非推测）

| 项目 | 事实 | 来源 |
|---|---|---|
| 通道 | ZMQ。Learner 跑 `TrainerServer`：**5588**（数据上行）+ **5589**（权重下行广播） | `launcher.py:233` `make_trainer_config()` |
| 上行时机 | **只在 `client.update()`** → `train_rlpd.py:215`，即**每个 episode 结束发一次增量批** | `train_rlpd.py:215`、`agentlace/trainer.py:324` |
| 上行内容 | `_data_store.get_latest_data(from_id)` → 自上次以来的**增量 transitions**（含图像） | `agentlace/trainer.py:325` |
| 压缩 | **LZ4**：`lz4.frame.compress(pickle.dumps(data))` | `agentlace/internal/utils.py:55` |
| 下行时机 | 每 `steps_per_update=50` 步广播一次网络参数 | `train_rlpd.py:340-342` |
| 断线处理 | `wait_for_server=True`，`timeout_ms=3000` | `train_rlpd.py:129-130` |
| 默认限流 | `rate_limit: None`（**无上限**） | `agentlace/trainer.py:35` |
| 双数据流 | 在线经验 `actor_env` + 干预经验 `actor_env_intvn`，分别上行 | `train_rlpd.py:266-267` |

### 1.1 带宽估算（这是决定方案可行性的核心数字）

单条 transition 的大小（按 HIL-SERL 默认 128×128 图像）：

```
图像: 2 路 × 128×128×3 (uint8) = 98,304 B
proprio + action + reward + mask ≈ 数百 B
合计 单条 ≈ 98.4 KB（未压缩）
LZ4 对 uint8 图像典型 2~3.5x → 单条 ≈ 28~49 KB
```

每个 episode 的上行量（设 episode = 100 步，`MAX_EPISODE_LENGTH=100`）：

```
未压缩: 100 × 98.4 KB ≈ 9.8 MB
LZ4 后:                ≈ 2.8~4.9 MB
```

**换算成持续带宽**（这是最关键的一步）：

| episode 时长（含 reset） | 平均上行带宽 |
|---|---|
| 15 s | 0.19 ~ 0.33 MB/s ≈ **1.5 ~ 2.6 Mbps** |
| 30 s | 0.09 ~ 0.16 MB/s ≈ **0.8 ~ 1.3 Mbps** |
| 60 s | 0.05 ~ 0.08 MB/s ≈ **0.4 ~ 0.7 Mbps** |

> **结论：上行带宽需求极低（个位数 Mbps），家庭宽带上行完全够。**
> 真正的风险不是带宽，而是 **① 上行链路的稳定性 ② ZMQ 无加密 ③ 3000ms 超时下的突发丢包**。

### 1.2 权重下行带宽

```
Student/Actor/Critic 网络参数量级: 数百万参数（ResNet10 编码器 + MLP）
单次广播 ≈ 数十 MB（LZ4 后可能 10~30 MB）
每 50 步一次，50 步 ≈ 5 s → 平均下行 ≈ 2~6 MB/s = 16~48 Mbps
```

> ⚠️ **下行反而比上行重得多**。如果云端上行带宽小（家用宽带上行 30~50 Mbps），
> 广播频率会成为瓶颈。缓解手段：调大 `steps_per_update`（如 100~200），
> 或只在 learner 侧定期存 checkpoint 让 actor 拉取（改架构，不推荐初版做）。

---

## 2. 目标架构

```
┌───────────────────────── 本地 WSL2（必须连机器人）─────────────────────────┐
│  机械臂 + Leader 臂 + 相机                                                  │
│      ↓ 串口 / USB                                                          │
│  so101_ros2 (ROS2 Humble)  ← 阶段 A 已完成                                 │
│      ↓ SO101Env (gym)                                                      │
│  train_rlpd.py --actor                                                     │
│    · 采样 action（本地前向推理）                                            │
│    · 读 /leader/joint_states 做人工干预                                     │
│    · 每个 episode 结束 client.update()  ────┐                              │
└────────────────────────────────────────────┼──────────────────────────────┘
                                             │ ZMQ 5588（上行 transitions，LZ4）
                                             │ ZMQ 5589（下行 网络参数）
                    加密隧道（WireGuard / Tailscale / SSH）
                                             │
┌────────────────────────────────────────────┼──────────────────────────────┐
│  云端 GPU 实例                              ↓                              │
│  train_rlpd.py（Learner）                                                  │
│    · ReplayBuffer + DemoBuffer                                             │
│    · RLPD 更新（Ensemble Critic + LayerNorm）                              │
│    · 每 50 步 publish_network()                                            │
│    · checkpoint / WandB 日志                                               │
└───────────────────────────────────────────────────────────────────────────┘
```

**关键设计决策：把 `SO101Env` 放在 Actor 进程内**（而不是走 ROS2 HTTP bridge）。
原因：观测要在本地生成、动作要在本地执行、干预信号也在本地。云端只吃 transition。

---

## 3. 云端资源规格

### 3.1 GPU 选型

| 档位 | 显存 | 适配度 | 说明 |
|---|---|---|---|
| RTX 4090 / 5090 | 24~32 GB | ✅ 首选 | HIL-SERL 论文配置 |
| A100 40G / L40S 48G | 40+ GB | ✅ 有余量 | 贵，对本任务过剩 |
| RTX 4080 / 3090 | 16~24 GB | ✅ 够用 | 性价比好 |
| **RTX 4060（你本地）** | **8 GB** | ⚠️ 勉强 | 本地跑要压 batch_size/分辨率 |
| T4 16G（云上便宜档） | 16 GB | ✅ 够用 | 慢但能跑 |

**注意**：你本地是 RTX 4060 8GB，所以"上云"的收益**不只是快，而是能开大 batch**：

| 超参 | 本地 8GB 建议 | 云端 24GB 建议 |
|---|---|---|
| `batch_size` | 64~128 | 256 |
| `critic_ensemble_size` | 2~3 | 10（论文默认） |
| `critic_subsample_size` | 2 | 2 |
| 图像分辨率 | 128×128（已是默认，别加大） | 128×128 |
| `steps_per_update` | 200（省下行带宽） | 50 |

### 3.2 其他要求

| 项目 | 要求 |
|---|---|
| 系统 | Ubuntu 22.04 或 24.04 + NVIDIA 驱动 + CUDA 12.x |
| Python | **3.10**（跟 HIL-SERL 一致；**别用 3.12**，因为 jax/tf 版本组合在 3.12 下更脆） |
| 磁盘 | ≥ 100 GB（checkpoint + buffer 会持续增长） |
| 网络 | 公网 IP 或可建立隧道；上行 ≥ 50 Mbps（权重广播需要） |
| 存储 | checkpoint 目录要持久化（实例销毁不丢数据） |

---

## 4. 分阶段实施路线

### 阶段 B0：本地补齐依赖 + 无云端跑通（**先做这个**）

**目的**：在引入"网络"这个变量之前，先确认算法侧能跑。

| # | 任务 | 说明 | 状态 |
|---|---|---|---|
| B0.1 | 装 ML 依赖（本机 Actor 侧） | **CPU 版 jax**（方案 A）+ flax/optax/distrax/chex + agentlace | ✅ **完成** |
| B0.2 | 用 `fake_env=True` 空跑双进程 | 不连机器人，验证 Learner+Actor 握手 + 数据流 + 训练循环 | ✅ **完成**（训练跑到 131/200 步） |
| B0.3 | 写 `SO101Env`（gym） | 观测=关节角+图像；动作=关节增量归一化到 [-1,1]；见 §5 | ⏳ |
| B0.4 | 写 `LeaderIntervention` wrapper | 复用现有 `/leader/joint_states`；动作语义必须与 policy 严格一致 | ⏳ |
| B0.5 | 写 reward（先规则版） | 用关节角/位置判成功，先不训 classifier | ⏳ |
| B0.6 | 接真机跑（B0.3~B0.5 完成后） | Actor 用真 SO101Env，Learner 仍在本机（`--ip 127.0.0.1`） | ⏳ |

> **B0.1/B0.2 实测结果**：
> - Actor 侧依赖只用 **CPU 版 jax**（`jaxlib` 87MB vs CUDA 版 5GB）。**本地不需要 CUDA 版** —— 训练在云端，本地只做策略前向推理。
> - 双进程握手成功：Learner 起 `TrainerServer`(5588/5589)，Actor 连上后 `update()=True`，数据流通，Learner 训练循环以 ~1.1 步/秒（CPU）推进。
> - 复现：`bash b02_run.sh learner` + `bash b02_run.sh actor`

---

## 4.1 B0.2 复现细节与本轮踩的坑

### 文件清单

| 文件 | 作用 |
|---|---|
| `actor_requirements.txt` | 本地 Actor 环境依赖（CPU 版 jax，含版本钉死原因） |
| `hil-serl/examples/experiments/dummy_test/config.py` | 纯 numpy 假环境 + 训练配置 |
| `hil-serl/examples/gen_dummy_demos.py` | 生成假 demo 数据（RLPD 离线数据是硬需求） |
| `b02_run.sh` | 双进程启动器（`learner` / `actor` / `stop`） |
| `test_agentlace_link.py` | agentlace 数据通路的隔离测试（排障用） |

### 本轮发现的 6 个坑

| # | 现象 | 根因 | 处理 |
|---|---|---|---|
| 1 | `import experiments.mappings` 直接失败 | 原版无条件 import 4 个 franka 实验，它们依赖 ROS1 `rospy` | 改成**宽容加载**：失败的跳过并记录原因 |
| 2 | `pip install git+https://...agentlace` 失败 | `~/.bashrc` 残留 Clash 代理（`127.0.0.1:7890`），**该代理连不上 GitHub** | 临时 `git config --global --unset http.proxy`，或手动 clone 后本地安装 |
| 3 | `agentlace` 装完 import 不到 | 我用了 `-e`（editable）安装，随后又删掉源码目录 | 改成**非 editable** 安装（源码目录可删） |
| 4 | Learner 报 `wandb.errors.UsageError: No API key` | wandb 默认 `mode="online"` | 加 `--debug`（`train_rlpd.py:50` 已有此 flag → `mode="disabled"`） |
| 5 | Learner 报 `EOFError: input()` | `--checkpoint_path` 目录存在就触发 `input("Press Enter to resume")` | checkpoint 目录与 demo 文件**分开**，且 ckpt 目录不要预先创建 |
| 6 | `--demo_path` 断言失败 | learner 路径硬要求离线数据（`train_rlpd.py:458`） | 用 `gen_dummy_demos.py` 生成（让 env 自己 rollout，保证格式一致） |

### ⚠️ 排障时的观察陷阱（值得记住）

Learner 有**两个**进度条，极易看错：

```
Filling up replay buffer: 0%     ← 等在线数据填够 training_starts（会长时间停在 0%）
learner: 18%|█▊ | 35/200         ← 真正的训练循环（已经在跑）
```

我一度以为卡死，实际是**只 grep 了第一个进度条**。判断训练是否在跑，要看 `learner:` 那个。

### 数据同步的时序特性（不是 bug）

- `client.update()` 第一次调用**必然返回 False** 并打 `Failed to get last update id` ——
  因为 server 端 `last_update_id_map` 还是空的（`agentlace/trainer.py:107` 的条件）。
  但**数据仍会送达 server**（隔离测试已证实）。
- 实际影响：`TrainerClient.__init__` 里那次 `update()` 失败属正常，后续每个 episode 结束的 `update()` 都返回 `True`。

### 阶段 B1：接入云端（同城/同网优先）

| # | 任务 | 说明 |
|---|---|---|
| B1.1 | 云端实例就绪 | 装 Ubuntu + CUDA + conda 环境（同 B0.1 的依赖，Python 3.10） |
| B1.2 | 克隆代码 + 传标定/数据 | `hil-serl` 代码、`lerobot`、以及本地采集的 **demo 数据 pkl**（用于 demo_buffer） |
| B1.3 | **建加密隧道** | **必做**。ZMQ 明文无加密，绝不能直接暴露 5588/5589 到公网。方案：Tailscale（最省事）/ WireGuard / SSH 端口转发 |
| B1.4 | 验证连通性 | 本地 `ping` + `nc -vz <云端隧道IP> 5588` |
| B1.5 | 跨网双进程启动 | Learner 起在云端（绑隧道网卡），Actor 用 `--ip <云端隧道IP>` |
| B1.6 | 观察首轮 episode 上行 | 抓 `ifstat`/`nload` 看实际带宽是否符合 §1.1 估算 |

### 阶段 B2：长跑与稳定性

| # | 任务 | 说明 |
|---|---|---|
| B2.1 | 断线恢复测试 | **故意断网**，看 Actor 是否卡死、恢复后能否续传（`wait_for_server=True` + 3000ms 超时会重试） |
| B2.2 | 心跳 + 安全停机 | **本地必须加**：一旦检测到 learner 失联超过 N 秒，Actor 应停止下发动作（否则机器人继续按旧策略乱动） |
| B2.3 | checkpoint 策略 | 云端定期存 ckpt，并**异步同步回本地**（否则实例销毁=全丢） |
| B2.4 | WandB | 云端配 `WANDB_API_KEY`；或走 `offline` 模式后同步 |
| B2.5 | buffer 落盘 | `buffer_period=1000` 会把 buffer 存成 pkl，注意云端磁盘增长 |

### 阶段 B3：加 reward classifier

| # | 任务 |
|---|---|
| B3.1 | 加第三视角相机（`so101_cameras.yaml` 加一路） |
| B3.2 | `record_success_fail.py` 采集成功/失败帧 |
| B3.3 | `train_reward_classifier.py` 训 ResNet10 分类器（**注意 SO-101 视觉域与 Franka 差异大，必须重采重训**） |
| B3.4 | 在线推理接进 env 的 `compute_reward` |

---

## 5. `SO101Env` 设计要点（B0.3 详述）

### 5.1 观测空间

```python
observation_space = gym.spaces.Dict({
    "state": gym.spaces.Dict({
        "joint_pos": Box(-pi, pi, (6,)),      # 5 关节 + 夹爪
        "joint_vel": Box(-inf, inf, (6,)),
        # 建议补: "tcp_pose": Box(-inf, inf, (7,))  ← 用 URDF 做 FK 得到
    }),
    "images": gym.spaces.Dict({
        "cam_main":  Box(0, 255, (128,128,3), uint8),
        "cam_wrist": Box(0, 255, (128,128,3), uint8),
        # 阶段 B3 再加 "cam_side"
    }),
})
```

**必须删掉** `tcp_force` / `tcp_torque`（SO-101 无 F/T 传感器），
`proprio_keys = ["joint_pos", "joint_vel"]`（+ 可选 `tcp_pose`）。

### 5.2 动作空间

```python
action_space = gym.spaces.Box(-1, 1, shape=(6,), dtype=np.float32)
# 语义：关节增量（相对当前关节角），第 6 维是夹爪
delta = action[:5] * ACTION_SCALE          # 建议 ACTION_SCALE ≈ 2~5 度/步
target = clip(joint_pos + delta, JOINT_LOW, JOINT_HIGH)
```

**为什么用关节增量而不是末端相对平移**（文档 3.4 节的建议）：
- 不需要实现 IK，直接复用阶段 A 已验证的关节级链路；
- 满足"相对量分布集中"这一核心优点；
- 与 Leader 臂干预信号**天然同构**（Leader 也给关节角）。

### 5.3 控制频率

HIL-SERL 硬编码 `hz=10`。你的 ROS2 链路是 25~50 Hz，**env 侧降采样到 10 Hz** 读取最新值即可。

### 5.4 安全（训练时必开）

| 机制 | 做法 |
|---|---|
| 单步限速 | `max_relative_target`（yaml，**必须写 float 如 `10.0`**，见阶段 A 坑 #8） |
| 关节限位 | env 侧 `np.clip` 到 URDF 的关节范围 |
| 工作空间 | 若启用 `tcp_pose`，可用 `ABS_POSE_LIMIT` 类似机制夹住 |
| 急停 | 保留物理急停；另加"learner 失联即停"的软件兜底（B2.2） |

---

## 6. 关键风险与对策

| # | 风险 | 影响 | 对策 |
|---|---|---|---|
| R1 | **ZMQ 明文 + 公网暴露** | 任何人可向 learner 注入假 transition / 窃取权重 | **必须**走 Tailscale/WireGuard/SSH 隧道，禁止直接开端口 |
| R2 | 上行链路抖动 → `client.update()` 超时 | 该 episode 数据丢失（静默） | 抓包确认丢包率；必要时本地做发送队列重试 |
| R3 | 权重下行带宽不足 | 参数广播滞后，actor 用旧策略，训练变慢 | 调大 `steps_per_update`；选上行带宽大的云厂商 |
| R4 | learner 失联但 actor 继续跑 | **机器人按旧策略持续动作，安全风险** | B2.2 心跳 + 自动停止下发 |
| R5 | 云端实例销毁丢 checkpoint | 训练白做 | 持久化卷 + 定期同步回本地 |
| R6 | 代码/数据版本不一致 | 两端 `TrainerConfig.version` 不匹配会报错 | 用 git 固定 commit；两端同一份代码 |
| R7 | Python 3.12 下 jax/tf 组合脆弱 | 装不上或运行崩 | 云端用 **Python 3.10**（跟 HIL-SERL 一致） |
| R8 | 本地 WSL2 网络特殊（NAT） | 云端可能连不回来 | Actor 是**主动连出**方，通常没问题；但隧道要选支持 WSL2 的方案 |
| R9 | demo 数据没同步到云端 | demo_buffer 为空，RLPD 退化成纯 SAC | 启动前确认云端 `demo_buffer` 非空（`len(demo_buffer)`） |
| R10 | 8GB 本地显存 vs 24GB 云端超参不一致 | 两端行为不一致，难复现 | 超参写进 config，两端用同一份，靠 profile 区分 |

---

## 7. 成本估算（粗略）

| 项目 | 估算 |
|---|---|
| 云 GPU（4090/A100 按需） | ¥3~15 / 小时 |
| 一次完整训练（HIL-SERL 论文级 ~3 小时） | ¥10~50 |
| 存储（100GB 持久卷） | ¥5~30 / 月 |
| 隧道（Tailscale 免费档 / VPS） | ¥0~30 / 月 |

> 注意：真机 RL 训练**需要人工全程在场干预**，所以云端省的是"等待时间"，
> 不是"人力时间"。按 3 小时训练算，选按需实例比包月划算。

---

## 8. 建议的执行顺序（决策点）

```
B0（本地，无网络变量）  ← 强烈建议先做完
  └─ 决策点 1：本地 8GB 能跑通吗？超参要压多小？
       ↓
B1（同城云 + 隧道）
  └─ 决策点 2：实测带宽/延迟是否满足？(抓 ifstat)
       ↓
B2（长跑稳定性）
  └─ 决策点 3：是否需要做断线重传/心跳停机？
       ↓
B3（reward classifier + 第三视角相机）
```

### 我的建议

1. **先做 B0**，且必须用 `fake_env=True` 先空跑一遍双进程 —— 这样能在**不连机器人**的情况下验证
   算法链路，风险最低。
2. **B1 先选同城云**（延迟 < 30ms），确认整条链路后再考虑更远的机房。
3. **隧道用 Tailscale**：配置成本最低，且对 WSL2 友好。
4. 如果云端方案遇到太多网络问题，**退路是租一台本地 GPU 机器**（同局域网），
   架构完全不变，只是网络变简单。

---

## 9. 与阶段 A 的衔接

阶段 A 已交付的资产在阶段 B 全部复用：

| 阶段 A 资产 | 阶段 B 用途 |
|---|---|
| `so101_ros2`（ROS2 链路） | `SO101Env` 的底层 |
| `/leader/joint_states` | 人工干预信号源 |
| `so101_env.sh` | Actor 进程的环境 |
| 标定文件 + 端口约定 | 不变 |
| `max_relative_target=10.0` | 训练时的单步限速 |
| 8 个坑的记录 | 避免重复踩 |

**唯一需要新增的硬件**：第三视角相机（阶段 B3 才需要，B0/B1 不需要）。
