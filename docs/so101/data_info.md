# HDF5 与 LeRobot 数据集说明

本文说明 `LeIsaac-SO101-PickOrange-v0` 任务中，从 IsaacLab/LeIsaac HDF5 数据采集，到后续转换为 LeRobot Dataset v3 的数据格式、字段映射、相似点和差异。

## 推荐命令

在 Isaac/LeIsaac 运行环境中先采集 HDF5：

```bash
python scripts/environments/teleoperation/teleop_se3_agent.py \
  --task=LeIsaac-SO101-PickOrange-v0 \
  --teleop_device=so101leader \
  --remote_endpoint=tcp://localhost:5556 \
  --num_envs=1 \
  --device=cuda \
  --enable_cameras \
  --record \
  --dataset_file=./datasets/pick_orange_act_run01.hdf5 \
  --step_hz=30
```

将 HDF5 文件转换为 LeRobot Dataset v3：

```bash
python scripts/convert/isaaclab2lerobotv3.py \
  --task_name=LeIsaac-SO101-PickOrange-v0 \
  --repo_id=local/pick_orange_act_run01 \
  --fps=30 \
  --hdf5_root=./datasets \
  --hdf5_files=pick_orange_act_run01.hdf5 \
  --task_description="Pick the orange to the plate" \
  --device=cuda
```

转换脚本期望使用 `lerobot==0.4.2` 和 `numpy==1.26.0`。不过，把 LeRobot 直接安装进 Isaac Sim 运行环境可能引入依赖冲突。更稳妥的流程是：先保证 HDF5 采集稳定完成，再在单独环境或后处理阶段转换为 LeRobot 数据集。

## HDF5 数据格式

HDF5 是 IsaacLab/LeIsaac 原生录制格式。它把完整仿真 episode 存在顶层 `data` group 下。

典型结构如下：

```text
data/
  attrs:
    total
    env_args
  demo_0/
    attrs:
      num_samples
      seed
      success
    actions
    processed_actions
    obs/
      joint_pos
      joint_vel
      joint_pos_rel
      joint_vel_rel
      actions
      wrist
      front
      ee_frame_state
      joint_pos_target
    states/
      ...
    initial_state/
      ...
  demo_1/
    ...
```

对 ACT/LeRobot 训练最关键的字段：

```text
actions
obs/joint_pos
obs/front
obs/wrist
success
```

在这个任务里，`front` 和 `wrist` 是来自 `TiledCameraCfg` 的 RGB 相机观测。LeIsaac 单臂任务模板中，它们配置为 `640x480`、`rgb`，相机更新周期为 `update_period=1/30.0`。

HDF5 不只保存训练策略需要的字段。它还可能包含仿真状态、初始状态、处理后的动作、关节速度、相对关节位置、末端执行器位姿状态、目标关节位置等。因此 HDF5 更适合做 replay、debug 和后续格式转换。

## LeRobot Dataset v3 格式

LeRobot 数据集是面向训练和推理的格式。它更关注逐帧样本和 feature metadata，而不是保留完整的原始仿真 episode 结构。

对于 SO101 PickOrange 任务，转换后的 LeRobot frame 由 `env.cfg.build_lerobot_frame(...)` 构建，包含：

```text
observation.state
observation.images.front
observation.images.wrist
action
task
```

主要 feature 含义：

| LeRobot key | 含义 | Shape |
| --- | --- | --- |
| `observation.state` | SO101 关节位置，使用 LeRobot motor scale | `(6,)` |
| `observation.images.front` | front 相机 RGB 图像 | `(480, 640, 3)` |
| `observation.images.wrist` | wrist 相机 RGB 图像 | `(480, 640, 3)` |
| `action` | SO101 动作，使用 LeRobot motor scale | `(6,)` |
| `task` | 语言任务指令 | string |

SO101 的 6 个关节字段名为：

```text
shoulder_pan.pos
shoulder_lift.pos
elbow_flex.pos
wrist_flex.pos
wrist_roll.pos
gripper.pos
```

LeRobot features 由 `build_feature_from_env(...)` 根据环境自动创建。相机 feature 来自 scene sensors，数据集 FPS 由转换命令中的 `--fps` 决定。

## 转换映射

转换脚本主要执行以下步骤：

1. 使用 `parse_env_cfg(...)` 加载任务配置。
2. 使用 `get_task_type(...)` 推断任务类型。
3. 创建环境，用于构建正确的 LeRobot feature schema。
4. 使用 `HDF5DatasetFileHandler` 打开 HDF5 文件。
5. 遍历 HDF5 中的 `demo_*` episode。
6. 跳过 `success` 属性为 false 的 episode。
7. 跳过帧数少于 10 的 episode。
8. 对每条有效 episode 跳过开头 5 帧。
9. 将剩余 HDF5 frame 转换成 LeRobot frame。
10. 使用 `dataset.save_episode()` 保存转换后的 episode。

字段映射：

| HDF5 来源 | LeRobot 目标 | 说明 |
| --- | --- | --- |
| `obs/joint_pos[t]` | `observation.state` | 从 LeIsaac radians/USD joint scale 转为 LeRobot motor scale |
| `obs/front[t]` | `observation.images.front` | RGB 图像，以 numpy array 形式写入 |
| `obs/wrist[t]` | `observation.images.wrist` | RGB 图像，以 numpy array 形式写入 |
| `actions[t]` | `action` | 当 action 维度匹配 SO101 6 个关节时，转为 LeRobot motor scale |
| task config 或 `--task_description` | `task` | `--task_description` 会覆盖任务默认文本 |
| `success` episode attr | 仅用于过滤 | 失败 episode 不会被转换 |

action/state 的数值转换会使用 LeIsaac robot limits 和 SO101 motor limits。HDF5 中的关节位置和动作是 Isaac/LeIsaac 表示，关节值通常是 radians。转换到 LeRobot 后，这些值会变成 LeRobot policy 使用的 SO101 motor-style range。

## 相似之处

两种格式表达的是同一批 demonstration 内容：

- PickOrange 任务的一条或多条 episode。
- 时间对齐的 observations 和 actions。
- 两路 RGB 相机视角：`front` 和 `wrist`。
- 6 维 SO101 robot state。
- 6 维 SO101 action。
- episode 级别的成功/失败信息。不过在当前转换流程中，LeRobot 只保留成功 episode。

如果采集时使用 `--step_hz=30`，转换时使用 `--fps=30`，两种格式描述的是同一个标称 30 FPS 的行为序列。

## 主要区别

| 维度 | HDF5 | LeRobot v3 |
| --- | --- | --- |
| 主要用途 | 仿真 replay、debug、原始数据归档 | 策略训练和推理 |
| 组织方式 | 一个 `.hdf5` 文件中包含多个 episode group | LeRobot dataset directory/repo，包含 metadata、episodes 和 media |
| 字段范围 | 保存较多仿真字段和策略字段 | 主要保留模型训练/推理需要的字段 |
| episode 过滤 | 可以保存所有 episode，包括失败 episode | 当前转换流程只保留成功 episode |
| 开头帧处理 | 原样保存 | 转换时跳过前 5 帧 |
| state/action 数值尺度 | Isaac/LeIsaac 表示 | LeRobot SO101 motor scale |
| 图像存储 | 存在 HDF5 array 中 | 存为 LeRobot image/video features |
| 语言指令 | 不一定是核心训练字段 | 存为 `task` |
| 依赖风险 | 在 Isaac/LeIsaac runtime 中稳定工作 | 需要 LeRobot 相关依赖 |

## 实践建议

当 Isaac Sim 环境对 Python 依赖比较敏感时，建议使用 HDF5 作为采集格式。它是 teleoperation 采集和 replay 最稳的格式。

转换后的 LeRobot 数据集更适合 ACT 训练和远程推理。转换结果应当匹配 `policy_inference.py --policy_type=lerobot-act` 期望的模型输入输出：`observation.state`、相机观测、语言任务和 action chunks。

为了和相机 FPS 以及 ACT 训练保持一致，建议使用：

```text
recording: --step_hz=30
conversion: --fps=30
inference: --step_hz=30
```

我读了 [data_info.md (line 1)](/home/nkzwchen/pi_project/data_info.md:1)。核心可以这样理解：HDF5 是按 episode/demo 保存的原始采集包，LeRobot v3 是把它拆成“逐帧表格 + 视频文件 + 元数据索引”的训练格式。
以你当前 LeRobot 数据集为例，结构大概是：
dataset/
  data/chunk-000/file-000.parquet
  videos/observation.images.cam_main/chunk-000/file-000.mp4
  videos/observation.images.cam_wrist/chunk-000/file-000.mp4
  meta/info.json
  meta/stats.json
  meta/tasks.parquet
  meta/episodes/chunk-000/file-000.parquet
对应关系如下。
LeRobot 文件	作用	大致对应 HDF5 内容
meta/info.json	数据集总说明：fps、robot_type、总 episode/frame 数、feature schema、数据/视频路径模板	HDF5 的整体结构描述，类似 data.attrs.env_args、相机/状态/action 的 schema；但不是原样保存
meta/tasks.parquet	语言任务表，把 task 文本映射到 task_index	--task_description 或任务配置里的 task 文本
meta/stats.json	全局统计量：mean/std/min/max/qxx，用于训练归一化	HDF5 没有直接等价物，是转换后从 actions、obs/joint_pos、图像等计算出来的
meta/episodes/...parquet	每个 episode 的索引：长度、任务、在 data parquet 里的起止行、在 mp4 里的时间范围、episode 级统计	HDF5 的 data/demo_0、demo_1 这些 episode group，以及 num_samples
data/chunk-000/file-000.parquet	逐帧低维数据表：action、observation.state、timestamp、frame_index、episode_index、task_index	demo_x/actions[t]、demo_x/obs/joint_pos[t]，以及转换生成的时间戳和索引
videos/observation.images.cam_main/...mp4	主视角相机视频，多个 episode 可以拼在同一个 mp4 里	HDF5 里的 demo_x/obs/front[t]，或你实际命名中的 main camera
videos/observation.images.cam_wrist/...mp4	腕部相机视频	HDF5 里的 demo_x/obs/wrist[t]

你文档里写的 HDF5 关键字段是：
data/demo_0/actions
data/demo_0/obs/joint_pos
data/demo_0/obs/front
data/demo_0/obs/wrist
data/demo_0.attrs.success
转换到 LeRobot 后主要变成：
actions[t]        -> data parquet: action
obs/joint_pos[t]  -> data parquet: observation.state
obs/front[t]      -> videos/observation.images.cam_main 或 front
obs/wrist[t]      -> videos/observation.images.cam_wrist 或 wrist
success           -> 只用于过滤，失败 episode 通常不写入 LeRobot
task_description  -> meta/tasks.parquet + data parquet 里的 task_index
还有一些 HDF5 里有、但 LeRobot 训练集默认不会保留的内容：
obs/joint_vel
obs/joint_pos_rel
obs/joint_vel_rel
obs/ee_frame_state
obs/joint_pos_target
processed_actions
states/
initial_state/
seed
success
这些更偏 replay/debug/仿真恢复。LeRobot ACT 训练通常只需要 observation.state、相机图像、action、task，所以转换时会丢掉很多仿真内部状态。
一个容易混淆的点：LeRobot 的 data/*.parquet 里通常不直接存图像数组。图像会被编码成 videos/...mp4，然后通过 meta/episodes 里的时间范围和 timestamp/frame_index 对齐回来。也就是说，LeRobot 不是“一个 episode 一个文件”，而是多个 episode 拼进共享 parquet/mp4，再用 metadata 还原 episode 边界。