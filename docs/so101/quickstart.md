# SO101 操作入口

从本仓库根目录执行。GPU 环境使用 JAX / serl_launcher / Agentlace；真机端额外需要 ROS2 图像、LeRobot、Tk/OpenCV GUI。现有环境参考 [环境检查脚本](../../so101_tools/check_cloud_env.py)、[依赖](../../serl_launcher/setup.py) 和配套 so101-ros2 文档。云端离线训练无需 ROS 或 USB。

## 设备配置

`native_io.py` 包含本机串口和标定位置，`so101_joint_pose_alignment.json` 记录标定哈希和主从臂偏置；移植时必须对自己的设备重做标定和对齐检查，不能复制其他机械臂的偏置。相机 ROI 由 `camera_observation.py` 及同目录 JSON 载入。不要同时启动占用两臂串口的 ROS 桥和原生录制器。

## 示范录制与标注

相机终端加载自己的 ROS 工作空间：

```bash
ros2 launch so101_bringup so101_record_cameras.launch.py
```

录制终端：

```bash
python so101_tools/record_demo_so101.py --control-hz 10 --episodes 3 --note demos
python so101_tools/label_demos.py recordings/<session>/*.pkl
```

C 开始；D 结束；人工退出接触并回程后 C 保存。原始数据保留，Q 标记排除。初始位置默认有前三关节 ±1.5° 随机化；`--no-randomize` 关闭。纯示范默认100步，在线Actor默认20秒执行预算。

## 云端训练 / 段后上传

```bash
export PYTHONPATH="$PWD/serl_launcher:$PWD/examples:${PYTHONPATH:-}"
python so101_tools/train_human.py train --data /path/reviewed   --out training_runs/run1 --continuous --update-schedule data-budget   --steps 10 --updates-per-transition 0.25 --max-runtime-seconds 3600
```

这是明确指定的起步配置，不代表此前真机实验的最佳值。`--steps` 是更新周期，不是遍历整个数据集的 epoch。需要一直更新时选择 `--update-schedule asynchronous`；预算模式则用完等待新数据。恢复使用 `--resume <checkpoint>` 并保持与存档配置一致。

建立 SSH 隧道后，在本地执行（地址与目录替换为自己的服务）：

```bash
ssh -N -L 5688:127.0.0.1:5688 -L 5689:127.0.0.1:5689 <user>@<host>
python so101_tools/train_human.py status --port 5688
python so101_tools/train_human.py actor --port 5688   --upload-host <user>@<host> --upload-dir /path/reviewed   --record-args --episodes 1 --note online
```

上传使用非交互 SSH，需预先配置密钥或已认证的 ControlMaster（用 `--ssh-control-path` 指定）；不在命令或仓库里保存密码。Actor 在窗口打开前检查权重并预热；本段结束、人工标注确认后上传，下一段边界更新模型。可以跳过提交，原始记录保留。

## 仅动作录制与回放

```bash
python so101_tools/record_actions_replay.py --control-hz 10 --episodes 1 --note actions_replay
```

不需要图像。人工退出接触后，在回放确认窗口按 C，按受限速度恢复本段起点再回放。主臂映射对齐容差10°，从臂起点验收3°；指令跟踪保护仍独立检查。D 停止，X 退出。回放文件含 source_frame 和双臂反馈，标为非专家，不应直接导入原训练池。

硬件命令需要现场观察、正确标定与运动空间。本次仓库整理仅做离线检查，未执行这些硬件命令。
