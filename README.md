# SO101 HIL-SERL：人工标注奖励训练

本仓库维护本地录制/actor 与云端 learner 的同一套代码。配套仓库为 `nkzwchen/so101-ros2`，开发分支 `feat/so101-human-reward-3d`。部署时记录并核对两个仓库的 commit；不要复制零散脚本覆盖环境。

## 数据与控制约定

- 动作是 pan、lift、elbow 三维相对关节目标请求，30 Hz、4°/步；以已接受目标为增量基准。人工和模型共用执行器。
- state 为 `(1,18)`，含目标误差、补偿、六关节位置和速度。三路图像为 `(1,224,224,3)`。
- RECORD/RETURN 冻结 wrist_flex、wrist_roll、gripper；归位阶段另有腕部调整流程。
- 录制原始奖励保持 0。每段结束后人工标注，learner 只接收已标注完整段，不使用奖励模型。
- 成功 observation 帧 k 对应 transition k-1 奖励 1，并在该步终止；人工失败整段奖励 0，末步终止。未标注、排除段、缺步段拒收。
- 图像质量标记保留并计数，不强制丢弃；标签本身不能修复过期图像。粗标签默认拒收，只有流程验证显式允许。

## 本地录制与复核

在现有 `hil-serl` 环境和 ROS 相机环境中运行。只有相机使用 ROS；两臂由 LeRobot 原生 IO 独占，不能同时开启串口桥。

```bash
python so101_tools/record_demo_so101.py --episodes 3 --note human_trial
DISPLAY=:0 python so101_tools/label_demos.py /path/session/*.pkl
```

标注器显示最终动作之后的图像，保存录制文件 SHA256。粗标签显示 PROVISIONAL；重新标成功/失败才算人工复核，单纯保存不会变更复核状态。

## 云端 GPU 训练

已有环境：`/root/gpufree-data/conda_envs/so101-hilserl/bin/python`。依赖沿用 serl_launcher / Agentlace / JAX CUDA，无需 ROS 或连接设备。

```bash
export PYTHONPATH="$PWD/serl_launcher:$PWD/examples:${PYTHONPATH:-}"
python so101_tools/train_human.py train --data /path/session --out training_runs/run1 \
  --steps 1000 --batch-size 32 --serve-seconds 3600
```

默认使用预训练 ResNet-10，三路图像 SAC。首次没有在线已标注数据时只采样 demonstrations；收到在线段后每批一半 demonstrations、一半 online。`--steps` 是本次梯度更新数，不是录制步数。每 `--checkpoint-every` 次更新（默认 100 次）及更新结束后保存完整优化器和参数，重新加载并比较动作。`--serve-seconds` 是更新完成后的服务保留时长；服务期间只收段/提供参数，不继续梯度更新，继续训练需再次启动并恢复检查点。当前数据保存在内存，训练规模应按云端 RAM 调整。

```bash
python so101_tools/train_human.py train --data /path/session --out training_runs/run2 \
  --resume training_runs/run1/checkpoint_1000.msgpack --steps 1000 --batch-size 32
```

恢复时模型配置必须一致；采样 RNG 不恢复，因此不是逐位相同的训练续跑。收录回执和指标保存在输出目录。数据/标签独立于 Git 上传，凭据不写入仓库。

粗标签仅验证链路：上述 train 命令增加 `--allow-provisional`，所得策略禁止驱动 actor。

## 本地到云端通信与人工奖励循环

SSH 隧道使用两端相同端口。Agentlace 仅供可信 SSH 隧道访问，不要公开其端口。

```bash
ssh -N -L 5688:127.0.0.1:5688 -L 5689:127.0.0.1:5689 -p SSH_PORT USER@HOST
python so101_tools/train_human.py status
python so101_tools/train_human.py policy-check
```

真机未来连接并完成验收后：

```bash
python so101_tools/train_human.py actor \
  --upload-host USER@HOST --upload-port SSH_PORT \
  --upload-dir /path/to/learner/inbox \
  --record-args --episodes 3 --note policy_trial
```

`--upload-dir` 必须与当前 learner 的 `--data` 是同一目录。SSH 需已配置免交互登录；也可用 `--ssh-control-path /path/to/socket` 复用已认证的连接。启动时会先检查上传目录，再打开硬件。

标注窗口新增动作曲线：上方显示各维归一化请求动作，下方显示前三关节实测速度模长（deg/s）。白色竖线是当前帧；绿色竖线是已确认成功帧，橙色是粗标签。点击/拖动曲线定位，按 `j` 跳到已标成功帧。所有帧号统一从 0 开始；最终观测没有后续动作，不绘制虚假的零动作。

actor 每段 RECORD 最长 **20 秒**（同时保留 30 Hz × 20 秒 = 600 步的步数上限），人工接管也计入这 20 秒。READY、RETURN、标注及上传不计时。超时结束本段执行并进入人工回程，保存原始数据与 `episode_timeout` 标记，之后仍由人工标成功或失败，不自动赋失败奖励。普通纯人工录制命令维持原有步数预算。

计时使用单调时钟，在循环起点、策略推理后下发动作前和 step 返回后检查。它是软件执行时限，不是能强制中断底层阻塞调用的硬实时急停；已经进入的串口/step 调用需返回后才能完成阶段切换。

每段完整流程：

1. READY 按 `c` 开始；空格切换人工接管，RECORD 按 `d` 结束执行。
2. RETURN 人工回程，按 `c` 完成回程并保存原始段。
3. 自动弹出本段标注窗口，此时控制循环暂停，不会进入下一段归位。选第一次成功的帧按 `s`，或按 `f` 标失败；按 **`w` 确认提交**并关闭窗口。
4. 自动校验人工标签，上传 pkl 与 labels.json，收到 learner 确认后进入下一段准备。

上传采用内容 ID 文件名，避免不同场次 ep_0000 覆盖；远端先写临时文件，再原子改名，最后提交 learner。每段同名 `.delivery.json` 记录 pending_label / pending_upload / pending_submit / submitted / excluded。

关闭窗口视为取消，不算确认。取消、上传失败或 learner 拒收会结束本次 actor 会话，保留数据和待提交状态，不自动开始下一段。标排除或缺步的段仅保留本地，不上传训练。恢复网络后补交：

```bash
python so101_tools/train_human.py retry-delivery \
  --file /path/session/ep_0000.pkl \
  --upload-host USER@HOST --upload-port SSH_PORT \
  --upload-dir /path/to/learner/inbox
```

pending_label 会重新打开窗口；已确认但网络失败的段直接重试上传和提交。已确认接收的相同内容不重复上传，回执丢失重试由 learner 去重。

actor 整场仍固定模型权重；本次实现不改变权重更新策略。learner 必须在线：梯度更新期间提交的数据可参与后续更新；若处于更新结束后的 serve 窗口，需要恢复训练才会学习新段。标注、网络传输都在段边界执行，不插入 env.step 控制循环。

服务端重复提交去重；已收录段改标签需停止并重建 replay，防止新旧奖励混用。训练入口拒绝旧的 `train_rlpd.py --exp_name so101`，避免把原始 0 奖励直接加入训练。

当前云端测试不等同于策略任务成功率或真机安全验收。上游项目文档保存在 README_upstream.md。

## 本次已验证的云端路径（2026-10-10）

- 代码：`/root/gpufree-data/so101-human-reward`（独立 worktree，保留原仓库工作区）。
- 数据：`/root/gpufree-data/demos/human_smoke_inbox`，10 段粗标签，1193 条转换后 transition。
- 最近检查点：`/root/gpufree-data/offline_test/human_reward_resume_20261010/checkpoint_5.msgpack`。
- 验证记录：[云端测试报告](docs/so101/cloud_human_reward_validation.md)。

继续有限步离线测试（不连接机器人）：

```bash
cd /root/gpufree-data/so101-human-reward
/root/gpufree-data/conda_envs/so101-hilserl/bin/python so101_tools/train_human.py train \
  --data /root/gpufree-data/demos/human_smoke_inbox \
  --resume /root/gpufree-data/offline_test/human_reward_resume_20261010/checkpoint_5.msgpack \
  --out /root/gpufree-data/offline_test/human_next_run \
  --steps 100 --batch-size 4 --allow-provisional
```

人工修改标签后，应从头建立正式训练 run，去掉 `--resume` 和 `--allow-provisional`；旧粗标签检查点的奖励含义已经改变，入口会拒绝混用。

## TensorBoard 实时看板

监控进程只读 JSON 日志，不阻塞或重启 learner。安装监控依赖（不需要 TensorFlow）：

```bash
python -m pip install -r so101_tools/requirements-monitor.txt
```

云端另开终端，将 `RUN_DIR` 改为当前 learner 的 `--out` 目录，`DATA_DIR` 改为其 `--data` 目录：

```bash
cd /root/gpufree-data/so101-human-reward
export RUN_DIR=/root/gpufree-data/training_runs/human_reviewed_20261010_j85vk4eh
export DATA_DIR=/root/gpufree-data/demos/human_reviewed_20261010_trial
/root/gpufree-data/conda_envs/so101-hilserl/bin/python so101_tools/training_monitor.py \
  --run "$RUN_DIR" --data "$DATA_DIR"
```

监控每 2 秒读取新增日志，旧日志自动回填。当前 learner 更新结束后，训练曲线不再增加；服务接收新人工标签时，episode 指标仍会更新。监控默认每次启动创建新的事件子目录；只启动一个监控进程，避免同一数据出现多个 run。

云端另一个终端启动网页：

```bash
/root/gpufree-data/conda_envs/so101-hilserl/bin/python -m tensorboard.main \
  --logdir /root/gpufree-data/training_runs \
  --host 127.0.0.1 --port 6006 --reload_interval 2
```

本次看板已启动，本地访问 **http://localhost:16006**（本地 16006 转发到云端 6006）。新会话需要转发端口时：

```bash
ssh -N -L 16006:127.0.0.1:6006 -p SSH_PORT USER@HOST
```

已有 SSH ControlMaster 时，也可以使用 `ssh -S /tmp/so101_train.sock -O forward -L 16006:127.0.0.1:6006 -p SSH_PORT USER@HOST` 加入端口转发，无需重开训练连接。

| TensorBoard 标签 | 含义 / 横轴 |
|---|---|
| train/actor/actor_loss、train/critic/critic_loss | 策略/价值损失；训练更新 step |
| train/critic/predicted_qs、target_qs | batch 平均预测 Q 和目标 Q；训练 step |
| train/critic/rewards | batch 平均奖励，不是成功率；训练 step |
| train/actor/temperature、entropy | SAC 温度与熵；训练 step |
| episodes/demo/* | 初始演示池统计；该池累计接收 episode 数 |
| episodes/online/* | 后续接收段统计；该池累计接收 episode 数 |
| capture_intervention_fraction | 全部原始 RECORD 步中的接管占比，0..1；不是只算成功帧前的裁剪部分 |
| capture_duration_s、capture_steps、capture_timeout | 每段耗时、原始步数、是否超时；缺失字段不伪造为 0 |
| human_return | 人工标成功为 1、失败为 0；与训练 batch reward 分开 |
| success_rate、last20_success_rate | 有效人工复核段的累计/最近 20 段成功率；不含粗标签 |
| reviewed_count、last20_count | 对应成功率的样本数 |
| autonomous_success_rate、autonomous_reviewed_count | 有策略元数据且全段无接管的人工复核段；零样本时只显示 count=0，不编造成功率 |
| timeout_rate | 已知超时状态的接收段中超时比例 |

成功率在人工标签提交后更新；已排除、未提交/被拒收的段不在 learner 接收统计中，因此看板不是所有尝试的无偏评估。最近 20 段按接收顺序计算，不按录制时间排序。不同模型版本暂未进一步分组，无接管成功率也不替代受控评测。

云端看不到还留在本地的待标注文件。若需查看本地待标注/待上传数量，在本地运行同一监控命令，附加 `--recordings /path/to/actor/session`；`local_queue/*` 横轴为队列变化次数。没有 delivery 状态的旧录制计入 untracked，不假定为未标注。`--once` 可仅导出一次。
