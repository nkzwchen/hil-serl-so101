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
python so101_tools/train_human.py actor --record-args --episodes 3 --note policy_trial
```

actor 在打开硬件前下载本次策略，整场固定权重以避免网络阻塞控制。空格切换人工接管，RETURN 始终人工回程；仍使用录制器原有归位、READY 和数据保存流程。录完人工标注，将 pkl 和同名 labels.json 上传到 learner 的 `--data` 目录（使用唯一文件名，避免不同场次 ep_0000 覆盖），然后：

```bash
python so101_tools/train_human.py submit --file session2_ep_0000.pkl
```

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
