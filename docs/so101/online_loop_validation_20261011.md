# 人工奖励在线训练闭环验证（2026-10-11）

本次补齐已审核数据接收后自动训练、接管数据额外进入示范池、段间权重刷新。未启动机械臂；所有动作输出检查都使用离线观测。

## 已完成的行为

- 连续 learner 根据新增在线 transition 提供训练额度。正式运行比例 0.25 周期/transition，每周期 2 次 Critic 更新、1 次 Actor/温度更新。无额度时等待人工标签，重复提交不增加额度。
- 默认 batch 64、CTA 2、discount 0.99 来自配置；写入 learning_config 和 checkpoint，恢复时校验。采样中在线与示范各占一半，接管数据同时加入两池。
- actor READY 按 c 后、RECORD 计时前拉取新版本；同一 run 且 step 增加才替换，先校验参数结构、形状、有限性及动作输出。网络失败保留旧版；不兼容版本拒绝新段。推理循环无网络请求。每段 meta.policy 保存版本。
- 成功帧 k：仅保留前 k 个 transition，最后 reward=1、mask=0。未成功/超时/人工截断：reward=0，最后 dones=True 但 mask=1，保留下一状态 bootstrap。这里“失败”表示本段没有达成成功，不定义新的物理吸收失败状态。
- 通信保护中断、缺步、明确排除段保留原始文件但不自动入库。原始数据不改写；标签需要人工审核、文件 hash 匹配。图像质量标记照常保留，不因本次改造自动修复或排除。
- SIGTERM/Ctrl-C 在更新周期结束后保存，保存模型、优化器、采样 RNG、额度起点；latest.json 指向完整检查点。接收回执原子写入，learner 输出目录加独占锁。
- 修复云端实测出的 Agentlace 退出竞态：先结束 REP 接收线程，再关闭 ZeroMQ socket，避免关闭时 assertion abort。

## 自动化与云端测试

本地 29 项测试通过；云端 27 项测试通过（云端不运行依赖本地 ROS 环境的 deadline 两项）。覆盖标签准入、success 对齐、截断 mask、接管双池、重复提交、额度恢复、策略版本检查、监控、提交重试和截止时间。

独立云端测试目录 `/root/gpufree-data/offline_test/online_loop_20261011_retry`，端口 5788，与正式服务隔离；测试结束已停止。

1. 恢复正式旧检查点 100，加载 10 段人工示范、937 transitions；初始额度 0，进入 waiting_for_labels。
2. 通过真实 RPC 提交已有人工标注在线段副本，接收 285 transitions，其中 235 个接管 transition 额外加入示范池。
3. 测试比例 0.02 提供 5 周期，GPU 从 100 自动更新到 105，默认 batch 64、CTA 2，指标均有限。
4. CPU 模拟 actor 在整个模拟段内保留 step 100；显式段间刷新后使用 step 105，并经过正式 actor 相同的参数/动作验证函数。离线零观测输出由约 `[-0.01598, 0.47203, -0.11283]` 变成 `[0.17439, 0.40705, 0.08583]`。这只证明权重切换，不是任务表现改善证据。
5. 重复提交返回 duplicate=True，额度仍为 0。
6. 正常退出保存、重新加载 checkpoint_105，保持 step 105、online 285、示范 1172；没有重复赠送训练额度，继续等待。
7. 检查点重载前后动作逐值一致；进程正常退出。

第一次集成测试发现 ZeroMQ 退出问题，修复后完整重跑通过；保留首次失败日志，不将其算作成功。

本地证据：`validation/online_loop_2026-10-11/unit_tests.log`、`cloud_unit_tests.log`、`cloud_lifecycle.json`、`deployment.json`。

## 正式部署与限制

正式 run：`/root/gpufree-data/training_runs/human_online_20261011`。从原 checkpoint_100 显式迁移，保留旧 run 和旧数据；使用已提交的真实人工标签，无新增粗标签。本次 runtime 3600 秒，到期保存退出。恢复及 actor 命令见仓库 README。

仅验证了训练闭环，没有做新的机械臂动作试验，也没有证明自主任务成功率。仍使用确定性策略，未加入随机探索。新增数据额度比例和训练量是保守起点，需要结合后续成功率、接管比例、Q 值和数据质量调整。入库确认不等于训练完成；actor 获取边界时已完成的最新版本，不等待指定更新次数。对同一 run 的硬杀/并发提交恢复不承诺逐位复现；进程正常保存后的模型、优化器与采样 RNG 均保留。

正式迁移后的实测结果：已用 285 条在线数据完成 71 个更新周期，从 step 100 到 171；checkpoint_171 已保存，RPC 状态为 waiting_for_labels、updates_remaining=0。示范池 1172 条、在线池 285 条（235 条接管数据双池复用）。TensorBoard HTTP Scalars 验证了新增 71 个训练点，最后 Critic loss 约 0.005865、Actor loss 约 2.55010；这些数值不等同于任务成功率。

本地经现有 SSH 隧道下载 step 171，用录制观测做 CPU 离线推理通过：中位耗时约 24.44 ms、最大 27.17 ms（10 次测量，不包含相机/串口/完整控制周期）。未执行任何真机动作。对应证据：`live_status.json`、`local_policy_check.log`、`tensorboard.json`。
