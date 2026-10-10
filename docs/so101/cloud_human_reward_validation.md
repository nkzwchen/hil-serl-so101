# SO101 云端人工奖励训练验证（2026-10-10）

完成云端训练端开发与离线联网验证；设备与相机未连接，本次未进行真机运动。训练数据使用用户授权的 97% 时间点粗标签，不能据此宣称策略任务成功或奖励已人工逐帧确认。

## 结果

| 检查 | 实测结果 |
|---|---|
| 真实录制数据 | 10 段，共 1193 条转换后 transition；200 条保留质量标记 |
| GPU | NVIDIA L40S，JAX CudaDevice(id=0) |
| 网络 | 预训练 ResNet-10，cam_main / cam_wrist / cam_main_roi，三维连续 SAC |
| 首次更新 | 前 9 段、1066 条 demo，完成 3 次更新，指标有限 |
| 新段提交 | 本地经 SSH 隧道提交第 10 段，127 条 online |
| 重复提交 | duplicate=true，不重复增加 replay |
| 无标签提交 | 服务端拒绝，没有进入 replay |
| 检查点 | 保存参数、目标网络、优化器、模型 RNG；重新加载动作逐位一致 |
| 恢复训练 | 从 step 3 恢复到 step 5，保留 1066 demo + 127 online，按半批比例采样 |
| 本地参数接收 | 成功，对真实 observation 输出三维有限且处于 [-1,1] 的动作 |
| CPU 推理短测 | 预热后 10 次：中位 24.63 ms，最大 27.02 ms |
| 本地回归 | 人工奖励 6 项、三关节契约 5 项、异步录制 5 项通过；录制状态机包含新增策略模式通过 |

云端 checkpoint_3 对首个 observation 动作为 `[0.5175864,-0.5956804,-0.6489156]`；本地 CPU 同参数输出 `[0.5173466,-0.5957462,-0.6490031]`。CPU/GPU 数值差异最大约 0.00024。CPU 时间不包含相机读取、串口通讯和完整 env.step，不能据此保证真机 30 Hz。

## 实现与限制

- actor 保存原始段，人工在段后标成功帧/失败；不连接奖励分类模型，不将未标注原始 0 奖励直接写入训练 replay。
- 两种控制使用相同相对动作执行器。策略 session 开始前下载权重，session 内固定权重；空格人工接管，RETURN 人工回程，元数据记录策略版本。
- learner 检查录制哈希、人工标签状态、三维动作/观测版本、频率、缩放和 ROI。正式入口默认拒绝粗标签，粗标签策略不能启动硬件 actor。
- 成功帧 k 奖励放在 transition k-1，之后数据只从训练派生数据截去，原始 pkl 不修改。
- 修复本地 NumPy 2 pickle 与云端 NumPy 1.26 的模块路径兼容性，无需升级云端 JAX 依赖。
- 在线提交以完整段去重。恢复时从磁盘原始段重建 replay 并保留 demo/online 分类；采样 RNG 不恢复，不保证逐位续跑。
- 训练以有限更新数运行，更新期间可收在线数据；更新结束后的 serve 窗口只提供通信服务。再次训练需恢复检查点。
- replay 当前保存在内存，尚未进行大规模长时训练压测。硬件整环周期、人工切换时运动行为和任务成功率仍需设备接入后验证。

## 产物

云端代码：`/root/gpufree-data/so101-human-reward`；配套 ROS 代码：`/root/gpufree-data/so101_ros2`。

原始录制：`/root/gpufree-data/demos/2026-10-10_221637_async_trial`。

测试 inbox：`/root/gpufree-data/demos/human_smoke_inbox`。

首轮结果：`/root/gpufree-data/offline_test/human_reward_20261010`。

恢复结果：`/root/gpufree-data/offline_test/human_reward_resume_20261010`，最新 `checkpoint_5.msgpack`。

本地日志：`validation/cloud_human_reward_2026-10-10/`，包括 cloud_train.log、cloud_resume.log、submit.json、submit_duplicate.json、rejected_unlabeled.log、policy_check.log、record_flow_test.log、resume_result.json、receipts.json。

凭据、录制 pkl 和模型文件不入 Git。两个仓库分支均为 `feat/so101-human-reward-3d`。可复现运行命令见仓库 README.md。
