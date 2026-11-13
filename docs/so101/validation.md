# 本次离线验证（2026-11-13）

使用 Python 3.12 / 已有 hil-serl 环境，所有控制均为模拟或隔离的生产方法测试，没有串口、相机或云端训练操作。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  so101_tools/test_joint_control.py so101_tools/test_human_reward.py \
  so101_tools/test_online_training.py so101_tools/test_async_recorder.py \
  so101_tools/test_training_keys.py so101_tools/test_fullframe_replay.py \
  so101_tools/test_takeover_settings.py so101_tools/test_control_panel.py
python so101_tools/test_record_flow.py
```

结果：**66项通过**；另行运行的录制流程场景全部通过。存在 Gymnasium 包装器访问、JAX 旧接口的弃用警告，不属于本轮错误修复范围。测试输出见 [validation 日志](validation-20261113.txt)。

本轮将旧四维关节测试更新为当前三维控制契约，保留目标累积、冻结关节、切换不跳变、错误动作拒绝和失效反馈拒绝等断言。不是通过放宽生产保护来令测试通过。

已核验：人工成功帧奖励与 mask、暂停边界、经验收录与预算、异步写盘、停止/接管按键、回放归位与反馈记录。没有重新验证真机回放效果、策略成功率、收敛时间或云端 GPU 吞吐。模拟流程中打印的运行频率不是硬件频率。
