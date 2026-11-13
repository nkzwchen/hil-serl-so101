# SO101 实现与验证对应

核对日期：2026-11-13；无需云端 GPU 的软件检查与历史真机记录分开列出。

| 工作 | 对应实现 / 验证 | 范围 |
|---|---|---|
| 三维相对动作与共用入口 | [joint_control](../../examples/experiments/so101/joint_control.py)、[测试](../../so101_tools/test_joint_control.py) | 请求动作、执行增量、受限原因；其余关节冻结 |
| 两次空格接管与段间权重检查 | [录制器](../../so101_tools/record_demo_so101.py)、[键盘测试](../../so101_tools/test_training_keys.py) | 第一次暂停，第二次更新差分基准；暂停不生成虚构动作 |
| 人工奖励 | [实现](../../examples/experiments/so101/human_reward.py)、[测试](../../so101_tools/test_human_reward.py) | 成功帧对应前一动作；成功 mask=0；普通失败截断保留自举 |
| 在线收录 / 训练预算 | [online_training](../../so101_tools/online_training.py)、[测试](../../so101_tools/test_online_training.py) | 整段标注后提交，不是未审核动作逐步进入 learner |
| 归位及双臂回放 | [扩展](../../so101_tools/record_fullframe_replay.py)、[测试](../../so101_tools/test_fullframe_replay.py) | 自动恢复本段起点前必须人工退出接触；回放不是专家示范 |
| 真机视频 | [来源与限制](video.md) | 展示装置和运动，无逐帧模式标签，不能推出纯策略成功率 |
| 接管前 3 步惩罚 | 当前 human_reward 未实现 | 需要单独设计、实现和对照；不宣称历史实验已完成 |
| 50%→60%、收敛缩短30% | 待补原始评测与训练预算记录 | 2026-11-13 用户确认暂无可提供的对应记录 |

## 补测协议

先固定插入完成条件、20秒执行预算是否包含暂停、是否允许人工接管；分别统计纯策略成功率和人工协助完成率。每段保存 trial、checkpoint、数据版本、成功标注、接管次数和时长、限制/质量标记与视频链接。收敛速度应预先定义目标成功率并按环境步数、更新次数或墙钟时间分别比较，使用多个随机种子，不能从单次演示推断。

项目叙述中的“试管夹取”与本实现的固定夹爪插接任务需区分。Git 提交日期是代码记录时间，不自动确定此前项目起止日期。
