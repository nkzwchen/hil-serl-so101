# 段后自动标注与上传（2026-10-10）

actor 已接入保存后回调：完成 RETURN、原始 pkl 与会话 meta.json 落盘后，再启动单段标注窗口。网络与标注不进入 env.step 控制循环，窗口关闭前不执行下一段归位。

选成功帧按 s，失败按 f，然后 w 确认保存并提交。窗口关闭或 q 表示取消，已有标签也不会因此自动提交。成功奖励仍由人工选择的帧决定，不由模型判断。

配置与命令见仓库 README 的“本地到云端通信与人工奖励循环”。actor 必须传入 upload-host、upload-port、upload-dir，上传目录对应 learner 的 data 目录；支持 ssh-control-path 复用现有已认证 SSH 会话，不保存账号密码。

实现包括：

- 人工标签、原始文件哈希、动作/观测契约校验通过后，才开始上传。
- 内容 ID 命名防止不同场次同名覆盖；文件在远端临时写入、校验 SHA256，完成后原子改名；标签最后上传。
- learner 确认接收后才写 submitted 回执；每段 .delivery.json 保存待标注、待上传、待提交及已提交状态。
- 取消或失败结束本次 actor 会话，保留原始数据供 retry-delivery 补交。未标注会重开窗口，网络失败无需重复标注。
- 排除/缺步段保留本地，不自动上传；未训练或粗标签 smoke 策略仍不可启动硬件。
- learner 的训练更新计划、actor 整场固定权重策略保持现状。本功能不自动延长 learner 的有限训练任务。

验证：

1. 12 项奖励/提交单元测试通过：成功提交、取消、粗标签拒收、网络失败重试、服务端拒收、排除段、奖励对齐等。
2. 录制状态机回归通过；策略模式验证回调在每段文件保存完成后调用。
3. 真实 Tk 窗口测试通过：确认保存并关闭，直接关闭不确认。测试使用合成数据和程序操作按钮，不替用户标注真实数据。
4. 真实 SSH 上传 + 云端协议接收通过：隔离的合成段上传后经过云端 load_episode 校验；模拟本地回执丢失再次提交返回 duplicate=true。
5. 测试未连接或驱动机械臂；未将合成段混入正式训练 inbox。硬件现场整段衔接仍待设备接入验证。

本地证据：validation/episode_delivery_2026-10-10/ 下的 unit_tests.log、record_flow.log、gui.log、cloud_link.log、cloud_server.log。云端隔离测试目录：/root/gpufree-data/offline_test/delivery_isolated_test。
