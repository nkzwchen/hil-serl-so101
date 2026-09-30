"""SO101 实验配置（挂到 CONFIG_MAPPING["so101"]）。

对应 HIL-SERL 的 TrainConfig 契约（train_rlpd.py 会读这些字段）：
    get_environment / image_keys / proprio_keys / setup_mode / batch_size /
    discount / max_steps / training_starts / steps_per_update / ...

前置条件（训练前必须已经在跑）：
    · follower + leader 的桥
    · follower 的 ros2_control（arm_controller / joint_state_broadcaster / gripper_controller）
    · 相机（若要用图像观测）
  这些由 so101_rl_env.launch.py 负责（F1 方案：不含 LeaderTeleopComponent，
  因为 env 要成为 follower 的唯一控制源）。

⚠️ 与 HIL-SERL 原始示例的差异：
    · **动作空间只有 3 维**：`shoulder_lift / elbow_flex / wrist_flex`。
      `shoulder_pan / wrist_roll / gripper` 全部冻结、**不进动作向量**
      （定义在 so101_home_pose.py 的 FROZEN_JOINTS）。
      因此 setup_mode = single-arm-fixed-gripper（普通 SACAgent，3 维全连续）。
      ⚠️ 不能用 single-arm-learned-gripper —— 那是 hybrid agent，会把最后一维
         当夹爪离散头，而我们的最后一维是 wrist_flex。
    · 不用 RelativeFrame（关节增量没有坐标系，理由见 env.py 的模块注释）
    · proprio_keys 不含 tcp_force/tcp_torque（SO101 无 F/T 传感器）
"""

import gymnasium as gym
import numpy as np

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

from experiments.config import DefaultTrainingConfig
from experiments.so101.env import SO101Env
from experiments.so101.so101_home_pose import N_ACTION, N_ALL


class TrainConfig(DefaultTrainingConfig):
    # ---- 观测 ----
    image_keys = ['cam_main', 'cam_wrist']
    classifier_keys = ['cam_main']          # 阶段 B3 训 reward classifier 用
    proprio_keys = ['joint_pos', 'joint_vel']   # 无 tcp_force/tcp_torque

    # ---- 网络 ----
    encoder_type = 'resnet-pretrained'
    # ★ 动作空间是 **3 维**（只有 shoulder_lift / elbow_flex / wrist_flex，
    #   见 so101_home_pose.py）：pan / wrist_roll / gripper 全部冻结，
    #   夹爪**不在动作里**。
    #   所以必须用 'single-arm-fixed-gripper' → 普通 SACAgent，3 维全连续。
    #   ⚠️ 不能用 'single-arm-learned-gripper'：那是 SACAgentHybridSingleArm，
    #      它把**最后一维**当夹爪离散头（`action_dim = shape[-1] - 1`），
    #      于是 wrist_flex 会被误当成夹爪头，动作语义整个错位。
    setup_mode = 'single-arm-fixed-gripper'

    # ---- 训练规模（真机调试先从小的开始）----
    batch_size = 64                 # 真机 + 本机 CPU learner 时压小；云端 24G 可到 256
    cta_ratio = 2
    discount = 0.99                 # 比默认 0.97 高（稀疏奖励下视野更长）
    max_steps = 5000
    training_starts = 20
    steps_per_update = 50
    random_steps = 0

    replay_buffer_capacity = 50000
    log_period = 10
    eval_period = 1000

    # 落盘（真机训练建议开，便于中断后续训）
    buffer_period = 1000
    checkpoint_period = 1000

    # ---- env 参数 ----
    ENV_HZ = 10.0
    # ★ 观测图像边长（两个相机都用它，正方形）。
    #   为什么不是默认的 128：相机源分辨率是 cam_main 640x480、cam_wrist 424x240，
    #   缩到 128 会把细节全丢掉 —— 标定时**看不出试管有没有进孔**，而且
    #   **策略看到的是同一张图，它也没法学**。
    #   取舍（本地只做推理，训练在云端）：
    #     本地推理开销：约 2~3 ms/两相机（实测 CPU ~1 TFLOPS），**可忽略**
    #     本地磁盘/上传：224 时约 30MB/段（100 帧），128 时约 10MB/段
    #     云端训练：图像大 4 倍，GPU 上问题不大
    #   为什么选 224 而不是 256/320：
    #     · cam_wrist 只有 424x240，**纵向最多 240**，再大就是插值放大（假分辨率）
    #     · 224 是 32 的整数倍，正好是 ResNet 的下采样步长（特征图 7x7 干净）
    #     · 224 也是 ImageNet 预训练 ResNet 的标准输入尺度
    #   cam_main 其实能给更高；将来若要分相机设置，改这里 + env 的 image_sizes。
    IMAGE_SIZE = 224
    ACTION_SCALE = 0.05             # rad/步；10Hz 下 ≈ 28.6°/s
    MAX_EPISODE_STEPS = 100
    DISPLAY_IMAGE = False           # True 会弹窗显示两路相机

    # ---- 起始位置随机化 ----
    # ★ 只随机化 **shoulder_pan** 一个关节 ⇒ 末端沿一条**水平圆弧**滑动 ±1cm。
    #   其余关节一动不动，所以三件事是**精确**成立的：管子倾角不变、
    #   末端高度不变（dz < 35nm，来自 pan 轴 0.0002° 的装配误差）、
    #   不存在姿态分支问题（根本不做 IK）。
    #   （用户 2026-09-30："随机性小一点，只有第一个维度有比较小的随机性，
    #     相当于只水平移动"；见 动作空间-冻结维度.md §3.7）
    #
    # ⚠️ 别再回到"三维框 + IK"：pan 解冻后 IK 有 1 维零空间（pan 转 + 手臂反绕
    #    能到同一个末端），真机上表现为起点**有时非常高/收起来**。实测 2000 次：
    #    shoulder_lift 跨度 36.2°、elbow_flex 52.6°、wrist_flex 31.5°，
    #    而末端位置(±2.5cm)和倾角(0.05°)却都正常 —— 当时的验证**没查姿态**。
    #    离线复核：phase0_randomize_check.py（含"请求位移 == 实际位移"的 1:1 对账，
    #    这一条才能抓住半径算错之类的量值 bug）。
    RANDOMIZE_START = True
    RANDOMIZE_HALF_M = 0.01         # ±1cm 水平弧（与 so101_home_pose.RANDOMIZE_HALF_M 一致）

    # ---- reward 判据（B0.5 占位）----
    # 五个臂关节的目标位姿（rad）。None 表示不判成功（reward 恒 0，仅用于先跑通链路）。
    # 你要做具体任务时改这里，或改成接 reward classifier。
    TARGET_JOINT_POSE = None
    TARGET_TOL = 0.15

    def get_environment(self, fake_env=False, save_video=False, classifier=False,
                        enable_keyboard=True, randomize_start=None):
        """创建 env 并套上 HIL-SERL 需要的 wrapper。

        注意 fake_env 参数：HIL-SERL 的 learner 侧会传 fake_env=True，
        但我们的 learner 只需要 observation_space / action_space 的形状，
        不需要真连机器人。所以 fake_env=True 时返回一个只提供 spaces 的壳。

        `enable_keyboard=False` 给录制器用：它自己管接管状态，不能让 env 的
        空格键监听同时抢（否则 RETURN 阶段会被空格翻转模式）。
        `randomize_start=None` 表示用配置里的值。

        ★ **录制必须也用这个函数**，不要自己 new 一个裸 SO101Env ——
          裸 env 的观测是嵌套的，不是训练要的扁平形态（见
          so101_recording_io 顶部的两种布局说明）。
        """
        if fake_env:
            return _FakeSpacesOnlyEnv(
                image_keys=self.image_keys,
                proprio_keys=self.proprio_keys,
                img=self.IMAGE_SIZE,      # ★ 必须与真实 env 一致
            )

        env = SO101Env(
            hz=self.ENV_HZ,
            action_scale=self.ACTION_SCALE,
            max_episode_steps=self.MAX_EPISODE_STEPS,
            display_image=self.DISPLAY_IMAGE,
            image_size=self.IMAGE_SIZE,
            randomize_start=(self.RANDOMIZE_START if randomize_start is None
                             else randomize_start),
            randomize_half_m=self.RANDOMIZE_HALF_M,
            enable_keyboard=enable_keyboard,
        )
        # ★ SERLObsWrapper 契约：把 {"state": {...}, "images": {...}} 拍平成
        #   {"state": <flatten array>, "cam_main": ..., "cam_wrist": ...}
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        # 设置 reward 判据
        if self.TARGET_JOINT_POSE is not None:
            env.set_target_joint_pose(
                np.asarray(self.TARGET_JOINT_POSE, dtype=np.float32), self.TARGET_TOL
            )
        return env

    def process_demos(self, demo):
        return demo


class _FakeSpacesOnlyEnv(gym.Env):
    """给 learner 侧用的空壳：只提供 observation_space / action_space。

    为什么需要：train_rlpd.py 里 learner 会调 get_environment(fake_env=True)
    来拿 spaces 构造 ReplayBuffer 和 agent，但不该连真机。

    ⚠️ spaces 必须与真实 env（经 SERLObsWrapper + ChunkingWrapper）**完全一致**，
       否则 learner 侧构造的 ReplayBuffer 与 actor 送来的数据形状不匹配。

    ⚠️⚠️ **必须带 obs_horizon 前导维！**（2026-09-29 修）
       `ChunkingWrapper(obs_horizon=1)` 会对每个 obs 做 `v[None]`，所以真实 env
       的 `observation_space` 与 `reset()` 出来的 obs 都是 `(1, ...)` 形状。
       官方示例（usb_pickup_insertion）**没有"只给 spaces 的空壳"** ——
       它无论 fake_env 真假都套同一串 wrapper，所以天然一致；
       我们这个空壳是自己写的，早先漏掉了这一维：
           空壳声明            vs   真实 env 实际
           state   (12,)             (1, 12)
           cam_*   (128,128,3)       (1, 128, 128, 3)
       实测确认（`TrainConfig.get_environment(fake_env=True/False)` 对比）：
       三个 key 形状全不一致 → learner 侧 ReplayBuffer 一 insert 就炸。
    """

    def __init__(self, image_keys, proprio_keys, img=128,
                 n_joints=N_ALL, n_action=N_ACTION, obs_horizon=1):
        # 每个 proprio key 贡献 n_joints 维（joint_pos / joint_vel 各 6 维 → 12）
        state_dim = n_joints * len(proprio_keys)
        self.observation_space = gym.spaces.Dict({
            'state': gym.spaces.Box(-np.inf, np.inf,
                                    (obs_horizon, state_dim), dtype=np.float32),
            **{
                k: gym.spaces.Box(0, 255, (obs_horizon, img, img, 3), dtype=np.uint8)
                for k in image_keys
            },
        })
        # 动作：3 维，只含受控关节（与 SO101Env.action_space 一致）。
        # ★ 动作**不带** obs_horizon 维：ChunkingWrapper 在 act_exec_horizon=None
        #   时直接透传 env.action_space。
        self.action_space = gym.spaces.Box(-1.0, 1.0, (n_action,), dtype=np.float32)

    def reset(self, **kw):
        return self.observation_space.sample(), {'succeed': False}

    def step(self, a):
        return self.observation_space.sample(), 0, False, False, {'succeed': False}

    def close(self):
        pass
