"""SO101 实验配置（挂到 CONFIG_MAPPING["so101"]）。

对应 HIL-SERL 的 TrainConfig 契约（train_rlpd.py 会读这些字段）：
    get_environment / image_keys / proprio_keys / setup_mode / batch_size /
    discount / max_steps / training_starts / steps_per_update / ...

前置条件：只启动 so101_record_cameras.launch.py 接收图像。
  环境通过 LeRobot 直接独占两臂串口，不能同时运行 ROS 串口桥、
  ros2_control 或其他原生跟随进程。fake_env 不连接机器人。

与 HIL-SERL 原始示例的差异：
    · 动作为三个受控关节的相对角度，三个连续分量。
    · pan 不冻结；wrist_flex、wrist_roll 和夹爪在录制/跟随时冻结。
    · setup_mode=single-arm-fixed-gripper；动作最后一维不是夹爪。
    · 不使用 RelativeFrame；proprio 保留关节位置与速度。
"""

import gymnasium as gym
import numpy as np

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

from experiments.config import DefaultTrainingConfig
SO101Env = None  # Hardware dependencies are imported only by the real actor.
from experiments.so101.joint_control import ACTION_SEMANTICS, OBSERVATION_SEMANTICS, PROPRIO_DIMS, DEFAULT_STEP_RAD, CONTROL_HZ
from experiments.so101.pose_alignment import ALIGNMENT
from experiments.so101.camera_observation import load_roi, load_main_roi
from experiments.so101.so101_home_pose import N_ACTION, N_ALL, CONTROLLED_JOINTS


class TrainConfig(DefaultTrainingConfig):
    # ---- 观测 ----
    image_keys = ['cam_main', 'cam_main_roi']
    classifier_keys = []          # 奖励只来自段后人工标注
    proprio_keys = ['joint_pos', 'joint_vel', 'command_error', 'hold_offset']   # 无 tcp_force/tcp_torque

    # ---- 网络 ----
    encoder_type = 'resnet-pretrained'
    # Three relative joint components; no hybrid gripper head.
    setup_mode = 'single-arm-fixed-gripper'

    # ---- 训练规模（真机调试先从小的开始）----
    batch_size = 256                # 云端训练；本地仅运行 actor
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
    ENV_HZ = 10.0  # Training/recording cadence; native feedback polling remains CONTROL_HZ=30.
    CONTROL_BACKEND = 'lerobot_native_v1'
    POSE_ALIGNMENT = ALIGNMENT
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
    ACTION_SCALE = DEFAULT_STEP_RAD  # 5.5 degrees per joint/request; recordings retain their original scale.
    ACTION_JOINTS = list(CONTROLLED_JOINTS)
    ACTION_SEMANTICS = ACTION_SEMANTICS
    OBSERVATION_SEMANTICS = OBSERVATION_SEMANTICS
    MAX_EPISODE_STEPS = 100         # 10-second demonstration budget at 10 Hz.
    DISPLAY_IMAGE = False           # True 会弹窗显示两路相机

    # ---- 起始位置随机化：前三个关节独立 ±1.5°，腕部补偿倾角 ----
    RANDOMIZE_START = True
    RANDOMIZE_HALF_M = 0.01  # Legacy metadata; no longer drives sampling.
    RANDOMIZE_JOINT_HALF_DEG = 1.5

    # ---- reward 判据（B0.5 占位）----
    # 五个臂关节的目标位姿（rad）。None 表示不判成功（reward 恒 0，仅用于先跑通链路）。
    # 训练奖励由段后人工标注生成，不在环境内判断。
    REWARD_SOURCE = 'human_episode_annotation'
    TARGET_JOINT_POSE = None
    TARGET_TOL = 0.15

    def __init__(self, full_frame_size=None, use_cameras=True):
        self.use_cameras = use_cameras
        self.full_frame_size = full_frame_size
        if not use_cameras:
            self.full_frame_size = None
            self.roi_spec = self.main_roi_spec = None
            self.image_keys = []
            return
        if full_frame_size:
            self.roi_spec = self.main_roi_spec = None
            self.image_keys = ["cam_main"]
            return
        self.roi_spec = load_roi()
        self.main_roi_spec = load_main_roi()
        if not self.roi_spec or not self.main_roi_spec:
            raise ValueError('Both cam_main ROI specifications are required')
        self.image_keys = ['cam_main', 'cam_main_roi']

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

        global SO101Env
        if SO101Env is None:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from experiments.so101.native_env import NativeSO101Env
            SO101Env = NativeSO101Env
        env = SO101Env(
            use_cameras=self.use_cameras,
            full_frame_size=self.full_frame_size,
            roi_spec=self.roi_spec,
            main_roi_spec=self.main_roi_spec,
            hz=self.ENV_HZ,
            action_scale=self.ACTION_SCALE,
            max_episode_steps=self.MAX_EPISODE_STEPS,
            display_image=self.DISPLAY_IMAGE,
            image_size=self.IMAGE_SIZE,
            randomize_start=(self.RANDOMIZE_START if randomize_start is None
                             else randomize_start),
            randomize_half_m=self.RANDOMIZE_HALF_M,
            randomize_joint_half_deg=self.RANDOMIZE_JOINT_HALF_DEG,
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
        # 按各 key 的真实维度求和：6+6+3+3=18。
        state_dim = sum(PROPRIO_DIMS[key] for key in proprio_keys)
        self.observation_space = gym.spaces.Dict({
            'state': gym.spaces.Box(-np.inf, np.inf,
                                    (obs_horizon, state_dim), dtype=np.float32),
            **{
                k: gym.spaces.Box(0, 255, (obs_horizon, img, img, 3), dtype=np.uint8)
                for k in image_keys
            },
        })
        # 动作：4 维相对关节角，与 SO101Env.action_space 一致。
        # ★ 动作**不带** obs_horizon 维：ChunkingWrapper 在 act_exec_horizon=None
        #   时直接透传 env.action_space。
        self.action_space = gym.spaces.Box(-1.0, 1.0, (n_action,), dtype=np.float32)

    def reset(self, **kw):
        return self.observation_space.sample(), {'succeed': False}

    def step(self, a):
        return self.observation_space.sample(), 0, False, False, {'succeed': False}

    def close(self):
        pass
