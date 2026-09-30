"""B0.2 用的最小假环境配置（不依赖 ROS2 / franka / 硬件）

目的：在**不连机器人**的情况下验证 Actor + Learner 双进程能否握手、能否跑训练循环。
所以这里刻意把 env 做成纯 numpy 的玩具环境，把「算法进程间通信」这个变量单独隔离出来。

用法：
    # Learner（本机）
    python examples/train_rlpd.py --exp_name dummy_test --learner --ip 127.0.0.1
    # Actor（本机，另开一个终端）
    python examples/train_rlpd.py --exp_name dummy_test --actor --ip 127.0.0.1
"""

import gymnasium as gym
import numpy as np

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

from experiments.config import DefaultTrainingConfig

# 关节数（对齐 SO101：5 关节 + 夹爪）
N_JOINTS = 6
IMG_SHAPE = (128, 128, 3)


class DummySO101Env(gym.Env):
    """纯 numpy 玩具环境。

    观测空间结构必须符合 SERLObsWrapper 的契约：
        {"state": {<proprio dict>}, "images": {<image dict>}}
    SERLObsWrapper 会把 images 展开到顶层、把 state 拍平。
    """

    def __init__(self, max_steps: int = 50):
        self.max_steps = max_steps
        self._step = 0

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(N_JOINTS,), dtype=np.float32)

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "joint_pos": gym.spaces.Box(-3.14, 3.14, shape=(N_JOINTS,), dtype=np.float32),
                        "joint_vel": gym.spaces.Box(-10.0, 10.0, shape=(N_JOINTS,), dtype=np.float32),
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        "cam_main": gym.spaces.Box(0, 255, shape=IMG_SHAPE, dtype=np.uint8),
                        "cam_wrist": gym.spaces.Box(0, 255, shape=IMG_SHAPE, dtype=np.uint8),
                    }
                ),
            }
        )

    def _obs(self):
        return {
            "state": {
                "joint_pos": np.zeros(N_JOINTS, dtype=np.float32),
                "joint_vel": np.zeros(N_JOINTS, dtype=np.float32),
            },
            "images": {
                "cam_main": np.zeros(IMG_SHAPE, dtype=np.uint8),
                "cam_wrist": np.zeros(IMG_SHAPE, dtype=np.uint8),
            },
        }

    def reset(self, **kwargs):
        self._step = 0
        return self._obs(), {"succeed": False}

    def step(self, action):
        self._step += 1
        # 随机奖励：约 20% 的步给奖励，模拟稀疏成功信号
        reward = int(np.random.rand() < 0.2)
        # 每 10 步强制给一次奖励，保证训练循环能走到「成功」分支
        if self._step % 10 == 0:
            reward = 1
        done = self._step >= self.max_steps or reward == 1
        return self._obs(), reward, done, False, {"succeed": bool(reward)}


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["cam_main", "cam_wrist"]
    classifier_keys = ["cam_main"]
    proprio_keys = ["joint_pos", "joint_vel"]
    encoder_type = "resnet-pretrained"
    buffer_period = 0            # B0.2 不落盘，避免写文件
    checkpoint_period = 0
    steps_per_update = 10        # 比默认 50 小，便于快速看到权重广播
    max_steps = 200              # 小规模，只为验证闭环
    training_starts = 10
    batch_size = 32              # CPU 上跑，压小
    cta_ratio = 2
    discount = 0.99
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = DummySO101Env()
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        return env

    def process_demos(self, demo):
        return demo
