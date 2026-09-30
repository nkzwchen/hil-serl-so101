#!/usr/bin/env python
"""生成 B0.2 用的假 demo 数据。

为什么需要它：
    train_rlpd.py 的 learner 路径有 `assert FLAGS.demo_path is not None`，
    且 RLPD 的离线数据（prior data）是硬需求——没有它就不是 RLPD 了。

做法：
    直接让 DummySO101Env 自己 rollout，保证 transition 的字段与形状
    和真实训练时 100% 一致（而不是手工构造，容易踩形状/字段的坑）。

用法：
    python gen_dummy_demos.py --out /path/to/dummy_demos.pkl --episodes 5
"""

import argparse
import os
import pickle as pkl
import sys

import numpy as np

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXAMPLES_DIR)

from experiments.dummy_test.config import TrainConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/nkzwchen/pi_project/checkpoints/dummy_test/demos.pkl")
    ap.add_argument("--episodes", type=int, default=5)
    args = ap.parse_args()

    cfg = TrainConfig()
    env = cfg.get_environment(fake_env=True)

    transitions = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, done, truncated, info = env.step(action)
            transitions.append(
                dict(
                    observations=obs,
                    actions=action,
                    next_observations=next_obs,
                    rewards=reward,
                    masks=float(1.0 - done),
                    dones=float(done),
                )
            )
            obs = next_obs
            if truncated:
                break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pkl.dump(transitions, f)

    print(f"✅ 生成 {len(transitions)} 条 transition -> {args.out}")
    t0 = transitions[0]
    print(f"   字段: {sorted(t0.keys())}")
    print(f"   observations 的 key: {sorted(t0['observations'].keys())}")
    for k, v in t0["observations"].items():
        print(f"     {k}: shape={np.asarray(v).shape} dtype={np.asarray(v).dtype}")
    print(f"   actions shape: {np.asarray(t0['actions']).shape}")


if __name__ == "__main__":
    main()
