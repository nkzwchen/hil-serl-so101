#!/usr/bin/env python3
"""Generate a SYNTHETIC recording so the labeler can be tested before real capture.

Why
---
label_demos.py must be verifiable before any real recording exists: does the
magnifier frame the right pixels, does frame stepping work, are labels saved
correctly? The synthetic data draws a tube approaching a hole frame by frame, so
whether the magnifier helps is obvious at a glance.

Output format is IDENTICAL to a real recording (HIL-SERL transitions, obs with the
obs_horizon leading dim), so it also validates:
  * handling of the (1,12) / (1,128,128,3) leading dim
  * detection of both flat and per-episode input layouts

Usage
-----
    python make_dummy_recording.py                  # 3 episodes into testdata/
    python make_dummy_recording.py --episodes 5 --out /tmp/x.pkl
"""
import argparse
import os
import pickle
import time

import math

import cv2
import numpy as np

W = H = 128          # 可用 --size 覆盖（与 env.IMAGE_SIZE 对齐）
N_ARM = 5
STATE_DIM = 12          # joint_pos(6) + joint_vel(6)
ACTION_DIM = 4          # pan / shoulder_lift / elbow_flex / wrist_flex
N_JOINTS = 6

# 合成场景用的底图（之前相机调试留下的证据图，有就用，没有就画个背景）
BASE_MAIN = 'cam_probe/cam_main_640x480.png'
BASE_WRIST = 'cam_probe/cam_wrist_424x240.png'


def _base(path, fallback_color):
    if os.path.exists(path):
        im = cv2.imread(path)
        if im is not None:
            return cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
    return np.full((H, W, 3), fallback_color, dtype=np.uint8)


def _draw_tube(img, tip_xy, length=26, width=3):
    """Draw a "test tube" as a line whose lower end is the tip."""
    x, y = tip_xy
    p1 = (int(x), int(y))
    p2 = (int(x + length * 0.55), int(y - length))
    cv2.line(img, p1, p2, (230, 230, 235), width)          # 管身
    cv2.circle(img, p1, width, (250, 250, 255), -1)        # 管尖
    return img


def make_episode(rng, n_frames, hole_xy, ok=True, ep_idx=0):
    """Build one episode: the tube tip approaches the hole over frames.
    
        SYNTHETIC fixture, not a real recording: a static photo plus a drawn
        tube, with frame numbers, a watermark, a moving block and noise so that
        motion is obvious and nobody mistakes it for a broken viewer.
        ok=False leaves a gap above the hole (i.e. it never gets inserted).
    """
    main0 = _base(BASE_MAIN, 60)
    wrist0 = _base(BASE_WRIST, 90)
    trans = []
    obs = None
    for i in range(n_frames):
        s = i / max(n_frames - 1, 1)
        s = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5           # min-jerk
        # 管尖起点偏右上，终点 = 孔位（失败时留 12px 间隙）
        end = np.array(hole_xy, float)
        if not ok:
            end = end + np.array([0.0, -12.0])
        start = end + np.array([26.0, -30.0])
        tip = start + (end - start) * s
        tip = tip + rng.normal(0, 0.4, 2)                     # 一点抖动
        jitter_tip = tuple(np.round(tip).astype(int))

        # 亮度轻微闪烁 + 噪声，避免看起来像死图
        flick = 1.0 + 0.06 * math.sin(i * 0.55)
        shift = int(round((i / max(n_frames - 1, 1)) * 60 - 30))   # 背景横移 ±30px
        def _prep(base, dx):
            im = np.clip(base.astype(np.float32) * flick, 0, 255).astype(np.uint8)
            im = np.roll(im, dx, axis=1)                # 整幅横移，模拟相机/场景相对运动
            n = rng.normal(0, 2.2, im.shape).astype(np.float32)
            return np.clip(im.astype(np.float32) + n, 0, 255).astype(np.uint8)

        img_m = _prep(main0, shift)
        img_w = _prep(wrist0, -shift // 2)
        # 明显的动态标记：黄色方块随帧横扫
        yb = (int(20 + 90 * s), int(60))
        cv2.rectangle(img_m, (yb[0], yb[1]), (yb[0] + 9, yb[1] + 9), (40, 210, 240), -1)
        cv2.rectangle(img_w, (yb[0], yb[1]), (yb[0] + 9, yb[1] + 9), (40, 210, 240), -1)
        # 孔位红圈（腕部视角）
        cv2.circle(img_w, (int(hole_xy[0]), int(hole_xy[1])), 7, (60, 60, 230), 1)
        _draw_tube(img_m, (jitter_tip[0] + 40, jitter_tip[1] + 34))
        _draw_tube(img_w, jitter_tip)
        for im, tag in ((img_m, 'cam_main'), (img_w, 'cam_wrist')):
            cv2.putText(im, f'f{i:03d} SYNTHETIC', (3, 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(im, f'ep{ep_idx}', (3, 124),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)

        state = np.zeros(STATE_DIM, dtype=np.float32)
        state[:N_JOINTS] = np.array(
            [0.331, 0.284 - 0.10 * s, 0.087 + 0.14 * s, -0.343 + 0.05 * s,
             0.0575, 0.3249], dtype=np.float32)
        state[N_JOINTS:] = rng.normal(0, 0.02, N_JOINTS).astype(np.float32)
        action = np.clip(np.array([0.2 * (1 - s), 0.4 - 0.8 * s,
                                   -0.3 + 0.6 * s, 0.2 * (1 - s)], dtype=np.float32)
                         + rng.normal(0, 0.05, ACTION_DIM), -1, 1).astype(np.float32)

        # ★ 与真实 obs 一致：obs_horizon 前导维
        o = {
            'state': state[None, :],                       # (1,12)
            'cam_main': img_m[:, :, ::-1][None, ...].copy(),   # cv2 BGR → RGB
            'cam_wrist': img_w[:, :, ::-1][None, ...].copy(),
        }
        if obs is None:
            obs = o
        done = (i == n_frames - 1)
        trans.append(dict(
            observations=obs,
            actions=action,
            next_observations=o,
            rewards=0.0,
            masks=0.0 if done else 1.0,
            dones=done,
            infos={'intervening': True, 'succeed': False},
        ))
        obs = o
    return trans


def main():
    global W, H          # ★ 必须放在任何使用 W/H 之前（argparse 默认值也用到）

    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--frames', type=int, default=100)
    ap.add_argument('--out', default='testdata/dummy_recording.pkl')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--size', type=int, default=W,
                    help='image side length (match env.IMAGE_SIZE)')
    args = ap.parse_args()
    W = H = int(args.size)

    rng = np.random.default_rng(args.seed)
    hole = (int(W * 0.46), int(H * 0.44))
    episodes = []
    for k in range(args.episodes):
        # 每段孔位略不同（模拟不同起点），第 2 段故意做成"没插进去"
        h = (hole[0] + int(rng.integers(-10, 11)), hole[1] + int(rng.integers(-10, 11)))
        ok = (k != 1)
        episodes.append({'transitions': make_episode(rng, args.frames, h, ok=ok,
                                                   ep_idx=k)})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    data = {
        'meta': {
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'note': 'SYNTHETIC test data - NOT a real robot recording. '
                    'Static photo + drawn tube; frame numbers, watermark and a '
                    'moving block are added on purpose so motion is obvious.',
            'env_hz': 10.0, 'action_scale': 0.05,
            'action_space': f'Box(-1,1,({ACTION_DIM},))',
            'image_shape': [1, W, H, 3], 'state_shape': [1, STATE_DIM],
            'joint_names': ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
                            'wrist_flex', 'wrist_roll', 'gripper'],
        },
        'episodes': episodes,
    }
    with open(args.out, 'wb') as f:
        pickle.dump(data, f)
    sz = os.path.getsize(args.out) / 1e6
    print(f'wrote {args.out}')
    print(f'   {args.episodes} episodes x {args.frames} frames x 2 cams '
          f'{W}x{H}  ->  {sz:.1f} MB')
    print('   episode #2 is deliberately a MISS (to exercise the FAIL label)')
    print(f'   labels will be written to {os.path.splitext(args.out)[0]}.labels.json')


if __name__ == '__main__':
    main()
