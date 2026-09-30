#!/usr/bin/env python3
"""导出复盘视频，供人眼检查画面到底拍到了什么。

三种模式
--------
    （默认）         只看 cam_main
    --cam cam_wrist  只看腕部
    --both           **主视角 + 腕部并排**（左 main 右 wrist）

每种模式导两份
--------------
* `*_native.mp4` —— **原样像素**（单相机 224×224；--both 时 448×224 横拼）。
  这是策略真正吃到的图，判断"模型能不能看清试管/孔"必须看这份。
* `*_3x.mp4`     —— 3 倍 Lanczos 放大 + 帧号水印，纯给人看。
  水印/红框只是复盘标注，**不在**训练数据里。

残帧检测
--------
`PARTIAL FRAME`：相机缓冲区只填了一部分 —— 画面上面一大块**整行全 0**、
下面还有正常内容。实测首批录制里 0.60%（55/9184）的帧是这样，
cam_main 恰好 66.5% 的行全 0、cam_wrist 70.1%。
⚠️ 不能用「整帧均值 < 5」判它：这种帧整帧均值仍有 ~58，会漏掉。
碰到就标红框 + 写字，拖进度条一眼能看到。

用法
----
    python export_videos.py --both                        # 默认 5 段并排
    python export_videos.py --both --eps a.pkl b.pkl      # 指定段落
    python export_videos.py --cam cam_wrist               # 只看腕部
"""

import argparse
import glob
import os
import pickle
import sys

import cv2
import numpy as np

FPS = 10.0          # = ENV_HZ
UPSCALE = 3
BAD_ROW_FRAC = 0.10  # 整行全 0 的比例超过它 → 判为残帧


# ------------------------------------------------------------------ 读取/工具
def read_frames(path, cam):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    out = []
    for t in d['transitions']:
        o = t['observations']
        # 兼容嵌套（首批录制）与规范（修复后）两种布局
        img = o['images'][cam] if 'images' in o else o[cam]
        img = np.asarray(img)
        if img.ndim == 4:            # 规范布局带 obs_horizon 前导维
            img = img[0]
        out.append(np.ascontiguousarray(img))
    return out


def partial_rows(img):
    """整行全 0 的行占比。残帧的特征就是上面一大块整行全 0。"""
    im = np.asarray(img)
    return float((im.max(axis=(1, 2)) == 0).mean())


def put(img, text, org, color=(80, 255, 80), fs=0.42, th=1):
    """带黑描边的文字：白底/黑底上都看得清。"""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2,
                cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)


def mark_bad(img, frac):
    """残帧：红框 + 说明。

    ⚠️ cv2 的 rectangle/putText 颜色是 **BGR 顺序**，但我们的数组是 RGB，
    所以"红"要写 (255, 0, 0) —— 写 (0,0,255) 会画成蓝色（踩过）。
    """
    h, w = img.shape[:2]
    cv2.rectangle(img, (2, 2), (w - 3, h - 3), (255, 0, 0), 6)
    put(img, f'PARTIAL FRAME  {frac * 100:.0f}% rows blank',
        (8, h - 12), color=(255, 60, 60), fs=0.5, th=1)


def up(img, n=UPSCALE):
    return cv2.resize(img, (img.shape[1] * n, img.shape[0] * n),
                      interpolation=cv2.INTER_LANCZOS4)


def write(path, frames):
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'avc1'), FPS, (w, h))
    if not vw.isOpened():
        print(f'  ❌ 打不开编码器：{path}')
        return False
    for f in frames:
        vw.write(f)
    vw.release()
    return True


# ------------------------------------------------------------------ 一段视频
def build_episode(path, cams, scale):
    """返回 (帧列表, 统计)。cams 为 1 或 2 个相机名。

    并排时 **左 = cams[0]，右 = cams[1]**，两边各自标名字与残帧。
    """
    per = {c: read_frames(path, c) for c in cams}
    n = min(len(v) for v in per.values())
    bad = {c: [partial_rows(f) for f in per[c][:n]] for c in cams}
    frames, nbad, badpos, half = [], 0, [], None
    for i in range(n):
        tiles = []
        for c in cams:
            t = per[c][i]
            if scale != 1:
                t = up(t, scale)
            put(t, c.replace('cam_', ''), (8, 20), color=(120, 200, 255), fs=0.45)
            tiles.append(t)
        half = tiles[0].shape[1]
        im = np.concatenate(tiles, axis=1)
        put(im, f'f{i + 1}/{n}   t={i / FPS:.1f}s', (8, im.shape[0] - 12))
        for j, c in enumerate(cams):
            if bad[c][i] > BAD_ROW_FRAC:
                # 只在对应的那一半上标红
                mark_bad(im[:, j * half:(j + 1) * half], bad[c][i])
                nbad += 1
                if i not in badpos:
                    badpos.append(i)
        frames.append(im)
    return frames, {'n': n, 'nbad': nbad, 'badpos': badpos[:8]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps', nargs='*', default=None)
    ap.add_argument('--cam', default='cam_main')
    ap.add_argument('--both', action='store_true',
                    help='主视角 + 腕部并排（左 main 右 wrist）')
    ap.add_argument('--out', default='review_videos')
    args = ap.parse_args()

    cams = ['cam_main', 'cam_wrist'] if args.both else [args.cam]
    suffix = 'both' if args.both else args.cam

    if args.eps:
        picks = args.eps
    else:
        cand = [
            'recordings/2026-09-29_221134_v1/ep_0002.pkl',   # 68 步，干净
            'recordings/2026-09-29_232918_v2/ep_0009.pkl',   # 95 步（最长）
            'recordings/2026-09-29_232918_v2/ep_0016.pkl',   # main 6 个残帧
            'recordings/2026-09-29_234817_v3/ep_0000.pkl',   # 65 步，干净
            'recordings/2026-09-29_234817_v3/ep_0002.pkl',   # wrist 1 个残帧
        ]
        picks = [p for p in cand if os.path.exists(p)]
        if not picks:
            picks = sorted(glob.glob('recordings/*/ep_*.pkl'))[:4]

    os.makedirs(args.out, exist_ok=True)
    print('=' * 90)
    print(f'导出复盘视频  {" + ".join(cams)}  →  {args.out}/')
    print('=' * 90)
    tiles_all = []
    for p in picks:
        sess = os.path.basename(os.path.dirname(p))
        name = os.path.basename(p).replace('.pkl', '')
        tag = f'{sess[-2:]}_{name}'
        nat, st = build_episode(p, cams, 1)
        big, _ = build_episode(p, cams, UPSCALE)
        ok1 = write(os.path.join(args.out, f'{tag}_{suffix}_native.mp4'), nat)
        ok2 = write(os.path.join(args.out, f'{tag}_{suffix}_3x.mp4'), big)
        tiles_all += big + [np.zeros_like(big[0])] * 3
        d = (np.asarray([np.abs(nat[i + 1].astype(np.int16) - nat[i].astype(np.int16)).mean()
                         for i in range(st['n'] - 1)]) if st['n'] > 1
             else np.array([0.0]))
        print(f'  {"✅" if ok1 and ok2 else "❌"} {tag:<20} {st["n"]:>3} 步  '
              f'{st["n"] / FPS:>5.1f}s  {nat[0].shape[1]}x{nat[0].shape[0]}(原生)  '
              f'帧间差 {np.median(d):>5.2f}/255  '
              + (f'⚠️ 残帧 {st["nbad"]} 个 {st["badpos"]}' if st['nbad'] else '残帧 0'))
    if tiles_all:
        outp = os.path.join(args.out, f'ALL_{suffix}_3x.mp4')
        write(outp, tiles_all)
        print(f'\n  📼 合并复盘片：{outp}   {len(tiles_all) / FPS:.1f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
