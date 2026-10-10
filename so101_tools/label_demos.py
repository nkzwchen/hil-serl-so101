#!/usr/bin/env python3
"""Offline success-frame labeler for demos (tkinter GUI with a magnifier).

Why a separate offline tool
---------------------------
Recording makes NO success judgement -- every attempt is stored as-is. Judging is
deferred to an offline pass where you can replay, scrub and magnify. Benefits:
  * the recorder needs no success criterion (reward is still all-zero here)
  * labeling can be REDONE any number of times without re-recording
  * the conversion step is deterministic and reuses the exact `observations`
    produced by the env, so they match `observation_space` by construction

Why the magnifier is mandatory
------------------------------
The policy obs is only 128x128 (cam_main 640x480 and cam_wrist 424x240 are both
INTER_AREA-downscaled). Judging whether the tube entered the hole on 128x128 is
guesswork, so this tool magnifies whatever you point at, 1x..16x (nearest
neighbour on source pixels, bilinear optional).

Input formats (auto-detected)
-----------------------------
  0) {'transitions': [...], 'meta': {...}}          -- record_demo_so101.py
                                                      (one episode per file)
  1) {'episodes': [{'transitions': [...]}, ...]}   -- multi-episode container
  2) [[t, t, ...], [t, t, ...]]                    -- already split per episode
  3) [t, t, t, ...]                                -- official record_demos.py flat
                                                      list (split on `dones`)
Each transition is the HIL-SERL format:
    dict(observations=..., actions=..., next_observations=...,
         rewards=..., masks=..., dones=..., infos=...)

Output
------
  <recording>.labels.json
    {"recording": "...", "episodes": {"0": {"status": "success", "frame": 42}}}

Usage
-----
    DISPLAY=:0 python label_demos.py recordings/so101_xxx.pkl
    DISPLAY=:0 python label_demos.py *.pkl          # label several files in a row

NOTE: on this machine the X server is only reachable via the unix socket, so
      DISPLAY=:0 works while DISPLAY=127.0.0.1:0.0 does NOT. The conda OpenCV is
      built with GUI: NONE, so cv2.imshow is unusable -- hence tkinter.
"""
import argparse
import hashlib
import glob
import json
import math
import os
import sys
import time

import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

from so101_recording_io import load_raw   # 唯一的格式分发（与 build_demos.py 共用）

# ---------------- 外观常数 ----------------
SRC_PX = 300          # 两个原始视角画布边长
ZOOM_PX = 460         # 放大镜画布边长
ZOOM_MIN, ZOOM_MAX = 1, 16
ZOOM_DEFAULT = 6
PLAY_FPS = 10.0       # 与 ENV_HZ 一致

BG = '#1e1e1e'
FG = '#e8e8e8'
ACCENT = '#4ec9b0'
WARN = '#e0a030'
ERR = '#e05252'


# ============================ 数据加载 ============================
def _squeeze(x):
    """Drop the obs_horizon leading dim: (1,D)->(D), (1,H,W,C)->(H,W,C)."""
    if x is None:
        return None
    a = np.asarray(x)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    elif a.ndim == 2 and a.shape[0] == 1:
        a = a[0]
    return a


def _ep_from_transitions(trs):
    """Extract the fields the labeler needs from a transition list."""
    frames = {'actions': [], 'intervening': [], 'joint_pos': [], 'states': []}
    cams = {}
    for t in trs:
        obs = t.get('observations', {}) or {}
        info = t.get('infos', {}) or {}
        for k, v in obs.items():
            if k == 'state':
                continue
            if k not in cams:
                cams[k] = []
            a = _squeeze(v)
            cams[k].append(a)
        st = _squeeze(obs.get('state'))
        frames['states'].append(st)
        frames['actions'].append(np.asarray(t.get('actions', []), dtype=float).reshape(-1))
        frames['intervening'].append(bool(info.get('intervening', False)))
    if trs:
        final = trs[-1]['next_observations']
        for k in cams:
            cams[k].append(_squeeze(final[k]))
        frames['states'].append(_squeeze(final.get('state')))
        frames['actions'].append(np.zeros_like(frames['actions'][-1]))
        frames['intervening'].append(False)
    T = len(trs) + bool(trs)
    for k in cams:
        frames[k] = cams[k]
    frames['n'] = T
    return frames


def load_recording(path):
    """Load a recording; return (episodes_of_frames, meta).

    Format dispatch lives in `so101_recording_io.load_raw` — the single reader
    shared with `build_demos.py`. This function only adapts raw transitions to
    the frame dicts the labeler renders.
    """
    raw_eps, meta = load_raw(path)
    return [_ep_from_transitions(t) for t in raw_eps], meta

    raise ValueError(f'unrecognized recording format: {type(data)}')


# ============================ 放大镜取景（纯函数，便于离线验证）============================
def zoom_crop_size(zoom, pane_px=460, src_px=None):
    """Return the magnifier crop side length in SOURCE pixels.
    
        `zoom` = how many screen pixels one source pixel becomes.
        crop = pane_px / zoom, clamped to the source image size.
        e.g. pane=460, zoom=6 -> crop 77px shown at 6x (clamped to the image).
    """
    c = float(pane_px) / max(zoom, 1.0)
    if src_px is not None:
        c = min(c, float(src_px))
    return max(4.0, c)


def zoom_crop(img, center, zoom, pane_px, nearest=True):
    """Crop a window around `center` and magnify it to pane_px x pane_px.
    
        The crop is clamped inside the image so edges never show black.
        Returns a PIL.Image (not numpy); the caller hands it to ImageTk.
    """
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    crop = min(zoom_crop_size(zoom, pane_px, max(w, 1)), float(min(w, h)))
    crop = max(crop, 1.0)
    cx, cy = center
    x0 = int(round(min(max(cx - crop / 2, 0), max(w - crop, 0))))
    y0 = int(round(min(max(cy - crop / 2, 0), max(h - crop, 0))))
    x1 = min(int(round(x0 + crop)), w)
    y1 = min(int(round(y0 + crop)), h)
    sub = img[y0:y1, x0:x1]
    if sub.size == 0:
        return None
    resample = Image.NEAREST if nearest else Image.BILINEAR
    return Image.fromarray(sub).resize((pane_px, pane_px), resample)


# ============================ GUI ============================
class Labeler:
    def __init__(self, root, paths, review_once=False):
        self.root = root
        self.review_once = review_once
        self.review_confirmed = False
        self.paths = list(paths)
        self.fi = 0                     # 当前文件下标
        self.episodes = []
        self.meta = {}
        self.labels = {}                # {ep_idx(int): {'status':..., 'frame':...}}
        self.ep = 0                     # 当前段
        self.fr = 0                     # 当前帧
        self.playing = False
        self._play_job = None
        self.zoom = ZOOM_DEFAULT
        self.zoom_src = None            # None = 自动（优先腕部）
        self.zoom_center = [0.0, 0.0]    # 源像素坐标 (x, y)，加载后按实际边长居中
        self.nearest = True
        self._imgs = {}                 # 缓存 PhotoImage，防 GC
        self._roi_drag = False

        self._build_ui()
        self._load_current_file()
        self._bind()
        self._refresh()

    # ---------------- UI 搭建 ----------------
    def _build_ui(self):
        self.root.title('Demo Labeler  -  frame-by-frame + magnifier')
        self.root.configure(bg=BG)

        top = tk.Frame(self.root, bg=BG)
        top.pack(fill='x', padx=8, pady=(8, 4))
        self.lbl_file = tk.Label(top, text='', bg=BG, fg=ACCENT,
                                 font=('DejaVu Sans Mono', 11, 'bold'), anchor='w')
        self.lbl_file.pack(side='left')
        self.lbl_state = tk.Label(top, text='', bg=BG, fg=FG,
                                  font=('DejaVu Sans Mono', 11), anchor='e')
        self.lbl_state.pack(side='right')

        mid = tk.Frame(self.root, bg=BG)
        mid.pack(padx=8)
        self.canv = {}
        self.rect = {}
        for col, key in enumerate(('cam_main', 'cam_wrist', 'cam_main_roi')):
            box = tk.Frame(mid, bg=BG)
            box.grid(row=0, column=col, padx=4)
            tk.Label(box, text=key, bg=BG, fg=FG,
                     font=('DejaVu Sans Mono', 10)).pack()
            c = tk.Canvas(box, width=SRC_PX, height=SRC_PX, bg='#000',
                          highlightthickness=1, highlightbackground='#444')
            c.pack()
            self.canv[key] = c
            self.rect[key] = c.create_rectangle(0, 0, 0, 0, outline=ACCENT, width=2)

        zbox = tk.Frame(mid, bg=BG)
        zbox.grid(row=0, column=3, padx=(12, 4))
        self.lbl_zoom = tk.Label(zbox, text='MAGNIFIER', bg=BG, fg=ACCENT,
                                 font=('DejaVu Sans Mono', 10))
        self.lbl_zoom.pack()
        self.canv['zoom'] = tk.Canvas(zbox, width=ZOOM_PX, height=ZOOM_PX, bg='#000',
                                      highlightthickness=1, highlightbackground=ACCENT)
        self.canv['zoom'].pack()
        self._zoom_img_id = self.canv['zoom'].create_image(0, 0, anchor='nw')

        # 画面下：帧滑块
        bot = tk.Frame(self.root, bg=BG)
        bot.pack(fill='x', padx=8, pady=(6, 2))
        self.scale = tk.Scale(bot, from_=0, to=1, orient='horizontal',
                              bg=BG, fg=FG, troughcolor='#333',
                              highlightthickness=0, showvalue=False,
                              command=self._on_scale)
        self.scale.pack(fill='x')

        # ★ 一排按钮：不能假设按键一定能送到窗口（焦点/窗口管理器都会影响），
        #   所以每个操作都有一个可点的入口。
        bar = tk.Frame(self.root, bg=BG)
        bar.pack(fill='x', padx=8, pady=(4, 0))

        def _btn(text, cmd, width=8):
            b = tk.Button(bar, text=text, command=cmd, width=width,
                          bg='#3a3a3a', fg=FG, activebackground=ACCENT,
                          activeforeground='#000', relief='raised', bd=1,
                          font=('DejaVu Sans', 10))
            b.pack(side='left', padx=2)
            return b

        _btn('|< first', lambda: self._goto_frame(0), 7)
        _btn('<< -10', lambda: self._step(-10), 7)
        _btn('< -1', lambda: self._step(-1), 6)
        self.btn_play = _btn('play', self._toggle_play, 8)
        _btn('+1 >', lambda: self._step(+1), 6)
        _btn('+10 >>', lambda: self._step(+10), 7)
        _btn('last >|', lambda: self._goto_frame(10 ** 9), 7)
        tk.Label(bar, text='   ', bg=BG).pack(side='left')
        _btn('OK success (s)', self._mark_success, 11)
        _btn('FAIL (f)', self._mark_failed, 11)
        _btn('clear (c)', self._clear, 8)
        tk.Label(bar, text='   ', bg=BG).pack(side='left')
        _btn('< prev ep (p)', lambda: self._step_ep(-1), 11)
        _btn('next ep (n) >', lambda: self._step_ep(+1), 11)
        _btn('confirm + submit (w)' if self.review_once else 'save (w)', self._save, 19 if self.review_once else 10)

        self.lbl_info = tk.Label(self.root, text='', bg=BG, fg=FG, justify='left',
                                 anchor='w', font=('DejaVu Sans Mono', 10))
        self.lbl_info.pack(fill='x', padx=10)
        self.lbl_help = tk.Label(self.root, text=self._help_text(), bg=BG, fg='#999',
                                 justify='left', anchor='w',
                                 font=('DejaVu Sans Mono', 9))
        self.lbl_help.pack(fill='x', padx=10, pady=(2, 8))

    @staticmethod
    def _help_text():
        return ('Arrows: step +-1   Shift+Arrows: +-10   Home/End: first/last   Space: play/pause\n'
                's: mark SUCCESS at this frame   f: mark this episode FAILED   c: clear\n'
                'n/p: next/prev episode    z/Z: zoom out/in    r: recenter zoom\n'
                't: switch zoom source    g: nearest/bilinear    click/drag = move zoom\n'
                'w: save labels    q: quit (auto-saves if anything was marked)')

    def _bind(self):
        r = self.root
        r.bind('<Left>', lambda e: self._step(-1))
        r.bind('<Right>', lambda e: self._step(+1))
        r.bind('<Shift-Left>', lambda e: self._step(-10))
        r.bind('<Shift-Right>', lambda e: self._step(+10))
        r.bind('<Home>', lambda e: self._goto_frame(0))
        r.bind('<End>', lambda e: self._goto_frame(self._n_frames() - 1))
        r.bind('<space>', lambda e: self._toggle_play())
        for k, fn in (('s', self._mark_success), ('f', self._mark_failed),
                      ('c', self._clear), ('n', lambda: self._step_ep(+1)),
                      ('p', lambda: self._step_ep(-1)), ('z', lambda: self._zoom(-2)),
                      ('Z', lambda: self._zoom(+2)), ('t', self._toggle_src),
                      ('g', self._toggle_interp), ('r', self._recenter),
                      ('w', self._save),
                      ('q', self._quit)):
            r.bind(k, lambda e, fn=fn: fn())
        for key in ('cam_main', 'cam_wrist', 'cam_main_roi'):
            c = self.canv[key]
            c.bind('<Button-1>', lambda e, k=key: self._pick(k, e))
            c.bind('<B1-Motion>', lambda e, k=key: self._pick(k, e))
        self.root.protocol('WM_DELETE_WINDOW', self._quit)

    # ---------------- 文件 / 段 ----------------
    def _load_current_file(self):
        path = self.paths[self.fi]
        self.episodes, self.meta = load_recording(path)
        with open(path, 'rb') as source:
            self.recording_sha256 = hashlib.file_digest(source, 'sha256').hexdigest()
        self.labels = {}
        lp = self._labels_path(path)
        if os.path.exists(lp):
            try:
                with open(lp, encoding='utf-8') as f:
                    d = json.load(f)
                self.labels = {int(k): v for k, v in d.get('episodes', {}).items()}
                print(f'  [load] existing labels: {lp}  ({len(self.labels)} episodes)')
            except Exception as e:
                print(f'  [warn] existing labels unreadable ({e}); starting blank')
        self.ep, self.fr = 0, 0
        if not self.episodes:
            raise ValueError(f'{path} contains no episodes')
        # 放大镜默认来源：优先腕部（近景，看插孔最有用）
        self.zoom_src = 'cam_wrist' if 'cam_wrist' in self.episodes[0] else None
        self.scale.configure(to=max(self._n_frames() - 1, 1))
        # 放大镜按实际图像边长居中（分辨率可配，不能写死 64）
        c = self._src_dim() / 2.0
        self.zoom_center = [c, c]

    @staticmethod
    def _labels_path(p):
        return os.path.splitext(p)[0] + '.labels.json'

    def _cur(self):
        return self.episodes[self.ep]

    def _n_frames(self):
        return max(self._cur()['n'], 1)

    # ---------------- 渲染 ----------------
    def _src_key(self):
        if self.zoom_src and self.zoom_src in self._cur():
            return self.zoom_src
        for k in ('cam_wrist', 'cam_main', 'cam_main_roi'):
            if k in self._cur():
                return k
        return None

    def _src_dim(self):
        """Side length (px) of the current camera's images.

        NOT hardcoded to 128: `env.IMAGE_SIZE` is configurable (currently 224),
        and the magnifier / ROI math must follow whatever the data actually is.
        """
        key = self._src_key()
        v = self._cur().get(key) if key else None
        if v:
            a = _squeeze(v[0])
            if a is not None and a.ndim >= 2:
                return float(max(a.shape[0], a.shape[1]))
        return 128.0

    def _get_frame_img(self, key, i):
        v = self._cur().get(key)
        if not v or i >= len(v):
            return None
        a = _squeeze(v[i])
        if a is None or a.size == 0:
            return None
        return np.ascontiguousarray(a.astype(np.uint8))

    def _draw_src(self, key):
        c = self.canv[key]
        img = self._get_frame_img(key, self.fr)
        c.delete('pic')
        if img is None:
            self._imgs[key] = None
            return
        im = Image.fromarray(img).resize((SRC_PX, SRC_PX), Image.NEAREST)
        ph = ImageTk.PhotoImage(im)
        self._imgs[key] = ph            # ★ 必须留引用，否则被 GC 掉变白
        c.create_image(0, 0, anchor='nw', image=ph, tags='pic')
        c.tag_lower('pic')

    def _draw_roi(self):
        """Draw the magnifier ROI box on both source views."""
        for key in ('cam_main', 'cam_wrist', 'cam_main_roi'):
            if key not in self._cur():
                continue
            side = self._src_dim()
            half = (zoom_crop_size(self.zoom, ZOOM_PX, side) / side * SRC_PX) / 2.0
            sx = SRC_PX / side
            x, y = self.zoom_center[0] * sx, self.zoom_center[1] * sx
            h = half * sx
            self.canv[key].coords(self.rect[key], x - h, y - h, x + h, y + h)
            active = (key == self._src_key())
            self.canv[key].itemconfigure(
                self.rect[key], outline=ACCENT if active else '#666',
                width=2 if active else 1)

    def _draw_zoom(self):
        key = self._src_key()
        c = self.canv['zoom']
        img = self._get_frame_img(key, self.fr) if key else None
        if img is None:
            self._imgs['zoom'] = None
            c.delete('all')
            self._zoom_img_id = c.create_image(0, 0, anchor='nw')
            return
        im = zoom_crop(img, self.zoom_center, self.zoom, ZOOM_PX, self.nearest)
        if im is None:
            return
        ph = ImageTk.PhotoImage(im)
        self._imgs['zoom'] = ph
        c.itemconfigure(self._zoom_img_id, image=ph)
        # 十字线标出取景中心
        c.delete('cross')
        if self.zoom > 1.2:
            c.create_line(ZOOM_PX / 2, 0, ZOOM_PX / 2, ZOOM_PX,
                          fill=ACCENT, dash=(4, 6), tags='cross')
            c.create_line(0, ZOOM_PX / 2, ZOOM_PX, ZOOM_PX / 2,
                          fill=ACCENT, dash=(4, 6), tags='cross')
        crop = zoom_crop_size(self.zoom, ZOOM_PX, self._src_dim())
        self.lbl_zoom.config(
            text=f'zoom [{key}]  crop {crop:.0f}x{crop:.0f}px  '
                 f'screen {ZOOM_PX / crop:.1f}x  '
                 f'{"nearest" if self.nearest else "bilinear"}')

    def _refresh(self):
        for key in ('cam_main', 'cam_wrist', 'cam_main_roi'):
            if key in self._cur():
                self._draw_src(key)
        self._draw_roi()
        self._draw_zoom()
        self.scale.set(self.fr)

        ep = self._cur()
        lab = self.labels.get(self.ep, {})
        st = lab.get('status', 'unlabeled')
        if st == 'success':
            stxt, scol = f'SUCCESS @ frame {lab.get("frame")}', ACCENT
        elif st == 'failed':
            stxt, scol = 'FAILED', ERR
        else:
            stxt, scol = '- unlabeled', WARN
        if lab.get('provisional', False):
            stxt += ' [PROVISIONAL - review required]'
            scol = WARN
        n_ok = sum(1 for v in self.labels.values() if v.get('status') == 'success')
        n_bad = sum(1 for v in self.labels.values() if v.get('status') == 'failed')
        self.lbl_file.config(
            text=f'[{self.fi+1}/{len(self.paths)}] {os.path.basename(self.paths[self.fi])}'
                 f'   ep {self.ep+1}/{len(self.episodes)}'
                 f'   labeled: {n_ok} ok / {n_bad} failed')
        self.lbl_state.config(text=stxt, fg=scol)
        if getattr(self, 'btn_play', None) is not None:
            self.btn_play.config(text='pause' if self.playing else 'play')

        a = ep['actions'][self.fr] if self.fr < len(ep['actions']) else np.zeros(3)
        stt = ep['states'][self.fr] if self.fr < len(ep['states']) else None
        jp = ''
        if stt is not None and stt.size >= 6:
            jp = ' '.join(f'{math.degrees(v):+6.1f}°' for v in (stt[6:12] if stt.size == 18 else stt[8:14] if stt.size == 20 else stt[:6]))
        itv = ep['intervening'][self.fr] if self.fr < len(ep['intervening']) else False
        play = 'PLAYING' if self.playing else 'paused'
        note = self.meta.get('note') or ''
        synth = ('   [NOTE] SYNTHETIC test data: static photo + drawn tube, '
                 'only the drawn bar moves' if 'SYNTHETIC' in note else '')
        self.lbl_info.config(
            text=(f'frame {self.fr+1}/{ep["n"]}   {play}   intervening: '
                  f'{"yes" if itv else "no"}'
                  f'   action = [{a[0]:+.3f}, {a[1]:+.3f}, {a[2]:+.3f}]'
                  + (f'\njoint_pos(6): {jp}' if jp else '')
                  + synth))

    # ---------------- 交互 ----------------
    def _step(self, d):
        self.fr = int(min(max(self.fr + d, 0), self._n_frames() - 1))
        self._refresh()

    def _goto_frame(self, i):
        self.fr = int(min(max(i, 0), self._n_frames() - 1))
        self._refresh()

    def _on_scale(self, v):
        i = int(float(v))
        if i != self.fr:
            self.fr = i
            self._refresh()

    def _step_ep(self, d):
        """Move to the next/prev episode, CROSSING FILE BOUNDARIES.

        Why: a recording session is written as one pkl per episode
        (`record_demo_so101.py` does this so a crash cannot lose the whole
        session). Without crossing files you would have to relaunch the labeler
        for every single episode -- unusable for 30 of them. Labels for the file
        being left are saved first.
        """
        new = self.ep + d
        if 0 <= new < len(self.episodes):
            self.ep = new
            self.fr = 0
        else:
            nfi = self.fi + d
            if not (0 <= nfi < len(self.paths)):
                return
            if self.labels:
                self._save(quiet=True)
            self.fi = nfi
            self._load_current_file()
            self.ep = 0 if d > 0 else len(self.episodes) - 1
            self.fr = 0
        self.scale.configure(to=max(self._n_frames() - 1, 1))
        self._refresh()

    def set_play(self, on):
        """Single entry point for play/pause (used by both the button and the spacebar)."""
        if on == self.playing:
            return
        self.playing = on
        if on:
            self._tick()
        elif self._play_job:
            try:
                self.root.after_cancel(self._play_job)
            except Exception:
                pass
            self._play_job = None
        self._refresh()

    def _toggle_play(self):
        self.set_play(not self.playing)

    def _tick(self):
        if not self.playing:
            return
        if self.fr < self._n_frames() - 1:
            self.fr += 1
            self._refresh()
            self._play_job = self.root.after(int(1000 / PLAY_FPS), self._tick)
        else:
            self.playing = False

    def _mark_success(self):
        if self.fr < 1:
            print('请选择 frame >= 1：第 0 帧之前没有可赋奖励的动作。')
            return
        self.labels[self.ep] = {'status': 'success', 'frame': int(self.fr), 'provisional': False, 'reviewed': True, 'source': 'human_review'}
        self._refresh()
        # 无条件尝试前进 —— 一段一个文件时也要能自动跳到**下一个文件**的第一段，
        # 否则标完还得手按 n，30 段会疯。
        self._step_ep(+1)

    def _mark_failed(self):
        self.labels[self.ep] = {'status': 'failed', 'provisional': False, 'reviewed': True, 'source': 'human_review'}
        self._refresh()
        self._step_ep(+1)

    def _clear(self):
        self.labels.pop(self.ep, None)
        self._refresh()

    def _zoom(self, d):
        self.zoom = int(min(max(self.zoom + d, ZOOM_MIN), ZOOM_MAX))
        self._draw_roi()
        self._draw_zoom()

    def _toggle_src(self):
        keys = [k for k in ('cam_main', 'cam_wrist', 'cam_main_roi') if k in self._cur()]
        if len(keys) < 2:
            return
        cur = self._src_key()
        self.zoom_src = keys[1] if cur == keys[0] else keys[0]
        self._draw_roi()
        self._draw_zoom()

    def _recenter(self):
        """Move the magnifier back to the image center."""
        c = self._src_dim() / 2.0
        self.zoom_center = [c, c]
        self._draw_roi()
        self._draw_zoom()

    def _toggle_interp(self):
        self.nearest = not self.nearest
        self._draw_zoom()

    def _pick(self, key, ev):
        if key not in self._cur():
            return
        self.zoom_src = key
        side = self._src_dim()
        s = side / SRC_PX
        self.zoom_center = [min(max(ev.x * s, 0.0), side),
                            min(max(ev.y * s, 0.0), side)]
        self._draw_roi()
        self._draw_zoom()

    # ---------------- 保存 / 退出 ----------------
    def _save(self, quiet=False):
        path = self.paths[self.fi]
        if self.review_once and not quiet:
            lab = self.labels.get(0, {})
            if not lab.get('reviewed') or lab.get('provisional') or lab.get('status') not in ('success', 'failed'):
                from tkinter import messagebox
                messagebox.showinfo('待人工确认', '请先标记成功帧或失败，再按 w 确认提交。')
                return
        out = {
            'recording': os.path.abspath(path),
            'recording_sha256': self.recording_sha256,
            'labeled_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_episodes': len(self.episodes),
            'episodes': {str(k): v for k, v in sorted(self.labels.items())},
        }
        lp = self._labels_path(path)
        tmp = lp + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        os.replace(tmp, lp)
        if not quiet:
            print(f'  [save] {lp}')
            if self.review_once:
                self.review_confirmed = True
                self.root.destroy()
        return lp

    def _quit(self):
        if self.review_once:
            self.root.destroy()
            return
        if self.labels:
            try:
                self._save(quiet=True)
                print(f'  [save] auto-saved on exit: {self._labels_path(self.paths[self.fi])}')
            except Exception as e:
                print(f'  [warn] auto-save failed: {e}')
        else:
            print('  (nothing labeled; no labels file written)')
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('recordings', nargs='+', help='recording pkl (globs ok)')
    ap.add_argument('--review-once', action='store_true', help='单段人工复核，w 保存并关闭提交')
    args = ap.parse_args()

    paths = []
    for p in args.recordings:
        g = sorted(glob.glob(p))
        paths.extend(g if g else [p])
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print('ERROR: no recording files found')
        return 2

    if not os.environ.get('DISPLAY'):
        print("WARNING: DISPLAY is not set. On this machine use DISPLAY=:0 "
              "(unix socket; 127.0.0.1:0.0 does NOT work).")
    root = tk.Tk()
    try:
        if args.review_once and len(paths) != 1:
            raise ValueError('--review-once requires exactly one recording')
        app = Labeler(root, paths, review_once=args.review_once)
        # ★ 主动抢键盘焦点：否则窗口刚出来时方向键/空格可能被窗口管理器
        #   发给别的窗口，表现就是"按了没反应"。按钮栏是兜底，这个是直接修好。
        try:
            root.focus_force()
            root.lift()
        except Exception:
            pass
    except Exception as e:
        print(f'ERROR: startup failed: {type(e).__name__}: {e}')
        return 1
    root.mainloop()
    return 0 if not args.review_once or app.review_confirmed else 3


if __name__ == '__main__':
    sys.exit(main())
