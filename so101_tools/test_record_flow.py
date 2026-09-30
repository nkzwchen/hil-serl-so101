#!/usr/bin/env python3
"""Test the record_demo_so101 state machine with a stub env and scripted keys.

Why this test exists
--------------------
The real recorder needs: cameras on, an interactive TTY, and a human doing the
insertion. None of that is available in a non-interactive check, so the state
machine (READY -> RECORD -> RETURN -> PRECISE), the key handling, the per-episode
file writing and the quality self-checks would otherwise be completely untested.

This replaces SO101Env with a stub and feeds a scripted key sequence, then asserts
on the files that got written. It NEVER touches the robot.

Run:  python test_record_flow.py
"""

import json
import os
import pickle
import shutil
import sys
import tempfile
import threading
import time

import gymnasium as gym
import numpy as np

# ─── 路径自解析（自动生成，勿手改：见 pack_so101_tools.py）───────────────
# 让脚本在两种布局下都能跑：
#   A) pi_project 布局： <P>/脚本.py              + <P>/hil-serl/examples/...
#   B) 仓库布局：       <repo>/so101_tools/脚本.py + <repo>/examples/...
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))


def _resolve_dirs():
    for _b in (_HERE, _os.path.dirname(_HERE)):
        if _os.path.isdir(_os.path.join(_b, 'examples', 'experiments')):
            return _b, _os.path.join(_b, 'examples')
        if _os.path.isdir(_os.path.join(_b, 'hil-serl', 'examples', 'experiments')):
            return _b, _os.path.join(_b, 'hil-serl', 'examples')
    _up = _os.path.dirname(_HERE)
    return _up, _os.path.join(_up, 'examples')


_ROOT, _EX = _resolve_dirs()
_EXPERIMENTS_DIR = _os.path.join(_EX, 'experiments')
_SO101_DIR = _os.path.join(_EXPERIMENTS_DIR, 'so101')
_sys.path.insert(0, _SO101_DIR)
_sys.path.insert(0, _EXPERIMENTS_DIR)
_sys.path.insert(0, _EX)
_sys.path.insert(0, _ROOT)
# ─────────────────────────────────────────────────────────────────────────

import record_demo_so101 as R  # noqa: E402
from so101_home_pose import N_ACTION, N_ARM, START_JOINT_POSE  # noqa: E402

IMG = 32          # 桩环境用小图，跑得快


# ----------------------------------------------------------------- stub env
class StubLogger:
    def info(self, *a, **k):
        pass

    def warn(self, *a, **k):
        pass


class StubNode:
    def get_logger(self):
        return StubLogger()

    def get_clock(self):
        class C:
            def now(self):
                class T:
                    def to_msg(self):
                        return None
                return T()
        return C()


class StubEnv(gym.Env):
    """Minimal stand-in for SO101Env. Only what the recorder touches.

    ★ 必须真的是 `gym.Env`：录制器现在把 env 包在 SERLObsWrapper +
      ChunkingWrapper 里（这样观测才会被拍平成训练要的形态），而
      gymnasium 的 Wrapper 需要内层 env 具备 `unwrapped` / `observation_space`。
    """

    def __init__(self, **kw):
        self.hz = 10.0
        self.action_scale = 0.05
        # 与真 SO101Env 同构：步数预算用**独立计数器**、且 reset() 清零。
        # （原来用 `n_steps % max_episode_steps` 的全局计数 + 6 步预算，
        #   桩会在任意位置自己结束，跟真 env 的行为不一样，测不出真问题。）
        self.max_episode_steps = int(kw.get('max_episode_steps', 100))
        self._step_count = 0
        self.image_size = IMG
        self.randomize_start = kw.get('randomize_start', True)
        self.randomize_half_m = kw.get('randomize_half_m', 0.025)
        self.joint_pos = np.array(START_JOINT_POSE, dtype=np.float32)
        self.leader_pos = np.array(START_JOINT_POSE, dtype=np.float32)
        self.action_space = None
        self.observation_space = _obs_space()
        self._images = {}
        self._tick = 0
        self._intervening = False
        self.node = StubNode()
        self.n_steps = 0
        self.n_resets = 0
        self._stop = threading.Event()
        self._get_obs()
        # ★ 模拟"活着的相机"：后台持续刷新 _images。
        #   没有它，前置检查里的"连续帧完全相同 ⇒ 相机卡死"会（正确地）报错。
        self._img_thread = threading.Thread(target=self._img_loop, daemon=True)
        self._img_thread.start()

    def _img_loop(self):
        while not self._stop.is_set():
            self._get_obs()
            time.sleep(0.02)

    # ---- recorder 用到的接口 ----
    def set_intervening(self, on):
        self._intervening = bool(on)

    def set_follower_torque(self, on):
        return True

    def set_intervene_deadband(self, rad):
        # 录制器进 RECORD 时会把死区归零（真 SO101Env 有这个接口）。
        # 桩环境必须也有，否则 AttributeError。
        self.intervene_deadband = float(rad)
        return self.intervene_deadband

    def _get_obs(self):
        self._tick += 1
        # 图像必须**每帧都变**，否则前置检查会判"相机卡死"。
        # ★ 返回**嵌套**形态、图像**不带**前导维 —— 与真 SO101Env 一致；
        #   拍平和加前导维是 SERLObsWrapper/ChunkingWrapper 的事。
        base = np.full((IMG, IMG, 3), self._tick % 251, dtype=np.uint8)
        self._images = {'cam_main': base.copy(), 'cam_wrist': (base + 7).copy()}
        return {'state': {'joint_pos': self.joint_pos.copy(),
                          'joint_vel': np.zeros(N_ARM + 1, dtype=np.float32)},
                'images': {k: v.copy() for k, v in self._images.items()}}

    def reset(self, seed=None, **kw):
        self.n_resets += 1
        self.joint_pos = np.array(START_JOINT_POSE, dtype=np.float32)
        self.joint_pos[1] += 0.02      # 假装随机起点
        self._step_count = 0           # 真 env 在 reset() 里也清零
        return self._get_obs(), {'succeed': False, 'start': {
            'randomized': True, 'dx_m': 0.011, 'dy_m': -0.008, 'dz_m': 0.004,
            'act_tilt_deg': 1.8, 'err_mm': 2.0, 'margin_rad': 1.2}}

    def step(self, action):
        self.n_steps += 1
        # 让主臂逐步靠近 "从臂 + 偏置"，好让 align_leader 收敛
        if not self._intervening:
            from so101_home_pose import leader_home_offset
            want = self.joint_pos[:N_ARM] + leader_home_offset(verbose=False)
            self.leader_pos[:N_ARM] += 0.5 * (want - self.leader_pos[:N_ARM])
        else:
            self.leader_pos[1] -= 0.01      # 人做插入
            self.joint_pos[1] -= 0.01
        self._step_count += 1
        done = self._step_count >= self.max_episode_steps
        info = {'succeed': False, 'intervening': self._intervening}
        if self._intervening:
            info['intervene_action'] = np.linspace(
                0.1, -0.2, N_ACTION).astype(np.float32)
        return self._get_obs(), 0.0, done, False, info

    def close(self):
        self._stop.set()
        try:
            self._img_thread.join(timeout=1.0)
        except Exception:
            pass


def _obs_space():
    import gymnasium as gym
    f32 = np.float32
    return gym.spaces.Dict({
        'state': gym.spaces.Dict({
            'joint_pos': gym.spaces.Box(-np.pi, np.pi, (6,), dtype=f32),
            'joint_vel': gym.spaces.Box(-np.inf, np.inf, (6,), dtype=f32)}),
        'images': gym.spaces.Dict({
            'cam_main': gym.spaces.Box(0, 255, (IMG, IMG, 3), dtype=np.uint8),
            'cam_wrist': gym.spaces.Box(0, 255, (IMG, IMG, 3), dtype=np.uint8)}),
    })


# ----------------------------------------------------------------- stub keys
class StubKeys:
    """Pops a scripted key sequence; None (no key) when the queue is empty.

    `gap`：每送出一个键之后，接下来这么多次 read_key() 都返回 None（"人没那么快
    按下一个键"）。没有它，脚本化的按键会在**每一次循环**被立刻取走 ——
    RECORD 阶段就会一步都没走完就收到 `d`，而录制器现在有"退段守卫"
    （< --min-steps 5 步自动丢弃），于是测试永远凑不够 2 段。
    """

    def __init__(self, seq, gap=6):
        self.seq = list(seq)
        self.ok = True
        self.consumed = []
        self.gap = int(gap)
        self._cool = 0
        self._push = []

    def read_key(self):
        if self._cool > 0:          # 刚按过一个键，冷却中
            self._cool -= 1
            return None
        if self._push:
            k = self._push.pop()
            self._cool = self.gap
            return k
        if self.seq:
            k = self.seq.pop(0)
            self.consumed.append(k)
            self._cool = self.gap
            return k
        return None

    def unread(self, ch):
        if ch:
            self._push.append(ch)

    def drain(self):
        # 真实实现会清掉缓冲；桩里"待按的键"就是 seq，不能被清掉，
        # 所以这里只清放回字符（与真实语义一致：进新阶段时旧放回作废）。
        self._push.clear()

    def reassert(self):
        pass

    def pressed(self):
        return False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ----------------------------------------------------------------- the test
def main():
    tmp = tempfile.mkdtemp(prefix='recflow_')
    # 2 段：每段  c(开始) -> [env 自己 done] -> c(回程结束) -> ... ；最后 q 退出
    # 注意 RECORD 里第 3 步会按 d 提前结束
    # 3 次尝试、只收 2 段：
    #   尝试1: c 开始 -> q 丢弃（操作失误）-> c 结束回程   ⇒ 不写文件、ep 不前进
    #   尝试2: c -> d 完成 -> c                          ⇒ 存成 ep_0000
    #   尝试3: c -> d 完成 -> c                          ⇒ 存成 ep_0001（达到 --episodes 2）
    script = ['c', 'q', 'c',   'c', 'd', 'c',   'c', 'd', 'c']
    keys = StubKeys(script)

    # ★ 录制器现在走 `TrainConfig().get_environment()`（这样才会套 SERLObsWrapper
    #   把嵌套观测拍平 —— 2026-09-29 首批 52 段就是因为没套而全部作废）。
    #   所以桩 env 要补到 **config 模块**里 SO101Env 这个名字上；只补源模块
    #   `experiments.so101.env.SO101Env` 没用（config 在 import 时就绑定了）。
    import experiments.so101.env as envmod
    import experiments.so101.config as cfgmod
    envmod.SO101Env = StubEnv
    cfgmod.SO101Env = StubEnv
    R.NonBlockingKey = lambda: keys
    import rclpy
    rclpy.init = lambda *a, **k: None
    rclpy.try_shutdown = lambda *a, **k: None

    argv = sys.argv
    sys.argv = ['record_demo_so101.py', '--out', tmp, '--episodes', '2', '--note', 'test']
    try:
        rc = R.main()
    finally:
        sys.argv = argv

    print(f'\n  main() 返回 {rc}')
    sess = [d for d in os.listdir(tmp) if os.path.isdir(os.path.join(tmp, d))]
    assert len(sess) == 1, f'应只有一个会话目录，实际 {sess}'
    sdir = os.path.join(tmp, sess[0])
    files = sorted(os.listdir(sdir))
    print(f'  会话目录 {sess[0]} 内容: {files}')

    eps = [f for f in files if f.startswith('ep_')]
    assert 'meta.json' in files, 'meta.json 没写出来'
    assert eps == ['ep_0000.pkl', 'ep_0001.pkl'], f'应正好两段且序号连续，实际 {eps}'

    with open(os.path.join(sdir, 'meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    print(f'  meta: image_size={meta["image_size"]} action_dim={meta["action_dim"]} '
          f'keys={meta["keys"]}')
    # ★ 和录制器共用同一张键位表（以前这里抄了一份字面量，改了键就假报错）
    assert meta['keys'] == R.KEYMAP, f'键位表与 R.KEYMAP 不一致: {meta["keys"]}'
    print(f'  meta.episodes = {len(meta["episodes"])} 段   '
          f'n_attempts={meta.get("n_attempts")}  n_discarded={meta.get("n_discarded")}')
    assert meta.get('n_discarded') == 1, '丢弃计数应为 1'
    assert meta.get('n_attempts') == 3, '尝试次数应为 3'
    for e in meta['episodes']:
        print(f'    {e["file"]}: {e["n_steps"]} 步, 接管 {e["n_intervened"]}, '
              f'{e["hz_actual"]}Hz, 起点倾角 {e["start_tilt_deg"]}, '
              f'回程 {e.get("return", {}).get("sec")}s')
        # ★ 退段守卫必须生效：1 步的空段不能落到磁盘上
        #   （2026-09-29 首批 52 段里有 14 段就是 1 步的空段）
        assert e['n_steps'] >= 5, \
            f'{e["file"]} 只有 {e["n_steps"]} 步 —— 退段守卫没生效（min-steps=5）'

    # 校验第 0 段的 transition 结构（必须与 HIL-SERL 格式一致）
    with open(os.path.join(sdir, eps[0]), 'rb') as f:
        d = pickle.load(f)
    trs = d['transitions']
    t = trs[0]
    print(f'\n  ep0: {len(trs)} 个 transition，keys={sorted(t.keys())}')
    for k in ('observations', 'actions', 'next_observations', 'rewards',
              'masks', 'dones', 'infos'):
        assert k in t, f'transition 缺 {k}'
    assert t['actions'].shape == (N_ACTION,), t['actions'].shape
    assert t['infos']['phase'] == 'record'
    assert trs[-1]['dones'] is True and trs[-1]['masks'] == 0.0
    assert all(x['infos']['intervening'] for x in trs), '记录段应该全程接管'
    print(f'  actions.shape={t["actions"].shape}  dones 末尾={trs[-1]["dones"]}  '
          f'masks 末尾={trs[-1]["masks"]}')
    print(f'  ep0.meta = {d["meta"]}')

    # 回程段不应写进 transitions（infos.phase 只有 'record'）
    assert all(x['infos']['phase'] == 'record' for x in trs), '回程不该被记录'
    print('\n  ✅ 状态机 / 按键 / 文件 / 自检 全部通过')
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
