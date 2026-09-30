#!/usr/bin/env python3
"""Record demo episodes for the SO101 tube-insertion task -- NO success judgement.

Design in one paragraph
-----------------------
The env (`SO101Env`) is the follower's ONLY control source, so the recorder drives
the loop through `env.step()` and the human steers via leader intervention
(mode B). Recording makes NO success judgement: every attempt is stored verbatim
and labelling happens offline in `label_demos.py`. That keeps this script simple
and means a wrong success criterion can never throw good data away.

Per-episode state machine (4 phases)
------------------------------------
    READY    mode A. Leader torque ON and aligned to `follower + LEADER_HOME_OFFSET`.
             The follower has already been reset to a RANDOM start pose.
             Human: hands off the leader.            [c] start recording
    RECORD   mode B. Leader torque OFF, human drives the insertion.
             The differential anchor is captured on entry, so there is no jump.
             Human: do the insertion.                [d] done / env `done`
    RETURN   STILL mode B -- the leader keeps controlling the follower.
             The human brings the follower back near the start pose, which also
             pulls the tube out and rewinds the leader toward its own home.
             Transitions are NOT recorded (metadata only).
             Human: bring it back.                   [c] finished returning
    PRECISE  mode A. `env.reset()` puts the follower exactly on a new random start
             (gravity compensation + measured feedback + smooth min-jerk ramp),
             then the leader is re-aligned and verified.
                                                    -> READY

Why the RETURN phase exists
---------------------------
Two problems, one motion:
  * the follower ends the episode inserted in the hole; a single script ramp from
    there back to a random start would be a large, abrupt move
  * with a DIFFERENTIAL mapping the leader does NOT re-home itself, so its
    absolute pose drifts by one insertion per episode. Simulation: with a
    [-18, -26, -9] deg insertion the leader hits its elbow limit on episode 4.
The human's rewind fixes both at once: the follower comes back near the start AND
the leader physically returns near its home.

Keys
----
    c   advance to the next phase (start recording / finished returning)
    d   "I finished inserting" -- end this episode's recording and SAVE it
    q   "I made a mistake" -- DISCARD this attempt (nothing is written, the
        episode index does not advance, so the next attempt reuses the number)
        in READY it means "skip this start pose, re-randomize"
    x   quit the session (already-saved episodes stay on disk; writes meta.json)
    h   print the keys allowed in the current phase

A discarded attempt is still followed by the RETURN phase, so you can bring the
arm back safely before the next attempt. Nothing is written to disk for it.

NOTE `d`, not the spacebar: the env's own spacebar listener is DISABLED here
(`enable_keyboard=False`) because a space press would also flip the mode, while
the RETURN phase must stay in mode B.

WARNING (accepted risk, by the user's decision)
-----------------------------------------------
In mode B the leader has NO torque. If you let go of it, it falls, and because the
mapping is differential the FOLLOWER FOLLOWS THE FALL (measured: the leader drops
~115 deg when released). So ALWAYS keep one hand on the leader while pressing a
key with the other hand. There is deliberately no automatic "leader was dropped"
guard in this script.

Usage
-----
    python record_demo_so101.py --out recordings --episodes 30 --note "batch1"
    python record_demo_so101.py --dry-run      # preflight + plan only, no motion
"""

import argparse
import math
import os
import pickle
import shutil
import sys
import time

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

from so101_keys import NonBlockingKey  # noqa: E402
from so101_home_pose import (  # noqa: E402
    ARM_JOINTS, CONTROLLED_IDX, FROZEN_IDX, HOME_CMD_FILE, LEADER_LIFT_DEFAULT_CM,
    N_ACTION, N_ARM, leader_home_offset, load_home_cmd, tube_tilt_deg,
)

PHASES = ('READY', 'RECORD', 'RETURN', 'PRECISE')
# ★ 键位表是**单一来源**：写进 meta.json 的也是它，测试脚本 import 的也是它。
#   （踩坑 2026-09-29：这张表在录制器里是字面量、测试里又抄了一份，
#     改了键之后录制器正常、测试却拿旧键位断言，白白报一个 AssertionError。）
KEYMAP = {
    'start_or_next': 'c',
    'save_this_episode': 'd',
    'discard_this_attempt': 'q',
    'skip_start': 'q (in READY)',
    'quit_session': 'x',
    'help': 'h',
}
# ★ env.reset() 返回的 info['start'] 里，录制器**必须**读到的键。
#   写错键名不会抛异常（全是 .get()），只会把元数据静默写成 0 —— 所以录制器
#   在第一次 reset 后会硬查这张表，test_env_reset.py 也会拿真机数据核对。
START_INFO_KEYS = ('dx_m', 'dy_m', 'dz_m', 'act_tilt_deg', 'randomized')
KEYS_BY_PHASE = {
    'READY': 'c = start recording   q = skip this start (re-randomize)   '
             'x = quit   h = help',
    'RECORD': 'd = I finished inserting (SAVE it)   '
              'q = MISTAKE - discard this attempt   x = quit   h = help',
    'RETURN': 'c = finished returning   q = mark this attempt as DISCARDED   '
              'x = quit   h = help',
    'PRECISE': '(automatic -- the script is moving the follower and the leader)',
}
# 每段的磁盘预算：2 相机 × 观测边长² × 3 字节 × 100 帧，再留 3 倍余量
BYTES_PER_STEP_AT_224 = 2 * 224 * 224 * 3


# ===================================================================== preflight
def preflight(args, env, kb):
    """All-or-nothing checks. Returns a list of problems (empty == OK)."""
    bad, warn = [], []

    # 1) gravity-compensation archive -- without it the start pose is 8.75 deg off
    if load_home_cmd() is None:
        bad.append(
            f'找不到归位指令存档 {HOME_CMD_FILE}\n'
            f'      → 先跑一次 `python phase1_home.py --no-leader`。\n'
            f'        没有它，从臂会停在名义起点下方约 3.6cm、管子歪到 8.75 度。')

    # 2) a TTY is mandatory: without it there are no phase keys
    if not kb.ok:
        bad.append('stdin 不是 TTY → 读不到按键，无法录制。\n'
                   '      → 在**交互式终端**里直接运行本脚本（不要用管道/重定向）。')

    # 3) joint states fresh
    t0 = time.time()
    while (env.joint_pos is None or env.leader_pos is None) and time.time() - t0 < 8.0:
        time.sleep(0.05)
    if env.joint_pos is None:
        bad.append('拿不到 /follower/joint_states → launch 在跑吗？')
    if env.leader_pos is None:
        bad.append('拿不到 /leader/joint_states → leader 桥在跑吗？')

    # 4) cameras: present, not all-zero, and NOT FROZEN
    # ⚠️ 必须走 `env.unwrapped`：gymnasium 的 Wrapper.__getattr__ **禁止**转发
    #    下划线开头的属性，所以包了 wrapper 之后 `env._images` 会直接 AttributeError。
    raw = env.unwrapped
    if env.joint_pos is not None:
        cams = list(getattr(raw, '_images', {}) or {})
        if not cams:
            bad.append('没有收到任何相机图像 → 是不是用了 use_cameras:=false？')
        else:
            shots = []
            for _ in range(6):
                time.sleep(0.15)
                shots.append({k: np.asarray(v).copy()
                              for k, v in (getattr(raw, '_images', {}) or {}).items()})
            for cam in cams:
                frames = [s[cam] for s in shots if cam in s]
                if not frames:
                    bad.append(f'{cam}: 采样期间没有新帧')
                    continue
                if all(f.size == 0 or not f.any() for f in frames):
                    bad.append(f'{cam}: 全是黑图（全 0）')
                    continue
                if all(np.array_equal(frames[0], f) for f in frames[1:]):
                    bad.append(f'{cam}: 连续 {len(frames)} 帧完全相同 → 相机卡死 '
                               f'（这是踩过的真实故障）')
            print(f'  相机            : {cams}  '
                  f'分辨率 {tuple(shots[0][cams[0]].shape)}')

    # 5) torque
    if not env.set_follower_torque(True):
        bad.append('/follower/set_torque ON 失败 → 从臂使不上力')

    # 6) disk
    need = args.episodes * 100 * BYTES_PER_STEP_AT_224 * 3
    p = os.path.abspath(args.out or '.')          # 目录可能还不存在，往上找到有的
    while p and not os.path.isdir(p):
        p = os.path.dirname(p)
    free = shutil.disk_usage(p or '.').free
    print(f'  磁盘剩余        : {free/1e9:.1f} GB   计划 {args.episodes} 段 '
          f'预计需要 {need/1e9:.1f} GB')
    if free < need:
        bad.append(f'磁盘可能不够：剩 {free/1e9:.1f}GB < 预计 {need/1e9:.1f}GB')
    return bad, warn


# ===================================================================== helpers
def poll(kb, accept):
    """Read one key, but only CONSUME it if it is in `accept`.

    Anything else is put back via `kb.unread()` so a key pressed slightly too
    early (e.g. `c` while the automatic leader alignment is still running) is not
    silently swallowed. Callers must not busy-spin: every loop here either sleeps
    or calls `env.step()` (which sleeps), so re-reading an unaccepted key is fine.
    """
    k = kb.read_key()
    if k is None or k in accept:
        return k
    kb.unread(k)
    return None



def cur_obs(env):
    """当前的**规范**观测 —— 与 `env.step()` 的返回**同源**。

    ⚠️ 不要用 `env._get_obs()`：那是裸 env 的方法，返回**嵌套**观测
    （`{'state': {...}, 'images': {...}}`），训练侧要的是扁平形态。
    2026-09-29 首批录制就是因为这个，52 段全部作废。

    `ChunkingWrapper` 把最近 `obs_horizon` 帧规范观测存在 `current_obs` 里，
    `stack_obs(current_obs)` 就是 `step()` 返回的那个东西，所以两者必然自洽
    （实测：`next_obs[k] == obs[k+1]`，state 与图像逐帧全等）。
    """
    cw = env
    for _ in range(4):                      # 剥 wrapper 找 current_obs
        if hasattr(cw, 'current_obs'):
            break
        cw = getattr(cw, 'env', None)
        if cw is None:
            return None
    try:
        from serl_launcher.wrappers.chunking import stack_obs
        return stack_obs(cw.current_obs)
    except Exception:
        return None


def check_obs_layout(env, obs, tag=''):
    """把观测逐 key 对齐 `env.observation_space`。不一致就抛错。

    这是**本该在第一次录制时就拦住**那个 bug 的检查：录制器用裸 env 时
    `observation_space` 是嵌套 Dict、而记下来的观测也是嵌套的，两边"自洽"，
    所以谁都不报错；错的是**两者都不是训练要的形态**。因此这里不能拿
    `env.observation_space` 当参照物，还要确认它本身就是**扁平**的。
    """
    sp = env.observation_space
    if not hasattr(sp, 'spaces'):
        raise RuntimeError(f'observation_space 不是 Dict：{type(sp)}')
    keys = set(sp.spaces)
    if 'images' in keys or 'state' not in keys or isinstance(sp.spaces['state'], dict):
        raise RuntimeError(
            f'{tag}观测空间是**嵌套**的（{sorted(keys)}），不是训练要的扁平形态。\n'
            f'     录制必须用 TrainConfig().get_environment()（会套 SERLObsWrapper）。')
    if not isinstance(obs, dict) or 'images' in obs:
        raise RuntimeError(
            f'{tag}观测是**嵌套**的（{sorted(obs) if isinstance(obs, dict) else type(obs)}），'
            f'不是训练要的扁平形态。\n'
            f'     录制器请用 cur_obs(env)，不要用 env._get_obs()。')
    bad = []
    for k, s in sp.spaces.items():
        if k not in obs:
            bad.append(f'缺 {k!r}')
            continue
        a = np.asarray(obs[k])
        if tuple(a.shape) != tuple(s.shape):
            bad.append(f'{k}: shape {tuple(a.shape)} != {tuple(s.shape)}')
        elif a.dtype != s.dtype:
            bad.append(f'{k}: dtype {a.dtype} != {s.dtype}')
    if bad:
        raise RuntimeError(f'{tag}观测与 observation_space 不符：' + '; '.join(bad))
    return True



def align_leader(env, kb, tol_deg=2.0, timeout=12.0, label='align'):
    """Drive the leader to `follower + LEADER_HOME_OFFSET` by stepping in mode A.

    In mode A `env.step()` calls `_drive_leader(cur)` every step, and the leader
    bridge rate-limits to 20 deg/step, so this is a bounded, smooth move.

    ★ 只消费 `x`（退出会话），**其它键一律放回**。
      踩坑（2026-09-29）：最初这里把 `q` 也消费掉（本意是"放弃对齐"），
      结果用户想在 READY 里按 `q`（跳过这个起点）时，按键在自动对齐阶段
      **被悄悄吃掉** —— 按键语义和用户预期对不上，而且毫无提示。
      对齐最多 12s，不能提前打断也不是问题。

    Returns (converged, residual_deg, quit_now).
    """
    off = leader_home_offset(verbose=False)
    t0, resid = time.time(), float('inf')
    while time.time() - t0 < timeout:
        if poll(kb, 'xX') in ('x', 'X'):
            return False, resid, True
        env.step(np.zeros(N_ACTION, dtype=np.float32))       # 策略动作为 0 → 从臂保持
        if env.leader_pos is None or env.joint_pos is None:
            continue
        want = np.asarray(env.joint_pos[:N_ARM]) + off
        resid = float(np.abs(np.asarray(env.leader_pos[:N_ARM]) - want).max())
        resid = math.degrees(resid)
        if resid < tol_deg:
            return True, resid, False
    return False, resid, False


def phase_tilt(env):
    """Current tube tilt (deg from vertical) from the measured follower pose."""
    if env.joint_pos is None:
        return float('nan')
    return tube_tilt_deg(np.asarray(env.joint_pos)[:N_ARM])


def print_banner(env, args):
    print('=' * 78)
    print('SO101 demo 录制（不做成功判断 —— 判断留给离线标定）')
    print('=' * 78)
    # ★ 这里打印的是**训练侧的规范空间**（扁平的 state(1,12) + cam_*(1,S,S,3)）。
    #   录制器必须用包装后的 env，所以看到的就该是这个形态；若还看到嵌套的
    #   state{joint_pos,...}/images{...}，说明又退回裸 SO101Env 了 ——
    #   check_obs_layout() 会直接抛错（2026-09-29 首批 52 段就是这样作废的）。
    sp = env.observation_space
    if hasattr(sp['state'], 'spaces'):          # 嵌套（不该出现）
        st = sp['state']
        st_shapes = '嵌套! ' + ', '.join(f'{k}{tuple(v.shape)}'
                                        for k, v in st.spaces.items())
    else:
        st_shapes = f'{tuple(sp["state"].shape)} {sp["state"].dtype}'
    img_txt = ', '.join(f'{k}{tuple(v.shape)}'
                        for k, v in sp.spaces.items() if k != 'state')
    print(f'  观测            : state[{st_shapes}]   {img_txt}')
    ctrl = ' / '.join(ARM_JOINTS[i] for i in CONTROLLED_IDX)
    print(f'  动作            : {N_ACTION} 维（{ctrl}）  scale={env.action_scale} rad/步')
    print(f'  控制频率        : {env.hz:.0f} Hz   每段上限 {env.max_episode_steps} 步 '
          f'= {env.max_episode_steps / env.hz:.0f}s')
    if env.randomize_start:
        print(f'  起始位置        : 随机化 ±{env.randomize_half_m*100:.1f}cm'
              f'（只转 shoulder_pan 的水平弧）')
    else:
        print('  起始位置        : 固定（--no-randomize）')
    print(f'  主臂抬高偏置    : {LEADER_LIFT_DEFAULT_CM:g} cm')
    print(f'  随机种子        : {args.seed}')
    print('=' * 78)


# ===================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='recordings', help='session parent directory')
    ap.add_argument('--episodes', type=int, default=30, help='how many episodes to record')
    ap.add_argument('--note', default='', help='short tag in the session directory name')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--min-steps', type=int, default=5, dest='min_steps',
                    help='短于这么多步的段自动判为丢弃（默认 5）—— 一次真实插入不可能'
                         '只有一两步，防的是 `d` 在 RECORD 刚开始就被按下')
    ap.add_argument('--no-randomize', action='store_true',
                    help='debug: fixed start pose instead of randomizing')
    ap.add_argument('--dry-run', action='store_true',
                    help='preflight + print the plan, then exit (no motion)')
    args = ap.parse_args()

    import rclpy
    from experiments.so101.env import SO101Env
    from experiments.so101.config import TrainConfig

    cfg = TrainConfig()
    rclpy.init()
    # ★★ 必须走 TrainConfig().get_environment()，拿**包装后**的 env ★★
    #   裸 SO101Env 的观测是嵌套的 {'state': {...}, 'images': {...}}，而训练侧要的是
    #   SERLObsWrapper + ChunkingWrapper 之后的扁平形态 {'state': (1,12), 'cam_*': (1,S,S,3)}。
    #   2026-09-29 的首批 3 个会话（52 段）就是这样录的，全部被 build_demos 拦下：
    #       observations 缺 key 'cam_main'; observations['state'] shape () != (1, 12)
    #   learner 侧会直接 KeyError。所以这里复用训练用的同一套 wrapper，
    #   从根上不可能再分叉。（gym.Wrapper 会把 joint_pos/set_intervening 等属性转发给内层。）
    env = cfg.get_environment(
        enable_keyboard=False,        # ★ 接管状态由本脚本掌管（见文件头 Keys 一节）
        randomize_start=(cfg.RANDOMIZE_START and not args.no_randomize),
    )
    try:
        with NonBlockingKey() as kb:
            print_banner(env, args)
            print('\n前置检查 ...')
            bad, _ = preflight(args, env, kb)
            if bad:
                print('\n❌ 前置检查未通过，拒绝开始（不改动机械臂）：')
                for b in bad:
                    print(f'   · {b}')
                return 2
            print('  ✅ 全部通过')

            if args.dry_run:
                print('\n--dry-run：不进入录制。')
                return 0

            outdir = os.path.join(
                args.out,
                time.strftime('%Y-%m-%d_%H%M%S') + (f'_{args.note}' if args.note else ''))
            os.makedirs(outdir, exist_ok=True)
            print(f'\n会话目录：{outdir}')
            meta = {
                'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                'note': args.note, 'seed': args.seed,
                'env_hz': cfg.ENV_HZ, 'action_scale': cfg.ACTION_SCALE,
                'max_episode_steps': cfg.MAX_EPISODE_STEPS,
                'image_size': cfg.IMAGE_SIZE, 'action_dim': N_ACTION,
                'joint_names': list(ARM_JOINTS),
                'controlled_joints': [ARM_JOINTS[i] for i in CONTROLLED_IDX],
                'frozen_joints': [ARM_JOINTS[i] for i in FROZEN_IDX if i < N_ARM],
                'randomize_start': bool(env.randomize_start),
                'randomize_half_m': env.randomize_half_m,
                'leader_lift_cm': LEADER_LIFT_DEFAULT_CM,
                'leader_home_offset_deg': [float(math.degrees(v))
                                           for v in leader_home_offset(verbose=False)],
                'gravity_comp_offset_deg': (load_home_cmd() or {}).get(
                    'follower_offset'),
                'keys': dict(KEYMAP),
                'note_on_return_phase':
                    'transitions during RETURN are NOT recorded (metadata only)',
                'episodes': [],
            }

            ep, quit_now = 0, False
            attempt, n_discarded = 0, 0
            while ep < args.episodes and not quit_now:
                attempt += 1
                discard, redo = False, False
                # ---------------- PRECISE: random start + leader alignment ----
                print(f'\n{"─" * 78}\n第 {ep + 1}/{args.episodes} 段  [PRECISE] 精确归位')
                env.set_intervening(False)           # 模式 A
                # 每次尝试都用不同的种子：被丢弃/跳过的尝试也要换个随机起点
                obs0, rinfo = env.reset(seed=args.seed + ep * 1000 + attempt)
                check_obs_layout(env, obs0, tag='[' + str(ep) + '] ')   # ★ 形态硬查
                st = rinfo.get('start', {})
                # ★ 契约检查：下面全部用 .get() 读，键名写错不会报错，只会**静默**
                #   把元数据写成 0.00cm —— 比 KeyError 更难发现。所以第一次就查。
                #   （踩坑 2026-09-29：随机化签名从 a_m/b_m 改成 dx_m/dy_m/dz_m，
                #     改了 env 却漏改录制器，元数据会悄悄全变 0。）
                missing = [k for k in START_INFO_KEYS if k not in st]
                if missing:
                    raise RuntimeError(
                        f'env 的 start_info 缺少 {missing}（实际有 {sorted(st)}）；\n'
                        f'     录制器的元数据会静默写成 0，先修 START_INFO_KEYS 或 env。')
                tilt = st.get('act_tilt_deg', float('nan'))
                if st.get('randomized'):
                    print(f'  随机起点：dx {st.get("dx_m", 0)*100:+.2f}cm / '
                          f'dy {st.get("dy_m", 0)*100:+.2f}cm / '
                          f'dz {st.get("dz_m", 0)*100:+.2f}cm')
                else:
                    print(f'  起点固定（随机化未生效：{st.get("reason", "--no-randomize")}）')
                print(f'  实际管子倾角 = {tilt:.2f}°（名义 1.66°）'
                      + ('   ⚠️ >3° → 重力补偿可能失配，建议重跑 phase1_home.py'
                         if tilt == tilt and tilt > 3.0 else ''))
                conv, resid, aq = align_leader(env, kb)
                print(f'  主臂对齐：{"✅" if conv else "⚠️ 未完全到位"} 残差 {resid:.1f}°')
                if aq:
                    quit_now = True
                    break

                # ---------------- READY ----------------
                kb.reassert()      # ★ Ctrl-Z + fg 会把终端退回 canonical，重设一次
                print(f'\n[READY] {KEYS_BY_PHASE["READY"]}')
                print('  ⚠️ 手离开主臂（模式 A 下主臂有力矩，会自己动）')
                t_wait = time.time()
                while True:
                    k = poll(kb, 'cCqQxXhH')
                    if k in ('c', 'C'):
                        break
                    if k in ('q', 'Q'):
                        # READY 阶段还没开始录，没有数据要丢 —— 只是换个随机起点
                        redo = True
                        print('  ↩ 跳过这个起点，重新随机一个')
                        break
                    if k in ('x', 'X'):
                        quit_now = True
                        break
                    if k in ('h', 'H'):
                        print(f'  [READY] {KEYS_BY_PHASE["READY"]}')
                    if time.time() - t_wait > 180:
                        print('  ⚠️ 已等 3 分钟没收到任何按键。若你确实在按：')
                        print('     终端模式可能被 Ctrl-Z/`fg` 弄丢了 —— 按 Ctrl-C 退出本脚本，')
                        print('     然后 `stty sane` 恢复终端，再重跑（本段不会保存）。')
                        t_wait = time.time()
                    time.sleep(0.05)
                if quit_now:
                    break
                if redo:
                    continue          # 回到 PRECISE，ep 不变

                # ---------------- RECORD ----------------
                env.set_intervening(True)            # 模式 B（差分基准在 _enter_mode 记录）
                # ★★ 两个必须在 RECORD 开始时做的设置 ★★
                # ① 死区归零：死区只决定 info 里报不报 intervene_action，不影响 mode B
                #    实际下发什么。默认 0.05 rad ≈ 2.86°，比 10Hz 下典型的人类单步
                #    位移还大 → 手臂动了 1~2° 而录下来的 action 是 0，transition 不自洽。
                #    （实测首批 52 段：81.1% 的步 action=0，其中大半手臂确实动了。）
                env.set_intervene_deadband(0.0)
                # ② 步数预算清零：PRECISE 阶段的 align_leader 会反复调 env.step()，
                #    对齐最长 12s（≈120 步）> max_episode_steps=100 → 不清零的话
                #    RECORD 的**第一步**就让 env 返回 done=True，录出只有 1 步、
                #    动作全 0 的空段。（实测首批 52 段里有 14 段就是这样来的。）
                env.unwrapped._step_count = 0
                kb.drain()
                kb.reassert()
                print(f'\n[RECORD] {KEYS_BY_PHASE["RECORD"]}')
                seg, n_itv, t_start = [], 0, time.time()
                done, truncated = False, False
                while not done:
                    k = poll(kb, 'dDqQxXhH')
                    if k in ('q', 'Q'):
                        # ★ 操作失误：本次**不记录**。仍然进 RETURN，让人把手臂安全带回。
                        discard = True
                        done = True
                        print('  ✖ 本次标为丢弃（不会写文件）；接下来把手臂带回起点')
                        break
                    if k in ('x', 'X'):
                        quit_now, discard = True, True
                        break
                    if k in ('h', 'H'):
                        print(f'  [RECORD] {KEYS_BY_PHASE["RECORD"]}')
                        continue
                    obs = cur_obs(env)
                    next_obs, rew, env_done, trunc, info = env.step(
                        np.zeros(N_ACTION, dtype=np.float32))
                    done = env_done
                    act = info.get('intervene_action',
                                   np.zeros(N_ACTION, dtype=np.float32))
                    itv = bool(info.get('intervening', False))
                    n_itv += int(itv)
                    if k in ('d', 'D'):
                        done = True                  # 人按 d：本段到此为止
                    if env_done and k not in ('d', 'D'):
                        # env 自己结束的（步数预算用满 / 判成功）—— 说清楚，
                        # 免得看起来像"按了 d 就没了"。踩坑：对齐阶段耗掉的
                        # 步数曾经让这里在第一步就触发（见上面的 _step_count 注释）。
                        truncated = bool(trunc) or len(seg) + 1 >= env.max_episode_steps
                        print(f'  ⓘ env 自己结束了本段（第 {len(seg) + 1} 步，'
                              f'step budget={env.max_episode_steps}）'
                              + ('，按"超长截断"记录' if truncated else ''))
                    seg.append(dict(
                        observations=obs, actions=np.asarray(act, dtype=np.float32),
                        next_observations=next_obs, rewards=0.0,
                        masks=0.0 if done else 1.0, dones=bool(done),
                        infos={'intervening': itv, 'phase': 'record'},
                    ))
                seg_sec = time.time() - t_start
                hz_act = len(seg) / max(seg_sec, 1e-6)

                # ★ 退化段守卫：一次真正的插入不可能只有一两步。
                #   2026-09-29 首批录制里有 14/52 段只有 **1 步**（record_sec≈0.12s），
                #   都是 `d` 在 RECORD 第一次轮询就到了 —— 一段插入都没发生过，
                #   却照样写进了数据集（要手工标成 failure 才不污染）。
                #   这里直接判为丢弃，并说清原因。
                if len(seg) < args.min_steps and not discard:
                    discard = True
                    print(f'  ✖ 只录到 {len(seg)} 步（< --min-steps {args.min_steps}），'
                          f'不可能是一次真实插入 → 自动丢弃')
                    print('     （若确实想留，用 --min-steps 1；正常情况下 `d` 要在'
                          '做完插入之后再按）')

                # ---------------- RETURN (mode B stays on) ----------------
                ret_meta = {}
                if not quit_now:
                    q_before = (np.asarray(env.joint_pos[:N_ARM]).copy()
                                if env.joint_pos is not None else None)
                    kb.reassert()
                    print(f'\n[RETURN] {KEYS_BY_PHASE["RETURN"]}')
                    print('  ★ 主臂**仍然控制**从臂。把从臂带回起点附近（管子顺势拔出），')
                    print('    这也会把主臂自己带回它的家 —— 解决主臂漂移的关键一步。')
                    t_ret0 = time.time()
                    while True:
                        k = poll(kb, 'cCqQxXhH')
                        if k in ('c', 'C'):
                            break
                        if k in ('q', 'Q'):
                            discard = not discard
                            print(f'  {"✖ 已标记为丢弃" if discard else "↩ 取消丢弃标记"}'
                                  f'（按 c 结束回程）')
                            continue
                        if k in ('x', 'X'):
                            quit_now, discard = True, True
                            break
                        if k in ('h', 'H'):
                            print(f'  [RETURN] {KEYS_BY_PHASE["RETURN"]}')
                            continue
                        env.step(np.zeros(N_ACTION, dtype=np.float32))   # 仍模式 B
                        if time.time() - t_ret0 > 60.0:
                            print('  ⚠️ 回程已超过 60s，还在继续（不强制打断）')
                            t_ret0 = time.time()
                    ret_sec = time.time() - t_ret0
                    q_after = (np.asarray(env.joint_pos[:N_ARM]).copy()
                               if env.joint_pos is not None else None)
                    travel = (float(np.abs(q_after - q_before).max())
                              if q_before is not None and q_after is not None else float('nan'))
                    ret_meta = {'sec': round(ret_sec, 2),
                                'joint_travel_deg': round(math.degrees(travel), 2),
                                'tilt_after_deg': round(phase_tilt(env), 2)}

                # ---------------- save ----------------
                if discard:
                    n_discarded += 1
                    print(f'\n  🗑  本次已丢弃，未写任何文件（第 {attempt} 次尝试）。'
                          f'累计丢弃 {n_discarded} 次；ep 序号仍是 {ep}。')
                    if quit_now:
                        break
                    continue
                epfile = os.path.join(outdir, f'ep_{ep:04d}.pkl')
                with open(epfile, 'wb') as f:
                    pickle.dump({'transitions': seg, 'meta': {
                        'index': ep, 'start_info': st,
                        'n_steps': len(seg), 'n_intervened': n_itv,
                        'intervened_frac': round(n_itv / max(len(seg), 1), 3),
                        'record_sec': round(seg_sec, 2),
                        'hz_actual': round(hz_act, 2),
                        'truncated': bool(truncated or quit_now),
                        'return': ret_meta,
                    }}, f)
                quality = []
                if n_itv == 0:
                    quality.append('❌ 全程没有接管步（忘了进模式 B？）这一段没用')
                if abs(hz_act - env.hz) > 1.0:
                    quality.append(f'⚠️ 实际频率 {hz_act:.1f}Hz 偏离目标 {env.hz:.0f}Hz '
                                   f'—— 动作语义是"每步"，频率变了含义就变了')
                print(f'\n  已存 {epfile}')
                print(f'  步数 {len(seg)}  接管 {n_itv} 步 ({n_itv/max(len(seg),1)*100:.0f}%)  '
                      f'{seg_sec:.1f}s ({hz_act:.1f}Hz)'
                      + (f'  回程 {ret_meta.get("sec")}s 位移 '
                         f'{ret_meta.get("joint_travel_deg")}°' if ret_meta else ''))
                for q_ in quality:
                    print(f'  {q_}')
                meta['episodes'].append({
                    'file': os.path.basename(epfile), 'n_steps': len(seg),
                    'n_intervened': n_itv, 'hz_actual': round(hz_act, 2),
                    'truncated': bool(truncated or quit_now),
                    'start_tilt_deg': st.get('act_tilt_deg'),
                    'return': ret_meta,
                })
                ep += 1

            meta['n_discarded'] = n_discarded
            meta['n_attempts'] = attempt
            with open(os.path.join(outdir, 'meta.json'), 'w', encoding='utf-8') as f:
                import json
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f'\n{"=" * 78}')
            print(f'会话结束：录了 {ep} 段（共尝试 {attempt} 次，丢弃 {n_discarded} 次），'
                  f'写在 {outdir}')
            print('  meta.json + ep_XXXX.pkl')
            print(f'  下一步：DISPLAY=:0 python label_demos.py {outdir}/*.pkl')
            print(f'{"=" * 78}')
            return 0
    except KeyboardInterrupt:
        print('\n中断（Ctrl-C）。')
        return 1
    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            rclpy.try_shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    sys.exit(main())
