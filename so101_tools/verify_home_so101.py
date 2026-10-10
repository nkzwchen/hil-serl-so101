#!/usr/bin/env python3
"""Validate fixed-start reset and recorder leader alignment; never record demos."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'hil-serl/examples'))
sys.path.insert(0, str(ROOT / 'hil-serl/examples/experiments/so101'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempts', type=int, default=1)
    parser.add_argument('--out', required=True)
    parser.add_argument('--execute', action='store_true', help='Actually move hardware')
    args = parser.parse_args()
    if not args.execute:
        parser.error('Requires --execute and an on-site operator; this moves both arms')
    if not 1 <= args.attempts <= 5:
        parser.error('--attempts must be between 1 and 5')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import rclpy
    from experiments.so101.config import TrainConfig
    from so101_home_pose import FROZEN_IDX, START_JOINT_POSE, JOINT_NAMES
    from record_demo_so101 import NonBlockingKey, preflight, align_leader, check_obs_layout

    result = dict(scope='Fixed-start home and leader alignment; no recording',
                  requested_attempts=args.attempts, rows=[], passed=False)
    cfg = TrainConfig()
    result['roi'] = dict(main=cfg.main_roi_spec, third=cfg.roi_spec)
    rclpy.init()
    env = cfg.get_environment(enable_keyboard=False, randomize_start=False)
    raw = env.unwrapped
    try:
        with NonBlockingKey() as kb:
            # Reuse recorder checks, but enable torque only after every check passes.
            checks = SimpleNamespace(dry_run=True, episodes=args.attempts, out=str(out))
            problems, warnings = preflight(checks, env, kb)
            result['preflight'] = dict(problems=problems, warnings=warnings)
            if problems:
                raise RuntimeError('Preflight failed: ' + '; '.join(problems))
            peers = raw.node.get_publishers_info_by_topic(
                '/follower/arm_controller/joint_trajectory')
            result['trajectory_publishers'] = [p.node_name for p in peers]
            if len(peers) != 1 or peers[0].node_name != raw.node.get_name():
                raise RuntimeError('Follower trajectory control source is not exclusive')
            if not raw.set_follower_torque(True):
                raise RuntimeError('Follower torque enable failed')
            for index in range(args.attempts):
                row = dict(index=index + 1, passed=False)
                result['rows'].append(row)
                row['before_joint_rad'] = raw.joint_pos.tolist()
                frozen_before = np.asarray(raw.joint_pos)[list(FROZEN_IDX)].copy()
                started = time.monotonic()
                print(f'HOME {index + 1}/{args.attempts}: fixed start', flush=True)
                env.set_intervening(False)
                obs, info = env.reset(seed=index)
                row['reset_seconds'] = time.monotonic() - started
                row['start'] = info['start']
                row['target_joint_rad'] = np.asarray(START_JOINT_POSE).tolist()
                row['after_reset_joint_rad'] = raw.joint_pos.tolist()
                check_obs_layout(env, obs, tag='home validation ')
                if not info['start'].get('accepted'):
                    raise RuntimeError('Reset did not report accepted')
                if not raw.set_leader_torque(True):
                    raise RuntimeError('Leader torque enable failed')
                started = time.monotonic()
                converged, residual, quit_now = align_leader(env, kb)
                row['alignment_seconds'] = time.monotonic() - started
                row['alignment_residual_deg'] = residual
                row['after_alignment_joint_rad'] = raw.joint_pos.tolist()
                row['leader_joint_rad'] = raw.leader_pos.tolist()
                row['frozen_measured_change_deg'] = dict(zip(
                    [JOINT_NAMES[i] for i in FROZEN_IDX],
                    np.degrees(np.asarray(raw.joint_pos)[list(FROZEN_IDX)] - frozen_before).tolist()))
                row['leader_alignment_converged'] = converged
                if quit_now:
                    raise RuntimeError('Leader alignment cancelled')
                if (time.monotonic() - raw._leader_stamp > .25 or
                        time.monotonic() - raw._joint_stamp > .25):
                    raise RuntimeError('Stale joint feedback after alignment')
                residual_hold = float(np.max(np.abs(np.asarray(raw.joint_pos)[:4] -
                                                     np.asarray(START_JOINT_POSE)[:4])))
                row['follower_hold_residual_deg'] = float(np.degrees(residual_hold))
                if residual_hold > raw.start_feedback_tol:
                    raise RuntimeError('Follower drifted during leader alignment')
                row['passed'] = True
                print(json.dumps(row, ensure_ascii=False), flush=True)
                (out / 'report.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
            result['passed'] = True
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        print(result['error'], flush=True)
    finally:
        (out / 'report.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
        env.close()
        rclpy.try_shutdown()
        print('Validation exited; bridge/controller remain running, torque was not released.', flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
