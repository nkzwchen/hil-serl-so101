#!/usr/bin/env python3
"""Read live images through SO101Env without reset(), step(), or torque calls."""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE / 'hil-serl' if (HERE / 'hil-serl').is_dir() else HERE.parent
sys.path.insert(0, str(REPO / 'examples'))


def main():
    import cv2
    import numpy as np
    import rclpy
    from experiments.so101.config import TrainConfig
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=20.)
    parser.add_argument('--out', default='cam_probe/live_observations')
    parser.add_argument('--test-rois', action='store_true',
                        help='Temporary 640x480 test rectangles; never save to training config')
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    cfg = TrainConfig()
    if args.test_rois:
        base = dict(version=1, camera='cam_main', source_size=[640, 480],
                    preprocess='letterbox_rgb_v1')
        cfg.main_roi_spec = {**base, 'xyxy': [140, 200, 500, 480]}
        cfg.roi_spec = {**base, 'xyxy': [275, 310, 440, 480]}
        cfg.image_keys = ['cam_main', 'cam_wrist', 'cam_main_roi']
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = cfg.get_environment(enable_keyboard=False)
    raw = env.unwrapped
    fake = cfg.get_environment(fake_env=True)
    result = dict(scope='Camera observations only; no robot motion or joint validation',
                  test_rois=args.test_rois, main_roi_spec=cfg.main_roi_spec,
                  roi_spec=cfg.roi_spec, samples=0, bad_camera_samples=0,
                  repeated_samples=0, shape_failures=0,
                  stale_or_invalid_samples=0, skew_samples=0,
                  max_source_skew_s=0., source_lag_s={})
    source_lags = {k: [] for k in ['cam_main', 'cam_wrist']}
    last_stamps = None
    try:
        deadline = time.monotonic() + 5.
        while time.monotonic() < deadline:
            with raw._lock:
                ready = all(k in raw._images for k in cfg.image_keys)
            if ready:
                break
            time.sleep(.05)
        start = time.monotonic()
        while time.monotonic() - start < args.seconds:
            obs = env.env.observation(raw._get_obs())
            quality = raw.last_observation_quality()
            stamps = [v['source_stamp_ns'] for v in quality['cameras'].values()]
            result['samples'] += 1
            stale = any(not v['valid'] or v['age_s'] is None or v['age_s'] > .25
                        for v in quality['cameras'].values())
            have_both_stamps = all(stamps)
            skewed = have_both_stamps and quality['camera_skew_s'] > quality['max_camera_skew_s']
            bad = stale or skewed
            result['stale_or_invalid_samples'] += int(stale)
            result['skew_samples'] += int(skewed)
            if have_both_stamps:
                result['max_source_skew_s'] = max(result['max_source_skew_s'], quality['camera_skew_s'])
            for key, stats in quality['cameras'].items():
                if stats['source_stamp_ns']:
                    source_lags[key].append(time.time() - stats['source_stamp_ns'] / 1e9)
            result['bad_camera_samples'] += int(bad)
            if last_stamps is not None and all(last_stamps) and have_both_stamps:
                result['repeated_samples'] += int(any(b <= a for a, b in zip(last_stamps, stamps)))
            last_stamps = stamps
            stacked = {k: v[None] for k, v in obs.items()}
            result['shape_failures'] += int(not fake.observation_space.contains(stacked))
            time.sleep(.1)
        result['cameras'] = raw.observation_quality()['cameras']
        result['source_lag_s'] = {k: {'min': min(v), 'median': float(np.median(v)), 'max': max(v)}
                                  for k, v in source_lags.items() if v}
        result['saved_views'] = []
        for key in cfg.image_keys:
            physical_key = 'cam_main' if key == 'cam_main_roi' else key
            if result['cameras'][physical_key]['accepted']:
                cv2.imwrite(str(out / f'{key}.png'), obs[key][:, :, ::-1])
                result['saved_views'].append(key)
        (out / 'report.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return int(result['shape_failures'] > 0 or result['bad_camera_samples'] > 0
                   or result['repeated_samples'] > 0)
    finally:
        env.close()
        rclpy.try_shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
