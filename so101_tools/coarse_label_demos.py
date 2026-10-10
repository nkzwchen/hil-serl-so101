#!/usr/bin/env python3
"""Generate explicitly provisional success labels without changing recordings."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime, timezone

from so101_recording_io import load_raw, labels_path


def coarse_label(transitions, fraction=0.97):
    if not transitions or not 0 < fraction <= 1:
        raise ValueError('Need a nonempty episode and 0 < fraction <= 1')
    # Observation k is reached by transition k-1. Prefer actual control elapsed
    # time over nominal FPS; these timestamps are NOT camera exposure times.
    controls = [t.get('infos', {}).get('joint_control', {}) for t in transitions]
    start = controls[0].get('step_started_monotonic_s')
    ends = [c.get('step_finished_monotonic_s') for c in controls]
    valid = all(isinstance(x, (float, int)) and math.isfinite(x)
                for x in [start, *ends])
    valid = valid and start < ends[-1] and all(a <= b for a, b in zip([start, *ends], ends))
    if valid:
        target = start + fraction * (ends[-1] - start)
        k = next(i + 1 for i, timestamp in enumerate(ends) if timestamp >= target)
        basis = 'control_monotonic_elapsed_time'
    else:
        k = max(1, math.ceil(fraction * len(transitions)))
        basis = 'transition_count_fallback'
    return dict(status='success', frame=k, provisional=True,
                source='user_requested_coarse_rule', reviewed=False,
                rule=dict(fraction=fraction, basis=basis,
                          indexing='observation k; reward on transition k-1'),
                transition_count=len(transitions),
                success_transition_quality_ok=bool(transitions[k-1].get('infos', {}).get('quality_ok', False)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('session', type=Path)
    ap.add_argument('--fraction', type=float, default=.97)
    args = ap.parse_args()
    for path in sorted(args.session.glob('ep_*.pkl')):
        dest = Path(labels_path(str(path)))
        if dest.exists():
            print(f'SKIP existing labels: {dest.name}')
            continue
        episodes, _ = load_raw(path)
        labels = {str(i): coarse_label(ts, args.fraction) for i, ts in enumerate(episodes)}
        with path.open('rb') as src:
            digest = hashlib.file_digest(src, 'sha256').hexdigest()
        out = dict(recording=path.name, recording_sha256=digest,
                   labeled_at=datetime.now(timezone.utc).isoformat(),
                   n_episodes=len(episodes), episodes=labels)
        # Exclusive creation never overwrites human labels.
        with dest.open('x', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(path.name, [(v['frame'], v['transition_count'], v['rule']['basis']) for v in labels.values()])


if __name__ == '__main__':
    main()
