#!/usr/bin/env python3
"""Read-only TensorBoard exporter for live learner logs and accepted human labels."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))


def scalars(tree, prefix='train'):
    for key, value in tree.items():
        tag = prefix + '/' + key
        if isinstance(value, dict):
            yield from scalars(value, tag)
        elif isinstance(value, (int, float)) and math.isfinite(value):
            yield tag, float(value)


def episode_scalars(rows):
    """One independent episode axis per demo/online pool; never mix provisional labels."""
    counts = {}
    history = {}
    for row in rows:
        pool = row.get('role', 'demo')
        if pool not in ('demo', 'online'):
            continue
        counts[pool] = counts.get(pool, 0) + 1
        step = counts[pool]
        prefix = f'episodes/{pool}'
        lab = row['label']
        reviewed = lab.get('reviewed') and not lab.get('provisional') and lab.get('source') == 'human_review'
        reviewed = reviewed and lab.get('status') in ('success', 'failed')
        yield prefix + '/received_count', step, step
        yield prefix + '/training_steps', row['transitions'], step
        yield prefix + '/quality_flagged_training_steps', row.get('quality_flagged', 0), step
        cap = row.get('capture', {})
        for key in ('steps', 'intervention_steps', 'intervention_fraction', 'duration_s', 'timeout'):
            value = cap.get(key)
            if value is not None and isinstance(value, (int, float)) and math.isfinite(value):
                yield prefix + '/capture_' + key, float(value), step
        h = history.setdefault(pool, [])
        h.append(dict(reviewed=bool(reviewed), success=lab.get('status') == 'success', cap=cap))
        valid = [r for r in h if r['reviewed']]
        yield prefix + '/reviewed_count', len(valid), step
        yield prefix + '/unreviewed_or_provisional_count', len(h)-len(valid), step
        if reviewed:
            yield prefix + '/human_return', float(lab['status'] == 'success'), step
        if valid:
            yield prefix + '/success_rate', sum(r['success'] for r in valid)/len(valid), step
            window = valid[-20:]
            yield prefix + '/last20_success_rate', sum(r['success'] for r in window)/len(window), step
            yield prefix + '/last20_count', len(window), step
        autonomous = [r for r in valid if r['cap'].get('policy_present') and r['cap'].get('intervention_fraction') == 0]
        yield prefix + '/autonomous_reviewed_count', len(autonomous), step
        if autonomous:
            yield prefix + '/autonomous_success_rate', sum(r['success'] for r in autonomous)/len(autonomous), step
        timed = [r['cap']['timeout'] for r in h if r['cap'].get('timeout') is not None]
        if timed:
            yield prefix + '/timeout_rate', sum(timed)/len(timed), step


class Monitor:
    def __init__(self, run, logdir, data=None, recordings=None):
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
        self.run, self.data = Path(run), Path(data) if data else None
        self.recordings = Path(recordings) if recordings else None
        self.writer = EventFileWriter(str(logdir))
        self.offset = 0
        self.metric_step = -1
        self.receipt_signature = None
        self.pending_signature = None
        self.cache = {}
        self.pending_step = 0

    def scalar(self, tag, value, step):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        self.writer.add_event(Event(wall_time=time.time(), step=int(step),
                                    summary=Summary(value=[Summary.Value(tag=tag, simple_value=float(value))])))

    def enrich(self, row):
        if 'capture' in row or not self.data:
            return row
        if row['id'] in self.cache:
            return self.cache[row['id']]
        from experiments.so101.human_reward import RecordingUnpickler, capture_summary
        path = (self.data / row['file']).resolve()
        if path.parent != self.data.resolve():
            raise ValueError('Receipt path outside supplied data directory')
        with path.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(f.read()).hexdigest()
            if digest != row['recording_sha256']:
                raise ValueError('Receipt recording hash mismatch')
            f.seek(0)
            raw = RecordingUnpickler(f).load()
        enriched = dict(row, capture=capture_summary(raw['transitions'], raw.get('meta', {})))
        self.cache[row['id']] = enriched
        return enriched

    def poll(self):
        metrics = self.run / 'metrics.jsonl'
        if metrics.exists():
            if metrics.stat().st_size < self.offset:
                raise ValueError('metrics.jsonl was truncated; use a fresh monitor directory')
            with metrics.open('rb') as f:
                f.seek(self.offset)
                while True:
                    position = f.tell()
                    line = f.readline()
                    if not line or not line.endswith(b'\n'):
                        self.offset = position
                        break
                    row = json.loads(line)
                    if row['step'] > self.metric_step:
                        for tag, value in scalars(row['metrics']):
                            self.scalar(tag, value, row['step'])
                        self.metric_step = row['step']
                    self.offset = f.tell()
        receipts = self.run / 'receipts.json'
        if receipts.exists():
            try:
                content = receipts.read_bytes()
                rows = json.loads(content)
            except (json.JSONDecodeError, OSError):
                # Existing learner versions replace this file non-atomically.
                rows = None
            if rows is not None and content != self.receipt_signature:
                rows = [self.enrich(r) for r in rows]
                for row in rows:
                    row['_new'] = row['id'] not in getattr(self, 'seen', set())
                new_steps = {}
                counts = {}
                for row in rows:
                    pool = row.get('role','demo'); counts[pool]=counts.get(pool,0)+1
                    if row['_new']: new_steps.setdefault(pool,set()).add(counts[pool])
                for tag, value, step in episode_scalars(rows):
                    pool=tag.split('/')[1]
                    if step in new_steps.get(pool,set()): self.scalar(tag,value,step)
                self.seen = {r['id'] for r in rows}
                self.receipt_signature = content
        # Pending labels live on the actor, not the learner. Enable only where those files exist.
        if self.recordings:
            pending = {}
            for path in sorted(self.recordings.glob('**/ep_*.pkl')):
                state_path = path.with_suffix('.delivery.json')
                try:
                    state = json.loads(state_path.read_text())['status'] if state_path.exists() else 'untracked'
                except (OSError, json.JSONDecodeError):
                    continue
                pending[state] = pending.get(state, 0) + 1
            signature = json.dumps(pending, sort_keys=True)
            if signature != self.pending_signature:
                self.pending_step += 1
                for state in ('pending_label','pending_upload','pending_submit','submitted','excluded','skipped','untracked'):
                    self.scalar('local_queue/'+state, pending.get(state,0), self.pending_step)
                self.pending_signature = signature
        self.writer.flush()

    def close(self):
        self.writer.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True, help='Learner output directory containing metrics.jsonl and receipts.json')
    ap.add_argument('--data', help='Original inbox, to enrich receipts from older learner versions')
    ap.add_argument('--recordings', help='Optional local actor recordings tree for pending delivery counts')
    ap.add_argument('--logdir', help='Fresh event directory (default: RUN/tensorboard/live_TIMESTAMP)')
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--interval', type=float, default=2.)
    args = ap.parse_args()
    if args.interval <= 0: ap.error('--interval must be positive')
    logdir = Path(args.logdir) if args.logdir else Path(args.run)/'tensorboard'/('live_'+time.strftime('%Y%m%d_%H%M%S'))
    logdir.mkdir(parents=True, exist_ok=True)
    if list(logdir.glob('events.out.tfevents.*')):
        ap.error('Use a fresh logdir; old event files are preserved to avoid duplicate histories')
    monitor = Monitor(args.run, logdir, args.data, args.recordings)
    print(f'TensorBoard events: {logdir}', flush=True)
    try:
        while True:
            monitor.poll()
            if args.once: break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.close()

if __name__ == '__main__': main()
