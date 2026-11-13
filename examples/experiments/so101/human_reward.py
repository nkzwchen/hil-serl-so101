"""Episode-level human rewards. No robot, ROS or learned reward dependencies."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pickle


class RecordingUnpickler(pickle.Unpickler):
    """Read trusted NumPy 2 recordings with the cloud's NumPy 1.26 runtime."""
    def find_class(self, module, name):
        if module.startswith('numpy._core') and int(np.__version__.split('.')[0]) < 2:
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)


def capture_summary(transitions, meta):
    """Full RECORD segment statistics, before success-frame trimming; unknown is not zero."""
    import math
    flags = [t.get('infos', {}).get('intervening') for t in transitions]
    known = all(isinstance(x, (bool, np.bool_)) for x in flags) and bool(flags)
    duration = meta.get('record_sec')
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
        duration = None
    return dict(steps=len(transitions),
                intervention_steps=sum(bool(x) for x in flags) if known else None,
                intervention_fraction=sum(bool(x) for x in flags)/len(flags) if known else None,
                duration_s=duration, timeout=meta.get('episode_timeout'),
                policy_present=bool(meta.get('policy')))


def load_episode(path, config, allow_provisional=False):
    import pickle
    path = Path(path)
    label_path = path.with_suffix('.labels.json')
    labels = json.loads(label_path.read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if labels.get('recording_sha256') != digest:
        raise ValueError(f'{path.name}: missing/mismatched recording hash; review labels again')
    lab = labels['episodes']['0']
    if lab.get('provisional', False):
        if not allow_provisional:
            raise ValueError('Provisional label requires --allow-provisional')
    elif not lab.get('reviewed', False) or lab.get('source') not in ('human_review', 'derived_human_review'):
        raise ValueError('Human review required')
    if lab.get('status') not in ('success', 'failed'):
        raise ValueError('Episode must be explicitly labeled success or failed')
    with path.open('rb') as f:
        raw = RecordingUnpickler(f).load()  # Only trusted, user-owned recording files.
    meta, ts = raw['meta'], raw['transitions']
    if lab.get('source') == 'derived_human_review':
        conversion = meta.get('conversion', {})
        original = conversion.get('source_label', {})
        spans = conversion.get('source_ranges', [])
        if (conversion.get('version') != '10hz_scale55_v1' or
            original.get('source') != 'human_review' or not original.get('reviewed') or original.get('provisional') or
            original.get('status') != lab.get('status') or
            not conversion.get('source_sha256') or conversion['source_sha256'] != lab.get('source_recording_sha256') or
            len(spans) != len(ts) or not spans):
            raise ValueError('Invalid derived label provenance')
        if any(not isinstance(s, list) or len(s)!=2 or not all(type(i) is int for i in s) or s[0]<0 or s[1]<=s[0] for s in spans):
            raise ValueError('Invalid source ranges')
        if any(a[1]!=b[0] for a,b in zip(spans,spans[1:])):
            raise ValueError('Derived episode crosses discarded source steps')
        if original['status']=='success' and (spans[-1][1]!=original.get('frame') or lab.get('frame')!=len(ts)):
            raise ValueError('Derived success frame is misaligned')
        kind=conversion.get('kind')
        if kind=='approximate_30hz_success_suffix':
            if not getattr(config, 'ALLOW_APPROXIMATE_DEMOS', False):
                raise ValueError('Approximate 30Hz conversion requires --allow-approximate-demos')
        elif kind!='native10_scale_only':
            raise ValueError('Unknown conversion kind')
    if meta.get('operator_excluded') or meta.get('recording', {}).get('has_gaps'):
        raise ValueError('Excluded episode or missing recording steps')
    if meta.get('safety_error'):
        raise ValueError('Safety-interrupted episode retained locally; exclude from automatic training')
    expected = dict(action_semantics=config.ACTION_SEMANTICS,
                    observation_semantics=config.OBSERVATION_SEMANTICS,
                    action_scale_rad=config.ACTION_SCALE, action_joints=config.ACTION_JOINTS,
                    env_hz=config.ENV_HZ, control_backend=config.CONTROL_BACKEND,
                    pose_alignment=config.POSE_ALIGNMENT, roi_spec=config.roi_spec,
                    main_roi_spec=config.main_roi_spec)
    for key, val in expected.items():
        if meta.get(key) != val:
            raise ValueError(f'{path.name}: incompatible {key}')
    success = lab['status'] == 'success'
    end = lab.get('frame') if success else len(ts)
    if isinstance(end, bool) or not isinstance(end, int) or not 1 <= end <= len(ts):
        raise ValueError('Success observation frame must be in 1..N')
    env = config.get_environment(fake_env=True)
    if success and end < len(ts) and ts[end].get('infos', {}).get('segment_start'):
        raise ValueError('Success frame lies at a pause boundary; select a frame after an executed action')
    result = []
    for i, raw_t in enumerate(ts[:end]):
        for field in ('observations', 'next_observations'):
            obs = raw_t[field]
            if set(obs) != set(env.observation_space.spaces):
                raise ValueError('Observation keys mismatch')
            for key, space in env.observation_space.spaces.items():
                a = np.asarray(obs[key])
                if a.shape != space.shape or a.dtype != space.dtype or not np.isfinite(a).all():
                    raise ValueError(f'Invalid {field}/{key}')
        if not env.action_space.contains(raw_t['actions']):
            raise ValueError('Invalid action')
        t = {key: raw_t[key] for key in ('observations', 'actions', 'next_observations')}
        final = i == end - 1
        boundary = i+1 < len(ts) and bool(ts[i+1].get('infos', {}).get('segment_start'))
        t.update(rewards=np.float32(success and final), masks=np.float32(not (success and final)), dones=final or boundary)
        result.append(t)
    receipt = dict(id=hashlib.sha256((digest + json.dumps(lab, sort_keys=True)).encode()).hexdigest(),
                   recording_sha256=digest, file=path.name, label=copy.deepcopy(lab),
                   replay_semantics='human_terminal_success_bootstrap_cutoff_v2',
                   termination='success' if success else ('timeout' if meta.get('episode_timeout') else 'manual_cutoff'),
                   intervention_indices=[i for i,t in enumerate(ts[:end]) if t.get('infos', {}).get('intervening') is True or isinstance(t.get('infos', {}).get('intervening'), np.bool_) and bool(t['infos']['intervening'])],
                   transitions=len(result), capture=capture_summary(ts, meta), quality_flagged=sum(not t.get('infos', {}).get('quality_ok', True) for t in ts[:end]))
    return result, receipt
