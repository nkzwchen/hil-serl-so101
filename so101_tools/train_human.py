#!/usr/bin/env python3
"""SO101 human-annotated SAC learner and SSH-tunneled episode/policy client."""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'examples'))
sys.path.insert(0, str(REPO / 'serl_launcher'))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')


def wire_config(port):
    from agentlace.trainer import TrainerConfig
    return TrainerConfig(port_number=port, broadcast_port=port+1,
                         request_types=['status', 'submit', 'policy'])


def client(args):
    from agentlace.trainer import TrainerClient
    class EpisodeClient(TrainerClient):
        def update(self):
            # This protocol has no raw transition stores; admission is episode RPC only.
            return True
    return EpisodeClient('human_episode_client', args.host, wire_config(args.port),
                         wait_for_server=False, timeout_ms=120000)


def make_agent(args, config):
    import numpy as np
    from serl_launcher.utils.launcher import make_sac_pixel_agent
    env = config.get_environment(fake_env=True)
    obs = {k: np.zeros(s.shape, s.dtype) for k, s in env.observation_space.spaces.items()}
    return make_sac_pixel_agent(args.seed, obs, np.zeros(env.action_space.shape, np.float32),
                               image_keys=config.image_keys, encoder_type=args.encoder,
                               discount=config.discount)


def train(args):
    import jax
    import numpy as np
    from flax import serialization
    from flax.core import freeze
    from agentlace.trainer import TrainerServer
    from experiments.so101.config import TrainConfig
    from experiments.so101.human_reward import load_episode
    cfg = TrainConfig()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    inbox = Path(args.data).resolve()
    demos, online, receipts = [], [], {}
    previous = {}
    if args.resume:
        manifest = Path(args.resume).parent / 'receipts.json'
        previous = {r['recording_sha256']: r for r in json.loads(manifest.read_text())}
    for path in sorted(inbox.glob('*.pkl')):
        ts, receipt = load_episode(path, cfg, args.allow_provisional)
        if receipt['id'] in receipts:
            continue  # Same episode under another filename must not double replay on restart.
        if any(r['recording_sha256'] == receipt['recording_sha256'] for r in receipts.values()):
            raise ValueError('Conflicting labels for duplicate recording in inbox')
        prior = previous.get(receipt['recording_sha256'])
        if prior and prior['id'] != receipt['id']:
            raise ValueError('Labels changed since checkpoint; start a new run')
        receipt['role'] = prior.get('role', 'demo') if prior else ('online' if args.resume else 'demo')
        (online if receipt['role'] == 'online' else demos).extend(ts)
        receipts[receipt['id']] = receipt
    if previous and not set(previous).issubset({r['recording_sha256'] for r in receipts.values()}):
        raise ValueError('Resume is missing previously ingested recordings')
    if not demos:
        raise ValueError('No labeled demonstrations')
    print(f'Loaded {len(receipts)} episodes / {len(demos)} transitions; devices={jax.devices()}', flush=True)
    agent = make_agent(args, cfg)
    spec = dict(encoder=args.encoder, seed=args.seed, action_semantics=cfg.ACTION_SEMANTICS,
                observation_semantics=cfg.OBSERVATION_SEMANTICS, image_keys=cfg.image_keys,
                reward_source=cfg.REWARD_SOURCE, allow_provisional=args.allow_provisional,
                action_scale_rad=cfg.ACTION_SCALE, env_hz=cfg.ENV_HZ,
                pose_alignment=cfg.POSE_ALIGNMENT, roi_spec=cfg.roi_spec, main_roi_spec=cfg.main_roi_spec)
    step = 0
    if args.resume:
        saved = json.loads(Path(args.resume + '.json').read_text())
        if saved['spec'] != spec:
            raise ValueError('Checkpoint configuration differs')
        agent = agent.replace(state=serialization.from_bytes(agent.state, Path(args.resume).read_bytes()))
        step = saved['step']
    lock = threading.Lock()
    snapshot = dict(params=jax.device_get(agent.state.params), spec=spec, step=step)

    def callback(kind, payload):
        try:
            if kind == 'status':
                with lock:
                    return dict(ok=True, step=snapshot['step'], demo_transitions=len(demos),
                                online_transitions=len(online), episodes=len(receipts), spec=spec)
            if kind == 'policy':
                with lock:
                    return dict(ok=True, **snapshot)
            if kind == 'submit':
                name = payload['file']
                if Path(name).name != name or not name.endswith('.pkl'):
                    raise ValueError('Only recording basenames within inbox are accepted')
                path = (inbox / name).resolve()
                if path.parent != inbox:
                    raise ValueError('Recording outside inbox')
                ts, receipt = load_episode(path, cfg, args.allow_provisional)
                with lock:
                    duplicate = receipt['id'] in receipts
                    if not duplicate:
                        if any(r['recording_sha256'] == receipt['recording_sha256'] for r in receipts.values()):
                            raise ValueError('Relabeled episode already ingested; restart learner to rebuild replay')
                        receipt['role'] = 'online'
                        online.extend(ts)
                        receipts[receipt['id']] = receipt
                        (out / 'receipts.json').write_text(json.dumps(list(receipts.values()), indent=2))
                    return dict(ok=True, duplicate=duplicate, transitions=len(ts))
            raise ValueError('Unknown request')
        except Exception as exc:
            return dict(ok=False, error=str(exc))

    server = TrainerServer(wire_config(args.port), request_callback=callback) if args.serve_seconds else None
    if server:
        server.start(threaded=True)
    def save_checkpoint():
        checkpoint = out / f'checkpoint_{step}.msgpack'
        tmp = checkpoint.with_suffix('.tmp')
        tmp.write_bytes(serialization.to_bytes(agent.state))
        tmp.replace(checkpoint)
        Path(str(checkpoint)+'.json').write_text(json.dumps(dict(spec=spec, step=step), indent=2))
        return checkpoint

    rng = np.random.default_rng(args.seed)
    (out / 'receipts.json').write_text(json.dumps(list(receipts.values()), indent=2))
    try:
        for _ in range(args.steps):
            with lock:
                n_online = args.batch_size // 2 if online else 0
                picked = [demos[i] for i in rng.integers(len(demos), size=args.batch_size-n_online)]
                if n_online:
                    picked += [online[i] for i in rng.integers(len(online), size=n_online)]
            batch = freeze(jax.tree_util.tree_map(lambda *xs: np.stack(xs), *picked))
            agent, metrics = agent.update(batch)
            metrics = jax.device_get(metrics)
            if not all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(metrics)):
                raise RuntimeError('Nonfinite training metrics')
            step += 1
            record = dict(step=step, metrics=jax.tree_util.tree_map(lambda x: np.asarray(x).tolist(), serialization.to_state_dict(metrics)))
            with (out / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            print(f'update {step}: metrics finite', flush=True)
            with lock:
                snapshot.update(params=jax.device_get(agent.state.params), step=step)
            if step % args.checkpoint_every == 0:
                save_checkpoint()
        checkpoint = save_checkpoint()
        restored = agent.replace(state=serialization.from_bytes(agent.state, checkpoint.read_bytes()))
        obs = demos[0]['observations']
        before = np.asarray(agent.sample_actions(obs, argmax=True))
        after = np.asarray(restored.sample_actions(obs, argmax=True))
        np.testing.assert_allclose(before, after, rtol=0, atol=0)
        assert after.shape == (3,) and np.isfinite(after).all() and (np.abs(after) <= 1).all()
        report = dict(step=step, checkpoint=str(checkpoint), devices=[str(d) for d in jax.devices()],
                      reload_action=after.tolist(), reload_equal=True, spec=spec,
                      demo_transitions=len(demos), online_transitions=len(online))
        (out / 'result.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        if server:
            print(f'Serving for {args.serve_seconds}s after training', flush=True)
            deadline = time.monotonic()+args.serve_seconds
            while time.monotonic() < deadline:
                time.sleep(.2)
    finally:
        if server:
            server.stop()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['train','status','submit','policy-check','actor','retry-delivery'])
    ap.add_argument('--data', default='recordings')
    ap.add_argument('--out', default='training_runs/human')
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--checkpoint-every', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--encoder', choices=['resnet','resnet-pretrained'], default='resnet-pretrained')
    ap.add_argument('--allow-provisional', action='store_true')
    ap.add_argument('--resume')
    ap.add_argument('--serve-seconds', type=int, default=0)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=5688)
    ap.add_argument('--file')
    ap.add_argument('--upload-host', help='SSH user@host for automatic episode upload')
    ap.add_argument('--upload-port', type=int, default=22)
    ap.add_argument('--upload-dir', help='Absolute cloud learner --data directory')
    ap.add_argument('--ssh-control-path', help='Optional existing SSH ControlMaster socket')
    ap.add_argument('--record-args', nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()
    if args.steps < 0 or args.batch_size < 1 or args.serve_seconds < 0 or args.checkpoint_every < 1:
        ap.error('Invalid counts')
    if args.mode == 'train':
        train(args)
        return
    delivery = None
    if args.mode in ('actor', 'retry-delivery'):
        if not args.upload_host or not args.upload_dir:
            ap.error('actor/retry-delivery requires --upload-host and --upload-dir')
        if args.mode == 'retry-delivery' and not args.file:
            ap.error('retry-delivery requires --file pointing to the local recording')
        from episode_delivery import SSHUploader, EpisodeDelivery
        from experiments.so101.config import TrainConfig
        uploader = SSHUploader(args.upload_host, args.upload_dir, args.upload_port, args.ssh_control_path)
        uploader.check()  # Before any hardware is opened.
    c = client(args)
    try:
        if args.mode in ('actor', 'retry-delivery'):
            delivery = EpisodeDelivery(TrainConfig(), uploader, lambda name: c.request('submit', {'file': name}))
        if args.mode == 'retry-delivery':
            status_path = Path(args.file).with_suffix('.delivery.json')
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            needs_review = status.get('status', 'pending_label') == 'pending_label'
            if not delivery.process(args.file, review=needs_review):
                raise RuntimeError('Pending delivery retained; see .delivery.json')
            return
        kind = 'policy' if args.mode in ('policy-check', 'actor') else args.mode
        result = c.request(kind, dict(file=args.file))
        if not result or not result.get('ok'):
            raise RuntimeError(result)
        if args.mode in ('policy-check', 'actor'):
            import jax
            import numpy as np
            from experiments.so101.config import TrainConfig
            args.encoder, args.seed = result['spec']['encoder'], result['spec']['seed']
            cfg = TrainConfig()
            for key, expected in dict(action_semantics=cfg.ACTION_SEMANTICS,
                                      observation_semantics=cfg.OBSERVATION_SEMANTICS,
                                      image_keys=cfg.image_keys, action_scale_rad=cfg.ACTION_SCALE,
                                      env_hz=cfg.ENV_HZ, pose_alignment=cfg.POSE_ALIGNMENT,
                                      roi_spec=cfg.roi_spec, main_roi_spec=cfg.main_roi_spec).items():
                if result['spec'].get(key) != expected:
                    raise ValueError(f'Policy configuration mismatch: {key}')
            agent = make_agent(args,cfg)
            agent = agent.replace(state=agent.state.replace(params=jax.device_put(result.pop('params'))))
            if args.file and args.mode == 'policy-check':
                from experiments.so101.human_reward import RecordingUnpickler
                with open(args.file, 'rb') as f:
                    obs = RecordingUnpickler(f).load()['transitions'][0]['observations']
            else:
                obs = {k: np.zeros(s.shape, s.dtype) for k,s in cfg.get_environment(fake_env=True).observation_space.spaces.items()}
            action = np.asarray(agent.sample_actions(obs, argmax=True))
            assert action.shape == (3,) and np.isfinite(action).all() and (np.abs(action)<=1).all()
            durations = []
            for _ in range(10):
                started = time.perf_counter()
                np.asarray(agent.sample_actions(obs, argmax=True))
                durations.append(time.perf_counter()-started)
            result['inference_ms_median'] = float(np.median(durations)*1000)
            result['inference_ms_max'] = float(max(durations)*1000)
            result['test_action'] = action.tolist()
            if args.mode == 'actor':
                if result['step'] < 1:
                    raise ValueError('Untrained policy cannot drive hardware')
                if result['spec'].get('allow_provisional'):
                    raise ValueError('Provisional smoke-test policy cannot drive hardware')
                # Download once before opening hardware; no network requests in control loop.
                import record_demo_so101
                sys.argv = ['record_demo_so101.py', *args.record_args]
                print('Policy loaded. SPACE toggles intervention. Each saved episode opens human review, then uploads.', flush=True)
                def policy(obs):
                    return np.asarray(agent.sample_actions(obs, argmax=True))
                policy.metadata = dict(step=result['step'], spec=result['spec'])
                rc = record_demo_so101.main(policy=policy, on_episode_saved=delivery.process)
                if rc:
                    raise RuntimeError(f'Actor stopped (exit {rc}); raw files and delivery status retained')
        print(json.dumps(result), flush=True)
    finally:
        c.stop()

if __name__ == '__main__':
    main()
