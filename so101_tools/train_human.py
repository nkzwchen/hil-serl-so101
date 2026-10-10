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


def validated_policy_agent(agent, candidate, obs):
    """Validate and warm up a candidate before the caller atomically swaps agents."""
    import jax
    import numpy as np
    candidate_agent = agent.replace(state=agent.state.replace(params=jax.device_put(candidate['params'])))
    if jax.tree_util.tree_structure(candidate_agent.state.params) != jax.tree_util.tree_structure(agent.state.params):
        raise ValueError('Policy parameter structure mismatch')
    for new, old in zip(jax.tree_util.tree_leaves(candidate_agent.state.params), jax.tree_util.tree_leaves(agent.state.params)):
        if new.shape != old.shape or not np.isfinite(np.asarray(new)).all():
            raise ValueError('Invalid policy parameters')
    probe = np.asarray(candidate_agent.sample_actions(obs, argmax=True))
    if probe.shape != (3,) or not np.isfinite(probe).all() or np.any(np.abs(probe)>1):
        raise ValueError('Invalid refreshed policy action')
    return candidate_agent


def stop_server(server):
    # Agentlace.stop closes REP from the caller thread while recv is active.
    # Let its receive timeout exit the worker before closing ZeroMQ resources.
    server.req_rep_server.is_kill = True
    thread = getattr(server, 'thread', None)
    if thread is not None:
        thread.join()
    server.stop()
    server.broadcast_server.socket.close(linger=0)
    server.broadcast_server.context.term()


def train(args):
    import fcntl
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.learner.lock').open('a') as run_lock:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train_locked(args, out)


def _train_locked(args, out):
    import jax
    import numpy as np
    import signal
    from flax import serialization
    from flax.core import freeze
    from agentlace.trainer import TrainerServer
    from experiments.so101.config import TrainConfig
    from experiments.so101.human_reward import load_episode
    from online_training import HumanReplay, UpdateBudget, REPLAY_SEMANTICS
    cfg = TrainConfig()
    inbox = Path(args.data).resolve()
    replay = HumanReplay(cfg.replay_buffer_capacity)
    previous = {}
    saved = {}
    if args.resume:
        manifest = Path(args.resume).parent / 'receipts.json'
        previous = {r['recording_sha256']: r for r in json.loads(manifest.read_text())}
        saved = json.loads(Path(args.resume + '.json').read_text())
    elif (out / 'receipts.json').exists():
        raise ValueError('Run already exists; use --resume or a new --out')
    for path in sorted(inbox.glob('*.pkl')):
        ts, receipt = load_episode(path, cfg, args.allow_provisional)
        prior = previous.get(receipt['recording_sha256'])
        if prior and prior['id'] != receipt['id']:
            raise ValueError('Labels changed since checkpoint; start a new run')
        role = prior.get('role', 'demo') if prior else ('online' if args.resume else 'demo')
        replay.add(ts, receipt, role)
    if previous and not set(previous).issubset({r['recording_sha256'] for r in replay.receipts.values()}):
        raise ValueError('Resume is missing previously ingested recordings')
    if not replay.demos:
        raise ValueError('No labeled demonstrations')
    print(f'Loaded {len(replay.receipts)} episodes / {len(replay.demos)} transitions; devices={jax.devices()}', flush=True)
    agent = make_agent(args, cfg)
    spec = dict(encoder=args.encoder, seed=args.seed, action_semantics=cfg.ACTION_SEMANTICS,
                observation_semantics=cfg.OBSERVATION_SEMANTICS, image_keys=cfg.image_keys,
                reward_source=cfg.REWARD_SOURCE, allow_provisional=args.allow_provisional,
                action_scale_rad=cfg.ACTION_SCALE, env_hz=cfg.ENV_HZ,
                pose_alignment=cfg.POSE_ALIGNMENT, roi_spec=cfg.roi_spec, main_roi_spec=cfg.main_roi_spec)
    continuous = getattr(args, 'continuous', False)
    batch_size = getattr(args, 'batch_size', None) or cfg.batch_size
    cta_ratio = getattr(args, 'cta_ratio', None) or cfg.cta_ratio
    max_runtime = getattr(args, 'max_runtime_seconds', 3600)
    learning = dict(batch_size=batch_size, cta_ratio=cta_ratio, discount=cfg.discount,
                    replay_semantics=REPLAY_SEMANTICS, capacity=cfg.replay_buffer_capacity)
    step = 0
    if args.resume:
        if saved.get('learning') and saved['learning'] != learning:
            raise ValueError('Resume learning configuration differs; keep batch size/CTA/discount/replay semantics unchanged')
        if saved['spec'] != spec:
            raise ValueError('Checkpoint configuration differs')
        if saved.get('learning', {}).get('replay_semantics') != REPLAY_SEMANTICS and not getattr(args, 'migrate_replay_semantics', False):
            raise ValueError('Legacy checkpoint: explicitly pass --migrate-replay-semantics to rebuild cutoff masks/intervention replay')
        agent = agent.replace(state=serialization.from_bytes(agent.state, Path(args.resume).read_bytes()))
        step = saved['step']
    budget = UpdateBudget(step, args.steps, getattr(args, 'updates_per_transition', .25))
    if continuous and saved.get('budget'):
        budget = UpdateBudget(**saved['budget'])
        if budget.updates_per_transition != args.updates_per_transition:
            raise ValueError('Resume must keep updates-per-transition; stored budget is authoritative')
    run_id = saved.get('run_id', str(out))
    rng = np.random.default_rng(args.seed)
    if saved.get('sampler_state'):
        rng.bit_generator.state = saved['sampler_state']
    lock = threading.RLock()
    stop = threading.Event()
    snapshot = dict(params=jax.device_get(agent.state.params), spec=spec, step=step, run_id=run_id)
    phase = 'starting'
    def atomic_json(path, data):
        tmp = path.with_suffix(path.suffix+'.tmp')
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)
    def persist_receipts():
        atomic_json(out / 'receipts.json', list(replay.receipts.values()))
    def status():
        return dict(ok=True, step=snapshot['step'], run_id=run_id, phase=phase,
                    demo_transitions=len(replay.demos), online_transitions=len(replay.online),
                    online_received=replay.online_received, intervention_transitions=replay.interventions_received,
                    episodes=len(replay.receipts), learning=learning, continuous=continuous,
                    updates_remaining=budget.remaining(step, replay.online_received) if continuous else None,
                    spec=spec)
    def callback(kind, payload):
        try:
            if kind == 'status':
                with lock:return status()
            if kind == 'policy':
                with lock:
                    if payload.get('run_id') == run_id and payload.get('step') == snapshot['step']:
                        return dict(ok=True, unchanged=True, step=snapshot['step'], run_id=run_id)
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
                    duplicate = not replay.add(ts, receipt, 'online')
                    # Retry durability even for duplicates if a prior write failed.
                    persist_receipts()
                    return dict(ok=True, duplicate=duplicate, transitions=len(ts),
                                intervention_transitions=len(receipt.get('intervention_indices', [])),
                                step=snapshot['step'], phase=phase)
            raise ValueError('Unknown request')
        except Exception as exc:
            return dict(ok=False, error=str(exc))
    server = TrainerServer(wire_config(args.port), request_callback=callback) if (continuous or args.serve_seconds) else None
    def save_checkpoint():
        checkpoint = out / f'checkpoint_{step}.msgpack'
        tmp = checkpoint.with_suffix('.tmp')
        tmp.write_bytes(serialization.to_bytes(agent.state))
        tmp.replace(checkpoint)
        with lock:
            persist_receipts()
            atomic_json(Path(str(checkpoint)+'.json'), dict(spec=spec, step=step, run_id=run_id,
                        learning=learning, budget=budget.state() if continuous else None,
                        sampler_state=rng.bit_generator.state, migrated_from=args.resume))
        atomic_json(out / 'latest.json', dict(checkpoint=str(checkpoint), step=step))
        return checkpoint
    old_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, lambda *_: stop.set())
    started = time.monotonic()
    initial_step = step
    last_saved = None
    try:
        persist_receipts()
        atomic_json(out / 'learning_config.json', dict(**learning, continuous=continuous,
                    budget=budget.state(), max_runtime_seconds=max_runtime, run_id=run_id))
        if server:server.start(threaded=True)
        while not stop.is_set():
            if continuous and time.monotonic()-started >= max_runtime:break
            if not continuous and step-initial_step >= args.steps:break
            with lock:
                credit = budget.remaining(step, replay.online_received)
            if continuous and credit == 0:
                phase = 'waiting_for_labels'
                if last_saved != step:
                    save_checkpoint(); last_saved = step
                    print(f'Waiting for reviewed data at step {step}; update budget exhausted', flush=True)
                stop.wait(.2)
                continue
            phase = 'training'
            for critic_index in range(cta_ratio):
                with lock:
                    picked = replay.sample(rng, batch_size)
                batch = freeze(jax.tree_util.tree_map(lambda *xs: np.stack(xs), *picked))
                networks = frozenset({'critic'}) if critic_index < cta_ratio-1 else frozenset({'critic', 'actor', 'temperature'})
                agent, metrics = agent.update(batch, networks_to_update=networks)
                metrics = jax.device_get(metrics)
                if not all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(metrics)):
                    raise RuntimeError('Nonfinite training metrics')
            step += 1
            record = dict(step=step, metrics=jax.tree_util.tree_map(lambda x: np.asarray(x).tolist(), serialization.to_state_dict(metrics)))
            record['metrics']['schedule'] = dict(online_received=replay.online_received,
                    intervention_received=replay.interventions_received, cta_ratio=cta_ratio,
                    updates_remaining=budget.remaining(step, replay.online_received) if continuous else 0)
            with (out / 'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(f'update {step}: metrics finite', flush=True)
            # Publish a coherent snapshot only after the entire update cycle finishes.
            parameters = jax.device_get(agent.state.params)
            with lock:snapshot.update(params=parameters, step=step)
            if step % args.checkpoint_every == 0:
                save_checkpoint(); last_saved = step
        phase = 'saving'
        checkpoint = save_checkpoint()
        restored = agent.replace(state=serialization.from_bytes(agent.state, checkpoint.read_bytes()))
        obs = replay.demos[0]['observations']
        before = np.asarray(agent.sample_actions(obs, argmax=True))
        after = np.asarray(restored.sample_actions(obs, argmax=True))
        np.testing.assert_allclose(before, after, rtol=0, atol=0)
        assert after.shape == (3,) and np.isfinite(after).all() and (np.abs(after) <= 1).all()
        report = dict(step=step, checkpoint=str(checkpoint), devices=[str(d) for d in jax.devices()],
                      reload_action=after.tolist(), reload_equal=True, spec=spec, learning=learning,
                      demo_transitions=len(replay.demos), online_transitions=len(replay.online))
        atomic_json(out / 'result.json', report)
        print(json.dumps(report), flush=True)
        if server and not continuous and not stop.is_set():
            phase = 'serving_only'
            print(f'Serving for {args.serve_seconds}s after training (no updates)', flush=True)
            stop.wait(args.serve_seconds)
    finally:
        if server:stop_server(server)
        for sig, handler in old_handlers.items():signal.signal(sig, handler)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['train','status','submit','policy-check','actor','retry-delivery'])
    ap.add_argument('--data', default='recordings')
    ap.add_argument('--out', default='training_runs/human')
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--batch-size', type=int, help='Default: TrainConfig.batch_size')
    ap.add_argument('--cta-ratio', type=int, help='Default: TrainConfig.cta_ratio')
    ap.add_argument('--continuous', action='store_true', help='Train on reviewed-data credits; wait when credits are exhausted')
    ap.add_argument('--updates-per-transition', type=float, default=.25)
    ap.add_argument('--max-runtime-seconds', type=float, default=3600)
    ap.add_argument('--migrate-replay-semantics', action='store_true')
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
    if args.steps < 0 or (args.batch_size is not None and args.batch_size < 2) or (args.cta_ratio is not None and args.cta_ratio < 1) or args.serve_seconds < 0 or args.checkpoint_every < 1:
        ap.error('Invalid counts')
    import math
    if not math.isfinite(args.updates_per_transition) or args.updates_per_transition < 0 or not math.isfinite(args.max_runtime_seconds) or args.max_runtime_seconds <= 0:
        ap.error('Invalid continuous training budget/runtime')
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
                # Initial download precedes hardware; refresh only at RECORD boundaries.
                import record_demo_so101
                sys.argv = ['record_demo_so101.py', *args.record_args]
                print('Policy loaded. SPACE toggles intervention. Each saved episode opens human review, then uploads.', flush=True)
                from online_training import EpisodePolicy
                def apply_policy(candidate):
                    nonlocal agent
                    agent = validated_policy_agent(agent, candidate, obs)
                policy = EpisodePolicy(result,
                    lambda obs: np.asarray(agent.sample_actions(obs, argmax=True)),
                    lambda meta: c.request('policy', {k:meta.get(k) for k in ('step','run_id')}),
                    apply_policy)
                rc = record_demo_so101.main(policy=policy, on_episode_saved=delivery.process)
                if rc:
                    raise RuntimeError(f'Actor stopped (exit {rc}); raw files and delivery status retained')
                result = dict(ok=True, initial_policy_step=result['step'],
                              final_policy_step=policy.metadata['step'], run_id=policy.metadata.get('run_id'))
        print(json.dumps(result), flush=True)
    finally:
        c.stop()

if __name__ == '__main__':
    main()
