"""Isolated GPU learner + CPU policy lifecycle test using reviewed data, never hardware.
Creates an independent inbox/run; does not submit test episodes to the production service.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--port', type=int, default=5788)
    args=ap.parse_args()
    root=Path(args.out).resolve();root.mkdir(parents=True, exist_ok=False)
    inbox=root/'inbox';inbox.mkdir()
    seed=root/'seed';seed.mkdir()
    original=Path(args.checkpoint)
    receipts=json.loads((original.parent/'receipts.json').read_text())
    demos=[r for r in receipts if r['role']=='demo']
    online=next(r for r in receipts if r['role']=='online')
    for r in demos:
        for suffix in ('.pkl','.labels.json'):
            source=(Path(args.data)/r['file']).with_suffix(suffix)
            shutil.copyfile(source,inbox/source.name)
    checkpoint=seed/original.name
    os.link(original,checkpoint)
    shutil.copyfile(Path(str(original)+'.json'),Path(str(checkpoint)+'.json'))
    (seed/'receipts.json').write_text(json.dumps(demos))
    # Actor-side validation uses CPU; the subprocess learner explicitly uses GPU.
    os.environ['JAX_PLATFORMS']='cpu'
    from train_human import client, make_agent, validated_policy_agent
    from online_training import EpisodePolicy
    from experiments.so101.config import TrainConfig
    import numpy as np
    py=sys.executable
    run=root/'run'
    base=[py,'-u',str(Path(__file__).with_name('train_human.py')),'train',
          '--data',str(inbox),'--out',str(run),'--continuous','--steps','0',
          '--updates-per-transition','.02','--max-runtime-seconds','1200',
          '--port',str(args.port),'--checkpoint-every','2']
    processes=[]
    client_obj=None
    deadline=time.monotonic()+1100
    def start(cp, legacy=False):
        env=dict(os.environ,JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false')
        log=(root/f'learner_{len(processes)}.log').open('w')
        p=subprocess.Popen(base+['--resume',str(cp)]+(['--migrate-replay-semantics'] if legacy else []),
                           stdout=log,stderr=subprocess.STDOUT,env=env)
        log.close();processes.append(p)
        while time.monotonic()<deadline:
            if p.poll() is not None:raise RuntimeError(f'Learner exited {p.returncode}; inspect logs in {root}')
            try:
                with socket.create_connection(('127.0.0.1',args.port),timeout=.2):break
            except OSError:time.sleep(.5)
        else:raise TimeoutError('Learner never opened RPC')
        return client(SimpleNamespace(host='127.0.0.1',port=args.port))
    def wait_status(predicate):
        while time.monotonic()<deadline:
            status=client_obj.request('status',{})
            if status and status.get('ok') and predicate(status):return status
            if processes[-1].poll() is not None:raise RuntimeError('Learner exited during test')
            time.sleep(.5)
        raise TimeoutError('Learner state did not converge')
    evidence={}
    try:
        client_obj=start(checkpoint,True)
        initial=wait_status(lambda s:s['phase']=='waiting_for_labels')
        evidence['initial']=initial
        first=client_obj.request('policy',{})
        cfg=TrainConfig()
        agent=make_agent(SimpleNamespace(seed=first['spec']['seed'],encoder=first['spec']['encoder']),cfg)
        obs={k:np.zeros(s.shape,s.dtype) for k,s in cfg.get_environment(fake_env=True).observation_space.spaces.items()}
        agent=validated_policy_agent(agent,first,obs)
        def apply(candidate):
            nonlocal agent
            agent=validated_policy_agent(agent,candidate,obs)
        policy=EpisodePolicy(first,lambda o:np.asarray(agent.sample_actions(o,argmax=True)),
                             lambda meta:client_obj.request('policy',meta),apply)
        action_before=policy(obs).tolist()
        for suffix in ('.pkl','.labels.json'):
            source=(Path(args.data)/online['file']).with_suffix(suffix)
            shutil.copyfile(source,inbox/source.name)
        reply=client_obj.request('submit',{'file':online['file']})
        assert reply['ok'] and not reply['duplicate'],reply
        evidence['submit']=reply
        target=initial['step']+int(online['transitions']*.02)
        final=wait_status(lambda s:s['step']==target and s['phase']=='waiting_for_labels')
        assert final['intervention_transitions']>0
        assert final['demo_transitions']==initial['demo_transitions']+final['intervention_transitions']
        assert policy.metadata['step']==initial['step']  # no mid-episode update
        assert policy.before_episode()
        assert policy.metadata['step']==target
        evidence['action_before']=action_before
        evidence['action_after']=policy(obs).tolist()
        assert not policy.before_episode()
        duplicate=client_obj.request('submit',{'file':online['file']})
        assert duplicate['ok'] and duplicate['duplicate']
        assert client_obj.request('status',{})['updates_remaining']==0
        evidence['duplicate']=duplicate;evidence['after_training']=final
        client_obj.stop();client_obj=None
        processes[-1].terminate();processes[-1].wait(timeout=120)
        assert processes[-1].returncode==0
        latest=json.loads((run/'latest.json').read_text())['checkpoint']
        client_obj=start(latest)
        resumed=wait_status(lambda s:s['phase']=='waiting_for_labels')
        assert resumed['step']==target and resumed['updates_remaining']==0
        assert resumed['intervention_transitions']==final['intervention_transitions']
        evidence['resumed']=resumed
        evidence['ok']=True
        (root/'evidence.json').write_text(json.dumps(evidence,indent=2))
        print(json.dumps(dict(ok=True,initial_step=initial['step'],final_step=target,
                             intervention_transitions=final['intervention_transitions'],evidence=str(root/'evidence.json'))),flush=True)
    finally:
        if client_obj:client_obj.stop()
        for p in processes:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=120)
                except subprocess.TimeoutExpired:p.kill();p.wait()

if __name__=='__main__':main()
