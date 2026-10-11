"""Build a separate migration dataset from trusted human-reviewed recordings.
Native 10Hz actions are rescaled exactly. Legacy sources retain only a contiguous,
quality-screened suffix ending at the human success frame (approximate resampling).
"""
import argparse,copy,hashlib,json,pickle,sys
from pathlib import Path
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'examples'),str(Path(__file__).resolve().parents[1]/'serl_launcher')]
from experiments.so101.config import TrainConfig

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def equal(a,b):return set(a)==set(b) and all(np.array_equal(a[k],b[k]) for k in a)

def ranges(ts,scale):
    """Return eligible windows without stitching across any rejected source step."""
    result=[];i=0
    while i<len(ts):
        start=ts[i]['infos']['joint_control']['step_started_monotonic_s'];best=None
        for j in range(i,min(i+8,len(ts))):
            dt=ts[j]['infos']['joint_control']['step_finished_monotonic_s']-start
            if .08<=dt<=.12 and (best is None or abs(dt-.1)<best[0]):best=(abs(dt-.1),j,dt)
            if dt>.12:break
        if best is None:i+=1;continue
        _,j,dt=best;w=ts[i:j+1];a=np.asarray([t['actions'] for t in w]);total=a.sum(0)
        valid=all(t['infos']['joint_control'].get('accepted') and not t['infos']['joint_control'].get('limited') for t in w)
        valid &= len({bool(t['infos'].get('intervening')) for t in w})==1
        # Keep the original conservative 4 degree audit boundary, despite new 5.5 scale.
        valid &= bool(np.all(abs(total)<=1+1e-6))
        valid &= bool(np.all(np.minimum(np.maximum(a,0).sum(0),-np.minimum(a,0).sum(0))<=.05))
        net=np.array(w[-1]['infos']['joint_control']['target_rad'])-np.array(w[0]['infos']['joint_control']['command_before_rad'])
        valid &= np.allclose(net,total*scale,atol=1e-6)
        for t,field in [(w[0],'observation_quality'),(w[-1],'next_observation_quality')]:
            q=t['infos'].get(field,{}).get('cameras',{}).get('cam_main',{});age=q.get('age_s')
            valid &= bool(q.get('valid') and age is not None and 0<=age<=.15)
        valid &= all(equal(x['next_observations'],y['observations']) for x,y in zip(w,w[1:]))
        if valid:result.append((i,j+1))
        i=j+1
    # Only the final continuous success suffix. No artificial failed labels or gaps.
    if not result or result[-1][1]!=len(ts):return []
    suffix=[result[-1]]
    for item in reversed(result[:-1]):
        if item[1]!=suffix[0][0]:break
        suffix.insert(0,item)
    return suffix

def convert(path,out,index,legacy=False):
    cfg=TrainConfig();labels=json.loads(path.with_suffix('.labels.json').read_text());sha=digest(path)
    if labels['recording_sha256']!=sha:raise ValueError(f'Label hash mismatch: {path}')
    lab=labels['episodes']['0']
    if lab.get('source')!='human_review' or not lab.get('reviewed') or lab.get('provisional') or lab.get('status') not in ('success','failed'):
        raise ValueError(f'Human reviewed label required: {path}')
    with path.open('rb') as f:raw=pickle.load(f)
    meta=raw['meta']
    if meta.get('operator_excluded') or meta.get('safety_error') or meta.get('recording',{}).get('has_gaps'):return dict(source=str(path),excluded=True)
    if meta['env_hz']!=(30 if legacy else 10):raise ValueError('Unexpected source frequency')
    end=lab['frame'] if lab['status']=='success' else len(raw['transitions'])
    if isinstance(end,bool) or not isinstance(end,int) or not 1<=end<=len(raw['transitions']):raise ValueError('Invalid label frame')
    ts=raw['transitions'][:end];scale=meta['action_scale_rad']
    spans=ranges(ts,scale) if legacy else [(i,i+1) for i in range(end)]
    if not spans:return dict(source=str(path),excluded=True,reason='No valid terminal suffix')
    converted=[]
    for start,stop in spans:
        w=ts[start:stop]
        if converted and not equal(converted[-1]['next_observations'],w[0]['observations']):raise ValueError('Fragment boundary observation mismatch')
        action=(np.sum([t['actions'] for t in w],axis=0)*scale/cfg.ACTION_SCALE).astype(np.float32)
        if np.any(abs(action)>1+1e-6):raise ValueError('Do not clip transformed actions')
        first=w[0]['infos'];last=w[-1]['infos'];jc0=first['joint_control'];jc1=last['joint_control']
        info=dict(intervening=bool(first.get('intervening')),quality_ok=all(t['infos'].get('quality_ok',True) for t in w),
            observation_quality=first.get('observation_quality'),next_observation_quality=last.get('next_observation_quality'),
            migration_source_range=[start,stop],migration_duration_s=jc1['step_finished_monotonic_s']-jc0['step_started_monotonic_s'])
        # Preserve physical requested displacement exactly, without mislabeling old diagnostics.
        info['migration_requested_delta_rad']=(action*cfg.ACTION_SCALE).tolist()
        converted.append(dict(observations=w[0]['observations'],next_observations=w[-1]['next_observations'],actions=action,infos=info))
    conversion=dict(version='10hz_scale55_v1',kind='approximate_30hz_success_suffix' if legacy else 'native10_scale_only',
        source=str(path.resolve()),source_sha256=sha,source_label=lab,source_hz=meta['env_hz'],source_scale_rad=scale,
        source_ranges=[list(s) for s in spans],source_end=end)
    newmeta=copy.deepcopy(meta);newmeta.update(env_hz=10.,action_scale_rad=cfg.ACTION_SCALE,n_steps=len(converted),
        conversion=conversion,n_intervened=sum(t['infos']['intervening'] for t in converted))
    # Preserve source-only aggregate metadata under one explicit key; no false converted timing.
    for k in ('record_sec','hz_actual','n_control_intervened','intervened_frac','return','policy','control_key_timings','capture_policy'):
        if k in newmeta:newmeta.setdefault('source_capture',{})[k]=newmeta.pop(k)
    newmeta['recording']=dict(has_gaps=False,conversion=True)
    newlab=dict(status=lab['status'],frame=len(converted) if lab['status']=='success' else None,
        reviewed=True,provisional=False,source='derived_human_review',source_recording_sha256=sha)
    for i,t in enumerate(converted):
        terminal=i==len(converted)-1;success=terminal and lab['status']=='success'
        t.update(rewards=np.float32(success),masks=np.float32(not success),dones=terminal)
    target=out/f'ep_{index:04d}.pkl'
    with target.open('xb') as f:pickle.dump(dict(meta=newmeta,transitions=converted),f,protocol=pickle.HIGHEST_PROTOCOL)
    target.with_suffix('.labels.json').write_text(json.dumps(dict(recording=str(target),recording_sha256=digest(target),episodes={'0':newlab}),indent=2))
    return dict(source=str(path),file=target.name,steps=len(converted),source_range=[spans[0][0],spans[-1][1]],kind=conversion['kind'],excluded=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--legacy',required=True);ap.add_argument('sources',nargs='+');args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);rows=[];index=0
    files=[(p,False) for d in args.sources for p in sorted(Path(d).glob('ep_*.pkl'))]
    legacy=Path(args.legacy);m=json.loads((legacy/'manifest.json').read_text())
    for e in m['episodes']:
        p=legacy/e['retained_file']
        if digest(p)!=e['sha256']:raise ValueError('Archived source changed')
        files.append((p,True))
    for p,old in files:
        row=convert(p,out,index,old);rows.append(row)
        if not row['excluded']:index+=1
    manifest=dict(target_hz=10,action_scale_deg=5.5,source_count=len(rows),episodes=rows)
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print('Prepared',index,'episodes/fragments;',sum(x.get('steps',0) for x in rows),'transitions')
if __name__=='__main__':main()
