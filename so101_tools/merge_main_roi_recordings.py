"""Lossless observation projection for trusted local recordings; sources untouched."""
import argparse, hashlib, json, pickle
from pathlib import Path
import numpy as np

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('sources',nargs='+');a=ap.parse_args()
    out=Path(a.out);out.mkdir(exist_ok=False)
    manifest={'image_keys':['cam_main','cam_main_roi'],'sources':a.sources,'episodes':[],
              'conversion':'remove_wrist_observation_only_v1','quality_note':'Historical quality diagnostics retained unchanged; may refer to wrist.'}
    for source in a.sources:
        for path in sorted(Path(source).glob('ep_*.pkl')):
            old_hash=digest(path);labelpath=path.with_suffix('.labels.json')
            labels=json.loads(labelpath.read_text())
            if labels.get('recording_sha256')!=old_hash:raise ValueError(f'Stale labels: {path}')
            with path.open('rb') as f:raw=pickle.load(f)
            index=len(manifest['episodes']);dest=out/f'ep_{index:04d}.pkl'
            for t in raw['transitions']:
                for field in ('observations','next_observations'):
                    obs=t[field]
                    if set(obs)!={'state','cam_main','cam_main_roi','cam_wrist'}:raise ValueError(f'Unexpected keys: {path}')
                    del obs['cam_wrist']
            raw['meta'].update(index=index,image_keys=manifest['image_keys'],
                source_recording=str(path.resolve()),source_recording_sha256=old_hash,
                conversion=manifest['conversion'])
            with dest.open('xb') as f:pickle.dump(raw,f,protocol=pickle.HIGHEST_PROTOCOL)
            # Re-read and verify all retained pixels, states, actions and step order.
            with dest.open('rb') as f:check=pickle.load(f)
            assert len(check['transitions'])==len(raw['transitions'])
            for x,y in zip(raw['transitions'],check['transitions']):
                for field in ('observations','next_observations'):
                    assert set(y[field])=={'state','cam_main','cam_main_roi'}
                    for key in x[field]:np.testing.assert_array_equal(x[field][key],y[field][key])
                np.testing.assert_array_equal(x['actions'],y['actions'])
            # Confirm user did not save new labels during conversion.
            if json.loads(labelpath.read_text())!=labels:raise RuntimeError(f'Labels changed during conversion: {path}')
            labels.update(recording=str(dest.resolve()),recording_sha256=digest(dest),
                          migration_source=str(labelpath.resolve()),source_recording_sha256=old_hash)
            dest.with_suffix('.labels.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2))
            manifest['episodes'].append(dict(file=dest.name,source=str(path),n_steps=len(raw['transitions']),label=labels['episodes']['0']))
            (out/'meta.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
            print(dest.name,len(raw['transitions']),flush=True)
    print('Completed',len(manifest['episodes']),flush=True)
if __name__=='__main__':main()
