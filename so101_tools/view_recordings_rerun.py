#!/usr/bin/env python3
"""Export SO101 v3 raw recordings to lossless, offline Rerun files."""
import argparse
import json
from pathlib import Path
import pickle
import sys

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

JOINTS = ('pan', 'lift', 'elbow', 'wrist_flex')
CAMERAS = ('cam_main', 'cam_main_roi', 'cam_wrist')


def blueprint(joint_names=JOINTS):
    images = rrb.Grid(*[
        rrb.Spatial2DView(origin=f'/images/{phase}/{cam}', name=f'{phase}: {cam}')
        for phase in ('before', 'after') for cam in CAMERAS
    ], grid_columns=3)
    joints = rrb.Grid(*[
        rrb.TimeSeriesView(origin=f'/joints/{joint}', name=f'{joint}: command / geometric goal / measured (deg)')
        for joint in joint_names
    ], grid_columns=2)
    plots = rrb.Horizontal(
        rrb.TimeSeriesView(origin='/actions', name='Requested joint increments (deg/step)'),
        rrb.TimeSeriesView(origin='/quality/age_ms', name='Image age (ms; limit 250)'),
        rrb.TimeSeriesView(origin='/quality/flags', name='Quality flags'),
    )
    return rrb.Blueprint(
        rrb.Vertical(images, rrb.Horizontal(joints, rrb.TextDocumentView(origin='/review', name='Step details'),
                                          column_shares=[3,1]), plots, row_shares=[3,3,2]),
        rrb.TimePanel(timeline='elapsed_s', duration_cursor=0),
        rrb.SelectionPanel(expanded=False), auto_layout=False, auto_views=False,
    )


def png_image(frame):
    image = np.asarray(frame)
    if image.shape == (1,224,224,3): image = image[0]
    if image.shape != (224,224,3) or image.dtype != np.uint8:
        raise ValueError(f'Unexpected RGB image: {image.shape} {image.dtype}')
    ok, encoded = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok: raise RuntimeError('PNG encoding failed')
    return rr.EncodedImage(contents=encoded.tobytes(), media_type="image/png")


def scalar(path, value):
    rr.log(path, rr.Scalars(float(value)))


def export_recording(source, output):
    with source.open('rb') as stream: data = pickle.load(stream)
    ts, meta = data['transitions'], data['meta']
    versions = {
        ('relative_joint_target_request_v3', 'joint_command_state_v3'): 4,
        ('relative_joint_target_request_v4_3d', 'joint_command_state_v4_3d'): 3,
    }
    dimensions = versions.get((meta.get('action_semantics'), meta.get('observation_semantics')))
    if dimensions is None:
        raise ValueError('Unsupported action/observation version pair')
    joints = JOINTS[:dimensions]
    state_dim = 12 + 2 * dimensions
    if not ts: raise ValueError('Empty recording')
    rr.init('SO101 recording review', recording_id=f'{source.parent.name}_{source.stem}', strict=True)
    output.mkdir(parents=True, exist_ok=True)
    target = output / f'{source.parent.name}_{source.stem}.rrd'
    rr.save(target, default_blueprint=blueprint(joints))
    for joint in joints:
        for name,color in [('command_deg',[255,170,60]),('measured_deg',[80,180,255]),
                           ('geometric_goal_deg',[100,220,130])]:
            rr.log(f'/joints/{joint}/{name}',rr.SeriesLines(names=[name],colors=[color]),static=True)
    origin = ts[0]['infos']['joint_control']['step_started_monotonic_s']
    bad_steps = []
    for i,t in enumerate(ts):
        info=t['infos']; control=info['joint_control']
        elapsed=control['step_started_monotonic_s']-origin
        rr.set_time('step',sequence=i)
        rr.set_time('elapsed_s',duration=elapsed)
        before=np.asarray(t['observations']['state']).reshape(-1)
        after=np.asarray(t['next_observations']['state']).reshape(-1)
        actions=np.asarray(t['actions'])
        if before.shape!=(state_dim,) or after.shape!=(state_dim,) or actions.shape!=(dimensions,):
            raise ValueError(f'Invalid state/action dimensions at step {i}')
        if not np.all(np.isfinite(np.r_[before,after,actions])):
            raise ValueError(f'Nonfinite state/action at step {i}')
        for phase,key in [('before','observations'),('after','next_observations')]:
            for cam in CAMERAS:
                rr.log(f'/images/{phase}/{cam}',png_image(t[key][cam]))
        command=np.asarray(control['target_rad'])
        applied=np.asarray(control['applied_delta_rad'])
        requested=actions*meta['action_scale_rad']
        for j,joint in enumerate(joints):
            scalar(f'/actions/{joint}',np.degrees(requested[j]))
            scalar(f'/execution/applied_delta_deg/{joint}',np.degrees(applied[j]))
            scalar(f'/joints/{joint}/command_deg',np.degrees(command[j]))
            scalar(f'/joints/{joint}/geometric_goal_deg',np.degrees(command[j]-after[dimensions+j]))
            scalar(f'/joints/{joint}/measured_deg',np.degrees(after[2*dimensions+j]))
            scalar(f'/diagnostics/tracking_error_deg/{joint}',np.degrees(command[j]-after[dimensions+j]-after[2*dimensions+j]))
        quality=info['next_observation_quality']
        scalar('/quality/age_ms/limit',250)
        for cam in ('cam_main','cam_wrist'):
            stats=quality['cameras'][cam]
            if stats.get('age_s') is not None: scalar(f'/quality/age_ms/{cam}',stats['age_s']*1000)
            scalar(f'/quality/repeat_streak/{cam}',info['camera_repeat_streak'].get(cam,0))
            scalar(f'/quality/frame_arrival_elapsed_s/{cam}',stats['arrival']-origin)
            scalar(f'/quality/processing_delay_ms/{cam}',stats.get('processing_delay_s',0)*1000)
        scalar('/quality/skew_ms',quality['camera_skew_s']*1000)
        scalar('/quality/flags/bad_step',not info['quality_ok'])
        scalar('/quality/flags/skew_warning',bool(info.get('quality_warnings')))
        scalar('/quality/flags/action_limited',control.get('limited',False))
        scalar('/timing/step_duration_ms',(control['step_finished_monotonic_s']-control['step_started_monotonic_s'])*1000)
        if not info['quality_ok']: bad_steps.append(i)
        detail={
            'source':str(source), 'step_zero_based':i, 'elapsed_s':elapsed,
            'quality_ok':info['quality_ok'], 'quality_problems':info['quality_problems'],
            'quality_warnings':info.get('quality_warnings',[]),
            'requested_action_normalized':actions.tolist(),
            'requested_delta_deg':np.degrees(requested).tolist(),
            'applied_delta_deg':np.degrees(applied).tolist(),
            'measured_before_deg':np.degrees(before[2*dimensions:3*dimensions]).tolist(),
            'measured_after_deg':np.degrees(after[2*dimensions:3*dimensions]).tolist(),
            'joint_reasons':control.get('joint_reasons'),
            'camera_source_stamp_ns':{c:quality['cameras'][c]['source_stamp_ns'] for c in ('cam_main','cam_wrist')},
            'timing_note':'Images are cached before/after this action, not synchronized exposure. Joint plots use after-state. step and elapsed_s are control timelines.',
        }
        rr.log('/review',rr.TextDocument(json.dumps(detail,ensure_ascii=False,indent=2)))
    rr.get_global_data_recording().flush()
    return dict(source=str(source),rrd=str(target),steps=len(ts),image_events=len(ts)*6,
                bad_steps=bad_steps,elapsed_s=elapsed,images='lossless PNG',state_layout=f'error[0:{dimensions}],offset[{dimensions}:{2*dimensions}],position[{2*dimensions}:{2*dimensions+6}],velocity[{2*dimensions+6}:{state_dim}]')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='+',type=Path,help='Session directories or ep_*.pkl files')
    parser.add_argument('--out',type=Path,default=Path('validation/rerun'))
    args=parser.parse_args()
    files=[]
    for path in args.inputs:
        files.extend(sorted(path.glob('ep_*.pkl')) if path.is_dir() else [path])
    reports=[]
    for source in files:
        result=export_recording(source,args.out)
        reports.append(result)
        print(f"Exported {result['steps']} steps: {result['rrd']}",flush=True)
    (args.out/'manifest.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2)+'\n')
    return 0


if __name__=='__main__':sys.exit(main())
