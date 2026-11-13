import ast
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import pickle
import numpy as np
import pytest
import record_fullframe_replay as r


def source():
    return dict(meta={},transitions=[dict(infos=dict(joint_control=dict(accepted=True,target_rad=[.1,.2,.3],command_before_rad=[0,0,0])))])

def test_discontinuous_replay_rejected():
    raw=source();raw['transitions'][0]['infos']['segment_start']=True
    with pytest.raises(ValueError):r.replay_targets(raw)
    raw=source();raw['meta']['operator_excluded']=True
    with pytest.raises(ValueError):r.replay_targets(raw)
    raw=source();raw['meta']['recording']={'has_gaps':True}
    with pytest.raises(ValueError):r.replay_targets(raw)

def test_command_bound_and_convergence():
    q=np.zeros(3);target=np.array([.2,-.3,.01])
    for _ in range(50):
        a=r.command_action(q,target,.096,.0175)
        assert np.max(abs(a*.096))<=.017501
        q+=a*.096
    np.testing.assert_allclose(q,target,atol=1e-6)

def test_full_image_preserved_and_shape_rejected():
    # Execute actual production method without constructing ROS/hardware.
    tree=ast.parse((Path(__file__).parents[1]/'examples/experiments/so101/env.py').read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='SO101Env')
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_process_image')
    image=np.arange(480*640*3,dtype=np.uint8).reshape(480,640,3)
    ns=dict(time=time,decode_rgb=lambda msg:image,np=np,Image=object)
    exec(compile(ast.Module(body=[method],type_ignores=[]),'env_method','exec'),ns)
    stats=dict(source_stamp_ns=0,arrival=0,accepted=0,errors=0,max_gap_s=0)
    b=SimpleNamespace(_lock=threading.Lock(),_camera_stats={'cam_main':stats},full_frame_size=(640,480),image_shape=image.shape,main_roi_spec=None,roi_spec=None,_images={},node=SimpleNamespace(get_logger=lambda:SimpleNamespace(error=lambda *a:None)))
    msg=SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=1,nanosec=0)))
    ns['_process_image'](b,msg,'cam_main',time.monotonic())
    np.testing.assert_array_equal(b._images['cam_main'],image)
    assert b._images['cam_main'] is not image
    b.image_shape=(1080,1920,3);msg.header.stamp.sec=2
    ns['_process_image'](b,msg,'cam_main',time.monotonic())
    assert stats['errors']==1 and not stats['valid']

def test_d_stops_before_first_motion_and_keeps_record(tmp_path,monkeypatch):
    raw=source();raw['transitions'][0]['infos']['joint_control']['target_rad']=[0,0,0]
    path=tmp_path/'ep_0000.pkl';path.write_bytes(pickle.dumps(raw))
    calls=[];b=SimpleNamespace(joint_pos=np.zeros(6),leader_pos=np.zeros(6),_joint_stamp=time.monotonic(),_leader_stamp=time.monotonic(),_command_hold=SimpleNamespace(target=lambda:np.zeros(5)),leader_alignment_target=lambda q:q,_native_io=SimpleNamespace(bounds={'follower':(np.full(3,-1),np.ones(3))}),max_episode_steps=100,hz=10,full_frame_size=(640,480),action_scale=.096,action_semantics='test',set_intervening=lambda x:None,_enter_mode=lambda x:None)
    env=SimpleNamespace(unwrapped=b,step=lambda a:calls.append(a))
    keys=iter(['c','d','x'])
    monkeypatch.setattr(r.recording,'poll',lambda *a:next(keys))
    monkeypatch.setattr(r.recording,'panel_phase',lambda *a:None)
    monkeypatch.setattr(r,'export_video',lambda p:None)
    assert not r.replay_episode(env,object(),str(path),{})
    assert calls==[]
    result=pickle.loads(next(tmp_path.glob('*replay*.pkl')).read_bytes())
    assert result['meta']['n_steps']==0 and 'operator_stop' in result['meta']['stop_reason']
    assert not result['meta']['automatic_training_eligible']

@pytest.mark.parametrize("leader_bias_deg", [0, 4.5, 9.5])
@pytest.mark.parametrize("initial_deg", [0, 6, 9, 45])
def test_replay_records_actual_feedback_and_target_indices(tmp_path,monkeypatch,initial_deg,leader_bias_deg):
    raw=source();path=tmp_path/'ep.pkl';path.write_bytes(pickle.dumps(raw))
    command=np.zeros(5);pos=np.zeros(6)
    command[:3]=np.deg2rad(initial_deg);pos[:3]=command[:3]
    b=SimpleNamespace(joint_pos=pos.copy(),leader_pos=pos.copy(),_joint_stamp=time.monotonic(),_leader_stamp=time.monotonic(),_command_hold=SimpleNamespace(target=lambda:command.copy()),leader_alignment_target=lambda q:q,_native_io=SimpleNamespace(bounds={'follower':(np.full(3,-1),np.ones(3))}),max_episode_steps=100,hz=10,full_frame_size=(640,480),action_scale=.096,action_semantics='test',set_intervening=lambda x:None,_enter_mode=lambda x:None,observation_quality=lambda:{'ok':True})
    obs={'cam_main':np.zeros((1,4,6,3),np.uint8),'state':np.zeros((1,18),np.float32)}
    def step(a):
        time.sleep(.002)  # Yield to the real async writer; hardware steps run at 10 Hz.
        command[:3]+=a*b.action_scale;b.joint_pos[:3]=command[:3];b.leader_pos[:3]=command[:3]
        b.leader_pos[2]+=np.deg2rad(leader_bias_deg)
        b._joint_stamp=b._leader_stamp=time.monotonic()
        return obs,0,False,False,{'joint_control':{'accepted':True,'limited':False},'intervening':False}
    env=SimpleNamespace(unwrapped=b,step=step)
    phase=[''];ready=[0]
    def panel(kb,p,*args):phase[0]=p
    def poll(*args):
        if phase[0]=='REPLAY_READY':
            ready[0]+=1;return 'c' if ready[0]==1 else 'x'
        return None
    monkeypatch.setattr(r.recording,'panel_phase',panel);monkeypatch.setattr(r.recording,'poll',poll)
    monkeypatch.setattr(r.recording,'cur_obs',lambda env:obs);monkeypatch.setattr(r,'export_video',lambda p:None)
    outcome=r.replay_episode(env,object(),str(path),{})
    assert outcome is False
    result=pickle.loads(next(tmp_path.glob('*replay*.pkl')).read_bytes())
    assert result['meta']['stop_reason']=='completed'
    alignment=[t for t in result['transitions'] if t['infos']['source_frame']==-1]
    assert len(alignment)>=3
    assert all(t['infos']['phase']=='replay_align' for t in alignment)
    for t in alignment[-3:]:
        assert max(abs(np.asarray(t['infos']['both_arms_after']['follower_rad'])[:3]))<=np.deg2rad(3)
    assert all(max(abs(t['actions']*.096))<=np.deg2rad(1)+1e-7 for t in result['transitions'])
    assert result['transitions'][0]['infos']['source_frame']==-1
    assert result['transitions'][-1]['infos']['source_frame']==0
    np.testing.assert_allclose(result['transitions'][-1]['infos']['both_arms_after']['follower_rad'][:3],[.1,.2,.3],atol=1e-5)

def test_no_camera_quality_still_checks_both_arms():
    tree=ast.parse((Path(__file__).parents[1]/'examples/experiments/so101/env.py').read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='SO101Env')
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_quality_locked')
    ns={'time':time};exec(compile(ast.Module(body=[method],type_ignores=[]),'quality','exec'),ns)
    b=SimpleNamespace(_camera_stats={},_joint_stamp=time.monotonic(),_leader_stamp=time.monotonic())
    result=ns['_quality_locked'](b)
    assert result['ok'] and result['cameras']=={} and result['camera_skew_s']==0
    b._leader_stamp=0
    assert not ns['_quality_locked'](b)['ok']

def test_action_only_export_does_not_start_encoder(tmp_path,monkeypatch):
    p=tmp_path/'actions.pkl';p.write_bytes(pickle.dumps({'transitions':[{'observations':{'state':np.zeros(18)}}]}))
    monkeypatch.setattr(r.subprocess,'Popen',lambda *a,**kw:pytest.fail('Must not encode images'))
    r.export_video(p)
