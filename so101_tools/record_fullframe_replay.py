"""Opt-in full-frame demonstrations and guarded follower-target replay.

Does not change the standard recorder's defaults. Replay is NOT an expert demo.
"""
import argparse
import json
import math
from pathlib import Path
import pickle
import subprocess
import sys
import time
import numpy as np
import record_demo_so101 as recording
from async_recorder import AsyncEpisodeRecorder

# Replay alignment tolerance includes the measured leader steady-state error.
LEADER_ALIGNMENT_TOL_RAD = math.radians(10)


def export_video(path):
    """Convenience review MP4; lossless RGB arrays and timestamps stay in pkl."""
    raw=pickle.load(open(path,'rb'));ts=raw['transitions']
    if not ts or 'cam_main' not in ts[0]['observations']:return
    image=np.asarray(ts[0]['observations']['cam_main']).squeeze()
    h,w,_=image.shape;dst=Path(path).with_suffix('.mp4')
    proc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{w}x{h}','-r',str(raw['meta']['env_hz']),'-i','-','-an','-c:v','libx264',
        '-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(dst)],stdin=subprocess.PIPE)
    try:
        for t in ts:
            a=np.asarray(t['observations']['cam_main']).squeeze()
            if a.shape!=image.shape or a.dtype!=np.uint8:raise ValueError('Inconsistent RGB frames')
            proc.stdin.write(a.tobytes())
        proc.stdin.close()
        if proc.wait()!=0:raise RuntimeError('ffmpeg failed')
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
    print('原图复盘视频（控制频率，不含回程）：',dst,flush=True)


def replay_targets(raw):
    if raw['meta'].get('recording', {}).get('has_gaps'):
        raise ValueError('Source contains missing steps')
    if raw['meta'].get('operator_excluded') or raw['meta'].get('safety_error'):
        raise ValueError('Excluded/unsafe source cannot be replayed')
    targets=[]
    for t in raw['transitions']:
        c=t['infos'].get('joint_control') or {}
        if not c.get('accepted') or t['infos'].get('segment_start'):
            raise ValueError('Rejected control or discontinuous source')
        q=np.asarray(c.get('target_rad'),dtype=float)
        if q.shape!=(3,) or not np.isfinite(q).all():raise ValueError('Missing valid recorded target')
        targets.append(q)
    if not targets:raise ValueError('Empty source')
    start=np.asarray(raw['transitions'][0]['infos']['joint_control']['command_before_rad'],float)
    if start.shape!=(3,) or not np.isfinite(start).all():raise ValueError('Invalid start')
    return start,targets


def command_action(current, target, scale, max_step):
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid scale')
    return (np.clip(np.asarray(target)-np.asarray(current),-max_step,max_step)/scale).astype(np.float32)


def feedback(env):
    b=env.unwrapped;now=time.monotonic()
    if b.joint_pos is None or b.leader_pos is None or max(now-b._joint_stamp,now-b._leader_stamp)>.25:
        raise RuntimeError('Fresh feedback required from BOTH arms')
    if not np.isfinite(b.joint_pos).all() or not np.isfinite(b.leader_pos).all():
        raise RuntimeError('Nonfinite arm feedback')
    return dict(follower_rad=b.joint_pos.tolist(),leader_rad=b.leader_pos.tolist(),
                follower_stamp=b._joint_stamp,leader_stamp=b._leader_stamp)


def replay_episode(env,kb,path,meta):
    b=env.unwrapped
    try:export_video(path)
    except Exception as exc:print("视频导出失败，原始pkl保留：",exc,flush=True)
    recording.panel_phase(kb,'REPLAY_READY','本段已保存。请先拔出管子、确保归位路径无障碍；C 受控归位后回放（双手离开），Q 跳过，X 退出')
    while True:
        k=recording.poll(kb,'cCqQxX')
        if k in ('q','Q'):return True
        if k in ('x','X'):return False
        if k in ('c','C'):break
        time.sleep(.02)
    raw=pickle.load(open(path,'rb'))
    try:
        start,targets=replay_targets(raw);fb=feedback(env)
        # After manual withdrawal, return to the recorded start with incremental
        # commands. Distance to start is not a tracking-error safety limit.
        current=b._command_hold.target()[:3]
        if not np.isfinite(current).all():raise ValueError('Invalid current command')
        if not hasattr(b,'leader_alignment_target'):raise ValueError('Native aligned backend required')
        lo,hi=b._native_io.bounds['follower']
        if any(np.any(q<lo) or np.any(q>hi) for q in [current,start,*targets]):raise ValueError('Source outside current bounds')
    except (ValueError,RuntimeError) as exc:
        print('拒绝回放：',exc,flush=True)
        return True
    dst=Path(path).with_name(Path(path).stem+'_replay_'+time.strftime('%H%M%S')+'.pkl')
    writer=AsyncEpisodeRecorder(dst)
    reason='completed';quit_session=False;count=0
    old_limit=b.max_episode_steps
    recording.panel_phase(kb,'REPLAY')
    try:
        b.set_intervening(False)
        b._enter_mode('policy')  # restores leader torque; failure refuses replay
        b.max_episode_steps=1_000_000;b._step_count=0
        # Start segment plus recorded target positions. Speed is deliberately
        # capped below the native leader-follow rate (15deg/s).
        for index,target in [(-1,start),*enumerate(targets)]:
            if index == -1:
                recording.panel_phase(kb,'REPLAY','正在受控恢复本段录制起点；D 停止，X 退出')
            elif index == 0:
                recording.panel_phase(kb,'REPLAY','两臂起点验收通过，正在回放；D 停止，X 退出')
            settled=0
            # Allow travel time and leader catch-up, not just a fixed 10 s.
            travel=float(np.max(abs(b._command_hold.target()[:3]-target)))
            leader_travel=float(np.max(abs(b.leader_alignment_target(b.joint_pos)[:3]-b.leader_pos[:3])))
            timeout=max(10., 3*travel/math.radians(10)+leader_travel/math.radians(15)+5.) if index == -1 else 5.
            deadline=time.monotonic()+timeout
            if index == -1:
                print(f'自动归位：最大关节距离 {math.degrees(travel):.1f}°，速度上限 10°/s，超时 {timeout:.1f}s；D 停止 / X 退出',flush=True)
            while True:
                key=recording.poll(kb,'dDxX')
                if key is not None:
                    quit_session=key in ('x','X');raise InterruptedError('operator_stop')
                before=feedback(env)
                if np.max(abs(b._command_hold.target()[:3]-np.asarray(before['follower_rad'])[:3]))>math.radians(8):
                    raise RuntimeError('Follower tracking lag exceeds 8 degrees')
                if not b.observation_quality()['ok']:raise RuntimeError('Replay observation stale/invalid')
                leader_goal=b.leader_alignment_target(b.joint_pos)[:3]
                leader_gap=np.max(abs(leader_goal-np.array(before['leader_rad'])[:3]))
                action=command_action(b._command_hold.target()[:3],target,b.action_scale,
                                      min(b.action_scale,math.radians(10)/b.hz))
                # Let leader catch up before advancing follower.
                if leader_gap>LEADER_ALIGNMENT_TOL_RAD:action[:]=0
                obs=recording.cur_obs(env)
                next_obs,reward,done,trunc,info=env.step(action)
                after=feedback(env);control=info.get('joint_control',{})
                packet=dict(observations=obs,next_observations=next_obs,actions=action,
                    rewards=0.,masks=1.,dones=False,infos=dict(info,phase='replay_align' if index == -1 else 'replay',
                    source_frame=index,expert_label=False,both_arms_before=before,both_arms_after=after,
                    replay_alignment=dict(leader_error_deg=np.rad2deg(leader_goal-np.asarray(before['leader_rad'])[:3]).tolist(),
                        follower_target_error_deg=np.rad2deg(target-np.asarray(after['follower_rad'])[:3]).tolist(),
                        waiting_for_leader=bool(leader_gap>LEADER_ALIGNMENT_TOL_RAD))))
                if not writer.submit(packet):raise RuntimeError('Replay writer overflow/error')
                count+=1
                if not control.get('accepted',False) or control.get('limited'):raise RuntimeError('Replay control rejected/limited')
                if done or trunc:raise RuntimeError('Environment terminated replay')
                command_ready=np.max(abs(b._command_hold.target()[:3]-target))<1e-5
                if index == -1:
                    follower_ready=np.max(abs(np.asarray(after['follower_rad'])[:3]-start))<=math.radians(3)
                    leader_start=b.leader_alignment_target(np.asarray(after['follower_rad']))[:3]
                    leader_ready=np.max(abs(leader_start-np.asarray(after['leader_rad'])[:3]))<=LEADER_ALIGNMENT_TOL_RAD
                    settled=settled+1 if command_ready and follower_ready and leader_ready else 0
                    if settled>=3:break
                elif command_ready and leader_gap<=LEADER_ALIGNMENT_TOL_RAD:break
                if time.monotonic()>deadline:
                    raise RuntimeError(f'Replay convergence timeout: source_frame={index}, '
                        f'leader_error_deg={np.rad2deg(leader_goal-np.asarray(after["leader_rad"])[:3]).round(2).tolist()}, '
                        f'follower_error_deg={np.rad2deg(target-np.asarray(after["follower_rad"])[:3]).round(2).tolist()}, '
                        f'command_ready={command_ready}')
    except (Exception,KeyboardInterrupt) as exc:
        reason=f'{type(exc).__name__}: {exc}';print('回放停止，保留记录：',reason,flush=True)
    finally:
        # No new trajectory command on error: keep last controller target.
        b.max_episode_steps=old_limit
        writer.finish(dict(data_role='robot_replay_not_expert',source_episode=str(Path(path).resolve()),
            stop_reason=reason,n_steps=count,env_hz=b.hz,full_frame_size=b.full_frame_size,
            action_scale_rad=b.action_scale,action_semantics=b.action_semantics,
            replay_mode='recorded_follower_targets_leader_aligned_follow',max_speed_deg_s=10.,
            leader_alignment_tolerance_deg=math.degrees(LEADER_ALIGNMENT_TOL_RAD),automatic_training_eligible=False))
        print('回放实测数据：',dst,flush=True)
        try:export_video(dst)
        except Exception as exc:print('回放视频导出失败，pkl保留：',exc,flush=True)
    if quit_session:return False
    # Return must be manual; never automatically withdraw an inserted tube.
    recording.panel_phase(kb,'REPLAY_READY','回放结束，当前保持。扶住主臂后按 C 开启人工回程；X 退出保持')
    while True:
        k=recording.poll(kb,'cCqQxX')
        if k in ('x','X','q','Q'):return False
        if k in ('c','C'):break
        time.sleep(.02)
    b.start_human_intervention();b._step_count=0
    recording.panel_phase(kb,'RETURN','人工回程中；回到起点附近后 C，X 退出')
    while True:
        k=recording.poll(kb,'cCqQxX')
        if k in ('c','C'):return True
        if k in ('x','X'):return False
        env.step(np.zeros(3,np.float32))


def main():
    ap=argparse.ArgumentParser(description=__doc__,add_help=False)
    ap.add_argument('--full-width',type=int,default=640)
    ap.add_argument('--full-height',type=int,default=480)
    args,rest=ap.parse_known_args()
    if '--help' in rest or '-h' in rest:
        print('额外选项: --full-width 640 --full-height 480（必须匹配当前发布的原图尺寸，不缩放）')
    if args.full_width<=0 or args.full_height<=0:raise ValueError('Positive image dimensions required')
    from experiments.so101.config import TrainConfig
    sys.argv=[sys.argv[0],*rest]
    if '--out' not in rest:sys.argv+=['--out','recordings_fullframe']
    return recording.main(config_factory=lambda:TrainConfig(full_frame_size=(args.full_width,args.full_height)),
                          after_local_save=replay_episode)

if __name__=='__main__':sys.exit(main())
