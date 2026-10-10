"""Recording/training env using verified LeRobot direct targets and ROS images."""
import math
import threading
import time
import numpy as np
from .env import SO101Env, JOINT_LOW, JOINT_HIGH
from joint_control import CONTROL_HZ
from native_io import NativeJointIO
from pose_alignment import ALIGNMENT, follower_to_leader, leader_to_follower
from so101_home_pose import CONTROLLED_IDX, CONTROLLED_JOINTS, START_JOINT_POSE, JOINT_NAMES, min_jerk


class NativeSO101Env(SO101Env):
    def __init__(self,*args,**kwargs):
        self.control_backend='lerobot_native_v1'
        self.pose_alignment=ALIGNMENT
        self._native_stop=threading.Event()
        self._native_feedback_error=None
        super().__init__(*args,**kwargs)
        try:
            self._native_io=NativeJointIO()
            lo,hi=self._native_io.bounds['follower']
            self._action_low=np.maximum(JOINT_LOW[:3],lo)
            self._action_high=np.minimum(JOINT_HIGH[:3],hi)
            if np.any(self._action_low>=self._action_high):raise RuntimeError('No valid native action interval')
            self._native_raw={}
            self._leader_follow_command=None
            self._refresh_native()
            self._native_thread=threading.Thread(target=self._native_poll,daemon=True)
            self._native_thread.start()
        except Exception:
            self.close()
            raise

    # Ignore ROS joint topics: only native encoder packets populate state.
    def _on_joint(self,msg):pass
    def _on_leader(self,msg):pass

    def _refresh_native(self):
        for arm in ['follower','leader']:
            q,stamp,raw=self._native_io.read(arm)
            with self._lock:
                if arm=='follower':
                    old=self._joint_pos;old_stamp=self._joint_stamp
                    self._joint_vel=np.zeros(6) if old is None or stamp<=old_stamp else (q-old)/(stamp-old_stamp)
                    self._joint_pos=q;self._joint_stamp=stamp
                else:self._leader_pos=q;self._leader_stamp=stamp
                self._native_raw[arm]=dict(raw=raw,received_monotonic_s=stamp)

    def _native_poll(self):
        while not self._native_stop.is_set():
            tick=time.monotonic()
            try:self._refresh_native()
            except Exception as exc:
                self._native_feedback_error=repr(exc)
                break  # Freshness checks prevent further control; no silent stale reuse.
            self._native_stop.wait(max(0,1/CONTROL_HZ-(time.monotonic()-tick)))

    def set_leader_torque(self,on):return self._native_torque('leader',on)
    def set_follower_torque(self,on):return self._native_torque('follower',on)
    def _prepare_start_pose(self):
        self.node.get_logger().info(
            f'初始化 wrist_roll 至 {math.degrees(START_JOINT_POSE[4]):.2f}°；仅归位阶段允许调整')
        result=self._native_io.home_wrist_roll(START_JOINT_POSE[4])
        measured=self._settled_joint_position()
        residual=abs(measured[4]-START_JOINT_POSE[4])
        if residual>math.radians(.5):
            raise RuntimeError(f'Reset wrist_roll failed acceptance: {math.degrees(residual):.2f}deg > 0.5deg')
        result.update(measured_deg=math.degrees(measured[4]),residual_deg=math.degrees(residual))
        return result
    def _native_torque(self,arm,on):
        try:
            result=self._native_io.torque(arm,on)
            if result and arm=='leader' and on:
                with self._native_io.lock:
                    goal=self._native_io.robots['leader'].bus.sync_read('Goal_Position')
                self._leader_follow_command=np.asarray([math.radians(goal[j]) for j in CONTROLLED_JOINTS])
            return result
        except Exception as exc:
            self.node.get_logger().error(f'Native {arm} torque failed: {exc}')
            return False

    def _publish_arm(self,target5):
        self._native_io.send('follower',np.asarray(target5)[:3])
        self._joint_control_status['native_send']=self._native_io.last_send['follower'].copy()
        with self._lock:self._joint_control_status['raw_feedback']=dict(self._native_raw)

    def leader_alignment_target(self,follower_pos):
        q=np.asarray(follower_pos,float).copy()
        mapped=follower_to_leader({j:math.degrees(q[i]) for i,j in enumerate(CONTROLLED_JOINTS)})
        q[:3]=[math.radians(mapped[j]) for j in CONTROLLED_JOINTS]
        leader=self.leader_pos
        if leader is not None:q[3:]=leader[3:]  # Tool joints are never driven.
        return q

    def leader_pose_in_follower_coordinates(self,leader_pos):
        q=np.asarray(leader_pos,float).copy()
        aligned=leader_to_follower({j:math.degrees(q[i]) for i,j in enumerate(CONTROLLED_JOINTS)})
        q[:3]=[math.radians(aligned[j]) for j in CONTROLLED_JOINTS]
        return q

    def _drive_leader(self,follower_pos):
        cur=self.leader_pos
        if cur is None or time.monotonic()-self._leader_stamp>.25:raise RuntimeError('Stale leader feedback')
        target=self.leader_alignment_target(follower_pos)[:3]
        # Alignment is deliberately slow; this is separate from action execution.
        previous=self._leader_follow_command
        if previous is None:previous=cur[:3].copy()
        delta=np.clip(target-previous,-math.radians(15)/self.hz,math.radians(15)/self.hz)
        lo,hi=self._native_io.bounds['leader']
        command=np.clip(previous+delta,lo,hi)
        if np.any(abs(command-cur[:3])>math.radians(8)):
            self.node.get_logger().warn('Leader alignment lag; keeping its last target without further correction')
            return
        self._native_io.send('leader',command)
        self._leader_follow_command=command

    def _publish_arm_ramp(self,q_to,n=25,v_max_deg=30.,d_min=.6,d_max=3.,hold_idx=None):
        if self.joint_pos is None:raise RuntimeError('Missing native feedback before return')
        # Begin at accepted servo goal, preserving static holding compensation.
        with self._native_io.lock:
            goal=self._native_io.robots['follower'].bus.sync_read('Goal_Position')
        start=np.asarray([math.radians(goal[j]) for j in JOINT_NAMES[:4]])
        target=np.clip(np.asarray(q_to,float)[:4],np.maximum(JOINT_LOW[:4],self._native_io.reset_bounds['follower'][0]),np.minimum(JOINT_HIGH[:4],self._native_io.reset_bounds['follower'][1]))
        if not np.isfinite(target).all():raise ValueError('Nonfinite native return target')
        lo,hi=self._native_io.reset_bounds['follower']
        invalid=(~np.isfinite(start)) | (start<lo-1e-8) | (start>hi+1e-8)
        if np.any(invalid):
            # An old register goal need not be a legal command in the current
            # calibrated coordinates. Never interpolate from it or silently clip it.
            cur=self.joint_pos
            if cur is None or time.monotonic()-self._joint_stamp>.25:
                raise RuntimeError('Invalid native return start goal and stale feedback; no command sent')
            measured=np.asarray(cur,float)[:4].copy()
            if not np.isfinite(measured).all() or np.any(measured<lo-1e-8) or np.any(measured>hi+1e-8):
                raise RuntimeError('Invalid native return start goal and measured position outside firmware limits; '
                                   f'goal_deg={np.degrees(start).tolist()}, '
                                   f'measured_deg={np.degrees(measured).tolist()}, '
                                   f'low_deg={np.degrees(lo).tolist()}, high_deg={np.degrees(hi).tolist()}; '
                                   'no command sent')
            self.node.get_logger().warn(
                'Native return: invalid stored Goal_Position; starting from fresh measured position. '
                f'joints={[JOINT_NAMES[:4][i] for i in np.flatnonzero(invalid)]}, '
                f'goal_deg={np.degrees(start).tolist()}, measured_deg={np.degrees(measured).tolist()}, '
                f'low_deg={np.degrees(lo).tolist()}, high_deg={np.degrees(hi).tolist()}')
            start=measured
        self._native_io.enable_reset_wrist_flex()
        disp=float(np.max(abs(target-start)))
        duration=max(d_min,1.875*math.degrees(disp)/v_max_deg,math.sqrt(5.774*math.degrees(disp)/60.))
        initial_gap=start-self.joint_pos[:4]
        begin=time.monotonic()
        while True:
            tick=time.monotonic()
            cur=self.joint_pos
            if cur is None or tick-self._joint_stamp>.25:raise RuntimeError('Stale native return feedback')
            fraction=min(1.,(tick-begin)/duration)
            command=start+(target-start)*min_jerk(fraction)
            if np.any(abs(command-cur[:4])>abs(initial_gap)+math.radians(8)):
                raise RuntimeError('Native return tracking lag; no further correction')
            self._native_io.send_reset_arm(command)
            if fraction>=1:break
            time.sleep(max(0,1/CONTROL_HZ-(time.monotonic()-tick)))
        return 0.0  # The ramp has already finished; settling must not wait its duration twice.

    def close(self):
        self._native_stop.set()
        thread=getattr(self,'_native_thread',None)
        if thread is not None:thread.join(timeout=2)
        io=getattr(self,'_native_io',None)
        if io is not None:
            # Preserve follower hold. Restore active leader motors that were on
            # before this env, holding their present pose rather than an old target.
            try:
                with io.lock:
                    bus=io.robots['leader'].bus
                    original=io.initial_torque.get('leader',{})
                    off=bus.sync_read('Torque_Enable',normalize=False)
                    restore=[j for j in CONTROLLED_JOINTS if int(original.get(j,0))==1 and int(off[j])==0]
                    if restore:
                        cur=bus.sync_read('Present_Position')
                        bus.sync_write('Goal_Position',{j:cur[j] for j in restore});bus.enable_torque(restore,num_retry=2)
            except Exception as exc:self.node.get_logger().error('Leader restore failed: '+repr(exc))
            finally:io.close()
        super().close()
