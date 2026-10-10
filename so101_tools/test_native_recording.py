"""Offline native I/O and approved-trace replay checks; no ports or torque opened."""
import json
import math
from pathlib import Path
import sys
import threading
import time
import unittest
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE/'hil-serl' if (HERE/'hil-serl').is_dir() else HERE.parent
MODULE=REPO/'examples/experiments/so101'
sys.path.insert(0,str(MODULE))
from native_io import NativeJointIO
from joint_control import DEFAULT_STEP_RAD, execute_joint_request, CONTROL_HZ
from so101_home_pose import CONTROLLED_JOINTS, JOINT_NAMES
from pose_alignment import follower_to_leader, leader_to_follower
sys.path.insert(0,str(MODULE.parents[1]))
from experiments.so101.native_env import NativeSO101Env


class Bus:
    def __init__(self):
        self.torque={j:1 for j in JOINT_NAMES}
        self.writes=[];self.enable_calls=[];self.disable_calls=[]
        self.position={j:10.*(i+1) for i,j in enumerate(JOINT_NAMES)}
        self.motors={j:SimpleNamespace(id=i+1) for i,j in enumerate(JOINT_NAMES)}
    def sync_read(self,register,normalize=True):
        return dict(self.torque if register=='Torque_Enable' else self.position)
    def _normalize(self,values):return values
    def sync_write(self,register,values):self.writes.append((register,dict(values)))
    def enable_torque(self,motors,**kwargs):
        self.enable_calls.append(list(motors))
        for j in motors:self.torque[j]=1
    def disable_torque(self,motors,**kwargs):
        self.disable_calls.append(list(motors))
        for j in motors:self.torque[j]=0


class NativeRecordingTests(unittest.TestCase):
    def io(self):
        io=NativeJointIO.__new__(NativeJointIO)
        io.lock=threading.RLock();io.last_send={}
        bus=Bus()
        def send_action(action):
            bus.sync_write('Goal_Position',{j.removesuffix('.pos'):v for j,v in action.items()})
            return action
        robot=SimpleNamespace(bus=bus,send_action=send_action)
        io.robots={'follower':robot,'leader':robot}
        io.bounds={a:(np.full(4,-3.),np.full(4,3.)) for a in io.robots}
        return io,bus

    def test_direct_send_units_and_frozen_tools(self):
        io,bus=self.io();target=np.radians([10.,20.,-30.,40.])
        actual=io.send('follower',target)
        np.testing.assert_allclose(actual,target)
        self.assertEqual(set(bus.writes[-1][1]),set(CONTROLLED_JOINTS))
        self.assertAlmostEqual(bus.writes[-1][1]['elbow_flex'],-30.)
        with self.assertRaises(ValueError):io.send('follower',[4,0,0,0])
        self.assertEqual(len(bus.writes),1)

    def test_torque_sync_only_released_active_joints(self):
        io,bus=self.io();bus.torque['elbow_flex']=0;bus.torque['wrist_roll']=0
        self.assertTrue(io.torque('leader',True))
        self.assertEqual(bus.writes[-1][1],{'elbow_flex':30.})
        self.assertEqual(bus.enable_calls,[['elbow_flex']])
        self.assertEqual(bus.torque['wrist_roll'],0)
        self.assertTrue(io.torque('leader',False))
        self.assertEqual(bus.disable_calls,[list(CONTROLLED_JOINTS)])
        self.assertEqual(bus.torque['gripper'],1)

    def test_single_packet_read_units(self):
        io,bus=self.io();q,stamp,raw=io.read('follower')
        np.testing.assert_allclose(q[:5],np.radians([10,20,30,40,50]))
        self.assertAlmostEqual(q[5],60*math.pi/100)
        self.assertEqual(raw,bus.position)

    def test_formal_env_inverse_alignment_preserves_tools_and_pan(self):
        env=NativeSO101Env.__new__(NativeSO101Env)
        env._lock=threading.RLock();env._leader_pos=np.array([0,0,0,0,.7,.8])
        follower=np.radians([19,19,1,-20,10,20])
        target=env.leader_alignment_target(follower)
        self.assertAlmostEqual(target[0],follower[0])
        converted=leader_to_follower(dict(zip(CONTROLLED_JOINTS,np.degrees(target[:4]))))
        np.testing.assert_allclose(list(converted.values()),np.degrees(follower[:4]))
        np.testing.assert_array_equal(target[4:],[.7,.8])

    def test_leader_return_ramp_accumulates_commands_not_measured_deadband(self):
        env=NativeSO101Env.__new__(NativeSO101Env)
        env._lock=threading.RLock();env._leader_pos=np.zeros(6);env._leader_stamp=time.monotonic()
        env._leader_follow_command=np.zeros(4);env.hz=30
        sent=[]
        env._native_io=SimpleNamespace(bounds={'leader':(np.full(4,-3.),np.full(4,3.))},
                                      send=lambda arm,q:sent.append(q.copy()))
        env.node=SimpleNamespace(get_logger=lambda:SimpleNamespace(warn=lambda _:None))
        env.leader_alignment_target=lambda _:np.r_[np.ones(4),0,0]
        for _ in range(4):env._drive_leader(np.zeros(6))
        np.testing.assert_allclose(sent[-1],np.full(4,math.radians(2)))

    def test_requested_action_clipping_is_visible(self):
        q=np.zeros(4);bounds=np.ones(4)*2
        target,status=execute_joint_request(q,q,q,[2,-3,.5,0],-bounds,bounds)
        np.testing.assert_allclose(status['requested_action'],[1,-1,.5,0])
        self.assertEqual(status['raw_requested_action'],[2,-3,.5,0])
        self.assertEqual(status['joint_reasons'][:2],[['action_range'],['action_range']])
        np.testing.assert_allclose(target,np.array([1,-1,.5,0])*DEFAULT_STEP_RAD)

    def test_approved_native_trace_reproduced_by_policy_requests(self):
        folder=MODULE.parents[3]/'validation/joint_relative_2026-10-09/native_222458'
        if not (folder/'states.jsonl').exists():self.skipTest('Local accepted trace unavailable')
        meta=json.loads((folder/'meta.json').read_text())
        rows=[json.loads(l) for l in (folder/'states.jsonl').read_text().splitlines()]
        command=np.radians([meta['initial_goal'][j] for j in CONTROLLED_JOINTS])
        lo=np.radians([meta['bounds_deg']['low'][j] for j in CONTROLLED_JOINTS])
        hi=np.radians([meta['bounds_deg']['high'][j] for j in CONTROLLED_JOINTS])
        count=0
        for row in rows:
            if row['event']!='step':continue
            wanted=np.radians([row['sent'][j+'.pos'] for j in CONTROLLED_JOINTS])
            measured=np.radians([row['follower'][j+'.pos'] for j in CONTROLLED_JOINTS])
            request=((wanted-command)/DEFAULT_STEP_RAD).astype(np.float32)
            command,status=execute_joint_request(command,measured,np.zeros(4),request,lo,hi)
            np.testing.assert_allclose(command,wanted,atol=1e-8)
            self.assertFalse(status['limited']);count+=1
        self.assertEqual(count,1508);self.assertEqual(CONTROL_HZ,30.)


if __name__=='__main__':unittest.main(verbosity=2)
