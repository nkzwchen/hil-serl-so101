"""Offline regression of production relative-joint step; no ROS nodes or motors."""
import ast
import math
from pathlib import Path
import sys
import time
import threading
from types import SimpleNamespace, MethodType
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE / 'hil-serl' if (HERE / 'hil-serl').is_dir() else HERE.parent
MODULE = REPO / 'examples/experiments/so101'
sys.path.insert(0, str(MODULE))
from joint_control import CommandHold, bounded_joint_delta, execute_joint_request
from so101_home_pose import START_JOINT_POSE, Q_LOW, Q_HIGH, hold_frozen


class JointTests(unittest.TestCase):
    def fixture(self):
        tree = ast.parse((MODULE / 'env.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SO101Env')
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('step', 'get_intervene_action', '_enter_mode', '_get_obs')]
        for method in methods:
            method.returns = None
            for arg in method.args.args:
                arg.annotation = None
        ns = dict(np=np, math=math, time=SimpleNamespace(monotonic=time.monotonic, sleep=lambda _: None),
                  bounded_joint_delta=bounded_joint_delta, execute_joint_request=execute_joint_request,
                  N_ACTION=4, N_ARM=5, CONTROLLED_IDX=(0,1,2,3), JOINT_LOW=Q_LOW,
                  JOINT_HIGH=Q_HIGH, hold_frozen=hold_frozen)
        exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), '<production joint step>', 'exec'), ns)
        q = START_JOINT_POSE[:5].copy(); hold=CommandHold()
        offset=np.array([0,-.01,-.02,.01,0]);hold.calibrate(q+offset,q)
        sent=[]
        env=SimpleNamespace(node=SimpleNamespace(get_logger=lambda: SimpleNamespace(warn=lambda _: None, info=lambda _: None)),
            joint_pos=np.r_[q,.32],leader_pos=np.r_[q,.32],_joint_stamp=time.monotonic(),_leader_stamp=time.monotonic(),
            action_scale=.02,_command_hold=hold,is_intervening=False,_enter_mode=lambda _: None,
            _lead_anchor=q.copy(),_foll_anchor=q.copy(),observation_quality=lambda: {'ok':True},
            _publish_arm=lambda target:sent.append(target.copy()),_drive_leader=lambda _:None,
            hz=10.,_get_obs=lambda:{},compute_reward=lambda _:0,_step_count=0,_episode_return=0,
            max_episode_steps=100,intervene_deadband=0.,last_observation_quality=lambda:{})
        for name in ('step','get_intervene_action'):setattr(env,name,MethodType(ns[name],env))
        env._production_enter_mode = MethodType(ns['_enter_mode'],env)
        env._production_get_obs = MethodType(ns['_get_obs'],env)
        return env,sent

    def test_action_dimension_finiteness_and_units(self):
        np.testing.assert_allclose(bounded_joint_delta([2,-2,.5,0],.02),[.02,-.02,.01,0])
        for bad in ([0,0,0],[0]*5,[np.nan,0,0,0]):
            with self.assertRaises(ValueError):bounded_joint_delta(bad)

    def test_zero_holds_command_despite_measurement_noise(self):
        env,sent=self.fixture();wanted=env._command_hold.target()
        for _ in range(10):
            env.joint_pos[1]+=.0001;env.step(np.zeros(4))
            np.testing.assert_array_equal(sent[-1],wanted)

    def test_increment_accumulates_without_measured_tracking_gate(self):
        env,sent=self.fixture();start=env._command_hold.target()
        env.step([.5,0,0,0]);env.step([.5,0,0,0])
        self.assertAlmostEqual(sent[-1][0]-start[0],.02)
        for _ in range(20):env.step([1,0,0,0])
        self.assertAlmostEqual(sent[-1][0]-start[0],.42)
        self.assertFalse(env._joint_control_status['limited'])

    def test_human_anchor_freezes_roll_and_ignores_camera_quality(self):
        env,sent=self.fixture();wanted=env._command_hold.target();env.is_intervening=True
        env.observation_quality=lambda:{'ok':False}
        env.step(np.zeros(4));np.testing.assert_array_equal(sent[-1],wanted)
        env.leader_pos[0]+=.03;env.leader_pos[4]+=.4;env.leader_pos[5]+=.4
        _,_,_,_,info=env.step(np.zeros(4))
        self.assertAlmostEqual(sent[-1][0]-wanted[0],.02)
        self.assertEqual(sent[-1][4],env.joint_pos[4])
        np.testing.assert_allclose(info['intervene_action'],info['joint_control']['requested_action'],atol=1e-8)
        env.step(np.zeros(4));self.assertAlmostEqual(sent[-1][0]-wanted[0],.03)

    def test_enter_human_with_tracking_lag_does_not_rebase_command(self):
        env,sent=self.fixture();wanted=env._command_hold.target()
        env.joint_pos[0]-=.015  # Real follower lags behind the held command.
        env.leader_pos[:4]+=np.array([.15,-.1,.07,.04])
        env.is_intervening=True;env._last_mode='policy'
        env.set_leader_torque=lambda _:True
        env._enter_mode=env._production_enter_mode
        env.step(np.zeros(4))
        np.testing.assert_allclose(sent[-1],wanted,atol=1e-12)
        np.testing.assert_array_equal(env._last_joint_action,np.zeros(4))
        for i in range(4):
            env.leader_pos[i]+=.005
            env.step(np.zeros(4))
            wanted[i]+=.005
            np.testing.assert_allclose(sent[-1],wanted,atol=1e-12)

    def test_reenter_human_uses_new_command_and_leader_baselines(self):
        env,sent=self.fixture();env.set_leader_torque=lambda _:True
        env._enter_mode=env._production_enter_mode;env._last_mode='policy'
        env.is_intervening=True;env.step(np.zeros(4))
        env.is_intervening=False;env.step([.5,0,0,0])
        wanted=env._command_hold.target()
        env.leader_pos[0]+=.1
        env.is_intervening=True;env.step(np.zeros(4))
        np.testing.assert_allclose(sent[-1],wanted,atol=1e-12)
        env.leader_pos[0]-=.005;env.step(np.zeros(4))
        wanted[0]-=.005
        np.testing.assert_allclose(sent[-1],wanted,atol=1e-12)

    def test_human_and_policy_share_request_and_applied_contract(self):
        human,hs=self.fixture();policy,ps=self.fixture()
        human.is_intervening=True;human.leader_pos[:4]+=[.04,.01,-.01,0]
        # Pan is already at its tracking boundary. Other joints must still update.
        for e in (human,policy):e.joint_pos[0]-=math.radians(3)
        _,_,_,_,hi=human.step(np.zeros(4))
        _,_,_,_,pi=policy.step([2,.5,-.5,0])
        np.testing.assert_allclose(hs[-1],ps[-1],atol=1e-9)
        np.testing.assert_allclose(hi['intervene_action'],[1,.5,-.5,0])
        self.assertAlmostEqual(hi['joint_control']['applied_delta_rad'][0],.02)
        self.assertEqual(hi['joint_control']['held_joints'],[])
        self.assertNotEqual(hi['joint_control']['applied_delta_rad'][1],0.)
        np.testing.assert_allclose(hi['joint_control']['applied_delta_rad'],pi['joint_control']['applied_delta_rad'])

    def test_feedback_lag_does_not_reverse_or_hold_request(self):
        env,sent=self.fixture();start=env._command_hold.target()
        env.joint_pos[2]+=math.radians(3)+.002
        env.step([.5,0,-1,.5])
        self.assertAlmostEqual(sent[-1][2],start[2]-.02)
        self.assertAlmostEqual(sent[-1][0]-start[0],.01)
        self.assertAlmostEqual(sent[-1][3]-start[3],.01)
        self.assertEqual(env._joint_control_status['joint_reasons'][2],[])

    def test_replaying_human_request_reproduces_published_target(self):
        human,hs=self.fixture();policy,ps=self.fixture();human.is_intervening=True
        human.leader_pos[:4]+=[.04,.01,0,-.05]
        for _ in range(8):
            _,_,_,_,info=human.step(np.zeros(4));policy.step(info['intervene_action'])
            np.testing.assert_allclose(hs[-1],ps[-1],atol=1e-8)

    def test_observation_exposes_full_command_and_compensation(self):
        env,_=self.fixture();env._lock=threading.RLock()
        env._joint_pos=env.joint_pos.copy();env._joint_vel=np.zeros(6,np.float32)
        env._images={};env._quality_locked=lambda:{};env.image_keys=[];env.display_image=False
        obs=env._production_get_obs()['state']
        self.assertEqual(obs['command_error'].shape,(4,))
        np.testing.assert_allclose(obs['joint_pos'][:4]+obs['command_error'],env._command_hold.target()[:4],atol=1e-7)
        np.testing.assert_allclose(obs['hold_offset'],env._command_hold.offset[:4],atol=1e-7)

    def test_camera_quality_is_diagnostic_for_both_sources(self):
        env,sent=self.fixture();env.observation_quality=lambda:{'ok':False}
        start=env._command_hold.target();env.step([.5,0,0,0])
        self.assertAlmostEqual(sent[-1][0]-start[0],.01)

    def test_missing_and_stale_leader_hold(self):
        for missing in (True,False):
            env,sent=self.fixture();wanted=env._command_hold.target();env.is_intervening=True
            if missing:env._lead_anchor=None
            else:env._leader_stamp-=1
            env.step(np.ones(4));np.testing.assert_array_equal(sent[-1],wanted)
            self.assertFalse(env._joint_control_status['accepted'])

    def test_stale_follower_refuses_publication(self):
        env,sent=self.fixture();env._joint_stamp-=1
        with self.assertRaises(RuntimeError):env.step(np.ones(4))
        self.assertEqual(sent,[])

    def test_no_pose_constraint_and_joint_limits(self):
        env,sent=self.fixture();env.joint_pos[3]+=.2
        env._command_hold.calibrate(env.joint_pos[:5],env.joint_pos[:5])
        env.step([0,0,0,.5]);self.assertTrue(env._joint_control_status['accepted'])
        env,sent=self.fixture();env.joint_pos[0]=Q_HIGH[0]-.005
        env._command_hold.calibrate(env.joint_pos[:5],env.joint_pos[:5])
        env.step([1,0,0,0]);self.assertAlmostEqual(sent[-1][0],Q_HIGH[0])


if __name__=='__main__':unittest.main(verbosity=2)
