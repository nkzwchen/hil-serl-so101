"""Offline checks of new three-joint recording/training contract."""
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
import numpy as np
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'hil-serl/examples'))
sys.path.insert(0,str(root/'hil-serl/examples/experiments/so101'))
from joint_control import bounded_joint_delta,execute_joint_request,PROPRIO_DIMS,ACTION_SEMANTICS
from native_io import NativeJointIO
from so101_home_pose import CONTROLLED_JOINTS,FROZEN_JOINTS,START_JOINT_POSE,randomize_joint_start_pose
from experiments.so101.config import TrainConfig
from pose_alignment import leader_to_follower,follower_to_leader


class ThreeJointTests(unittest.TestCase):
    def test_contract_and_fake_training_environment(self):
        self.assertEqual(CONTROLLED_JOINTS,('shoulder_pan','shoulder_lift','elbow_flex'))
        self.assertIn('wrist_flex',FROZEN_JOINTS)
        self.assertEqual(sum(PROPRIO_DIMS.values()),18)
        env=TrainConfig().get_environment(fake_env=True)
        self.assertEqual(env.action_space.shape,(3,))
        self.assertEqual(env.observation_space['state'].shape,(1,18))
        with self.assertRaises(ValueError):bounded_joint_delta([0,0,0,1])

    def test_policy_and_human_share_three_dimensional_executor(self):
        request=np.array([.1,-.3,.2]);zeros=np.zeros(3)
        policy,status=execute_joint_request(zeros,zeros,zeros,request,-np.ones(3),np.ones(3))
        human,hstatus=execute_joint_request(zeros,zeros,zeros,np.array(status['requested_delta_rad'])/np.radians(4),-np.ones(3),np.ones(3))
        np.testing.assert_allclose(policy,human)
        self.assertEqual(len(hstatus['target_rad']),3)

    def test_serial_normal_send_excludes_all_frozen_joints(self):
        io=NativeJointIO.__new__(NativeJointIO);io.lock=threading.RLock();io.last_send={}
        writes=[]
        def send(action):writes.append(action);return action
        io.robots={'follower':SimpleNamespace(send_action=send)}
        io.bounds={'follower':(-np.ones(3),np.ones(3))}
        io.reset_bounds={'follower':(-np.ones(4),np.ones(4))}
        io.send('follower',[.1,.2,.3])
        self.assertEqual(set(writes[-1]),{j+'.pos' for j in CONTROLLED_JOINTS})
        with self.assertRaises(ValueError):io.send('follower',[.1,.2,.3,.4])
        self.assertEqual(len(writes),1)
        io.send_reset_arm([.1,.2,.3,.4])
        self.assertIn('wrist_flex.pos',writes[-1])
        self.assertNotIn('gripper.pos',writes[-1])
        self.assertNotIn('wrist_roll.pos',writes[-1])

    def test_alignment_accepts_only_active_joint_keys(self):
        p=dict(zip(CONTROLLED_JOINTS,[0,1,2]))
        self.assertEqual(set(leader_to_follower(p)),set(p))
        np.testing.assert_allclose(list(follower_to_leader(leader_to_follower(p)).values()),list(p.values()))

    def test_randomization_still_initializes_wrist_pitch(self):
        q,info=randomize_joint_start_pose(np.random.default_rng(1),q_cur=START_JOINT_POSE)
        self.assertTrue(info['ok'])
        self.assertTrue(np.all(np.abs(info['joint_delta_deg'][:3])<=1.5))
        self.assertAlmostEqual(sum(info['joint_delta_deg'][1:4]),0.)

if __name__=='__main__':unittest.main()
