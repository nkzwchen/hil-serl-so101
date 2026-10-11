"""Offline tests: reward alignment and the human-label admission boundary."""
import hashlib
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'examples'), str(Path(__file__).resolve().parents[1]/'serl_launcher')]
import numpy as np
from experiments.so101.config import TrainConfig
from experiments.so101.human_reward import load_episode
from coarse_label_demos import coarse_label

class HumanRewardTests(unittest.TestCase):
    def test_derived_success_requires_contiguous_frame_mapping(self):
        self.meta['conversion']=dict(version='10hz_scale55_v1',kind='native10_scale_only',
            source_label=dict(status='success',frame=4,reviewed=True,source='human_review'),
            source_sha256='source-digest',source_ranges=[[0,1],[1,2],[2,3],[3,4]])
        self.lab.update(source='derived_human_review',frame=4,source_recording_sha256='source-digest')
        self.save()
        ts,_=load_episode(self.path,self.cfg)
        self.assertEqual(float(ts[-1]['rewards']),1.)
        self.meta['conversion']['source_ranges'][2]=[2,4]
        self.save()
        with self.assertRaisesRegex(ValueError,'discarded'):
            load_episode(self.path,self.cfg)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'ep.pkl'
        self.cfg = TrainConfig()
        cfg = self.cfg
        env = cfg.get_environment(fake_env=True)
        obs = {k:np.zeros(s.shape,s.dtype) for k,s in env.observation_space.spaces.items()}
        self.ts = [dict(observations=obs,next_observations=obs,actions=np.zeros(3,np.float32),infos={}) for _ in range(4)]
        self.meta = {k:getattr(cfg,v) for k,v in dict(action_semantics='ACTION_SEMANTICS',observation_semantics='OBSERVATION_SEMANTICS',action_scale_rad='ACTION_SCALE',action_joints='ACTION_JOINTS',env_hz='ENV_HZ',control_backend='CONTROL_BACKEND',pose_alignment='POSE_ALIGNMENT').items()}
        self.meta.update(roi_spec=cfg.roi_spec,main_roi_spec=cfg.main_roi_spec)
        self.lab = dict(status='success',frame=3,reviewed=True,source='human_review')
        self.save()

    def save(self):
        self.path.write_bytes(pickle.dumps(dict(transitions=self.ts,meta=self.meta)))
        self.path.with_suffix('.labels.json').write_text(json.dumps(dict(recording_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),episodes={'0':self.lab})))

    def test_success_rewards_preceding_action(self):
        ts,receipt=load_episode(self.path,self.cfg)
        self.assertEqual([float(t['rewards']) for t in ts],[0,0,1])
        self.assertEqual([float(t['masks']) for t in ts],[1,1,0])
        self.assertEqual(receipt['transitions'],3)

    def test_explicit_failure(self):
        self.lab=dict(status='failed',reviewed=True,source='human_review');self.save()
        ts,_=load_episode(self.path,self.cfg)
        self.assertEqual(len(ts),4)
        self.assertTrue(all(t['rewards']==0 for t in ts))
        self.assertTrue(ts[-1]['dones'])
        self.assertEqual(float(ts[-1]['masks']),1.)

    def test_timeout_bootstraps_and_interventions_trim_with_success(self):
        self.lab=dict(status='failed',reviewed=True,source='human_review')
        self.meta['episode_timeout']=True
        self.ts[1]['infos']['intervening']=True
        self.ts[3]['infos']['intervening']=True
        self.save()
        ts,r=load_episode(self.path,self.cfg)
        self.assertEqual(r['termination'],'timeout')
        self.assertEqual(ts[-1]['masks'],1.)
        self.lab=dict(status='success',frame=3,reviewed=True,source='human_review')
        self.save()
        ts,r=load_episode(self.path,self.cfg)
        self.assertEqual(r['intervention_indices'],[1])
        self.assertEqual(ts[-1]['masks'],0.)

    def test_safety_interruption_not_an_absorbing_failure(self):
        self.meta['safety_error']='feedback stale';self.save()
        with self.assertRaisesRegex(ValueError,'Safety-interrupted'):
            load_episode(self.path,self.cfg)

    def test_provisional_opt_in(self):
        self.lab=coarse_label(self.ts);self.save()
        with self.assertRaisesRegex(ValueError,'Provisional'):load_episode(self.path,self.cfg)
        self.assertEqual(len(load_episode(self.path,self.cfg,True)[0]),4)

    def test_missing_or_changed_recording_rejected(self):
        self.path.write_bytes(self.path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'hash'):load_episode(self.path,self.cfg)

    def test_shape_or_nonfinite_action_rejected(self):
        self.ts[0]['actions'][0]=np.nan;self.save()
        with self.assertRaisesRegex(ValueError,'action'):load_episode(self.path,self.cfg)

    def test_zero_frame_excluded_and_gaps_rejected(self):
        self.lab['frame']=0;self.save()
        with self.assertRaises(ValueError):load_episode(self.path,self.cfg)
        self.lab['frame']=3
        for meta in ({'operator_excluded':True},{'recording':{'has_gaps':True}}):
            self.meta.update(meta);self.save()
            with self.assertRaises(ValueError):load_episode(self.path,self.cfg)

if __name__=='__main__':unittest.main()
