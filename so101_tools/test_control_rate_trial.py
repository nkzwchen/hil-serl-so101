"""Offline integration check: 10 Hz override is recorded without changing training defaults."""
import contextlib
import io
import pickle
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import test_record_flow as flow
import record_demo_so101 as recorder
from experiments.so101 import config

class ControlRateTrialTests(unittest.TestCase):
    def test_demo_cli_rate_dry_run(self):
        instances=[]
        class Env(flow.StubEnv):
            def __init__(self,*a,**kw):
                super().__init__(*a,**kw);instances.append(self)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(config,'SO101Env',Env), \
             patch.object(recorder,'NonBlockingKey',return_value=flow.StubKeys([])), \
             patch.object(sys,'argv',['record','--terminal-controls','--dry-run','--control-hz','10','--out',directory]), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(recorder.main(),0)
        self.assertEqual(instances[0].hz,10)
        self.assertEqual(instances[0].max_episode_steps,100)
        self.assertEqual(config.TrainConfig.ENV_HZ,10)

    def test_trial_rate_metadata_and_deadline(self):
        cycles=[];calls=[];instances=[]
        class Env(flow.StubEnv):
            def __init__(self,*a,**kw):
                super().__init__(*a,**kw);instances.append(self)
            def begin_control_cycle(self):cycles.append(1)
        def policy(obs):
            calls.append(1)
            return np.zeros(3,np.float32)
        original=config.TrainConfig.ENV_HZ
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(config,'SO101Env',Env), \
                 patch.object(recorder,'NonBlockingKey',return_value=flow.StubKeys(['c','c'])), \
                 patch.object(recorder,'actor_deadline_expired',side_effect=lambda d:d is not None and len(calls)>=2), \
                 patch.object(sys,'argv',['record','--terminal-controls','--out',directory,'--episodes','1']), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(recorder.main(policy=policy,control_hz=10),0)
            with next(Path(directory).glob('*/ep_*.pkl')).open('rb') as f:data=pickle.load(f)
            self.assertEqual(data['meta']['env_hz'],10)
            self.assertEqual(instances[0].hz,10)
            self.assertEqual(instances[0].max_episode_steps,200)
            self.assertEqual(config.TrainConfig.ENV_HZ,original)
            self.assertEqual(len(cycles),2)
            self.assertFalse(list(Path(directory).glob('*/ep_*.delivery.json')))

if __name__=='__main__':unittest.main()
