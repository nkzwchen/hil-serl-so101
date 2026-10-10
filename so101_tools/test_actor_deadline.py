"""No hardware: timeout prevents late actions and preserves the completed prefix."""
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

class DeadlineTests(unittest.TestCase):
    def test_monotonic_boundary(self):
        with patch.object(recorder.time, 'monotonic', return_value=19.999):
            self.assertFalse(recorder.actor_deadline_expired(20.))
        with patch.object(recorder.time, 'monotonic', return_value=20.):
            self.assertTrue(recorder.actor_deadline_expired(20.))
            self.assertFalse(recorder.actor_deadline_expired(None))

    def test_slow_inference_action_is_not_sent(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            def policy(obs):
                calls.append(1)
                return np.zeros(3,np.float32)
            with patch.object(config, 'SO101Env', flow.StubEnv), \
                 patch.object(recorder, 'NonBlockingKey', return_value=flow.StubKeys(['c','c'])), \
                 patch.object(recorder, 'actor_deadline_expired', side_effect=lambda d:d is not None and len(calls)>=2), \
                 patch.object(sys, 'argv', ['record', '--out',directory,'--episodes','1']), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(recorder.main(policy=policy),0)
            files=list(Path(directory).glob('*/ep_*.pkl'))
            self.assertEqual(len(files),1)
            with files[0].open('rb') as f: data=pickle.load(f)
            self.assertEqual(len(data['transitions']),1)
            self.assertTrue(data['meta']['episode_timeout'])
            self.assertTrue(data['meta']['truncated'])
            self.assertEqual(data['meta']['episode_time_limit_s'],20.)
            self.assertTrue(data['transitions'][-1]['dones'])
            self.assertEqual(data['transitions'][-1]['rewards'],0.)

if __name__=='__main__':unittest.main()
