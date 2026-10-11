"""Offline regression: keys arriving during inference precede robot commands."""
import contextlib
import io
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import test_record_flow as flow
import record_demo_so101 as recorder
from experiments.so101 import config

class TrainingKeyTests(unittest.TestCase):
    def run_key(self, key):
        phase=['']; queued=[]; executed=[]; calls=[]
        class Env(flow.StubEnv):
            def step(self, action):
                if phase[0]=='RECORD':executed.append(self.is_intervening)
                return super().step(action)
        def panel(kb,p,detail=''):phase[0]=p
        def poll(kb,accept):
            if phase[0] in ('READY','RETURN'):return 'c'
            if phase[0]=='RECORD':
                if queued:return queued.pop(0)
                if executed:return 'd'
            return None
        def policy(obs):
            calls.append(time.monotonic())
            queued.append(key)
            time.sleep(.05)  # Simulate a key arriving in a blocking policy call.
            return np.zeros(3,np.float32)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(config,'SO101Env',Env), \
             patch.object(recorder,'NonBlockingKey',return_value=flow.StubKeys([])), \
             patch.object(recorder,'panel_phase',side_effect=panel), \
             patch.object(recorder,'poll',side_effect=poll), \
             patch.object(sys,'argv',['record','--terminal-controls','--out',directory,'--episodes','1']), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(recorder.main(policy=policy),0)
        self.assertEqual(len(calls),1)
        return executed
    def test_during_inference_stop_sends_no_record_action(self):
        self.assertEqual(self.run_key('d'),[])
    def test_during_inference_takeover_sends_only_human_action(self):
        self.assertEqual(self.run_key(' '),[True])

if __name__=='__main__':unittest.main()
