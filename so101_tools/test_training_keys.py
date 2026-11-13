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
        phase=['']; queued=[]; executed=[]; calls=[]; paused_polls=[]
        class Env(flow.StubEnv):
            def step(self, action):
                if phase[0]=='RECORD':executed.append(self.is_intervening)
                return super().step(action)
        def panel(kb,p,detail=''):phase[0]=p
        def poll(kb,accept):
            if phase[0] in ('READY','RETURN'):return 'c'
            if phase[0]=='TAKEOVER_PAUSE':
                self.assertEqual(executed, [])
                paused_polls.append(1)
                return ' ' if len(paused_polls) == 5 else None
            if phase[0]=='RECORD':
                if queued:return queued.pop(0)
                if executed:return 'd'
            return None
        def policy(obs):
            calls.append(time.monotonic())
            queued.append(key)
            time.sleep(.05)  # Simulate a key arriving in a blocking policy call.
            return np.zeros(3,np.float32)
        refresh_phases = []
        policy.before_episode = lambda: refresh_phases.append(phase[0])
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(config,'SO101Env',Env), \
             patch.object(recorder,'NonBlockingKey',return_value=flow.StubKeys([])), \
             patch.object(recorder,'panel_phase',side_effect=panel), \
             patch.object(recorder,'poll',side_effect=poll), \
             patch.object(sys,'argv',['record','--terminal-controls','--out',directory,'--episodes','1']), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(recorder.main(policy=policy),0)
        self.assertEqual(refresh_phases, [''])  # Before CHECK/READY and window use.
        self.assertEqual(len(calls),1)
        return executed
    def test_second_space_refreshes_anchor_even_if_previous_mode_human(self):
        from unittest.mock import Mock
        from experiments.so101.env import SO101Env, N_ARM
        env = object.__new__(SO101Env)
        env._joint_stamp = env._leader_stamp = time.monotonic()
        import threading
        env._lock = threading.Lock()
        env._joint_pos = np.zeros(6)
        env._leader_pos = np.ones(6) * .12
        env._last_mode = 'human'
        env._command_hold = Mock()
        env._command_hold.target.return_value = np.ones(N_ARM) * .2
        env._command_hold.offset = np.zeros(N_ARM)
        env.set_leader_torque = Mock(return_value=True)
        env.leader_pose_in_follower_coordinates = lambda q: q
        env.node = Mock()
        env.start_human_intervention()
        np.testing.assert_allclose(env._lead_anchor, .12)
        np.testing.assert_allclose(env._foll_anchor, .2)
        self.assertTrue(env.is_intervening)
        env._leader_stamp -= 1
        with self.assertRaises(RuntimeError):
            env.start_human_intervention()

    def test_during_inference_stop_sends_no_record_action(self):
        self.assertEqual(self.run_key('d'),[])
    def test_during_inference_takeover_sends_only_human_action(self):
        self.assertEqual(self.run_key(' '),[True])

if __name__=='__main__':unittest.main()
