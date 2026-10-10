"""Regression tests for native reset ramps; never opens serial ports."""
import math
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project / 'hil-serl/examples'))
sys.path.insert(0, str(project / 'hil-serl/examples/experiments/so101'))
from experiments.so101 import native_env
from so101_home_pose import JOINT_NAMES
RESET_JOINTS = JOINT_NAMES[:4]


class ReturnTests(unittest.TestCase):
    def make_env(self, goal, measured=0., stale=False):
        env = native_env.NativeSO101Env.__new__(native_env.NativeSO101Env)
        env._lock = threading.RLock()
        env._joint_pos = np.full(6, measured)
        env._joint_stamp = 99. if stale else 100.
        env._action_low, env._action_high = np.full(4, -.5), np.full(4, .5)
        commands, warnings = [], []
        def send(arm, command):
            self.assertTrue(np.all(np.abs(command) <= .5 + 1e-8))
            commands.append(command.copy())
            env._joint_pos[:4] = command
        bus = SimpleNamespace(sync_read=lambda _: dict(zip(RESET_JOINTS, np.degrees(np.full(4, goal)))))
        env._native_io = SimpleNamespace(lock=threading.RLock(), robots={'follower': SimpleNamespace(bus=bus)},
                                         reset_bounds={'follower': (env._action_low, env._action_high)},
                                         send_reset_arm=lambda command: send('follower', command),
                                         enable_reset_wrist_flex=lambda: None)
        env.node = SimpleNamespace(get_logger=lambda: SimpleNamespace(warn=warnings.append))
        clock = [100.]
        def now():
            result = clock[0]
            clock[0] += .05
            if not stale:
                env._joint_stamp = result
            return result
        fake_time = SimpleNamespace(monotonic=now, sleep=lambda _: None)
        return env, commands, warnings, fake_time

    def test_invalid_stored_goal_uses_fresh_legal_feedback(self):
        env, commands, warnings, clock = self.make_env(goal=1.)
        with patch.object(native_env, 'time', clock):
            env._publish_arm_ramp(np.full(4, .1))
        self.assertTrue(warnings)
        self.assertLess(commands[0].max(), .02)
        np.testing.assert_allclose(commands[-1], .1)

    def test_invalid_goal_and_stale_feedback_send_nothing(self):
        env, commands, _, clock = self.make_env(goal=1., stale=True)
        with patch.object(native_env, 'time', clock):
            with self.assertRaisesRegex(RuntimeError, 'stale feedback'):
                env._publish_arm_ramp(np.zeros(4))
        self.assertFalse(commands)

    def test_invalid_goal_and_invalid_feedback_send_nothing(self):
        env, commands, _, clock = self.make_env(goal=1., measured=.7)
        with patch.object(native_env, 'time', clock):
            with self.assertRaisesRegex(RuntimeError, 'measured position outside firmware limits'):
                env._publish_arm_ramp(np.zeros(4))
        self.assertFalse(commands)

    def test_valid_goal_retains_static_holding_offset(self):
        env, commands, warnings, clock = self.make_env(goal=.05)
        with patch.object(native_env, 'time', clock):
            env._publish_arm_ramp(np.full(4, .1))
        self.assertFalse(warnings)
        self.assertGreaterEqual(commands[0].min(), .05)
        np.testing.assert_allclose(commands[-1], .1)


if __name__ == '__main__':
    unittest.main()
