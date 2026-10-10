"""Offline feedback recovery and timing diagnostics; no serial devices opened."""
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from collections import deque
import numpy as np
from experiments.so101 import native_env
from native_io import NativeJointIO


class FeedbackTests(unittest.TestCase):
    def make_env(self):
        env=native_env.NativeSO101Env.__new__(native_env.NativeSO101Env)
        env._native_feedback_error=None
        env._native_last_feedback_error=None
        env._native_feedback_failures=0
        env._native_stop=threading.Event()
        env._joint_stamp=env._leader_stamp=10.
        env._joint_pos=np.zeros(6)
        env._lock=threading.RLock()
        env.action_scale=.07
        logs=[]
        env.node=SimpleNamespace(get_logger=lambda:SimpleNamespace(error=logs.append,info=logs.append))
        env._native_io=SimpleNamespace(diagnostics=lambda:[])
        return env,logs

    def test_transient_failure_recovers_and_keeps_error_history(self):
        env,logs=self.make_env()
        calls=[]
        def refresh():
            calls.append(1)
            if len(calls)==1:raise OSError('mock packet timeout')
            self.assertEqual(env._joint_stamp,10.)
            env._joint_stamp=20.
            env._native_stop.set()
        env._refresh_native=refresh
        env._native_poll()
        self.assertEqual(len(calls),2)
        self.assertIsNone(env._native_feedback_error)
        self.assertIn('mock packet timeout',env._native_last_feedback_error)
        self.assertEqual(env._native_feedback_failures,1)
        self.assertTrue(any('recovered' in line for line in logs))

    def test_persistent_failure_does_not_refresh_stamp(self):
        env,_=self.make_env()
        def refresh():
            if env._native_feedback_failures==2:env._native_stop.set()
            raise OSError('offline')
        env._refresh_native=refresh
        env._native_poll()
        self.assertEqual(env._joint_stamp,10.)
        self.assertEqual(env._native_feedback_failures,3)

    def test_stale_feedback_rejected_before_command(self):
        env,logs=self.make_env()
        with self.assertRaisesRegex(RuntimeError,'No fresh joint feedback'):
            env.step(np.zeros(3))
        self.assertTrue(any('Native control stopped' in line for line in logs))

    def test_serial_write_rechecks_freshness_after_lock_acquisition(self):
        io=NativeJointIO.__new__(NativeJointIO)
        class AgingLock:
            def __enter__(self):
                io._last_read_stamp['follower']=time.monotonic()-1.
            def __exit__(self,*args):pass
        io.lock=AgingLock()
        io._diagnostic_lock=threading.Lock()
        io._io_events=deque(maxlen=64)
        io._last_read_stamp={'follower':time.monotonic()}
        io.bounds={'follower':(np.full(3,-1.),np.full(3,1.))}
        # No robot object: any attempt to actually write would fail this test.
        with self.assertRaisesRegex(RuntimeError,'no command sent'):
            io.send('follower',np.zeros(3))
        self.assertIn('serial write',io.diagnostics()[-1]['error'])

    def test_operation_records_failure_and_releases_lock(self):
        io=NativeJointIO.__new__(NativeJointIO)
        io.lock=threading.RLock()
        io._diagnostic_lock=threading.Lock()
        io._io_events=deque(maxlen=64)
        with self.assertRaises(OSError):
            with io._operation('read','leader'):
                raise OSError('packet failure')
        with io._operation('send','follower'):pass
        events=io.diagnostics()
        self.assertIn('packet failure',events[0]['error'])
        self.assertIsNone(events[1]['error'])
        self.assertGreaterEqual(events[0]['lock_wait_s'],0)
        self.assertGreaterEqual(events[0]['io_duration_s'],0)
        events[0]['error']='changed'
        self.assertIn('packet failure',io.diagnostics()[0]['error'])


if __name__=='__main__':unittest.main()
