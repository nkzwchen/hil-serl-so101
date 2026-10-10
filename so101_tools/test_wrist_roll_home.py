"""Reset-only roll regression tests; all buses are fake."""
import math
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'hil-serl/examples/experiments/so101'))
import native_io
from so101_home_pose import START_JOINT_POSE


class Bus:
    def __init__(self,torque):
        self.calibration={'wrist_roll':SimpleNamespace(range_min=0,range_max=4095)}
        self.motors={'wrist_roll':SimpleNamespace(model='sts3215')}
        self.model_resolution_table={'sts3215':4096}
        self.position=-3.3846
        self.torque=torque
        self.commands=[]
        self.enabled=[]
    def read(self,register,joint,**kwargs):
        assert joint=='wrist_roll'
        return {'Min_Position_Limit':0,'Max_Position_Limit':4095,
                'Present_Position':self.position,'Torque_Enable':self.torque}[register]
    def write(self,register,joint,value,**kwargs):
        assert register=='Goal_Position' and joint=='wrist_roll'
        self.commands.append(value)
        self.position=value
    def enable_torque(self,joints,**kwargs):
        self.enabled.append(joints)
        self.torque=1


class RollHomeTests(unittest.TestCase):
    def run_home(self,torque=1,target=START_JOINT_POSE[4]):
        bus=Bus(torque)
        io=native_io.NativeJointIO.__new__(native_io.NativeJointIO)
        io.lock=threading.RLock()
        io.robots={'follower':SimpleNamespace(bus=bus)}
        clock=[0.]
        def now():
            clock[0]+=.005
            return clock[0]
        def sleep(seconds):clock[0]+=seconds
        with patch.object(native_io,'time',SimpleNamespace(monotonic=now,sleep=sleep)):
            result=io.home_wrist_roll(target)
        return bus,result

    def test_only_roll_moves_slowly_to_nominal(self):
        bus,result=self.run_home()
        self.assertAlmostEqual(bus.commands[-1],math.degrees(START_JOINT_POSE[4]))
        self.assertLess(max(np.diff(bus.commands)),.5)
        self.assertEqual(result['phase'],'reset_only')
        self.assertFalse(bus.enabled)

    def test_off_roll_enabled_at_present_position(self):
        bus,_=self.run_home(torque=0)
        self.assertAlmostEqual(bus.commands[0],-3.3846)
        self.assertEqual(bus.enabled,[['wrist_roll']])

    def test_invalid_target_sends_nothing(self):
        bus=Bus(0)
        io=native_io.NativeJointIO.__new__(native_io.NativeJointIO)
        io.lock=threading.RLock();io.robots={'follower':SimpleNamespace(bus=bus)}
        with self.assertRaisesRegex(RuntimeError,'outside firmware limits'):
            io.home_wrist_roll(math.radians(200))
        self.assertFalse(bus.commands)
        self.assertFalse(bus.enabled)

    def test_stalled_roll_stops_further_correction(self):
        bus=Bus(1)
        bus.write=lambda register,joint,value,**kwargs:bus.commands.append(value)
        io=native_io.NativeJointIO.__new__(native_io.NativeJointIO)
        io.lock=threading.RLock();io.robots={'follower':SimpleNamespace(bus=bus)}
        clock=[0.]
        def now():clock[0]+=.1;return clock[0]
        with patch.object(native_io,'time',SimpleNamespace(monotonic=now,sleep=lambda _:None)):
            with self.assertRaisesRegex(RuntimeError,'tracking lag'):
                io.home_wrist_roll(math.radians(20))
        self.assertLess(max(bus.commands)-bus.position,8.)


if __name__=='__main__':unittest.main()
