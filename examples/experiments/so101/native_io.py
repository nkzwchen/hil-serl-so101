"""Exclusive LeRobot joint I/O. ROS is only used for images in native mode."""
import math
import os
from pathlib import Path
import threading
import time
import numpy as np
from pose_alignment import check_calibration_file
from so101_home_pose import JOINT_NAMES, CONTROLLED_JOINTS

PORTS = dict(leader='/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6083913-if00',
             follower='/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6058742-if00')
CALIBRATION = Path('/home/nkzwchen/.cache/huggingface/calibration')


def require_free_ports():
    devices={os.path.realpath(p) for p in PORTS.values()}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            for fd in (proc/'fd').iterdir():
                if os.path.realpath(fd) in devices:
                    raise RuntimeError(f'Serial port occupied by PID {proc.name}; stop arm bridges/follow tests first')
        except (PermissionError, FileNotFoundError, ProcessLookupError):continue


class NativeJointIO:
    def __init__(self):
        require_free_ports()
        from lerobot.robots.so_follower.so_follower import SOFollower
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        from lerobot.teleoperators.so_leader.so_leader import SOLeader
        from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
        self.lock=threading.RLock()
        self.robots=dict(
            leader=SOLeader(SOLeaderTeleopConfig(port=PORTS['leader'],id='so101_leader_main',
                calibration_dir=CALIBRATION/'teleoperators/so_leader',use_degrees=True)),
            follower=SOFollower(SOFollowerRobotConfig(port=PORTS['follower'],id='so101_follower_main',
                calibration_dir=CALIBRATION/'robots/so_follower',use_degrees=True,
                max_relative_target=None,disable_torque_on_disconnect=False)))
        self.initial_torque={}
        self.bounds={}
        self.reset_bounds={}
        self.last_send={}
        try:
            for arm,robot in self.robots.items():
                check_calibration_file(robot.calibration_fpath,arm)
                robot.bus.connect()  # No configure(), calibration or gain writes.
                self.initial_torque[arm]=robot.bus.sync_read('Torque_Enable',normalize=False)
                actual=robot.bus.read_calibration()
                lo=[];hi=[]
                for j in JOINT_NAMES[:4]:
                    c=robot.calibration[j];mid=(c.range_min+c.range_max)/2
                    lo.append(math.radians((actual[j].range_min-mid)*360/4095))
                    hi.append(math.radians((actual[j].range_max-mid)*360/4095))
                self.reset_bounds[arm]=(np.asarray(lo),np.asarray(hi))
                self.bounds[arm]=(np.asarray(lo[:3]),np.asarray(hi[:3]))
        except Exception:
            self.close()
            raise

    def read(self,arm):
        with self.lock:
            robot=self.robots[arm]
            raw=robot.bus.sync_read('Present_Position',normalize=False)
            values=robot.bus._normalize({robot.bus.motors[j].id:v for j,v in raw.items()})
            norm={j:values[robot.bus.motors[j].id] for j in JOINT_NAMES}
            q=np.asarray([math.radians(norm[j]) if j!='gripper' else norm[j]*math.pi/100 for j in JOINT_NAMES])
            return q,time.monotonic(),raw

    def home_wrist_roll(self, target_rad):
        """Reset-only exception: slowly home follower roll, never the gripper."""
        from so101_home_pose import min_jerk
        joint='wrist_roll'
        target=float(target_rad)
        with self.lock:
            bus=self.robots['follower'].bus
            cal=bus.calibration[joint]
            midpoint=(cal.range_min+cal.range_max)/2
            resolution=bus.model_resolution_table[bus.motors[joint].model]-1
            low=(bus.read('Min_Position_Limit',joint,normalize=False)-midpoint)*2*math.pi/resolution
            high=(bus.read('Max_Position_Limit',joint,normalize=False)-midpoint)*2*math.pi/resolution
            start=math.radians(bus.read('Present_Position',joint))
            if not all(math.isfinite(v) for v in (start,target,low,high)) or not low<=start<=high or not low<=target<=high:
                raise RuntimeError('Reset wrist_roll start/target outside firmware limits; no command sent')
            if not int(bus.read('Torque_Enable',joint,normalize=False)):
                bus.write('Goal_Position',joint,math.degrees(start))
                bus.enable_torque([joint],num_retry=2)
            if int(bus.read('Torque_Enable',joint,normalize=False))!=1:
                raise RuntimeError('Reset wrist_roll torque enable failed')
        displacement=math.degrees(abs(target-start))
        duration=max(.6,1.875*displacement/10.,math.sqrt(5.774*displacement/30.))
        begin=time.monotonic()
        while True:
            tick=time.monotonic()
            fraction=min(1.,(tick-begin)/duration)
            command=start+(target-start)*min_jerk(fraction)
            with self.lock:
                measured=math.radians(bus.read('Present_Position',joint))
                if not math.isfinite(measured) or abs(command-measured)>math.radians(8):
                    raise RuntimeError('Reset wrist_roll tracking lag; no further correction')
                bus.write('Goal_Position',joint,math.degrees(command))
            if fraction>=1:break
            time.sleep(max(0,1/30.-(time.monotonic()-tick)))
        return {'target_deg':math.degrees(target),'start_deg':math.degrees(start),
                'duration_s':duration,'phase':'reset_only'}

    def send(self,arm,target):
        return self._send(arm,target,CONTROLLED_JOINTS,self.bounds[arm])

    def enable_reset_wrist_flex(self):
        with self.lock:
            bus=self.robots['follower'].bus
            joint='wrist_flex'
            if not int(bus.read('Torque_Enable',joint,normalize=False)):
                current=bus.read('Present_Position',joint)
                bus.write('Goal_Position',joint,current)
                bus.enable_torque([joint],num_retry=2)
            if int(bus.read('Torque_Enable',joint,normalize=False))!=1:
                raise RuntimeError('Reset wrist_flex torque enable failed')

    def send_reset_arm(self,target):
        return self._send('follower',target,JOINT_NAMES[:4],self.reset_bounds['follower'])

    def _send(self,arm,target,joints,bounds):
        q=np.asarray(target,float)
        if q.shape!=(len(joints),) or not np.isfinite(q).all():raise ValueError('Finite joint targets of the correct dimension required')
        lo,hi=bounds
        outside=(q<lo-1e-8) | (q>hi+1e-8)
        if np.any(outside):
            details='; '.join(f'{joints[i]}={math.degrees(q[i]):.4f}deg '
                             f'allowed=[{math.degrees(lo[i]):.4f}, {math.degrees(hi[i]):.4f}]deg'
                             for i in np.flatnonzero(outside))
            raise ValueError(arm+' target outside firmware limits: '+details)
        with self.lock:
            robot=self.robots[arm]
            action={j+'.pos':math.degrees(q[i]) for i,j in enumerate(joints)}
            started=time.monotonic()
            if arm=='follower':sent=robot.send_action(action)
            else:
                robot.bus.sync_write('Goal_Position',{j:action[j+'.pos'] for j in joints})
                sent=action
            result=np.asarray([math.radians(sent[j+'.pos']) for j in joints])
            self.last_send[arm]=dict(requested_target_rad=q.tolist(),sent_target_rad=result.tolist(),
                                     started_monotonic_s=started,finished_monotonic_s=time.monotonic())
            return result

    def torque(self,arm,on):
        with self.lock:
            bus=self.robots[arm].bus
            state=bus.sync_read('Torque_Enable',normalize=False)
            if on:
                off=[j for j in CONTROLLED_JOINTS if int(state[j])!=1]
                if off:
                    cur=bus.sync_read('Present_Position')
                    bus.sync_write('Goal_Position',{j:cur[j] for j in off})
                    bus.enable_torque(off,num_retry=2)
            else:bus.disable_torque(CONTROLLED_JOINTS,num_retry=2)
            confirmed=bus.sync_read('Torque_Enable',normalize=False)
            return all(int(confirmed[j])==int(on) for j in CONTROLLED_JOINTS)

    def close(self):
        with self.lock:
            for robot in self.robots.values():
                if robot.bus.is_connected:robot.bus.disconnect(disable_torque=False)
