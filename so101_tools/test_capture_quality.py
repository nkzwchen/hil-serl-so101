"""Offline regression tests. No ROS node, serial port or motor is created."""
import ast
import math
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
REPO = HERE / 'hil-serl' if (HERE / 'hil-serl').is_dir() else HERE.parent
MODULE = REPO / 'examples/experiments/so101'
sys.path.insert(0, str(MODULE))
from camera_observation import decode_rgb, crop_roi, validate_roi, CAMERA_KEYS
from latest_frame import LatestFrameWorker

# Compile actual production methods without importing or constructing ROS nodes.
tree = ast.parse((MODULE / 'env.py').read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SO101Env')
names = {'_on_image', '_process_image', '_image_worker_error', '_quality_locked', 'observation_quality', '_get_obs',
         'last_observation_quality', '_settled_joint_position', '_publish_arm_ramp'}
methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
for method in methods:
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
ns = dict(np=np, cv2=cv2, time=time, math=math, decode_rgb=decode_rgb, crop_roi=crop_roi,
          N_ARM=5, FROZEN_IDX=(4,), ARM_JOINTS=list('abcde'),
          JOINT_LOW=np.full(5, -2.), JOINT_HIGH=np.full(5, 2.),
          min_jerk=lambda x: 10*x**3 - 15*x**4 + 6*x**5,
          JointTrajectory=lambda: SimpleNamespace(header=SimpleNamespace(), points=[]),
          JointTrajectoryPoint=lambda: SimpleNamespace(time_from_start=SimpleNamespace()))
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(
    name='Harness', bases=[], keywords=[], body=methods, decorator_list=[])],
    type_ignores=[])), str(MODULE / 'env.py'), 'exec'), ns)


def msg(rgb, stamp=1, padding=0, encoding='rgb8'):
    h, w = rgb.shape[:2]
    rows = np.full((h, w * 3 + padding), 255, np.uint8)
    rows[:, :w * 3] = rgb.reshape(h, -1)
    return SimpleNamespace(height=h, width=w, step=rows.shape[1], encoding=encoding,
                           data=rows.tobytes(), header=SimpleNamespace(
                               stamp=SimpleNamespace(sec=stamp, nanosec=0)))


def harness():
    obj = ns['Harness']()
    obj._lock = threading.Lock()
    obj._images = {}
    obj.img_size = 8
    obj.roi_spec = None
    obj.main_roi_spec = None
    obj.image_keys = list(CAMERA_KEYS)
    obj.display_image = False
    obj._joint_pos = obj._joint_vel = np.zeros(6)
    obj._joint_stamp = obj._leader_stamp = time.monotonic()
    obj._camera_stats = {k: dict(received=0, accepted=0, errors=0, arrival=0.,
        source_stamp_ns=0, last_error='', valid=False, first_arrival=0., max_gap_s=0.,
        pending_replaced=0, duplicate_messages=0, processing_expired=0, processing_delay_s=0.)
        for k in CAMERA_KEYS}
    obj.node = SimpleNamespace(get_logger=lambda: SimpleNamespace(error=lambda x: None))
    class ImmediateWorker:
        def __init__(self, key): self.key = key
        def submit(self, message, arrival):
            obj._process_image(message, self.key, arrival)
            return False
    obj._image_workers = {key: ImmediateWorker(key) for key in CAMERA_KEYS}
    return obj


class QualityTests(unittest.TestCase):
    def test_stride_and_color(self):
        rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
        np.testing.assert_array_equal(decode_rgb(msg(rgb, padding=8)), rgb)
        np.testing.assert_array_equal(decode_rgb(msg(rgb[:, :, ::-1], padding=8,
                                                   encoding='bgr8')), rgb)
        broken = msg(rgb); broken.data = broken.data[:-1]
        with self.assertRaises(ValueError): decode_rgb(broken)
        with self.assertRaises(ValueError): decode_rgb(msg(rgb, encoding='8UC3'))

    def test_missing_stale_and_repeated(self):
        obj = harness(); frame = np.zeros((8, 8, 3), np.uint8)
        obj._on_image(msg(frame), 'cam_main')
        self.assertFalse(obj.observation_quality()['ok'])  # missing wrist
        obj._on_image(msg(frame), 'cam_wrist')
        self.assertTrue(obj.observation_quality()['ok'])  # stationary scene is valid
        obj._on_image(msg(frame), 'cam_main')
        self.assertTrue(obj.observation_quality()['ok'])
        self.assertEqual(obj._camera_stats['cam_main']['errors'], 0)
        self.assertEqual(obj._camera_stats['cam_main']['duplicate_messages'], 1)
        obj._on_image(msg(frame, stamp=2), 'cam_main')
        obj._on_image(msg(frame, stamp=2), 'cam_wrist')
        self.assertTrue(obj.observation_quality()['ok'])
        obj._camera_stats['cam_main']['arrival'] -= 1
        self.assertFalse(obj.observation_quality()['ok'])

    def test_skew_is_warning_and_backwards_is_error(self):
        obj = harness(); frame = np.zeros((8,8,3), np.uint8)
        obj._on_image(msg(frame, stamp=1), 'cam_main')
        obj._on_image(msg(frame, stamp=2), 'cam_wrist')
        q = obj.observation_quality()
        self.assertTrue(q['ok'])
        self.assertFalse(q['camera_sync_required'])
        self.assertTrue(q['warnings'])
        obj._on_image(msg(frame, stamp=1), 'cam_wrist')
        self.assertFalse(obj.observation_quality()['ok'])
        self.assertEqual(obj._camera_stats['cam_wrist']['errors'], 1)

    def test_processing_delay_never_refreshes_expired_image(self):
        obj = harness(); frame = np.zeros((8,8,3), np.uint8)
        obj._on_image(msg(frame, stamp=1), 'cam_main')
        arrival = obj._camera_stats['cam_main']['arrival']
        obj._process_image(msg(frame, stamp=2), 'cam_main', time.monotonic()-.3)
        self.assertEqual(obj._camera_stats['cam_main']['arrival'], arrival)
        self.assertEqual(obj._camera_stats['cam_main']['source_stamp_ns'], 1_000_000_000)
        self.assertEqual(obj._camera_stats['cam_main']['processing_expired'], 1)
        with patch.object(ns['cv2'], 'resize', side_effect=lambda *a, **k: frame):
            with patch.object(time, 'monotonic', side_effect=[100., 100.3]):
                obj._process_image(msg(frame, stamp=2), 'cam_main', 100.)
        self.assertEqual(obj._camera_stats['cam_main']['arrival'], arrival)
        self.assertEqual(obj._camera_stats['cam_main']['processing_expired'], 2)

    def test_latest_workers_replace_pending_and_are_independent(self):
        blocked = threading.Event(); release = threading.Event()
        latest = threading.Event(); other_done = threading.Event()
        processed, errors = [], []
        def main_callback(value, arrival):
            if value == 0:
                blocked.set()
                release.wait(2.)
            processed.append((value, arrival))
            if value == 2: latest.set()
        main = LatestFrameWorker(main_callback, 'test_main', errors.append)
        wrist = LatestFrameWorker(lambda *args: other_done.set(), 'test_wrist', errors.append)
        try:
            self.assertFalse(main.submit(0,10.))
            self.assertTrue(blocked.wait(1.))
            self.assertFalse(main.submit(1,11.))
            self.assertTrue(main.submit(2,12.))
            wrist.submit(3,13.)
            self.assertTrue(other_done.wait(1.))
            release.set()
            self.assertTrue(latest.wait(1.))
            self.assertEqual(processed, [(0,10.),(2,12.)])
            self.assertEqual(errors, [])
        finally:
            release.set(); main.close(); wrist.close()
        self.assertFalse(main._thread.is_alive())
        self.assertFalse(wrist._thread.is_alive())

    def test_roi_atomic_and_resolution(self):
        spec = dict(version=1, camera='cam_main', source_size=[8, 8],
                    xyxy=[2, 2, 6, 4], preprocess='letterbox_rgb_v1')
        obj = harness(); obj.roi_spec = spec
        obj.image_keys.append('cam_main_roi')
        frame = np.full((8, 8, 3), 37, np.uint8)
        obj._on_image(msg(frame), 'cam_main')
        crop = obj._images['cam_main_roi']
        self.assertEqual(crop.shape, (8, 8, 3))
        self.assertTrue((crop[2:6] == 37).all())
        self.assertTrue((crop[:2] == 0).all())
        old = obj._images['cam_main'].copy()
        obj._on_image(msg(np.ones((9, 8, 3), np.uint8), stamp=2), 'cam_main')
        self.assertFalse(obj._camera_stats['cam_main']['valid'])
        np.testing.assert_array_equal(old, obj._images['cam_main'])
        with self.assertRaises(ValueError): validate_roi({**spec, 'xyxy': [2, 2, 2, 4]})

    def test_independent_main_and_detail_crops(self):
        obj = harness()
        frame = np.zeros((8, 8, 3), np.uint8)
        frame[:, :4] = [255, 0, 0]
        frame[:, 4:] = [0, 255, 0]
        base = dict(version=1, camera='cam_main', source_size=[8, 8],
                    preprocess='letterbox_rgb_v1')
        obj.main_roi_spec = {**base, 'xyxy': [0, 0, 4, 8]}
        obj.roi_spec = {**base, 'xyxy': [4, 0, 8, 8]}
        obj._on_image(msg(frame), 'cam_main')
        np.testing.assert_array_equal(obj._images['cam_main'][4, 4], [255, 0, 0])
        np.testing.assert_array_equal(obj._images['cam_main_roi'][4, 4], [0, 255, 0])

    def test_home_duration_and_limits(self):
        obj = harness()
        obj.joint_pos = np.zeros(6)
        sent = []
        obj.node = SimpleNamespace(get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: None)))
        obj.traj_pub = SimpleNamespace(publish=sent.append)
        duration = obj._publish_arm_ramp([1.5, 0., 0., 0., 0.])
        self.assertGreater(duration, 3.)
        self.assertLessEqual(1.875 * math.degrees(1.5) / duration, 30. + 1e-9)
        self.assertLessEqual(5.774 * math.degrees(1.5) / duration**2, 60. + 1e-9)
        np.testing.assert_allclose(sent[0].points[-1].positions, [1.5, 0, 0, 0, 0])
        obj._publish_arm_ramp([8., 0., 0., 0., 0.])
        self.assertEqual(sent[-1].points[-1].positions[0], 2.)

    def test_fresh_stable_home_feedback(self):
        obj = harness(); obj.start_settle_s = 0.
        done = threading.Event()
        def update():
            while not done.is_set():
                with obj._lock:
                    obj._joint_stamp = time.monotonic()
                time.sleep(0.01)
        thread = threading.Thread(target=update)
        thread.start()
        try:
            np.testing.assert_array_equal(obj._settled_joint_position(), np.zeros(5))
        finally:
            done.set(); thread.join()

    def test_stale_home_feedback_rejected(self):
        obj = harness(); obj.start_settle_s = 1.
        obj._joint_stamp -= 1
        with self.assertRaises(RuntimeError): obj._settled_joint_position()


if __name__ == '__main__':
    unittest.main()
