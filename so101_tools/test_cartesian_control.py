#!/usr/bin/env python3
"""Offline checks against the real URDF and production step(); no robot nodes."""
import ast
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace, MethodType
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE / 'hil-serl' if (HERE / 'hil-serl').is_dir() else HERE.parent
MODULE = REPO / 'examples/experiments/so101'
sys.path.insert(0, str(MODULE))
from cartesian_control import CommandHold, LocalCartesianIK, bounded_delta, limit_delta
from so101_kin import SO101Kin
from so101_home_pose import START_JOINT_POSE, Q_LOW as JOINT_LOW, Q_HIGH as JOINT_HIGH, hold_frozen


class CartesianTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kin = SO101Kin(tcp_offset=0.)
        cls.ik = LocalCartesianIK(cls.kin)
        cls.q = np.asarray(START_JOINT_POSE[:5]).copy()

    def test_vector_limit_and_validation(self):
        self.assertAlmostEqual(np.linalg.norm(bounded_delta([1, 1, 1])), .002)
        np.testing.assert_allclose(limit_delta([.004, .001, 0], .002), np.array([.004, .001, 0]) * .002 / np.hypot(.004, .001))
        for bad in ([1, 2], [np.nan, 0, 0], [np.inf, 0, 0]):
            with self.assertRaises(ValueError):
                bounded_delta(bad)

    def test_real_urdf_small_steps_and_continuity(self):
        rng = np.random.default_rng(19)
        accepted = 0
        for _ in range(100):
            q0 = self.q.copy()
            jitter = rng.uniform(-.02, .02, 3)
            q0[:3] += jitter
            q0[3] -= jitter[1] + jitter[2]
            delta = bounded_delta(rng.uniform(-1, 1, 3))
            q, status = self.ik.solve(q0, delta)
            if not status['accepted']:
                np.testing.assert_array_equal(q, q0)
                continue
            accepted += 1
            self.assertLessEqual(np.max(np.abs(q-q0)), np.radians(2) + 1e-12)
            self.assertEqual(q[4], q0[4])
            self.assertLessEqual(self.ik.tilt(q), np.radians(3))
            actual = self.kin.fk(q)[0] - self.kin.fk(q0)[0]
            np.testing.assert_allclose(actual, status['applied_delta_m'], atol=2e-5)
            self.assertTrue(np.all(q >= self.kin.q_low) and np.all(q <= self.kin.q_high))
        self.assertEqual(accepted, 100)

    def test_reject_bad_tilt_and_unreachable(self):
        q = self.q.copy(); q[3] += .2
        result, status = self.ik.solve(q, [.001, 0, 0])
        self.assertFalse(status['accepted'])
        np.testing.assert_array_equal(result, q)
        result, status = self.ik.solve(self.q, [10., 0, 0])
        self.assertFalse(status['accepted'])
        np.testing.assert_array_equal(result, self.q)

    def test_exhausted_ik_budget_holds(self):
        ik = LocalCartesianIK(self.kin, time_budget_s=1e-12)
        result, status = ik.solve(self.q, [.001, 0, 0])
        self.assertFalse(status['accepted'])
        self.assertEqual(status['reason'], 'ik_time_budget')
        np.testing.assert_array_equal(result, self.q)







    def test_production_reset_metadata_for_all_start_modes(self):
        # Execute the production reset with simulated feedback and publishers.
        tree = ast.parse((MODULE / 'env.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SO101Env')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'reset')
        recorder = ast.parse((HERE / 'record_demo_so101.py').read_text())
        contract = next(n.value for n in recorder.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'START_INFO_KEYS' for t in n.targets))
        required = ast.literal_eval(contract)
        class Base:
            def reset(self, **kwargs):
                pass
        cases = [(mode, 0., 2., True)
                 for mode in ('fixed', 'random_ok', 'random_rejected')]
        cases += [('fixed', .39, 2., True), ('fixed', .44, 1.63, True),
                  ('fixed', .48, 2., True), ('fixed', .50, 2., True),
                  ('fixed', 1., 2., True), ('fixed', 3., 2., True), ('fixed', 3.01, 2., False), ('fixed', .44, 3.01, False)]
        acceptance_limit = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == 'START_ACCEPTANCE_TOL'
                                    for t in n.targets))
        acceptance_limit = eval(compile(ast.Expression(acceptance_limit), '<acceptance limit>', 'eval'),
                            {'math': math})
        for mode, residual_deg, tilt_deg, accepted in cases:
            with self.subTest(mode=mode, residual_deg=residual_deg, tilt_deg=tilt_deg):
                info = dict(ok=mode == 'random_ok', reason='test_rejected',
                            dx_m=.001, dy_m=.002, dz_m=0., err_mm=0.,
                            tube_angle_deg=2., d_tilt_deg=0., margin_rad=1., iters=1,
                            randomization_mode='joint_uniform_v1', joint_half_deg=1.5,
                            joint_delta_deg=[.57,0.,0.,0.,0.])
                random_q = self.q.copy(); random_q[0] += .01
                ns = dict(np=np, math=math, time=time, Base=Base, N_ARM=5,
                          START_JOINT_POSE=START_JOINT_POSE, CONTROLLED_IDX=(0,1,2,3),
                          FROZEN_IDX=(4,5), JOINT_LOW=JOINT_LOW, JOINT_HIGH=JOINT_HIGH,
                          START_FEEDBACK_MAX=math.radians(3),
                          START_ACCEPTANCE_TOL=acceptance_limit,
                          randomize_joint_start_pose=lambda *a, **kw: (random_q, info),
                          tube_tilt_deg=lambda q: tilt_deg, tool_z=lambda q: .16)
                code = ast.ClassDef(name='ResetHarness', bases=[ast.Name(id='Base', ctx=ast.Load())],
                                    keywords=[], body=[method], decorator_list=[])
                exec(compile(ast.fix_missing_locations(ast.Module(body=[code], type_ignores=[])),
                             '<production reset>', 'exec'), ns)
                env = ns['ResetHarness']()
                env.joint_pos = np.r_[self.q, .32]
                env._joint_stamp = time.monotonic()
                env.randomize_start = mode != 'fixed'
                env.randomize_half_m = .01
                env.randomize_joint_half_deg = 1.5
                env._kin = None; env._rng = np.random.default_rng(0)
                env.home_offset = np.zeros(5)
                env.start_feedback_iters = 2
                env.start_feedback_tol = math.radians(.4)
                env._command_hold = CommandHold()
                warnings = []
                env.node = SimpleNamespace(get_logger=lambda: SimpleNamespace(info=lambda _: None, warn=warnings.append))
                published = []
                def publish(q, **kwargs):
                    published.append(q.copy())
                    env.joint_pos[:5] = q
                    if residual_deg:
                        # A stable physical stop, despite bounded feedback corrections.
                        env.joint_pos[:5] = self.q
                        env.joint_pos[0] += math.radians(residual_deg)
                    return 0.
                env._publish_arm_ramp = publish
                env._settled_joint_position = lambda _: env.joint_pos[:5].copy()
                env._get_obs = lambda: {}
                if not accepted:
                    with self.assertRaisesRegex(RuntimeError, '归位未通过验收'):
                        env.reset(seed=1)
                    self.assertIsNone(env._command_hold.command)
                    continue
                _, result = env.reset(seed=1)
                start = result['start']
                self.assertTrue(start['accepted'])
                self.assertAlmostEqual(start['resid_deg'], residual_deg)
                self.assertAlmostEqual(start['precision_tol_deg'], .4)
                self.assertAlmostEqual(start['acceptance_tol_deg'], 3.)
                self.assertEqual(start['precision_met'], residual_deg <= .4)
                self.assertEqual(start['accepted_outside_precision'], residual_deg > .4)
                if residual_deg > .4:
                    self.assertEqual(len(published), 3)  # Still performs both corrections.
                    self.assertTrue(any('放宽后的验收范围' in msg for msg in warnings))
                self.assertFalse(set(required) - set(start))
                if mode == 'random_ok':
                    self.assertTrue(start['randomized'])
                    self.assertEqual(start['dx_m'], .001)
                    self.assertEqual(start['dy_m'], .002)
                else:
                    self.assertFalse(start['randomized'])
                    self.assertEqual([start[k] for k in ('dx_m', 'dy_m', 'dz_m')], [0.,0.,0.])
                if mode == 'random_rejected':
                    self.assertEqual(start['reason'], 'test_rejected')

    def test_joint_randomization_bounds_pitch_and_seed(self):
        from so101_home_pose import randomize_joint_start_pose
        rng = np.random.default_rng(2026)
        samples = []
        frozen = self.q.copy(); frozen[4] = .06
        for _ in range(500):
            q, info = randomize_joint_start_pose(rng, q_cur=frozen, kin=self.kin)
            self.assertTrue(info['ok'])
            delta = np.degrees(q - self.q)
            self.assertTrue(np.all(np.abs(delta[:3]) <= 1.5 + 1e-12))
            self.assertAlmostEqual(delta[3], -delta[1]-delta[2], places=10)
            self.assertEqual(q[4], frozen[4])
            self.assertLessEqual(info['tube_angle_deg'], 3.)
            expected = self.kin.fk(q)[0] - self.kin.fk(frozen)[0]
            np.testing.assert_allclose([info[k] for k in ('dx_m','dy_m','dz_m')], expected)
            samples.append(delta[:3])
        samples = np.asarray(samples)
        self.assertTrue(np.all(samples.min(axis=0) < -1.4))
        self.assertTrue(np.all(samples.max(axis=0) > 1.4))
        for seed in (0, 1, 12):
            a, _ = randomize_joint_start_pose(np.random.default_rng(seed), kin=self.kin)
            b, _ = randomize_joint_start_pose(np.random.default_rng(seed), kin=self.kin)
            np.testing.assert_array_equal(a, b)
        for bad in (-1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                randomize_joint_start_pose(rng, half_deg=bad, kin=self.kin)

    def test_torque_failure_does_not_enter_mode(self):
        tree = ast.parse((MODULE / 'env.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SO101Env')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_enter_mode')
        ns = dict(np=np, math=math, N_ARM=5)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<production mode>', 'exec'), ns)
        for mode, old in [('human', 'policy'), ('policy', 'human')]:
            anchor = self.q.copy()
            env = SimpleNamespace(_last_mode=old, _lead_anchor=anchor, _foll_anchor=anchor,
                                  set_leader_torque=lambda _: False)
            with self.assertRaises(RuntimeError):
                ns['_enter_mode'](env, mode)
            self.assertEqual(env._last_mode, old)
            self.assertIs(env._lead_anchor, anchor)



if __name__ == '__main__':
    names = ['test_vector_limit_and_validation', 'test_real_urdf_small_steps_and_continuity',
             'test_reject_bad_tilt_and_unreachable', 'test_exhausted_ik_budget_holds',
             'test_production_reset_metadata_for_all_start_modes', 'test_joint_randomization_bounds_pitch_and_seed', 'test_torque_failure_does_not_enter_mode']
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(CartesianTests(name) for name in names))
    sys.exit(not result.wasSuccessful())
