"""Local base-frame tool-origin increments, bounded IK, and calibrated command holding."""
import time
import numpy as np

ACTION_SEMANTICS = 'base_tool_origin_delta_v2'
ACTION_DIM = 3
DEFAULT_STEP_M = 0.002


def bounded_delta(action, step_m=DEFAULT_STEP_M):
    a = np.asarray(action, dtype=float)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError('Cartesian action must be three finite values')
    if not np.isfinite(step_m) or step_m <= 0:
        raise ValueError('step_m must be finite and positive')
    a = np.clip(a, -1., 1.)
    return limit_delta(a * step_m, step_m)

def limit_delta(delta, step_m):
    if not np.isfinite(step_m) or step_m <= 0:
        raise ValueError('step_m must be finite and positive')
    d = np.asarray(delta, float)
    if d.shape != (3,) or not np.all(np.isfinite(d)):
        raise ValueError("Displacement must contain three finite values")
    return d / max(1., np.linalg.norm(d) / step_m)


class CommandHold:
    def __init__(self):
        self.command = None
        self.offset = None

    def calibrate(self, command, measured):
        self.command = np.asarray(command, float).copy()
        self.offset = self.command - np.asarray(measured, float)

    def target(self, geometric=None):
        if self.command is None:
            raise RuntimeError('Run an accepted reset before step()')
        return self.command.copy() if geometric is None else np.asarray(geometric) + self.offset

    def remember(self, command):
        self.command = np.asarray(command, float).copy()


class LocalCartesianIK:
    def __init__(self, kin, active=(0, 1, 2, 3), max_joint_step_deg=2., tilt_limit_deg=3., time_budget_s=.04):
        self.kin = kin
        self.active = np.asarray(active, int)
        self.max_joint_step = np.radians(max_joint_step_deg)
        self.tilt_limit = np.radians(tilt_limit_deg)
        self.time_budget_s = float(time_budget_s)

    def tilt(self, q):
        _, R = self.kin.fk(q)
        d = R @ self.kin.tube_axis
        return np.arccos(np.clip(-d[2], -1., 1.))

    def solve(self, q0, delta):
        q0 = np.asarray(q0, float).copy()
        delta = np.asarray(delta, float)
        if q0.shape != (5,) or delta.shape != (3,) or not np.all(np.isfinite(q0)) or not np.all(np.isfinite(delta)):
            raise ValueError('Invalid joint state or tool-origin delta')
        deadline = time.perf_counter() + self.time_budget_s
        p0, _ = self.kin.fk(q0)
        if np.linalg.norm(delta) < 1e-12:
            return q0, dict(accepted=True, fraction=0., applied_delta_m=[0., 0., 0.])
        if self.tilt(q0) > self.tilt_limit:
            return q0, dict(accepted=False, reason='initial_tilt', applied_delta_m=[0., 0., 0.])
        low = np.maximum(self.kin.q_low[self.active], q0[self.active] - self.max_joint_step)
        high = np.minimum(self.kin.q_high[self.active], q0[self.active] + self.max_joint_step)
        for fraction in (1., .5, .25, .125):
            target = p0 + fraction * delta
            q = q0.copy()
            for _ in range(60):
                if time.perf_counter() >= deadline:
                    return q0, dict(accepted=False, reason='ik_time_budget', applied_delta_m=[0., 0., 0.])
                p, R = self.kin.fk(q)
                error = target - p
                if np.linalg.norm(error) <= 2e-5:
                    break
                J = self.kin.jacobian(q)
                Jp = J[:3, self.active]
                pinv = np.linalg.pinv(Jp, rcond=1e-5)
                dq = Jp.T @ np.linalg.solve(Jp @ Jp.T + 1e-6 * np.eye(3), error)
                # Direction preference only in the position-task null space.
                d = R @ self.kin.tube_axis
                skew = np.array([[0., -d[2], d[1]], [d[2], 0., -d[0]], [-d[1], d[0], 0.]])
                Jd = -skew @ J[3:, self.active]
                N = np.eye(len(self.active)) - pinv @ Jp
                dq += N @ (.1 * (q0[self.active] - q[self.active]) + .05 * Jd.T @ (np.array([0., 0., -1.]) - d))
                q[self.active] = np.clip(q[self.active] + np.clip(dq, -.02, .02), low, high)
            actual, _ = self.kin.fk(q)
            if np.linalg.norm(actual - target) <= 2e-5 and self.tilt(q) <= self.tilt_limit and np.all(q >= self.kin.q_low) and np.all(q <= self.kin.q_high):
                return q, dict(accepted=True, fraction=fraction,
                              applied_delta_m=(fraction * delta).tolist(),
                              predicted_delta_m=(actual-p0).tolist(),
                              position_error_m=float(np.linalg.norm(actual-target)),
                              tilt_deg=float(np.degrees(self.tilt(q))))
        return q0, dict(accepted=False, reason='unreachable_within_limits', applied_delta_m=[0., 0., 0.])
