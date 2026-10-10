"""Relative commands for three controlled joints; no Cartesian IK."""
import numpy as np

ACTION_SEMANTICS = 'relative_joint_target_request_v4_3d'
OBSERVATION_SEMANTICS = 'joint_command_state_v4_3d'
PROPRIO_DIMS = dict(joint_pos=6, joint_vel=6, command_error=3, hold_offset=3)
ACTION_DIM = 3
DEFAULT_STEP_RAD = float(np.radians(4.))
CONTROL_HZ = 30.0


def bounded_joint_delta(action, step_rad=DEFAULT_STEP_RAD):
    a = np.asarray(action, float)
    if a.shape != (ACTION_DIM,) or not np.all(np.isfinite(a)):
        raise ValueError('Joint action must contain three finite values')
    if not np.isfinite(step_rad) or step_rad <= 0:
        raise ValueError('Joint action scale must be finite and positive')
    return np.clip(a, -1., 1.) * step_rad


def execute_joint_request(command, measured, offset, request, low, high, step_rad=DEFAULT_STEP_RAD):
    """Both sources request increments from the previous accepted target.

    No measured-position tracking gate. Measurements/hold offsets are telemetry;
    finite action saturation and joint-position bounds apply equally to both sources.
    """
    command=np.asarray(command,float);measured=np.asarray(measured,float);offset=np.asarray(offset,float)
    low=np.asarray(low,float);high=np.asarray(high,float)
    if any(x.shape!=(ACTION_DIM,) or not np.all(np.isfinite(x)) for x in (command,measured,offset,low,high)):
        raise ValueError('Executor requires three finite values for each joint vector')
    raw_request=np.asarray(request,float)
    delta=bounded_joint_delta(raw_request,step_rad)
    target=command.copy();reasons=[[] for _ in range(ACTION_DIM)]
    for j in range(ACTION_DIM):
        if abs(delta[j])<=1e-12:continue
        if abs(raw_request[j])>1.:reasons[j].append('action_range')
        goal=command[j]+delta[j]
        joint_goal=np.clip(goal,low[j],high[j])
        if abs(joint_goal-goal)>1e-12:reasons[j].append('joint_limit')
        target[j]=joint_goal
    applied=target-command
    return target,dict(accepted=True,requested_action=(delta/step_rad).tolist(),
        raw_requested_action=raw_request.tolist(),
        requested_delta_rad=delta.tolist(),applied_delta_rad=applied.tolist(),
        command_before_rad=command.tolist(),target_rad=target.tolist(),measured_before_rad=measured.tolist(),
        applied_action=(applied/step_rad).tolist(),limited=any(reasons),joint_reasons=reasons,
        held_joints=[j for j in range(ACTION_DIM) if abs(delta[j])>1e-12 and abs(applied[j])<=1e-12])


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
