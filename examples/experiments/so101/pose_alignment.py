"""Provisional coordinate alignment for native SO101 comparison tools, in degrees.

Only lift/elbow/wrist are corrected. Relative actions need no extra offset:
the same constant cancels between current and anchor poses.
This does not establish physical zero calibration or change training semantics.
"""
import hashlib
import json
import math
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name('so101_joint_pose_alignment.json')
ALIGNMENT = json.loads(CONFIG_PATH.read_text())
OFFSETS_DEG = ALIGNMENT['offset_deg']
if ALIGNMENT['units'] != 'degrees' or set(OFFSETS_DEG) != {'shoulder_lift', 'elbow_flex', 'wrist_flex'}:
    raise ValueError('Alignment must correct only lift, elbow and wrist in degrees')
if not all(math.isfinite(v) for v in OFFSETS_DEG.values()):
    raise ValueError('Nonfinite pose offset')


def leader_to_follower(positions):
    """Convert absolute leader readings; untouched keys, including pan, pass through."""
    result = dict(positions)
    for joint, offset in OFFSETS_DEG.items():
        if joint not in result:
            continue
        value = result[joint]
        if not math.isfinite(value):
            raise ValueError('Nonfinite joint pose: ' + joint)
        result[joint] = value + offset
    return result


def follower_to_leader(positions):
    """Inverse absolute mapping, used for leader return targets."""
    result = dict(positions)
    for joint, offset in OFFSETS_DEG.items():
        if joint not in result:
            continue
        value = result[joint]
        if not math.isfinite(value):
            raise ValueError('Nonfinite joint pose: ' + joint)
        result[joint] = value - offset
    return result


def check_calibration_file(path, arm):
    """Reject offsets derived from a different software calibration frame."""
    expected = ALIGNMENT['calibration'][arm]
    if Path(path).resolve() != Path(expected['path']).resolve():
        raise ValueError('Pose alignment calibration path mismatch: ' + arm)
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected['sha256']:
        raise ValueError('Calibration changed; revalidate pose offsets for ' + arm)
