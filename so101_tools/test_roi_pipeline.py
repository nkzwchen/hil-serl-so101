"""Verify three-view observation spaces, conversion and real replay insertion offline."""
import copy
import json
import pickle
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch
import gymnasium as gym
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE / 'hil-serl' if (HERE / 'hil-serl').is_dir() else HERE.parent
sys.path.insert(0, str(REPO / 'examples'))
from experiments.so101 import config
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
import build_demos

spec = dict(version=1, camera='cam_main', source_size=[640, 480],
            xyxy=[200, 150, 360, 290], preprocess='letterbox_rgb_v1')


class NoHardwareEnv(gym.Env):
    def __init__(self, **kw):
        keys = ['cam_main', 'cam_wrist'] + (['cam_main_roi'] if kw['roi_spec'] else [])
        size = kw['image_size']
        self.observation_space = gym.spaces.Dict({
            'state': gym.spaces.Dict({
                **{k:gym.spaces.Box(-1.,1.,(6,),np.float32) for k in ['joint_pos','joint_vel']},
                'command_error':gym.spaces.Box(-np.inf,np.inf,(4,),np.float32),
                'hold_offset':gym.spaces.Box(-np.inf,np.inf,(4,),np.float32)}),
            'images': gym.spaces.Dict({k: gym.spaces.Box(0, 255, (size, size, 3), np.uint8)
                                      for k in keys})})
        self.action_space = gym.spaces.Box(-1., 1., (4,), np.float32)

    def reset(self, **kw):
        return self.observation_space.sample(), {}


with patch.object(config, 'load_roi', return_value=spec), \
        patch.object(config, 'load_main_roi', return_value=spec), \
        patch.object(config, 'SO101Env', NoHardwareEnv):
    cfg = config.TrainConfig()
    real = cfg.get_environment()
    fake = cfg.get_environment(fake_env=True)
    obs, _ = real.reset()
    assert set(obs) == set(fake.observation_space.spaces)
    assert fake.observation_space.contains(obs)
    assert obs['state'].shape==(1,20)
    check, space, actions = build_demos.build_space_checker()
    buffer = MemoryEfficientReplayBufferDataStore(space, actions, capacity=32,
                                                  image_keys=cfg.image_keys)
    for i in range(12):
        next_obs, _ = real.reset()
        t = dict(observations=obs, next_observations=next_obs,
                 actions=np.zeros(4, np.float32), rewards=0., masks=1., dones=False)
        assert not check(t), check(t)
        buffer.insert(copy.deepcopy(t))
        obs = next_obs
    batch = buffer.sample(4)
    assert batch['observations']['cam_main_roi'].shape == (4, 1, 224, 224, 3)
    old = copy.deepcopy(t)
    del old['observations']['cam_main_roi']
    assert check(old), 'Old two-view data must fail the three-view schema'
    with tempfile.TemporaryDirectory() as folder:
        rec = Path(folder) / 'ep_0000.pkl'
        labels = rec.with_suffix('.labels.json')
        labels.write_text(json.dumps({'episodes': {'0': {'status': 'success', 'frame': 2}}}))
        with rec.open('wb') as file:
            pickle.dump({'transitions': [t, t, t], 'meta': {'roi_spec': spec, 'main_roi_spec': spec, 'action_semantics': cfg.ACTION_SEMANTICS, 'observation_semantics': cfg.OBSERVATION_SEMANTICS, 'action_scale_rad': cfg.ACTION_SCALE, 'action_joints': cfg.ACTION_JOINTS, 'env_hz':cfg.ENV_HZ, 'control_backend':cfg.CONTROL_BACKEND, 'pose_alignment':cfg.POSE_ALIGNMENT}}, file)
        with patch.object(sys, 'argv', ['build_demos.py', str(rec), '--dry-run']):
            assert build_demos.main() == 0
        with rec.open('wb') as file:
            pickle.dump({'transitions':[t], 'meta':dict(roi_spec=spec,main_roi_spec=spec,
                action_semantics='relative_joint_delta_v1',observation_semantics=cfg.OBSERVATION_SEMANTICS,
                action_scale_rad=cfg.ACTION_SCALE,action_joints=cfg.ACTION_JOINTS)},file)
        with patch.object(sys,'argv',['build_demos.py',str(rec),'--dry-run','--all-failed']):
            assert build_demos.main()!=0, 'Old executed-action labels must fail even with new observation shape'
        with rec.open('wb') as file:
            pickle.dump({'transitions': [t, t, t], 'meta': {'roi_spec': None, 'action_semantics': cfg.ACTION_SEMANTICS, 'observation_semantics': cfg.OBSERVATION_SEMANTICS, 'action_scale_rad': cfg.ACTION_SCALE, 'action_joints': cfg.ACTION_JOINTS, 'env_hz':cfg.ENV_HZ, 'control_backend':cfg.CONTROL_BACKEND, 'pose_alignment':cfg.POSE_ALIGNMENT}}, file)
        with patch.object(sys, 'argv', ['build_demos.py', str(rec), '--dry-run', '--all-failed']):
            assert build_demos.main() != 0, 'ROI mismatch must not bypass validation'
    with tempfile.TemporaryDirectory() as folder:
        rec = Path(folder) / 'legacy.pkl'
        with rec.open('wb') as file:
            pickle.dump({'transitions': [t], 'meta': {'roi_spec': spec, 'main_roi_spec': spec}}, file)
        with patch.object(sys, 'argv', ['build_demos.py', str(rec), '--dry-run', '--all-failed']):
            assert build_demos.main() != 0, 'Legacy semantics must fail even for three-component actions'
    print('PASS: three-view spaces, conversion, ROI mismatch rejection, ReplayBuffer insert/sample')
