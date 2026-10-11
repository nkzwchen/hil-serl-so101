import json,tempfile,unittest
from pathlib import Path
import numpy as np
from flax import struct,serialization
from warm_start import load_weights
from prepare_10hz import ranges

@struct.dataclass
class State:
    params: object
    target_params: object
    optimizer: object
    step: int
@struct.dataclass
class Agent:
    state: State

class MigrationTests(unittest.TestCase):
    def test_exact_networks_fresh_optimizer_no_source_write(self):
        new=Agent(State({'x':np.zeros(2)}, {'x':np.zeros(2)},np.zeros(2),0))
        old=State({'x':np.ones(2)}, {'x':np.ones(2)*2},np.ones(2)*9,2083)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'old'/'checkpoint.msgpack';p.parent.mkdir()
            payload=serialization.to_bytes(old);p.write_bytes(payload)
            spec=dict(env_hz=30,action_scale_rad=float(np.radians(4)),encoder='resnet-pretrained',allow_provisional=False)
            Path(str(p)+'.json').write_text(json.dumps(dict(spec=spec,step=2083)))
            target=dict(spec,env_hz=10,action_scale_rad=float(np.radians(5.5)))
            agent,info=load_weights(new,p,target,Path(directory)/'new')
            np.testing.assert_array_equal(agent.state.params['x'],old.params['x'])
            np.testing.assert_array_equal(agent.state.target_params['x'],old.target_params['x'])
            np.testing.assert_array_equal(agent.state.optimizer,new.state.optimizer)
            self.assertEqual(agent.state.step,0);self.assertEqual(p.read_bytes(),payload)
            with self.assertRaises(ValueError):load_weights(new,p,target,p.parent)
            with self.assertRaises(ValueError):load_weights(new,p,dict(target,encoder='other'),Path(directory)/'new')

    def test_legacy_does_not_stitch_across_bad_window(self):
        ts=[];scale=np.radians(4)
        for i in range(9):
            obs={'state':np.array([i])};nxt={'state':np.array([i+1])}
            q={'cameras':{'cam_main':{'valid':True,'age_s':.01}}}
            jc=dict(step_started_monotonic_s=i/30,step_finished_monotonic_s=(i+1)/30,
                accepted=True,limited=i==4,command_before_rad=np.full(3,i*.1*scale),target_rad=np.full(3,(i+1)*.1*scale))
            ts.append(dict(observations=obs,next_observations=nxt,actions=np.full(3,.1),infos=dict(intervening=True,joint_control=jc,observation_quality=q,next_observation_quality=q)))
        self.assertEqual(ranges(ts,scale),[(6,9)])
        ts[-1]['infos']['joint_control']['limited']=True
        self.assertEqual(ranges(ts,scale),[])

if __name__=='__main__':unittest.main()
