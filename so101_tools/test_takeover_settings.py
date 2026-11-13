import contextlib,io,pickle,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from takeover_settings import TakeoverBudget,load_limit,save_limit
import test_record_flow as flow
import record_demo_so101 as recorder
from experiments.so101 import config

class TakeoverTests(unittest.TestCase):
    def test_persistence_and_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'settings.json'
            self.assertEqual(load_limit(p),0)
            save_limit(2,p);self.assertEqual(load_limit(p),2)
            for bad in (-1,True,2.5,1000):
                with self.assertRaises(ValueError):save_limit(bad,p)
            self.assertEqual(load_limit(p),2)
            p.write_text('broken');self.assertEqual(load_limit(p),0)
    def test_count_only_policy_to_human(self):
        b=TakeoverBudget(2)
        for _ in range(2):
            self.assertFalse(b.press_space(False));self.assertFalse(b.press_space(True))
        self.assertTrue(b.press_space(False));self.assertEqual(b.count,2)
        self.assertEqual(TakeoverBudget(2).count,0)
        b=TakeoverBudget(0)
        for _ in range(20):self.assertFalse(b.press_space(False))
    def test_third_takeover_ends_record_without_extra_action(self):
        phase=[''];keys=iter([None,None,' ',' ',' ',' ',' ',' ',' ']);executed=[]
        class Env(flow.StubEnv):
            def step(self,a):
                if phase[0]=='RECORD':executed.append(self.is_intervening)
                return super().step(a)
        class Keys(flow.StubKeys):
            def get_max_takeovers(self):return 2
        def poll(kb,accept):
            if phase[0] in ('READY','RETURN'):return 'c'
            if phase[0] in ('RECORD','TAKEOVER_PAUSE'):return next(keys,'d')
            return None
        with tempfile.TemporaryDirectory() as d:
            with patch.object(config,'SO101Env',Env), \
                 patch.object(recorder,'NonBlockingKey',return_value=Keys([])), \
                 patch.object(recorder,'panel_phase',side_effect=lambda kb,p,detail='':phase.__setitem__(0,p)), \
                 patch.object(recorder,'poll',side_effect=poll), \
                 patch.object(sys,'argv',['record','--terminal-controls','--out',d,'--episodes','1']), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(recorder.main(policy=lambda obs:np.zeros(3,np.float32)),0)
            with next(Path(d).glob('*/ep_*.pkl')).open('rb') as f:data=pickle.load(f)
            self.assertEqual(executed,[False])
            self.assertEqual(data['meta']['takeover_count'],2)
            self.assertTrue(data['meta']['takeover_limit_reached'])
            self.assertFalse(data['meta']['operator_excluded'])
            self.assertEqual(data['meta']['max_takeovers'],2)

if __name__=='__main__':unittest.main()
