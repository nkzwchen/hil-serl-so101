"""Hardware-free tests for replay admission, budget accounting and policy boundaries."""
import unittest
import numpy as np
from online_training import HumanReplay, UpdateBudget, EpisodePolicy


class OnlineTests(unittest.TestCase):
    def receipt(self, ident, indices=()):
        return dict(id=ident, recording_sha256=ident, intervention_indices=list(indices))

    def test_interventions_added_to_both_pools_once(self):
        r=HumanReplay()
        r.add(['d0','d1'],self.receipt('demo'),'demo')
        r.add(['p','h','p2','h2'],self.receipt('online',[1,3]),'online')
        self.assertEqual(list(r.demos),['d0','d1','h','h2'])
        self.assertEqual(len(r.online),4)
        self.assertFalse(r.add(['p','h','p2','h2'],self.receipt('online',[1,3]),'online'))
        self.assertEqual(r.online_received,4)
        batch=r.sample(np.random.default_rng(0),64)
        self.assertEqual(len(batch),64)
        self.assertTrue(all(x in r.demos for x in batch[:32]))
        self.assertTrue(all(x in r.online for x in batch[32:]))

    def test_conflicting_label_and_capacity(self):
        r=HumanReplay(2)
        r.add([1,2,3],self.receipt('a'),'online')
        self.assertEqual(list(r.online),[2,3])
        self.assertEqual(r.online_received,3)
        with self.assertRaises(ValueError):r.add([4],dict(id='b',recording_sha256='a'),'online')

    def test_budget_survives_checkpoint_and_duplicate_receipts(self):
        b=UpdateBudget(100,0,.25)
        self.assertEqual(b.remaining(100,285),71)
        self.assertEqual(b.remaining(171,285),0)
        resumed=UpdateBudget(**b.state())
        self.assertEqual(resumed.remaining(171,285),0)
        self.assertEqual(resumed.remaining(171,289),1)

    def test_policy_only_fetches_at_boundary(self):
        calls=[];applied=[]
        old=dict(ok=True,step=1,spec={},run_id='run')
        new=dict(old,step=2)
        p=EpisodePolicy(old, lambda obs:0,lambda meta:calls.append(meta.copy()) or new,applied.append)
        p(0);p(0)
        self.assertFalse(calls)
        self.assertTrue(p.before_episode())
        self.assertEqual(p.metadata['step'],2)
        p(0)
        self.assertEqual(len(calls),1)
        self.assertFalse(p.before_episode())
        self.assertEqual(len(applied),1)

    def test_bad_policy_and_network_failure_preserve_version(self):
        old=dict(ok=True,step=1,spec={},run_id='run')
        def offline(_):raise OSError('offline')
        p=EpisodePolicy(old,lambda obs:0,offline,lambda _:None)
        self.assertFalse(p.before_episode())
        self.assertEqual(p.metadata['step'],1)
        p.fetch=lambda _:dict(old,step=2,run_id='other')
        with self.assertRaises(ValueError):p.before_episode()
        p.fetch=lambda _:dict(old,step=2)
        def bad(_):raise ValueError('invalid tensors')
        p.apply=bad
        with self.assertRaises(ValueError):p.before_episode()
        self.assertEqual(p.metadata['step'],1)

if __name__=='__main__':unittest.main()
