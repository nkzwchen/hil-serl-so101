import json
from pathlib import Path
import tempfile
import unittest
from training_monitor import Monitor, episode_scalars
from experiments.so101.human_reward import capture_summary


def receipt(i, success=True, role='online', fraction=0., provisional=False):
    return dict(id=str(i),role=role,label=dict(status='success' if success else 'failed',reviewed=not provisional,
                provisional=provisional,source='human_review'),transitions=3,quality_flagged=0,
                capture=dict(steps=5,intervention_fraction=fraction,policy_present=True,timeout=False))


class MonitorTests(unittest.TestCase):
    def test_denominators_and_provisional(self):
        rows=[receipt(1),receipt(2,False,fraction=.5),receipt(3,provisional=True),receipt(4,fraction=None)]
        data={tag:v for tag,v,step in episode_scalars(rows) if step==4}
        self.assertEqual(data['episodes/online/reviewed_count'],3)
        self.assertAlmostEqual(data['episodes/online/success_rate'],2/3)
        self.assertEqual(data['episodes/online/autonomous_reviewed_count'],1)
        self.assertEqual(data['episodes/online/autonomous_success_rate'],1)
        self.assertEqual(data['episodes/online/unreviewed_or_provisional_count'],1)

    def test_demo_and_online_do_not_mix(self):
        data={tag:v for tag,v,step in episode_scalars([receipt(1,role='demo'),receipt(2,False)])}
        self.assertEqual(data['episodes/demo/success_rate'],1)
        self.assertEqual(data['episodes/online/success_rate'],0)

    def test_last20_and_no_autonomous_rate_when_empty(self):
        rows=[receipt(i,success=i>=2,fraction=.1) for i in range(22)]
        data={tag:v for tag,v,step in episode_scalars(rows) if step==22}
        self.assertEqual(data['episodes/online/last20_count'],20)
        self.assertEqual(data['episodes/online/last20_success_rate'],1)
        self.assertNotIn('episodes/online/autonomous_success_rate',data)

    def test_capture_full_segment_and_missing_feedback(self):
        cap=capture_summary([{'infos':{'intervening':True}},{'infos':{'intervening':False}}],{'record_sec':1.2})
        self.assertEqual(cap['intervention_fraction'],.5)
        self.assertEqual(cap['steps'],2)
        self.assertIsNone(cap['timeout'])
        self.assertIsNone(capture_summary([{'infos':{}}],{})['intervention_fraction'])

    def test_live_append_partial_lines_and_receipt_dedup(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp); log=run/'events';m=Monitor(run,log)
            line=json.dumps({'step':1,'metrics':{'critic':{'critic_loss':.25}}})
            (run/'metrics.jsonl').write_text(line)
            (run/'receipts.json').write_text(json.dumps([receipt(1)]))
            m.poll();m.poll()
            with (run/'metrics.jsonl').open('a') as f:f.write('\n')
            m.poll()
            (run/'receipts.json').write_text('{')
            m.poll()  # Partial receipt write must be retried, not counted as empty data.
            (run/'receipts.json').write_text(json.dumps([receipt(1),receipt(2,False)]))
            m.poll();m.poll();m.close()
            events=EventAccumulator(str(log));events.Reload()
            self.assertEqual(len(events.Scalars('train/critic/critic_loss')),1)
            rates=events.Scalars('episodes/online/success_rate')
            self.assertEqual([e.step for e in rates],[1,2])
            self.assertEqual([e.value for e in rates],[1.,.5])

    def test_pending_is_local_and_resets_to_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'recordings';data.mkdir()
            (data/'ep_0000.pkl').touch()
            state=data/'ep_0000.delivery.json';state.write_text(json.dumps({'status':'pending_label'}))
            m=Monitor(root,root/'events',recordings=data);m.poll()
            state.write_text(json.dumps({'status':'submitted'}));m.poll();m.close()
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            e=EventAccumulator(str(root/'events'));e.Reload()
            self.assertEqual([v.value for v in e.Scalars('local_queue/pending_label')],[1,0])

if __name__=='__main__':unittest.main()
