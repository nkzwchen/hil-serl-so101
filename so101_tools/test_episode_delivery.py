"""Offline delivery tests; no GUI, SSH or robot is opened."""
import json
from pathlib import Path
import subprocess
import unittest
from test_human_reward import HumanRewardTests
from episode_delivery import EpisodeDelivery

class DeliveryTests(unittest.TestCase):
    setUp = HumanRewardTests.setUp
    save = HumanRewardTests.save

    def flow(self, reviewer=lambda p: None, reply=None):
        self.uploads=[]; self.submissions=[]
        def submit(name):
            self.submissions.append(name)
            return reply if reply is not None else {'ok': True, 'transitions':3}
        return EpisodeDelivery(self.cfg, lambda p,n:self.uploads.append((p,n)), submit, reviewer)

    def status(self):
        return json.loads(self.path.with_suffix('.delivery.json').read_text())

    def test_review_then_upload_pair_and_ack(self):
        flow=self.flow()
        self.assertTrue(flow.process(self.path))
        self.assertEqual(self.status()['status'],'submitted')
        self.assertEqual(len(self.uploads),2)
        self.assertEqual(Path(self.uploads[0][1]).with_suffix('.labels.json').name,self.uploads[1][1])
        self.assertEqual(self.submissions,[self.uploads[0][1]])
        self.assertTrue(flow.process(self.path))
        self.assertEqual(len(self.uploads),2)

    def test_close_without_confirmation_never_uploads_even_existing_label(self):
        def cancel(p): raise subprocess.CalledProcessError(3,'labeler')
        flow=self.flow(reviewer=cancel)
        self.assertFalse(flow.process(self.path))
        self.assertFalse(self.uploads)
        self.assertEqual(self.status()['status'],'pending_label')

    def test_unreviewed_labels_never_upload(self):
        self.lab['provisional']=True; self.save()
        flow=self.flow()
        self.assertFalse(flow.process(self.path))
        self.assertFalse(self.uploads)

    def test_upload_failure_persists_and_retry_succeeds(self):
        flow=self.flow()
        def fail(p,n): raise OSError('network unavailable')
        flow.uploader=fail
        self.assertFalse(flow.process(self.path))
        self.assertEqual(self.status()['status'],'pending_upload')
        flow=self.flow()
        self.assertTrue(flow.process(self.path,review=False))
        self.assertEqual(self.status()['status'],'submitted')

    def test_server_rejection_not_marked_success(self):
        flow=self.flow(reply={'ok':False,'error':'incompatible metadata'})
        self.assertFalse(flow.process(self.path))
        self.assertEqual(self.status()['status'],'pending_submit')

    def test_exclusion_skips_all_network_work(self):
        flow=self.flow()
        self.assertTrue(flow.process(self.path,{'operator_excluded':True}))
        self.assertFalse(self.uploads)
        self.assertEqual(self.status()['status'],'excluded')

    def test_restart_deduplicates_alternate_filenames(self):
        import contextlib
        import io
        from types import SimpleNamespace
        from unittest.mock import patch
        from train_human import train
        duplicate = self.path.with_name('renamed.pkl')
        duplicate.write_bytes(self.path.read_bytes())
        duplicate.with_suffix('.labels.json').write_bytes(self.path.with_suffix('.labels.json').read_bytes())
        args = SimpleNamespace(data=str(self.path.parent), out=str(self.path.parent/'run'), resume=None, allow_provisional=False)
        output = io.StringIO()
        with patch('train_human.make_agent', side_effect=RuntimeError('stop before model creation')):
            with contextlib.redirect_stdout(output):
                with self.assertRaisesRegex(RuntimeError, 'stop before model creation'):
                    train(args)
        self.assertIn('Loaded 1 episodes / 3 transitions', output.getvalue())

if __name__=='__main__':unittest.main()
