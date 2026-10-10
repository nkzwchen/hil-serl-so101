"""Offline checks; no ROS or hardware access."""
import pickle
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np
from async_recorder import AsyncEpisodeRecorder


def packet(value=0):
    return {'observations': {'state': np.array([value])},
            'next_observations': {'state': np.array([value + 1])},
            'actions': np.array([value]), 'infos': {'intervening': True, 'joint_control': {'target_rad': [value]}},
            'dones': False, 'masks': 1., 'rewards': 0.}


class RecorderTests(unittest.TestCase):
    def test_snapshot_order_terminal_and_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ep.pkl'
            writer = AsyncEpisodeRecorder(path)
            p = packet()
            self.assertTrue(writer.submit(p))
            p['actions'][0] = 99
            p['infos']['joint_control']['target_rad'][0] = 99
            self.assertTrue(writer.submit(packet(1)))
            meta = writer.finish({'quality_failed': False})
            with path.open('rb') as stream:
                transitions = pickle.load(stream)['transitions']
            self.assertEqual([t['infos']['recording_step'] for t in transitions], [0, 1])
            self.assertEqual(transitions[0]['actions'][0], 0)
            self.assertEqual(transitions[0]['infos']['joint_control']['target_rad'], [0])
            self.assertFalse(transitions[0]['dones'])
            self.assertTrue(transitions[-1]['dones'])
            self.assertEqual(transitions[-1]['masks'], 0.)
            self.assertFalse(meta['recording']['has_gaps'])
            self.assertFalse(writer.journal.exists())

    def test_full_queue_does_not_wait_and_marks_missing_step(self):
        started, release = threading.Event(), threading.Event()
        original = AsyncEpisodeRecorder._write_step
        def blocked(self, stream, data):
            started.set()
            if not release.wait(5):
                raise RuntimeError('Test writer timed out')
            original(self, stream, data)
        with tempfile.TemporaryDirectory() as directory, patch.object(AsyncEpisodeRecorder, '_write_step', blocked):
            writer = AsyncEpisodeRecorder(Path(directory) / 'ep.pkl', capacity=1)
            try:
                self.assertTrue(writer.submit(packet()))
                self.assertTrue(started.wait(2))
                self.assertTrue(writer.submit(packet(1)))
                # Disk worker is still blocked: submit must return independently.
                self.assertFalse(writer.submit(packet(2)))
            finally:
                release.set()
                meta = writer.finish({'quality_failed': False})
            self.assertEqual(meta['recording']['dropped_steps'][0]['step'], 2)
            self.assertEqual(meta['n_steps'], 2)
            self.assertEqual(meta['recording']['n_control_steps'], 3)
            self.assertTrue(meta['quality_failed'])
            self.assertTrue(meta['requires_gap_review'])

    def test_gap_ends_previous_contiguous_run(self):
        started, release, second_written = threading.Event(), threading.Event(), threading.Event()
        original = AsyncEpisodeRecorder._write_step
        def blocked(self, stream, data):
            if data['infos']['recording_step'] == 0:
                started.set()
                if not release.wait(5):
                    raise RuntimeError('Test writer timed out')
            original(self, stream, data)
            if data['infos']['recording_step'] == 1:
                second_written.set()
        with tempfile.TemporaryDirectory() as directory, patch.object(AsyncEpisodeRecorder, '_write_step', blocked):
            writer = AsyncEpisodeRecorder(Path(directory) / 'ep.pkl', capacity=1)
            try:
                writer.submit(packet())
                self.assertTrue(started.wait(2))
                writer.submit(packet(1))
                self.assertFalse(writer.submit(packet(2)))
                release.set()
                self.assertTrue(second_written.wait(2))
                self.assertTrue(writer.submit(packet(3)))
            finally:
                release.set()
                writer.finish({'quality_failed': False})
            with writer.destination.open('rb') as stream:
                transitions = pickle.load(stream)['transitions']
            self.assertEqual([t['infos']['recording_step'] for t in transitions], [0, 1, 3])
            self.assertTrue(transitions[1]['dones'])
            self.assertEqual(transitions[1]['masks'], 0.)

    def test_disk_failure_preserves_journal_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(AsyncEpisodeRecorder, '_write_step', side_effect=OSError('disk full')):
            writer = AsyncEpisodeRecorder(Path(directory) / 'ep.pkl')
            writer.submit(packet())
            with self.assertRaisesRegex(RuntimeError, 'disk full'):
                writer.finish({})
            self.assertTrue(writer.journal.exists())
            self.assertFalse(writer.destination.exists())

    def test_abort_retains_completed_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = AsyncEpisodeRecorder(Path(directory) / 'ep.pkl')
            writer.submit(packet(7))
            writer.abort()
            with writer.journal.open('rb') as stream:
                self.assertEqual(pickle.load(stream)['actions'][0], 7)
            self.assertFalse(writer.destination.exists())


if __name__ == '__main__':
    unittest.main()
