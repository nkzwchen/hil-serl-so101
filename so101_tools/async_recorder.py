"""Bounded, single-producer offline writer. Never calls robot APIs."""
import copy
import os
from pathlib import Path
import pickle
import queue
import threading
import time


class AsyncEpisodeRecorder:
    """Snapshot each completed step; write a recoverable pickle stream off-thread.

    submit never waits for disk/queue space. On overflow a small gap record is
    retained and the episode is explicitly marked unsuitable for direct training.
    finish may wait, and must only be called after leaving the control loop.
    """
    def __init__(self, destination, capacity=32):
        if capacity < 1:
            raise ValueError('Queue capacity must be positive')
        self.destination = Path(destination)
        self.journal = self.destination.with_suffix('.steps.partial')
        self.queue = queue.Queue(maxsize=capacity)
        self.stop = threading.Event()
        self.error = None
        self.dropped = []
        self.submitted = 0
        self.high_water = 0
        self.max_queue_delay_s = 0.0
        self.closed = False
        self.thread = threading.Thread(target=self._run, name='episode-writer', daemon=True)
        self.thread.start()

    def submit(self, transition):
        if self.closed:
            raise RuntimeError('Recorder already closed')
        step = self.submitted
        self.submitted += 1
        if self.error is not None or self.queue.full():
            self.dropped.append({'step': step, 'reason': 'writer_error' if self.error else 'queue_full',
                                 'monotonic_s': time.monotonic()})
            return False
        # Own the snapshot: no later mutation by the env/operator reaches the writer.
        packet = copy.deepcopy(transition)
        packet['infos']['recording_step'] = step
        packet['infos']['recording_enqueued_monotonic_s'] = time.monotonic()
        try:
            self.queue.put_nowait(packet)
        except queue.Full:
            self.dropped.append({'step': step, 'reason': 'queue_full', 'monotonic_s': time.monotonic()})
            return False
        self.high_water = max(self.high_water, self.queue.qsize())
        return True

    def _write_step(self, stream, packet):
        delay = time.monotonic() - packet['infos']['recording_enqueued_monotonic_s']
        self.max_queue_delay_s = max(self.max_queue_delay_s, delay)
        pickle.dump(packet, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()

    def _run(self):
        try:
            with self.journal.open('xb') as stream:
                while not self.stop.is_set() or not self.queue.empty():
                    try:
                        packet = self.queue.get(timeout=.05)
                    except queue.Empty:
                        continue
                    self._write_step(stream, packet)
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'

    def finish(self, meta):
        self.closed = True
        self.stop.set()
        self.thread.join()
        if self.error:
            raise RuntimeError(f'Recorder failed: {self.error}; recoverable journal: {self.journal}')
        transitions = []
        with self.journal.open('rb') as stream:
            while True:
                try:
                    transitions.append(pickle.load(stream))
                except EOFError:
                    break
        for before, after in zip(transitions, transitions[1:]):
            if after['infos']['recording_step'] != before['infos']['recording_step'] + 1:
                before['dones'] = True
                before['masks'] = 0.0
        if transitions:
            transitions[-1]['dones'] = True
            transitions[-1]['masks'] = 0.0
        meta = dict(meta)
        meta['n_steps'] = len(transitions)
        meta['n_control_intervened'] = meta.get('n_intervened', 0)
        meta['n_intervened'] = sum(bool(t['infos'].get('intervening')) for t in transitions)
        meta['intervened_frac'] = round(meta['n_intervened'] / max(len(transitions), 1), 3)
        meta['recording'] = {'mode': 'bounded_async_journal_v1', 'queue_capacity': self.queue.maxsize,
                             'queue_high_water': self.high_water,
                             'max_queue_delay_s': self.max_queue_delay_s,
                             'n_control_steps': self.submitted, 'dropped_steps': self.dropped,
                             'has_gaps': bool(self.dropped)}
        if self.dropped:
            meta['quality_failed'] = True
            meta['requires_gap_review'] = True
        payload = {'transitions': transitions, 'meta': meta}
        temporary = self.destination.with_suffix('.pkl.partial')
        # Final assembly is off the control loop too (finish is a phase boundary).
        with temporary.open('wb') as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.destination)
        self.journal.unlink()
        return meta

    def abort(self):
        """Drain accepted steps on exceptional exit, retaining the partial journal."""
        self.closed = True
        self.stop.set()
        self.thread.join()
