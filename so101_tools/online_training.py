"""Pure replay/budget and episode-boundary policy helpers (no hardware imports)."""
from collections import deque
import math

REPLAY_SEMANTICS = 'human_terminal_success_bootstrap_cutoff_v2'


class HumanReplay:
    def __init__(self, capacity=50000):
        self.demos = deque(maxlen=capacity)
        self.online = deque(maxlen=capacity)
        self.receipts = {}
        self.online_received = 0
        self.interventions_received = 0

    def add(self, transitions, receipt, role):
        if receipt['id'] in self.receipts:
            return False
        if any(r['recording_sha256'] == receipt['recording_sha256'] for r in self.receipts.values()):
            raise ValueError('Relabeled episode already ingested; rebuild replay explicitly')
        receipt = dict(receipt, role=role)
        if role == 'demo':
            self.demos.extend(transitions)
        elif role == 'online':
            self.online.extend(transitions)
            indices = receipt.get('intervention_indices', [])
            self.demos.extend(transitions[i] for i in indices)
            self.online_received += len(transitions)
            self.interventions_received += len(indices)
        else:
            raise ValueError('Unknown replay role')
        self.receipts[receipt['id']] = receipt
        return True

    def sample(self, rng, batch_size):
        if not self.demos:
            raise ValueError('No demonstration samples')
        n = batch_size // 2 if self.online else 0
        return ([self.demos[i] for i in rng.integers(len(self.demos), size=batch_size-n)] +
                [self.online[i] for i in rng.integers(len(self.online), size=n)])


class UpdateBudget:
    """One credit = cta_ratio critic updates and one actor/temperature update."""
    def __init__(self, base_step, initial_updates, updates_per_transition):
        if initial_updates < 0 or not math.isfinite(updates_per_transition) or updates_per_transition < 0:
            raise ValueError('Invalid update budget')
        self.base_step = base_step
        self.initial_updates = initial_updates
        self.updates_per_transition = updates_per_transition

    def remaining(self, step, online_received):
        return max(0, self.base_step + self.initial_updates +
                   math.floor(online_received*self.updates_per_transition) - step)

    def state(self):
        return vars(self).copy()


class EpisodePolicy:
    """Fetch/apply is explicit and only called before RECORD; no RPC in __call__."""
    def __init__(self, snapshot, action, fetch, apply):
        self.metadata = self._metadata(snapshot)
        self.action, self.fetch, self.apply = action, fetch, apply

    @staticmethod
    def _metadata(snapshot):
        return {k: snapshot[k] for k in ('step', 'spec', 'run_id') if k in snapshot}

    def before_episode(self):
        try:
            candidate = self.fetch(self.metadata)
        except Exception as exc:
            print(f'Policy refresh unavailable; retaining step {self.metadata["step"]}: {exc}', flush=True)
            return False
        if not candidate or not candidate.get('ok'):
            print('Policy refresh unavailable; retaining current policy', flush=True)
            return False
        if candidate.get('unchanged'):
            return False
        if candidate['spec'] != self.metadata['spec'] or candidate['spec'].get('allow_provisional'):
            raise ValueError('Refusing incompatible/provisional policy at episode boundary')
        if candidate.get('run_id') != self.metadata.get('run_id'):
            raise ValueError('Learner run changed; restart actor explicitly to select the new run')
        if candidate['step'] <= self.metadata['step']:
            return False
        # apply must validate/warm up candidate before replacing the active agent.
        self.apply(candidate)
        self.metadata = self._metadata(candidate)
        print(f'Policy refreshed before RECORD: step {self.metadata["step"]}', flush=True)
        return True

    def __call__(self, obs):
        return self.action(obs)
