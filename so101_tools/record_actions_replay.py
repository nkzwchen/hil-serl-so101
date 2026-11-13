"""Action-only recording and dual-arm replay: no camera subscriptions or images."""
import sys
import record_demo_so101 as recording
from record_fullframe_replay import replay_episode


def main():
    from experiments.so101.config import TrainConfig
    if not any(x=='--out' or x.startswith('--out=') for x in sys.argv[1:]):
        sys.argv+=['--out','recordings_actions']
    return recording.main(config_factory=lambda:TrainConfig(use_cameras=False),
                          after_local_save=replay_episode)

if __name__=='__main__':sys.exit(main())
