"""Durable, episode-boundary human review and SSH delivery (never in step())."""
import json
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys


def write_status(path, status):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


class SSHUploader:
    def __init__(self, host, directory, port=22, control_path=None):
        if not host or host.startswith('-') or not Path(directory).is_absolute():
            raise ValueError('SSH host and absolute learner inbox directory required')
        self.directory = directory
        self.command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-p', str(port)]
        if control_path:
            self.command += ['-S', control_path]
        self.command += [host]

    def remote(self, code, *args, stdin=None):
        command = shlex.join(['python3', '-c', code, *map(str, args)])
        subprocess.run([*self.command, command], stdin=stdin, check=True, timeout=300)

    def check(self):
        self.remote('import os,sys; p=sys.argv[1]; assert os.path.isdir(p) and os.access(p,os.W_OK), p', self.directory)

    def __call__(self, path, name):
        # Never expose a partial pkl or label file to the learner. Labels go last.
        code = '''import os,sys,tempfile,hashlib
from pathlib import Path
root=Path(sys.argv[1]); name=sys.argv[2]
assert Path(name).name==name
fd,tmp=tempfile.mkstemp(prefix='.upload-',dir=root)
try:
 with os.fdopen(fd,'wb') as f:
  digest=hashlib.sha256()
  while True:
   block=sys.stdin.buffer.read(1024*1024)
   if not block: break
   digest.update(block); f.write(block)
  f.flush(); os.fsync(f.fileno())
 assert digest.hexdigest()==sys.argv[3], 'Incomplete or corrupted upload'
 os.replace(tmp,root/name)
finally:
 if os.path.exists(tmp): os.unlink(tmp)
'''
        with Path(path).open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
            source.seek(0)
            self.remote(code, self.directory, name, digest, stdin=source)


def review_window(path):
    env = os.environ.copy()
    env.setdefault('DISPLAY', ':0')
    subprocess.run([sys.executable, str(Path(__file__).with_name('label_demos.py')),
                    '--review-once', str(path)], env=env, check=True)


class EpisodeDelivery:
    def __init__(self, config, uploader, submit, reviewer=review_window):
        self.config, self.uploader, self.submit, self.reviewer = config, uploader, submit, reviewer

    def process(self, path, meta=None, review=True):
        from experiments.so101.human_reward import load_episode
        path = Path(path).resolve()
        status_path = path.with_suffix('.delivery.json')
        status = dict(recording=str(path), status='pending_label')
        if meta and (meta.get('operator_excluded') or meta.get('recording', {}).get('has_gaps')):
            status['status'] = 'excluded'
            write_status(status_path, status)
            print('本段已标记排除或存在缺步，仅保留本地记录，不上传训练。', flush=True)
            return True
        if status_path.exists():
            status = json.loads(status_path.read_text())
        try:
            write_status(status_path, status)
            if review and status.get('status') != 'submitted':
                print('控制暂停：请在窗口中标成功帧或失败，按 w 确认并提交。', flush=True)
                self.reviewer(path)
            transitions, receipt = load_episode(path, self.config, allow_provisional=False)
            del transitions
            if status.get('status') == 'submitted' and status.get('id') == receipt['id']:
                return True
            name = 'episode_' + receipt['id'] + '.pkl'
            status.update(status='pending_upload', id=receipt['id'], remote_file=name, error=None)
            write_status(status_path, status)
            self.uploader(path, name)
            self.uploader(path.with_suffix('.labels.json'), Path(name).with_suffix('.labels.json').name)
            status['status'] = 'pending_submit'
            write_status(status_path, status)
            reply = self.submit(name)
            if not reply or not reply.get('ok'):
                raise RuntimeError(f'Learner refused episode: {reply}')
            status.update(status='submitted', acknowledgement=reply)
            write_status(status_path, status)
            print(f'人工标签已确认，云端已接收 {reply.get("transitions")} 步。', flush=True)
            return True
        except (Exception, KeyboardInterrupt) as exc:
            status['error'] = f'{type(exc).__name__}: {exc}'
            write_status(status_path, status)
            print(f'本段待提交，原始数据保留：{status_path}\n{status["error"]}', flush=True)
            return False
