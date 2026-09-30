#!/usr/bin/env python3
"""把本机录制的 demo 构建好并上传到云端，供 learner 用 `--demo_path` 加载。

为什么需要它
------------
HIL-SERL 的 learner 侧 `train_rlpd.py` 里有 `assert FLAGS.demo_path is not None`
—— RLPD 的离线 prior data 是**硬需求**，没有它就不是 RLPD 了。
而 demo 是在**终端侧**（有相机、有机器人）录制+人工标定的，必须传到**云端**。

它做三件事
----------
  1. 调 `build_demos.py` 构建（稀疏奖励 + 逐条 space 校验）
  2. `scp` 到云端的 `demo_data/`
  3. 打印云端起 learner 的命令（含正确的 `--demo_path`）

"全失败也上传"
--------------
默认：只有标了成功帧的段会进 demo。
加 `--all-failed`：一段成功都没有时，把所有段完整上传（reward 全 0）。
    ⚠️ 这样上传的数据里没有任何成功样本，RLPD 只能当行为先验用，
       学不到"成功"是什么。有了成功段之后应当重跑，不要与旧包混用。

用法
----
    python push_demos.py --recordings recordings/2026-09-30_004807_v1/
    python push_demos.py --recordings recordings/*/ --all-failed
    python push_demos.py --recordings recordings/x/ --cloud root@120.209.70.195 \
        --port 30555 --dry-run
"""

import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('recordings', nargs='+', help='会话目录 / pkl / glob')
    ap.add_argument('--cloud', default=os.environ.get('SO101_CLOUD', 'root@120.209.70.195'),
                    help='云端 ssh 目标（可用环境变量 SO101_CLOUD 覆盖）')
    ap.add_argument('--port', type=int, default=int(os.environ.get('SO101_PORT', 30555)))
    ap.add_argument('--remote-dir',
                    default='/root/gpufree-data/so101-hil-serl/demo_data')
    ap.add_argument('--out', default='demo_data', help='本地构建输出目录')
    ap.add_argument('--all-failed', action='store_true',
                    help='一段成功都没有时，把所有段完整上传（reward 全 0）')
    ap.add_argument('--allow-frame0', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='只构建不上传')
    ap.add_argument('--start-learner', action='store_true',
                    help='上传后顺手在云端起 learner（后台）')
    args = ap.parse_args()

    # ---------- 1) 构建 ----------
    cmd = [sys.executable, os.path.join(HERE, 'build_demos.py'),
           *args.recordings, '--out', args.out]
    if args.all_failed:
        cmd.append('--all-failed')
    if args.allow_frame0:
        cmd.append('--allow-frame0')
    print('=' * 84)
    print('1) 构建 demo')
    print('=' * 84)
    print('  $ ' + ' '.join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print('\n❌ 构建失败。常见原因：')
        print('   · 还没标定 → python so101_tools/label_demos.py <录制文件>')
        print('   · 全标成了 failure 但没加 --all-failed')
        print('   · 成功帧标在了第 0 帧（它前面没有 transition 可给奖励）')
        return r.returncode

    pkls = sorted(glob.glob(os.path.join(args.out, '*.pkl')), key=os.path.getmtime)
    if not pkls:
        print(f'❌ {args.out}/ 里没找到 pkl')
        return 1
    pkl = pkls[-1]                      # 最新那个
    size = os.path.getsize(pkl) / 1e6
    print(f'\n  产出：{pkl}  ({size:.1f} MB)')

    if args.dry_run:
        print('\n--dry-run：不上传。')
        return 0

    # ---------- 2) 上传 ----------
    print()
    print('=' * 84)
    print(f'2) 上传到 {args.cloud}:{args.remote_dir}')
    print('=' * 84)
    ssh_base = ['-P', str(args.port), '-o', 'BatchMode=yes']
    subprocess.run(['ssh', *ssh_base, args.cloud,
                    f'mkdir -p {args.remote_dir}'], check=True)
    r = subprocess.run(['scp', *ssh_base, pkl, f'{args.cloud}:{args.remote_dir}/'])
    if r.returncode != 0:
        print('❌ scp 失败（检查密钥/端口）')
        return 1
    remote_path = f'{args.remote_dir}/{os.path.basename(pkl)}'
    print(f'  ✅ 已上传 → {remote_path}')

    # ---------- 3) 给命令 ----------
    print()
    print('=' * 84)
    print('3) 在云端起 learner')
    print('=' * 84)
    env = '/root/gpufree-data/conda_envs/so101-hilserl'
    repo = args.remote_dir.rsplit('/demo_data', 1)[0]
    start = (f'{env}/bin/python {repo}/examples/train_rlpd.py --learner '
             f'--exp_name so101 --demo_path {remote_path}')
    print(f'  {start}')
    if args.start_learner:
        log = f'{repo}/learner.log'
        subprocess.run(['ssh', *ssh_base, args.cloud,
                        f'cd {repo}/examples && nohup env PYTHONPATH={repo}/examples '
                        f'bash -c "{start} --debug" > {log} 2>&1 & echo started'],
                       check=True)
        print(f'  已在云端后台启动，日志 {log}')
        print(f'  跟踪：ssh -p {args.port} {args.cloud} "tail -f {log}"')
    print()
    print('  终端侧 actor（另开一个终端，注意 IP 走 SSH 隧道或云端内网）：')
    print('    python examples/train_rlpd.py --actor --ip <地址> --exp_name so101')
    return 0


if __name__ == '__main__':
    sys.exit(main())
