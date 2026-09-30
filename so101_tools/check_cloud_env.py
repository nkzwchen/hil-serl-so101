#!/usr/bin/env python3
"""云端 learner 环境的自检脚本。

为什么要单独一个文件
--------------------
远程执行时用 `ssh host 'python -c "..."'` 会陷入多层引号转义地狱，很容易
把命令写坏却不报错（本会话就踩过）。所以固定成一个文件，传上去跑。

用法（在云端）
--------------
    /root/gpufree-data/conda_envs/so101-hilserl/bin/python check_cloud_env.py
"""

import importlib
import sys
import time

NEEDED = [
    ('numpy', None), ('cv2', None), ('jax', None), ('jaxlib', None),
    ('flax', None), ('optax', None), ('distrax', None), ('chex', None),
    ('gymnasium', None), ('gym', None), ('scipy', None),
    ('agentlace', None), ('serl_launcher', None), ('wandb', None),
]

EXPECT = {          # 期望的版本（与 requirements.txt 对齐）；None = 不检查
    'numpy': '1.',
    'jax': '0.4.35',
    'flax': '0.10.4',
    'optax': '0.2.5',
    'distrax': '0.1.5',
    'scipy': '1.11.4',
    'gymnasium': '0.29.1',
}


def main():
    print('=' * 70)
    print('云端 learner 环境自检')
    print('=' * 70)
    print(f'  python {sys.version.split()[0]}  ({sys.executable})')
    bad = []

    print('\n-- 包 --')
    for name, _ in NEEDED:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, '__version__', '?')
            note = ''
            exp = EXPECT.get(name)
            if exp and not str(ver).startswith(exp):
                note = f'  ⚠️ 期望 {exp}*'
                bad.append(f'{name} 版本 {ver} != {exp}*')
            print(f'  {name:<16} {ver:<12}{note}')
        except Exception as e:
            print(f'  {name:<16} ❌ {type(e).__name__}: {e}')
            bad.append(f'{name} 导入失败')

    print('\n-- GPU --')
    try:
        import jax
        import jax.numpy as jnp
        devs = jax.devices()
        print(f'  devices: {devs}')
        if not any(d.platform == 'gpu' for d in devs):
            bad.append('jax 看不到 GPU')
            print('  ❌ jax 只看到 CPU —— jaxlib 可能不是 cuda12 版')
            print('     修：pip install -U "jax[cuda12]==0.4.35"')
        else:
            x = jnp.ones((4000, 4000))
            t = time.time()
            y = x @ x
            y.block_until_ready()
            dt = time.time() - t
            print(f'  4000³ fp32 matmul: {dt * 1000:.0f} ms  '
                  f'({2 * 4000 ** 3 / dt / 1e12:.1f} TFLOPS)  '
                  f'-> {float(y[0, 0])}')
    except Exception as e:
        print(f'  ❌ GPU 测试失败: {type(e).__name__}: {e}')
        bad.append('GPU 测试失败')

    print('\n-- 训练栈能不能真的建起来 --')
    try:
        import jax
        from serl_launcher.agents.continuous.sac import SACAgent
        from serl_launcher.utils.launcher import make_trainer_config
        cfg = make_trainer_config()
        print(f'  make_trainer_config() -> request_port={getattr(cfg, "port_number", "?")} '
              f'broadcast_port={getattr(cfg, "broadcast_port", "?")}')
        print(f'  SACAgent: {SACAgent.__name__}')
        print(f'  jax backend: {jax.default_backend()}')
    except Exception as e:
        print(f'  ❌ {type(e).__name__}: {e}')
        bad.append('训练栈构建失败')

    print('\n' + '=' * 70)
    if bad:
        print(f'  ❌ {len(bad)} 项有问题：')
        for b in bad:
            print(f'     · {b}')
        return 1
    print('  ✅ 全部通过，可以跑 learner')
    return 0


if __name__ == '__main__':
    sys.exit(main())
