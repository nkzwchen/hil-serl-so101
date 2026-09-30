#!/usr/bin/env python3
"""把 pi_project 根目录的脚本搬进 hil-serl 仓库，并改造成「路径自解析」。

为什么
------
那些脚本里硬编码了 29 处 `/home/nkzwchen/pi_project/...`。要让两端
（以及别的机器）共用同一套代码，必须让它们自己找到 `examples/` 与
`examples/experiments/so101`，而不是假设某个人名下的绝对路径。

支持的两种布局
--------------
  A) pi_project 布局（现状）:  <P>/脚本.py            + <P>/hil-serl/examples/...
  B) 仓库布局（搬运后）:      <repo>/so101_tools/脚本.py + <repo>/examples/...

做法
----
对每个 .py：
  1) 在第一个用到项目路径的行**之前**注入一段自解析头（`_ROOT / _EX /
     _EXPERIMENTS_DIR / _SO101_DIR`）；
  2) 把 4 个字面量按「长的先替」换成这些变量（避免
     `.../examples` 把 `.../examples/experiments` 误伤）；
  3) 确保 `import os` 存在；
  4) py_compile 必须通过，否则**不写盘**并报错。

安全
----
每个文件都是「读入 → 全文替换 → 断言 → 编译 → 才写」。任何一步不过就
整个脚本退出，不留半个改坏的文件。（曾经用切片替换把文件改烂过一次，
所以这里全部走整串替换 + 断言。）

用法
----
    python pack_so101_tools.py --repo /home/nkzwchen/pi_project/hil-serl --dry-run
    python pack_so101_tools.py --repo /home/nkzwchen/pi_project/hil-serl
"""

import argparse
import os
import py_compile
import shutil
import sys
import tempfile

P = '/home/nkzwchen/pi_project'
LIT = [                      # 顺序 = 长 → 短，替换时必须按这个顺序
    (f'{P}/hil-serl/examples/experiments/so101', '_SO101_DIR'),
    (f'{P}/hil-serl/examples/experiments', '_EXPERIMENTS_DIR'),
    (f'{P}/hil-serl/examples', '_EX'),
    (P, '_ROOT'),
]

HEADER = '''# ─── 路径自解析（自动生成，勿手改：见 pack_so101_tools.py）───────────────
# 让脚本在两种布局下都能跑：
#   A) pi_project 布局： <P>/脚本.py              + <P>/hil-serl/examples/...
#   B) 仓库布局：       <repo>/so101_tools/脚本.py + <repo>/examples/...
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))


def _resolve_dirs():
    for _b in (_HERE, _os.path.dirname(_HERE)):
        if _os.path.isdir(_os.path.join(_b, 'examples', 'experiments')):
            return _b, _os.path.join(_b, 'examples')
        if _os.path.isdir(_os.path.join(_b, 'hil-serl', 'examples', 'experiments')):
            return _b, _os.path.join(_b, 'hil-serl', 'examples')
    _up = _os.path.dirname(_HERE)
    return _up, _os.path.join(_up, 'examples')


_ROOT, _EX = _resolve_dirs()
_EXPERIMENTS_DIR = _os.path.join(_EX, 'experiments')
_SO101_DIR = _os.path.join(_EXPERIMENTS_DIR, 'so101')
_sys.path.insert(0, _SO101_DIR)
_sys.path.insert(0, _EXPERIMENTS_DIR)
_sys.path.insert(0, _EX)
_sys.path.insert(0, _ROOT)
# ─────────────────────────────────────────────────────────────────────────
'''

TOOLS = [
    # 录制 / 标定 / 构建 demo —— 核心流水线
    'record_demo_so101.py', 'label_demos.py', 'build_demos.py',
    'so101_recording_io.py', 'so101_keys.py',
    'make_dummy_recording.py', 'audit_recordings.py', 'fix_recordings.py',
    # 相机体检
    'check_stride.py', 'grab_stills.py', 'export_videos.py',
    'probe_usb_cam_stride.py',
    # 归位 / 标定 / 遥操作 / 几何
    'phase1_home.py', 'phase3_cartesian_teleop.py', 'phase0_randomize_check.py',
    'phase0_fk_tf_check.py', 'phase0_v2v3.py', 'diag_reset_drift.py',
    'free_arm.py', 'sample_pose.py',
    # 测试
    'test_record_flow.py', 'test_env_reset.py', 'test_torque_jump.py',
    'test_so101_env.py', 'test_so101_intervention.py', 'test_leader_bus.py',
    'test_leader_drive.py', 'test_leader_follows_follower.py',
    'test_space_toggle.py', 'test_space_file_fallback.py',
    'test_agentlace_link.py', 'ping_client.py',
]


def transform(src_path):
    """返回改造后的源码文本；不改动盘上文件。"""
    import re
    s = open(src_path, encoding='utf-8').read()
    lines = s.split('\n')
    idx = next((i for i, ln in enumerate(lines) if P in ln), None)
    if idx is None:
        return s, 0, False            # 没有硬编码路径，原样搬运
    head = '\n'.join(lines[:idx])

    # ⚠️ 踩坑：第一次实现只替换了引号**里面**的路径，于是
    #      sys.path.insert(0, '<P>/...') → sys.path.insert(0, '_SO101_DIR')
    #    变成一个**字符串字面量** —— 能编译、能跑，但路径完全没生效。
    #    所以替换必须连引号一起吃掉；而且下面还要断言这个 bug 不再出现。
    re_insert = re.compile(r"^\s*sys\.path\.insert\(0,\s*'%s[^']*'\s*\)\s*$"
                           % re.escape(P))
    re_assign = re.compile(r"^\s*_SO101_DIR\s*=\s*'%s[^']*'\s*$" % re.escape(P))

    out_lines, n_sub, dropped = [], 0, 0
    for ln in lines[idx:]:
        if re_insert.match(ln) or re_assign.match(ln):
            dropped += 1          # 由自解析头统一插入/定义，这行删掉
            continue
        new_ln = ln
        for lit, var in LIT:      # LIT 已是「长 → 短」顺序
            for q in ("'", '"'):  # 连引号一起换，避免变成字符串字面量
                tok = f'{q}{lit}{q}'
                c = new_ln.count(tok)
                if c:
                    new_ln = new_ln.replace(tok, var)
                    n_sub += c
        # 兜底：万一还有不带引号的引用，也换掉
        for lit, var in LIT:
            c = new_ln.count(lit)
            if c:
                new_ln = new_ln.replace(lit, var)
                n_sub += c
        out_lines.append(new_ln)
    tail = '\n'.join(out_lines)

    if '\nimport os\n' not in '\n' + head:
        head = head.rstrip('\n') + '\nimport os\n'
    out = head.rstrip('\n') + '\n\n' + HEADER + tail

    # ★ 断言：不允许出现「引号包住的变量名」这种假替换
    bad = re.findall(r"'(?:_ROOT|_EX|_EXPERIMENTS_DIR|_SO101_DIR)'", out)
    if bad:
        raise RuntimeError(f'出现假替换（字符串字面量）{set(bad)} —— 替换逻辑有 bug')
    return out, n_sub, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=f'{P}/hil-serl')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    srcdir = P
    dstdir = os.path.join(args.repo, 'so101_tools')
    if not args.dry_run:
        os.makedirs(dstdir, exist_ok=True)

    print('=' * 88)
    print(f'打包 so101_tools → {dstdir}   {"(dry-run)" if args.dry_run else ""}')
    print('=' * 88)
    print(f"  {'文件':<34}{'替换处数':>9}{'行数':>7}  状态")
    ok = bad = skip = 0
    for fn in TOOLS:
        src = os.path.join(srcdir, fn)
        if not os.path.exists(src):
            print(f'  {fn:<34}{"-":>9}{"-":>7}  ⚠️ 源文件不存在，跳过')
            skip += 1
            continue
        try:
            out, n_sub, dropped = transform(src)
        except Exception as e:
            print(f'  {fn:<34}{"-":>9}{"-":>7}  ❌ 改造失败: {e}')
            bad += 1
            continue
        # ★ 写盘前必须编译通过（写到临时文件里编，不碰目标）
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False,
                                        encoding='utf-8') as tf:
            tf.write(out)
            tmp = tf.name
        try:
            py_compile.compile(tmp, cfile=tmp + 'c', doraise=True)
        except Exception as e:
            print(f'  {fn:<34}{n_sub:>9}{len(out.splitlines()):>7}  ❌ 编译不过: {e}')
            os.unlink(tmp)
            bad += 1
            continue
        os.unlink(tmp)
        os.unlink(tmp + 'c') if os.path.exists(tmp + 'c') else None
        if not args.dry_run:
            dst = os.path.join(dstdir, fn)
            with open(dst, 'w', encoding='utf-8') as f:
                f.write(out)
            os.chmod(dst, 0o644)
        print(f'  {fn:<34}{n_sub:>9}{len(out.splitlines()):>7}  '
              f'{"✅ 已写入" if not args.dry_run else "✅ 可写入"}'
              + (f'  删掉 {dropped} 行 insert' if dropped else
                 '  (无硬编码路径)'))
        ok += 1
    print(f'\n  成功 {ok}  失败 {bad}  跳过 {skip}')
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
