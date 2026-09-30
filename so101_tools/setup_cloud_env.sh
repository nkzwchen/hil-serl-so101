#!/usr/bin/env bash
# setup_cloud_env.sh —— 在云端建「只有 learner」的 conda 环境（Python 3.10，无 ROS2）
#
# 为什么单独一个环境
# ------------------
# 终端侧要 ROS2（rclpy，Python 3.12，RoboStack）；云端只跑 learner，
# 不需要 ROS，用 Python 3.10 + jax[cuda12] 更省事、依赖冲突更少。
# 两边共用**同一份代码**（同一个 GitHub 仓库），区别只在启动参数
# `--learner` / `--actor`。
#
# 磁盘注意
# --------
# 这类机器常常 `/` 很小（本机云主机只有 ~17GB 可用）而大盘挂在别处。
# 所以 conda envs / pkgs / pip cache **全部指到大盘**。
#
# 用法
# ----
#   bash setup_cloud_env.sh                 # 全流程：clone + 建环境 + 校验
#   bash setup_cloud_env.sh --skip-clone    # 只建环境（仓库已经在了）
#   ROOT=/data/xxx bash setup_cloud_env.sh  # 换大盘位置
set -euo pipefail

ROOT="${ROOT:-/root/gpufree-data}"
REPO_DIR="${REPO_DIR:-$ROOT/so101-hil-serl}"
ENV_PREFIX="${ENV_PREFIX:-$ROOT/conda_envs/so101-hilserl}"
CONDA="${CONDA:-/opt/conda/bin/conda}"
REPO_URL="${REPO_URL:-git@github.com:nkzwchen/hil-serl-so101.git}"
PY_VER="${PY_VER:-3.10}"
SKIP_CLONE=0
[ "${1:-}" = "--skip-clone" ] && SKIP_CLONE=1

# 把缓存都挪到大盘（否则默认写 /opt/conda/pkgs 和 ~/.cache，可能撑爆 /）
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$ROOT/conda_pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/pip_cache}"
mkdir -p "$ROOT" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

say "0/6 环境变量"
echo "  ROOT        = $ROOT"
echo "  REPO_DIR    = $REPO_DIR"
echo "  ENV_PREFIX  = $ENV_PREFIX"
echo "  conda       = $CONDA"
echo "  conda pkgs  = $CONDA_PKGS_DIRS"
echo "  pip cache   = $PIP_CACHE_DIR"
df -h "$ROOT" | tail -1

if [ "$SKIP_CLONE" = 0 ]; then
  say "1/6 克隆/更新仓库"
  if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" fetch --all --prune
    git -C "$REPO_DIR" checkout main
    git -C "$REPO_DIR" pull --ff-only
  else
    git clone "$REPO_URL" "$REPO_DIR"
  fi
  git -C "$REPO_DIR" log --oneline -1
else
  say "1/6 跳过克隆"
fi

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  say "2/6 建 conda 环境（python $PY_VER）→ $ENV_PREFIX"
  "$CONDA" create -y -p "$ENV_PREFIX" "python=$PY_VER"
else
  say "2/6 conda 环境已存在，跳过创建"
fi
PY="$ENV_PREFIX/bin/python"
"$PY" -V

say "3/6 升级 pip（并钉 setuptools，避免新版移除 distutils 之类的问题）"
"$PY" -m pip install -q -U pip wheel 'setuptools<81'

say "4/6 装 serl_launcher 的依赖（jax[cuda12]、flax、gymnasium…）"
"$PY" -m pip install -r "$REPO_DIR/serl_launcher/requirements.txt"

say "5/6 装 agentlace（train_rlpd 的进程间通信，上游没写进 requirements）"
"$PY" -m pip install -q agentlace==0.1.3

say "5.5/6 以 editable 方式装 serl_launcher（这样改代码不用重装）"
"$PY" -m pip install -q -e "$REPO_DIR/serl_launcher"

say "6/6 校验"
"$PY" - <<'PYEOF'
import importlib, sys
print(f'  python {sys.version.split()[0]}')
for m in ('jax', 'flax', 'optax', 'distrax', 'gymnasium', 'agentlace', 'serl_launcher'):
    try:
        mod = importlib.import_module(m)
        print(f'  {m:<16} ✅ {getattr(mod, "__version__", "?")}')
    except Exception as e:
        print(f'  {m:<16} ❌ {type(e).__name__}: {e}')
        sys.exit(1)
import jax
devs = jax.devices()
print(f'  jax devices      {devs}')
if not any(d.platform == "gpu" for d in devs):
    print('  ⚠️ 没看到 GPU —— 检查 jaxlib 是否是 cuda12 版：')
    print('     pip install -U "jax[cuda12]==0.4.35"')
    sys.exit(2)
print('  ✅ GPU 可用')
PYEOF

say "完成"
cat <<EOF
  conda 环境: $ENV_PREFIX
  仓库      : $REPO_DIR
  起 learner: $ENV_PREFIX/bin/python $REPO_DIR/examples/train_rlpd.py --learner ...
  或        : bash $REPO_DIR/so101_tools/b02_run.sh learner   （PY=$ENV_PREFIX/bin/python）
EOF
