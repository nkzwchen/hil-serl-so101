#!/usr/bin/env bash
# b02_run.sh —— 双进程（Learner / Actor）握手与训练循环的启动器
#
# 为什么需要它
# ------------
# `train_rlpd.py` 的 learner 侧有 `assert FLAGS.demo_path is not None`（RLPD 的离线
# prior data 是硬需求），而且 checkpoint 目录只要存在就会 `input("Press Enter to
# resume")` —— 在非交互场景下直接 EOFError。这个脚本把这些坑都绕过去了。
#
# 用法
# ----
#   bash b02_run.sh learner                      # 在本机起 learner
#   bash b02_run.sh actor --ip 127.0.0.1         # 在本机起 actor
#   bash b02_run.sh actor --ip 127.0.0.1 --remote-cloud   # actor 在终端、learner 在云端
#   bash b02_run.sh stop                         # 停掉本机所有 train_rlpd
#
# 全部路径都可用环境变量覆盖，不假设用户名：
#   PY        python 解释器           （默认：当前 PATH 里的 python）
#   EXAMPLES  hil-serl/examples 目录  （默认：本脚本上一级 + examples）
#   LOGDIR    日志目录                （默认：仓库根）
#   DEMOS     demo pkl                （默认：$LOGDIR/demos/dummy_demos.pkl）
#   CKPT      checkpoint 目录         （默认：$LOGDIR/checkpoints/dummy_test，**故意不存在**）
#   ENV_NAME  exp_name                （默认 dummy_test）
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"

# 两种布局：<repo>/so101_tools/  或  <project>/  (仓库在 <project>/hil-serl)
if [ -d "$REPO/examples/experiments" ]; then
  EXAMPLES="${EXAMPLES:-$REPO/examples}"
elif [ -d "$(dirname "$REPO")/hil-serl/examples/experiments" ]; then
  REPO="$(dirname "$REPO")/hil-serl"
  EXAMPLES="${EXAMPLES:-$REPO/examples}"
else
  echo "❌ 找不到 examples/ 目录，请用 EXAMPLES=<path> 指定" >&2; exit 1
fi

PY="${PY:-python}"
LOGDIR="${LOGDIR:-$REPO}"
DEMOS="${DEMOS:-$LOGDIR/demos/dummy_demos.pkl}"
# ⚠️ ckpt 目录**必须不存在**：存在就会触发 train_rlpd 的交互式 resume 提示
CKPT="${CKPT:-$LOGDIR/checkpoints/dummy_test}"
ENV_NAME="${ENV_NAME:-dummy_test}"

MODE="${1:-}"; shift || true
ARGS=("$@")
IP="127.0.0.1"
for ((i=0; i<${#ARGS[@]}; i++)); do
  [ "${ARGS[$i]}" = "--ip" ] && IP="${ARGS[$((i+1))]:-127.0.0.1}"
done

ensure_demos() {
  [ -f "$DEMOS" ] && return 0
  echo "  demo 不存在，先生成：$DEMOS"
  mkdir -p "$(dirname "$DEMOS")"
  ( cd "$EXAMPLES" && env -u PYTHONPATH PYTHONPATH="$EXAMPLES" "$PY" gen_dummy_demos.py \
      --out "$DEMOS" ) || { echo "❌ 生成 demo 失败" >&2; return 1; }
}

run_one() {
  local role="$1" log="$2"
  ensure_demos || return 1
  cd "$EXAMPLES" || exit 1
  # --debug 关掉 wandb（云端再配 WANDB_API_KEY）
  nohup env -u PYTHONPATH PYTHONPATH="$EXAMPLES" "$PY" train_rlpd.py \
      --exp_name "$ENV_NAME" --"$role" --ip "$IP" --debug \
      --checkpoint_path "$CKPT" --demo_path "$DEMOS" \
      > "$log" 2>&1 &
  echo "  $role 已启动 pid=$!  ip=$IP  日志: $log"
  echo "  跟踪: tail -f $log"
}

stop_all() {
  # 方括号技巧：避免匹配到本脚本自身
  ps -eo pid,args 2>/dev/null | grep "[t]rain_rlpd.py" | awk '{print $1}' | while read -r p; do
    kill -9 "$p" 2>/dev/null && echo "  已停止 pid=$p"
  done
}

case "$MODE" in
  learner) run_one learner "${LOGDIR}/b02_learner.log" ;;
  actor)   run_one actor   "${LOGDIR}/b02_actor.log"   ;;
  stop)    stop_all ;;
  *) echo "用法: bash $0 {learner|actor|stop} [--ip <addr>]"; exit 1 ;;
esac
