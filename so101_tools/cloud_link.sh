#!/usr/bin/env bash
# cloud_link.sh —— 打通「终端 actor → 云端 learner」的 SSH 隧道
#
# 为什么要隧道
# ------------
# agentlace 用两个 TCP 端口：
#   · 5588  请求端口（actor 拉动作、learner 返动作）
#   · 5589  广播端口（learner 把新网络权重推给 actor）
# 云端只对外开放 SSH 端口（120.209.70.195:30555），5588/5589 不出网。
# 所以在本机把这两个端口**转发**到云端，actor 就能用 `--ip 127.0.0.1` 连上。
#
# 用法
# ----
#   bash cloud_link.sh start      # 开隧道（后台），并打印 actor 命令
#   bash cloud_link.sh status     # 看隧道/端口状态
#   bash cloud_link.sh stop       # 关隧道
#
# 可用环境变量覆盖：
#   SO101_CLOUD=root@120.209.70.195   SO101_PORT=30555
set -uo pipefail

CLOUD="${SO101_CLOUD:-root@120.209.70.195}"
PORT="${SO101_PORT:-30555}"
REQ_PORT="${SO101_REQ_PORT:-5588}"
BC_PORT="${SO101_BC_PORT:-5589}"
PIDFILE="${TMPDIR:-/tmp}/so101_cloud_tunnel.pid"

is_up() {
  # 端口上有监听 = 隧道在
  (exec 3<>/dev/tcp/127.0.0.1/"$1") 2>/dev/null && { exec 3<&-; return 0; } || return 1
}

case "${1:-}" in
  start)
    # ★ 先清残留：上一次没 stop 干净时，端口会被旧进程占着，
    #   新建隧道会以 "bind: Address already in use" 失败（踩过）。
    stale=$(pgrep -f "ssh .*-L ${REQ_PORT}:localhost:${REQ_PORT}" 2>/dev/null | tr '\n' ' ')
    if [ -n "$stale" ]; then
      echo "  清理残留隧道: $stale"
      for p in $stale; do kill "$p" 2>/dev/null; done
      sleep 1
    fi
    if is_up "$REQ_PORT" && is_up "$BC_PORT"; then
      echo "  隧道已在（$REQ_PORT/$BC_PORT 都在监听）"
    else
      echo "  建立隧道 $CLOUD  ($REQ_PORT, $BC_PORT) ..."
      # -N 不执行远端命令；-f 后台；ServerAliveInterval 防掉线
      ssh -f -N -p "$PORT" \
          -o BatchMode=yes -o ExitOnForwardFailure=yes \
          -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
          -L "${REQ_PORT}:localhost:${REQ_PORT}" \
          -L "${BC_PORT}:localhost:${BC_PORT}" \
          "$CLOUD" || { echo "  ❌ 隧道建立失败"; exit 1; }
      pid=$(pgrep -f "ssh .*-L ${REQ_PORT}:localhost:${REQ_PORT}" | head -1)
      [ -n "$pid" ] && echo "$pid" > "$PIDFILE"
      sleep 1
    fi
    is_up "$REQ_PORT" && echo "  ✅ $REQ_PORT 已转发" || echo "  ❌ $REQ_PORT 未通"
    is_up "$BC_PORT"  && echo "  ✅ $BC_PORT 已转发"  || echo "  ❌ $BC_PORT 未通"
    echo
    echo "  终端侧 actor 这样启动（注意是 127.0.0.1，走隧道）："
    echo "    cd ${SO101_REPO:-<repo>}/examples"
    echo "    PYTHONPATH=\$PWD python train_rlpd.py --actor --ip 127.0.0.1 --exp_name so101"
    echo "  或：bash $(dirname "${BASH_SOURCE[0]}")/b02_run.sh actor --ip 127.0.0.1"
    ;;
  status)
    echo "  云端      : $CLOUD (ssh -p $PORT)"
    for p in "$REQ_PORT" "$BC_PORT"; do
      is_up "$p" && echo "  端口 $p : ✅ 通" || echo "  端口 $p : ❌ 不通"
    done
    if [ -f "$PIDFILE" ]; then
      pid=$(cat "$PIDFILE")
      kill -0 "$pid" 2>/dev/null && echo "  隧道 pid : $pid（在跑）" || echo "  隧道 pid : $pid（已不在）"
    fi
    echo "  云端 learner 是否在跑："
    ssh -p "$PORT" -o BatchMode=yes "$CLOUD" \
        'ps -eo pid,etime,args | grep "[t]rain_rlpd.py" | head -3' 2>/dev/null || true
    ;;
  stop)
    if [ -f "$PIDFILE" ]; then
      kill "$(cat "$PIDFILE")" 2>/dev/null && echo "  已停隧道 pid=$(cat $PIDFILE)"
      rm -f "$PIDFILE"
    fi
    # 兜底：按命令行特征清理
    pkill -f "ssh .*-L ${REQ_PORT}:localhost:${REQ_PORT}" 2>/dev/null && echo "  已清理残留隧道"
    echo "  完成"
    ;;
  *)
    echo "用法: bash $0 {start|status|stop}"; exit 1 ;;
esac
