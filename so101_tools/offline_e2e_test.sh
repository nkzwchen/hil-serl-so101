#!/usr/bin/env bash
# offline_e2e_test.sh —— 离线端到端测试（不接设备/相机）
#
# 测什么
# ------
#   1) 云端环境自检（版本 / GPU / 训练栈）
#   2) 云端 learner 能不能起来、能不能加载 demo（RLPD 的 prior data）
#   3) 本地 actor 能不能**经 SSH 隧道**连上云端 learner 并握手
#   4) 双边训练循环的实际速率
#   5) demo 上传链路
#
# 为什么是离线
# ------------
# 用 examples/experiments/dummy_test/（纯 numpy 玩具环境）当 env，把"算法进程间
# 通信"这个变量单独隔离出来 —— 设备没接的时候只有它能测。
#
# 用法
# ----
#   bash offline_e2e_test.sh            # 全部
#   bash offline_e2e_test.sh --quick    # 缩短等待时间
#
# 产物
# ----
#   /root/gpufree-data/offline_test/<时间戳>/   云端侧日志
#   /tmp/so101_offline_test/<时间戳>/           本地侧日志
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
CLOUD="${SO101_CLOUD:-root@120.209.70.195}"
PORT="${SO101_PORT:-30555}"
CLOUD_ROOT="${SO101_CLOUD_ROOT:-/root/gpufree-data}"
CLOUD_PY="$CLOUD_ROOT/conda_envs/so101-hilserl/bin/python"
STAMP="$(date +%Y-%m-%d_%H%M%S)"
LOCAL_DIR="/tmp/so101_offline_test/$STAMP"
REMOTE_DIR="$CLOUD_ROOT/offline_test/$STAMP"
WAIT_START=60      # 等 learner 起来
WAIT_RUN=75        # 跑一段看速率
[ "${1:-}" = "--quick" ] && { WAIT_START=30; WAIT_RUN=35; }

mkdir -p "$LOCAL_DIR"
SSH="ssh -p $PORT -o BatchMode=yes -o StrictHostKeyChecking=accept-new $CLOUD"
PASS=0; FAIL=0
note() { printf '\n\033[1;36m──── %s\033[0m\n' "$*"; }
ok()   { echo "  ✅ $*"; PASS=$((PASS+1)); }
bad()  { echo "  ❌ $*"; FAIL=$((FAIL+1)); }

echo "=================================================================="
echo " SO101 HIL-SERL 离线端到端测试   $STAMP"
echo "=================================================================="
echo "  云端      : $CLOUD (ssh -p $PORT)"
echo "  本地日志  : $LOCAL_DIR"
echo "  云端日志  : $REMOTE_DIR"

# ---------------------------------------------------------------- 0) 环境
note "0) 云端环境自检"
$SSH "$CLOUD_PY $CLOUD_ROOT/check_cloud_env.py" > "$LOCAL_DIR/cloud_env.txt" 2>&1
if grep -q "全部通过" "$LOCAL_DIR/cloud_env.txt"; then
  grep -E "python|jax|jaxlib|devices|matmul|request_port|backend" "$LOCAL_DIR/cloud_env.txt" | sed 's/^/  /'
  ok "云端环境自检通过"
else
  tail -12 "$LOCAL_DIR/cloud_env.txt" | sed 's/^/  /'
  bad "云端环境自检未通过"
fi

# ---------------------------------------------------------------- 1) 隧道
note "1) SSH 隧道（云端只开 SSH，5588/5589 靠转发）"
bash "$HERE/cloud_link.sh" start > "$LOCAL_DIR/tunnel.txt" 2>&1
if grep -q "5588 已转发" "$LOCAL_DIR/tunnel.txt" && grep -q "5589 已转发" "$LOCAL_DIR/tunnel.txt"; then
  ok "5588 / 5589 已转发"
else
  bad "隧道没通"; cat "$LOCAL_DIR/tunnel.txt"; exit 1
fi

# ---------------------------------------------------------------- 2) learner
note "2) 云端起 learner（dummy_test + 假 demo）"
$SSH "set -e
  cd $CLOUD_ROOT/so101-hil-serl && git pull -q --ff-only
  mkdir -p $CLOUD_ROOT/demos $REMOTE_DIR
  [ -f $CLOUD_ROOT/demos/dummy_demos.pkl ] || ( cd $CLOUD_ROOT/so101-hil-serl/examples && \
      PYTHONPATH=\$PWD $CLOUD_PY gen_dummy_demos.py --out $CLOUD_ROOT/demos/dummy_demos.pkl >/dev/null )
  pkill -f '[t]rain_rlpd.py' || true
  rm -rf $CLOUD_ROOT/checkpoints/dummy_test          # 必须不存在，否则等 input()
  cd $CLOUD_ROOT/so101-hil-serl/examples
  nohup env PYTHONPATH=\$PWD $CLOUD_PY train_rlpd.py \
    --exp_name dummy_test --learner --ip 127.0.0.1 --debug \
    --checkpoint_path $CLOUD_ROOT/checkpoints/dummy_test \
    --demo_path $CLOUD_ROOT/demos/dummy_demos.pkl \
    > $REMOTE_DIR/learner.log 2>&1 &
  echo learner_pid=\$!" > "$LOCAL_DIR/learner_start.txt" 2>&1
grep -q "learner_pid=" "$LOCAL_DIR/learner_start.txt" && ok "learner 已在云端后台启动" \
  || { bad "learner 启动失败"; cat "$LOCAL_DIR/learner_start.txt"; }

# 等 learner 就绪（出现 "Filling up replay buffer" 就是起来了）
note "3) 等 learner 就绪（最多 ${WAIT_START}s）"
T0=$(date +%s); READY=0
for _ in $(seq 1 "$WAIT_START"); do
  sleep 1
  if $SSH "grep -q 'Filling up replay buffer' $REMOTE_DIR/learner.log 2>/dev/null"; then
    READY=$(( $(date +%s) - T0 )); break
  fi
done
[ "$READY" -gt 0 ] && ok "learner 就绪，用时 ${READY}s（含 JAX/ResNet 初始化）" || bad "learner ${WAIT_START}s 内没就绪"
$SSH "grep -E 'demo buffer|online buffer' $REMOTE_DIR/learner.log | head -2" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g;s/^/  /'

# ---------------------------------------------------------------- 4) actor
note "4) 本地起 actor（经 127.0.0.1 隧道连云端）"
pkill -f "[t]rain_rlpd.py" 2>/dev/null; sleep 1
cd "$REPO/examples"
nohup env -u PYTHONPATH PYTHONPATH="$REPO/examples" \
  "${SO101_LOCAL_PY:-python}" train_rlpd.py \
  --exp_name dummy_test --actor --ip 127.0.0.1 --debug \
  --checkpoint_path /tmp/so101_offline_test/ckpt_actor \
  --demo_path "$CLOUD_ROOT/demos/dummy_demos.pkl" \
  > "$LOCAL_DIR/actor.log" 2>&1 &
ACTOR_PID=$!
echo "  actor pid=$ACTOR_PID"
sleep 8
if grep -q "starting actor loop" "$LOCAL_DIR/actor.log"; then
  ok "actor 起来了"
else
  bad "actor 没起来"; tail -8 "$LOCAL_DIR/actor.log"; fi

# ---------------------------------------------------------------- 5) 握手
note "5) 握手确认（learner 必须把初始网络推给 actor）"
T0=$(date +%s); HS=0
for _ in $(seq 1 60); do
  sleep 1
  if $SSH "grep -q 'sent initial network to actor' $REMOTE_DIR/learner.log"; then
    HS=$(( $(date +%s) - T0 )); break
  fi
done
[ "$HS" -gt 0 ] && ok "握手成功：learner 已把初始网络推给 actor（${HS}s）" || bad "握手没完成"

# ---------------------------------------------------------------- 6) 速率
note "6) 跑 ${WAIT_RUN}s 看双边速率"
sleep "$WAIT_RUN"
$SSH "tr '\r' '\n' < $REMOTE_DIR/learner.log | grep -oE 'learner: *[0-9]+%\|[^]]*\] *[0-9]+/[0-9]+ \[[^]]*\]' | tail -1" \
   > "$LOCAL_DIR/learner_rate.txt" 2>&1
tr '\r' '\n' < "$LOCAL_DIR/actor.log" | grep -oE '[0-9]+/[0-9]+ \[[^]]*\]' | tail -1 > "$LOCAL_DIR/actor_rate.txt"
echo "  learner 最后进度: $(cat "$LOCAL_DIR/learner_rate.txt")"
echo "  actor   最后进度: $(cat "$LOCAL_DIR/actor_rate.txt")"
LEARN_STEPS=$($SSH "tr '\r' '\n' < $REMOTE_DIR/learner.log | grep -oE 'learner: *[0-9]+%' | tail -1 | tr -dc '0-9'")
ACT_STEPS=$(tr '\r' '\n' < "$LOCAL_DIR/actor.log" | grep -oE '[0-9]+/[0-9]+' | tail -1 | cut -d/ -f1)
[ "${LEARN_STEPS:-0}" -gt 0 ] && ok "learner 已在训练（进度 ${LEARN_STEPS}%）" || bad "learner 没有训练进度"
[ "${ACT_STEPS:-0}" -gt 0 ] && ok "actor 已在交互（已完成 ${ACT_STEPS} 步）" || bad "actor 没有步进"

# GPU 占用
$SSH "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader" > "$LOCAL_DIR/gpu.txt" 2>&1
echo "  云端 GPU: $(cat "$LOCAL_DIR/gpu.txt")"

# ---------------------------------------------------------------- 7) 上传
note "7) demo 上传链路"
LOCAL_UP="/tmp/so101_offline_test/upload_$STAMP"
"${SO101_LOCAL_PY:-python}" "$HERE/push_demos.py" \
    recordings_new/2026-09-30_004807_v1/ --out "$LOCAL_UP" --dry-run \
    > "$LOCAL_DIR/upload.txt" 2>&1 || true
if grep -qE "产出：.*pkl" "$LOCAL_DIR/upload.txt"; then
  grep -E "产出：" "$LOCAL_DIR/upload.txt" | sed 's/^/  /'
  ok "build_demos 能产出 demo（--dry-run，未重复上传）"
else
  bad "构建失败"; tail -6 "$LOCAL_DIR/upload.txt"; fi

# ---------------------------------------------------------------- 收尾
note "8) 收尾"
kill "$ACTOR_PID" 2>/dev/null && echo "  已停本地 actor"
$SSH "pkill -f '[t]rain_rlpd.py' || true" && echo "  已停云端 learner"
bash "$HERE/cloud_link.sh" stop >/dev/null 2>&1 && echo "  已停隧道"
# 云端日志拷回来留证
scp -P "$PORT" -o BatchMode=yes -q "$CLOUD:$REMOTE_DIR/learner.log" "$LOCAL_DIR/" 2>/dev/null \
  && echo "  云端日志已拷回 $LOCAL_DIR/learner.log"

echo
echo "=================================================================="
echo " 结果：通过 $PASS 项，失败 $FAIL 项"
echo " 日志：$LOCAL_DIR"
echo "=================================================================="
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
