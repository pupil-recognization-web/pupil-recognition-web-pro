#!/usr/bin/env bash
# 一键体检：健康状态 + 冒烟测试
#   ./tools/check.sh            # 服务在跑时也能用（冒烟会多加载一份模型）
#   ./tools/check.sh --stop     # 先停服务再跑，腾出内存
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PUPIL_PYTHON:-/mnt/backup/miniconda3/envs/yolo/bin/python}"
[ -x "$PY" ] || PY=python3

if [ "${1:-}" = "--stop" ]; then
  echo "== 先停服务 =="
  "$HERE/stop.sh" || true
  sleep 1
fi

echo
"$HERE/tools/healthcheck.sh" "$@" || true
echo
echo "== 冒烟测试 =="
cd "$HERE" && MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-cache}" "$PY" tools/smoke_test.py "$@" 2>&1 | tail -6
