#!/usr/bin/env bash
# 安全提交：跑冒烟测试 + 隐私扫描，都通过才提交
#   ./tools/commit.sh -m "改了什么"          # 只提交
#   ./tools/commit.sh -m "改了什么" --push    # 提交并推送
#   ./tools/commit.sh -m "改了什么" --no-test # 跳过冒烟（紧急情况）
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PUPIL_PYTHON:-/mnt/backup/miniconda3/envs/yolo/bin/python}"
[ -x "$PY" ] || PY=python3

MSG=""; PUSH=0; TEST=1
while [ $# -gt 0 ]; do
  case "$1" in
    -m) MSG="${2:-}"; shift 2;;
    --push) PUSH=1; shift;;
    --no-test) TEST=0; shift;;
    *) MSG="$1"; shift;;
  esac
done
cd "$HERE"
[ -n "$MSG" ] || { echo "用法：./tools/commit.sh -m \"提交说明\" [--push] [--no-test]"; exit 1; }

if [ -z "$(git status --porcelain)" ]; then
  echo "没有改动，无需提交"; exit 0
fi

echo "== 1/3 语法检查 =="
if ! "$PY" -m compileall -q . >/tmp/compile.$$ 2>&1; then
  echo "❌ 有文件语法错误："; tail -5 /tmp/compile.$$; rm -f /tmp/compile.$$; exit 1
fi
rm -f /tmp/compile.$$; echo "  ✅ 通过"

echo "== 2/3 隐私扫描 =="
git add -A
if ! "$PY" tools/privacy_scan.py; then
  echo "❌ 隐私扫描未通过，已撤销暂存"; git reset -q; exit 1
fi

if [ "$TEST" -eq 1 ]; then
  echo "== 3/3 冒烟测试（约 30–60 秒）=="
  if ! MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-cache}" "$PY" tools/smoke_test.py >/tmp/smoke.$$ 2>&1; then
    echo "❌ 冒烟未通过，已撤销暂存："; tail -12 /tmp/smoke.$$; rm -f /tmp/smoke.$$; git reset -q; exit 1
  fi
  tail -1 /tmp/smoke.$$; rm -f /tmp/smoke.$$
else
  echo "== 3/3 已跳过冒烟测试（--no-test）=="
fi

git commit -q -m "$MSG" && echo "✅ 已提交：$(git log --oneline | head -1)"
if [ "$PUSH" -eq 1 ]; then
  git push -q origin main && echo "✅ 已推送 $(git remote get-url origin)"
fi
