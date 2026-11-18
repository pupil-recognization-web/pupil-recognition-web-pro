#!/usr/bin/env bash
# 一眼看健康状态：端口、内存、最近错误、日志大小
# 用法：./tools/healthcheck.sh [端口...]    默认检查 8080 8081
set -u
PORTS=("$@"); [ ${#PORTS[@]} -eq 0 ] && PORTS=(8080 8081)
HERE="$(cd "$(dirname "$0")/.." && pwd)"
FAIL=0

echo "== 服务端口 =="
for p in "${PORTS[@]}"; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://127.0.0.1:$p/" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then echo "  ✅ $p 正常"; else echo "  ❌ $p 无响应（$code）"; FAIL=1; fi
done

echo "== 内存 =="
free -h | awk 'NR==1||NR==2'
avail=$(free -m | awk '/Mem|内存/{print $7}')
if [ -n "$avail" ] && [ "$avail" -lt 800 ]; then
  echo "  ⚠️ 可用内存仅 ${avail} MB，可能被 OOM Kill（建议先停一个服务）"
fi

echo "== 进程（PID / 内存 / 端口）=="
ps -eo pid,rss,args | grep "[p]ython.*app\.py" | while read -r pid rss args; do
  mb=$((rss / 1024))
  cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null)
  printf '  PID %-7s %5s MB  %s\n' "$pid" "$mb" "$(basename "${cwd:-?}")"
done || echo "  （无 app.py 进程）"

echo "== 日志 =="
for log in "$HERE/app.log" "$HERE/app.log.1"; do
  [ -f "$log" ] && printf '  %-40s %s\n' "$log" "$(du -h "$log" | cut -f1)"
done

echo "== 最近错误（app.log 尾部含 Traceback 的行）=="
if [ -f "$HERE/app.log" ]; then
  # 只看最近 200 行，避免历史噪音
  tail -200 "$HERE/app.log" | grep -c "Traceback" | while read -r n; do
    if [ "$n" -eq 0 ]; then echo "  （最近 200 行无错误）"; else echo "  最近 200 行有 $n 处 Traceback（详见 app.log）"; fi
  done
else
  echo "  （无日志）"
fi

exit $FAIL
