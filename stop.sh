#!/bin/bash
# 停止专业版（按端口）
PORT="${PUPIL_PORT:-8080}"
if fuser -k -n tcp "$PORT" >/dev/null 2>&1; then
  sleep 1; echo "已停止：端口 $PORT"
else
  echo "端口 $PORT 上没有在运行的服务"
fi
