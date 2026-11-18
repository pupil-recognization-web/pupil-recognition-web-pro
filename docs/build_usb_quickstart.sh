#!/usr/bin/env bash
# 把五份文档 + 快速开始同步到 U 盘（覆盖）
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-/media/$USER/UBUNTU 22_0/YOLO/web}"
[ -d "$DEST" ] || { echo "目标目录不存在: $DEST"; exit 1; }
for b in 01_使用报告_轻量版.pdf 02_使用报告_专业版.pdf 03_技术报告_论文.pdf \
         04_部署手册.pdf 05_双版本介绍.pptx 快速开始.txt; do
  cp -f "$HERE/$b" "$DEST/$b"
  printf "  %-26s → U盘\n" "$b"
done
sync
echo "完成：$DEST"
