#!/usr/bin/env bash
# 备份"不可重建"的东西：代码快照 + 患者历史 + 标定 + 基线
#   ./tools/backup.sh [目标目录]      默认 /mnt/backup/pupil_backups
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-/mnt/backup/pupil_backups}"
NAME="$(basename "$HERE")"
STAMP="$(date +%Y%m%d_%H%M)"
OUT="$DEST/${NAME}_${STAMP}.tar.gz"

mkdir -p "$DEST" || { echo "无法创建备份目录: $DEST"; exit 1; }
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/$(basename "$HERE")"

cd "$HERE"
# 1) 代码（用 git 快照，避免带 .git 与产物）
if git rev-parse --git-dir >/dev/null 2>&1; then
  git archive HEAD | tar -x -C "$TMP/$(basename "$HERE")"
  git describe --tags --always > "$TMP/$(basename "$HERE")/GIT_VERSION" 2>/dev/null || \
    git rev-parse --short HEAD > "$TMP/$(basename "$HERE")/GIT_VERSION"
else
  rsync -a --exclude outputs --exclude cache --exclude __pycache__ ./ "$TMP/$(basename "$HERE")/"
fi

# 权重已随 git 提交（远端可拉回），备份里去掉以控制体积
find "$TMP" -name "*.pt" -delete 2>/dev/null || true

# 2) 不可重建的数据
for sub in outputs/patients outputs/stimulus_marks; do
  [ -d "$sub" ] && { mkdir -p "$TMP/$(basename "$HERE")/$(dirname "$sub")"; cp -a "$sub" "$TMP/$(basename "$HERE")/$sub"; }
done
[ -f tools/baseline.json ] && mkdir -p "$TMP/$(basename "$HERE")/tools" && cp tools/baseline.json "$TMP/$(basename "$HERE")/tools/"

tar -czf "$OUT" -C "$TMP" "$(basename "$HERE")"
echo "✅ 已备份: $OUT ($(du -h "$OUT" | cut -f1))"

# 3) 只保留最近 10 份
ls -1t "$DEST"/${NAME}_*.tar.gz 2>/dev/null | tail -n +11 | while read -r old; do
  rm -f "$old"; echo "   清理旧备份: $(basename "$old")"
done
