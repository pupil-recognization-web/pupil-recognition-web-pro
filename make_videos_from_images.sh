#!/bin/bash
# 把 openEDS/openEDS/openEDS/<子目录>/ 下的图片序列合成视频
# 输出到 openEDS/openEDS/videos/<子目录>.mp4（H.264）
# 用法: ./make_videos_from_images.sh [源根目录] [输出目录] [fps]

set -e

SRC="${1:-/home/qingque/qingque_yolo/openEDS/openEDS/openEDS}"
OUT="${2:-/home/qingque/qingque_yolo/openEDS/openEDS/videos}"
FPS="${3:-10}"

mkdir -p "$OUT"

if [ ! -d "$SRC" ]; then
  echo "❌ 源目录不存在: $SRC"
  exit 1
fi

# 遍历每个子目录（S_0, S_1, ...）
for sub in "$SRC"/*/; do
  name=$(basename "$sub")
  # 找纯数字 .png（排除 label_*.npy, *.npy）
  mapfile -t imgs < <(find "$sub" -maxdepth 1 -type f -name "*.png" \
    -printf "%f\n" | grep -E '^[0-9]+\.png$' | sort -n)
  if [ ${#imgs[@]} -eq 0 ]; then
    echo "⏭  $name: 无图片，跳过"
    continue
  fi

  # 建临时目录，重命名成 0001.png 0002.png 供 ffmpeg 序列输入
  tmp=$(mktemp -d)
  i=0
  for f in "${imgs[@]}"; do
    i=$((i+1))
    cp "$sub/$f" "$tmp/$(printf '%04d.png' $i)"
  done

  dst="$OUT/${name}.mp4"
  echo "🎬 $name: ${#imgs[@]} 张 -> $dst"
  ffmpeg -y -loglevel error -framerate "$FPS" \
    -i "$tmp/%04d.png" \
    -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p -an \
    "$dst" && echo "✅ $dst" || echo "❌ 失败: $name"

  rm -rf "$tmp"
done

echo ""
echo "========================================"
echo "完成，输出目录: $OUT"
ls -lh "$OUT"/*.mp4 2>/dev/null | tail -20
