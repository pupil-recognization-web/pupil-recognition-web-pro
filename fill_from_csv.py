#!/usr/bin/env python3
"""按已有 CSV 的 ``detected`` 标记补帧（不重复跑 YOLO）。

用途：某次推理跑完发现漏检段，想只对 ``detected == 0`` 的帧补 mask，
同时保留原有检出帧的椭圆结果。

补帧逻辑本身在 :mod:`flow_fill_backend`，本文件只负责读写与命令行。
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path

import cv2

from flow_fill_backend import draw, fill_masks, mask_from_ellipse


def read_csv(csv_path):
    """读逐帧 CSV 的 ``(frame, detected, major, minor, cx, cy)``。"""
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "frame": int(r["frame"]),
                "detected": int(r["detected"]),
                "major": float(r["major"]) if r["major"] not in ("", None) else None,
                "minor": float(r["minor"]) if r["minor"] not in ("", None) else None,
                "cx": float(r["cx"]) if r["cx"] not in ("", None) else None,
                "cy": float(r["cy"]) if r["cy"] not in ("", None) else None,
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="按已有 CSV 补帧")
    ap.add_argument("--video", required=True, help="原视频")
    ap.add_argument("--csv", required=True, help="已有的逐帧 CSV")
    ap.add_argument("--out", default="/tmp/filled_from_csv.mp4")
    ap.add_argument("--out_csv", default="/tmp/filled_from_csv.csv")
    args = ap.parse_args()

    rows = read_csv(args.csv)
    detected = [r["detected"] for r in rows]
    print(f"CSV 帧数: {len(rows)}，detected=1: {sum(detected)}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames, masks, grays = [], [], []
    for idx in range(len(rows)):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        masks.append(
            mask_from_ellipse((h, w), rows[idx]["cx"], rows[idx]["cy"],
                              rows[idx]["major"], rows[idx]["minor"])
            if detected[idx] == 1 else None
        )
    cap.release()
    print(f"读取视频帧: {len(frames)}")

    filled = fill_masks(frames, masks, detected, grays=grays)
    n_fixed = sum(1 for k in range(len(frames)) if detected[k] == 0 and filled[k] is not None)
    print(f"补全帧数: {n_fixed}")

    raw = args.out.replace(".mp4", "_raw.mp4")
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for k, frame in enumerate(frames):
        writer.write(draw(frame, filled[k] if detected[k] == 0 else masks[k]))
    writer.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", args.out], check=True)
    os.remove(raw)
    print(f"✅ 视频: {args.out}")

    with open(args.out_csv, "w", newline="") as f:
        f.write("frame,detected_orig,detected_filled\n")
        for k in range(len(frames)):
            f.write(f"{k},{detected[k]},{1 if filled[k] is not None else 0}\n")
    print(f"✅ CSV: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
