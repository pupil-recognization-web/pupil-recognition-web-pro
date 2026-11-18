#!/usr/bin/env python3
"""对清晰视频重跑分割，并对 ``detected == 0`` 的帧做光流补全。

与旧版的区别：模型加载/单帧推理复用 :mod:`inference`，补帧复用
:mod:`flow_fill_backend`，不再各留一份拷贝。
"""
from __future__ import annotations

import argparse
import os
import subprocess

import cv2

from config import CONF_THRESHOLD, DEFAULT_MODEL, IOU_THRESHOLD
from flow_fill_backend import draw, fill_masks, mask_from_ellipse
from inference import _process_frame, load_model


def main() -> int:
    ap = argparse.ArgumentParser(description="清晰视频分割 + 漏检帧光流补全")
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    ap.add_argument("--iou", type=float, default=IOU_THRESHOLD)
    ap.add_argument("--max_frames", type=int, default=2000)
    ap.add_argument("--out", default="/tmp/filled_clear.mp4")
    ap.add_argument("--csv", default="/tmp/filled_clear.csv")
    args = ap.parse_args()

    model = load_model(args.model)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_use = min(total, args.max_frames) if total > 0 and args.max_frames else total

    frames, masks, grays, detected = [], [], [], []
    majors, minors, centers = [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or (n_use and i >= n_use):
            break
        _, info = _process_frame(frame, model, args.conf, args.iou)
        p = info["pupil"]
        major, minor, center = p["major"], p["minor"], p["center"]

        frames.append(frame)
        grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        masks.append(
            mask_from_ellipse((h, w), center[0], center[1], major, minor)
            if major is not None and center is not None else None
        )
        detected.append(1 if major is not None else 0)
        majors.append(major)
        minors.append(minor)
        centers.append(center)
        i += 1
    cap.release()
    print(f"帧数: {len(frames)}，原始检出: {sum(detected)}")

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

    with open(args.csv, "w", newline="") as f:
        f.write("frame,detected_orig,detected_filled,major,minor,cx,cy\n")
        for k in range(len(frames)):
            cx = centers[k][0] if centers[k] is not None else ""
            cy = centers[k][1] if centers[k] is not None else ""
            f.write(f"{k},{detected[k]},{1 if filled[k] is not None else 0},"
                    f"{majors[k] if majors[k] is not None else ''},"
                    f"{minors[k] if minors[k] is not None else ''},{cx},{cy}\n")
    print(f"✅ CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
