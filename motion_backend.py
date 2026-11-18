"""
瞳孔运动特征提取：视频逐帧（可选去模糊）→ YOLO 分割 → 中心/长短轴 → 速度/加速度/面积变化率
"""
import os
import csv
import tempfile
import yaml
from pathlib import Path

import cv2
import numpy as np

from config import DEBLUR_VIDEO_DIR, DEFAULT_MODEL, CONF_THRESHOLD, IOU_THRESHOLD, MOTION_OUT_DIR
from inference import load_model, _process_frame
import deblur_backend as db

MOTION_OUT_DIR = Path(MOTION_OUT_DIR)
MOTION_OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_motion_features(video_path, model_name=DEFAULT_MODEL, conf=CONF_THRESHOLD,
                            iou=IOU_THRESHOLD, max_frames=600,
                            deblur_on=False, deblur_expert="C0", progress=None):
    """
    返回:
      rows: list[list] 逐帧特征
      csv_path: str
      summary: str
      trajectory_png: str
      speed_png: str
    """
    model = load_model(model_name)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_use = min(total, max_frames) if total > 0 and max_frames else total

    rows = []
    prev = None  # (cx, cy, major, minor, t)
    n_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if n_use and n_frames >= n_use:
            break

        if deblur_on:
            frame_proc = db.deblur_frame(frame, deblur_expert)
        else:
            frame_proc = frame

        _, info = _process_frame(frame_proc, model, conf, iou)
        p = info["pupil"]
        cx, cy = (p["center"] if p["center"] else (None, None))
        major, minor = p["major"], p["minor"]
        t = n_frames / fps if fps else 0.0

        area = (np.pi * major * minor / 4) if (major and minor) else None
        speed = accel = None
        if prev is not None and cx is not None and prev[0] is not None:
            dt = t - prev[4]
            if dt > 0:
                dx = cx - prev[0]
                dy = cy - prev[1]
                speed = float(np.hypot(dx, dy) / dt)
                if rows and rows[-1][8] is not None and rows[-1][8] > 0:
                    accel = float((speed - rows[-1][8]) / dt)
        area_rate = None
        if prev is not None and area is not None and prev[3] and prev[2]:
            prev_area = np.pi * prev[2] * prev[3] / 4
            if prev_area > 0:
                area_rate = float((area - prev_area) / prev_area)

        rows.append([
            n_frames, round(t, 3), int(major is not None),
            "" if major is None else round(major, 2),
            "" if minor is None else round(minor, 2),
            "" if cx is None else round(cx, 1),
            "" if cy is None else round(cy, 1),
            "" if area is None else round(area, 1),
            "" if speed is None else round(speed, 2),
            "" if accel is None else round(accel, 2),
            "" if area_rate is None else round(area_rate, 4),
        ])

        prev = (cx, cy, major, minor, t) if cx is not None else None
        n_frames += 1

        if progress is not None and n_frames % 10 == 0:
            try:
                progress(n_frames / max(n_use, 1), desc=f"帧 {n_frames}/{n_use}")
            except Exception:
                pass

    cap.release()

    csv_path = MOTION_OUT_DIR / f"{Path(video_path).stem}_motion.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "time_sec", "detected", "major", "minor", "cx", "cy",
                     "area", "speed_px_s", "accel_px_s2", "area_rate"])
        wr.writerows(rows)

    # 轨迹图
    traj_png = str(MOTION_OUT_DIR / f"{Path(video_path).stem}_trajectory.png")
    pts = [(r[5], r[6]) for r in rows if r[5] != "" and r[6] != ""]
    if len(pts) >= 2:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(xs, ys, lw=0.8, alpha=0.7, color="tab:green")
        ax.scatter(xs[0], ys[0], color="red", s=30, label="Start", zorder=5)
        ax.scatter(xs[-1], ys[-1], color="blue", s=30, label="End", zorder=5)
        ax.set_xlabel("cx (px)")
        ax.set_ylabel("cy (px)")
        ax.set_title(f"Pupil center trajectory")
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(traj_png, dpi=150)
        plt.close(fig)

    # 速度图
    speed_png = str(MOTION_OUT_DIR / f"{Path(video_path).stem}_speed.png")
    spd_pts = [(r[1], r[8]) for r in rows if r[8] != ""]
    if len(spd_pts) >= 2:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ts = [p[0] for p in spd_pts]
        vs = [p[1] for p in spd_pts]
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(ts, vs, lw=1.0, color="tab:red")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (px/s)")
        ax.set_title("Pupil center speed")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(speed_png, dpi=150)
        plt.close(fig)

    det = sum(1 for r in rows if r[2] == 1)
    speeds = [r[8] for r in rows if r[8] != ""]
    summary = (
        f"视频: {Path(video_path).name}\n"
        f"帧数: {len(rows)}，检出: {det}（{det/max(len(rows),1)*100:.1f}%）\n"
        f"速度均值: {np.mean(speeds):.2f} px/s\n" if speeds else
        f"视频: {Path(video_path).name}\n帧数: {len(rows)}，检出: {det}\n"
    )
    if speeds:
        summary += f"速度最大: {np.max(speeds):.2f} px/s\n"
    summary += f"CSV: {csv_path}"

    return rows, str(csv_path), summary, traj_png, speed_png


def deblur_video_only(video_path, expert="general_v2", max_frames=600, progress=None):
    """
    纯去模糊：逐帧 DeblurUNet，输出清晰视频，不跑 YOLO。
    返回: (out_path, summary)
    """
    import subprocess
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_use = min(total, max_frames) if total > 0 and max_frames else total

    out_dir = Path(DEBLUR_VIDEO_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{Path(video_path).stem}_{expert}_deblur.mp4"
    raw = out_dir / f"{name}.raw.mp4"
    final = out_dir / name

    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if n_use and n >= n_use:
            break
        out = db.deblur_frame(frame, expert)
        if out.shape[:2] != (h, w):
            out = cv2.resize(out, (w, h))
        writer.write(out)
        n += 1
        if progress is not None and n % 10 == 0:
            try:
                progress(n / max(n_use, 1), desc=f"去模糊 {n}/{n_use}")
            except Exception:
                pass
    cap.release()
    writer.release()

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(final)],
            check=True,
        )
        os.remove(raw)
    except Exception as e:
        print(f"ffmpeg 失败: {e}，保留 {raw}")
        final = raw

    summary = f"输入: {Path(video_path).name}\n专家: {expert}\n帧数: {n}\n输出: {final}"
    return str(final), summary
