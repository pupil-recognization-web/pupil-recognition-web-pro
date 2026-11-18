"""产物路径的统一入口。

以前"某个 CSV 对应的视频/ gaze / 标定文件在哪"这套规则散落在
analysis_backend、app.py、vogeo_backend、pdf_export 各处，各自拼路径、各自猜文件名；
改一次目录结构要改五处。这里收口成一组函数，其它模块只调用它们。
"""
from __future__ import annotations

import json
from pathlib import Path

from config import PDF_DIR, PLOT_DIR, STIMULUS_DIR, VIDEO_OUTPUT_DIR, VOGE0_OUTPUT_DIR


def seg_dir() -> Path:
    return Path(VIDEO_OUTPUT_DIR)


def plot_dir() -> Path:
    return Path(PLOT_DIR)


def pdf_dir() -> Path:
    return Path(PDF_DIR)


def csv_path(name: str) -> Path:
    return seg_dir() / name


def stim_path(video_stem: str) -> Path:
    return Path(STIMULUS_DIR) / f"{video_stem}.json"


def run_sidecar(csv_name: str) -> Path:
    return seg_dir() / f"{Path(csv_name).stem}.run.json"


def vogeo_dir(csv_stem: str) -> Path:
    return Path(VOGE0_OUTPUT_DIR) / csv_stem


def gaze_csv(csv_stem: str) -> Path:
    return vogeo_dir(csv_stem) / "gaze.csv"


def overlay_mp4(csv_stem: str) -> Path:
    return vogeo_dir(csv_stem) / "overlay.mp4"


def find_video(csv_name: str):
    """CSV → 源视频。优先推理时写的 run.json，再按命名约定找几个候选。"""
    sidecar = run_sidecar(csv_name)
    if sidecar.exists():
        try:
            v = json.loads(sidecar.read_text(encoding="utf-8")).get("video")
            if v and Path(v).exists():
                return Path(v)
        except Exception:  # noqa: BLE001
            pass

    stem = Path(csv_name).stem
    candidates = [
        seg_dir() / f"{stem}.mp4",
        seg_dir() / f"{stem}_h264.mp4",
        seg_dir() / f"{stem}_base_h264.mp4",
        seg_dir() / f"{stem}_plr_h264.mp4",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None
