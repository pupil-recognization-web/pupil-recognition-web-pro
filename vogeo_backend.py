"""VOGeo-Gaze 后端：调用 ``python -m vogeo_gaze``，把结果收进本项目 ``outputs/vogeo_results``。

相对旧版的修正（这正是"3D 模型不生成"的根因）
----------------------------------------------
1. **解释器**：旧版用字符串 ``"python"`` 起子进程。若服务的 PATH 里 ``python``
   不是当前这套环境（例如 conda base 而不是 ``envs/yolo``），子进程会 import 失败、
   returncode≠0，而界面**只 print 到控制台**，用户只看到“overlay 不存在”。现在
   固定用 ``sys.executable`` —— 与启动本服务的解释器完全一致。
2. **设备**：旧版默认写死 ``--device cuda``，没有 GPU / 驱动未加载的机器必失败。
   现在默认不传 ``--device``，交给 VOGeo 自己判断（``PUPIL_DEVICE`` 可强制）。
3. **产物定位**：旧版靠 stdout 的 ``[done]`` 和 4 个猜测路径找结果。现在直接传
   ``--out-dir <临时目录>``，再按目录约定收编，不再猜。
4. **可诊断**：stdout/stderr 全量写到 ``<结果目录>/vogeo.log``，失败时把日志尾部
   一并返回给界面。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from config import VOGE0_DEVICE, VOGE0_OUTPUT_DIR, VOGE0_ROOT, VOGE0_TIMEOUT
import store

OUTPUT_DIR = Path(VOGE0_OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#: 收编的产物文件名
ARTIFACTS = ("gaze.csv", "overlay.mp4", "run_info.json")


def _env():
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(VOGE0_ROOT) + ((":" + prev) if prev else "")
    return env


def _tail(text: str, n: int = 12) -> str:
    lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


def run_vogeo(video_path, out_name=None, device=None, timeout=None):
    """对单个视频跑 VOGeo-Gaze。

    返回 ``(overlay_path|None, gaze_csv|None, status_text)``。
    ``status_text`` 无论成败都会说明发生了什么、日志在哪。
    """
    video = Path(str(video_path))
    stem = out_name or video.stem
    dst_dir = store.vogeo_dir(stem)
    dst_dir.mkdir(parents=True, exist_ok=True)
    log_path = dst_dir / "vogeo.log"

    tmp_out = Path(tempfile.mkdtemp(prefix="vogeo_", dir="/tmp"))
    cmd = [sys.executable, "-m", "vogeo_gaze", str(video),
           "--overlay-only", "--out-dir", str(tmp_out)]
    dev = VOGE0_DEVICE if device is None else str(device)
    if dev:
        cmd += ["--device", dev]
    timeout = timeout or VOGE0_TIMEOUT

    try:
        proc = subprocess.run(cmd, cwd=str(VOGE0_ROOT), env=_env(),
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        msg = f"❌ VOGeo-Gaze 超时（{timeout}s）。可先用更短的视频或设置 PUPIL_VOGEO_TIMEOUT。"
        log_path.write_text(msg + f"\n命令: {' '.join(cmd)}\n", encoding="utf-8")
        return None, None, msg
    except Exception as e:  # noqa: BLE001 - 起不来也要把原因带回去
        msg = f"❌ VOGeo-Gaze 启动失败: {e}"
        log_path.write_text(msg + f"\n命令: {' '.join(cmd)}\n", encoding="utf-8")
        return None, None, msg

    log_path.write_text(
        f"命令: {' '.join(cmd)}\n退出码: {proc.returncode}\n\n"
        f"--- stdout ---\n{proc.stdout or ''}\n--- stderr ---\n{proc.stderr or ''}\n",
        encoding="utf-8")

    if proc.returncode != 0:
        return None, None, (
            f"❌ VOGeo-Gaze 运行失败（returncode={proc.returncode}），日志: {log_path}\n"
            f"{_tail(proc.stderr or proc.stdout)}")

    # 产物目录：默认 <out-dir>/<视频名>，兜底取唯一的子目录
    src_dir = tmp_out / video.stem
    if not src_dir.is_dir():
        subs = [p for p in tmp_out.iterdir() if p.is_dir()]
        src_dir = subs[0] if subs else None
    if src_dir is None:
        return None, None, (
            f"⚠️ VOGeo-Gaze 退出码 0 但没产出目录，日志: {log_path}\n"
            f"{_tail(proc.stdout)}")

    for name in ARTIFACTS:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
    shutil.rmtree(tmp_out, ignore_errors=True)

    overlay = dst_dir / "overlay.mp4"
    gaze_csv = dst_dir / "gaze.csv"
    if not overlay.exists() and not gaze_csv.exists():
        return None, None, (
            f"⚠️ VOGeo-Gaze 跑完但没有 gaze.csv / overlay.mp4，日志: {log_path}\n"
            f"{_tail(proc.stdout)}")

    lines = ["✅ VOGeo-Gaze 完成", f"输出目录: {dst_dir}"]
    if gaze_csv.exists():
        try:
            import csv as _csv
            import numpy as _np
            with open(gaze_csv) as f:
                rows = list(_csv.DictReader(f))
            if rows:
                th = [float(r["theta_h"]) for r in rows if r.get("theta_h") not in ("", None)]
                tv = [float(r["theta_v"]) for r in rows if r.get("theta_v") not in ("", None)]
                if th:
                    lines.append(f"theta_h: 均值 {_np.mean(th):.2f}°，范围 [{min(th):.2f}, {max(th):.2f}]")
                if tv:
                    lines.append(f"theta_v: 均值 {_np.mean(tv):.2f}°，范围 [{min(tv):.2f}, {max(tv):.2f}]")
                lines.append(f"帧数: {len(rows)}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"gaze.csv 解析失败: {e}")
    if not overlay.exists():
        lines.append("⚠️ 没有 overlay.mp4（可能源视频只有一个视角）")

    return (str(overlay) if overlay.exists() else None,
            str(gaze_csv) if gaze_csv.exists() else None,
            "\n".join(lines))


def status_for(stem: str) -> str:
    """查看某个 CSV 对应的 3D 结果状态（给界面用，不触发计算）。"""
    d = OUTPUT_DIR / stem
    if not d.exists():
        return f"未生成：{d} 不存在"
    have = [n for n in ARTIFACTS if (d / n).exists()]
    if not have:
        tail = ""
        log = d / "vogeo.log"
        if log.exists():
            tail = "\n" + _tail(log.read_text(encoding="utf-8", errors="ignore"))
        return f"目录存在但为空（生成失败或中断）：{d}{tail}"
    return f"已有: {', '.join(have)}\n目录: {d}"


def plot_gaze(csv_path, plot_dir=None, pupil_csv=None):
    """画 gaze 静态图：theta_h/theta_v 时序 + 2D 轨迹 + （可选）按瞳孔着色的轨迹。

    返回图片路径列表；``pupil_csv`` 为逐帧分割 CSV，用于叠加瞳孔直径。
    """
    import csv as _csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    out_dir = Path(plot_dir) if plot_dir else OUTPUT_DIR / csv_path.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.parent.name or csv_path.stem

    rows = []
    with open(csv_path) as f:
        for r in _csv.DictReader(f):
            try:
                rows.append({
                    "t": float(r["timestamp"]),
                    "th": float(r["theta_h"]),
                    "tv": float(r["theta_v"]),
                    "blink": int(r.get("blink_index") or 0),
                })
            except Exception:  # noqa: BLE001 - 跳过坏行
                continue
    if not rows:
        return []

    t = [r["t"] for r in rows]
    th = [r["th"] for r in rows]
    tv = [r["tv"] for r in rows]
    blink = [r["blink"] for r in rows]
    plots = []

    # 1. theta_h / theta_v 时序（眨眼段打灰底）
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, vals, label, color in [
        (axes[0], th, "theta_h (deg)", "tab:blue"),
        (axes[1], tv, "theta_v (deg)", "tab:orange"),
    ]:
        ax.plot(t, vals, color=color, lw=1.0)
        for i, b in enumerate(blink):
            if b == 1:
                ax.axvspan(t[i] - 0.05, t[i] + 0.05, color="lightgray", alpha=0.4, lw=0)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[1].set_xlabel("Time (s)")
    axes[0].set_title(f"Gaze angles over time - {stem}")
    fig.tight_layout()
    p = str(out_dir / f"{stem}_gaze_theta.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    plots.append(p)

    # 2. 2D 轨迹
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(th, tv, lw=0.8, alpha=0.7, color="tab:green")
    ax.scatter(th[0], tv[0], color="red", s=30, label="Start", zorder=5)
    ax.scatter(th[-1], tv[-1], color="blue", s=30, label="End", zorder=5)
    ax.set_xlabel("theta_h (deg)")
    ax.set_ylabel("theta_v (deg)")
    ax.set_title(f"Gaze 2D trajectory - {stem}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = str(out_dir / f"{stem}_gaze_traj.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    plots.append(p)

    # 3. 按瞳孔直径着色的轨迹（需要逐帧分割 CSV）
    pupil_data = None
    if pupil_csv and Path(pupil_csv).exists():
        try:
            import pandas as pd
            df = pd.read_csv(pupil_csv)
            if "pupil_mm" in df.columns and df["pupil_mm"].notna().any():
                col, unit = "pupil_mm", "mm"
            elif "major" in df.columns:
                col, unit = "major", "px"
            else:
                col, unit = None, None
            if col:
                valid = df[df["detected"] == 1][["time_sec", col]].dropna()
                if len(valid) >= 2:
                    pupil_data = (valid["time_sec"].tolist(), valid[col].tolist(), unit)
        except Exception as e:  # noqa: BLE001
            print(f"pupil_csv 解析失败: {e}")

    if pupil_data is not None:
        p_t, p_v, p_unit = pupil_data
        p_t_arr, p_v_arr = np.array(p_t), np.array(p_v)
        t_arr, th_arr, tv_arr = np.array(t), np.array(th), np.array(tv)
        colors = np.array([
            p_v_arr[int(np.argmin(np.abs(p_t_arr - ti)))] if len(p_t_arr) else np.nan
            for ti in t_arr
        ])
        if np.any(~np.isnan(colors)):
            fig, ax = plt.subplots(figsize=(6, 6))
            sc = ax.scatter(th_arr, tv_arr, c=colors, cmap="viridis", s=8, alpha=0.7)
            plt.colorbar(sc, ax=ax, label=f"Pupil diameter ({p_unit})")
            ax.set_xlabel("theta_h (deg)")
            ax.set_ylabel("theta_v (deg)")
            ax.set_title(f"Gaze 2D trajectory colored by pupil - {stem}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            p = str(out_dir / f"{stem}_gaze_traj_pupil.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            plots.append(p)

    return plots
