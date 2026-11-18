"""
瞳孔分割结果分析：读 CSV，出统计表 + 医学风格图。
兼容普通 CSV（8 列）和 PLR CSV（12 列，含 iris_detected/iris_major/iris_minor/pupil_mm）。
"""
import csv
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    OUTPUT_DIR, PLOT_DIR, QUALITY_DETECT_RATE_FAIR, QUALITY_DETECT_RATE_GOOD,
    RELIABLE_DETECT_RATE, STIMULUS_DIR, VIDEO_OUTPUT_DIR,
)
import plr_core
import store

CSV_DIR = Path(VIDEO_OUTPUT_DIR)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


def list_csv_files():
    if not CSV_DIR.exists():
        return []
    files = list(CSV_DIR.glob("*.csv"))
    def key(f):
        m = re.search(r"S_(\d+)", f.name)
        return int(m.group(1)) if m else 0
    return [f.name for f in sorted(files, key=key)]


def _find_video_for_csv(csv_name):
    """CSV → 源视频；具体规则在 store.find_video（run.json 优先，其次命名约定）。"""
    return store.find_video(csv_name)


def _read_rows(csv_name):
    path = CSV_DIR / csv_name
    if not path.exists():
        return None
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        has_plr = "iris_major" in (reader.fieldnames or [])
        for r in reader:
            try:
                row = {
                    "frame": int(r["frame"]),
                    "time_sec": float(r["time_sec"]),
                    "detected": int(r["detected"]),
                    "major": float(r["major"]) if r["major"] not in ("", None) else None,
                    "minor": float(r["minor"]) if r["minor"] not in ("", None) else None,
                    "cx": float(r["cx"]) if r["cx"] not in ("", None) else None,
                    "cy": float(r["cy"]) if r["cy"] not in ("", None) else None,
                    "conf": float(r["conf"]),
                }
                if has_plr:
                    row["iris_detected"] = int(r.get("iris_detected") or 0)
                    row["iris_major"] = float(r["iris_major"]) if r.get("iris_major") not in ("", None) else None
                    row["iris_minor"] = float(r["iris_minor"]) if r.get("iris_minor") not in ("", None) else None
                    row["pupil_mm"] = float(r["pupil_mm"]) if r.get("pupil_mm") not in ("", None) else None
                rows.append(row)
            except Exception:
                continue
    return rows


def _is_plr(rows):
    return any("iris_major" in r for r in rows[:1])


def _plot_fresh(out_path, csv_path) -> bool:
    """图比数据新就复用，避免同一份 CSV 反复点分析时重画（约 100 ms/图）。"""
    try:
        o = Path(out_path)
        c = Path(csv_path) if Path(csv_path).is_absolute() else CSV_DIR / csv_path
        return o.exists() and c.exists() and o.stat().st_mtime >= c.stat().st_mtime
    except Exception:  # noqa: BLE001
        return False


def _plot_timeseries(rows, csv_name):
    out = Path(PLOT_DIR) / f"{csv_name.replace('.csv', '')}_timeseries.png"
    if _plot_fresh(out, csv_name):
        return str(out)
    times = [r["time_sec"] for r in rows]
    majors = [r["major"] if r["major"] is not None else np.nan for r in rows]
    minors = [r["minor"] if r["minor"] is not None else np.nan for r in rows]

    fig, ax = plt.subplots(figsize=(10, 4))
    for r in rows:
        if r["detected"] == 0:
            ax.axvspan(r["time_sec"] - 0.05, r["time_sec"] + 0.05,
                       color="lightgray", alpha=0.5, lw=0)
    ax.plot(times, majors, label="Long axis (major)", color="tab:blue", lw=1.2)
    ax.plot(times, minors, label="Short axis (minor)", color="tab:orange", lw=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pupil diameter (px)")
    ax.set_title(f"Pupil diameter over time - {csv_name}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def _plot_distribution(rows, csv_name):
    out = Path(PLOT_DIR) / f"{csv_name.replace('.csv', '')}_dist.png"
    if _plot_fresh(out, csv_name):
        return str(out)
    majors = [r["major"] for r in rows if r["detected"] == 1 and r["major"] is not None]
    minors = [r["minor"] for r in rows if r["detected"] == 1 and r["minor"] is not None]
    if not majors:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(majors, bins=30, color="tab:blue", alpha=0.75, edgecolor="white")
    axes[0].set_title("Long axis distribution")
    axes[0].set_xlabel("Diameter (px)")
    axes[0].set_ylabel("Frame count")
    axes[1].hist(minors, bins=30, color="tab:orange", alpha=0.75, edgecolor="white")
    axes[1].set_title("Short axis distribution")
    axes[1].set_xlabel("Diameter (px)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def _plot_trajectory(rows, csv_name):
    out = Path(PLOT_DIR) / f"{csv_name.replace('.csv', '')}_trajectory.png"
    if _plot_fresh(out, csv_name):
        return str(out)
    pts = [(r["cx"], r["cy"]) for r in rows if r["detected"] == 1 and r["cx"] is not None and r["cy"] is not None]
    if len(pts) < 2:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(xs, ys, lw=0.8, alpha=0.7, color="tab:green")
    ax.scatter(xs[0], ys[0], color="red", s=30, label="Start", zorder=5)
    ax.scatter(xs[-1], ys[-1], color="blue", s=30, label="End", zorder=5)
    ax.set_xlabel("cx (px)")
    ax.set_ylabel("cy (px)")
    ax.set_title(f"Pupil center trajectory - {csv_name}")
    ax.invert_yaxis()
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def _plot_plr(rows, csv_name):
    """PLR 时序图：瞳孔 mm、虹膜长轴、双轴叠加"""
    has_pupil_mm = any(r.get("pupil_mm") is not None for r in rows)
    has_iris = any(r.get("iris_major") is not None for r in rows)
    if not has_pupil_mm and not has_iris:
        return None

    times = [r["time_sec"] for r in rows]
    pupil_mm = [r.get("pupil_mm") if r.get("pupil_mm") is not None else np.nan for r in rows]
    iris_major = [r.get("iris_major") if r.get("iris_major") is not None else np.nan for r in rows]
    iris_minor = [r.get("iris_minor") if r.get("iris_minor") is not None else np.nan for r in rows]

    fig, ax1 = plt.subplots(figsize=(10, 4))
    color1 = "tab:blue"
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Pupil diameter (mm)", color=color1)
    ax1.plot(times, pupil_mm, color=color1, lw=1.3, label="Pupil (mm)")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    if has_iris:
        ax2 = ax1.twinx()
        color2 = "tab:green"
        ax2.set_ylabel("Iris diameter (px)", color=color2)
        ax2.plot(times, iris_major, color=color2, lw=1.0, alpha=0.8, label="Iris major (px)")
        ax2.plot(times, iris_minor, color="tab:olive", lw=1.0, alpha=0.6, label="Iris minor (px)")
        ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    if has_iris:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax1.legend(loc="best")

    ax1.set_title(f"PLR view (pupil mm + iris px) - {csv_name}")
    fig.tight_layout()
    out = Path(PLOT_DIR) / f"{csv_name.replace('.csv', '')}_plr.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def analyze_csv(csv_name):
    """返回 (summary_table, frame_table, abnormal_table, summary_text, images)"""
    if not csv_name:
        return [], [], [], "请选择 CSV", []
    rows = _read_rows(csv_name)
    if rows is None:
        return [], [], [], f"文件不存在: {csv_name}", []
    if not rows:
        return [], [], [], "CSV 为空或格式不对", []

    plr_mode = _is_plr(rows)

    n = len(rows)
    det = [r for r in rows if r["detected"] == 1]
    majors = [r["major"] for r in det if r["major"] is not None]
    minors = [r["minor"] for r in det if r["minor"] is not None]
    confs = [r["conf"] for r in det]

    # 视频与模型概况
    summary = []

    # --- 视频参数 ---
    video_path = _find_video_for_csv(csv_name)
    if video_path is not None:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        duration = (total / fps) if fps > 0 else 0.0
        summary += [
            ["--- 视频参数 ---", ""],
            ["文件名", video_path.name],
            ["分辨率", f"{vw} x {vh}"],
            ["帧率 (fps)", f"{fps:.2f}"],
            ["总帧数", total],
            ["时长 (s)", f"{duration:.2f}"],
            ["处理帧数", n],
        ]
    else:
        summary += [
            ["--- 视频参数 ---", ""],
            ["文件名", "（未找到对应视频）"],
            ["处理帧数", n],
        ]

    # --- 模型检出质量 ---
    summary += [
        ["--- 模型检出质量 ---", ""],
        ["检出帧数", len(det)],
        ["未检出帧数", n - len(det)],
        ["检出率", f"{len(det)/n*100:.1f}%"],
    ]
    if confs:
        summary += [
            ["置信度均值", f"{np.mean(confs):.3f}"],
            ["置信度最小", f"{min(confs):.3f}"],
            ["置信度标准差", f"{np.std(confs):.3f}"],
            ["低置信度帧数 (conf<0.5)", sum(1 for c in confs if c < 0.5)],
        ]

    # 最长连续未检出段、段数量
    max_gap = 0
    cur_gap = 0
    n_gaps = 0
    in_gap = False
    for r in rows:
        if r["detected"] == 0:
            cur_gap += 1
            if not in_gap:
                n_gaps += 1
                in_gap = True
            max_gap = max(max_gap, cur_gap)
        else:
            cur_gap = 0
            in_gap = False
    summary += [
        ["最长连续未检出帧数", max_gap],
        ["未检出段数量", n_gaps],
    ]

    # --- 分段检出率 ---
    if n >= 3:
        third = n // 3
        segs = [(0, third, "前 1/3"), (third, 2*third, "中 1/3"), (2*third, n, "后 1/3")]
        summary += [["--- 分段检出率 ---", ""]]
        for a, b, nm in segs:
            if b > a:
                seg_det = sum(1 for r in rows[a:b] if r["detected"] == 1)
                summary.append([nm, f"{seg_det/(b-a)*100:.1f}%"])

    # 逐帧表：PLR 模式 12 列，非 PLR 8 列补空
    if plr_mode:
        frame_table = []
        for r in rows[:200]:
            frame_table.append([
                r["frame"], r["time_sec"], r["detected"],
                "" if r["major"] is None else round(r["major"], 2),
                "" if r["minor"] is None else round(r["minor"], 2),
                "" if r["cx"] is None else round(r["cx"], 1),
                "" if r["cy"] is None else round(r["cy"], 1),
                round(r["conf"], 3),
                r.get("iris_detected", 0),
                "" if r.get("iris_major") is None else round(r["iris_major"], 2),
                "" if r.get("iris_minor") is None else round(r["iris_minor"], 2),
                "" if r.get("pupil_mm") is None else round(r["pupil_mm"], 2),
            ])
    else:
        frame_table = []
        for r in rows[:200]:
            frame_table.append([
                r["frame"], r["time_sec"], r["detected"],
                "" if r["major"] is None else round(r["major"], 2),
                "" if r["minor"] is None else round(r["minor"], 2),
                "" if r["cx"] is None else round(r["cx"], 1),
                "" if r["cy"] is None else round(r["cy"], 1),
                round(r["conf"], 3),
                "", "", "", "",   # PLR 四列占位（非 PLR 模式为空）
            ])

    abnormal = []
    if majors and minors:
        m_mean, m_std = np.mean(majors), np.std(majors)
        n_mean, n_std = np.mean(minors), np.std(minors)
        for r in det:
            if r["major"] is None or r["minor"] is None:
                continue
            if (m_std > 0 and abs(r["major"] - m_mean) > 2 * m_std) or \
               (n_std > 0 and abs(r["minor"] - n_mean) > 2 * n_std):
                abnormal.append([r["frame"], round(r["major"], 2), round(r["minor"], 2),
                                 round(r["conf"], 3), "长短轴偏离"])
    for r in rows:
        if r["detected"] == 0:
            abnormal.append([r["frame"], "", "", round(r["conf"], 3), "未检出"])
        elif plr_mode and r.get("iris_detected") == 0:
            abnormal.append([r["frame"],
                             "" if r["major"] is None else round(r["major"], 2),
                             "" if r["minor"] is None else round(r["minor"], 2),
                             round(r["conf"], 3), "虹膜缺失"])

    images = []
    ts = _plot_timeseries(rows, csv_name)
    if ts:
        images.append(ts)
    dist = _plot_distribution(rows, csv_name)
    if dist:
        images.append(dist)
    traj = _plot_trajectory(rows, csv_name)
    if traj:
        images.append(traj)

    mode_text = "PLR（pupil + iris）" if plr_mode else "普通（pupil only）"
    text = (f"文件: {csv_name}\n"
            f"模式: {mode_text}\n"
            f"总帧数 {n}，检出 {len(det)}，检出率 {len(det)/n*100:.1f}%\n"
            f"异常帧 {len(abnormal)} 个")
    return summary, frame_table, abnormal, text, images


# ============================================================
# PLR 参数提取
# ============================================================
def _load_stimulus_time(csv_name):
    """从 outputs/stimulus_marks 读刺激时刻，返回 ``(秒, 文件路径)`` 或 ``(None, None)``。

    旧版用 ``csv_stem.startswith(video_stem)`` 做前缀匹配，``1_1`` 会误配到
    ``1_10``；现在只接受“完全相同”或“视频名 + 分隔符”开头，并在多个候选里
    取最长（最具体）的一个。精确匹配额外加权，避免 ``S_1`` 抢走 ``S_10`` 的标定。
    """
    import json

    marks_dir = Path(STIMULUS_DIR)
    if not marks_dir.exists():
        return None, None

    csv_stem = Path(csv_name).stem
    best = None  # (匹配长度, 文件, 时刻)
    for f in sorted(marks_dir.glob("*.json")):
        try:
            with open(f) as fp:
                data = json.load(fp)
        except Exception:  # noqa: BLE001 - 坏文件跳过即可
            continue
        video_stem = Path(data.get("video", "")).stem
        if not video_stem:
            continue
        if csv_stem == video_stem:
            match_len = len(video_stem) + 1
        elif csv_stem.startswith(video_stem) and csv_stem[len(video_stem)] in "_-.":
            match_len = len(video_stem)
        else:
            continue
        t = data.get("current")
        try:
            t = float(t) if t is not None else None
        except (TypeError, ValueError):
            t = None
        if t is None or t <= 0:
            continue
        if best is None or match_len > best[0]:
            best = (match_len, f, t)

    if best is None:
        return None, None
    return best[2], str(best[1])


def analyze_plr(csv_name, stimulus_time=None, plot_dir=None):
    """提取 14 项 PLR 参数并出图。

    返回 ``(params_table, summary_text, plot_path)``。
    计算全部委托给 :mod:`plr_core`（与命令行 ``plr_params.py`` 同一份实现）。

    刺激时刻优先级：手动输入 > stimulus_marks 标定文件 > 自动检测（可信度低）。
    """
    path = CSV_DIR / csv_name
    if not path.exists():
        return [], f"文件不存在: {csv_name}", None

    stim_source = None
    if stimulus_time is None or stimulus_time <= 0:
        t_mark, mark_file = _load_stimulus_time(csv_name)
        if t_mark is not None:
            stimulus_time = t_mark
            stim_source = f"刺激标定文件 ({Path(mark_file).name})"

    result = plr_core.analyze(path, stimulus_time=stimulus_time,
                              stimulus_source=stim_source)
    if result is None:
        return [], (f"刺激后窗口无有效数据（需 ≥{plr_core.PLR_MIN_VALID_POINTS} 帧）或有效点太少，"
                    "无法提取 PLR 参数；请核对刺激标定与视频质量"), None

    curve, params, quality = result.curve, result.params, result.quality
    table = plr_core.params_table(params, curve.unit, quality=quality, response=result.response)

    plot_path = None
    if plot_dir:
        out_dir = Path(plot_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(curve.t, curve.d, color="tab:blue", lw=1.5,
                label=f"Pupil diameter ({curve.unit})")
        ax.axvline(curve.stimulus_time, color="green", ls="--", lw=1.2,
                   label=f"Stimulus ({curve.stimulus_time:.2f}s)")
        ax.axvline(curve.t_min, color="red", ls="--", lw=1.2,
                   label=f"Min ({curve.t_min:.2f}s)")
        if curve.recovery_idx is not None:
            ax.axvline(curve.t_recovery, color="orange", ls="--", lw=1.2,
                       label=f"Recovery 50% ({curve.t_recovery:.2f}s)")
        if curve.min_idx > curve.stim_idx:
            ax.axvspan(curve.stimulus_time, curve.t_min, color="red", alpha=0.05)
        # 刺激后 2 s 收缩窗（v2 的判定范围）
        ax.axvspan(curve.stimulus_time, curve.stimulus_time + 2.0, color="green", alpha=0.04)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Pupil diameter ({curve.unit})")
        ax.set_title(f"PLR analysis - {csv_name}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        plot_path = str(out_dir / f"{Path(csv_name).stem}_plr_analysis.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    mode = "PLR（pupil_mm）" if curve.unit == "mm" else "普通（major, px）"
    latency_txt = f"{params.latency:.2f}s" if np.isfinite(params.latency) else "未检出"
    if result.response:
        mcv_txt = f"{params.mcv:.3f} {curve.unit}/s"
        npi_txt = f"{params.npi_like}（0–5，≥3 正常）"
    else:
        mcv_txt = "—（未引出收缩）"
        npi_txt = "不适用（未引出收缩）"
    text = (f"文件: {csv_name}\n"
            f"模式: {mode}（{quality.mode}）\n"
            f"刺激时间: {curve.stimulus_time:.2f}s（{curve.stimulus_source}）\n"
            f"响应判定: {'✅ 有收缩' if result.response else '⚠️ 未引出（Δ<5%）'}\n"
            f"收缩幅度: {params.amplitude:.3f} {curve.unit}（{params.percent:.1f}%）\n"
            f"MCV: {mcv_txt}｜潜伏期: {latency_txt}\n"
            f"NPi 近似: {npi_txt}")
    if quality.notes:
        text += "\n\n质量提示:\n" + "\n".join(f"- {n}" for n in quality.notes)

    return table, text, plot_path


def reliability_warnings(csv_name):
    """数据可靠性提示（检出率过低 / 非 PLR 模式 / 无刺激标定）。

    列表为空表示没有需要提醒的问题；界面把非空结果放在“概览”最前面，
    PDF 里也据此加一句限定，避免把 1.3% 检出率算出来的 NPi 当结论用。
    """
    rows = _read_rows(csv_name)
    if not rows:
        return []

    n = len(rows)
    det = sum(1 for r in rows if r["detected"] == 1)
    rate = det / n * 100 if n else 0.0

    out = []
    if rate < RELIABLE_DETECT_RATE:
        out.append(f"⚠️ 检出率仅 {rate:.1f}%（{det}/{n} 帧），PLR/临床判读结论不可靠，"
                   "建议先检查视频质量或 ROI 是否裁对眼睛。")
    elif rate < QUALITY_DETECT_RATE_FAIR:
        out.append(f"⚠️ 检出率 {rate:.1f}% 偏低，PLR 曲线可能有断点，解读需谨慎。")

    if not _is_plr(rows):
        out.append("ℹ️ 该 CSV 无 pupil_mm 列（非 PLR 模式），直径以像素计，"
                   "以 mm 为口径的临床参考范围不适用。")

    t_mark, _ = _load_stimulus_time(csv_name)
    if t_mark is None:
        out.append("ℹ️ 未找到刺激标定文件，刺激时刻为自动推断，潜伏期/收缩时间可能失真。")

    return out


# ============================================================
# 模块一：纯像素形态与运动分析（不判读病症）
# ============================================================
def analyze_morphology(csv_name, plot_dir=None):
    """
    从逐帧 CSV 计算形态与运动信息，不做病症判读。
    返回 (morph_table, motion_table, summary_text, plot_paths)
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not csv_name:
        return [], [], "请先选择 CSV", []
    rows = _read_rows(csv_name)
    if rows is None:
        return [], [], f"文件不存在: {csv_name}", []
    if not rows:
        return [], [], "CSV 为空或格式不对", []

    n = len(rows)
    det = [r for r in rows if r["detected"] == 1]
    n_det = len(det)

    majors = [r["major"] for r in det if r["major"] is not None]
    minors = [r["minor"] for r in det if r["minor"] is not None]

    # 面积、长短轴比（逐检出帧）
    areas = []
    ratios = []
    for r in det:
        if r["major"] is not None and r["minor"] is not None:
            a = np.pi * r["major"] * r["minor"] / 4.0
            areas.append(a)
            if r["major"] > 0:
                ratios.append(r["minor"] / r["major"])

    # 形态表
    morph = []
    morph.append(["总帧数", n])
    morph.append(["检出帧数", n_det])
    morph.append(["检出率", f"{n_det/n*100:.1f}%"])
    if majors:
        morph.append(["长轴均值 (px)", f"{np.mean(majors):.2f}"])
        morph.append(["长轴标准差", f"{np.std(majors):.2f}"])
        morph.append(["长轴最小/最大", f"{min(majors):.1f} / {max(majors):.1f}"])
    if minors:
        morph.append(["短轴均值 (px)", f"{np.mean(minors):.2f}"])
        morph.append(["短轴标准差", f"{np.std(minors):.2f}"])
        morph.append(["短轴最小/最大", f"{min(minors):.1f} / {max(minors):.1f}"])
    if areas:
        morph.append(["面积均值 (px²)", f"{np.mean(areas):.1f}"])
        morph.append(["面积标准差", f"{np.std(areas):.1f}"])
    if ratios:
        morph.append(["长短轴比均值", f"{np.mean(ratios):.3f}"])
        morph.append(["长短轴比标准差", f"{np.std(ratios):.3f}"])
        morph.append(["长短轴比最小/最大", f"{min(ratios):.3f} / {max(ratios):.3f}"])

    # 虹膜信息（如果有）
    if any("iris_major" in r for r in rows[:1]):
        iris_det = [r for r in rows if r.get("iris_detected") == 1 and r.get("iris_major") is not None]
        iris_majors = [r["iris_major"] for r in iris_det]
        iris_minors = [r["iris_minor"] for r in iris_det if r.get("iris_minor") is not None]
        pupil_iris_ratios = []
        for r in rows:
            if r.get("iris_major") and r["major"] is not None and r["iris_major"] > 0:
                pupil_iris_ratios.append(r["major"] / r["iris_major"])
        morph.append(["虹膜检出帧数", len(iris_det)])
        if iris_majors:
            morph.append(["虹膜长轴均值 (px)", f"{np.mean(iris_majors):.2f}"])
        if iris_minors:
            morph.append(["虹膜短轴均值 (px)", f"{np.mean(iris_minors):.2f}"])
        if pupil_iris_ratios:
            morph.append(["瞳孔/虹膜比例均值", f"{np.mean(pupil_iris_ratios):.3f}"])
            morph.append(["瞳孔/虹膜比例标准差", f"{np.std(pupil_iris_ratios):.3f}"])

    # 运动：速度、加速度、面积变化率（逐检出帧，时间相邻）
    times = [r["time_sec"] for r in rows]
    speeds = []
    accels = []
    area_rates = []
    prev = None
    prev_speed = None
    for r in rows:
        if r["detected"] != 1 or r["cx"] is None or r["cy"] is None:
            prev = None
            prev_speed = None
            continue
        if prev is not None:
            dt = r["time_sec"] - prev["t"]
            if dt > 0:
                dx = r["cx"] - prev["x"]
                dy = r["cy"] - prev["y"]
                sp = np.hypot(dx, dy) / dt
                speeds.append(sp)
                if prev_speed is not None:
                    accels.append((sp - prev_speed) / dt)
                prev_speed = sp
                # 面积变化率
                if r["major"] and r["minor"] and prev["major"] and prev["minor"]:
                    a_cur = np.pi * r["major"] * r["minor"] / 4.0
                    a_prev = np.pi * prev["major"] * prev["minor"] / 4.0
                    if a_prev > 0:
                        area_rates.append((a_cur - a_prev) / a_prev / dt * 100.0)
        prev = {"t": r["time_sec"], "x": r["cx"], "y": r["cy"],
                "major": r["major"], "minor": r["minor"]}

    # 轨迹长度、稳定性
    pts = [(r["cx"], r["cy"]) for r in det if r["cx"] is not None and r["cy"] is not None]
    traj_len = 0.0
    for i in range(1, len(pts)):
        traj_len += np.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1])
    stability = None
    if len(pts) >= 2:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        stability = float(np.std(xs) + np.std(ys))

    motion = []
    if speeds:
        motion.append(["速度均值 (px/s)", f"{np.mean(speeds):.2f}"])
        motion.append(["速度最大 (px/s)", f"{max(speeds):.2f}"])
        motion.append(["速度标准差", f"{np.std(speeds):.2f}"])
    if accels:
        motion.append(["加速度均值 (px/s²)", f"{np.mean(accels):.2f}"])
        motion.append(["加速度最大 (px/s²)", f"{max(accels):.2f}"])
    if area_rates:
        motion.append(["面积变化率均值 (%/s)", f"{np.mean(area_rates):.3f}"])
        motion.append(["面积变化率最大 (%/s)", f"{max(area_rates):.3f}"])
    motion.append(["轨迹长度 (px)", f"{traj_len:.1f}"])
    if stability is not None:
        motion.append(["注视稳定性（离散度, px）", f"{stability:.2f}"])

    # 图表
    plots = []
    if plot_dir:
        import os
        from pathlib import Path as _Path
        out_dir = _Path(plot_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = _Path(csv_name).stem

        # 三张形态图只依赖 CSV：都比 CSV 新就整块跳过（省 ~300 ms）
        _expected = [out_dir / f"{stem}_morph_trajectory.png",
                     out_dir / f"{stem}_morph_diameter.png",
                     out_dir / f"{stem}_morph_speed.png"]
        _skip = all(_plot_fresh(x, csv_name) for x in _expected)
        if _skip:
            plots = [str(x) for x in _expected]

        # 轨迹图
        if not _skip and len(pts) >= 2:
            fig, ax = plt.subplots(figsize=(5, 5))
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, lw=0.8, alpha=0.7, color="tab:green")
            ax.scatter(xs[0], ys[0], color="red", s=30, label="Start", zorder=5)
            ax.scatter(xs[-1], ys[-1], color="blue", s=30, label="End", zorder=5)
            ax.set_xlabel("cx (px)")
            ax.set_ylabel("cy (px)")
            ax.set_title(f"Pupil center trajectory - {csv_name}")
            ax.invert_yaxis()
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            p = str(out_dir / f"{stem}_morph_trajectory.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            plots.append(p)

        # 直径曲线
        t_det = [r["time_sec"] for r in det if r["major"] is not None]
        m_det = [r["major"] for r in det if r["major"] is not None]
        if not _skip and len(t_det) >= 2:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(t_det, m_det, lw=1.2, color="tab:blue", label="Major axis (px)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Pupil major axis (px)")
            ax.set_title(f"Pupil diameter over time - {csv_name}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            p = str(out_dir / f"{stem}_morph_diameter.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            plots.append(p)

        # 速度曲线
        if not _skip and speeds:
            # 速度对应的时间：取每次计算速度的当前帧时间
            t_speeds = []
            prev_t = None
            for r in rows:
                if r["detected"] != 1 or r["cx"] is None:
                    prev_t = None
                    continue
                if prev_t is not None:
                    t_speeds.append(r["time_sec"])
                prev_t = r["time_sec"]
            t_speeds = t_speeds[:len(speeds)]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(t_speeds, speeds, lw=1.0, color="tab:red")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Speed (px/s)")
            ax.set_title(f"Pupil center speed - {csv_name}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            p = str(out_dir / f"{stem}_morph_speed.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            plots.append(p)

    summary = (f"文件: {csv_name}\n"
               f"总帧数 {n}，检出 {n_det}，检出率 {n_det/n*100:.1f}%\n"
               f"轨迹长度 {traj_len:.1f} px"
               + (f"，速度均值 {np.mean(speeds):.2f} px/s" if speeds else ""))

    return morph, motion, summary, plots


# ============================================================
# 模块四（第一档）：单眼临床判读
# ============================================================
def clinical_interpretation(csv_name, gaze_csv=None, stimulus_time=None, full=False):
    """
    从 CSV 直接生成单眼临床判读（第一档：单眼 + PLR 参数）。
    返回 (interpret_table, summary_text)。
    interpret_table: [[判读项, 结果, 依据], ...]
    """
    if not csv_name:
        return [], "请先选择 CSV"

    # 调 analyze_plr 拿 14 项参数
    plr_params, plr_text, _ = analyze_plr(csv_name, stimulus_time=stimulus_time, plot_dir=None)
    if not plr_params:
        return [], "PLR 参数提取失败，无法判读"

    # 把 [[参数名, 值, 参考范围?], ...] 转成 dict，兼容两元素和三元素
    p = {}
    unit = "px"
    for item in plr_params:
        if len(item) >= 2:
            k, v = item[0], item[1]
        else:
            continue
        p[k] = v
        if "(mm)" in str(k):
            unit = "mm"

    def _get(prefix):
        for k, v in p.items():
            if k.startswith(prefix):
                try:
                    return float(v)
                except Exception:
                    return None
        return None

    # v2 新增行：响应判定 / 数据质量
    response_flag = None
    quality_note = ""
    for k, v in p.items():
        if k.startswith("响应判定"):
            response_flag = "未引出" not in str(v)
        elif k.startswith("数据质量"):
            quality_note = str(v)

    D0 = _get("基线直径")
    Dmin = _get("最小直径")
    amp = _get("收缩幅度")
    pct = _get("收缩百分比")
    mcv = _get("最大收缩速度")
    npi = _get("NPi 近似评分")
    latency = _get("潜伏期")

    rows = []

    # 1. 直径判读
    if D0 is not None:
        if D0 < 2.0:
            rows.append(["针尖样瞳孔", "⚠️ 是", f"基线直径 {D0:.2f}{unit} < 2mm；排查阿片类、脑桥病变、有机磷、缩瞳药"])
        elif D0 > 5.0:
            rows.append(["瞳孔散大", "⚠️ 是", f"基线直径 {D0:.2f}{unit} > 5mm；排查脑疝、动眼神经受压、药物、暗环境"])
            if D0 >= 9.0:
                rows.append(["散大程度", "⚠️ 极度散大（>9mm）",
                             f"直径 {D0:.2f}{unit} > 9mm；需除外药物（阿托品/东莨菪碱）、眼球外伤与虹膜损伤"])
            elif D0 >= 6.0:
                rows.append(["散大程度", "⚠️ 散大固定区间（6–9mm）",
                             f"直径 {D0:.2f}{unit} 在 6–9mm；提示晚期脑疝/脑干功能衰竭，"
                             "若伴对光反射消失须紧急降颅压（脑死亡需整套规范评估，不能单靠瞳孔）"])
            else:
                rows.append(["散大程度", "⚠️ 轻度散大（5–6mm）", f"直径 {D0:.2f}{unit} 在 5–6mm；需结合对光反射判断"])
        else:
            rows.append(["直径正常", "✅ 是", f"基线直径 {D0:.2f}{unit} 在 2–5mm 范围"])
    else:
        rows.append(["直径判读", "—", "无有效直径数据（需双类模型或瞳孔检出）"])

    # 2. 对光反射判读（v2：先看"有没有引出收缩"，再谈强弱）
    if response_flag is False:
        detail = ("刺激后 2 s 内瞳孔未见收缩" if pct <= 0.05
                  else f"刺激后 2 s 内收缩仅 {pct:.1f}%（阈值 5%）")
        rows.append(["对光反射未引出", "⚠️ 需复核",
                     f"{detail}；先核对刺激时刻/光源/录像亮度，不能据此直接判为反射消失"])
    elif amp is not None and pct is not None:
        if pct < 5.0:
            rows.append(["对光反射消失", "⚠️ 是", f"收缩幅度 {amp:.2f}{unit}（{pct:.1f}% 基线）< 5%；提示脑疝/视神经损伤/中脑损害"])
        elif pct < 15.0:
            rows.append(["对光反射减弱", "⚠️ 部分", f"收缩幅度 {amp:.2f}{unit}（{pct:.1f}% 基线）偏低"])
        else:
            rows.append(["对光反射存在", "✅ 是", f"收缩幅度 {amp:.2f}{unit}（{pct:.1f}% 基线）"])
    else:
        rows.append(["对光反射判读", "—", "无收缩幅度数据（需光刺激或自动检测）"])

    # 3. MCV 判读（未引出收缩时 MCV 无意义）
    if response_flag is False:
        rows.append(["MCV 判读", "—", "未引出收缩，MCV 不可评估（先确认刺激/光源）"])
    elif mcv is not None:
        if mcv < 0.5:
            rows.append(["MCV 极低", "⚠️ 是", f"最大收缩速度 {mcv:.3f}{unit}/s；提示反应迟钝/强直性瞳孔"])
        elif mcv < 1.0:
            rows.append(["MCV 偏低", "⚠️ 部分", f"最大收缩速度 {mcv:.3f}{unit}/s"])
        else:
            rows.append(["MCV 正常", "✅ 是", f"最大收缩速度 {mcv:.3f}{unit}/s"])
    else:
        rows.append(["MCV 判读", "—", "无 MCV 数据"])

    # 4. NPi 判读
    if response_flag is False:
        rows.append(["NPi 判读", "—", "未引出收缩，NPi 不适用"])
    elif npi is not None:
        if npi < 3.0:
            rows.append(["NPi 异常", "⚠️ 是", f"NPi 近似 {npi} < 3；提示神经功能异常"])
        else:
            rows.append(["NPi 正常", "✅ 是", f"NPi 近似 {npi} ≥ 3"])
    else:
        rows.append(["NPi 判读", "—", "无 NPi 数据"])

    # 5. PLR 变差（直径正常但 MCV 低/NPi 低）
    if D0 is not None and 2.0 <= D0 <= 5.0:
        if (mcv is not None and mcv < 1.0) or (npi is not None and npi < 3.0):
            rows.append(["PLR 变差（尺寸正常）", "⚠️ 是",
                         "直径正常但 MCV 低/NPi<3；提示早期颅压高、视神经损伤、脑病"])
        else:
            rows.append(["PLR 变差（尺寸正常）", "✅ 否", "直径正常且 MCV/NPi 正常"])

    # 5b. 单眼提示（Horner/脑疝/阿-罗/强直性）
    if D0 is not None and 1.0 <= D0 <= 2.0:
        rows.append(["小瞳提示（Horner 鉴别）", "⚠️ 是", f"直径 {D0:.2f}{unit} 在 1–2mm，需排查 Horner（结合眼睑/面部少汗）"])
    elif full:
        rows.append(["小瞳提示（Horner 鉴别）", "✅ 否", f"直径 {D0:.2f}{unit} 不在 1–2mm" if D0 is not None else "无直径数据"])
    if D0 is not None and D0 > 5.0 and pct is not None and pct < 5.0:
        rows.append(["散大固定提示（脑疝/动眼神经）", "⚠️ 是", f"直径 {D0:.2f}{unit} > 5mm 且对光反射消失，需排查脑疝/动眼神经受压"])
    elif full:
        rows.append(["散大固定提示（脑疝/动眼神经）", "✅ 否", "未同时满足 >5mm + 反射消失"])
    if pct is not None and pct < 5.0:
        rows.append(["阿-罗瞳孔鉴别", "⚠️ 需人工", "对光反射消失，需人工查调节反射（阿-罗：对光反射消失但调节反射存在）"])
    elif full:
        rows.append(["阿-罗瞳孔鉴别", "✅ 否", "对光反射未消失，不支持阿-罗瞳孔"])
    if mcv is not None and mcv < 0.5:
        rows.append(["强直性瞳孔鉴别", "⚠️ 需人工", "MCV 极低，需人工查调节反射（强直性：收缩极慢、调节反射也慢）"])
    elif full:
        rows.append(["强直性瞳孔鉴别", "✅ 否", f"MCV {mcv:.2f} mm/s 正常，不支持强直性瞳孔"])

    # 5c. 视线相关病症提示（单眼可提示，确诊需眼动/临床）
    if D0 is not None and 1.0 <= D0 <= 2.0:
        rows.append(["Horner 综合征（瞳孔部分）提示", "⚠️ 需排查", f"直径 {D0:.2f}{unit} 在 1–2mm，需排查 Horner（结合眼睑下垂/面部少汗）"])
    else:
        rows.append(["Horner 综合征（瞳孔部分）提示", "✅ 否", "直径不在 1–2mm"])

    # 6. 形态判读（长短轴比）
    # 从 CSV 里读 major/minor，算长短轴比均值
    rows_csv = _read_rows(csv_name)
    if rows_csv:
        ratios = []
        for r in rows_csv:
            if r["detected"] == 1 and r["major"] and r["minor"] and r["major"] > 0:
                ratios.append(r["minor"] / r["major"])
        if ratios:
            import numpy as np
            ratio_mean = float(np.mean(ratios))
            if ratio_mean < 0.8:
                rows.append(["形态欠规则", "⚠️ 是", f"长短轴比均值 {ratio_mean:.3f} < 0.8；提示阿-罗瞳孔/虹膜异常"])
            else:
                rows.append(["形态规则", "✅ 是", f"长短轴比均值 {ratio_mean:.3f} ≥ 0.8"])
            if ratio_mean >= 0.95:
                rows.append(["圆度描述", "✅ 接近圆形", f"长短轴比 {ratio_mean:.3f} ≥ 0.95"])
            elif ratio_mean >= 0.85:
                rows.append(["圆度描述", "✅ 轻度椭圆", f"长短轴比 {ratio_mean:.3f} 在 0.85–0.95"])
            elif ratio_mean >= 0.80:
                rows.append(["圆度描述", "⚠️ 明显椭圆", f"长短轴比 {ratio_mean:.3f} 在 0.80–0.85；建议结合 gaze 漂移判断是否假性"])

    # 7. gaze 稳定性（质控，需要 gaze.csv）
    if gaze_csv is not None and Path(gaze_csv).exists():
        try:
            import csv as _csv
            import numpy as np
            th_list, tv_list = [], []
            with open(gaze_csv) as f:
                reader = _csv.DictReader(f)
                for r in reader:
                    try:
                        th_list.append(float(r["theta_h"]))
                        tv_list.append(float(r["theta_v"]))
                    except Exception:
                        continue
            if len(th_list) >= 10:
                th_std = float(np.std(th_list))
                tv_std = float(np.std(tv_list))
                th_range = float(np.max(th_list) - np.min(th_list))
                tv_range = float(np.max(tv_list) - np.min(tv_list))
                if th_std > 2.0 or tv_std > 2.0:
                    rows.append(["gaze 稳定性（PLR 质控）", "⚠️ 漂移",
                                 f"theta_h std={th_std:.2f}°, theta_v std={tv_std:.2f}°；视线漂移，PLR 参数可信度低"])
                else:
                    rows.append(["gaze 稳定性（PLR 质控）", "✅ 稳定",
                                 f"theta_h std={th_std:.2f}°, theta_v std={tv_std:.2f}°；视线稳定"])
            else:
                rows.append(["gaze 稳定性（PLR 质控）", "—", "gaze 数据不足（<10 帧）"])
        except Exception as e:
            rows.append(["gaze 稳定性（PLR 质控）", "—", f"gaze.csv 解析失败: {e}"])
    else:
        rows.append(["gaze 稳定性（PLR 质控）", "—", "无 gaze.csv（需 VOGeo-Gaze 结果）"])

    # 综合建议
    warnings = [r for r in rows if r[1].startswith("⚠️")]
    if warnings:
        summary = f"检出 {len(warnings)} 项异常提示：\n" + "\n".join(f"- {w[0]}：{w[2]}" for w in warnings)
    else:
        summary = "未检出明显异常（基于单眼 + PLR 参数第一档判读）。"

    summary += ("\n\n注：本判读为单眼第一档，未含双眼对比、调节反射、眼睑、眼动。最终诊断需临床结合。"
                "\n依据共识：双侧差 >1mm、瞳孔散大、对光反射消失提示 ICP 升高/脑疝，需紧急降颅压；"
                "新发变化比长期稳定的不等大更危险。")
    if quality_note:
        summary += f"\n数据质量：{quality_note}"

    return rows, summary




# ============================================================
# 双眼叠加对比图
# ============================================================
def compare_eyes_plots(csv_left, csv_right, plot_dir=None):
    """
    左右眼叠加对比图。返回 (plot_paths, summary_text)。
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows_l = _read_rows(csv_left)
    rows_r = _read_rows(csv_right)
    if not rows_l or not rows_r:
        return [], "CSV 读取失败"

    plots = []
    if plot_dir:
        out_dir = Path(plot_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"dual_{Path(csv_left).stem}_vs_{Path(csv_right).stem}"

        # 1. 直径-时间叠加
        tl = [r["time_sec"] for r in rows_l if r["detected"] == 1 and r["major"] is not None]
        ml = [r["major"] for r in rows_l if r["detected"] == 1 and r["major"] is not None]
        tr = [r["time_sec"] for r in rows_r if r["detected"] == 1 and r["major"] is not None]
        mr = [r["major"] for r in rows_r if r["detected"] == 1 and r["major"] is not None]
        if tl and tr:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(tl, ml, color="tab:blue", lw=1.2, label="Left")
            ax.plot(tr, mr, color="tab:orange", lw=1.2, label="Right")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Pupil major axis (px)")
            ax.set_title(f"Diameter overlay - {Path(csv_left).stem} vs {Path(csv_right).stem}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            p = str(out_dir / f"{stem}_diameter.png")
            fig.savefig(p, dpi=150); plt.close(fig)
            plots.append(p)

        # 2. 速度-时间叠加
        def speeds(rows):
            ts, vs = [], []
            prev = None
            for r in rows:
                if r["detected"] != 1 or r["cx"] is None:
                    prev = None
                    continue
                if prev is not None:
                    dt = r["time_sec"] - prev["t"]
                    if dt > 0:
                        sp = np.hypot(r["cx"] - prev["x"], r["cy"] - prev["y"]) / dt
                        ts.append(r["time_sec"]); vs.append(sp)
                prev = {"t": r["time_sec"], "x": r["cx"], "y": r["cy"]}
            return ts, vs
        tsl, vsl = speeds(rows_l)
        tsr, vsr = speeds(rows_r)
        if tsl and tsr:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(tsl, vsl, color="tab:blue", lw=1.0, label="Left")
            ax.plot(tsr, vsr, color="tab:orange", lw=1.0, label="Right")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Speed (px/s)")
            ax.set_title("Speed overlay")
            ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            p = str(out_dir / f"{stem}_speed.png")
            fig.savefig(p, dpi=150); plt.close(fig)
            plots.append(p)

        # 3. 轨迹叠加（以各自中心为原点归一化）
        def traj(rows):
            pts = [(r["cx"], r["cy"]) for r in rows if r["detected"] == 1 and r["cx"] is not None]
            if not pts:
                return [], []
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            mx, my = np.mean(xs), np.mean(ys)
            return [x - mx for x in xs], [y - my for y in ys]
        xl, yl = traj(rows_l)
        xr, yr = traj(rows_r)
        if xl and xr:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(xl, yl, color="tab:blue", lw=0.8, alpha=0.7, label="Left")
            ax.plot(xr, yr, color="tab:orange", lw=0.8, alpha=0.7, label="Right")
            ax.set_xlabel("cx (px, centered)")
            ax.set_ylabel("cy (px, centered)")
            ax.set_title("Trajectory overlay (centered)")
            ax.invert_yaxis()
            ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            p = str(out_dir / f"{stem}_trajectory.png")
            fig.savefig(p, dpi=150); plt.close(fig)
            plots.append(p)

        # 4. 长短轴分布叠加
        if ml and mr:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(ml, bins=30, color="tab:blue", alpha=0.5, label="Left")
            ax.hist(mr, bins=30, color="tab:orange", alpha=0.5, label="Right")
            ax.set_xlabel("Major axis (px)")
            ax.set_ylabel("Frame count")
            ax.set_title("Major axis distribution overlay")
            ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            p = str(out_dir / f"{stem}_dist.png")
            fig.savefig(p, dpi=150); plt.close(fig)
            plots.append(p)

        # 5. 差值曲线（按帧对齐，左 - 右）
        n = min(len(rows_l), len(rows_r))
        t_diff, d_diff = [], []
        for i in range(n):
            a = rows_l[i]; b = rows_r[i]
            if a["detected"] == 1 and b["detected"] == 1 and a["major"] is not None and b["major"] is not None:
                t_diff.append(a["time_sec"])
                d_diff.append(a["major"] - b["major"])
        if t_diff:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(t_diff, d_diff, color="tab:purple", lw=1.0)
            ax.axhline(0, color="gray", lw=0.8, ls="--")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Left - Right major axis (px)")
            ax.set_title("Diameter difference (Left - Right)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            p = str(out_dir / f"{stem}_diff.png")
            fig.savefig(p, dpi=150); plt.close(fig)
            plots.append(p)

    summary = f"左眼: {csv_left}\n右眼: {csv_right}\n对比图 {len(plots)} 张"
    return plots, summary


# ============================================================
# 双眼 PLR 曲线叠加
# ============================================================
def compare_eyes_plr_plots(csv_left, csv_right, plot_dir=None, stim_time_l=None, stim_time_r=None):
    """左右眼 PLR 曲线叠加（瞳孔直径随时间 + 刺激/最小值/恢复点标注）。"""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows_l = _read_rows(csv_left)
    rows_r = _read_rows(csv_right)
    if not rows_l or not rows_r:
        return []

    def series(rows):
        if any("pupil_mm" in r for r in rows[:1]) and any(r.get("pupil_mm") is not None for r in rows):
            col, unit = "pupil_mm", "mm"
        else:
            col, unit = "major", "px"
        t = [r["time_sec"] for r in rows if r["detected"] == 1 and r.get(col) is not None]
        d = [r[col] for r in rows if r["detected"] == 1 and r.get(col) is not None]
        return t, d, unit

    tl, dl, unit_l = series(rows_l)
    tr, dr, unit_r = series(rows_r)
    if not tl or not tr:
        return []
    unit = unit_l if unit_l == unit_r else unit_l

    plots = []
    if plot_dir:
        out_dir = Path(plot_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"dual_{Path(csv_left).stem}_vs_{Path(csv_right).stem}"

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(tl, dl, color="tab:blue", lw=1.2, label="Left")
        ax.plot(tr, dr, color="tab:orange", lw=1.2, label="Right")
        if stim_time_l is not None:
            ax.axvline(stim_time_l, color="tab:blue", ls="--", lw=1.0, alpha=0.6,
                       label=f"Left stimulus ({stim_time_l:.2f}s)")
        if stim_time_r is not None:
            ax.axvline(stim_time_r, color="tab:orange", ls="--", lw=1.0, alpha=0.6,
                       label=f"Right stimulus ({stim_time_r:.2f}s)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Pupil diameter ({unit})")
        ax.set_title("PLR curve overlay (Left vs Right)")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = str(out_dir / f"{stem}_plr_overlay.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        plots.append(p)

        # 归一化到基线（百分比变化）
        def norm(t, d):
            if not d: return [], []
            base = np.median(d[: max(3, len(d)//10)])
            if base == 0: return t, [0]*len(d)
            return t, [(x / base - 1) * 100 for x in d]
        tnl, dnl = norm(tl, dl)
        tnr, dnr = norm(tr, dr)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(tnl, dnl, color="tab:blue", lw=1.2, label="Left")
        ax.plot(tnr, dnr, color="tab:orange", lw=1.2, label="Right")
        if stim_time_l is not None:
            ax.axvline(stim_time_l, color="tab:blue", ls="--", lw=1.0, alpha=0.6,
                       label=f"Left stimulus ({stim_time_l:.2f}s)")
        if stim_time_r is not None:
            ax.axvline(stim_time_r, color="tab:orange", ls="--", lw=1.0, alpha=0.6,
                       label=f"Right stimulus ({stim_time_r:.2f}s)")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pupil change vs baseline (%)")
        ax.set_title("Normalized PLR overlay")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = str(out_dir / f"{stem}_plr_norm.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        plots.append(p)

    return plots, f"PLR 对比图 {len(plots)} 张"

# ============================================================
# PPT 病症分类对照（静态表）
# ============================================================
def ppt_reference_single(interp_rows=None):
    """
    PPT 里单眼能判的病症对照。
    interp_rows 是 clinical_interpretation 返回的 [[判读项, 结果, 依据], ...]。
    每行病症都给明确判读结果（✅ 是 / ✅ 否 / ⚠️ 需人工），不留 —。
    """
    idx = {}
    if interp_rows:
        for r in interp_rows:
            if len(r) >= 2:
                idx[r[0]] = (r[1], r[2] if len(r) > 2 else "")

    def find(keyword):
        for k, v in idx.items():
            if keyword in k:
                return v
        return None

    ref = [
        ("正常瞳孔（2–5mm，反射灵敏）", "直径正常", "直径 + 收缩幅度", "未检出直径异常"),
        ("双侧针尖样瞳孔（<2mm）", "针尖样瞳孔", "直径", "直径未 <2mm"),
        ("双侧瞳孔散大固定（6–9mm，反射消失）", "散大固定提示", "直径 + 收缩幅度", "未同时满足 >5mm + 反射消失"),
        ("双侧瞳孔散大对称（反射尚存）", "瞳孔散大", "直径 + 收缩幅度", "直径未 >5mm"),
        ("对光反射消失", "对光反射消失", "收缩幅度 / MCV", "收缩幅度未 <5% 基线"),
        ("PLR 变差（尺寸正常但 NPi 下降）", "PLR 变差", "MCV + NPi", "MCV/NPi 未达异常阈值"),
        ("强直性瞳孔（Adie，收缩极慢）", "强直性瞳孔鉴别", "MCV + 调节反射", "MCV 未极低"),
        ("阿-罗瞳孔（对光反射消失 + 调节反射存在）", "阿-罗瞳孔鉴别", "对光反射 + 调节反射", "对光反射未消失"),
        ("形态欠规则（阿-罗瞳孔形态）", "形态欠规则", "长短轴比 / 圆度", "长短轴比未 <0.8"),
        ("动眼神经麻痹（瞳孔受累型）提示", "动眼神经麻痹（瞳孔受累型）提示", "单侧/双侧散大 + 反射消失", "直径未 >5mm 或对光反射未消失"),
        ("动眼神经麻痹（瞳孔回避型）提示", "动眼神经麻痹（瞳孔回避型）提示", "瞳孔正常 + 需眼动", "瞳孔异常或反射未达正常"),
        ("Horner 综合征（瞳孔部分）提示", "Horner 综合征（瞳孔部分）提示", "直径 1–2mm + 眼睑/少汗", "直径不在 1–2mm"),
        ("PLR 质控（视线稳定）", "gaze 稳定性", "theta_h/theta_v 标准差 < 2°", "视线漂移或无非 gaze 数据"),
    ]

    out = []
    for item in ref:
        name, keyword, basis = item[0], item[1], item[2]
        neg = item[3] if len(item) > 3 else "未触发"
        v = find(keyword)
        if v is None:
            out.append([name, "✅ 否", neg, basis])
        else:
            result, reason = v
            out.append([name, result, reason, basis])
    return out



def ppt_reference_dual(dual_rows=None):
    """
    PPT 里需要双眼才能判的病症对照。
    dual_rows 是 _dual_interp 返回的 [[判读项, 左眼, 右眼, 差值, 判读], ...]。
    每行病症都给明确判读结果，不留 —。
    """
    def to_float(x):
        try:
            return float(str(x).replace("%", "").replace("s", "").replace("mm", "").strip())
        except Exception:
            return None

    idx = {}
    if dual_rows:
        for r in dual_rows:
            if len(r) >= 5:
                idx[r[0]] = (r[1], r[2], r[3], r[4])

    def find(keyword):
        for k, v in idx.items():
            if keyword in k:
                return v
        return None

    def judge(name, keyword, basis, threshold, neg_reason):
        v = find(keyword)
        if v is None:
            return [name, "✅ 否", "无对应数据", basis]
        a, b, d_str, verdict = v
        compare = f"左 {a} / 右 {b} / 差 {d_str}"
        return [name, verdict, compare, basis]

    # 生理性不等大单独处理（差 <1mm 是 ✅）
    v = find("基线直径")
    if v:
        a, b, d_str, _ = v
        d = to_float(d_str)
        if d is not None and d < 1.0:
            physio = ["生理性不等大（<1mm）", f"✅ 是（差 {d_str} < 1mm）", f"左 {a} / 右 {b} / 差 {d_str}", "双侧直径差 <1mm"]
        else:
            physio = ["生理性不等大（<1mm）", f"✅ 否（差 {d_str} ≥ 1mm）", f"左 {a} / 右 {b} / 差 {d_str}", "双侧直径差 <1mm"]
    else:
        physio = ["生理性不等大（<1mm）", "✅ 否", "无对应数据", "双侧直径差 <1mm"]

    return [
        judge("病理性不等大（>1mm）", "基线直径", "双侧直径差 >1mm", 1.0, "差 ≤1mm"),
        physio,
        judge("单侧散大（脑疝预警）", "单侧散大", "单侧 >5mm + 双侧差 >1mm", 1.0, "未触发"),
        judge("单侧缩小（Horner 瞳孔部分）", "单侧缩小", "单侧 1–2mm + 双侧差 >1mm", 1.0, "未触发"),
        judge("双侧散大（双侧 >5mm）", "双侧散大", "双侧 >5mm", None, "非双侧 >5mm"),
        judge("双侧 MCV 差（单侧神经受损）", "最大收缩速度", "MCV 差 >0.5", 0.5, "差 ≤0.5"),
        judge("双侧 NPi 差（单侧反应异常）", "NPi 近似评分", "NPi 差 >0.5", 0.5, "差 ≤0.5"),
        judge("双侧收缩幅度差（单侧反应弱）", "收缩百分比", "收缩百分比差 >10%", 10.0, "差 ≤10%"),
        judge("双侧潜伏期差（单侧传导慢）", "潜伏期", "潜伏期差 >0.1s", 0.1, "差 ≤0.1s"),
        judge("双侧恢复时间差（单侧恢复慢）", "恢复时间", "恢复时间差 >0.5s", 0.5, "差 ≤0.5s"),
        judge("眼外肌运动受限（示意性）", "双眼 gaze 水平差", "双眼 gaze 水平差 >5°（示意，需同人同步）", 5.0, "差 ≤5° 或 无数据"),
        judge("复视（示意性）", "双眼 gaze 水平差", "双眼 gaze 不一致（示意，需同人同步）", 5.0, "差 ≤5° 或 无数据"),
        judge("滑车神经麻痹（垂直位差异，示意性）", "双眼 gaze 垂直差", "垂直 gaze 差 >5°（示意，需同人同步）", 5.0, "差 ≤5° 或 无数据"),
    ]




def compare_eyes_gaze_plots(l_gaze_csv, r_gaze_csv, plot_dir=None):
    """
    双眼 gaze 对比图：theta_h/theta_v 时序叠加 + 2D 轨迹叠加。
    返回 plot_paths。
    """
    import csv as _csv
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def read_gaze(p):
        t, th, tv = [], [], []
        with open(p) as f:
            for r in _csv.DictReader(f):
                try:
                    t.append(float(r["timestamp"]))
                    th.append(float(r["theta_h"]))
                    tv.append(float(r["theta_v"]))
                except Exception:
                    continue
        return np.array(t), np.array(th), np.array(tv)

    l_t, l_th, l_tv = read_gaze(l_gaze_csv)
    r_t, r_th, r_tv = read_gaze(r_gaze_csv)
    if len(l_t) == 0 or len(r_t) == 0:
        return []

    if plot_dir is None:
        plot_dir = PLOT_DIR
    out_dir = Path(plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"dual_gaze_{Path(l_gaze_csv).parent.name}_vs_{Path(r_gaze_csv).parent.name}"

    plots = []

    # 1. theta_h/theta_v 时序叠加（双子图）
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(l_t, l_th, color="tab:blue", lw=1.0, label="Left")
    axes[0].plot(r_t, r_th, color="tab:orange", lw=1.0, label="Right")
    axes[0].set_ylabel("theta_h (deg)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"Gaze horizontal overlay - Left vs Right")
    axes[1].plot(l_t, l_tv, color="tab:blue", lw=1.0, label="Left")
    axes[1].plot(r_t, r_tv, color="tab:orange", lw=1.0, label="Right")
    axes[1].set_ylabel("theta_v (deg)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    p = str(out_dir / f"{stem}_theta_overlay.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    plots.append(p)

    # 2. 2D 轨迹叠加（各自中心化）
    def center(th, tv):
        return th - np.mean(th), tv - np.mean(tv)
    l_th_c, l_tv_c = center(l_th, l_tv)
    r_th_c, r_tv_c = center(r_th, r_tv)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(l_th_c, l_tv_c, color="tab:blue", lw=0.8, alpha=0.7, label="Left")
    ax.plot(r_th_c, r_tv_c, color="tab:orange", lw=0.8, alpha=0.7, label="Right")
    ax.set_xlabel("theta_h centered (deg)")
    ax.set_ylabel("theta_v centered (deg)")
    ax.set_title("Gaze 2D trajectory overlay (centered)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = str(out_dir / f"{stem}_traj_overlay.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    plots.append(p)

    return plots


def data_quality_summary(csv_name, gaze_csv=None):
    """返回数据质量摘要 [[指标, 值, 结论], ...]，供网页和 PDF 共用。"""
    rows = _read_rows(csv_name)
    if not rows:
        return []
    n = len(rows)
    det = [r for r in rows if r["detected"] == 1]
    out = []

    # 检出率（阈值集中在 config，避免和界面/PDF 各写一套）
    rate = len(det) / n * 100 if n else 0
    if rate >= QUALITY_DETECT_RATE_GOOD:
        verdict = "Good"
    elif rate >= QUALITY_DETECT_RATE_FAIR:
        verdict = "Fair"
    else:
        verdict = "Low — review needed"
    out.append(["Detection rate", f"{rate:.1f}%", verdict])

    # iris_conf / open_eye_score 从 gaze.csv 读（VOGeo-Gaze 输出）
    if gaze_csv and Path(gaze_csv).exists():
        try:
            import csv as _csv
            import numpy as _np
            with open(gaze_csv) as f:
                g = list(_csv.DictReader(f))
            iris_confs = [float(r["iris_conf"]) for r in g if r.get("iris_conf") not in ("", None)]
            if iris_confs:
                med = float(_np.median(iris_confs))
                verdict = "Suspiciously high — iris diameter may be over-confident" if med > 0.98 else "OK"
                out.append(["Iris confidence (median)", f"{med:.3f}", verdict])
            open_eyes = [float(r["open_eye_score"]) for r in g if r.get("open_eye_score") not in ("", None)]
            if open_eyes:
                med = float(_np.median(open_eyes))
                verdict = "Low — blink_index unreliable, review PLR segments" if med < 0.70 else "OK"
                out.append(["Open-eye score (median)", f"{med:.3f}", verdict])
        except Exception:
            pass

    # gaze_ok 均值
    if gaze_csv and Path(gaze_csv).exists():
        try:
            import csv as _csv
            with open(gaze_csv) as f:
                g = list(_csv.DictReader(f))
            ok = [float(r["gaze_ok"]) for r in g if r.get("gaze_ok") not in ("", None)]
            if ok:
                import numpy as np
                m = float(np.mean(ok))
                verdict = "Some frames fell back to clip gaze" if m < 0.9 else "OK"
                out.append(["gaze_ok (mean)", f"{m:.3f}", verdict])
        except Exception:
            pass

    return out
