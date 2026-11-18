"""PLR（瞳孔对光反射）参数提取——全项目唯一实现。

背景
----
旧版有两份几乎逐行重复的实现：``analysis_backend.analyze_plr``（网页/PDF 用）
与 ``plr_params.py``（命令行用）。两边公式、阈值、平滑窗口各自维护，改一处
另一处就悄悄漂移。本模块把计算抽出来，两个入口都只负责“取数据 / 排版”。

数值口径（与原实现保持一致）
----------------------------
* 直径优先 ``pupil_mm``，否则 ``major``（px）。
* 只用 ``detected == 1`` 且直径非空的帧。
* 序列做 window=5 的居中滑动中值平滑。
* 基线 ``D0`` = 刺激时刻之前的平滑值中位数（不足 3 点则取前 1/10）。
* ``MCV`` = 收缩段内最大的下降斜率绝对值；归一化 = ``MCV / D0``。
* 恢复时间 = 从最小点回到 ``D0 - 0.5 * (D0 - Dmin)`` 所需时间。
* ``NPi 近似 = 1.5*s_lat + 2.0*s_mcv + 1.5*s_amp``（0–5，仅供趋势判断，
  不等价于 NPi-200 专有算法）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import PLR_MIN_VALID_POINTS, PLR_SMOOTH_WINDOW

#: 参数顺序、显示名与参考范围（网页表格与 PDF 共用）
REF_RANGE = {
    "基线直径": "2–5 mm",
    "最小直径": "—",
    "收缩幅度": ">15% baseline",
    "收缩百分比": ">15%",
    "潜伏期": "<0.2 s",
    "收缩时间": "<1.0 s",
    "平均收缩速度": "—",
    "最大收缩速度 MCV": ">1.0 mm/s",
    "归一化 MCV": "—",
    "恢复时间": "—",
    "平均扩张速度": "—",
    "最大扩张速度": "—",
    "NPi 近似评分": "≥3（正常）",
    "双侧差异": "<1 mm",
}


@dataclass
class PlrCurve:
    """一条完整的 PLR 曲线及其关键索引。"""

    csv_path: Path
    column: str
    unit: str
    t: np.ndarray
    d: np.ndarray
    stimulus_time: float
    stimulus_source: str
    stim_idx: int
    min_idx: int
    recovery_idx: Optional[int]

    @property
    def t_min(self) -> float:
        return float(self.t[self.min_idx])

    @property
    def t_recovery(self) -> Optional[float]:
        return float(self.t[self.recovery_idx]) if self.recovery_idx is not None else None


@dataclass
class PlrParams:
    """14 项 PLR 参数（数值部分，排版交给调用方）。"""

    baseline: float
    minimum: float
    amplitude: float
    percent: float
    latency: float
    contraction_time: float
    contraction_speed_avg: float
    mcv: float
    mcv_norm: float
    recovery_time: float
    dilation_speed_avg: float
    max_dilation_speed: float
    npi_like: float


def pick_diameter_col(df: pd.DataFrame):
    """优先 ``pupil_mm``（PLR 模式），否则 ``major``（像素）。"""
    if "pupil_mm" in df.columns and df["pupil_mm"].notna().any():
        return "pupil_mm", "mm"
    if "major" in df.columns:
        return "major", "px"
    raise ValueError("CSV 里没有 pupil_mm 或 major 列")


def smooth(series: pd.Series, window: int = PLR_SMOOTH_WINDOW) -> np.ndarray:
    """居中滑动中值平滑，抑制逐帧抖动。"""
    return series.rolling(window=window, center=True, min_periods=1).median().values


def detect_stimulus(t, d, min_drop_ratio: float = 0.01, window: int = PLR_SMOOTH_WINDOW) -> float:
    """自动检测收缩起点：连续下降且累计降幅超过基线一定比例。

    可信度低（录不到刺激光源时只能靠瞳孔自己下降），仅作兜底。
    """
    if len(d) < window + 2:
        return float(t[0])
    baseline = np.nanmedian(d[: max(3, len(d) // 10)])
    diff = np.diff(d)
    for i in range(len(diff) - window):
        seg = diff[i:i + window]
        if np.all(seg < 0) and (baseline - d[i + window]) / max(baseline, 1e-6) > min_drop_ratio:
            return float(t[i])
    return float(t[0] + (t[-1] - t[0]) * 0.2)


def npi_like(latency, mcv_norm, amplitude_ratio) -> float:
    """NPi 近似评分（0–5，≥3 正常）。三项加权，仅用于趋势判断。"""
    s_lat = max(0.0, min(1.0, (0.5 - latency) / 0.5)) if latency is not None else 0.0
    s_mcv = max(0.0, min(1.0, mcv_norm / 3.0)) if mcv_norm else 0.0
    s_amp = max(0.0, min(1.0, amplitude_ratio / 0.4)) if amplitude_ratio else 0.0
    return round(1.5 * s_lat + 2.0 * s_mcv + 1.5 * s_amp, 2)


def extract(csv_path, stimulus_time: Optional[float] = None,
            smooth_window: int = PLR_SMOOTH_WINDOW,
            stimulus_source: Optional[str] = None):
    """读取 CSV 并提取 PLR 曲线与参数。

    ``stimulus_source`` 仅用于在报告里标注刺激时刻的来源（手动 / 标定文件），
    不传时若给了 ``stimulus_time`` 就记为“手动输入”。

    返回 ``(curve, params)``；有效数据不足时返回 ``None``。
    """
    path = Path(csv_path)
    df = pd.read_csv(path)

    column, unit = pick_diameter_col(df)
    if "detected" in df.columns:
        valid = df[df["detected"] == 1].copy()
    else:
        valid = df.copy()
    valid = valid[valid[column].notna()]
    if len(valid) < PLR_MIN_VALID_POINTS:
        return None

    t = valid["time_sec"].values.astype(float)
    d = smooth(pd.Series(valid[column].values.astype(float)), window=smooth_window)

    # ---- 刺激时刻 ----
    if stimulus_time is not None and stimulus_time > 0:
        stim_t = float(stimulus_time)
        stim_source = stimulus_source or "手动输入"
    else:
        stim_t, stim_source = detect_stimulus(t, d), "自动检测（瞳孔下降，可信度低）"

    # ---- 基线 ----
    base_mask = t < stim_t
    if base_mask.sum() >= 3:
        D0 = float(np.nanmedian(d[base_mask]))
    else:
        D0 = float(np.nanmedian(d[: max(3, len(d) // 10)]))

    # ---- 最小点 ----
    min_idx = int(np.nanargmin(d))
    Dmin = float(d[min_idx])
    t_min = float(t[min_idx])
    stim_idx = min(int(np.argmax(t >= stim_t)), min_idx)

    # ---- 收缩段 ----
    contraction_time = max(t_min - t[stim_idx], 0.0)
    amplitude = D0 - Dmin
    percent = (amplitude / D0 * 100.0) if D0 > 0 else 0.0
    speed_avg = (amplitude / contraction_time) if contraction_time > 0 else 0.0

    # 潜伏期：刺激后第一次显著下降（阈值 0.5%/帧）
    latency = None
    if min_idx > stim_idx:
        for i in range(stim_idx, min_idx):
            if d[i + 1] < d[i] * 0.995:
                latency = t[i] - stim_t
                break
    if latency is None:
        latency = contraction_time

    # MCV：收缩段最大降斜率
    mcv = mcv_norm = 0.0
    if min_idx > stim_idx + 1:
        seg_t = t[stim_idx:min_idx + 1]
        seg_d = d[stim_idx:min_idx + 1]
        slopes = np.diff(seg_d) / np.maximum(np.diff(seg_t), 1e-6)
        if len(slopes) > 0:
            mcv = float(-np.nanmin(slopes))
            mcv_norm = mcv / D0 if D0 > 0 else 0.0

    # 恢复段：回到基线与最小点的中点
    recovery_threshold = D0 - 0.5 * (D0 - Dmin)
    recovery_idx = None
    for i in range(min_idx, len(d)):
        if d[i] >= recovery_threshold:
            recovery_idx = i
            break

    recovery_time = dilation_avg = dilation_max = 0.0
    if recovery_idx is not None and recovery_idx > min_idx:
        recovery_time = float(t[recovery_idx] - t[min_idx])
        seg_t = t[min_idx:recovery_idx + 1]
        seg_d = d[min_idx:recovery_idx + 1]
        slopes = np.diff(seg_d) / np.maximum(np.diff(seg_t), 1e-6)
        if len(slopes) > 0:
            dilation_avg = float(np.nanmean(slopes))
            dilation_max = float(np.nanmax(slopes))

    params = PlrParams(
        baseline=D0,
        minimum=Dmin,
        amplitude=amplitude,
        percent=percent,
        latency=float(latency),
        contraction_time=contraction_time,
        contraction_speed_avg=speed_avg,
        mcv=mcv,
        mcv_norm=mcv_norm,
        recovery_time=recovery_time,
        dilation_speed_avg=dilation_avg,
        max_dilation_speed=dilation_max,
        npi_like=npi_like(latency, mcv_norm, amplitude / D0 if D0 > 0 else None),
    )
    curve = PlrCurve(
        csv_path=path, column=column, unit=unit, t=t, d=d,
        stimulus_time=stim_t, stimulus_source=stim_source,
        stim_idx=stim_idx, min_idx=min_idx, recovery_idx=recovery_idx,
    )
    return curve, params


def params_table(params: PlrParams, unit: str, quality=None, response=None):
    """``[参数名, 值, 参考范围]`` 行，供网页表格 / PDF 复用。

    传 ``quality`` / ``response``（v2 才有）时，额外补充响应判定、帧率、
    潜伏期分辨率、数据质量四行，让"这个数字能不能信"直接写在表里。
    """
    u = unit
    rows = [
        [f"基线直径 ({u})", round(params.baseline, 3), REF_RANGE["基线直径"]],
        [f"最小直径 ({u})", round(params.minimum, 3), REF_RANGE["最小直径"]],
        [f"收缩幅度 ({u})", round(params.amplitude, 3), REF_RANGE["收缩幅度"]],
        ["收缩百分比 (%)", round(params.percent, 1), REF_RANGE["收缩百分比"]],
        ["潜伏期 (s)", round(params.latency, 3), REF_RANGE["潜伏期"]],
        ["收缩时间 (s)", round(params.contraction_time, 3), REF_RANGE["收缩时间"]],
        [f"平均收缩速度 ({u}/s)", round(params.contraction_speed_avg, 3), REF_RANGE["平均收缩速度"]],
        [f"最大收缩速度 MCV ({u}/s)", round(params.mcv, 3), REF_RANGE["最大收缩速度 MCV"]],
        ["归一化 MCV (基线%/s)", round(params.mcv_norm * 100.0, 2), REF_RANGE["归一化 MCV"]],
        ["恢复时间 (s)", round(params.recovery_time, 3), REF_RANGE["恢复时间"]],
        [f"平均扩张速度 ({u}/s)", round(params.dilation_speed_avg, 3), REF_RANGE["平均扩张速度"]],
        [f"最大扩张速度 ({u}/s)", round(params.max_dilation_speed, 3), REF_RANGE["最大扩张速度"]],
        ["NPi 近似评分 (0-5)", params.npi_like, REF_RANGE["NPi 近似评分"]],
        ["双侧差异", "单路数据，无法计算", REF_RANGE["双侧差异"]],
    ]
    if response is not None:
        rows.insert(4, [
            "响应判定",
            "✅ 有收缩" if response else "⚠️ 未引出（Δ<5%）",
            "刺激后窗口内收缩 ≥5% 基线",
        ])
    if quality is not None:
        rows += [
            ["帧率 (fps)", round(quality.fps, 1), "≥15 fps 时潜伏期才有意义"],
            ["潜伏期分辨率 (s)", round(quality.latency_resolution, 3), "1 帧量化，非仪器精度"],
            ["数据质量",
             f"检出 {quality.frames_detected}/{quality.frames_total}，"
             f"使用 {quality.frames_used}，剔除闭眼 {quality.frames_dropped_closed}",
             "｜".join(quality.notes) if quality.notes else "—"],
        ]
    return rows


def params_dict(params: PlrParams, unit: str, quality=None, response=None) -> dict:
    """``{参数名: 值}``，供命令行 / 序列化使用。"""
    return {row[0]: row[1] for row in params_table(params, unit, quality, response)}


# ============================================================
# v2：刺激窗 + 基线窗 + 无响应判定（与临床共识对齐）
# ============================================================
def _fps(t: np.ndarray) -> float:
    if len(t) < 2:
        return 0.0
    dt = float(np.median(np.diff(t)))
    return 1.0 / dt if dt > 0 else 0.0


def _eye_closed_frames(csv_path: Path):
    """读同名 VOGeo gaze.csv，返回 ``open_eye_score < 0.5`` 的帧号集合。

    没有 gaze.csv 时返回空集合（退化为不做眨眼剔除）。
    """
    import csv as _csv

    from config import VOGE0_OUTPUT_DIR

    gaze = Path(VOGE0_OUTPUT_DIR) / Path(csv_path).stem / "gaze.csv"
    if not gaze.exists():
        return set()
    closed = set()
    try:
        with open(gaze) as f:
            for row in _csv.DictReader(f):
                try:
                    if float(row.get("open_eye_score") or 1.0) < 0.5:
                        closed.add(int(row["frame"]))
                except Exception:  # noqa: BLE001 - 单行坏数据忽略
                    continue
    except Exception:  # noqa: BLE001
        return set()
    return closed


@dataclass
class PlrQuality:
    """数据质量：决定这些 PLR 数字能不能当真。"""

    mode: str
    fps: float
    frames_total: int
    frames_detected: int
    frames_used: int
    frames_dropped_closed: int
    latency_resolution: float
    notes: list


@dataclass
class PlrResult:
    curve: PlrCurve
    params: PlrParams
    quality: PlrQuality
    response: bool


def analyze(csv_path, stimulus_time=None, mode=None, smooth_window=None,
            stimulus_source=None):
    """统一入口：按 ``mode`` 走 v2（默认）或 legacy。

    返回 :class:`PlrResult`；数据不足或刺激后窗口无有效帧时返回 ``None``。
    """
    from config import (
        PLR_BASE_WINDOW, PLR_LATENCY_DROP, PLR_MIN_RESPONSE_PCT, PLR_MODE,
        PLR_RESPONSE_WINDOW, PLR_SMOOTH_SEC,
    )

    mode = (mode or PLR_MODE).lower()
    path = Path(csv_path)
    if not path.exists():
        return None

    df = pd.read_csv(path)
    try:
        column, unit = pick_diameter_col(df)
    except ValueError:
        return None

    frames_total = len(df)
    valid = df[df["detected"] == 1] if "detected" in df.columns else df.copy()
    valid = valid[valid[column].notna()]
    frames_detected = len(valid)
    if frames_detected < PLR_MIN_VALID_POINTS:
        return None

    # legacy：完全保留旧实现，仅用于复现历史结果
    if mode == "legacy":
        out = extract(path, stimulus_time=stimulus_time,
                      smooth_window=smooth_window or PLR_SMOOTH_WINDOW)
        if out is None:
            return None
        curve, params = out
        fps = _fps(curve.t)
        q = PlrQuality("legacy", fps, frames_total, frames_detected, len(curve.t), 0,
                       1.0 / fps if fps else 0.0,
                       ["legacy：全序列 argmin，只用于复现旧结果，不建议用于临床判读"])
        return PlrResult(curve, params, q, params.percent >= PLR_MIN_RESPONSE_PCT)

    # ---- v2 ----
    closed = _eye_closed_frames(path)
    dropped = 0
    if closed and "frame" in valid.columns:
        keep = ~valid["frame"].isin(closed)
        dropped = int((~keep).sum())
        if int(keep.sum()) >= PLR_MIN_VALID_POINTS:
            valid = valid[keep]

    t = valid["time_sec"].values.astype(float)
    fps = _fps(t)
    win = smooth_window or max(1, int(round(PLR_SMOOTH_SEC * fps)))
    if win % 2 == 0:
        win += 1
    d = smooth(pd.Series(valid[column].values.astype(float)), window=win)

    if stimulus_time is not None and stimulus_time > 0:
        stim_t = float(stimulus_time)
        # 吸附到最近的采样帧：亚帧差异不应改变基线窗包含哪些帧
        stim_t = float(t[int(np.argmin(np.abs(t - stim_t)))])
        src = stimulus_source or "手动输入"
    else:
        # 自动检测固定用 legacy 的平滑窗口，保证 v2/legacy 得到同一个刺激时刻
        d_detect = smooth(pd.Series(valid[column].values.astype(float)),
                          window=PLR_SMOOTH_WINDOW)
        stim_t = detect_stimulus(t, d_detect)
        src = "自动检测（瞳孔下降，可信度低）"

    # 基线：刺激前固定窗口
    b0, b1 = PLR_BASE_WINDOW
    base = d[(t >= stim_t + b0) & (t < stim_t + b1)]
    if base.size < 3:
        pre = d[t < stim_t]
        base = pre[-5:] if pre.size >= 3 else d[: max(3, len(d) // 10)]
    D0 = float(np.median(base))

    # 收缩：只在刺激后窗口里找最小点
    r0, r1 = PLR_RESPONSE_WINDOW
    resp = (t >= stim_t + r0) & (t <= stim_t + r1)
    if resp.sum() < 3:
        return None
    tt, dd = t[resp], d[resp]
    min_i = int(np.argmin(dd))
    Dmin, t_min = float(dd[min_i]), float(tt[min_i])
    raw_delta = D0 - Dmin                      # 正=收缩，负=刺激后反而扩大
    amplitude = max(0.0, raw_delta)
    percent = (amplitude / D0 * 100.0) if D0 > 0 else 0.0
    response = percent >= PLR_MIN_RESPONSE_PCT

    # MCV：收缩段最大下降斜率
    mcv = mcv_norm = 0.0
    seg = (t >= stim_t + r0) & (t <= t_min)
    if seg.sum() >= 2:
        st, sd = t[seg], d[seg]
        if len(sd) >= 3:
            # 3 点最小二乘斜率：比相邻两点差分稳，抑制单帧抖动
            slopes = [np.polyfit(st[i:i + 3], sd[i:i + 3], 1)[0] for i in range(len(sd) - 2)]
        else:
            slopes = list(np.diff(sd) / np.maximum(np.diff(st), 1e-6))
        if len(slopes):
            mcv = float(max(0.0, -min(slopes)))
            mcv_norm = mcv / D0 if D0 > 0 else 0.0

    # 潜伏期：较基线下降 PLR_LATENCY_DROP 即算开始收缩；无则 NaN
    latency = float("nan")
    thr = D0 * (1.0 - PLR_LATENCY_DROP)
    for i in range(len(tt)):
        if tt[i] > t_min:
            break
        if dd[i] <= thr and i + 2 < len(dd) and dd[i + 1] <= thr and dd[i + 2] <= thr:
            latency = float(tt[i] - stim_t)
            break

    contraction_time = max(t_min - stim_t, 0.0)
    speed_avg = (amplitude / contraction_time) if contraction_time > 0 else 0.0

    # 恢复：回到 (Dmin + 90% 幅度)
    rec_thr = Dmin + 0.9 * amplitude
    recovery_idx = None
    for i in range(len(t)):
        if t[i] > t_min and d[i] >= rec_thr:
            recovery_idx = i
            break
    recovery_time = dilation_avg = dilation_max = 0.0
    if recovery_idx is not None and recovery_idx > int(np.argmin(np.abs(t - t_min))):
        i_min = int(np.argmin(np.abs(t - t_min)))
        recovery_time = float(t[recovery_idx] - t[i_min])
        slopes = np.diff(d[i_min:recovery_idx + 1]) / np.maximum(np.diff(t[i_min:recovery_idx + 1]), 1e-6)
        if slopes.size:
            dilation_avg = float(np.nanmean(slopes))
            dilation_max = float(np.nanmax(slopes))

    # NPi 近似：潜伏期 + 收缩速度 + 收缩幅度 三项加权（0–5）
    # 有 mm 标定时用绝对速度（mm/s）归一，否则退回"基线相对速度"
    from config import PLR_MCV_REF_MM_S

    s_lat = max(0.0, min(1.0, (0.5 - latency) / 0.5)) if np.isfinite(latency) else 0.0
    if unit == "mm":
        s_mcv = max(0.0, min(1.0, mcv / PLR_MCV_REF_MM_S))
    else:
        s_mcv = max(0.0, min(1.0, mcv_norm / 3.0))
    s_amp = max(0.0, min(1.0, (percent / 100.0) / 0.4))
    npi = round(1.5 * s_lat + 2.0 * s_mcv + 1.5 * s_amp, 2)

    params = PlrParams(
        baseline=D0, minimum=Dmin, amplitude=amplitude, percent=percent,
        latency=latency, contraction_time=contraction_time,
        contraction_speed_avg=speed_avg, mcv=mcv, mcv_norm=mcv_norm,
        recovery_time=recovery_time, dilation_speed_avg=dilation_avg,
        max_dilation_speed=dilation_max, npi_like=npi,
    )
    curve = PlrCurve(
        csv_path=path, column=column, unit=unit, t=t, d=d,
        stimulus_time=stim_t, stimulus_source=src,
        stim_idx=int(np.argmin(np.abs(t - stim_t))),
        min_idx=int(np.argmin(np.abs(t - t_min))),
        recovery_idx=recovery_idx,
    )

    notes = []
    if src.startswith("自动"):
        notes.append("刺激时刻为自动推断，潜伏期/收缩时间不可靠")
    if fps and fps < 15:
        notes.append(f"帧率 {fps:.1f} fps，潜伏期分辨率约 {1.0/fps:.2f}s（临床阈值 0.2s，仅能分档看）")
    if dropped:
        notes.append(f"按 open_eye_score<0.5 剔除闭眼帧 {dropped} 帧")
    if raw_delta < 0:
        notes.append(f"刺激后瞳孔反而扩大 {-raw_delta:.3f}{unit}（{-raw_delta / D0 * 100:.1f}%），未见收缩")
    if not response:
        notes.append(f"刺激后 {r1:.0f}s 内瞳孔无明显收缩（Δ={percent:.1f}%），"
                     "需核对刺激时刻/光源/录像亮度，勿直接判为反射消失")
    if unit != "mm":
        notes.append("该 CSV 无 pupil_mm（非 PLR 模式），NPi 用像素速度归一，仅供趋势参考")

    quality = PlrQuality("v2", fps, frames_total, frames_detected, len(t), dropped,
                         1.0 / fps if fps else 0.0, notes)
    return PlrResult(curve, params, quality, response)
