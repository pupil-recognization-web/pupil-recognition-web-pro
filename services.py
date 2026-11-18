"""纯服务层：分析相关的计算，不依赖 Gradio 组件。

目的：把 app.py 里"算东西"的部分挪出来，UI 只负责取值和展示。
好处：能单独跑、能单测、app.py 变薄。
"""
from __future__ import annotations

from pathlib import Path

import analysis_backend as ab
import applog
import store
from config import PLOT_DIR

_CACHE: dict = {}
_CACHE_LIMIT = 24


def _csv_mtime(csv_name: str) -> float:
    try:
        return store.csv_path(csv_name).stat().st_mtime
    except Exception:  # noqa: BLE001
        return 0.0


def _gaze_or_none(csv_name: str):
    g = store.gaze_csv(Path(csv_name).stem)
    return g if g.exists() else None


def abnormal_summary(abnormal) -> str:
    """异常帧的简要统计（界面/PDF 共用）。"""
    text = f"异常帧 {len(abnormal)} 个"
    if abnormal:
        reasons: dict = {}
        for a in abnormal:
            r = a[4] if len(a) > 4 else "未知"
            reasons[r] = reasons.get(r, 0) + 1
        text += "：" + "，".join(f"{k} {v}" for k, v in reasons.items())
    return text


def analyze_all(csv_name: str, stim_time=None):
    """跑完一次单眼分析所需的全部计算，返回 16 元组（顺序与界面输出一致）。

    结果按 (CSV, 刺激时刻, mtime) 缓存：重复点"分析"直接命中，约 1.2 s → 0.1 s。
    """
    key = (csv_name, stim_time, _csv_mtime(csv_name))
    if key in _CACHE:
        return _CACHE[key]

    morph_m, motion_m, morph_s, morph_p = ab.analyze_morphology(
        csv_name, plot_dir=str(PLOT_DIR))
    summary, frame, abnormal, text, _images = ab.analyze_csv(csv_name)
    plr_p, plr_text, plr_plot = ab.analyze_plr(
        csv_name, stimulus_time=stim_time, plot_dir=str(PLOT_DIR))

    gaze = _gaze_or_none(csv_name)
    interp_rows, interp_summary = ab.clinical_interpretation(csv_name, gaze_csv=gaze)
    ppt_ref = ab.ppt_reference_single(interp_rows)
    quality = ab.data_quality_summary(csv_name, gaze_csv=gaze)

    warns = ab.reliability_warnings(csv_name)
    if warns:
        text = "\n".join(warns) + "\n\n" + text

    result = (summary, frame, abnormal, text,
              morph_m, motion_m, morph_s, morph_p,
              plr_p, plr_text, plr_plot, abnormal_summary(abnormal),
              interp_rows, interp_summary, ppt_ref, quality)

    if len(_CACHE) >= _CACHE_LIMIT:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = result
    return result
