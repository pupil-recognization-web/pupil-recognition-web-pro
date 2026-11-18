"""单眼/双眼分析结果导出 PDF（HTML -> wkhtmltopdf）。"""
import csv as _csv
import applog
import html
import subprocess
from datetime import datetime
from pathlib import Path

import analysis_backend as ab
import store

from config import PDF_DIR, PLOT_DIR as _PLOT_DIR

OUT_DIR = Path(PDF_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR = Path(_PLOT_DIR)

CSS = """
<style>
  @page { margin: 0; }
  body { font-family: "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Droid Sans Fallback", sans-serif;
         font-size: 11px; color: #222; margin: 0; }
  h1 { font-size: 20px; margin: 0 0 4px 0; border-bottom: 2px solid #333; padding-bottom: 4px; }
  .subtitle { color: #555; font-size: 12px; margin-bottom: 4px; }
  .meta { color: #666; font-size: 10px; margin-bottom: 10px; }
  h2 { font-size: 14px; margin: 18px 0 6px 0; border-bottom: 1px solid #999; padding-bottom: 3px; }
  table { border-collapse: collapse; width: 100%; margin: 4px 0 10px 0; page-break-inside: auto; }
  tr { page-break-inside: avoid; page-break-after: auto; }
  th, td { border: 1px solid #bbb; padding: 3px 5px; vertical-align: top; text-align: left; }
  th { background: #e0e0e0; color: #222; font-weight: bold; border-bottom: 2px solid #333; }
  tr:nth-child(even) td { background: #f7f7f7; }
  thead { display: table-header-group; }
  tbody { display: table-row-group; }
  td.num { text-align: center; }
  td.txt { text-align: left; }
  img { max-width: 100%; margin: 6px 0; page-break-inside: avoid; }
  .page-break { page-break-after: always; }
  .note { font-size: 9px; color: #666; margin: 2px 0 8px 0; font-style: italic; }
  .caption { font-size: 9px; color: #555; margin: 2px 0 10px 0; text-align: left; }
  .exec-summary { background: #f5f5f5; border-left: 4px solid #333; padding: 6px 10px; margin: 8px 0 12px 0; font-size: 11px; }
  .impression-zh { background: #f5f5f5; border-left: 4px solid #333; padding: 6px 10px; margin: 8px 0 6px 0; }
  .impression-en { background: #fafafa; border-left: 4px solid #999; padding: 6px 10px; margin: 0 0 12px 0; color: #444; }
  .impression-zh p, .impression-en p { margin: 2px 0; }
  .disclaimer { margin-top: 18px; padding: 6px 10px; border: 1px solid #ccc; background: #fafafa; font-size: 9px; color: #555; }
  .footer-note { margin-top: 8px; font-size: 8px; color: #999; text-align: right; }
  .cover { text-align: center; padding-top: 80mm; }
  .cover h1 { font-size: 30px; border: none; margin-bottom: 10px; }
  .cover .cover-sub { font-size: 16px; color: #555; margin-bottom: 30px; }
  .cover .cover-meta { font-size: 11px; color: #666; line-height: 1.8; }
  .cover hr { border: none; border-top: 1px solid #333; margin: 8px auto; width: 60%; }
  .cover .cover-foot { margin-top: 60mm; font-size: 9px; color: #999; border-top: 1px solid #ccc; padding-top: 6px; }
</style>
"""

_EN_NAMES = {
    "针尖样瞳孔": "pinpoint pupil", "瞳孔散大": "mydriasis", "直径正常": "normal diameter",
    "对光反射消失": "absent light reflex", "对光反射减弱": "reduced light reflex", "对光反射存在": "present light reflex",
    "MCV 极低": "very low MCV", "MCV 偏低": "low MCV", "MCV 正常": "normal MCV",
    "NPi 异常": "abnormal NPi", "NPi 正常": "normal NPi",
    "PLR 变差（尺寸正常）": "degraded PLR (normal size)",
    "小瞳提示（Horner 鉴别）": "small pupil (Horner differential)",
    "散大固定提示（脑疝/动眼神经）": "fixed dilated (herniation/CN III)",
    "阿-罗瞳孔鉴别": "Argyll-Robertson differential", "强直性瞳孔鉴别": "tonic pupil differential",
    "Horner 综合征（瞳孔部分）提示": "Horner syndrome (pupil) hint",
    "形态欠规则": "irregular pupil shape", "形态规则": "regular pupil shape",
    "gaze 稳定性（PLR 质控）": "gaze stability (PLR QC)",
    "单侧散大（左）": "unilateral mydriasis (left)",
    "单侧散大（右）": "unilateral mydriasis (right)",
    "单侧缩小（左）": "unilateral miosis (left)",
    "单侧缩小（右）": "unilateral miosis (right)",
    "双侧散大": "bilateral mydriasis",
    "散大程度": "mydriasis severity",
    "圆度描述": "roundness description",
    "基线直径": "baseline diameter", "收缩幅度": "constriction amplitude",
    "收缩百分比": "constriction percent", "最大收缩速度": "max constriction velocity",
    "NPi 近似评分": "NPi-like score", "潜伏期": "latency", "收缩时间": "constriction time",
    "恢复时间": "recovery time", "双眼 gaze 水平差": "gaze horizontal difference",
    "双眼 gaze 垂直差": "gaze vertical difference",
}


_COND_EN = {
    "正常瞳孔（2–5mm，反射灵敏）": "normal pupil (2-5mm, reactive)",
    "双侧针尖样瞳孔（<2mm）": "bilateral pinpoint (<2mm)",
    "双侧瞳孔散大固定（6–9mm，反射消失）": "bilateral fixed dilated (6-9mm, absent reflex)",
    "双侧瞳孔散大对称（反射尚存）": "bilateral dilated symmetric (reflex present)",
    "对光反射消失": "absent light reflex",
    "PLR 变差（尺寸正常但 NPi 下降）": "degraded PLR (normal size, NPi drop)",
    "强直性瞳孔（Adie，收缩极慢）": "tonic pupil (Adie, very slow)",
    "阿-罗瞳孔（对光反射消失 + 调节反射存在）": "Argyll-Robertson (absent light, present accommodation)",
    "形态欠规则（阿-罗瞳孔形态）": "irregular shape (AR pupil)",
    "动眼神经麻痹（瞳孔受累型）提示": "CN III palsy (pupil-involved) hint",
    "动眼神经麻痹（瞳孔回避型）提示": "CN III palsy (pupil-sparing) hint",
    "Horner 综合征（瞳孔部分）提示": "Horner syndrome (pupil) hint",
    "PLR 质控（视线稳定）": "PLR QC (gaze stable)",
    "病理性不等大（>1mm）": "pathological anisocoria (>1mm)",
    "生理性不等大（<1mm）": "physiological anisocoria (<1mm)",
    "单侧散大（脑疝预警）": "unilateral mydriasis (herniation warning)",
    "单侧缩小（Horner 瞳孔部分）": "unilateral miosis (Horner pupil)",
    "双侧散大（双侧 >5mm）": "bilateral mydriasis (>5mm)",
    "双侧 MCV 差（单侧神经受损）": "bilateral MCV difference (unilateral nerve)",
    "双侧 NPi 差（单侧反应异常）": "bilateral NPi difference (unilateral response)",
    "双侧收缩幅度差（单侧反应弱）": "bilateral amplitude difference (weak response)",
    "双侧潜伏期差（单侧传导慢）": "bilateral latency difference (slow conduction)",
    "双侧恢复时间差（单侧恢复慢）": "bilateral recovery difference (slow recovery)",
    "眼外肌运动受限（示意性）": "extraocular movement restriction (indicative)",
    "复视（示意性）": "diplopia (indicative)",
    "滑车神经麻痹（垂直位差异，示意性）": "trochlear palsy (vertical difference, indicative)",
}


def _esc(x):
    return html.escape(str(x)) if x is not None else ""


def _is_num(s):
    try:
        float(str(s).replace("%", "").replace("°", "").replace(",", "").strip())
        return True
    except Exception:
        return False


def _table_html(rows, header=True):
    if not rows:
        return "<p>N/A</p>"
    out = ["<table>"]
    if header:
        out.append("<thead><tr>" + "".join(f"<th>{_esc(c)}</th>" for c in rows[0]) + "</tr></thead>")
        data_rows = rows[1:]
    else:
        data_rows = rows
    out.append("<tbody>")
    for row in data_rows:
        cells = []
        for c in row:
            cls = "num" if _is_num(c) else "txt"
            cells.append(f'<td class="{cls}">{_esc(c)}</td>')
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def _img_html(path, width="100%", caption=None):
    p = Path(path).resolve()
    if not p.exists():
        return f"<p>[image not found: {_esc(p.name)}]</p>"
    h = f'<img src="file://{p}" style="width:{width};">'
    if caption:
        h += f'<p class="caption">{_esc(caption)}</p>'
    return h


def _parse_dataset_model(stem):
    dataset, model = "Unknown", "Unknown"
    low = stem.lower()
    if "openeds" in low or stem.startswith("S_"):
        dataset = "OpenEDS"
    elif "lpw" in low:
        dataset = "LPW"
    for m in ("baseline", "mac", "strip"):
        if m in low:
            model = m
            break
    return dataset, model


def _cover_html(stem, dataset, model, ts):
    report_id = f"{stem}_{ts.replace('-', '').replace(':', '').replace(' ', '_')}"
    return f'''<div class="cover">
  <h1>Pupil Analysis Report</h1>
  <div class="cover-sub">{_esc(stem)}</div>
  <hr>
  <div class="cover-meta">
    <b>Report ID:</b> {_esc(report_id)}<br>
    <b>Dataset:</b> {_esc(dataset)} &nbsp;|&nbsp; <b>Model:</b> {_esc(model)}<br>
    <b>Generated:</b> {ts}
  </div>
  <hr>
  <div class="cover-foot">
    Automated analysis report. For research and decision-support use only.<br>
    Not a clinical diagnosis.
  </div>
</div>
<div class="page-break"></div>'''


def _section_video_overview(csv_name):
    summary, _, _, _, _ = ab.analyze_csv(csv_name)
    rows = [r for r in summary if len(r) >= 2 and not str(r[0]).startswith("---")]
    note = "<p class=\"note\">本表汇总视频参数和模型检出质量，用于判断后续 PLR/gaze 参数的可靠性。检出率 <80% 或低置信度帧较多时，PLR 参数需谨慎解读。</p>"
    return "<h2>Video &amp; Model Overview</h2>" + note + _table_html([["Metric", "Value"]] + rows)


def _section_3d(csv_name):
    import vogeo_backend as vb
    stem = Path(csv_name).stem
    d = vb.OUTPUT_DIR / stem
    gaze_csv = d / "gaze.csv"
    overlay = d / "overlay.mp4"
    parts = ["<h2>3D Gaze Model</h2>", "<p class=\"note\">本表汇总 gaze 角度（theta_h/theta_v）的均值和波动。std 越大表示视线越不稳定；Gaze stability: Drifting 表示该段 gaze 漂移，PLR 参数可信度降低。</p>"]
    if not gaze_csv.exists():
        parts.append("<p>No VOGeo-Gaze result for this CSV.</p>")
        return "".join(parts)
    try:
        with open(gaze_csv) as f:
            rows = list(_csv.DictReader(f))
        th = [float(r["theta_h"]) for r in rows if r.get("theta_h") not in ("", None)]
        tv = [float(r["theta_v"]) for r in rows if r.get("theta_v") not in ("", None)]
        import numpy as np
        th_std = float(np.std(th)) if th else 0.0
        tv_std = float(np.std(tv)) if tv else 0.0
        stable = "Stable" if (th_std < 2.0 and tv_std < 2.0) else "Drifting"
        rows_t = [
            ["Metric", "Value"],
            ["theta_h mean (deg)", f"{sum(th)/len(th):.2f}" if th else "N/A"],
            ["theta_h range (deg)", f"[{min(th):.2f}, {max(th):.2f}]" if th else "N/A"],
            ["theta_h std (deg)", f"{th_std:.2f}"],
            ["theta_v mean (deg)", f"{sum(tv)/len(tv):.2f}" if tv else "N/A"],
            ["theta_v range (deg)", f"[{min(tv):.2f}, {max(tv):.2f}]" if tv else "N/A"],
            ["theta_v std (deg)", f"{tv_std:.2f}"],
            ["Gaze stability", stable],
            ["Frames", len(rows)],
        ]
        parts.append(_table_html(rows_t))
    except Exception as e:
        parts.append(f"<p>gaze.csv parse error: {_esc(e)}</p>")
    if overlay.exists():
        try:
            import cv2
            cap = cv2.VideoCapture(str(overlay))
            ok, frame = cap.read()
            cap.release()
            if ok:
                tmp = OUT_DIR / f"{stem}_overlay_first.png"
                cv2.imwrite(str(tmp), frame)
                parts.append(_img_html(tmp, width="70%", caption="VOGeo-Gaze overlay (first frame)"))
        except Exception as e:
            parts.append(f"<p>overlay frame error: {_esc(e)}</p>")
    gaze_names = ["Gaze angles over time", "Gaze 2D trajectory", "Gaze trajectory colored by pupil"]
    for i, p in enumerate(sorted(d.glob("*_gaze_*.png"))[:3]):
        parts.append(_img_html(p, width="80%", caption=gaze_names[i] if i < len(gaze_names) else None))
    return "".join(parts)


def _section_morph(csv_name):
    morph_m, motion_m, _, morph_p = ab.analyze_morphology(csv_name, plot_dir=str(PLOT_DIR))
    parts = ["<h2>Morphology &amp; Motion</h2>", "<p class=\"note\">本表展示瞳孔形态（长短轴、面积、长短轴比）和运动（速度、加速度、轨迹长度、稳定性）的逐帧统计。长短轴比 <0.8 提示形态欠规则；速度/加速度异常高可能对应眨眼或伪影。</p>"]
    parts.append(_table_html([["Morphology", "Value"]] + [[k, v] for k, v in morph_m]))
    parts.append(_table_html([["Motion", "Value"]] + [[k, v] for k, v in motion_m]))
    names = ["Pupil center trajectory", "Pupil diameter over time", "Pupil center speed"]
    for i, p in enumerate(morph_p):
        parts.append(_img_html(p, width="80%", caption=names[i] if i < len(names) else None))
    return "".join(parts)


def _section_plr(csv_name, stimulus_time=None):
    params, text, plot = ab.analyze_plr(csv_name, stimulus_time=stimulus_time, plot_dir=str(PLOT_DIR))
    parts = ["<h2>PLR Parameters</h2>", "<p class=\"note\">本表为 14 项 PLR 参数，Reference 列给出常用参考范围（非诊断标准）。重点看：基线直径（2–5mm）、收缩幅度（>15% 基线）、MCV（>1.0 mm/s）、NPi（≥3）。</p>"]
    if params:
        parts.append(_table_html([["Parameter", "Value", "Reference"]] + params))
    if plot:
        parts.append(_img_html(plot, width="85%", caption="PLR curve (stimulus, minimum, recovery)"))
    return "".join(parts)


def _section_clinical(csv_name, full=False):
    import vogeo_backend as vb
    stem = Path(csv_name).stem
    gaze_csv = store.gaze_csv(stem)
    interp, summary = ab.clinical_interpretation(csv_name, gaze_csv=gaze_csv if gaze_csv.exists() else None, full=full)
    ppt = ab.ppt_reference_single(interp)
    parts = ["<h2>Clinical Interpretation</h2>", "<p class=\"note\">本表为逐项判读，⚠️ 表示异常/需关注，✅ 表示正常/未触发。下方 Condition Reference 表为病症对照，列出 13 类瞳孔异常的判读条件和本视频结果。</p>"]
    if interp:
        parts.append(_table_html([["判读项", "结果", "依据"]] + interp))
    parts.append("<h2>Condition Reference</h2>")
    if ppt:
        parts.append(_table_html([["病症", "判读", "依据（本视频）", "判读条件"]] + ppt))
    parts.append(f"<pre style='white-space:pre-wrap;font-family:inherit;'>{_esc(summary)}</pre>")
    return "".join(parts)


def _section_abnormal(csv_name):
    _, _, abnormal, _, _ = ab.analyze_csv(csv_name)
    parts = ["<h2>Abnormal Frames</h2>", "<p class=\"note\">本表列出检测异常帧（未检出/长短轴偏离/虹膜缺失）。异常帧比例高时，PLR 曲线需结合原始视频复核。</p>"]
    if not abnormal:
        parts.append("<p>No abnormal frames.</p>")
    else:
        reasons = {}
        for a in abnormal:
            r = a[4] if len(a) > 4 else "unknown"
            reasons[r] = reasons.get(r, 0) + 1
        parts.append(f"<p><b>Total abnormal frames:</b> {len(abnormal)} &nbsp;|&nbsp; "
                     + " &nbsp;|&nbsp; ".join(f"{k}: {v}" for k, v in reasons.items()) + "</p>")
        rows = [["Frame", "Major", "Minor", "Conf", "Reason"]]
        for a in abnormal:
            rows.append([str(x) for x in a])
        parts.append(_table_html(rows))
    return "".join(parts)


def _plr_get(plr, prefix):
    for item in plr:
        if len(item) >= 2 and str(item[0]).startswith(prefix):
            try:
                return float(item[1])
            except Exception:
                return None
    return None


def _section_impression_single(csv_name):
    import vogeo_backend as vb
    stem = Path(csv_name).stem
    gaze_csv = store.gaze_csv(stem)
    plr, _, _ = ab.analyze_plr(csv_name)
    interp, _ = ab.clinical_interpretation(csv_name, gaze_csv=gaze_csv if gaze_csv.exists() else None)

    d0 = _plr_get(plr, "基线直径")
    pct = _plr_get(plr, "收缩百分比")
    amp = _plr_get(plr, "收缩幅度")
    mcv = _plr_get(plr, "最大收缩速度")
    npi = _plr_get(plr, "NPi 近似评分")

    zh, en = [], []
    if d0 is not None:
        if d0 < 2.0:
            zh.append(f"基线瞳孔直径 {d0:.2f} mm，低于正常 2 mm。")
            en.append(f"Baseline pupil diameter was {d0:.2f} mm, below the normal 2 mm.")
        elif d0 > 5.0:
            zh.append(f"基线瞳孔直径 {d0:.2f} mm，高于正常 5 mm。")
            en.append(f"Baseline pupil diameter was {d0:.2f} mm, above the normal 5 mm.")
        else:
            zh.append(f"基线瞳孔直径 {d0:.2f} mm，在正常 2–5 mm 范围内。")
            en.append(f"Baseline pupil diameter was {d0:.2f} mm, within the normal 2-5 mm range.")
    if pct is not None and amp is not None:
        if pct < 5:
            zh.append(f"光刺激后收缩幅度 {amp:.2f} mm（{pct:.1f}% 基线），对光反射消失。")
            en.append(f"Light stimulation elicited a constriction of {amp:.2f} mm ({pct:.1f}% of baseline), consistent with an absent pupillary light reflex.")
        elif pct < 15:
            zh.append(f"光刺激后收缩幅度 {amp:.2f} mm（{pct:.1f}% 基线），对光反射减弱。")
            en.append(f"Light stimulation elicited a constriction of {amp:.2f} mm ({pct:.1f}% of baseline), consistent with a reduced pupillary light reflex.")
        else:
            zh.append(f"光刺激后收缩幅度 {amp:.2f} mm（{pct:.1f}% 基线），对光反射存在。")
            en.append(f"Light stimulation elicited a constriction of {amp:.2f} mm ({pct:.1f}% of baseline), consistent with a present pupillary light reflex.")
    if mcv is not None and npi is not None:
        zh.append(f"最大收缩速度 {mcv:.2f} mm/s，NPi 近似评分 {npi:.2f}。")
        en.append(f"Maximum constriction velocity was {mcv:.2f} mm/s, and the NPi-like score was {npi:.2f}.")
    warns = [r for r in interp if len(r) >= 3 and str(r[1]).startswith("⚠️")]
    if warns:
        zh.append(f"检出 {len(warns)} 项异常提示：" + "；".join(f"{r[0]}（{r[2]}）" for r in warns) + "。")
        en_names = [_EN_NAMES.get(str(r[0]), str(r[0])) for r in warns]
        en.append(f"{len(warns)} notable finding(s): " + "; ".join(en_names) + ". See Chinese section for details.")
        # 相关病症
        try:
            ppt = ab.ppt_reference_single(interp)
            conds = [p0 for p0 in ppt if len(p0) >= 2 and str(p0[1]).startswith("⚠️")
                     and "PLR 质控" not in str(p0[0])]
            if conds:
                zh.append("相关病症：" + "；".join(c[0] for c in conds) + "。")
                en.append("related conditions: " + "; ".join(_COND_EN.get(str(c[0]), str(c[0])) for c in conds) + ".")
        except Exception:
            applog.log_here()
    else:
        zh.append("未检出明显异常。")
        en.append("No significant abnormalities were detected in this automated assessment.")
    zh.append("本报告为自动化分析，仅供研究与决策支持，不构成临床诊断，需结合临床。")
    en.append("This is an automated analysis for research and decision-support purposes only. It does not constitute a clinical diagnosis and requires clinical correlation.")

    zh_html = "".join(f"<p>{_esc(x)}</p>" for x in zh)
    en_html = "".join(f"<p>{_esc(x)}</p>" for x in en)
    return (f'<h2>结论 / Impression</h2>'
            f'<div class="impression-zh">{zh_html}</div>'
            f'<div class="impression-en">{en_html}</div>')


def _section_impression_dual(csv_left, csv_right):
    import vogeo_backend as vb
    l_stem, r_stem = Path(csv_left).stem, Path(csv_right).stem
    l_gaze = store.gaze_csv(l_stem)
    r_gaze = store.gaze_csv(r_stem)
    l_plr, _, _ = ab.analyze_plr(csv_left)
    r_plr, _, _ = ab.analyze_plr(csv_right)
    l_interp, _ = ab.clinical_interpretation(csv_left, gaze_csv=l_gaze if l_gaze.exists() else None)
    r_interp, _ = ab.clinical_interpretation(csv_right, gaze_csv=r_gaze if r_gaze.exists() else None)

    def _eye_summary(plr, interp):
        d0 = _plr_get(plr, "基线直径")
        pct = _plr_get(plr, "收缩百分比")
        amp = _plr_get(plr, "收缩幅度")
        mcv = _plr_get(plr, "最大收缩速度")
        npi = _plr_get(plr, "NPi 近似评分")
        zh, en = [], []
        if d0 is not None:
            if d0 < 2.0:
                zh.append(f"基线直径 {d0:.2f} mm（低于正常 2 mm）")
                en.append(f"baseline diameter {d0:.2f} mm (below normal 2 mm)")
            elif d0 > 5.0:
                zh.append(f"基线直径 {d0:.2f} mm（高于正常 5 mm）")
                en.append(f"baseline diameter {d0:.2f} mm (above normal 5 mm)")
            else:
                zh.append(f"基线直径 {d0:.2f} mm（正常 2–5 mm）")
                en.append(f"baseline diameter {d0:.2f} mm (normal 2-5 mm)")
        if pct is not None and amp is not None:
            if pct < 5:
                zh.append(f"收缩幅度 {amp:.2f} mm（{pct:.1f}% 基线，对光反射消失）")
                en.append(f"constriction {amp:.2f} mm ({pct:.1f}% of baseline, absent light reflex)")
            elif pct < 15:
                zh.append(f"收缩幅度 {amp:.2f} mm（{pct:.1f}% 基线，对光反射减弱）")
                en.append(f"constriction {amp:.2f} mm ({pct:.1f}% of baseline, reduced light reflex)")
            else:
                zh.append(f"收缩幅度 {amp:.2f} mm（{pct:.1f}% 基线，对光反射存在）")
                en.append(f"constriction {amp:.2f} mm ({pct:.1f}% of baseline, present light reflex)")
        if mcv is not None and npi is not None:
            zh.append(f"MCV {mcv:.2f} mm/s，NPi {npi:.2f}")
            en.append(f"MCV {mcv:.2f} mm/s, NPi {npi:.2f}")
        warns = [r for r in interp if len(r) >= 3 and str(r[1]).startswith("⚠️")]
        if warns:
            zh.append(f"异常 {len(warns)} 项：" + "；".join(f"{r[0]}" for r in warns))
            en.append(f"{len(warns)} finding(s): " + "; ".join(_EN_NAMES.get(str(r[0]), str(r[0])) for r in warns))
            try:
                ppt = ab.ppt_reference_single(interp)
                conds = [p0 for p0 in ppt if len(p0) >= 2 and str(p0[1]).startswith("⚠️")
                         and "PLR 质控" not in str(p0[0])]
                if conds:
                    zh.append("相关病症：" + "；".join(c[0] for c in conds))
                    en.append("related conditions: " + "; ".join(_COND_EN.get(str(c[0]), str(c[0])) for c in conds))
            except Exception:
                applog.log_here()
        else:
            zh.append("未检出明显异常")
            en.append("no significant abnormality")
        return "；".join(zh) + "。", "; ".join(en) + "."

    l_zh, l_en = _eye_summary(l_plr, l_interp)
    r_zh, r_en = _eye_summary(r_plr, r_interp)

    d0_l, d0_r = _plr_get(l_plr, "基线直径"), _plr_get(r_plr, "基线直径")
    pct_l, pct_r = _plr_get(l_plr, "收缩百分比"), _plr_get(r_plr, "收缩百分比")
    mcv_l, mcv_r = _plr_get(l_plr, "最大收缩速度"), _plr_get(r_plr, "最大收缩速度")
    npi_l, npi_r = _plr_get(l_plr, "NPi 近似评分"), _plr_get(r_plr, "NPi 近似评分")

    zh_dual, en_dual = [], []
    if d0_l is not None and d0_r is not None:
        diff = abs(d0_l - d0_r)
        if diff > 1.0:
            zh_dual.append(f"双眼基线差 {diff:.2f} mm，超过 1 mm 阈值")
            en_dual.append(f"inter-eye baseline difference {diff:.2f} mm, exceeding 1 mm")
        else:
            zh_dual.append(f"双眼基线差 {diff:.2f} mm，在 1 mm 阈值内")
            en_dual.append(f"inter-eye baseline difference {diff:.2f} mm, within 1 mm")
    if pct_l is not None and pct_r is not None:
        zh_dual.append(f"左眼收缩 {pct_l:.1f}% vs 右眼 {pct_r:.1f}%")
        en_dual.append(f"constriction {pct_l:.1f}% (left) vs {pct_r:.1f}% (right)")
    if mcv_l is not None and mcv_r is not None:
        zh_dual.append(f"左眼 MCV {mcv_l:.2f} vs 右眼 {mcv_r:.2f} mm/s")
        en_dual.append(f"MCV {mcv_l:.2f} vs {mcv_r:.2f} mm/s")
    try:
        dual_rows = _dual_interp(l_plr, r_plr, l_interp, r_interp, None)
        warns = [r for r in dual_rows if len(r) >= 5 and str(r[4]).startswith("⚠️")]
        if warns:
            zh_dual.append(f"双眼判读检出 {len(warns)} 项异常：" + "；".join(f"{r[0]}" for r in warns))
            en_dual.append(f"{len(warns)} dual-eye finding(s): " + "; ".join(_EN_NAMES.get(str(r[0]), str(r[0])) for r in warns))
            try:
                ppt_dual = ab.ppt_reference_dual(dual_rows)
                exclude = ("双侧 MCV 差", "双侧 NPi 差", "双侧收缩幅度差",
                           "双侧潜伏期差", "双侧恢复时间差", "PLR 质控")
                conds = [p0 for p0 in ppt_dual if len(p0) >= 2 and str(p0[1]).startswith("⚠️")
                         and not any(x in str(p0[0]) for x in exclude)]
                if conds:
                    zh_dual.append("相关病症：" + "；".join(c[0] for c in conds))
                    en_dual.append("related conditions: " + "; ".join(_COND_EN.get(str(c[0]), str(c[0])) for c in conds))
            except Exception:
                applog.log_here()
        else:
            zh_dual.append("双眼判读未检出明显异常")
            en_dual.append("no significant dual-eye abnormality")
    except Exception:
        applog.log_here()

    zh = [
        f"<p><b>左眼：</b>{_esc(l_zh)}</p>",
        f"<p><b>右眼：</b>{_esc(r_zh)}</p>",
        f"<p><b>双眼对比：</b>{_esc('；'.join(zh_dual))}。</p>",
        "<p>本报告为自动化分析，仅供研究与决策支持，不构成临床诊断，需结合临床。</p>",
    ]
    en = [
        f"<p><b>Left eye:</b> {_esc(l_en)}</p>",
        f"<p><b>Right eye:</b> {_esc(r_en)}</p>",
        f"<p><b>Dual-eye:</b> {_esc('; '.join(en_dual))}.</p>",
        "<p>This is an automated analysis for research and decision-support purposes only. It does not constitute a clinical diagnosis and requires clinical correlation.</p>",
    ]
    return (f'<h2>结论 / Impression</h2>'
            f'<div class="impression-zh">{"".join(zh)}</div>'
            f'<div class="impression-en">{"".join(en)}</div>')


def _dual_interp(lpl, rpl, l_interp_rows=None, r_interp_rows=None, gaze_diff=None):
    def to_dict(params):
        d = {}
        for item in params:
            if len(item) < 2:
                continue
            k, v = item[0], item[1]
            try:
                d[str(k).split(" (")[0].strip()] = float(v)
            except Exception:
                applog.log_here()
        return d

    dl, dr = to_dict(lpl), to_dict(rpl)

    def get_val(d, prefix):
        for k, v in d.items():
            if k.startswith(prefix):
                return v
        return None

    rows = []

    def add(name, unit="", warn_th=0.5, warn_text=""):
        a, b = get_val(dl, name), get_val(dr, name)
        if a is None or b is None:
            return
        d = abs(a - b)
        if warn_th is not None:
            if d > warn_th:
                verdict = f"⚠️ 是（差 {d:.3f}{unit} > {warn_th}{unit}）" + (f" {warn_text}" if warn_text else "")
            else:
                verdict = f"✅ 否（差 {d:.3f}{unit} ≤ {warn_th}{unit}）"
        else:
            verdict = "—"
        rows.append([name, f"{a:.3f}{unit}", f"{b:.3f}{unit}", f"{d:.3f}{unit}", verdict])

    a, b = get_val(dl, "基线直径"), get_val(dr, "基线直径")
    if a is not None and b is not None:
        d = abs(a - b)
        verdict = ("⚠️ 病理性不等大（>1mm；共识：提示 ICP 升高/脑疝，需紧急降颅压）"
                   if d > 1.0 else "✅ 生理性（<1mm）")
        rows.append(["基线直径", f"{a:.2f}", f"{b:.2f}", f"{d:.2f}", verdict])
        if a > 5.0 and b > 5.0:
            rows.append(["双侧散大", f"{a:.2f}", f"{b:.2f}", "—", "⚠️ 是（双侧 >5mm）；提示双侧散大，需排查脑疝/药物/代谢"])
        else:
            rows.append(["双侧散大", f"{a:.2f}", f"{b:.2f}", "—", "✅ 否（非双侧 >5mm）"])

    add("收缩幅度", warn_th=0.1, warn_text="单侧收缩幅度差 >0.1")
    add("收缩百分比", "%", warn_th=10.0, warn_text="单侧反应弱")
    add("最大收缩速度", warn_th=0.5, warn_text="单侧神经受损")
    add("NPi 近似评分", warn_th=0.5, warn_text="单侧反应异常")
    add("潜伏期", "s", warn_th=0.1, warn_text="单侧传导慢")
    add("收缩时间", "s", warn_th=0.2, warn_text="单侧收缩慢 >0.2s")
    add("恢复时间", "s", warn_th=0.5, warn_text="单侧恢复慢")

    if gaze_diff is not None:
        th_d, tv_d = gaze_diff["th_diff"], gaze_diff["tv_diff"]
        th_v = "⚠️ 是（差 >5°）" if th_d > 5.0 else "✅ 否（差 ≤5°）"
        tv_v = "⚠️ 是（差 >5°）" if tv_d > 5.0 else "✅ 否（差 ≤5°）"
        rows.append(["双眼 gaze 水平差", f"{gaze_diff['th_mean_l']:.2f}°", f"{gaze_diff['th_mean_r']:.2f}°", f"{th_d:.2f}°", th_v + "（示意，需同人同步）"])
        rows.append(["双眼 gaze 垂直差", f"{gaze_diff['tv_mean_l']:.2f}°", f"{gaze_diff['tv_mean_r']:.2f}°", f"{tv_d:.2f}°", tv_v + "（示意，需同人同步）"])
    else:
        rows.append(["双眼 gaze 水平差", "—", "—", "—", "无 gaze.csv 或数据不足"])
        rows.append(["双眼 gaze 垂直差", "—", "—", "—", "无 gaze.csv 或数据不足"])

    return rows


def _read_gaze_mean(gaze_csv):
    import csv as _csv2
    import numpy as _np
    th, tv = [], []
    with open(gaze_csv) as f:
        for r in _csv2.DictReader(f):
            try:
                th.append(float(r["theta_h"]))
                tv.append(float(r["theta_v"]))
            except Exception:
                applog.log_here()
    if not th or not tv:
        return None
    return {"th_mean": float(_np.mean(th)), "tv_mean": float(_np.mean(tv)),
            "th_std": float(_np.std(th)), "tv_std": float(_np.std(tv))}


def _section_dual_comparison(csv_left, csv_right):
    parts = ["<h2>Dual-Eye Comparison</h2>", "<p class=\"note\">本组图对比左右眼的直径、速度、轨迹、分布和差值。Diameter difference 图中超出 ±1mm 的帧提示病理性不等大。</p>"]
    try:
        morph_cmp, _ = ab.compare_eyes_plots(csv_left, csv_right, plot_dir=str(PLOT_DIR))
        names = ["Diameter overlay (L vs R)", "Speed overlay (L vs R)",
                 "Trajectory overlay (L vs R, centered)", "Major axis distribution (L vs R)",
                 "Diameter difference (L - R)"]
        for i, p in enumerate(morph_cmp):
            parts.append(_img_html(p, width="85%", caption=names[i] if i < len(names) else None))
    except Exception as e:
        parts.append(f"<p>morph comparison error: {_esc(e)}</p>")
    try:
        plr_cmp, _ = ab.compare_eyes_plr_plots(csv_left, csv_right, plot_dir=str(PLOT_DIR))
        names = ["PLR curve overlay (L vs R)", "Normalized PLR overlay (L vs R)"]
        for i, p in enumerate(plr_cmp):
            parts.append(_img_html(p, width="85%", caption=names[i] if i < len(names) else None))
    except Exception as e:
        parts.append(f"<p>PLR comparison error: {_esc(e)}</p>")
    return "".join(parts)


def _section_dual_clinical(csv_left, csv_right):
    import vogeo_backend as vb
    l_stem, r_stem = Path(csv_left).stem, Path(csv_right).stem
    l_gaze = store.gaze_csv(l_stem)
    r_gaze = store.gaze_csv(r_stem)

    l_plr, _, _ = ab.analyze_plr(csv_left)
    r_plr, _, _ = ab.analyze_plr(csv_right)
    l_interp, _ = ab.clinical_interpretation(csv_left, gaze_csv=l_gaze if l_gaze.exists() else None)
    r_interp, _ = ab.clinical_interpretation(csv_right, gaze_csv=r_gaze if r_gaze.exists() else None)

    gaze_diff = None
    if l_gaze.exists() and r_gaze.exists():
        lm, rm = _read_gaze_mean(l_gaze), _read_gaze_mean(r_gaze)
        if lm and rm:
            gaze_diff = {
                "th_mean_l": lm["th_mean"], "th_mean_r": rm["th_mean"],
                "tv_mean_l": lm["tv_mean"], "tv_mean_r": rm["tv_mean"],
                "th_diff": abs(lm["th_mean"] - rm["th_mean"]),
                "tv_diff": abs(lm["tv_mean"] - rm["tv_mean"]),
            }

    parts = ["<h2>Dual-Eye Interpretation</h2>", "<p class=\"note\">本表逐项对比左右眼 PLR 参数，Diff 列 > 阈值时标 ⚠️。下方 Condition Reference 表为双眼病症对照。</p>"]
    dual_rows = _dual_interp(l_plr, r_plr, l_interp, r_interp, gaze_diff)
    if dual_rows:
        parts.append(_table_html([["Item", "Left", "Right", "Diff", "Verdict"]] + dual_rows))

    try:
        ppt_dual = ab.ppt_reference_dual(dual_rows)
        if ppt_dual:
            parts.append("<h2>Dual-Eye Condition Reference</h2>")
            parts.append(_table_html([["Condition", "Verdict", "L/R Comparison", "Criteria"]] + ppt_dual))
    except Exception as e:
        parts.append(f"<p>condition reference error: {_esc(e)}</p>")

    if l_gaze.exists() and r_gaze.exists():
        try:
            gaze_plots = ab.compare_eyes_gaze_plots(l_gaze, r_gaze, plot_dir=str(PLOT_DIR))
            names = ["Gaze angles overlay (L vs R)", "Gaze 2D trajectory overlay (L vs R)"]
            for i, gp in enumerate(gaze_plots):
                parts.append(_img_html(gp, width="85%", caption=names[i] if i < len(names) else None))
        except Exception as e:
            parts.append(f"<p>gaze comparison error: {_esc(e)}</p>")
    return "".join(parts)


def _build_html(csv_name, dual=False, csv_right=None, stimulus_time=None, stimulus_time_right=None):
    stem = Path(csv_name).stem
    dataset, model = _parse_dataset_model(stem)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = f"Pupil Analysis Report - {stem}"

    cover_stem = f"dual_{stem}_vs_{Path(csv_right).stem}" if (dual and csv_right) else stem
    body = [_cover_html(cover_stem, dataset, model, ts)]

    if dual and csv_right:
        # ===== 双眼 =====
        r_stem = Path(csv_right).stem
        r_dataset, r_model = _parse_dataset_model(r_stem)
        body.append('<div class="page-break"></div>')
        body.append(_section_impression_dual(csv_name, csv_right))
        body.append(f"<h1>Left Eye - {_esc(stem)}</h1>")
        body.append('<div class="subtitle">Pupil Segmentation and PLR Analysis Report (Left Eye)</div>')
        body.append(f'<div class="meta"><b>Dataset:</b> {_esc(dataset)} &nbsp;|&nbsp; <b>Model:</b> {_esc(model)} &nbsp;|&nbsp; <b>Source:</b> {_esc(stem)}.csv &nbsp;|&nbsp; <b>Generated:</b> {ts}</div>')
        body.append(_section_video_overview(csv_name))
        body.append(_section_3d(csv_name))
        body.append(_section_morph(csv_name))
        body.append(_section_plr(csv_name, stimulus_time=stimulus_time))
        body.append(_section_clinical(csv_name, full=True))
        body.append(_section_abnormal(csv_name))
        body.append('<div class="page-break"></div>')
        body.append(f"<h1>Right Eye - {_esc(r_stem)}</h1>")
        body.append('<div class="subtitle">Pupil Segmentation and PLR Analysis Report (Right Eye)</div>')
        body.append(f'<div class="meta"><b>Dataset:</b> {_esc(r_dataset)} &nbsp;|&nbsp; <b>Model:</b> {_esc(r_model)} &nbsp;|&nbsp; <b>Source:</b> {_esc(r_stem)}.csv &nbsp;|&nbsp; <b>Generated:</b> {ts}</div>')
        body.append(_section_video_overview(csv_right))
        body.append(_section_3d(csv_right))
        body.append(_section_morph(csv_right))
        body.append(_section_plr(csv_right, stimulus_time=stimulus_time_right))
        body.append(_section_clinical(csv_right, full=True))
        body.append(_section_abnormal(csv_right))
        body.append('<div class="page-break"></div>')
        body.append("<h1>Dual-Eye Comparison</h1>")
        body.append(_section_dual_comparison(csv_name, csv_right))
        body.append(_section_dual_clinical(csv_name, csv_right))
        body.append('<div class="disclaimer"><b>Disclaimer:</b> This report is generated by an automated algorithm '
                    'for research and decision-support purposes only. It does not constitute a clinical diagnosis.</div>')
    else:
        # ===== 单眼 =====
        body.append('<div class="page-break"></div>')
        body.append(f"<h1>{_esc(title)}</h1>")
        body.append('<div class="subtitle">Pupil Segmentation and PLR Analysis Report</div>')
        body.append(f'<div class="meta"><b>Dataset:</b> {_esc(dataset)} &nbsp;|&nbsp; <b>Model:</b> {_esc(model)} &nbsp;|&nbsp; <b>Source:</b> {_esc(stem)}.csv &nbsp;|&nbsp; <b>Generated:</b> {ts}</div>')
        try:
            import vogeo_backend as vb
            gaze_csv = store.gaze_csv(stem)
            _, summary_text = ab.clinical_interpretation(csv_name, gaze_csv=gaze_csv if gaze_csv.exists() else None)
            first = summary_text.split("\n\n")[0].split("\n")[0] if summary_text else ""
            if first:
                body.append(f'<div class="exec-summary"><b>Summary:</b> {_esc(first)}</div>')
        except Exception:
            applog.log_here()
        body.append(_section_impression_single(csv_name))
        body.append(_section_video_overview(csv_name))
        body.append(_section_3d(csv_name))
        body.append(_section_morph(csv_name))
        body.append(_section_plr(csv_name, stimulus_time=stimulus_time))
        body.append(_section_clinical(csv_name))
        body.append(_section_abnormal(csv_name))
        body.append('<div class="disclaimer"><b>Disclaimer:</b> This report is generated by an automated algorithm '
                    'for research and decision-support purposes only. It does not constitute a clinical diagnosis.</div>')

    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{_esc(title)}</title>{CSS}</head><body>{''.join(body)}</body></html>"


def _html_to_pdf(html_str, out_pdf):
    tmp_html = Path("/tmp/_report.html")
    tmp_html.write_text(html_str, encoding="utf-8")
    cmd = [
        "wkhtmltopdf", "--enable-local-file-access", "--encoding", "utf-8",
        "--page-size", "A4",
        "--margin-top", "16mm", "--margin-bottom", "16mm",
        "--margin-left", "12mm", "--margin-right", "12mm",
        "--header-font-size", "8", "--header-center", "Pupil Analysis Report", "--header-line",
        "--footer-font-size", "8", "--footer-left", "Pupil Recognition Web",
        "--footer-right", "Page [page] / [topage]",
        str(tmp_html), str(out_pdf),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"wkhtmltopdf failed: {r.stderr[-800:]}")
    return str(out_pdf)


def export_single_eye_pdf(csv_name, stimulus_time=None):
    csv_name = Path(csv_name).name
    stem = Path(csv_name).stem
    html_str = _build_html(csv_name, stimulus_time=stimulus_time)
    out = OUT_DIR / f"{stem}_report.pdf"
    _html_to_pdf(html_str, out)
    return str(out)


def export_dual_eye_pdf(csv_left, csv_right, stimulus_time_l=None, stimulus_time_r=None):
    csv_left = Path(csv_left).name
    csv_right = Path(csv_right).name
    l_stem, r_stem = Path(csv_left).stem, Path(csv_right).stem
    html_str = _build_html(csv_left, dual=True, csv_right=csv_right,
                           stimulus_time=stimulus_time_l, stimulus_time_right=stimulus_time_r)
    out = OUT_DIR / f"dual_{l_stem}_vs_{r_stem}_report.pdf"
    _html_to_pdf(html_str, out)
    return str(out)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        print(export_single_eye_pdf(sys.argv[1]))
    elif len(sys.argv) == 3:
        print(export_dual_eye_pdf(sys.argv[1], sys.argv[2]))
    else:
        print("usage: python pdf_export.py <csv> | python pdf_export.py <csv_left> <csv_right>")
