#!/usr/bin/env python3
"""从 PLR 曲线提取临床参数（命令行入口）。

计算逻辑与网页 / PDF 完全一致，统一在 :mod:`plr_core`；本文件只负责参数解析、
排版与出图。旧版这里有一份与 ``analysis_backend.analyze_plr`` 逐行重复的实现，
两边公式各自维护，本版已合并。

用法::

    python plr_params.py --csv outputs/segmentation_results/S_1_baseline_plr_seg.csv \\
        --stimulus 1.49 --plot /tmp/S_1_plr.png --output /tmp/S_1_plr.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - 必须先设好后端再导入 pyplot

import plr_core  # noqa: E402


def _plot(curve, path):
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
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Pupil diameter ({curve.unit})")
    ax.set_title(f"PLR analysis - {Path(curve.csv_path).name}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="从 PLR 曲线提取临床参数（适配 pupil_recognition_web CSV）")
    ap.add_argument("--csv", required=True, help="逐帧 CSV")
    ap.add_argument("--stimulus", type=float, default=None,
                    help="光刺激开始时间(秒)，不传则自动检测")
    ap.add_argument("--output", default=None, help="参数输出 txt")
    ap.add_argument("--plot", default=None, help="分析图 PNG")
    ap.add_argument("--mode", default=None, choices=["v2", "legacy"],
                    help="v2=刺激窗+基线窗（默认），legacy=旧版全序列 argmin")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ 文件不存在: {csv_path}")
        return 1

    print("=" * 60)
    print("📊 瞳孔对光反射 (PLR) 参数提取")
    print("=" * 60)

    result = plr_core.analyze(csv_path, stimulus_time=args.stimulus, mode=args.mode)
    if result is None:
        print(f"⚠️ 刺激后窗口无有效数据或有效点太少（<{plr_core.PLR_MIN_VALID_POINTS}），无法提取参数")
        return 1

    curve, params, quality = result.curve, result.params, result.quality
    table = plr_core.params_table(params, curve.unit, quality=quality, response=result.response)

    print(f"\n📍 光刺激开始时间: {curve.stimulus_time:.2f}s（{curve.stimulus_source}）")
    print(f"📏 直径单位: {curve.unit}｜分析模式: {quality.mode}")
    print(f"🩺 响应判定: {'✅ 有收缩' if result.response else '⚠️ 未引出（Δ<5%）'}")
    print("\n📈 提取的临床参数:")
    print("-" * 50)
    for name, value, _ref in table:
        print(f"  {name:24s}: {value}")
    print("-" * 50)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "瞳孔对光反射(PLR)参数提取结果",
            f"数据来源: {csv_path}",
            f"光刺激开始时间: {curve.stimulus_time:.2f}s（{curve.stimulus_source}）",
            f"直径单位: {curve.unit}",
            "",
        ]
        lines += [f"{name}: {value}" for name, value, _ref in table]
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"💾 参数已保存: {out}")

    if args.plot:
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        _plot(curve, args.plot)
        print(f"📊 分析图已保存: {args.plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
