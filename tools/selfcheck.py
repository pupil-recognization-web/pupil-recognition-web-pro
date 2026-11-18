#!/usr/bin/env python3
"""离线自检：不启动 Web，也能确认“配置、模型、依赖、分析链路”都正常。

用法::

    python tools/selfcheck.py                # 全量自检
    python tools/selfcheck.py --csv S_1_baseline_plr_seg.csv   # 只看一份 CSV
    python tools/selfcheck.py --quiet        # 只报错，不打印明细

设计目标：把旧版“必须点开网页、上传视频才知道哪儿坏了”的问题，变成一条命令
就能定位。检查项分三级：
    FAIL  必须修（缺依赖、缺权重、CSV 解析不出来）
    WARN  环境没配好但代码没问题（数据集目录不在、缺刺激标定、检出率过低）
    OK    正常
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 让 matplotlib 用可写缓存目录，避免只读 HOME 下报警
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")

results = []  # (level, title, detail)


def check(level, title, detail=""):
    results.append((level, title, detail))


def section(name):
    if not QUIET:
        print(f"\n== {name} ==")


def main() -> int:
    global QUIET
    ap = argparse.ArgumentParser(description="pupil_recognition_web_v2 自检")
    ap.add_argument("--csv", default=None, help="只检查指定 CSV（默认全部）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    QUIET = args.quiet

    # ---------- 依赖 ----------
    section("Python 依赖")
    for mod, name in [
        ("numpy", "numpy"), ("pandas", "pandas"), ("cv2", "opencv"),
        ("matplotlib", "matplotlib"), ("yaml", "PyYAML"), ("gradio", "gradio"),
        ("torch", "torch"), ("ultralytics", "ultralytics"),
    ]:
        try:
            __import__(mod)
            check("OK", f"import {name}")
        except Exception as e:  # noqa: BLE001
            check("FAIL", f"import {name}", str(e))
    try:
        import mediapipe  # noqa: F401
        check("OK", "import mediapipe（ROI 裁剪可用）")
    except Exception as e:  # noqa: BLE001
        check("WARN", "import mediapipe（ROI 裁剪不可用）", str(e))

    # ---------- 配置与目录 ----------
    section("配置")
    import config

    check("OK", f"项目目录 {config.BASE_DIR}")
    check("OK", f"产出目录 {config.OUTPUT_DIR}")
    for name, d in config.DATASET_DIRS.items():
        if Path(d).is_dir():
            n = len(list(Path(d).glob("*.mp4")))
            check("OK", f"数据集 {name}", f"{d}（{n} 个 mp4）")
        else:
            check("WARN", f"数据集 {name} 不存在", str(d))
    if Path(config.VOGE0_ROOT).is_dir():
        check("OK", "VOGeo-Gaze 仓库", str(config.VOGE0_ROOT))
    else:
        check("WARN", "VOGeo-Gaze 仓库不存在（3D 注视功能不可用）", str(config.VOGE0_ROOT))
    if Path(config.MEDIAPIPE_MODEL).exists():
        check("OK", "MediaPipe 模型", str(config.MEDIAPIPE_MODEL))
    else:
        check("WARN", "MediaPipe 模型缺失（ROI 裁剪不可用）", str(config.MEDIAPIPE_MODEL))

    # ---------- 模型权重 ----------
    section("模型权重")
    import model_info

    for key, info in config.MODEL_REGISTRY.items():
        d = Path(info["dir"])
        for fn in ("model.pt", "model.yaml", "plr_model.pt", "plr_model.yaml"):
            if (d / fn).exists():
                check("OK", f"{key}/{fn}")
            else:
                check("FAIL", f"{key}/{fn} 缺失", str(d / fn))

    shared, msg = model_info.plr_is_shared()
    check("WARN" if shared else "OK", "PLR 权重一致性", msg)

    # ---------- 分析链路 ----------
    section("分析链路")
    import analysis_backend as ab
    import plr_core

    csvs = [args.csv] if args.csv else ab.list_csv_files()
    if not csvs:
        check("WARN", "没有可分析的 CSV", str(config.VIDEO_OUTPUT_DIR))
    for name in csvs:
        rows = ab._read_rows(name)
        if not rows:
            check("FAIL", f"CSV 解析失败: {name}")
            continue
        n = len(rows)
        det = sum(1 for r in rows if r["detected"] == 1)
        check("OK", f"CSV {name}", f"{n} 帧，检出 {det}（{det / n * 100:.1f}%）")

        for w in ab.reliability_warnings(name):
            check("WARN", f"可靠性 {name}", w)

        # 形态/运动（不写图）
        morph, motion, _, _ = ab.analyze_morphology(name, plot_dir=None)
        check("OK", f"形态/运动 {name}", f"{len(morph)} + {len(motion)} 项指标")

        # PLR（不写图；plot_dir=None 时纯计算）
        table, text, _ = ab.analyze_plr(name, plot_dir=None)
        if table:
            check("OK", f"PLR {name}", text.splitlines()[-1])
        else:
            check("WARN", f"PLR {name}", text)

        # 单眼判读
        interp, _ = ab.clinical_interpretation(name)
        check("OK", f"临床判读 {name}", f"{len(interp)} 项")

        # 导出用的 PLR 纯函数（与 CLI 共用），v2 模式带质量标记
        res = plr_core.analyze(config.VIDEO_OUTPUT_DIR / name)
        if res:
            check("OK", f"plr_core {name}",
                  f"mode={res.quality.mode} 响应={'有' if res.response else '未引出'} "
                  f"NPi={res.params.npi_like}")
            for note in res.quality.notes:
                check("INFO", f"PLR 质量 {name}", note)
        else:
            check("WARN", f"plr_core {name}", "刺激后窗口无有效数据")

    # ---------- 汇总 ----------
    fails = [r for r in results if r[0] == "FAIL"]
    warns = [r for r in results if r[0] == "WARN"]
    infos = [r for r in results if r[0] == "INFO"]
    if not QUIET:
        for level, title, detail in results:
            if level == "FAIL" or (level == "WARN" and detail):
                print(f"[{level}] {title}" + (f" — {detail}" if detail else ""))
    print(f"\n自检完成：{len(results)} 项，FAIL {len(fails)}，WARN {len(warns)}，INFO {len(infos)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
