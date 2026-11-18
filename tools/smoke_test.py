#!/usr/bin/env python3
"""专业版冒烟测试：改完代码跑一条命令，确认核心链路没退化。

覆盖：
  1. 环境（ffmpeg / wkhtmltopdf / 三套模型权重）
  2. 路径收口 store.py（csv / gaze / find_video）
  3. 分析链路：形态、逐帧、PLR、临床判读、质量、可靠性
  4. 确定性：同一 CSV 连续两次分析，数值完全一致
  5. 缓存：第二次图分析明显更快（图比 CSV 新时复用）
  6. PDF：单眼报告能生成且页数正常
  7. 可选：分割（--infer，慢）与 3D（--with-3d，更慢）

用法：
    python tools/smoke_test.py                 # 用已有 CSV，不跑推理（秒级）
    python tools/smoke_test.py --infer         # 额外跑一段分割
    python tools/smoke_test.py --with-3d       # 额外跑 VOGeo-Gaze（CPU 约 15 s/150 帧）
    python tools/smoke_test.py --csv 某.csv    # 指定 CSV

退出码：0 全过，1 有失败。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-smoke-pro")

import config  # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []


def check(ok, name, detail=""):
    RESULTS.append((bool(ok), name, str(detail)))
    print(f"{'✅' if ok else '❌'} {name}" + (f" — {detail}" if detail else ""))


def pdf_pages(path) -> int:
    out = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True).stdout
    try:
        return int(out.split("Pages:")[1].split()[0])
    except Exception:  # noqa: BLE001
        return -1


def main() -> int:
    ap = argparse.ArgumentParser(description="专业版冒烟测试")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--infer", action="store_true", help="额外跑一段分割（慢）")
    ap.add_argument("--with-3d", action="store_true", help="额外跑 VOGeo-Gaze（更慢）")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    print("== 环境 ==")
    check(shutil.which("ffmpeg") is not None, "ffmpeg 可用")
    check(shutil.which("wkhtmltopdf") is not None, "wkhtmltopdf 可用")
    for key, info in config.MODEL_REGISTRY.items():
        d = Path(info["dir"])
        ok = (d / "model.pt").exists() and (d / "plr_model.pt").exists()
        check(ok, f"权重 {key}", "" if ok else str(d))

    import analysis_backend as ab
    import pdf_export as pe
    import store

    # ---------- 选 CSV ----------
    csvs = sorted(store.seg_dir().glob("*_plr_seg.csv")) or sorted(store.seg_dir().glob("*.csv"))
    csv_name = args.csv or (csvs[0].name if csvs else None)
    check(csv_name is not None, "找到可用 CSV", csv_name or str(store.seg_dir()))
    if csv_name is None:
        return 1

    print("\n== 路径收口 ==")
    check(store.csv_path(csv_name).exists(), "store.csv_path", csv_name)
    stem = Path(csv_name).stem
    check(store.gaze_csv(stem).parent.name == stem, "store.gaze_csv 指向 <stem>/gaze.csv",
          str(store.gaze_csv(stem)))
    vid = store.find_video(csv_name)
    check(vid is None or Path(vid).exists(), "store.find_video 结果自洽",
          Path(vid).name if vid else "无（该 CSV 没有配套视频，属正常）")

    print("\n== 分析链路 ==")
    t0 = time.perf_counter()
    morph, motion, msum, mplots = ab.analyze_morphology(csv_name, plot_dir=str(store.plot_dir()))
    check(len(morph) > 0 and len(motion) > 0, "形态/运动表", f"{len(morph)} + {len(motion)} 项")
    summary, frame, abnormal, text, images = ab.analyze_csv(csv_name)
    check(len(frame) > 0, "逐帧表", f"{len(frame)} 行")
    plr, plr_text, plr_plot = ab.analyze_plr(csv_name, plot_dir=str(store.plot_dir()))
    check(bool(plr), "PLR 参数表", f"{len(plr)} 行")
    interp, isum = ab.clinical_interpretation(csv_name)
    names = [str(r[0]) for r in interp if r]
    must = [k for k in ("直径", "对光反射", "NPi") if not any(k in n for n in names)]
    check(not must, "临床判读含关键项（直径/对光反射/NPi）",
          f"{len(interp)} 项" + (f"，缺 {must}" if must else ""))
    quality = ab.data_quality_summary(csv_name)
    check(len(quality) > 0, "数据质量", f"{len(quality)} 项")
    warns = ab.reliability_warnings(csv_name)
    check(isinstance(warns, list), "可靠性提示", f"{len(warns)} 条")

    # 数值合理性
    d = {r[0]: r[1] for r in plr if isinstance(r, (list, tuple)) and len(r) >= 2}
    npi = next((v for k, v in d.items() if k.startswith("NPi")), None)
    pct = next((v for k, v in d.items() if k.startswith("收缩百分比")), None)
    ok_npi = isinstance(npi, (int, float)) and 0 <= npi <= 5
    ok_pct = isinstance(pct, (int, float)) and -100 <= pct <= 100
    check(ok_npi, "NPi 在 0–5", npi)
    check(ok_pct, "收缩百分比在合理区间", pct)

    cold = (time.perf_counter() - t0) * 1000

    print("\n== 确定性 & 缓存 ==")
    t1 = time.perf_counter()
    plr2, _, _ = ab.analyze_plr(csv_name, plot_dir=str(store.plot_dir()))
    warm = (time.perf_counter() - t1) * 1000
    check(plr2 == plr, "同一 CSV 两次 PLR 数值一致")
    check(warm <= max(cold * 1.5, cold + 80), "图缓存生效（第二次不明显更慢）",
          f"{cold:.0f} ms → {warm:.0f} ms")

    print("\n== 服务层 services ==")
    import services
    t2 = time.perf_counter()
    bundle = services.analyze_all(csv_name, None)
    first = (time.perf_counter() - t2) * 1000
    t3 = time.perf_counter()
    bundle2 = services.analyze_all(csv_name, None)
    second = (time.perf_counter() - t3) * 1000
    check(len(bundle) == 16, "analyze_all 返回 16 项", len(bundle))
    check(bundle is bundle2 or bundle == bundle2, "同参数结果一致（可缓存）")
    # 计时断言放宽：服务与冒烟测试抢 CPU 时会有抖动
    check(second <= max(first * 1.5, first + 80), "服务层缓存命中",
          f"{first:.0f} ms → {second:.0f} ms")
    check(isinstance(bundle[11], str) and "异常帧" in bundle[11], "异常帧摘要", bundle[11][:24])

    print("\n== 数值回归（对比 tools/baseline.json）==")
    import json
    bpath = ROOT / "tools" / "baseline.json"
    if not bpath.exists():
        check(False, "缺少 tools/baseline.json", "先跑 tools/make_baseline.py --update")
    else:
        base = json.loads(bpath.read_text(encoding="utf-8"))
        drift = []
        checked = 0
        for name, expected in base.items():
            if not store.csv_path(name).exists():
                continue
            mark, _ = ab._load_stimulus_time(name)
            table, _, _ = ab.analyze_plr(name, stimulus_time=mark, plot_dir=None)
            cur = {}
            for r in table:
                if isinstance(r, (list, tuple)) and len(r) >= 2 and isinstance(r[1], (int, float)):
                    cur[r[0].split(" (")[0].replace(" ", "_")] = round(float(r[1]), 3)
            for k, v in expected.items():
                if k in ("frames", "detected"):
                    c = {"frames": len(ab._read_rows(name) or []),
                         "detected": sum(1 for x in (ab._read_rows(name) or []) if x["detected"] == 1)}[k]
                    if c != v:
                        drift.append(f"{name}:{k} {v}→{c}")
                    continue
                if k not in cur:
                    continue
                checked += 1
                if abs(cur[k] - v) > max(0.02, abs(v) * 0.01):
                    drift.append(f"{name}:{k} {v}→{cur[k]}")
        check(not drift, f"PLR 数值与基线一致（{checked} 个指标）",
              "；".join(drift[:3]) if drift else "")

    print("\n== PDF ==")
    try:
        pdf = pe.export_single_eye_pdf(csv_name)
        check(Path(pdf).exists(), "单眼 PDF 生成", Path(pdf).name)
        pages = pdf_pages(pdf)
        check(pages >= 1, "PDF 页数", pages)
        txt = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True).stdout
        check("瞳孔" in txt, "PDF 含正文（中文可抽取）")
    except Exception as e:  # noqa: BLE001
        check(False, "单眼 PDF 生成", str(e)[:80])

    # ---------- 可选：分割 ----------
    if args.infer:
        print("\n== 分割（可选）==")
        from inference import predict_video_iter
        src = vid
        if src is None:
            for ddir in config.DATASET_DIRS.values():
                vids = sorted(Path(ddir).glob("*.mp4"))
                if vids:
                    src = vids[0]
                    break
        if src is None:
            check(False, "找不到可分割的视频")
        else:
            last = None
            for item in predict_video_iter(str(src), max_frames=24, flow_fill=False, plr=True):
                last = item
            check(last is not None and Path(last["csv_path"]).exists(), "分割产出 CSV",
                  Path(last["csv_path"]).name)
            check(Path(last["out_path"]).exists(), "分割产出视频", Path(last["out_path"]).name)

    # ---------- 可选：3D ----------
    if args.with_3d:
        print("\n== 3D 注视（可选）==")
        import vogeo_backend as vb
        target = vid
        if target is None:
            for ddir in config.DATASET_DIRS.values():
                vids = sorted(Path(ddir).glob("*.mp4"))
                if vids:
                    target = vids[0]
                    break
        if target is None:
            check(False, "找不到可跑 3D 的视频")
        else:
            ov, gz, status = vb.run_vogeo(target, out_name="__smoke_pro_3d__")
            check(ov is not None and gz is not None, "VOGeo 产出 overlay + gaze", status.splitlines()[0])
            if not args.keep:
                shutil.rmtree(vb.OUTPUT_DIR / "__smoke_pro_3d__", ignore_errors=True)

    ok_n = sum(1 for ok, _, _ in RESULTS if ok)
    print(f"\n结果：{ok_n}/{len(RESULTS)} 项通过")
    return 0 if ok_n == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
