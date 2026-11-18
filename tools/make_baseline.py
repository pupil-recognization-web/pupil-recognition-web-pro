#!/usr/bin/env python3
"""把当前 9 份 CSV 的 PLR 数值固化成基线，供 smoke_test 比对。

用法：python tools/make_baseline.py [--update]
  · 不带 --update：生成 baseline.json 到 stdout（人工确认后再落盘）
  · 带 --update：直接覆盖 tools/baseline.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-baseline")

import analysis_backend as ab
import store


def snapshot():
    out = {}
    for csv_path in sorted(store.seg_dir().glob("*.csv")):
        name = csv_path.name
        rows = ab._read_rows(name) or []
        if not rows:
            continue
        mark, _ = ab._load_stimulus_time(name)
        # 关键：把当时用的刺激时刻固化成基线的一部分，避免以后重标导致误报
        table, text, _ = ab.analyze_plr(name, stimulus_time=mark, plot_dir=None)
        d = {r[0]: r[1] for r in table if isinstance(r, (list, tuple)) and len(r) >= 2}
        keep = {}
        for k, v in d.items():
            if isinstance(v, (int, float)):
                key = (k.split(" (")[0].replace(" ", "_"))
                keep[key] = round(float(v), 3)
        keep["frames"] = len(rows)
        keep["detected"] = sum(1 for r in rows if r["detected"] == 1)
        out[name] = dict(keep, _stim=(round(mark, 4) if mark else None))
    return out


if __name__ == "__main__":
    data = snapshot()
    if "--update" in sys.argv:
        p = ROOT / "tools" / "baseline.json"
        p.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
        print(f"已写入 {p}（{len(data)} 份 CSV）")
    else:
        print(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True))
