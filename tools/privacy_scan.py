#!/usr/bin/env python3
"""提交前隐私扫描：确保患者信息、手机号、身份证号不会被推到 GitHub。

用法：python tools/privacy_scan.py [--staged]
  扫描暂存区（--staged，默认）或全部跟踪文件。
退出码：0 = 干净，1 = 发现必须处理的内容。
"""
from __future__ import annotations

import re
import subprocess
import sys

TEXT_EXT = {".py", ".md", ".txt", ".sh", ".yml", ".yaml", ".json", ".service", ".html", ".css"}

# 必须拦下：手机号、身份证号
HARD = [
    ("手机号", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("身份证号", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
]
# 提醒人工确认：床号 + 姓名、疑似真实姓名
SOFT = [
    ("床号+姓名", re.compile(r"\d{1,3}\s*床\s*[\u4e00-\u9fa5]{2,4}")),
]


def tracked_files(staged: bool) -> list[str]:
    cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"] if staged else ["git", "ls-files"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    return [f for f in out.splitlines() if f.strip()]


def main() -> int:
    staged = "--all" not in sys.argv
    files = tracked_files(staged)
    hard_hits, soft_hits = [], []
    for f in files:
        if "." not in f or f.rsplit(".", 1)[-1].lower() not in {e.lstrip(".") for e in TEXT_EXT}:
            continue
        try:
            text = open(f, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for name, pat in HARD:
            for m in pat.finditer(text):
                hard_hits.append((f, name, m.group(0)))
        for name, pat in SOFT:
            for m in pat.finditer(text):
                soft_hits.append((f, name, m.group(0)))

    if hard_hits:
        print("❌ 发现必须处理的内容：")
        for f, name, v in hard_hits[:10]:
            print(f"   {f}: {name} {v}")
        return 1
    if soft_hits:
        print("⚠️ 以下内容请确认不是真实患者信息：")
        for f, name, v in soft_hits[:10]:
            print(f"   {f}: {name} {v}")
    else:
        print("✅ 隐私扫描通过（无手机号/身份证号/床号+姓名）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
