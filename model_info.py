"""模型权重体检：把“选了没生效”这类静默行为变成看得见的提示。

为什么需要
----------
旧版界面上有 baseline / mac / strip 三个模型可选，勾 PLR 后会加载
``<变体>/plr_model.pt``。实际三份 PLR 权重是同一个文件（md5 相同），三份
``plr_model.yaml`` 也逐字节相同——选 mac 或 strip 再勾 PLR，拿到的仍是那份
双类 baseline 权重，用户完全无从察觉。

这里用文件指纹把事实算出来，界面据此提示；行为保持原样（本版不擅自换模型），
只是不再假装“选择生效”。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from config import MODEL_REGISTRY


def file_md5(path) -> str | None:
    """文件 md5；不存在返回 None。"""
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _short(md5: str | None) -> str:
    return (md5 or "缺失")[:8]


def fingerprint() -> dict:
    """每个变体的权重/配置指纹。

    返回 ``{变体: {"model_pt", "plr_pt", "plr_yaml"}}``，值为 md5 或 None。
    """
    out = {}
    for key, info in MODEL_REGISTRY.items():
        d = Path(info["dir"])
        out[key] = {
            "model_pt": file_md5(d / "model.pt"),
            "plr_pt": file_md5(d / "plr_model.pt"),
            "plr_yaml": file_md5(d / "plr_model.yaml"),
        }
    return out


def plr_is_shared() -> tuple[bool, str]:
    """PLR 权重是否在所有变体间共享。

    返回 ``(是否共享, 说明)``。文件缺失或只有一个变体时视为“无法判断”。
    """
    fp = fingerprint()
    plr = {k: v["plr_pt"] for k, v in fp.items()}
    present = {k: v for k, v in plr.items() if v}
    if len(present) < 2:
        return False, "PLR 权重不完整或只有一个变体，无法比较（是否缺少 .pt 文件？）"
    uniq = set(present.values())
    if len(uniq) == 1:
        only = next(iter(uniq))
        names = " / ".join(sorted(present))
        return True, (
            f"PLR 模式：{names} 的 plr_model.pt 完全相同（md5 {_short(only)}…），"
            "对应 plr_model.yaml 也一致 —— 勾选不同“模型”不会改变 PLR 结果，"
            "实际始终加载同一份双类权重。"
        )
    groups = {}
    for k, v in present.items():
        groups.setdefault(v, []).append(k)
    detail = "；".join("、".join(g) + f" = {_short(v)}…" for v, g in groups.items())
    return False, f"PLR 模式：各变体权重互不相同（{detail}），按所选模型加载。"


def variant_fingerprints_text() -> str:
    """多行指纹表，给“环境信息”类 Textbox 用。"""
    fp = fingerprint()
    lines = []
    for key, info in MODEL_REGISTRY.items():
        f = fp[key]
        lines.append(
            f"{key:<9} model.pt={_short(f['model_pt'])}  "
            f"plr_model.pt={_short(f['plr_pt'])}  plr_model.yaml={_short(f['plr_yaml'])}"
        )
    return "\n".join(lines)


def plr_mode_note(model_name: str) -> str:
    """勾选 PLR 时显示在界面上的提示。"""
    shared, msg = plr_is_shared()
    if not shared:
        return msg
    info = MODEL_REGISTRY.get(model_name, {})
    label = info.get("label", model_name)
    return (
        f"⚠️ 已选「{label}」，但 PLR 权重在三个变体间是共享的："
        "勾 PLR 后实际使用同一份双类（pupil+iris）权重，专家模型选择对 PLR 无效。"
    )
