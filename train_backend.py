"""
瞳孔分割训练后端：直接调用三个训练脚本，参数通过命令行传递。
"""
import os
import re
import signal
import subprocess
import time
from pathlib import Path

from config import (
    DATASET_YAMLS, STATS_CACHE, STATS_CHECKPOINT, STATS_TRAIN_SCRIPT, TRAIN_LOG_DIR,
    TRAIN_PROJECT_DIR, TRAIN_SCRIPTS_DIR, TRAIN_YAML_DIR, YOLO_ROOT,
)

SCRIPTS_DIR = Path(TRAIN_SCRIPTS_DIR)
LOG_DIR = Path(TRAIN_LOG_DIR)
LOG_DIR.mkdir(parents=True, exist_ok=True)

_PROC = None
_LOG_PATH = None

# 模型 -> (脚本, 默认输出目录)
#: 模型 YAML -> (训练脚本, 输出工程目录)。Web 训练统一加 `_web` 后缀，与手动训练隔离。
MODEL_MAP = {
    "yolo26_bifpn_seg.yaml": {
        "script": SCRIPTS_DIR / "train_yolo26_bifpn_loss_v2.py",
        "project": TRAIN_PROJECT_DIR / "yolo26_bifpn_loss_v2_web",
    },
    "yolo26_bifpn_mac_seg.yaml": {
        "script": SCRIPTS_DIR / "train_yolo26_bifpn_mac_loss.py",
        "project": TRAIN_PROJECT_DIR / "yolo26_bifpn_mac_loss_web",
    },
    "yolo26_bifpn_strip_seg.yaml": {
        "script": SCRIPTS_DIR / "train_yolo26_bifpn_strip_loss.py",
        "project": TRAIN_PROJECT_DIR / "yolo26_bifpn_strip_loss_web",
    },
}

DATASET_MAP = {name: Path(path) for name, path in DATASET_YAMLS.items()}


def is_running() -> bool:
    global _PROC
    return _PROC is not None and _PROC.poll() is None


def start_training(yaml_name: str, dataset_name: str, epochs: int, batch: int,
                   lr0: float, device: str, imgsz: int = 640) -> str:
    global _PROC, _LOG_PATH
    if is_running():
        return f"⚠️ 已有训练在跑（PID {_PROC.pid}），先停止再启动"

    info = MODEL_MAP.get(yaml_name)
    if info is None:
        return f"❌ 未知 YAML: {yaml_name}"
    script = info["script"]
    if not script.exists():
        return f"❌ 训练脚本不存在: {script}"

    data_path = DATASET_MAP.get(dataset_name)
    if data_path is None or not data_path.exists():
        return f"❌ 数据集不存在: {dataset_name}"

    ts = time.strftime("%Y%m%d_%H%M%S")
    _LOG_PATH = LOG_DIR / f"train_{ts}.log"

    cmd = [
        "python", str(script),
        "--yaml", str(Path(TRAIN_YAML_DIR) / yaml_name),
        "--data", str(data_path),
        "--epochs", str(int(epochs)),
        "--batch", str(int(batch)),
        "--imgsz", str(int(imgsz)),
        "--device", str(device),
        "--lr0", str(float(lr0)),
        "--project", str(info["project"]),
        "--name", "bifpn_run",
    ]

    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONPATH"] = str(YOLO_ROOT) + ":" + env.get("PYTHONPATH", "")

    log_f = open(_LOG_PATH, "w")
    _PROC = subprocess.Popen(
        cmd, cwd=str(YOLO_ROOT), stdout=log_f, stderr=subprocess.STDOUT,
        env=env, preexec_fn=os.setsid,
    )
    log_f.write(f"[启动] PID={_PROC.pid}\n命令: {' '.join(cmd)}\n\n")
    log_f.flush()
    return f"✅ 训练已启动，PID={_PROC.pid}\n日志: {_LOG_PATH}"


def stop_training() -> str:
    global _PROC
    if not is_running():
        return "没有正在跑的训练"
    pid = _PROC.pid
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        time.sleep(1)
        if _PROC.poll() is None:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        return f"🛑 已停止训练 PID={pid}"
    except Exception as e:
        return f"停止失败: {e}"


def read_log(max_lines: int = 200) -> str:
    if _LOG_PATH is None or not Path(_LOG_PATH).exists():
        return "（还没有训练日志）"
    try:
        with open(_LOG_PATH, "r", errors="ignore") as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:])
    except Exception as e:
        return f"读日志失败: {e}"


def parse_progress() -> str:
    if _LOG_PATH is None or not Path(_LOG_PATH).exists():
        return ""
    try:
        with open(_LOG_PATH, "r", errors="ignore") as f:
            text = f.read()
        matches = __import__("re").findall(r"(\d+)/(\d+)", text)
        if matches:
            cur, total = matches[-1]
            return f"epoch {cur}/{total}"
        if "results saved" in text or "train complete" in text.lower():
            return "训练完成"
        if "traceback" in text.lower() or "error" in text.lower():
            return "训练出错，看日志"
        return "训练中"
    except Exception:
        return ""




# ============================================================
# 运动光流特征模型训练（StatsSeqTransformer）
# ============================================================
STATS_LOG_DIR = LOG_DIR
# STATS_TRAIN_SCRIPT / STATS_CHECKPOINT / STATS_CACHE 均在 config 中定义

_STATS_PROC = None
_STATS_LOG_PATH = None


def is_stats_running() -> bool:
    global _STATS_PROC
    return _STATS_PROC is not None and _STATS_PROC.poll() is None


def start_stats_training(resume: bool = False) -> str:
    global _STATS_PROC, _STATS_LOG_PATH
    if is_stats_running():
        return f"⚠️ 已有运动光流训练在跑（PID {_STATS_PROC.pid}），先停止再启动"
    if not STATS_TRAIN_SCRIPT.exists():
        return f"❌ 训练脚本不存在: {STATS_TRAIN_SCRIPT}"
    ts = time.strftime("%Y%m%d_%H%M%S")
    _STATS_LOG_PATH = STATS_LOG_DIR / f"stats_train_{ts}.log"
    cmd = ["python", str(STATS_TRAIN_SCRIPT)]
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONPATH"] = str(YOLO_ROOT) + ":" + env.get("PYTHONPATH", "")
    log_f = open(_STATS_LOG_PATH, "w")
    _STATS_PROC = subprocess.Popen(
        cmd, cwd=str(YOLO_ROOT), stdout=log_f, stderr=subprocess.STDOUT,
        env=env, preexec_fn=os.setsid,
    )
    log_f.write(f"[启动] PID={_STATS_PROC.pid}\n命令: {' '.join(cmd)}\n\n")
    log_f.flush()
    return f"✅ 运动光流训练已启动，PID={_STATS_PROC.pid}\n日志: {_STATS_LOG_PATH}"


def stop_stats_training() -> str:
    global _STATS_PROC
    if not is_stats_running():
        return "没有正在跑的运动光流训练"
    pid = _STATS_PROC.pid
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        time.sleep(1)
        if _STATS_PROC.poll() is None:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        return f"🛑 已停止运动光流训练 PID={pid}"
    except Exception as e:
        return f"停止失败: {e}"


def read_stats_log(max_lines: int = 200) -> str:
    if _STATS_LOG_PATH is None or not Path(_STATS_LOG_PATH).exists():
        return "（还没有运动光流训练日志）"
    try:
        with open(_STATS_LOG_PATH, "r", errors="ignore") as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:])
    except Exception as e:
        return f"读日志失败: {e}"


def parse_stats_progress() -> str:
    if _STATS_LOG_PATH is None or not Path(_STATS_LOG_PATH).exists():
        return ""
    try:
        with open(_STATS_LOG_PATH, "r", errors="ignore") as f:
            text = f.read()
        matches = re.findall(r"Epoch (\d+): train_loss=([\d.]+), val_loss=([\d.]+)", text)
        if matches:
            ep, tr, va = matches[-1]
            return f"Epoch {ep} | train={tr} | val={va}"
        if "训练完成" in text:
            return "训练完成"
        if "traceback" in text.lower() or "error" in text.lower():
            return "训练出错，看日志"
        return "训练中"
    except Exception:
        return ""


def stats_checkpoint_info() -> str:
    lines = []
    lines.append(f"脚本: {STATS_TRAIN_SCRIPT} {'✅' if STATS_TRAIN_SCRIPT.exists() else '❌ 不存在'}")
    lines.append(f"缓存: {STATS_CACHE} {'✅' if STATS_CACHE.exists() else '（首次训练会构建，较慢）'}")
    if STATS_CHECKPOINT.exists():
        try:
            import torch
            ckpt = torch.load(STATS_CHECKPOINT, map_location="cpu", weights_only=False)
            lines.append(f"检查点: ✅ epoch={ckpt.get('epoch')}, best_val_loss={ckpt.get('best_val_loss'):.6f}")
        except Exception as e:
            lines.append(f"检查点: ✅ 存在但读取失败({e})")
    else:
        lines.append("检查点: ❌ 不存在（从头训练）")
    return "\n".join(lines)
