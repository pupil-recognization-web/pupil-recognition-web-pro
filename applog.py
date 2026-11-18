"""统一错误日志。

以前代码里大量 `except Exception: pass`，出问题时界面只是"没反应"，日志里什么都没有。
现在统一走这里：异常写进项目根目录的 app.log（带时间戳、位置、完整 traceback），
不影响主流程（写日志本身失败也不会再抛）。
"""
from __future__ import annotations

import traceback
from datetime import datetime
from pathlib import Path

from config import BASE_DIR

LOG_PATH = Path(BASE_DIR) / "app.log"
MAX_BYTES = 5 * 1024 * 1024      # 超过 5 MB 就滚动一次，只保留一份 .1


def _rotate_if_needed() -> None:
    try:
        if LOG_PATH.exists() and LOG_PATH.stat().st_size > MAX_BYTES:
            backup = LOG_PATH.with_suffix(".log.1")
            if backup.exists():
                backup.unlink()
            LOG_PATH.rename(backup)
    except Exception:  # noqa: BLE001
        pass


def _is_user_error() -> bool:
    """Gradio 的 gr.Error 是给用户看的校验提示，不是 bug，不必记日志。"""
    import sys

    exc = sys.exc_info()[1]
    if exc is None:
        return False
    return type(exc).__name__ in ("Error", "ErrorStatus") and type(exc).__module__.startswith("gradio")


def log_exc(context: str) -> None:
    """记录当前异常（只能在 except 块里调用）。"""
    if _is_user_error():
        return
    try:
        _rotate_if_needed()
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] {context}\n")
            f.write(traceback.format_exc())
    except Exception:  # noqa: BLE001 - 日志失败绝不向上抛
        pass


def log_here() -> None:
    """记录当前异常，并自动带上调用位置（文件:行），替代 except 里的 pass。"""
    import inspect

    frame = inspect.currentframe()
    caller = frame.f_back if frame else None
    where = f"{Path(caller.f_code.co_filename).name}:{caller.f_lineno}" if caller else "?"
    log_exc(where)


def log_msg(message: str) -> None:
    """记录一条普通信息（用于长任务进度、回退路径等）。"""
    try:
        _rotate_if_needed()
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}\n")
    except Exception:  # noqa: BLE001
        pass
