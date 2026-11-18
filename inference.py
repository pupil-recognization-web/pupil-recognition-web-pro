"""YOLO 分割推理：单帧、逐帧视频、逐帧 CSV、光流补帧。

本版相对旧版的改动（对外行为保持不变）
--------------------------------------
1. 全部路径/阈值来自 :mod:`config`，不再有绝对路径字面量。
2. **内存**：旧版把整段视频的原始帧和 overlay 帧同时留在内存里
   （600 帧 720p ≈ 3GB，极易 OOM）。现在 overlay 直接落盘成中间视频，
   补帧阶段再流式读回，峰值内存只跟“单段连续漏检长度”有关（默认每段最多 64 帧）。
3. **换算常数**：像素到毫米的角膜参照 ``CORNEA_MM`` 从 config 取，便于按设备标定。
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from config import (
    CONF_THRESHOLD, CORNEA_MM, DEFAULT_MODEL, IMG_SIZE, IOU_THRESHOLD,
    MODEL_REGISTRY, VIDEO_FPS, VIDEO_MAX_FRAMES, VIDEO_OUTPUT_DIR, YOLO_DEVICE,
)
from flow_fill_backend import draw as draw_mask
from flow_fill_backend import mask_from_ellipse, warp_mask

_MODELS = {}


def _pick_device():
    """选推理设备：显式配置优先；否则 GPU 显存 <1.2 GB 时退回 CPU。

    这台机器常有别的任务占着显存（实测被占 5.6/8 GiB），
    与其让 YOLO 在推理中途 OOM，不如提前降级，只是慢一点。
    """
    if YOLO_DEVICE:
        return YOLO_DEVICE
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        if free < 1.2e9:
            import applog
            applog.log_msg(f"显存仅 {free/1e9:.1f} GB，本次推理自动用 CPU")
            return "cpu"
    except Exception:  # noqa: BLE001
        pass
    return None
_MODEL_LIMIT = 2          # 最多同时保留几个模型（3 变体 × PLR/非PLR 最多 6 个，太吃内存）

#: 补帧时单段最多缓存多少帧（超过则分段，用上一段末尾的 mask 继续做前向参考）
FILL_CHUNK = 64

_FARNEBACK = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                  poly_n=5, poly_sigma=1.2, flags=0)


# ============================================================
# 模型加载
# ============================================================
def _prepare_yaml(src_yaml: Path, nc: int = 1) -> str:
    """把模型 YAML 的 nc/scale 改成推理所需值，写到临时文件。"""
    with open(src_yaml) as f:
        cfg = yaml.safe_load(f)
    cfg["nc"] = nc
    cfg["scale"] = "n"
    fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="yolo_")
    os.close(fd)
    with open(tmp, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
    return tmp


def load_model(name: str = DEFAULT_MODEL, plr: bool = False):
    """按 ``(模型名, 是否 PLR)`` 缓存加载 YOLO。

    PLR 模式用 ``plr_model.pt``（2 类：pupil + iris），否则 ``model.pt``（1 类）。
    """
    key = (name, plr)
    if key in _MODELS:
        return _MODELS[key]

    info = MODEL_REGISTRY[name]
    base_dir = Path(info["dir"])
    if plr:
        yaml_path, weights, nc = base_dir / "plr_model.yaml", base_dir / "plr_model.pt", 2
    else:
        yaml_path, weights, nc = base_dir / "model.yaml", base_dir / "model.pt", 1
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML 不存在: {yaml_path}")
    if not weights.exists():
        raise FileNotFoundError(f"权重不存在: {weights}")

    from ultralytics import YOLO
    model = YOLO(_prepare_yaml(yaml_path, nc=nc))
    model.load(str(weights))
    if len(_MODELS) >= _MODEL_LIMIT:
        _MODELS.pop(next(iter(_MODELS)))     # 简单的 FIFO 淘汰
    _MODELS[key] = model
    return model


# ============================================================
# 单帧
# ============================================================
def _mask_to_orig_size(mask, h, w):
    mask = mask.astype(np.uint8) * 255
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def _ellipse_from_mask(mask):
    """从 mask 提取最大轮廓并拟合椭圆，返回 ``(major, minor, center)``。"""
    if mask is None:
        return None, None, None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None
    c = max(contours, key=cv2.contourArea)
    if len(c) < 5:
        return None, None, None
    (cx, cy), (d1, d2), _ = cv2.fitEllipse(c)
    return max(d1, d2), min(d1, d2), (cx, cy)


def _process_frame(image_bgr, model, conf, iou, plr: bool = False):
    """单帧推理，返回 ``(overlay_bgr, info)``。"""
    h, w = image_bgr.shape[:2]
    results = model.predict(image_bgr, conf=conf, iou=iou, imgsz=IMG_SIZE,
                            device=_pick_device(), verbose=False)
    if not results:
        info = {"pupil": {"major": None, "minor": None, "center": None}, "detections": []}
        if plr:
            info["iris"] = {"major": None, "minor": None, "center": None}
            info["mm"] = {"pupil_mm": None, "iris_mm": None, "cornea_mm": CORNEA_MM}
        return image_bgr.copy(), info

    r = results[0]
    pupil_mask = iris_mask = None
    if r.masks is not None and r.boxes is not None:
        classes = r.boxes.cls.cpu().numpy().astype(int)
        masks = r.masks.data.cpu().numpy()
        for i, cls_id in enumerate(classes):
            m = _mask_to_orig_size(masks[i], h, w)
            if cls_id == 0:
                pupil_mask = m if pupil_mask is None else cv2.bitwise_or(pupil_mask, m)
            elif plr and cls_id == 1:
                iris_mask = m if iris_mask is None else cv2.bitwise_or(iris_mask, m)

    major, minor, center = _ellipse_from_mask(pupil_mask)

    overlay = image_bgr.copy()
    if pupil_mask is not None:
        contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
        if center is not None:
            cv2.ellipse(overlay, ((center[0], center[1]), (major, minor), 0), (255, 0, 0), 2)

    info = {
        "pupil": {
            "major": float(major) if major is not None else None,
            "minor": float(minor) if minor is not None else None,
            "center": [float(center[0]), float(center[1])] if center is not None else None,
        },
        "detections": [
            {"cls": r.names[int(b.cls)], "conf": float(b.conf)}
            for b in r.boxes
        ] if r.boxes is not None else [],
    }

    if plr:
        i_major, i_minor, i_center = _ellipse_from_mask(iris_mask)
        if iris_mask is not None:
            contours_i, _ = cv2.findContours(iris_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours_i, -1, (0, 255, 0), 2)
            if i_center is not None:
                cv2.ellipse(overlay, ((i_center[0], i_center[1]), (i_major, i_minor), 0), (0, 255, 255), 1)

        iris_mm = float(i_major) if i_major is not None else None
        pupil_mm = None
        if major is not None and iris_mm is not None and iris_mm > 0:
            # 角膜缘参照：虹膜外接椭圆长轴 ≈ 成人角膜水平直径
            pupil_mm = float(major) / float(iris_mm) * CORNEA_MM

        info["iris"] = {
            "major": float(i_major) if i_major is not None else None,
            "minor": float(i_minor) if i_minor is not None else None,
            "center": [float(i_center[0]), float(i_center[1])] if i_center is not None else None,
        }
        info["mm"] = {"pupil_mm": pupil_mm, "iris_mm": iris_mm, "cornea_mm": CORNEA_MM}

    return overlay, info


def predict(image, model_name=DEFAULT_MODEL, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, plr: bool = False):
    """图片推理，返回 ``(overlay_rgb, info)``。"""
    if isinstance(image, Image.Image):
        image_rgb = np.array(image.convert("RGB"))
    else:
        image_rgb = np.array(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    model = load_model(model_name, plr=plr)
    overlay_bgr, info = _process_frame(image_bgr, model, conf, iou, plr=plr)
    return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB), info


# ============================================================
# 视频
# ============================================================
def _csv_header(plr: bool):
    head = ["frame", "time_sec", "detected", "major", "minor", "cx", "cy", "conf"]
    if plr:
        head += ["iris_detected", "iris_major", "iris_minor", "pupil_mm",
                 "iris_cx", "iris_cy"]
    return head


def _row_from_info(frame_idx, fps, info, plr: bool):
    p = info["pupil"]
    conf = max([d["conf"] for d in info["detections"]], default=0.0)
    row = [
        frame_idx,
        round(frame_idx / fps, 3) if fps else 0.0,
        int(p["major"] is not None),
        round(p["major"], 2) if p["major"] is not None else "",
        round(p["minor"], 2) if p["minor"] is not None else "",
        round(p["center"][0], 1) if p["center"] is not None else "",
        round(p["center"][1], 1) if p["center"] is not None else "",
        round(conf, 3),
    ]
    if plr:
        ir = info.get("iris", {})
        mm = info.get("mm", {})
        row += [
            int(ir.get("major") is not None),
            round(ir["major"], 2) if ir.get("major") is not None else "",
            round(ir["minor"], 2) if ir.get("minor") is not None else "",
            round(mm["pupil_mm"], 2) if mm.get("pupil_mm") is not None else "",
            round(ir["center"][0], 1) if ir.get("center") is not None else "",
            round(ir["center"][1], 1) if ir.get("center") is not None else "",
        ]
    return row


def _ffmpeg_h264(src, dst, timeout=300):
    """mp4v 转 H.264（浏览器可播）。失败时降级为原文件。"""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dst)],
            check=True, capture_output=True, timeout=timeout,
        )
        return str(dst)
    except Exception as e:  # noqa: BLE001 - 转码失败要降级而不是让整条流水线崩掉
        print(f"ffmpeg 转码失败: {e}")
        return str(src)


def _gap_ranges(detected):
    """``detected`` 布尔序列 -> 连续漏检段 ``[(start, end_exclusive), ...]``。"""
    gaps, i, n = [], 0, len(detected)
    while i < n:
        if detected[i]:
            i += 1
            continue
        j = i
        while j < n and not detected[j]:
            j += 1
        gaps.append((i, j))
        i = j
    return gaps


class _FrameReader:
    """顺序读帧，允许回看一帧（漏检段需要后向参考帧）。"""

    def __init__(self, cap, n):
        self.cap = cap
        self.n = n
        self.idx = 0
        self._peek = None

    def read(self):
        if self._peek is not None:
            frame, self._peek = self._peek, None
            self.idx += 1
            return frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.idx >= self.n:
            return None, None
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        self.idx += 1
        return frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def peek(self):
        if self._peek is None and self.idx < self.n:
            ok, frame = self.cap.read()
            if ok:
                self._peek = frame
        return self._peek


def _flow(src_gray, dst_gray):
    return cv2.calcOpticalFlowFarneback(src_gray, dst_gray, None, **_FARNEBACK)


def _fill_and_write(src_path, base_path, rows, size, fps, out_raw):
    """补帧并写出视频，单次流式读取。

    * 检出帧：直接从中间视频取 overlay（与旧版输出一致）；
    * 漏检帧：前后最近检出帧的 mask 经 Farneback 光流 warp，按 1/帧距 加权融合；
    * 峰值内存 ≈ ``FILL_CHUNK`` 帧，与视频总长无关。
    """
    n = len(rows)
    w, h = size
    detected = [int(r[2]) == 1 for r in rows]
    gap_end = {s: e for s, e in _gap_ranges(detected)}

    def mask_at(idx):
        r = rows[idx]
        if not detected[idx] or r[3] == "" or r[4] == "" or r[5] == "" or r[6] == "":
            return None
        return mask_from_ellipse((h, w), r[5], r[6], r[3], r[4])

    cap_src = cv2.VideoCapture(str(src_path))
    cap_base = cv2.VideoCapture(str(base_path))
    writer = cv2.VideoWriter(str(out_raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    reader = _FrameReader(cap_src, n)

    n_filled = 0
    prev_gray = prev_mask = prev_idx = None
    k = 0
    try:
        while k < n:
            if detected[k]:
                frame, gray = reader.read()
                if frame is None:
                    break
                ok, overlay = cap_base.read()
                writer.write(overlay if ok else frame)
                prev_gray, prev_mask, prev_idx = gray, mask_at(k), k
                k += 1
                continue

            end = gap_end.get(k, k + 1)
            seg_start = k
            while seg_start < end:
                seg_end = min(seg_start + FILL_CHUNK, end)
                buffered = []
                for _ in range(seg_start, seg_end):
                    f, g = reader.read()
                    if f is None:
                        break
                    buffered.append((f, g))
                if not buffered:
                    k = seg_end
                    break

                # 只在最后一段预读后向参考帧（下一个检出帧或视频末尾）
                next_gray = next_mask = next_idx = None
                if seg_end == end and end < n:
                    peek = reader.peek()
                    if peek is not None:
                        next_gray = cv2.cvtColor(peek, cv2.COLOR_BGR2GRAY)
                        next_mask, next_idx = mask_at(end), end

                last = None
                for off, (f, g) in enumerate(buffered):
                    idx = seg_start + off
                    acc = np.zeros((h, w), dtype=np.float32)
                    wsum = 0.0
                    if prev_mask is not None and prev_gray is not None:
                        weight = 1.0 / (idx - prev_idx + 1)
                        acc += warp_mask(prev_mask, _flow(prev_gray, g), (h, w)).astype(np.float32) * weight
                        wsum += weight
                    if next_mask is not None and next_gray is not None:
                        weight = 1.0 / (next_idx - idx + 1)
                        acc += warp_mask(next_mask, _flow(next_gray, g), (h, w)).astype(np.float32) * weight
                        wsum += weight

                    filled = ((acc / wsum) > 127.5).astype(np.uint8) * 255 if wsum > 0 else None
                    if filled is not None:
                        n_filled += 1
                    writer.write(draw_mask(f, filled))
                    last = (filled, g, idx)

                k = seg_end
                if last is not None:
                    prev_mask, prev_gray, prev_idx = last
                seg_start = seg_end
    finally:
        cap_src.release()
        cap_base.release()
        writer.release()

    return n_filled


def predict_video_iter(video_path, model_name=DEFAULT_MODEL, conf=CONF_THRESHOLD,
                       iou=IOU_THRESHOLD, max_frames=VIDEO_MAX_FRAMES, progress=None,
                       flow_fill: bool = False, plr: bool = False):
    """逐帧推理生成器。

    * 每 5 帧 yield ``{"partial": True, "rows": [...]}``；
    * 结束 yield ``{"partial": False, "stage", "out_path", "stats", "rows", "csv_path"}``；
      ``flow_fill=True`` 时先 yield ``stage="infer_done"``，补帧完成后再 yield ``stage="flow_done"``。
    """
    model = load_model(model_name, plr=plr)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频: {video_path}")

    fps = VIDEO_FPS or cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_use = min(total, max_frames) if total > 0 and max_frames else (total or max_frames)

    if flow_fill:
        mode_tag = "_plr_flow_seg" if plr else "_flow_seg"
    else:
        mode_tag = "_plr_seg" if plr else "_seg"
    out_path = Path(VIDEO_OUTPUT_DIR) / f"{Path(video_path).stem}_{model_name}{mode_tag}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rows, pupils = [], []
    n_frames = n_det = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if n_use and n_frames >= n_use:
            break

        overlay, info = _process_frame(frame, model, conf, iou, plr=plr)
        writer.write(overlay)                      # overlay 直接落盘，不再驻留内存
        rows.append(_row_from_info(n_frames, fps, info, plr))
        if info["pupil"]["major"] is not None:
            n_det += 1
            pupils.append(info["pupil"])
        n_frames += 1

        if n_frames % 5 == 0:
            if progress is not None:
                try:
                    last = rows[-1]
                    progress(n_frames / max(n_use, 1),
                             desc=f"帧 {n_frames}/{n_use} | major={last[3]} | conf={last[7]}")
                except Exception:  # noqa: BLE001 - 进度条异常不影响推理
                    pass
            yield {"partial": True, "rows": list(rows), "stage": "infer"}

    cap.release()
    writer.release()

    csv_path = str(out_path.with_suffix(".csv"))
    if rows:
        with open(csv_path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(_csv_header(plr))
            wr.writerows(rows)

    # 记录来源视频与参数：日后想单独重建 3D 注视模型时，不必重跑整段推理
    try:
        Path(csv_path).with_suffix(".run.json").write_text(json.dumps({
            "video": str(Path(video_path).resolve()),
            "model": model_name,
            "plr": bool(plr),
            "flow_fill": bool(flow_fill),
            "frames": n_frames,
            "fps": fps,
            "csv": Path(csv_path).name,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:  # noqa: BLE001 - 元数据写不了不该影响主流程
        print(f"写 run.json 失败: {e}")

    tmp_dir = tempfile.mkdtemp(prefix="seg_", dir="/tmp")
    base_h264 = Path(tmp_dir) / (out_path.stem + "_base_h264.mp4")
    base_final = _ffmpeg_h264(out_path, base_h264)

    stats = {
        "frames": n_frames,
        "detected": n_det,
        "detect_rate": (n_det / n_frames) if n_frames else 0.0,
        "fps": fps,
        "out_path": base_final,
        "csv_path": csv_path,
    }
    if pupils:
        stats["pupil_major_mean"] = float(np.mean([p["major"] for p in pupils]))
        stats["pupil_minor_mean"] = float(np.mean([p["minor"] for p in pupils]))

    if not (flow_fill and rows):
        if base_final != str(out_path) and Path(out_path).exists():
            os.remove(out_path)
        yield {"partial": False, "stage": "done", "out_path": base_final,
               "stats": dict(stats), "rows": list(rows), "csv_path": csv_path}
        return

    if progress is not None:
        try:
            progress(1.0, desc="已完成基础分割推理，开始光流模型优化…")
        except Exception:  # noqa: BLE001
            pass
    yield {"partial": False, "stage": "infer_done", "out_path": base_final,
           "stats": dict(stats), "rows": list(rows), "csv_path": csv_path}

    raw_filled = Path(tmp_dir) / (out_path.stem + "_filled_raw.mp4")
    # 补帧阶段优先读未转码的 mp4v 中间文件（少一代 H.264 压缩），没有才退回 base_final
    base_src = str(out_path) if Path(out_path).exists() else base_final
    n_filled = _fill_and_write(video_path, base_src, rows, (w, h), fps, raw_filled)

    final_path = str(out_path.with_suffix("")) + "_h264.mp4"
    try:
        final_path = _ffmpeg_h264(raw_filled, final_path)
        for p in (raw_filled, out_path, Path(base_final)):
            if Path(p).exists() and str(p) != final_path:
                os.remove(p)
    except Exception as e:  # noqa: BLE001
        print(f"ffmpeg 补帧转码失败: {e}")
        final_path = str(raw_filled)

    stats["out_path"] = final_path
    stats["flow_filled"] = n_filled
    if progress is not None:
        try:
            progress(1.0, desc=f"光流模型优化完成，补帧 {n_filled} 帧")
        except Exception:  # noqa: BLE001
            pass
    yield {"partial": False, "stage": "flow_done", "out_path": final_path,
           "stats": dict(stats), "rows": list(rows), "csv_path": csv_path}
