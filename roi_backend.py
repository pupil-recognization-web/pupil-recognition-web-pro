"""
ROI 裁剪：用 MediaPipe FaceLandmarker（Tasks API，新版）检测眼睛区域，裁剪 ROI。
"""
import os
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    _MP_AVAILABLE = True
except Exception:
    _MP_AVAILABLE = False
    mp = None

from config import MEDIAPIPE_MODEL

_MODEL_PATH = str(MEDIAPIPE_MODEL)

# MediaPipe 眼睛关键点索引（478 点，0-467 面部，468-477 虹膜）
LEFT_EYE_IDX = [33, 133, 157, 158, 159, 160, 161, 173]
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398, 466]

_LANDMARKER = None


def _get_landmarker(running_mode="IMAGE"):
    """懒加载 FaceLandmarker，按 running_mode 缓存。"""
    global _LANDMARKER
    if not _MP_AVAILABLE:
        return None
    if not os.path.exists(_MODEL_PATH):
        print(f"[roi] 模型不存在: {_MODEL_PATH}")
        return None
    if _LANDMARKER is not None:
        return _LANDMARKER
    try:
        base = python.BaseOptions(model_asset_path=_MODEL_PATH)
        if running_mode == "VIDEO":
            rm = vision.RunningMode.VIDEO
        else:
            rm = vision.RunningMode.IMAGE
        options = vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=rm,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _LANDMARKER = vision.FaceLandmarker.create_from_options(options)
        print(f"[roi] FaceLandmarker 加载完成（{running_mode}）")
        return _LANDMARKER
    except Exception as e:
        print(f"[roi] FaceLandmarker 加载失败: {e}")
        return None


def _landmarks_to_roi(landmarks, w, h, idx_list, padding):
    """关键点列表 → 外接矩形 ROI（加 padding）。"""
    xs = [landmarks[i].x * w for i in idx_list]
    ys = [landmarks[i].y * h for i in idx_list]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    bw, bh = x2 - x1, y2 - y1
    pad_w, pad_h = bw * padding, bh * padding
    x1 = max(0, int(x1 - pad_w))
    y1 = max(0, int(y1 - pad_h))
    x2 = min(w, int(x2 + pad_w))
    y2 = min(h, int(y2 + pad_h))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def detect_eye_roi_image(frame_bgr, padding=0.3, which="both"):
    """
    单帧图像检测眼睛 ROI（IMAGE 模式）。
    which: "left" / "right" / "both"
    返回 [(x1, y1, x2, y2), ...]。
    """
    lm = _get_landmarker("IMAGE")
    if lm is None:
        return []
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    try:
        result = lm.detect(mp_image)
    except Exception as e:
        print(f"[roi] detect 失败: {e}")
        return []
    if not result.face_landmarks:
        return []
    landmarks = result.face_landmarks[0]
    rois = []
    if which in ("left", "both"):
        roi = _landmarks_to_roi(landmarks, w, h, LEFT_EYE_IDX, padding)
        if roi:
            rois.append(roi)
    if which in ("right", "both"):
        roi = _landmarks_to_roi(landmarks, w, h, RIGHT_EYE_IDX, padding)
        if roi:
            rois.append(roi)
    return rois


def crop_roi(frame_bgr, roi):
    """裁剪 ROI，返回 (roi_img, (x1, y1))。"""
    x1, y1, x2, y2 = roi
    return frame_bgr[y1:y2, x1:x2].copy(), (x1, y1)


def draw_roi(frame_bgr, rois):
    """在原图上画 ROI 框（调试用）。"""
    out = frame_bgr.copy()
    for (x1, y1, x2, y2) in rois:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out


def close():
    """释放 FaceLandmarker。"""
    global _LANDMARKER
    if _LANDMARKER is not None:
        try:
            _LANDMARKER.close()
        except Exception:
            pass
        _LANDMARKER = None


if __name__ == "__main__":
    # 最小测试：读一张图，检测 ROI
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_face.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print(f"读不到图片: {img_path}")
        sys.exit(1)
    rois = detect_eye_roi_image(img)
    print(f"ROI: {rois}")
    if rois:
        out = draw_roi(img, rois)
        cv2.imwrite("/tmp/roi_test.jpg", out)
        print("保存: /tmp/roi_test.jpg")


def detect_eye_roi_video(video_path, which="both", padding=0.3, fallback="prev",
                         max_frames=2000, progress=None):
    """
    逐帧检测 ROI，输出：
      rois_per_frame: list[dict] 每帧 {"frame": int, "left": (x1,y1,x2,y2)|None, "right": ...}
      stats: dict
    fallback: "prev" 沿用上一帧 / "full" 不裁剪（全图）/ "skip" 跳过该帧
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_use = min(total, max_frames) if total > 0 and max_frames else total

    rois_per_frame = []
    prev_left, prev_right = None, None
    n_detect_left = n_detect_right = 0
    n_fallback_left = n_fallback_right = 0
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret or (n_use and i >= n_use):
            break
        h, w = frame.shape[:2]
        rois = detect_eye_roi_image(frame, padding=padding, which=which)
        left_roi = right_roi = None
        if which in ("left", "both") and len(rois) >= (1 if which == "left" else 1):
            idx = 0
            if which == "both":
                idx = 0
            left_roi = rois[idx] if idx < len(rois) else None
        if which in ("right", "both") and len(rois) >= (1 if which == "right" else 2):
            idx = 1 if which == "both" else 0
            right_roi = rois[idx] if idx < len(rois) else None

        if left_roi is not None:
            n_detect_left += 1
            prev_left = left_roi
        else:
            if fallback == "prev" and prev_left is not None:
                left_roi = prev_left
                n_fallback_left += 1
            elif fallback == "full":
                left_roi = (0, 0, w, h)
                n_fallback_left += 1
        if right_roi is not None:
            n_detect_right += 1
            prev_right = right_roi
        else:
            if fallback == "prev" and prev_right is not None:
                right_roi = prev_right
                n_fallback_right += 1
            elif fallback == "full":
                right_roi = (0, 0, w, h)
                n_fallback_right += 1

        rois_per_frame.append({"frame": i, "left": left_roi, "right": right_roi})
        i += 1
        if progress is not None and i % 10 == 0:
            try:
                progress(i / max(n_use, 1), desc=f"ROI {i}/{n_use}")
            except Exception:
                pass
    cap.release()

    stats = {
        "frames": len(rois_per_frame),
        "detect_left": n_detect_left,
        "detect_right": n_detect_right,
        "fallback_left": n_fallback_left,
        "fallback_right": n_fallback_right,
    }
    return rois_per_frame, stats


def write_roi_csv(rois_per_frame, path):
    import csv
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "eye", "x1", "y1", "x2", "y2", "valid"])
        for r in rois_per_frame:
            for eye in ("left", "right"):
                roi = r.get(eye)
                if roi is None:
                    wr.writerow([r["frame"], eye, "", "", "", "", 0])
                else:
                    x1, y1, x2, y2 = roi
                    wr.writerow([r["frame"], eye, x1, y1, x2, y2, 1])


def crop_video_with_roi(video_path, rois_per_frame, eye="left", out_path=None,
                        debug_path=None, max_frames=2000, progress=None):
    """
    按 rois_per_frame 裁剪视频，输出裁剪视频（+可选画框图）。
    eye: "left" / "right"
    返回 (out_path, debug_path, stats)
    """
    import subprocess
    from pathlib import Path

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_use = min(total, max_frames) if total > 0 and max_frames else total

    # 先扫一遍确定输出尺寸（取所有 ROI 的最大宽高）
    max_w = max_h = 0
    for r in rois_per_frame[:n_use]:
        roi = r.get(eye)
        if roi is None:
            continue
        x1, y1, x2, y2 = roi
        max_w = max(max_w, x2 - x1)
        max_h = max(max_h, y2 - y1)
    if max_w <= 0 or max_h <= 0:
        cap.release()
        raise RuntimeError(f"没有有效 ROI（eye={eye}）")

    stem = Path(video_path).stem
    out_path = out_path or f"/tmp/{stem}_roi_{eye}.mp4"
    raw = out_path.replace(".mp4", "_raw.mp4")
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (max_w, max_h))
    dbg_writer = None
    if debug_path:
        raw_dbg = debug_path.replace(".mp4", "_raw.mp4")
        dbg_writer = cv2.VideoWriter(raw_dbg, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret or (n_use and i >= n_use):
            break
        roi = rois_per_frame[i].get(eye) if i < len(rois_per_frame) else None
        if roi is not None:
            x1, y1, x2, y2 = roi
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                crop = cv2.resize(frame, (max_w, max_h))
            elif crop.shape[1] != max_w or crop.shape[0] != max_h:
                crop = cv2.resize(crop, (max_w, max_h))
            writer.write(crop)
            if dbg_writer is not None:
                dbg_writer.write(draw_roi(frame, [roi]))
        i += 1
        if progress is not None and i % 10 == 0:
            try:
                progress(i / max(n_use, 1), desc=f"裁剪 {i}/{n_use}")
            except Exception:
                pass
    cap.release()
    writer.release()
    if dbg_writer is not None:
        dbg_writer.release()
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw_dbg,
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", debug_path], check=True)
            os.remove(raw_dbg)
        except Exception:
            debug_path = raw_dbg
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path], check=True)
        os.remove(raw)
    except Exception:
        out_path = raw
    stats = {"frames": i, "eye": eye, "out_w": max_w, "out_h": max_h}
    return out_path, debug_path, stats
