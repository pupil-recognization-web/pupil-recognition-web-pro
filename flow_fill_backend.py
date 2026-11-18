"""光流补帧的**唯一**实现。

从 CSV 的椭圆参数重建 mask，对 ``detected == 0`` 的连续段用前后检出帧的 mask
做 Farneback 光流 warp，再按 1/帧距 加权融合。

本模块原先在 ``inference.py``、``fill_from_csv.py``、``fill_clear_masks.py``
里各有一份逐行重复的拷贝，改一处另两处就漂；现在只有这里一份，其余模块 import。
"""
import cv2
import numpy as np


def mask_from_ellipse(shape, cx, cy, major, minor):
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if cx is None or cy is None or major is None or minor is None:
        return None
    cv2.ellipse(mask, ((cx, cy), (major, minor), 0), 255, -1)
    return mask


def compute_flow(src_gray, dst_gray):
    return cv2.calcOpticalFlowFarneback(
        src_gray, dst_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )


def warp_mask(mask_src, flow, dst_shape):
    h, w = dst_shape
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (gx + flow[..., 0]).astype(np.float32)
    map_y = (gy + flow[..., 1]).astype(np.float32)
    return cv2.remap(mask_src, map_x, map_y, cv2.INTER_NEAREST)


def fill_masks(frames, masks, detected, grays=None):
    """对漏检段做光流补全，返回与 ``masks`` 同长的 mask 列表。

    ``frames`` 可以是 BGR 帧列表；若已预先算好灰度图，传 ``grays`` 可省掉重复
    的 ``cvtColor``（补帧只用到灰度）。
    """
    n = len(frames)
    filled = list(masks)

    def gray(idx):
        if idx is None:
            return None
        return grays[idx] if grays is not None else cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)

    i = 0
    while i < n:
        if detected[i] == 1:
            i += 1
            continue
        j = i
        while j < n and detected[j] == 0:
            j += 1
        prev_idx = next((k for k in range(i - 1, -1, -1) if detected[k] == 1 and masks[k] is not None), None)
        next_idx = next((k for k in range(j, n) if detected[k] == 1 and masks[k] is not None), None)
        if prev_idx is None and next_idx is None:
            i = j
            continue
        prev_gray, next_gray = gray(prev_idx), gray(next_idx)
        for k in range(i, j):
            cur_gray = gray(k)
            h, w = cur_gray.shape
            acc = np.zeros((h, w), dtype=np.float32)
            wsum = 0.0
            if prev_idx is not None:
                acc += warp_mask(masks[prev_idx], compute_flow(prev_gray, cur_gray), (h, w)).astype(np.float32) / (k - prev_idx + 1)
                wsum += 1.0 / (k - prev_idx + 1)
            if next_idx is not None:
                acc += warp_mask(masks[next_idx], compute_flow(next_gray, cur_gray), (h, w)).astype(np.float32) / (next_idx - k + 1)
                wsum += 1.0 / (next_idx - k + 1)
            if wsum > 0:
                filled[k] = ((acc / wsum) > 127.5).astype(np.uint8) * 255
        i = j
    return filled


def draw(frame, mask):
    out = frame.copy()
    if mask is not None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 0, 255), 2)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if len(c) >= 5:
                cv2.ellipse(out, cv2.fitEllipse(c), (255, 0, 0), 2)
    return out
