"""
光流统计量计算 + 去模糊专家推荐（占位）。
统计量算法与 LPW/scripts/generate_blur_flow_stats.py 的 compute_stats 完全一致。
不生成中间 npy 文件，直接内存计算。
"""
import os
import cv2
import numpy as np

STATS_KEYS = [
    'mean_speed', 'max_speed', 'min_speed', 'std_speed', 'median_speed',
    'p25_speed', 'p75_speed', 'p95_speed', 'p99_speed',
    'mean_dx', 'mean_dy', 'std_dx', 'std_dy', 'mean_ang', 'dir_consistency',
    'motion_ratio', 'valid_ratio',
    'mean_div', 'std_div', 'mean_curl', 'std_curl',
    'entropy', 'hv_ratio', 'speed_concentration',
]

MOTION_THRESHOLD = 0.5
VALID_THRESHOLD = 0.01
INTERVAL = 5   # 与训练数据一致：第 i 帧 vs 第 i+5 帧


def compute_flow(gray1, gray2):
    return cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )


def compute_stats(flow):
    """与 generate_blur_flow_stats.py 的 compute_stats 逐行一致。"""
    dx = flow[..., 0].astype(np.float64)
    dy = flow[..., 1].astype(np.float64)
    mag = np.sqrt(dx ** 2 + dy ** 2)
    ang = np.arctan2(dy, dx)

    mean_dx = float(np.mean(dx))
    mean_dy = float(np.mean(dy))
    mean_ang = float(np.arctan2(mean_dy, mean_dx))

    div = np.gradient(dx, axis=1) + np.gradient(dy, axis=0)
    curl = np.gradient(dy, axis=1) - np.gradient(dx, axis=0)

    hist, _ = np.histogram(mag.flatten(), bins=50, density=True)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log(hist))) if len(hist) > 0 else 0.0

    hv_ratio = float(mean_dx / mean_dy) if abs(mean_dy) > 1e-6 else 0.0
    speed_concentration = float(np.sum(mag ** 2) / (np.sum(mag) ** 2 + 1e-8))

    return {
        'mean_speed': float(np.mean(mag)),
        'max_speed': float(np.max(mag)),
        'min_speed': float(np.min(mag)),
        'std_speed': float(np.std(mag)),
        'median_speed': float(np.median(mag)),
        'p25_speed': float(np.percentile(mag, 25)),
        'p75_speed': float(np.percentile(mag, 75)),
        'p95_speed': float(np.percentile(mag, 95)),
        'p99_speed': float(np.percentile(mag, 99)),
        'mean_dx': mean_dx,
        'mean_dy': mean_dy,
        'std_dx': float(np.std(dx)),
        'std_dy': float(np.std(dy)),
        'mean_ang': mean_ang,
        'dir_consistency': float(np.mean(np.cos(ang - mean_ang))),
        'motion_ratio': float(np.mean(mag > MOTION_THRESHOLD)),
        'valid_ratio': float(np.mean(mag > VALID_THRESHOLD)),
        'mean_div': float(np.mean(div)),
        'std_div': float(np.std(div)),
        'mean_curl': float(np.mean(curl)),
        'std_curl': float(np.std(curl)),
        'entropy': entropy,
        'hv_ratio': hv_ratio,
        'speed_concentration': speed_concentration,
    }


def stats_from_video_pair(video_path, frame_i=0, interval=INTERVAL):
    """
    读视频第 frame_i 帧和第 frame_i+interval 帧，算统计量。
    返回 (stats, info_text)。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
    ret1, f1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i + interval)
    ret2, f2 = cap.read()
    cap.release()

    if not ret1 or not ret2:
        raise RuntimeError(f"读帧失败: 需要帧 {frame_i} 和 {frame_i + interval}，总帧数 {total}")

    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    flow = compute_flow(g1, g2)
    stats = compute_stats(flow)
    info = f"视频: {os.path.basename(str(video_path))}\n帧对: {frame_i} → {frame_i + interval}（间隔 {interval}）"
    return stats, flow, info


def stats_from_two_images(img1_bgr, img2_bgr):
    """单图两帧模式（间隔 1，和训练数据间隔不同，仅作兜底）。"""
    g1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)
    flow = compute_flow(g1, g2)
    stats = compute_stats(flow)
    info = "两图模式（间隔 1，与训练数据间隔 5 不同，统计量分布可能偏）"
    return stats, flow, info


def stats_to_table(stats):
    return [[k, round(float(stats[k]), 6)] for k in STATS_KEYS]


def recommend_expert(stats):
    """
    根据光流统计量推荐去模糊专家。
    当前为基于阈值的规则选择，后续可接入 StatsSeqTransformer 预测清晰统计量后再选。
    """
    speed = stats['mean_speed']
    cons = stats['dir_consistency']
    entropy = stats['entropy']
    if speed > 5.0 and cons < 0.3:
        return "general_v2", f"mean_speed={speed:.2f} 高、dir_consistency={cons:.2f} 低 → 运动剧烈方向乱，强去模糊"
    if speed < 0.5:
        return "general_v2", f"mean_speed={speed:.2f} 低 → 运动平缓，通用专家"
    if entropy > 4.0:
        return "general_v2", f"entropy={entropy:.2f} 高 → 运动分布散，通用专家"
    return "general_v2", f"mean_speed={speed:.2f}, dir_consistency={cons:.2f} → 默认 general_v2"


def flow_visualization(flow, step=16):
    """画光流箭头图（降采样），返回 RGB 图给 gr.Image"""
    h, w = flow.shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return vis
