"""集中配置：全项目唯一的路径与参数来源。

设计原则
--------
1. **不硬编码绝对路径**。任何模块都从本文件取路径，不再出现
   ``/home/qingque/qingque_yolo/...`` 这类字面量。
2. **可用环境变量整体搬迁**。换机器 / 换目录时只改环境变量，不动代码：

   ========================  ==================================================
   ``PUPIL_YOLO_ROOT``       ultralytics、数据集、训练脚本所在的工程根
   ``PUPIL_VOGEO_ROOT``      VOGeo-Gaze 仓库根
   ``PUPIL_OPENEDS_DIR``     OpenEDS 视频目录
   ``PUPIL_LPW_DIR``         LPW 视频目录
   ``PUPIL_DEBLUR_DIR``      去模糊专家权重根（默认 ``$PUPIL_YOLO_ROOT/LPW/models``）
   ``PUPIL_MODELS_DIR``      本项目的模型目录（默认 ``<项目>/models``）
   ``PUPIL_OUTPUT_DIR``      产出目录（默认 ``<项目>/outputs``）
   ``PUPIL_PORT``            Web 端口（默认 8080）
   ``PUPIL_DEVICE``          VOGeo-Gaze 推理设备（默认 cuda）
   ========================  ==================================================

3. **单一数值定义**。视频帧数上限、界面滑块范围、置信度阈值等只在此定义，
   UI 与后端读同一份，避免 ``config=600`` 而界面能拉到 2000 的不一致。
"""
from __future__ import annotations

import os
from pathlib import Path

# ============================================================
# 路径
# ============================================================
BASE_DIR = Path(os.environ.get("PUPIL_PROJECT_DIR") or Path(__file__).resolve().parent)

MODELS_DIR = Path(os.environ.get("PUPIL_MODELS_DIR") or BASE_DIR / "models")
EXAMPLES_DIR = BASE_DIR / "examples"
OUTPUT_DIR = Path(os.environ.get("PUPIL_OUTPUT_DIR") or BASE_DIR / "outputs")

#: 上游工程根（ultralytics 源码、训练脚本、数据集都挂在它下面）
YOLO_ROOT = Path(os.environ.get("PUPIL_YOLO_ROOT") or "/home/qingque/qingque_yolo")
#: VOGeo-Gaze 仓库根
VOGE0_ROOT = Path(os.environ.get("PUPIL_VOGEO_ROOT") or YOLO_ROOT / "VOGeo-Gaze")
#: 去模糊专家权重根（每个专家一个子目录，内含 best.pth）
DEBLUR_MODEL_DIR = Path(os.environ.get("PUPIL_DEBLUR_DIR") or YOLO_ROOT / "LPW" / "models")

#: 数据集名 -> 视频目录
DATASET_DIRS = {
    "OpenEDS": Path(os.environ.get("PUPIL_OPENEDS_DIR") or YOLO_ROOT / "openEDS/openEDS/videos"),
    "LPW": Path(os.environ.get("PUPIL_LPW_DIR") or YOLO_ROOT / "LPW/videos"),
}

# --- 产出子目录（全部相对 OUTPUT_DIR，便于整体迁移）---
VIDEO_OUTPUT_DIR = OUTPUT_DIR / "segmentation_results"   # 逐帧 CSV + 分割视频
PLOT_DIR = OUTPUT_DIR / "plots"                          # 全部分析图
PDF_DIR = OUTPUT_DIR / "pdf_reports"
STIMULUS_DIR = OUTPUT_DIR / "stimulus_marks"             # 刺激标定 JSON
DEBLUR_VIDEO_DIR = OUTPUT_DIR / "deblur_videos"
MOTION_OUT_DIR = OUTPUT_DIR / "motion"
VOGE0_OUTPUT_DIR = OUTPUT_DIR / "vogeo_results"
ROI_OUTPUT_DIR = OUTPUT_DIR / "roi_results"

CACHE_DIR = BASE_DIR / "cache"
RAW_VIDEO_DIR = CACHE_DIR / "raw_videos"                 # 从数据集拷来的原视频副本
TRAIN_LOG_DIR = BASE_DIR / "train_logs"

#: 训练脚本与数据集配置（Web 训练 Tab 调它们）
TRAIN_SCRIPTS_DIR = YOLO_ROOT / "scripts" / "yolo_training"
TRAIN_YAML_DIR = YOLO_ROOT / "ultralytics" / "cfg" / "models" / "26"
TRAIN_PROJECT_DIR = YOLO_ROOT / "runs" / "segment"
DATASET_YAMLS = {"OpenEDS pupil": YOLO_ROOT / "openEDS/yolo_dataset/dataset.yaml"}

#: StatsSeqTransformer（运动光流预定义模型）训练相关
STATS_TRAIN_SCRIPT = YOLO_ROOT / "LPW/scripts/flow_stats_predictor/train_stats_regression_transformer.py"
STATS_MODEL_DIR = YOLO_ROOT / "LPW/models"
STATS_CHECKPOINT = STATS_MODEL_DIR / "stats_regression_tf_checkpoint.pth"
STATS_CACHE = STATS_MODEL_DIR / "stats_regression_tf_cache.pkl"

#: MediaPipe FaceLandmarker 权重
MEDIAPIPE_MODEL = MODELS_DIR / "mediapipe" / "face_landmarker.task"

# ============================================================
# 推理参数
# ============================================================
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMG_SIZE = 640

VIDEO_FPS = None                      # None = 用原视频帧率
VIDEO_MAX_FRAMES = 600                # 单次处理帧数默认值
VIDEO_MAX_FRAMES_MIN = 10             # 界面滑块下限
VIDEO_MAX_FRAMES_MAX = 2000           # 界面滑块上限（与后端共用，避免不一致）

#: 角膜缘参照：成人角膜水平直径中位数，用于 px -> mm 换算
CORNEA_MM = 11.5
#: 补帧用的光流帧间隔（与训练数据一致）
FLOW_INTERVAL = 5

# ============================================================
# 分析参数
# ============================================================
#: PLR 分析模式：``v2`` = 刺激窗 + 基线窗 + 无响应判定（默认）；
#: ``legacy`` = 旧版“全序列 argmin”，仅用于复现历史数字。
PLR_MODE = os.environ.get("PUPIL_PLR_MODE", "v2").strip().lower()

PLR_SMOOTH_WINDOW = 5                 # legacy 模式的中值平滑窗口（帧）
#: v2：平滑按“时间”而非帧数（秒）——10 fps 与 30 fps 结果可比
PLR_SMOOTH_SEC = 0.2
#: v2：基线窗（相对刺激时刻，秒）
PLR_BASE_WINDOW = (-0.6, 0.0)
#: v2：收缩窗（相对刺激时刻，秒）。对光反射在 2 s 内基本完成
PLR_RESPONSE_WINDOW = (0.0, 2.0)
#: v2：刺激后收缩幅度低于该百分比视为“未引出 / 无响应”，不再硬算参数
PLR_MIN_RESPONSE_PCT = 5.0
#: v2：潜伏期判据——较基线下降该比例即认为开始收缩
PLR_LATENCY_DROP = 0.02
#: v2：NPi 的速率分项锚点（mm/s）。约 2 mm/s 视为"正常收缩速度"满分
PLR_MCV_REF_MM_S = 2.0
PLR_MIN_VALID_POINTS = 10             # 少于该点数不提取 PLR
QUALITY_DETECT_RATE_GOOD = 90.0       # 检出率 >= 该值：Good
QUALITY_DETECT_RATE_FAIR = 70.0       # 检出率 >= 该值：Fair，否则 Low
#: 低于该检出率时，PLR / 判读结论按“不可靠”处理并显式提示
RELIABLE_DETECT_RATE = 30.0

# ============================================================
# 模型注册表
# ============================================================
MODEL_REGISTRY = {
    "baseline": {"label": "综合最优（BiFPN + 进阶损失）", "dir": MODELS_DIR / "baseline"},
    "mac": {"label": "边缘模糊（+MAC）", "dir": MODELS_DIR / "mac"},
    "strip": {"label": "遮挡（+Strip，实验性）", "dir": MODELS_DIR / "strip"},
}
DEFAULT_MODEL = "baseline"

TRAIN_MODEL_YAMLS = [
    "yolo26_bifpn_seg.yaml",
    "yolo26_bifpn_mac_seg.yaml",
    "yolo26_bifpn_strip_seg.yaml",
]

# ============================================================
# Web 服务
# ============================================================
SERVER_HOST = os.environ.get("PUPIL_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("PUPIL_PORT", "8080"))
#: VOGeo-Gaze 设备：空字符串表示交给它自己判断（有 CUDA 用 cuda，否则 cpu）。
#: 旧版写死 "cuda"，在没有 GPU / 驱动未装的机器上会直接失败，且界面无提示。
#: YOLO 推理设备：空字符串 = 自动（有 CUDA 用 GPU）；可设 "cpu" 在 GPU 被占用时降级
YOLO_DEVICE = os.environ.get("PUPIL_YOLO_DEVICE", "").strip()
VOGE0_DEVICE = os.environ.get("PUPIL_DEVICE", "").strip()
#: VOGeo-Gaze 超时（秒）；CPU 上 600 帧约需 1–2 分钟，默认给足
VOGE0_TIMEOUT = int(os.environ.get("PUPIL_VOGEO_TIMEOUT", "1800"))


def ensure_dirs() -> None:
    """建好运行期需要的目录（幂等）。"""
    for d in (
        OUTPUT_DIR, VIDEO_OUTPUT_DIR, PLOT_DIR, PDF_DIR, STIMULUS_DIR,
        DEBLUR_VIDEO_DIR, MOTION_OUT_DIR, VOGE0_OUTPUT_DIR, ROI_OUTPUT_DIR,
        CACHE_DIR, RAW_VIDEO_DIR, TRAIN_LOG_DIR,
    ):
        Path(d).mkdir(parents=True, exist_ok=True)


def dataset_video_dir(dataset: str):
    """数据集名 -> Path；未知数据集返回 None。"""
    d = DATASET_DIRS.get(dataset)
    return Path(d) if d else None


def allowed_paths():
    """Gradio 需要显式放行的静态路径。"""
    return [
        str(RAW_VIDEO_DIR), str(DEBLUR_VIDEO_DIR), str(PLOT_DIR),
        str(VOGE0_OUTPUT_DIR), str(ROI_OUTPUT_DIR), str(PDF_DIR),
    ]


ensure_dirs()
