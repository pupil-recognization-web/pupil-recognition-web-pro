"""
独立推理脚本：兼容 best.pt（cv4 用 Conv）与当前 head.py（cv4 用 StripConv）的差异。
不改 head.py，运行时临时替换 Segment.__init__ 的 cv4 构建逻辑。
"""
import sys
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# ============ 配置（路径来自 config，可用环境变量覆盖） ============
from config import CONF_THRESHOLD as CONF, IMG_SIZE, IOU_THRESHOLD as IOU, TRAIN_YAML_DIR, YOLO_ROOT

YOLO_ROOT = str(YOLO_ROOT)
MODEL_YAML = os.environ.get(
    "STANDALONE_YAML", str(Path(TRAIN_YAML_DIR) / "yolo26_bifpn_seg.yaml"))
WEIGHTS = os.environ.get(
    "STANDALONE_WEIGHTS",
    str(Path(YOLO_ROOT) / "runs/segment/yolo26_bifpn_loss_v2/bifpn_run/weights/best.pt"))
# ================================================================

# 1) 优先用本地 ultralytics 源码
sys.path.insert(0, YOLO_ROOT)

import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import Conv
from ultralytics.nn.modules.head import Segment, Segment26


def _patch_segment_cv4():
    """把 Segment / Segment26 的 cv4 构建改回 Conv 版本（与 best.pt 一致）。"""
    def _init(self, nc=80, nm=32, npr=256, reg_max=16, end2end=False, ch=()):
        # 复用 Detect.__init__ 的逻辑（与官方 Segment.__init__ 一致，只改 cv4）
        from ultralytics.nn.modules.head import Detect
        Detect.__init__(self, nc, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr
        from ultralytics.nn.modules.block import Proto, Proto26
        # Segment26 用 Proto26，Segment 用 Proto
        if isinstance(self, Segment26):
            self.proto = Proto26(ch, self.npr, self.nm, nc)
        else:
            self.proto = Proto(ch[0], self.npr, self.nm)
        c4 = max(ch[0] // 4, self.nm)
        # ★ 关键：用 Conv，而不是 StripConv
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1))
            for x in ch
        )
        if end2end:
            import copy
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    Segment.__init__ = _init
    # Segment26 继承 Segment，但它的 __init__ 自己又调了 super()，这里也覆盖
    def _init26(self, nc=80, nm=32, npr=256, reg_max=16, end2end=False, ch=()):
        Segment.__init__(self, nc, nm, npr, reg_max, end2end, ch)
    Segment26.__init__ = _init26


def load_model():
    import yaml
    _patch_segment_cv4()
    with open(MODEL_YAML) as f:
        cfg = yaml.safe_load(f)
    cfg["nc"] = 1
    cfg["scale"] = "n"
    model = YOLO(cfg)
    model.load(WEIGHTS)
    return model


def infer(image_path, model=None):
    if model is None:
        model = load_model()
    results = model.predict(image_path, conf=CONF, iou=IOU, imgsz=IMG_SIZE, verbose=False)
    if not results:
        return None, None
    r = results[0]
    return r, r.plot()  # BGR ndarray


if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else f"{YOLO_ROOT}/pictures/basic_pictures/1.webp"
    print(f"加载模型: {WEIGHTS}")
    model = load_model()
    print("推理:", img_path)
    r, annotated = infer(img_path, model)
    if r is None:
        print("无结果")
        sys.exit(1)
    print("boxes:", len(r.boxes) if r.boxes is not None else 0)
    print("masks:", r.masks is not None)
    out_path = "/tmp/infer_out.jpg"
    cv2.imwrite(out_path, annotated)
    print("结果图:", out_path)
