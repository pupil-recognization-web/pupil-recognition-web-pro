"""
DeblurUNetV2 / DeblurUNet 推理封装，结构和 deblur_infer.py 完全一致。
"""
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from config import DEBLUR_MODEL_DIR

MODEL_BASE = Path(DEBLUR_MODEL_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MODELS = {}


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(cat))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class EPA(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.dw7 = nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=False)
        self.fuse = nn.Conv2d(channels * 3, channels, 1, bias=False)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y1 = self.dw3(x)
        y2 = self.dw5(x)
        y3 = self.dw7(x)
        y = self.fuse(torch.cat([y1, y2, y3], dim=1))
        return x + x * self.se(y)


class DeblurUNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self._block(3, 32)
        self.enc2 = self._block(32, 64)
        self.enc3 = self._block(64, 128)
        self.enc4 = self._block(128, 256)
        self.bottleneck = self._block(256, 256)
        self.epa_bottleneck = EPA(256)
        self.up4 = self._up(256, 256)
        self.cbam4 = CBAM(128)
        self.dec4 = self._block(256 + 256, 128)
        self.up3 = self._up(128, 128)
        self.cbam3 = CBAM(64)
        self.dec3 = self._block(128 + 128, 64)
        self.up2 = self._up(64, 64)
        self.cbam2 = CBAM(32)
        self.dec2 = self._block(64 + 64, 32)
        self.up1 = self._up(32, 32)
        self.cbam1 = CBAM(32)
        self.dec1 = self._block(32 + 32, 32)
        self.final = nn.Conv2d(32, 3, 3, padding=1)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def _up(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        e4 = self.enc4(nn.MaxPool2d(2)(e3))
        b = self.bottleneck(nn.MaxPool2d(2)(e4))
        b = self.epa_bottleneck(b)
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d4 = self.cbam4(d4)
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d3 = self.cbam3(d3)
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d2 = self.cbam2(d2)
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        d1 = self.cbam1(d1)
        residual = self.final(d1)
        return torch.clamp(x + residual, 0, 1)


class DeblurUNet(nn.Module):
    def __init__(self, chs=(16, 32, 64, 128)):
        super().__init__()
        c1, c2, c3, c4 = chs
        self.enc1 = self._block(3, c1)
        self.enc2 = self._block(c1, c2)
        self.enc3 = self._block(c2, c3)
        self.enc4 = self._block(c3, c4)
        self.bottleneck = self._block(c4, c4)
        self.up4 = self._up(c4, c4)
        self.dec4 = self._block(c4 + c4, c3)
        self.up3 = self._up(c3, c3)
        self.dec3 = self._block(c3 + c3, c2)
        self.up2 = self._up(c2, c2)
        self.dec2 = self._block(c2 + c2, c1)
        self.up1 = self._up(c1, c1)
        self.dec1 = self._block(c1 + c1, c1)
        self.final = nn.Conv2d(c1, 3, 3, padding=1)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def _up(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        e4 = self.enc4(nn.MaxPool2d(2)(e3))
        b = self.bottleneck(nn.MaxPool2d(2)(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


def list_experts():
    names = []
    for d in sorted(MODEL_BASE.iterdir()):
        if d.is_dir() and (d / "best.pth").exists():
            names.append(d.name)
    names.sort(key=lambda n: (0 if n.endswith("_v2") else 1, n))
    return names


def load_deblur(expert: str = "general_v2"):
    if expert in _MODELS:
        return _MODELS[expert]
    path = MODEL_BASE / expert / "best.pth"
    if not path.exists():
        raise FileNotFoundError(f"专家权重不存在: {path}")
    sd = torch.load(path, map_location=DEVICE)
    has_epa = any(k.startswith("epa_bottleneck.") for k in sd.keys())
    has_cbam = any(k.startswith("cbam1.") for k in sd.keys())
    if has_epa or has_cbam:
        model = DeblurUNetV2().to(DEVICE)
    else:
        c1 = sd["enc1.0.weight"].shape[0]
        chs = (16, 32, 64, 128) if c1 == 16 else (32, 64, 128, 256)
        model = DeblurUNet(chs=chs).to(DEVICE)
    model.load_state_dict(sd)
    model.eval()
    _MODELS[expert] = model
    return model


def deblur_image(image_bgr: np.ndarray, expert: str = "general_v2") -> np.ndarray:
    model = load_deblur(expert)
    img = cv2.resize(image_bgr, (640, 480))
    x = img.astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        y = model(x)
    y = y.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (np.clip(y, 0, 1) * 255.0).astype(np.uint8)


def deblur_frame(image_bgr: np.ndarray, expert: str = "general_v2") -> np.ndarray:
    h, w = image_bgr.shape[:2]
    out = deblur_image(image_bgr, expert)
    if out.shape[:2] != (h, w):
        out = cv2.resize(out, (w, h))
    return out
