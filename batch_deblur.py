"""批量对图片目录去模糊（命令行工具，不参与 Web 流程）。

用法::

    python batch_deblur.py --input <图片目录> [--output <输出目录>] [--expert general_v2]
"""
import argparse
import os
from pathlib import Path

import cv2
from tqdm import tqdm

import deblur_backend as db
from config import YOLO_ROOT

DEFAULT_INPUT = Path(YOLO_ROOT) / "openEDS/yolo_dataset_real/images/train"
DEFAULT_OUTPUT = Path(YOLO_ROOT) / "openEDS/yolo_dataset_real/images/train_deblur"
EXPERT_NAME = "general_v2"

def main():
    ap = argparse.ArgumentParser(description="批量去模糊")
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--expert", default=EXPERT_NAME)
    args = ap.parse_args()

    INPUT_DIR = Path(args.input)
    OUTPUT_DIR = Path(args.output)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    expert = args.expert

    exts = ("*.png", "*.jpg", "*.jpeg")
    img_paths = []
    for ext in exts:
        img_paths.extend(list(INPUT_DIR.glob(ext)))
    
    print(f"找到 {len(img_paths)} 张图片，准备使用专家模型 [{expert}] 进行去模糊...")
    
    for img_path in tqdm(img_paths, desc="Deblurring"):
        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"读取失败: {img_path}")
            continue
            
        # 调用后端模型推理
        try:
            # 假设你的 deblur_frame 函数签名是 (image_bgr, expert)
            out_img = db.deblur_frame(img, expert=expert)
            
            # 保存到输出目录（保持原文件名）
            out_path = OUTPUT_DIR / img_path.name
            cv2.imwrite(str(out_path), out_img)
        except Exception as e:
            print(f"处理 {img_path.name} 时出错: {e}")

    print(f"\n✅ 批量去模糊完成！结果已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
