"""
对比三份 results.csv 的训练曲线，按 epoch 对齐。
只保留最关键指标，最后拼成一张总图。
用法: python plot_compare.py <csv1> <csv2> <csv3> [输出目录]
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

from config import OUTPUT_DIR

def load_clean(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["epoch"])
    df["epoch"] = df["epoch"].astype(int)
    df = df.sort_values(["epoch", "time"]).drop_duplicates(subset="epoch", keep="last")
    df = df.sort_values("epoch").reset_index(drop=True)
    return df

def main():
    if len(sys.argv) < 4:
        print("用法: python plot_compare.py <csv1> <csv2> <csv3> [输出目录]")
        sys.exit(1)
    paths = sys.argv[1:4]
    out_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = ["baseline", "MAC", "Strip"]
    dfs = [load_clean(p) for p in paths]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
    })

    # 1) Loss：train/val seg_loss
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col, title in [
        (axes[0], "train/seg_loss", "Train Seg Loss"),
        (axes[1], "val/seg_loss", "Val Seg Loss"),
    ]:
        for df, name, c in zip(dfs, names, colors):
            if col in df.columns:
                ax.plot(df["epoch"], df[col], lw=1.5, label=name, color=c)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    p1 = out_dir / "compare_loss.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"✅ {p1}")

    # 2) mAP50-95 box/mask
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col, title in [
        (axes[0], "metrics/mAP50-95(B)", "Box mAP50-95"),
        (axes[1], "metrics/mAP50-95(M)", "Mask mAP50-95"),
    ]:
        for df, name, c in zip(dfs, names, colors):
            if col in df.columns:
                ax.plot(df["epoch"], df[col], lw=1.5, label=name, color=c)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 1.02)
        ax.legend()
    fig.tight_layout()
    p2 = out_dir / "compare_mAP.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"✅ {p2}")

    # 3) 前 28 epoch 公平对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col, title in [
        (axes[0], "metrics/mAP50-95(B)", "Box mAP50-95 (first 28 ep)"),
        (axes[1], "metrics/mAP50-95(M)", "Mask mAP50-95 (first 28 ep)"),
    ]:
        for df, name, c in zip(dfs, names, colors):
            if col in df.columns:
                d = df[df["epoch"] <= 28]
                ax.plot(d["epoch"], d[col], lw=1.5, label=name, color=c)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 1.02)
        ax.legend()
    fig.tight_layout()
    p3 = out_dir / "compare_first28_mAP.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"✅ {p3}")

    # 4) 三张拼成一张总图
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    for ax, p, title in [
        (axes[0], p1, "Seg Loss"),
        (axes[1], p2, "mAP50-95"),
        (axes[2], p3, "mAP50-95 (first 28 ep)"),
    ]:
        ax.imshow(mpimg.imread(p))
        ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=6)
    fig.tight_layout()
    p_combined = out_dir / "compare_combined.png"
    fig.savefig(p_combined, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ {p_combined}")

    # 5) ep28 指标表
    print("\n=== epoch 28 指标对比 ===")
    rows = []
    for df, name in zip(dfs, names):
        r = df[df["epoch"] == 28]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append({
            "model": name,
            "Box mAP50-95": round(r.get("metrics/mAP50-95(B)", float("nan")), 4),
            "Mask mAP50-95": round(r.get("metrics/mAP50-95(M)", float("nan")), 4),
            "Val Seg Loss": round(r.get("val/seg_loss", float("nan")), 4),
        })
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    main()
