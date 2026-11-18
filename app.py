import gradio as gr
import applog
import cv2
import sys
import csv
import json
import numpy as np
import os
import re
import shutil
from pathlib import Path
from PIL import Image

from config import (
    BASE_DIR, CONF_THRESHOLD, DATASET_DIRS, DATASET_YAMLS, DEBLUR_VIDEO_DIR, DEFAULT_MODEL,
    EXAMPLES_DIR, IOU_THRESHOLD, MODEL_REGISTRY, OUTPUT_DIR, PLOT_DIR, RAW_VIDEO_DIR,
    SERVER_HOST, SERVER_PORT, STIMULUS_DIR, TRAIN_MODEL_YAMLS, VIDEO_MAX_FRAMES,
    VIDEO_MAX_FRAMES_MAX, VIDEO_MAX_FRAMES_MIN, VIDEO_OUTPUT_DIR, allowed_paths,
)
from inference import predict, predict_video_iter
import analysis_backend as ab
import deblur_backend as db
import motion_backend as mb
import train_backend as tb
import flow_stats_backend as fsb
import roi_backend as rb
import pdf_export as pe
import model_info
import store
import services

model_choices = [(v["label"], k) for k, v in MODEL_REGISTRY.items()]

DEBLUR_DIR = str(DEBLUR_VIDEO_DIR)
COPY_DIR = str(RAW_VIDEO_DIR)
os.makedirs(COPY_DIR, exist_ok=True)


def list_deblur_videos():
    p = Path(DEBLUR_DIR)
    if not p.exists():
        return []
    files = [f.name for f in p.glob("*.mp4")]
    files.sort(key=lambda n: (0 if "general_v2" in n else 1, n))
    return files


def _natural_key(name: str):
    """自然排序：S_2 在 S_10 前；1_1 在 1_10 前。"""
    stem = Path(name).stem
    key = []
    for part in re.split(r"(\d+)", stem):
        if part == "":
            continue
        key.append((0, int(part), "") if part.isdigit() else (1, 0, part.lower()))
    return (key, name.lower())


def list_videos(dataset: str):
    d = DATASET_DIRS.get(dataset)
    if not d or not os.path.isdir(d):
        return []
    files = [f for f in os.listdir(d) if f.lower().endswith(".mp4")]
    try:
        return sorted(files, key=_natural_key)
    except Exception:
        return sorted(files)


def list_stim_marks():
    marks_dir = Path(STIMULUS_DIR)
    if not marks_dir.exists():
        return []
    return sorted([f.name for f in marks_dir.glob("*.json")])


def _resolve_stim_time(use_stim, mark_name=None, use_upload=False, uploaded=None):
    """统一的刺激时刻解析：``标记文件 > 上传文件``，未启用返回 None。

    旧版在单眼分析、单眼 PDF、双眼分析里各抄了三遍同样的 if/elif 分支，
    这里收敛成一处，避免哪天只改了一处。
    """
    if not use_stim:
        return None
    if use_upload and uploaded is not None:
        return _parse_stim_file(uploaded)
    if mark_name:
        p = Path(STIMULUS_DIR) / mark_name
        if p.exists():
            import json
            try:
                with open(p) as f:
                    return json.load(f).get("current")
            except Exception as e:  # noqa: BLE001
                print(f"刺激标定文件解析失败: {e}")
    return None


def find_source_video(csv_name):
    """反查某个 CSV 对应的源视频。

    优先用推理时写下的 ``<stem>.run.json``；没有就按“视频名是 CSV stem 的前缀”
    在数据集目录、cache/raw_videos、assets 里找最长匹配（后缀须是分隔符，
    避免 ``S_1`` 抢走 ``S_10``）。
    """
    stem = Path(csv_name).stem
    sidecar = Path(VIDEO_OUTPUT_DIR) / f"{stem}.run.json"
    if sidecar.exists():
        try:
            with open(sidecar) as f:
                video = json.load(f).get("video")
            if video and Path(video).exists():
                return video
        except Exception as e:  # noqa: BLE001
            print(f"{sidecar.name} 解析失败: {e}")

    roots = [Path(d) for d in DATASET_DIRS.values()] + [
        Path(RAW_VIDEO_DIR), Path(BASE_DIR) / "assets",
    ]
    best = None
    for root in roots:
        if not root.is_dir():
            continue
        for f in root.rglob("*.mp4"):
            vs = f.stem
            if stem == vs or (stem.startswith(vs) and stem[len(vs)] in "_-."):
                if best is None or len(vs) > len(Path(best).stem):
                    best = str(f)
    return best


def _parse_stim_file(stim_file):
    """解析刺激文件，返回刺激时刻（秒）或 None。"""
    import json
    if stim_file is None:
        return None
    path = stim_file if isinstance(stim_file, str) else getattr(stim_file, "name", None)
    if not path or not os.path.exists(path):
        return None
    try:
        if path.endswith(".json"):
            with open(path) as f:
                data = json.load(f)
            for key in ["current", "stimulus_time", "time"]:
                if key in data and data[key] is not None:
                    return float(data[key])
            if "marks" in data and data["marks"]:
                return float(data["marks"][-1].get("time"))
            return None
        elif path.endswith(".csv"):
            import csv as _csv
            with open(path) as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    for key in ["stimulus_time", "time", "stimulus"]:
                        if key in row and row[key]:
                            return float(row[key])
            return None
    except Exception as e:
        print(f"刺激文件解析失败: {e}")
        return None
    return None


def load_stim_mark(name):
    if not name:
        return ""
    import json
    p = Path(STIMULUS_DIR) / name
    if not p.exists():
        return f"文件不存在: {p}"
    try:
        with open(p) as f:
            data = json.load(f)
        lines = [
            f"视频: {data.get('video')}",
            f"数据集: {data.get('dataset')}",
            f"当前刺激时刻: {data.get('current')}",
            f"标定次数: {len(data.get('marks', []))}",
            "",
            "历史标定:",
        ]
        for m in data.get("marks", []):
            lines.append(f"  {m.get('time'):.3f}s  ({m.get('source')}, {m.get('timestamp')})")
        return "\n".join(lines)
    except Exception as e:
        return f"读取失败: {e}"


def load_dataset_video(dataset, name):
    if not dataset or not name:
        return None
    d = DATASET_DIRS.get(dataset)
    if not d:
        return None
    src_path = os.path.join(d, name)
    if not os.path.exists(src_path):
        return None
    dst = os.path.join(COPY_DIR, f"{dataset}_orig_{name}")
    if not os.path.exists(dst):
        shutil.copy(src_path, dst)
    return dst


# ================= 瞳孔分割推理 =================

def run_image(image, model_name, conf, iou, flow_opt, plr=False, vogeo=False):
    if image is None:
        raise gr.Error("请先上传一张图片")
    overlay, info = predict(image, model_name=model_name, conf=conf, iou=iou, plr=plr)
    p = info["pupil"]
    lines = []
    if p["major"] is not None:
        lines.append(f"瞳孔椭圆长轴: {p['major']:.1f} px")
        lines.append(f"瞳孔椭圆短轴: {p['minor']:.1f} px")
        lines.append(f"椭圆中心: ({p['center'][0]:.1f}, {p['center'][1]:.1f})")
    else:
        lines.append("瞳孔: 未检测到")

    if plr:
        ir = info.get("iris", {})
        mm = info.get("mm", {})
        lines.append("")
        if ir.get("major") is not None:
            lines.append(f"虹膜椭圆长轴: {ir['major']:.1f} px")
            lines.append(f"虹膜椭圆短轴: {ir['minor']:.1f} px")
            lines.append(f"虹膜中心: ({ir['center'][0]:.1f}, {ir['center'][1]:.1f})")
            if mm.get("pupil_mm") is not None:
                lines.append(f"估算瞳孔直径: {mm['pupil_mm']:.2f} mm（角膜缘参照 {mm['cornea_mm']} mm）")
        else:
            lines.append("虹膜: 未检测到")

    lines.append("")
    lines.append(f"检测框数量: {len(info['detections'])}")
    for d in info["detections"]:
        lines.append(f"  {d['cls']}: {d['conf']:.3f}")
    lines.append("")
    lines.append(f"光流优化: {'已勾选' if flow_opt else '未勾选'}")
    if vogeo:
        lines.append("注：图片模式不支持 3D 动态注视模型（需视频序列）")
    return overlay, "\n".join(lines)


def run_video_common(video_path, model_name, conf, iou, max_frames, flow_opt, plr=False, vogeo=False, progress=None):
    rows = []
    final_csv = None
    final_out = None
    for item in predict_video_iter(
        video_path, model_name=model_name, conf=conf, iou=iou,
        max_frames=int(max_frames), progress=progress,
        flow_fill=bool(flow_opt), plr=bool(plr),
    ):
        if item["partial"]:
            rows = item["rows"]
            yield gr.update(), gr.update(), gr.update(), rows
        else:
            stats = item["stats"]
            final_csv = item.get("csv_path")
            lines = [
                f"输出视频: {item['out_path']}",
                f"处理帧数: {stats['frames']}",
                f"检出帧数: {stats['detected']}",
                f"检出率: {stats['detect_rate']*100:.1f}%",
                f"原视频 FPS: {stats['fps']:.1f}",
            ]
            if "pupil_major_mean" in stats:
                lines.append(f"瞳孔长轴均值: {stats['pupil_major_mean']:.1f} px")
                lines.append(f"瞳孔短轴均值: {stats['pupil_minor_mean']:.1f} px")
            if final_csv:
                lines.append(f"CSV: {final_csv}")
            if plr:
                lines.append("模式: PLR（pupil + iris，CSV 含 iris_major/iris_minor/pupil_mm）")
            if flow_opt:
                if item.get("stage") == "infer_done":
                    lines.append("状态: 已完成基础分割推理，光流模型优化中…")
                elif item.get("stage") == "flow_done":
                    lines.append(f"状态: 光流模型优化完成，补帧 {stats.get('flow_filled', 0)} 帧")
                else:
                    lines.append("状态: 光流优化已启用")
            else:
                lines.append(f"光流优化: {'已勾选' if flow_opt else '未勾选'}")
            yield item["out_path"], "\n".join(lines), final_csv, item["rows"]


    # 3D 注视模型：跑完后把结果状态回填到统计框，不再“静默消失”
    if vogeo and plr and final_csv:
        import vogeo_backend as vb
        csv_stem = Path(final_csv).stem
        overlay, gaze_csv, status = vb.run_vogeo(video_path, out_name=csv_stem)
        yield (item["out_path"],
               "\n".join(lines) + "\n\n--- 3D 动态注视模型 ---\n" + status,
               final_csv, item["rows"])

def run_video(video_path, model_name, conf, iou, max_frames, flow_opt, plr=False, vogeo=False, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("请先上传一个视频或选择修复结果")
    yield from run_video_common(video_path, model_name, conf, iou, max_frames, flow_opt, plr, vogeo, progress)


def run_video_from_deblur(deblur_name, model_name, conf, iou, max_frames, flow_opt, plr=False, vogeo=False, progress=gr.Progress()):
    if not deblur_name:
        raise gr.Error("请先选择修复结果视频")
    path = os.path.join(DEBLUR_DIR, deblur_name)
    if not os.path.exists(path):
        raise gr.Error(f"修复结果不存在: {path}")
    yield from run_video_common(path, model_name, conf, iou, max_frames, flow_opt, plr, vogeo, progress)


def run_dataset(video_name, dataset_name, model_name, conf, iou, max_frames, flow_opt, plr=False, vogeo=False, progress=gr.Progress()):
    if not dataset_name or not video_name:
        raise gr.Error("请先选择数据集和视频")
    d = DATASET_DIRS.get(dataset_name)
    if not d:
        raise gr.Error(f"未知数据集: {dataset_name}")
    path = os.path.join(d, video_name)
    if not os.path.exists(path):
        raise gr.Error(f"视频不存在: {path}")
    yield from run_video_common(path, model_name, conf, iou, max_frames, flow_opt, plr, vogeo, progress)


def restart_app():
    import threading, time, os
    def _do():
        time.sleep(1.0)
        os.execv(sys.executable, [sys.executable] + sys.argv)
    threading.Thread(target=_do, daemon=True).start()
    return "🔄 正在重启，请等 2~3 秒后刷新页面"


CSS = """
.gradio-container { max-width: 100% !important; width: 100% !important; padding: 8px 16px !important; }
h1, .prose h1 { font-size: 34px !important; font-weight: 700 !important; }
.gradio-container .tabitem { width: 100%; }
.dual-overlay { max-width: 420px !important; margin: 0 auto; }
.dual-overlay video { max-height: 320px !important; object-fit: contain; }
"""

HEAD = """
<script>
function styleTabs() {
    const bigs = ["瞳孔分割推理", "瞳孔分割训练", "瞳孔运动特征提取", "运动光流特征提取", "瞳孔分割结果分析", "双眼瞳孔分割结果分析", "ROI 裁剪", "刺激标定"];
    document.querySelectorAll('button[role="tab"]').forEach(btn => {
        const t = (btn.textContent || "").trim();
        if (bigs.includes(t)) {
            btn.style.fontSize = "20px";
            btn.style.fontWeight = "600";
            btn.style.padding = "14px 0";
            btn.style.flex = "1 1 0";
        } else {
            btn.style.fontSize = "16px";
            btn.style.fontWeight = "400";
            btn.style.padding = "8px 12px";
            btn.style.flex = "0 1 auto";
        }
    });
}
setTimeout(styleTabs, 300);
setTimeout(styleTabs, 1000);
setTimeout(styleTabs, 2000);
const _obs = new MutationObserver(() => styleTabs());
_obs.observe(document.body, {childList: true, subtree: true});
</script>
"""


with gr.Blocks(title="瞳孔识别平台") as demo:
    gr.Markdown("# 瞳孔识别平台")
    with gr.Row():
        restart_btn = gr.Button("🔄 重启服务", variant="secondary", size="sm", scale=0)
        restart_status = gr.Textbox(label="", show_label=False, lines=1, max_lines=1, scale=3)
    restart_btn.click(restart_app, outputs=[restart_status])

    with gr.Tabs(elem_classes="main-tabs"):
        # ============ 1. 瞳孔分割推理 ============
        with gr.Tab("瞳孔分割推理", elem_classes="big-tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_dd = gr.Dropdown(choices=model_choices, value=DEFAULT_MODEL, label="模型")
                    plr_note = gr.Textbox(
                        label="PLR 模式说明", lines=2, interactive=False, visible=False,
                        show_label=False,
                    )
                    with gr.Accordion("参数", open=False):
                        conf_slider = gr.Slider(0.0, 1.0, value=CONF_THRESHOLD, label="置信度阈值")
                        iou_slider = gr.Slider(0.0, 1.0, value=IOU_THRESHOLD, label="IoU 阈值")

            def _plr_notice(model_name, plr_on):
                """勾 PLR 时提示“模型选择对 PLR 无效”（三份 PLR 权重其实是同一份）。"""
                if not plr_on:
                    return gr.update(value="", visible=False)
                return gr.update(value=model_info.plr_mode_note(model_name), visible=True)

            with gr.Tabs():
                with gr.Tab("图片推理"):
                    with gr.Row():
                        with gr.Column():
                            img_in = gr.Image(type="pil", label="输入图片")
                            flow_opt_img = gr.Checkbox(label="推理后结果光流模型优化", value=False)
                            plr_opt_img = gr.Checkbox(label="PLR 模式（双类：瞳孔 + 虹膜，估算 mm；生成 3D 模型需勾选下方选项）", value=False)
                            vogeo_opt_img = gr.Checkbox(label="额外生成 3D 动态注视模型（VOGeo-Gaze）", value=False, visible=False)
                            btn_img = gr.Button("识别图片", variant="primary")
                        with gr.Column():
                            img_out = gr.Image(label="分割结果（红=瞳孔轮廓，蓝=瞳孔椭圆，绿=虹膜轮廓）")
                            text_img = gr.Textbox(label="检测详情", lines=16)
                    plr_opt_img.change(lambda x: gr.update(visible=bool(x)), inputs=[plr_opt_img], outputs=[vogeo_opt_img])
                    plr_opt_img.change(_plr_notice, inputs=[model_dd, plr_opt_img], outputs=[plr_note])

                    btn_img.click(run_image,
                                  inputs=[img_in, model_dd, conf_slider, iou_slider, flow_opt_img, plr_opt_img, vogeo_opt_img],
                                  outputs=[img_out, text_img])
                    if EXAMPLES_DIR.exists() and any(EXAMPLES_DIR.iterdir()):
                        gr.Examples(examples=[[str(p)] for p in EXAMPLES_DIR.glob("*") if p.is_file()],
                                    inputs=[img_in])

                with gr.Tab("视频推理"):
                    use_deblur_vid = gr.Checkbox(
                        label="从运动特征提取修复结果中选择视频（outputs/deblur_videos）",
                        value=False,
                    )
                    vid_in = gr.Video(label="上传视频")
                    with gr.Row(visible=False) as deblur_row:
                        deblur_vid_dd = gr.Dropdown(
                            choices=list_deblur_videos(),
                            value=(list_deblur_videos()[0] if list_deblur_videos() else None),
                            label="选择修复结果视频",
                            scale=3,
                        )
                        refresh_deblur_vid = gr.Button("刷新修复结果列表", scale=1)
                    with gr.Row():
                        with gr.Column(scale=1):
                            deblur_preview = gr.Video(label="修复视频（未加 mask）", interactive=False, visible=False)
                            max_frames_slider = gr.Slider(VIDEO_MAX_FRAMES_MIN, VIDEO_MAX_FRAMES_MAX, value=VIDEO_MAX_FRAMES, step=10, label="最多处理帧数")
                            flow_opt_vid = gr.Checkbox(label="推理后结果光流模型优化", value=False)
                            plr_opt_vid = gr.Checkbox(label="PLR 模式（双类：瞳孔 + 虹膜，估算 mm；生成 3D 模型需勾选下方选项）", value=False)
                            vogeo_opt_vid = gr.Checkbox(label="额外生成 3D 动态注视模型（VOGeo-Gaze）", value=False, visible=False)
                            with gr.Row():
                                btn_vid = gr.Button("识别视频", variant="primary")
                                btn_vid_deblur = gr.Button("识别修复结果视频", variant="primary", visible=False)
                        with gr.Column(scale=1):
                            vid_out = gr.Video(label="分割结果视频（带 mask）", format="mp4")
                            text_vid = gr.Textbox(label="分割统计", lines=12)
                    df_out = gr.Dataframe(
                        headers=["frame", "time_sec", "detected", "major", "minor", "cx", "cy", "conf",
                                 "iris_detected", "iris_major", "iris_minor", "pupil_mm"],
                        datatype=["number"] * 12,
                        label="逐帧分割数据（推理中实时更新；PLR 模式多 4 列）",
                        wrap=True,
                        row_count=(10, "dynamic"),
                    )
                    csv_out = gr.File(label="下载分割结果 CSV")

                    def _toggle_deblur_vid(use_db):
                        return (
                            gr.update(visible=not use_db),
                            gr.update(visible=use_db),
                            gr.update(visible=use_db),
                            gr.update(visible=not use_db),
                            gr.update(visible=use_db),
                        )

                    use_deblur_vid.change(
                        _toggle_deblur_vid,
                        inputs=[use_deblur_vid],
                        outputs=[vid_in, deblur_row, deblur_preview, btn_vid, btn_vid_deblur],
                    )
                    refresh_deblur_vid.click(
                        lambda: gr.update(choices=list_deblur_videos()),
                        outputs=[deblur_vid_dd],
                    )
                    deblur_vid_dd.change(
                        lambda name: os.path.join(DEBLUR_DIR, name) if name else None,
                        inputs=[deblur_vid_dd],
                        outputs=[deblur_preview],
                    )
                    plr_opt_vid.change(lambda x: gr.update(visible=bool(x)), inputs=[plr_opt_vid], outputs=[vogeo_opt_vid])
                    plr_opt_vid.change(_plr_notice, inputs=[model_dd, plr_opt_vid], outputs=[plr_note])

                    btn_vid.click(
                        run_video,
                        inputs=[vid_in, model_dd, conf_slider, iou_slider, max_frames_slider, flow_opt_vid, plr_opt_vid, vogeo_opt_vid],
                        outputs=[vid_out, text_vid, csv_out, df_out],
                    )
                    btn_vid_deblur.click(
                        run_video_from_deblur,
                        inputs=[deblur_vid_dd, model_dd, conf_slider, iou_slider, max_frames_slider, flow_opt_vid, plr_opt_vid, vogeo_opt_vid],
                        outputs=[vid_out, text_vid, csv_out, df_out],
                    )

                with gr.Tab("数据集测试"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            dataset_dd = gr.Dropdown(
                                choices=list(DATASET_DIRS.keys()),
                                value="OpenEDS",
                                label="数据集",
                            )
                            refresh_btn = gr.Button("刷新视频列表")
                            video_dd = gr.Dropdown(
                                choices=list_videos("OpenEDS"),
                                value=(list_videos("OpenEDS")[0] if list_videos("OpenEDS") else None),
                                label="选择视频",
                            )
                            vid_in2 = gr.Video(label="原视频", interactive=False)
                            max_frames_slider2 = gr.Slider(VIDEO_MAX_FRAMES_MIN, VIDEO_MAX_FRAMES_MAX, value=VIDEO_MAX_FRAMES, step=10, label="最多处理帧数")
                            flow_opt_openeds = gr.Checkbox(label="推理后结果光流模型优化", value=False)
                            plr_opt_openeds = gr.Checkbox(label="PLR 模式（双类：瞳孔 + 虹膜，估算 mm；生成 3D 模型需勾选下方选项）", value=False)
                            vogeo_opt_openeds = gr.Checkbox(label="额外生成 3D 动态注视模型（VOGeo-Gaze）", value=False, visible=False)
                            btn_openeds = gr.Button("推理", variant="primary")
                        with gr.Column(scale=2):
                            vid_out2 = gr.Video(label="分割结果视频（带 mask）", format="mp4")
                            text_out2 = gr.Textbox(label="分割统计", lines=12)
                    df_out2 = gr.Dataframe(
                        headers=["frame", "time_sec", "detected", "major", "minor", "cx", "cy", "conf",
                                 "iris_detected", "iris_major", "iris_minor", "pupil_mm"],
                        datatype=["number"] * 12,
                        label="逐帧分割数据（推理中实时更新；PLR 模式多 4 列）",
                        wrap=True,
                        row_count=(10, "dynamic"),
                    )
                    csv_out2 = gr.File(label="下载分割结果 CSV")

                    dataset_dd.change(
                        lambda ds: gr.update(choices=list_videos(ds), value=(list_videos(ds)[0] if list_videos(ds) else None)),
                        inputs=[dataset_dd],
                        outputs=[video_dd],
                    )
                    video_dd.change(load_dataset_video, inputs=[dataset_dd, video_dd], outputs=[vid_in2])
                    refresh_btn.click(
                        lambda ds: gr.update(choices=list_videos(ds)),
                        inputs=[dataset_dd],
                        outputs=[video_dd],
                    )
                    plr_opt_openeds.change(lambda x: gr.update(visible=bool(x)), inputs=[plr_opt_openeds], outputs=[vogeo_opt_openeds])
                    plr_opt_openeds.change(_plr_notice, inputs=[model_dd, plr_opt_openeds], outputs=[plr_note])
                    model_dd.change(_plr_notice, inputs=[model_dd, plr_opt_img], outputs=[plr_note])

                    btn_openeds.click(
                        run_dataset,
                        inputs=[video_dd, dataset_dd, model_dd, conf_slider, iou_slider, max_frames_slider2, flow_opt_openeds, plr_opt_openeds, vogeo_opt_openeds],
                        outputs=[vid_out2, text_out2, csv_out2, df_out2],
                    )

        # ============ 2. 瞳孔分割训练 ============
        with gr.Tab("瞳孔分割训练", elem_classes="big-tab"):
            gr.Markdown("### 瞳孔分割训练")
            with gr.Row():
                with gr.Column(scale=1):
                    train_yaml_dd = gr.Dropdown(
                        choices=TRAIN_MODEL_YAMLS,
                        value=TRAIN_MODEL_YAMLS[0],
                        label="模型 YAML",
                    )
                    train_dataset_dd = gr.Dropdown(
                        choices=list(DATASET_YAMLS.keys()),
                        value=next(iter(DATASET_YAMLS)),
                        label="数据集",
                    )
                    train_epochs = gr.Number(value=100, label="Epochs")
                    train_batch = gr.Number(value=8, label="Batch Size")
                    train_lr = gr.Number(value=0.0005, label="LR0")
                    train_imgsz = gr.Number(value=640, label="imgsz")
                    train_device = gr.Textbox(value="0", label="device（0=GPU，cpu=CPU）")
                    with gr.Row():
                        btn_train_start = gr.Button("开始训练", variant="primary")
                        btn_train_stop = gr.Button("停止训练", variant="stop")
                        btn_train_refresh = gr.Button("刷新日志")
                with gr.Column(scale=2):
                    train_status = gr.Textbox(label="状态", lines=2, interactive=False)
                    train_log = gr.Textbox(label="训练日志（尾部 30 行）", lines=15, interactive=False,
                                           elem_classes="train-log")
            train_timer = gr.Timer(value=3.0, active=True)

            def _start(yaml_name, dataset_name, epochs, batch, lr0, imgsz, device):
                msg = tb.start_training(yaml_name, dataset_name, epochs, batch, lr0, device, imgsz)
                return msg, tb.read_log()

            def _stop():
                msg = tb.stop_training()
                return msg, tb.read_log()

            def _refresh():
                status = tb.parse_progress()
                running = "运行中" if tb.is_running() else "未运行"
                return f"{running} | {status}", tb.read_log()

            btn_train_start.click(_start,
                                  inputs=[train_yaml_dd, train_dataset_dd, train_epochs, train_batch,
                                          train_lr, train_imgsz, train_device],
                                  outputs=[train_status, train_log])
            btn_train_stop.click(_stop, outputs=[train_status, train_log])
            btn_train_refresh.click(_refresh, outputs=[train_status, train_log])
            train_timer.tick(_refresh, outputs=[train_status, train_log])

        # ============ 3. 瞳孔运动特征提取 ============
        with gr.Tab("瞳孔运动特征提取", elem_classes="big-tab"):
            gr.Markdown("### 瞳孔去模糊")
            gr.Markdown("> 只调用 DeblurUNet 逐帧去模糊，不跑 YOLO、不做分割。")

            with gr.Row():
                with gr.Column(scale=1):
                    dataset_dd_motion = gr.Dropdown(
                        choices=list(DATASET_DIRS.keys()),
                        value="LPW",
                        label="数据集",
                    )
                    refresh_motion = gr.Button("刷新视频列表")
                    video_dd_motion = gr.Dropdown(
                        choices=list_videos("LPW"),
                        value=(list_videos("LPW")[0] if list_videos("LPW") else None),
                        label="选择视频",
                    )
                    preview_motion = gr.Video(label="原视频", interactive=False)
                    _experts = db.list_experts() or ["general_v2"]
                    expert_motion = gr.Dropdown(
                        choices=["智能（自动根据光流统计量选择）"] + _experts,
                        value=_experts[0],
                        label="去模糊专家",
                    )
                    max_frames_motion = gr.Slider(VIDEO_MAX_FRAMES_MIN, VIDEO_MAX_FRAMES_MAX, value=VIDEO_MAX_FRAMES, step=10, label="最多处理帧数")
                    btn_deblur = gr.Button("去模糊", variant="primary")
                with gr.Column(scale=2):
                    deblur_out = gr.Video(label="去模糊结果")
                    deblur_summary = gr.Textbox(label="统计", lines=6)

            def _motion_list(ds):
                vs = list_videos(ds)
                return gr.update(choices=vs, value=(vs[0] if vs else None))

            def _motion_load(ds, name):
                if not ds or not name:
                    return None
                d = DATASET_DIRS.get(ds)
                if not d:
                    return None
                src_path = os.path.join(d, name)
                if not os.path.exists(src_path):
                    return None
                dst = os.path.join(COPY_DIR, f"motion_{ds}_{name}")
                if not os.path.exists(dst):
                    shutil.copy(src_path, dst)
                return dst

            def _run_deblur(ds, video_name, expert, max_frames, progress=gr.Progress()):
                if not ds or not video_name:
                    raise gr.Error("请先选择数据集和视频")
                d = DATASET_DIRS.get(ds)
                if not d:
                    raise gr.Error(f"未知数据集: {ds}")
                path = os.path.join(d, video_name)
                if not os.path.exists(path):
                    raise gr.Error(f"视频不存在: {path}")

                is_auto = expert.startswith("智能")
                if is_auto:
                    try:
                        stats, _, _ = fsb.stats_from_video_pair(path, frame_i=0)
                        rec, reason = fsb.recommend_expert(stats)
                        expert = rec
                        auto_note = f"\n[智能选择] 推荐: {expert}\n理由: {reason}"
                    except Exception as e:
                        expert = "general_v2"
                        auto_note = f"\n[智能选择] 统计量计算失败({e})，暂用 general_v2"
                else:
                    auto_note = ""

                out, summary = mb.deblur_video_only(
                    path, expert=expert, max_frames=int(max_frames), progress=progress
                )
                return out, summary + auto_note

            dataset_dd_motion.change(_motion_list, inputs=[dataset_dd_motion], outputs=[video_dd_motion])
            video_dd_motion.change(_motion_load, inputs=[dataset_dd_motion, video_dd_motion], outputs=[preview_motion])
            refresh_motion.click(lambda ds: _motion_list(ds), inputs=[dataset_dd_motion], outputs=[video_dd_motion])
            btn_deblur.click(
                _run_deblur,
                inputs=[dataset_dd_motion, video_dd_motion, expert_motion, max_frames_motion],
                outputs=[deblur_out, deblur_summary],
            )

        # ============ 3b. 运动光流特征提取（大 Tab，含训练 + 计算两个子 Tab） ============
        with gr.Tab("运动光流特征提取", elem_classes="big-tab"):
            with gr.Tabs():
                # ---- 3b-1. 运动光流预定义模型训练 ----
                with gr.Tab("运动光流预定义模型训练"):
                    gr.Markdown("### 运动光流预定义模型训练（StatsSeqTransformer）")
                    gr.Markdown("> 从模糊光流统计量序列回归清晰统计量序列，用于帮助运动光流特征计算。")
                    with gr.Row():
                        with gr.Column(scale=1):
                            stats_info = gr.Textbox(
                                label="预定义模型环境信息", lines=3, interactive=False,
                                value=tb.stats_checkpoint_info(),
                            )
                            stats_refresh_info = gr.Button("刷新环境信息")
                            with gr.Row():
                                btn_stats_start = gr.Button("开始/继续训练", variant="primary")
                                btn_stats_fresh = gr.Button("从头训练（删检查点）", variant="stop")
                                btn_stats_stop = gr.Button("停止训练", variant="stop")
                                btn_stats_log_refresh = gr.Button("刷新日志")
                        with gr.Column(scale=2):
                            stats_status = gr.Textbox(label="状态", lines=2, interactive=False)
                            stats_log = gr.Textbox(label="训练日志（尾部 30 行）", lines=15, interactive=False,
                                                   elem_classes="train-log")
                    stats_timer = gr.Timer(value=3.0, active=True)

                    def _stats_start():
                        msg = tb.start_stats_training(resume=True)
                        return msg, tb.read_stats_log()

                    def _stats_fresh():
                        try:
                            if tb.STATS_CHECKPOINT.exists():
                                tb.STATS_CHECKPOINT.unlink()
                                note = "已删除检查点，"
                            else:
                                note = "无检查点，"
                        except Exception as e:
                            return f"删除检查点失败: {e}", tb.read_stats_log()
                        msg = tb.start_stats_training(resume=False)
                        return note + msg, tb.read_stats_log()

                    def _stats_stop():
                        msg = tb.stop_stats_training()
                        return msg, tb.read_stats_log()

                    def _stats_refresh():
                        status = tb.parse_stats_progress()
                        running = "运行中" if tb.is_stats_running() else "未运行"
                        return f"{running} | {status}", tb.read_stats_log()

                    def _stats_refresh_info():
                        return tb.stats_checkpoint_info()

                    btn_stats_start.click(_stats_start, outputs=[stats_status, stats_log])
                    btn_stats_fresh.click(_stats_fresh, outputs=[stats_status, stats_log])
                    btn_stats_stop.click(_stats_stop, outputs=[stats_status, stats_log])
                    btn_stats_log_refresh.click(_stats_refresh, outputs=[stats_status, stats_log])
                    stats_refresh_info.click(_stats_refresh_info, outputs=[stats_info])
                    stats_timer.tick(_stats_refresh, outputs=[stats_status, stats_log])

                # ---- 3b-2. 运动光流特征计算 ----
                with gr.Tab("运动光流特征计算"):
                    gr.Markdown("### 运动光流特征计算")
                    gr.Markdown("> 输入视频，取第 i 帧与第 i+5 帧（与训练数据间隔一致）计算 24 维光流统计量，并给出推荐去模糊专家。")

                    with gr.Row():
                        with gr.Column(scale=1):
                            flow_dataset_dd = gr.Dropdown(
                                choices=list(DATASET_DIRS.keys()),
                                value="LPW",
                                label="数据集",
                            )
                            flow_refresh = gr.Button("刷新视频列表")
                            flow_video_dd = gr.Dropdown(
                                choices=list_videos("LPW"),
                                value=(list_videos("LPW")[0] if list_videos("LPW") else None),
                                label="选择视频",
                            )
                            flow_preview = gr.Video(label="原视频", interactive=False)
                            flow_frame_i = gr.Number(value=0, label="起始帧 i（统计第 i 帧 vs i+5 帧）")
                            flow_use_model = gr.Checkbox(label="是否使用预定义模型辅助", value=False)
                            btn_flow_stats = gr.Button("计算光流统计量", variant="primary")
                            btn_build_flow_index = gr.Button("构建视频统计量索引", variant="primary")
                        with gr.Column(scale=2):
                            flow_stats_df = gr.Dataframe(
                                headers=["统计量", "值"],
                                datatype=["str", "number"],
                                label="24 维光流统计量",
                                wrap=True,
                                row_count=(12, "dynamic"),
                            )
                            flow_recommend = gr.Textbox(label="推荐去模糊专家", lines=5)
                            flow_vis = gr.Image(label="光流可视化（HSV）")

                    def _flow_list(ds):
                        vs = list_videos(ds)
                        return gr.update(choices=vs, value=(vs[0] if vs else None))

                    def _flow_load(ds, name):
                        if not ds or not name:
                            return None
                        d = DATASET_DIRS.get(ds)
                        if not d:
                            return None
                        src_path = os.path.join(d, name)
                        if not os.path.exists(src_path):
                            return None
                        dst = os.path.join(COPY_DIR, f"flow_{ds}_{name}")
                        if not os.path.exists(dst):
                            shutil.copy(src_path, dst)
                        return dst

                    def _run_flow_stats(ds, video_name, frame_i, use_model):
                        if not ds or not video_name:
                            raise gr.Error("请先选择数据集和视频")
                        d = DATASET_DIRS.get(ds)
                        if not d:
                            raise gr.Error(f"未知数据集: {ds}")
                        path = os.path.join(d, video_name)
                        if not os.path.exists(path):
                            raise gr.Error(f"视频不存在: {path}")
                        try:
                            stats, flow, info = fsb.stats_from_video_pair(path, frame_i=int(frame_i))
                        except Exception as e:
                            raise gr.Error(f"计算失败: {e}")
                        table = fsb.stats_to_table(stats)
                        expert, reason = fsb.recommend_expert(stats)
                        text = f"推荐专家: {expert}\n理由: {reason}\n\n{info}"
                        if use_model:
                            text += "\n\n预定义模型辅助: 已勾选（功能待接入）"
                        vis = fsb.flow_visualization(flow)
                        return table, text, vis

                    def _build_flow_index(ds, video_name, frame_i):
                        if not ds or not video_name:
                            return "请先选择数据集和视频。\n（构建视频统计量索引：功能待接入）"
                        return (f"构建视频统计量索引：功能待接入\n"
                                f"数据集: {ds}\n视频: {video_name}\n起始帧: {frame_i}")

                    flow_dataset_dd.change(_flow_list, inputs=[flow_dataset_dd], outputs=[flow_video_dd])
                    flow_video_dd.change(_flow_load, inputs=[flow_dataset_dd, flow_video_dd], outputs=[flow_preview])
                    flow_refresh.click(lambda ds: _flow_list(ds), inputs=[flow_dataset_dd], outputs=[flow_video_dd])
                    btn_flow_stats.click(
                        _run_flow_stats,
                        inputs=[flow_dataset_dd, flow_video_dd, flow_frame_i, flow_use_model],
                        outputs=[flow_stats_df, flow_recommend, flow_vis],
                    )
                    btn_build_flow_index.click(
                        _build_flow_index,
                        inputs=[flow_dataset_dd, flow_video_dd, flow_frame_i],
                        outputs=[flow_recommend],
                    )

        # ============ 4. 瞳孔分割结果分析 ============
        with gr.Tab("瞳孔分割结果分析", elem_classes="big-tab"):
            gr.Markdown("### 瞳孔分割结果分析")
            with gr.Row():
                with gr.Column(scale=1):
                    refresh_csv_btn = gr.Button("刷新 CSV 列表")
                    csv_dd = gr.Dropdown(
                        choices=ab.list_csv_files(),
                        value=(ab.list_csv_files()[0] if ab.list_csv_files() else None),
                        label="选择逐帧 CSV（来自 outputs/segmentation_results）",
                    )
                    use_stim_file = gr.Checkbox(label="导入刺激文件", value=False)
                    stim_source_radio = gr.Radio(
                        choices=["从 stimulus_marks 选择", "上传文件"],
                        value="从 stimulus_marks 选择",
                        label="刺激来源",
                        visible=False,
                    )
                    stim_marks_select = gr.Dropdown(choices=list_stim_marks(), label="选择 stimulus_marks 文件", visible=False)
                    refresh_stim_marks = gr.Button("刷新 stimulus_marks 列表", visible=False)
                    stim_file = gr.File(label="上传刺激文件", file_types=[".csv", ".json"], visible=False)
                    with gr.Row():
                        btn_analyze = gr.Button("分析", variant="primary")
                        btn_export_single_pdf = gr.Button("导出 PDF 报告", variant="secondary")
                with gr.Column(scale=2):
                    analysis_summary = gr.Textbox(label="概览", lines=3)
                    single_pdf_status = gr.Textbox(label="PDF 导出状态", lines=1, interactive=False)
                    single_pdf_file = gr.File(label="PDF 报告", interactive=False)
            with gr.Tabs():
                with gr.Tab("形态与运动"):
                    gr.Markdown("> 纯像素形态与运动信息，不做病症判读。")
                    morph_summary = gr.Textbox(label="形态与运动概览", lines=3)
                    with gr.Row():
                        with gr.Column():
                            morph_table = gr.Dataframe(
                                headers=["形态指标", "值"], datatype=["str", "str"],
                                label="形态信息", wrap=True, row_count=(12, "dynamic"),
                            )
                        with gr.Column():
                            motion_table = gr.Dataframe(
                                headers=["运动指标", "值"], datatype=["str", "str"],
                                label="运动信息", wrap=True, row_count=(10, "dynamic"),
                            )
                    morph_gallery = gr.Gallery(label="可视化", columns=2, height="auto", object_fit="contain")

                with gr.Tab("PLR 参数分析"):
                    gr.Markdown("> 使用上方选择的 CSV。PLR CSV（含 pupil_mm）按 mm 计算，普通 CSV 按 major（px）计算。刺激时刻自动检测。")
                    with gr.Row():
                        with gr.Column(scale=1):
                            plr_summary = gr.Textbox(label="概览", lines=5)
                        with gr.Column(scale=2):
                            plr_params_df = gr.Dataframe(
                                headers=["参数", "值", "参考范围"],
                                datatype=["str", "str", "str"],
                                label="PLR 临床参数",
                                wrap=True,
                                row_count=(15, "dynamic"),
                            )
                            plr_plot = gr.Image(label="PLR 分析图")

                with gr.Tab("3D 动态注视模型"):
                    gr.Markdown("> 显示该 CSV 对应的 VOGeo-Gaze 结果（overlay 视频 + gaze 摘要）。需先在视频推理时勾选 PLR + 额外生成 3D 模型。")
                    with gr.Row():
                        with gr.Column(scale=1):
                            btn_load_vogeo = gr.Button("加载 3D 模型结果", variant="primary")
                            btn_regen_vogeo = gr.Button("（重新）生成 3D 模型", variant="secondary")
                            vogeo_info = gr.Textbox(label="状态", lines=6, interactive=False)
                        with gr.Column(scale=2):
                            vogeo_overlay = gr.Video(label="VOGeo-Gaze overlay", format="mp4")
                            vogeo_gaze_summary = gr.Textbox(label="gaze 摘要", lines=6, interactive=False)
                            
                    gaze_gallery = gr.Gallery(label="Gaze 静态图", columns=2, height="auto", object_fit="contain")

                    def _load_vogeo(csv_name):
                        if not csv_name:
                            raise gr.Error("请先在上方选择 CSV")
                        import vogeo_backend as vb
                        stem = Path(csv_name).stem
                        d = vb.OUTPUT_DIR / stem
                        overlay = d / "overlay.mp4"
                        csv_g = d / "gaze.csv"
                        if not (overlay.exists() or csv_g.exists()):
                            # 目录缺失或为空时，把 vogeo.log 的尾部一起显示出来
                            return vb.status_for(stem), None, "", []
                        info = f"目录: {d}"
                        summary = ""
                        if csv_g.exists():
                            try:
                                import csv as _csv
                                import numpy as _np
                                with open(csv_g) as f:
                                    rows = list(_csv.DictReader(f))
                                if rows:
                                    th = [float(r["theta_h"]) for r in rows if r.get("theta_h") not in ("", None)]
                                    tv = [float(r["theta_v"]) for r in rows if r.get("theta_v") not in ("", None)]
                                    if th:
                                        summary += f"theta_h: 均值 {_np.mean(th):.2f}°，范围 [{min(th):.2f}, {max(th):.2f}]\n"
                                    if tv:
                                        summary += f"theta_v: 均值 {_np.mean(tv):.2f}°，范围 [{min(tv):.2f}, {max(tv):.2f}]\n"
                                    summary += f"帧数: {len(rows)}"
                            except Exception as e:
                                summary = f"gaze.csv 解析失败: {e}"
                        gaze_plots = []
                        if csv_g.exists():
                            pupil_csv = ab.CSV_DIR / csv_name
                            gaze_plots = vb.plot_gaze(csv_g, plot_dir=vb.OUTPUT_DIR / stem, pupil_csv=pupil_csv)
                        return info, str(overlay) if overlay.exists() else None, summary, gaze_plots


                    def _regen_vogeo(csv_name):
                        """对已有 CSV 直接重建 3D 注视模型（不必重跑分割推理）。"""
                        if not csv_name:
                            raise gr.Error("请先在上方选择 CSV")
                        import vogeo_backend as vb
                        stem = Path(csv_name).stem
                        video = find_source_video(csv_name)
                        if not video:
                            return (f"❌ 找不到 {stem} 的源视频（缺 {stem}.run.json，"
                                    f"数据集里也没有匹配的 mp4）。请先在“视频推理”里重跑一次。"), None, "", []
                        overlay, gaze_csv, status = vb.run_vogeo(video, out_name=stem)
                        info, ov, summary, plots = _load_vogeo(csv_name)
                        return f"{status}\n\n{info}", ov, summary, plots

                    btn_regen_vogeo.click(
                        _regen_vogeo,
                        inputs=[csv_dd],
                        outputs=[vogeo_info, vogeo_overlay, vogeo_gaze_summary, gaze_gallery],
                    )



                with gr.Tab("临床判读"):
                    gr.Markdown("> 基于单眼 + PLR 参数的第一档判读（正常/针尖样/散大/对光反射消失/PLR 变差/形态）。")
                    interpret_summary = gr.Textbox(label="综合建议", lines=8, interactive=False)
                    interpret_df = gr.Dataframe(
                        headers=["判读项", "结果", "依据"],
                        datatype=["str", "str", "str"],
                        label="临床判读",
                        wrap=True,
                        column_widths=["25%", "15%", "60%"],
                        row_count=(8, "dynamic"),
                    )
                    gr.Markdown("### PPT 病症分类对照（全部类别）")
                    single_ppt_ref_df = gr.Dataframe(
                        headers=["病症", "判读", "依据（本视频）", "判读条件"],
                        datatype=["str", "str", "str", "str"],
                        label="单眼病症对照（含本视频判读）",
                        wrap=True,
                        column_widths=["30%", "18%", "32%", "20%"],
                        row_count=(10, "dynamic"),
                    )

                with gr.Tab("视频与模型概况"):
                    summary_df = gr.Dataframe(headers=["指标", "值"], datatype=["str", "str"],
                                              label="视频与模型概况", wrap=True)
                    gr.Markdown("### 数据质量说明")
                    quality_df = gr.Dataframe(headers=["指标", "值", "结论"], datatype=["str", "str", "str"],
                                              label="数据质量摘要", wrap=True)
                    gr.Markdown("### 方法说明")
                    gr.Markdown(
                        "- **分割**：YOLO26-seg + BiFPN，在 OpenEDS 上训练。\n"
                        "- **瞳孔直径 (mm)**：`pupil_px / iris_px × 11.5 mm`，以虹膜直径作为标尺。\n"
                        "- **PLR 参数**：从 `pupil_mm` 时间序列推导；刺激时刻来自标定文件或自动检测。\n"
                        "- **3D 注视**：VOGeo-Gaze 双球眼球模型；gaze 为光轴方向。\n"
                        "- **免责声明**：自动化分析，不构成临床诊断。"
                    )
                with gr.Tab("逐帧数据 + 异常帧（前 200 行）"):
                    frame_df = gr.Dataframe(
                        headers=["frame", "time_sec", "detected", "major", "minor", "cx", "cy", "conf",
                                 "iris_detected", "iris_major", "iris_minor", "pupil_mm"],
                        datatype=["number"] * 12,
                        label="逐帧分割数据（PLR 模式多 4 列）", wrap=True, row_count=(10, "dynamic"),
                    )
                    gr.Markdown("### 异常帧（数据质量）")
                    abnormal_summary = gr.Textbox(label="异常统计", lines=2, interactive=False)
                    abnormal_df = gr.Dataframe(
                        headers=["frame", "major", "minor", "conf", "原因"],
                        datatype=["number", "number", "number", "number", "str"],
                        label="异常帧（未检出 / 长短轴偏离 / 虹膜缺失）", wrap=True, row_count=(8, "dynamic"),
                    )


            refresh_csv_btn.click(lambda: gr.update(choices=ab.list_csv_files()), outputs=[csv_dd])
            def _run_all_analysis(csv_name, use_stim_file=False, stim_file=None, stim_mark_name=None):
                """UI 只负责取刺激时刻，计算全部交给 services.analyze_all。"""
                if not csv_name:
                    raise gr.Error("请先选择 CSV")
                stim = _resolve_stim_time(use_stim_file, stim_mark_name, False, stim_file)
                return services.analyze_all(csv_name, stim)

            use_stim_file.change(lambda x: gr.update(visible=bool(x)), inputs=[use_stim_file], outputs=[stim_source_radio])
            stim_source_radio.change(
                lambda x: (gr.update(visible=(x == "从 stimulus_marks 选择")), gr.update(visible=(x == "上传文件"))),
                inputs=[stim_source_radio],
                outputs=[stim_marks_select, stim_file],
            )

            refresh_stim_marks.click(lambda: gr.update(choices=list_stim_marks()), outputs=[stim_marks_select])

            btn_analyze.click(
                _run_all_analysis,
                inputs=[csv_dd, use_stim_file, stim_file, stim_marks_select],
                outputs=[summary_df, frame_df, abnormal_df, analysis_summary,
                         morph_table, motion_table, morph_summary, morph_gallery,
                         plr_params_df, plr_summary, plr_plot, abnormal_summary,
                         interpret_df, interpret_summary, single_ppt_ref_df, quality_df],
            )

            def _export_single_pdf(csv_name, use_stim_file=False, stim_file=None, stim_mark_name=None):
                if not csv_name:
                    raise gr.Error("请先选择 CSV")
                stim_time = _resolve_stim_time(use_stim_file, stim_mark_name, False, stim_file)
                try:
                    path = pe.export_single_eye_pdf(csv_name, stimulus_time=stim_time)
                    return f"✅ 已生成: {path}", path
                except Exception as e:
                    return f"❌ 导出失败: {e}", None
            
            

            btn_export_single_pdf.click(
                _export_single_pdf,
                inputs=[csv_dd, use_stim_file, stim_file, stim_marks_select],
                outputs=[single_pdf_status, single_pdf_file],
            )

            btn_load_vogeo.click(
                _load_vogeo,
                inputs=[csv_dd],
                outputs=[vogeo_info, vogeo_overlay, vogeo_gaze_summary, gaze_gallery],
            )
            csv_dd.change(
                _load_vogeo,
                inputs=[csv_dd],
                outputs=[vogeo_info, vogeo_overlay, vogeo_gaze_summary, gaze_gallery],
            )

        # ============ 5. 双眼瞳孔分割结果分析 ============
        with gr.Tab("双眼瞳孔分割结果分析", elem_classes="big-tab"):
            gr.Markdown("### 双眼瞳孔分割结果分析（双眼 = 两个单眼）")
            with gr.Row():
                with gr.Column(scale=1):
                    refresh_dual_btn = gr.Button("刷新 CSV 列表")
                    dual_left_dd = gr.Dropdown(choices=ab.list_csv_files(), label="左眼 CSV")
                    use_stim_l = gr.Checkbox(label="左眼导入刺激文件", value=False)
                    stim_marks_select_l = gr.Dropdown(choices=list_stim_marks(), label="左眼 stimulus_marks 文件", visible=False)
                    refresh_stim_marks_l = gr.Button("刷新左眼 stimulus_marks 列表", visible=False)
                    use_upload_l = gr.Checkbox(label="左眼改用上传文件", value=False, visible=False)
                    stim_file_l = gr.File(label="左眼上传刺激文件", file_types=[".csv", ".json"], visible=False)
                    dual_right_dd = gr.Dropdown(choices=ab.list_csv_files(), label="右眼 CSV")
                    use_stim_r = gr.Checkbox(label="右眼导入刺激文件", value=False)
                    stim_marks_select_r = gr.Dropdown(choices=list_stim_marks(), label="右眼 stimulus_marks 文件", visible=False)
                    refresh_stim_marks_r = gr.Button("刷新右眼 stimulus_marks 列表", visible=False)
                    use_upload_r = gr.Checkbox(label="右眼改用上传文件", value=False, visible=False)
                    stim_file_r = gr.File(label="右眼上传刺激文件", file_types=[".csv", ".json"], visible=False)
                    with gr.Row():
                        btn_dual_analyze = gr.Button("分析", variant="primary")
                        btn_export_dual_pdf = gr.Button("导出 PDF 报告", variant="secondary")
                    dual_pdf_status = gr.Textbox(label="PDF 导出状态", lines=1, interactive=False)
                    dual_pdf_file = gr.File(label="PDF 报告", interactive=False)
                with gr.Column(scale=2):
                    dual_summary = gr.Textbox(label="概览", lines=4, interactive=False)

            with gr.Tabs():
                with gr.Tab("形态与运动"):
                    with gr.Row():
                        with gr.Column():
                            left_morph_table = gr.Dataframe(headers=["形态指标", "值"], datatype=["str", "str"],
                                                            label="左眼形态", wrap=True, row_count=(12, "dynamic"))
                            left_motion_table = gr.Dataframe(headers=["运动指标", "值"], datatype=["str", "str"],
                                                             label="左眼运动", wrap=True, row_count=(10, "dynamic"))
                        with gr.Column():
                            right_morph_table = gr.Dataframe(headers=["形态指标", "值"], datatype=["str", "str"],
                                                             label="右眼形态", wrap=True, row_count=(12, "dynamic"))
                            right_motion_table = gr.Dataframe(headers=["运动指标", "值"], datatype=["str", "str"],
                                                              label="右眼运动", wrap=True, row_count=(10, "dynamic"))
                    gr.Markdown("### 对比图（左右眼叠加）")
                    dual_morph_gallery = gr.Gallery(label="形态与运动对比", columns=2, height="auto", object_fit="contain")

                with gr.Tab("PLR 参数分析"):
                    with gr.Row():
                        with gr.Column():
                            left_plr_df = gr.Dataframe(headers=["参数", "值", "参考范围"], datatype=["str", "str", "str"],
                                                       label="左眼 PLR", wrap=True, row_count=(15, "dynamic"))
                        with gr.Column():
                            right_plr_df = gr.Dataframe(headers=["参数", "值", "参考范围"], datatype=["str", "str", "str"],
                                                        label="右眼 PLR", wrap=True, row_count=(15, "dynamic"))
                    gr.Markdown("### 对比图（PLR 曲线叠加）")
                    dual_plr_gallery = gr.Gallery(label="PLR 对比", columns=1, height="auto", object_fit="contain")

                with gr.Tab("3D 动态注视模型"):
                    gr.Markdown("> 左右眼各跑 VOGeo-Gaze，显示 overlay 视频 + gaze 静态图 + 对比图。需先为左右眼 CSV 生成 gaze.csv。")
                    with gr.Row():
                        with gr.Column(scale=1):
                            btn_dual_vogeo = gr.Button("加载左右眼 3D 模型", variant="primary")
                            dual_vogeo_info = gr.Textbox(label="状态", lines=4, interactive=False)
                    with gr.Row():
                        with gr.Column(scale=1):
                            left_overlay = gr.Video(label="左眼 VOGeo-Gaze overlay", format="mp4", elem_classes="dual-overlay-video")
                        with gr.Column(scale=1):
                            right_overlay = gr.Video(label="右眼 VOGeo-Gaze overlay", format="mp4", elem_classes="dual-overlay-video")
                    gr.Markdown("### 左眼 gaze 静态图")
                    left_gaze_gallery = gr.Gallery(label="左眼 gaze 图", columns=2, height="auto", object_fit="contain")
                    gr.Markdown("### 右眼 gaze 静态图")
                    right_gaze_gallery = gr.Gallery(label="右眼 gaze 图", columns=2, height="auto", object_fit="contain")
                    gr.Markdown("### 左右眼对比图（theta_h/theta_v 叠加）")
                    dual_gaze_gallery = gr.Gallery(label="双眼对比图", columns=1, height="auto", object_fit="contain")

                with gr.Tab("临床判读"):
                    with gr.Row():
                        with gr.Column():
                            left_interp_df = gr.Dataframe(headers=["判读项", "结果", "依据"],
                                                           datatype=["str", "str", "str"],
                                                           label="左眼临床判读", wrap=True, row_count=(10, "dynamic"))
                            left_ppt_ref_df = gr.Dataframe(
                                headers=["病症", "判读", "依据（本视频）", "判读条件"],
                                datatype=["str", "str", "str", "str"],
                                label="左眼病症对照（含本视频判读）",
                                wrap=True, row_count=(8, "dynamic"),
                            )
                        with gr.Column():
                            right_interp_df = gr.Dataframe(headers=["判读项", "结果", "依据"],
                                                            datatype=["str", "str", "str"],
                                                            label="右眼临床判读", wrap=True, row_count=(10, "dynamic"))
                            right_ppt_ref_df = gr.Dataframe(
                                headers=["病症", "判读", "依据（本视频）", "判读条件"],
                                datatype=["str", "str", "str", "str"],
                                label="右眼病症对照（含本视频判读）",
                                wrap=True, row_count=(8, "dynamic"),
                            )
                    gr.Markdown("### 双眼判读（对比 + 双眼病症）")
                    dual_interp_df = gr.Dataframe(headers=["判读项", "左眼", "右眼", "差值", "判读"],
                                                   datatype=["str"] * 5,
                                                   label="双眼判读", wrap=True, row_count=(10, "dynamic"))
                    dual_interp_summary = gr.Textbox(label="双眼判读摘要", lines=6, interactive=False)
                    gr.Markdown("### PPT 病症分类对照（全部类别）")
                    dual_ppt_ref_df = gr.Dataframe(
                        headers=["病症", "判读", "左右眼对比", "依据"],
                        datatype=["str", "str", "str", "str"],
                        label="双眼病症对照（含本视频判读）",
                        wrap=True,
                        row_count=(10, "dynamic"),
                    )

                with gr.Tab("视频与模型概况"):
                    with gr.Row():
                        with gr.Column():
                            left_summary_df = gr.Dataframe(headers=["指标", "值"], datatype=["str", "str"],
                                                           label="左眼视频与模型概况", wrap=True)
                        with gr.Column():
                            right_summary_df = gr.Dataframe(headers=["指标", "值"], datatype=["str", "str"],
                                                            label="右眼视频与模型概况", wrap=True)

                with gr.Tab("逐帧数据 + 异常帧（前 200 行）"):
                    with gr.Row():
                        with gr.Column():
                            left_frame_df = gr.Dataframe(
                                headers=["frame", "time_sec", "detected", "major", "minor", "cx", "cy", "conf",
                                         "iris_detected", "iris_major", "iris_minor", "pupil_mm"],
                                datatype=["number"] * 12,
                                label="左眼逐帧数据（前 200 行）", wrap=True, row_count=(10, "dynamic"))
                            left_abnormal_summary = gr.Textbox(label="左眼异常统计", lines=2, interactive=False)
                            left_abnormal_df = gr.Dataframe(
                                headers=["frame", "major", "minor", "conf", "原因"],
                                datatype=["number", "number", "number", "number", "str"],
                                label="左眼异常帧", wrap=True, row_count=(8, "dynamic"))
                        with gr.Column():
                            right_frame_df = gr.Dataframe(
                                headers=["frame", "time_sec", "detected", "major", "minor", "cx", "cy", "conf",
                                         "iris_detected", "iris_major", "iris_minor", "pupil_mm"],
                                datatype=["number"] * 12,
                                label="右眼逐帧数据（前 200 行）", wrap=True, row_count=(10, "dynamic"))
                            right_abnormal_summary = gr.Textbox(label="右眼异常统计", lines=2, interactive=False)
                            right_abnormal_df = gr.Dataframe(
                                headers=["frame", "major", "minor", "conf", "原因"],
                                datatype=["number", "number", "number", "number", "str"],
                                label="右眼异常帧", wrap=True, row_count=(8, "dynamic"))

            def _dual_interp(lpl, rpl, l_text, r_text, gaze_diff=None, l_interp_rows=None, r_interp_rows=None):
                def to_dict(params):
                    d = {}
                    for item in params:
                        if len(item) < 2:
                            continue
                        k, v = item[0], item[1]
                        key = str(k).split(" (")[0].strip()
                        try:
                            d[key] = float(v)
                        except Exception:
                            applog.log_here()
                    return d
                dl, dr = to_dict(lpl), to_dict(rpl)

                def get_val(d, prefix):
                    for k, v in d.items():
                        if k.startswith(prefix):
                            return v
                    return None

                rows = []

                def add(name, unit="", warn_th=0.5, warn_text=""):
                    a = get_val(dl, name)
                    b = get_val(dr, name)
                    if a is None or b is None:
                        return
                    d = abs(a - b)
                    if warn_th is not None:
                        if d > warn_th:
                            verdict = f"⚠️ 是（差 {d:.3f}{unit} > {warn_th}{unit}）" + (f" {warn_text}" if warn_text else "")
                        else:
                            verdict = f"✅ 否（差 {d:.3f}{unit} ≤ {warn_th}{unit}）"
                    else:
                        verdict = "—"
                    rows.append([name, f"{a:.3f}{unit}", f"{b:.3f}{unit}", f"{d:.3f}{unit}", verdict])

                # 基线直径（特殊：>1mm 病理性）
                a = get_val(dl, "基线直径")
                b = get_val(dr, "基线直径")
                if a is not None and b is not None:
                    d = abs(a - b)
                    verdict = ("⚠️ 病理性不等大（>1mm；共识：提示 ICP 升高/脑疝，需紧急降颅压）"
                               if d > 1.0 else "✅ 生理性（<1mm；长期稳定、反射灵敏则无需处理）")
                    rows.append(["基线直径", f"{a:.2f}", f"{b:.2f}", f"{d:.2f}", verdict])
                    # 单侧散大（左）
                    if a > 5.0 and (a - b) > 1.0:
                        v = "⚠️ 是（脑疝预警）"
                    elif a > 5.0:
                        v = "⚠️ 散大（双侧差未超 1mm）"
                    else:
                        v = "✅ 否（≤5mm）"
                    rows.append(["单侧散大（左）", f"{a:.2f}", f"{b:.2f}", f"{a-b:+.2f}", v])

                    # 单侧散大（右）
                    if b > 5.0 and (b - a) > 1.0:
                        v = "⚠️ 是（脑疝预警）"
                    elif b > 5.0:
                        v = "⚠️ 散大（双侧差未超 1mm）"
                    else:
                        v = "✅ 否（≤5mm）"
                    rows.append(["单侧散大（右）", f"{a:.2f}", f"{b:.2f}", f"{b-a:+.2f}", v])

                if a > 5.0 and b > 5.0:
                    rows.append(["双侧散大", f"{a:.2f}", f"{b:.2f}", "—", "⚠️ 是（双侧 >5mm）；提示双侧散大，需排查脑疝/药物/代谢"])
                else:
                    rows.append(["双侧散大", f"{a:.2f}", f"{b:.2f}", "—", "✅ 否（非双侧 >5mm）"])
                    # 单侧缩小（左）
                    if 1.0 <= a <= 2.0 and (b - a) > 1.0:
                        v = "⚠️ 是（Horner 瞳孔部分）"
                    elif 1.0 <= a <= 2.0:
                        v = "⚠️ 小瞳（双侧差未超 1mm）"
                    else:
                        v = "✅ 否（不在 1–2mm）"
                    rows.append(["单侧缩小（左）", f"{a:.2f}", f"{b:.2f}", f"{b-a:+.2f}", v])

                    # 单侧缩小（右）
                    if 1.0 <= b <= 2.0 and (a - b) > 1.0:
                        v = "⚠️ 是（Horner 瞳孔部分）"
                    elif 1.0 <= b <= 2.0:
                        v = "⚠️ 小瞳（双侧差未超 1mm）"
                    else:
                        v = "✅ 否（不在 1–2mm）"
                    rows.append(["单侧缩小（右）", f"{a:.2f}", f"{b:.2f}", f"{a-b:+.2f}", v])

                add("收缩幅度", warn_th=0.1, warn_text="单侧收缩幅度差 >0.1")
                add("收缩百分比", "%", warn_th=10.0, warn_text="单侧反应弱")
                add("最大收缩速度", warn_th=0.5, warn_text="单侧神经受损")
                add("NPi 近似评分", warn_th=0.5, warn_text="单侧反应异常")
                add("潜伏期", "s", warn_th=0.1, warn_text="单侧传导慢")
                add("收缩时间", "s", warn_th=0.2, warn_text="单侧收缩慢 >0.2s")
                add("恢复时间", "s", warn_th=0.5, warn_text="单侧恢复慢")

                # 动眼神经麻痹（受累型/回避型，示意）
                # 受累型：单眼散大固定 + 双眼 gaze 差异大
                l_fixed = any("散大固定" in str(r[0]) and "⚠️" in str(r[1]) for r in (l_interp_rows or []))
                r_fixed = any("散大固定" in str(r[0]) and "⚠️" in str(r[1]) for r in (r_interp_rows or []))
                gaze_available = gaze_diff is not None
                gaze_flag = gaze_available and (gaze_diff["th_diff"] > 5.0 or gaze_diff["tv_diff"] > 5.0)
                if (l_fixed or r_fixed) and gaze_flag:
                    rows.append(["动眼神经麻痹（受累型，示意性）", "—", "—", "—", "⚠️ 需排查（散大固定 + gaze 差异 >5°，需同人同步）"])
                elif l_fixed or r_fixed:
                    rows.append(["动眼神经麻痹（受累型，示意性）", "—", "—", "—", "— 需眼动（散大固定，无 gaze 差异数据）"])
                else:
                    rows.append(["动眼神经麻痹（受累型，示意性）", "—", "—", "—", "✅ 否（无散大固定）"])

                # 回避型：瞳孔正常 + gaze 差异大
                l_normal = any("直径正常" in str(r[0]) and "✅" in str(r[1]) for r in (l_interp_rows or []))
                r_normal = any("直径正常" in str(r[0]) and "✅" in str(r[1]) for r in (r_interp_rows or []))
                if l_normal and r_normal and gaze_flag:
                    rows.append(["动眼神经麻痹（回避型，示意性）", "—", "—", "—", "⚠️ 需排查（瞳孔正常 + gaze 差异 >5°，需同人同步）"])
                elif l_normal and r_normal and gaze_available:
                    rows.append(["动眼神经麻痹（回避型，示意性）", "—", "—", "—", "✅ 否（瞳孔正常，gaze 差异 ≤5°）"])
                elif l_normal and r_normal:
                    rows.append(["动眼神经麻痹（回避型，示意性）", "—", "—", "—", "— 需眼动（瞳孔正常，无 gaze 差异数据）"])
                else:
                    rows.append(["动眼神经麻痹（回避型，示意性）", "—", "—", "—", "✅ 否（瞳孔异常）"])

                # 双眼 gaze 差异（示意，需同一人左右眼同步）
                if gaze_diff is not None:
                    th_d = gaze_diff["th_diff"]
                    tv_d = gaze_diff["tv_diff"]
                    th_verdict = "⚠️ 是（差 >5°）" if th_d > 5.0 else "✅ 否（差 ≤5°）"
                    tv_verdict = "⚠️ 是（差 >5°）" if tv_d > 5.0 else "✅ 否（差 ≤5°）"
                    rows.append(["双眼 gaze 水平差", f"{gaze_diff['th_mean_l']:.2f}°", f"{gaze_diff['th_mean_r']:.2f}°", f"{th_d:.2f}°", th_verdict + "（示意，需同人同步）"])
                    rows.append(["双眼 gaze 垂直差", f"{gaze_diff['tv_mean_l']:.2f}°", f"{gaze_diff['tv_mean_r']:.2f}°", f"{tv_d:.2f}°", tv_verdict + "（示意，需同人同步）"])
                else:
                    rows.append(["双眼 gaze 水平差", "—", "—", "—", "无 gaze.csv 或数据不足"])
                    rows.append(["双眼 gaze 垂直差", "—", "—", "—", "无 gaze.csv 或数据不足"])

                summary_lines = []
                warns = [r for r in rows if len(r) >= 5 and str(r[4]).startswith("⚠️")]
                if warns:
                    summary_lines.append(f"双眼判读：检出 {len(warns)} 项异常提示")
                    for w in warns:
                        summary_lines.append(f"- {w[0]}：{w[4]}")
                else:
                    summary_lines.append("双眼判读：未检出明显异常")
                summary_lines.append("")
                summary_lines.append("注：双眼对比基于单眼参数差值，需双眼同步采集才可完全排除视线不一致误差。")

                text = "\n".join(summary_lines)
                return rows, text

            def _run_dual_all(csv_l, csv_r, use_stim_l=False, stim_mark_l=None, use_upload_l=False, stim_file_l=None, use_stim_r=False, stim_mark_r=None, use_upload_r=False, stim_file_r=None):
                if not csv_l or not csv_r:
                    raise gr.Error("请选择左右眼 CSV")
                plot_dir = str(Path(PLOT_DIR))
                lm, lmo, ls, lp = ab.analyze_morphology(csv_l, plot_dir=plot_dir)
                rm, rmo, rs, rp = ab.analyze_morphology(csv_r, plot_dir=plot_dir)
                l_summary, l_frame, l_abnormal, l_text, l_images = ab.analyze_csv(csv_l)
                r_summary, r_frame, r_abnormal, r_text, r_images = ab.analyze_csv(csv_r)
                morph_cmp, _ = ab.compare_eyes_plots(csv_l, csv_r, plot_dir=plot_dir)
                stim_time_l = _resolve_stim_time(use_stim_l, stim_mark_l, use_upload_l, stim_file_l)
                stim_time_r = _resolve_stim_time(use_stim_r, stim_mark_r, use_upload_r, stim_file_r)
                lpl, lplt, lplp = ab.analyze_plr(csv_l, stimulus_time=stim_time_l, plot_dir=plot_dir)
                rpl, rplt, rplp = ab.analyze_plr(csv_r, stimulus_time=stim_time_r, plot_dir=plot_dir)

                # 双眼 gaze 差异（需要两个 gaze.csv，需同一人左右眼同步才有效）
                import vogeo_backend as _vb
                l_gaze_csv = _vb.OUTPUT_DIR / Path(csv_l).stem / "gaze.csv"
                r_gaze_csv = _vb.OUTPUT_DIR / Path(csv_r).stem / "gaze.csv"
                gaze_diff = None
                if l_gaze_csv.exists() and r_gaze_csv.exists():
                    try:
                        import csv as _csv
                        import numpy as _np
                        def _read_gaze(p):
                            th, tv = [], []
                            with open(p) as f:
                                for row in _csv.DictReader(f):
                                    try:
                                        th.append(float(row["theta_h"]))
                                        tv.append(float(row["theta_v"]))
                                    except Exception:
                                        applog.log_here()
                            return _np.array(th), _np.array(tv)
                        l_th, l_tv = _read_gaze(l_gaze_csv)
                        r_th, r_tv = _read_gaze(r_gaze_csv)
                        if len(l_th) >= 10 and len(r_th) >= 10:
                            gaze_diff = {
                                "th_mean_l": float(_np.mean(l_th)),
                                "th_mean_r": float(_np.mean(r_th)),
                                "tv_mean_l": float(_np.mean(l_tv)),
                                "tv_mean_r": float(_np.mean(r_tv)),
                                "th_diff": float(abs(_np.mean(l_th) - _np.mean(r_th))),
                                "tv_diff": float(abs(_np.mean(l_tv) - _np.mean(r_tv))),
                            }
                    except Exception as e:
                        print(f"gaze diff 失败: {e}")
                plr_cmp, _plr_cmp_text = ab.compare_eyes_plr_plots(
                    csv_l, csv_r, plot_dir=plot_dir,
                    stim_time_l=stim_time_l, stim_time_r=stim_time_r,
                )
                l_gaze = __import__("vogeo_backend").OUTPUT_DIR / Path(csv_l).stem / "gaze.csv"
                l_interp, l_interp_s = ab.clinical_interpretation(csv_l, gaze_csv=l_gaze if l_gaze.exists() else None, full=True)
                r_gaze = __import__("vogeo_backend").OUTPUT_DIR / Path(csv_r).stem / "gaze.csv"
                r_interp, r_interp_s = ab.clinical_interpretation(csv_r, gaze_csv=r_gaze if r_gaze.exists() else None, full=True)
                l_ppt_ref = ab.ppt_reference_single(l_interp)
                r_ppt_ref = ab.ppt_reference_single(r_interp)
                dual_interp_rows, dual_interp_text = _dual_interp(lpl, rpl, l_interp_s, r_interp_s, gaze_diff, l_interp, r_interp)
                ppt_ref_dual = ab.ppt_reference_dual(dual_interp_rows)
                warn_lines = []
                for tag, csv_ in (("左眼", csv_l), ("右眼", csv_r)):
                    warn_lines += [f"【{tag}】{w}" for w in ab.reliability_warnings(csv_)]
                summary = f"左眼: {csv_l}\n右眼: {csv_r}\n形态对比 {len(morph_cmp)} 张，PLR 对比 {len(plr_cmp)} 张"
                if warn_lines:
                    summary = "\n".join(warn_lines) + "\n\n" + summary
                l_abn_text = f"异常帧 {len(l_abnormal)} 个" if l_abnormal else "无异常帧"
                r_abn_text = f"异常帧 {len(r_abnormal)} 个" if r_abnormal else "无异常帧"
                return (lm, lmo, rm, rmo, morph_cmp,
                        lpl, rpl, plr_cmp,
                        l_interp, l_ppt_ref, r_interp, r_ppt_ref,
                        dual_interp_rows, dual_interp_text,
                        summary, ppt_ref_dual,
                        l_summary, r_summary,
                        l_frame, l_abn_text, l_abnormal,
                        r_frame, r_abn_text, r_abnormal)

            def _ensure_h264(mp4_path):
                """mp4v → H.264，返回 H.264 路径（存在则跳过）。"""
                import subprocess
                mp4 = Path(mp4_path)
                if not mp4.exists():
                    return None
                h264 = mp4.with_name(mp4.stem + "_h264.mp4")
                if h264.exists():
                    return h264
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
                         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(h264)],
                        check=True, capture_output=True, timeout=300,
                    )
                    return h264
                except Exception as e:
                    print(f"H.264 转码失败: {e}")
                    return mp4

            def _load_dual_vogeo(csv_l, csv_r):
                if not csv_l or not csv_r:
                    raise gr.Error("请选择左右眼 CSV")
                import vogeo_backend as vb
                l_stem = Path(csv_l).stem
                r_stem = Path(csv_r).stem
                l_dir = store.vogeo_dir(l_stem)
                r_dir = store.vogeo_dir(r_stem)
                info_lines = []
                l_overlay = None
                r_overlay = None
                l_h264 = _ensure_h264(l_dir / "overlay.mp4")
                if l_h264 is not None:
                    l_overlay = str(l_h264)
                    info_lines.append(f"左眼 overlay: {l_h264.name}")
                else:
                    info_lines.append(f"左眼 overlay 未生成 — {vb.status_for(l_stem)}")
                r_h264 = _ensure_h264(r_dir / "overlay.mp4")
                if r_h264 is not None:
                    r_overlay = str(r_h264)
                    info_lines.append(f"右眼 overlay: {r_h264.name}")
                else:
                    info_lines.append(f"右眼 overlay 未生成 — {vb.status_for(r_stem)}")
                l_plots, r_plots = [], []
                if (l_dir / "gaze.csv").exists():
                    l_plots = vb.plot_gaze(l_dir / "gaze.csv", plot_dir=vb.OUTPUT_DIR / l_stem)
                if (r_dir / "gaze.csv").exists():
                    r_plots = vb.plot_gaze(r_dir / "gaze.csv", plot_dir=vb.OUTPUT_DIR / r_stem)

                # 对比图
                dual_plots = []
                if (l_dir / "gaze.csv").exists() and (r_dir / "gaze.csv").exists():
                    dual_plots = ab.compare_eyes_gaze_plots(
                        l_dir / "gaze.csv", r_dir / "gaze.csv",
                        plot_dir=str(Path(PLOT_DIR)),
                    )

                info_lines.append(f"左眼 gaze 图: {len(l_plots)} 张")
                info_lines.append(f"右眼 gaze 图: {len(r_plots)} 张")
                info_lines.append(f"双眼对比图: {len(dual_plots)} 张")
                return "\n".join(info_lines), l_overlay, r_overlay, l_plots, r_plots, dual_plots

            btn_dual_vogeo.click(
                _load_dual_vogeo,
                inputs=[dual_left_dd, dual_right_dd],
                outputs=[dual_vogeo_info, left_overlay, right_overlay,
                         left_gaze_gallery, right_gaze_gallery, dual_gaze_gallery],
            )

            refresh_dual_btn.click(
                lambda: (gr.update(choices=ab.list_csv_files()), gr.update(choices=ab.list_csv_files())),
                outputs=[dual_left_dd, dual_right_dd],
            )

            use_stim_l.change(
                lambda x: (gr.update(visible=bool(x)), gr.update(visible=bool(x)), gr.update(visible=bool(x))),
                inputs=[use_stim_l],
                outputs=[stim_marks_select_l, refresh_stim_marks_l, use_upload_l],
            )
            use_upload_l.change(
                lambda x: (gr.update(visible=not x), gr.update(visible=not x), gr.update(visible=bool(x))),
                inputs=[use_upload_l],
                outputs=[stim_marks_select_l, refresh_stim_marks_l, stim_file_l],
            )
            use_stim_r.change(
                lambda x: (gr.update(visible=bool(x)), gr.update(visible=bool(x)), gr.update(visible=bool(x))),
                inputs=[use_stim_r],
                outputs=[stim_marks_select_r, refresh_stim_marks_r, use_upload_r],
            )
            use_upload_r.change(
                lambda x: (gr.update(visible=not x), gr.update(visible=not x), gr.update(visible=bool(x))),
                inputs=[use_upload_r],
                outputs=[stim_marks_select_r, refresh_stim_marks_r, stim_file_r],
            )

            refresh_stim_marks_l.click(lambda: gr.update(choices=list_stim_marks()), outputs=[stim_marks_select_l])
            refresh_stim_marks_r.click(lambda: gr.update(choices=list_stim_marks()), outputs=[stim_marks_select_r])

            btn_dual_analyze.click(
                _run_dual_all,
                inputs=[dual_left_dd, dual_right_dd, use_stim_l, stim_marks_select_l, use_upload_l, stim_file_l, use_stim_r, stim_marks_select_r, use_upload_r, stim_file_r],
                outputs=[left_morph_table, left_motion_table, right_morph_table, right_motion_table,
                         dual_morph_gallery,
                         left_plr_df, right_plr_df, dual_plr_gallery,
                         left_interp_df, left_ppt_ref_df, right_interp_df, right_ppt_ref_df,
                         dual_interp_df, dual_interp_summary,
                         dual_summary, dual_ppt_ref_df,
                         left_summary_df, right_summary_df,
                         left_frame_df, left_abnormal_summary, left_abnormal_df,
                         right_frame_df, right_abnormal_summary, right_abnormal_df],
            )

            def _export_dual_pdf(csv_l, csv_r, use_stim_l=False, stim_mark_l=None, use_upload_l=False, stim_file_l=None,
                                 use_stim_r=False, stim_mark_r=None, use_upload_r=False, stim_file_r=None):
                if not csv_l or not csv_r:
                    raise gr.Error("请选择左右眼 CSV")
                stim_l = _resolve_stim_time(use_stim_l, stim_mark_l, use_upload_l, stim_file_l)
                stim_r = _resolve_stim_time(use_stim_r, stim_mark_r, use_upload_r, stim_file_r)
                try:
                    path = pe.export_dual_eye_pdf(csv_l, csv_r, stimulus_time_l=stim_l, stimulus_time_r=stim_r)
                    return f"✅ 已生成: {path}", path
                except Exception as e:
                    return f"❌ 导出失败: {e}", None

            btn_export_dual_pdf.click(
                _export_dual_pdf,
                inputs=[dual_left_dd, dual_right_dd, use_stim_l, stim_marks_select_l, use_upload_l, stim_file_l,
                        use_stim_r, stim_marks_select_r, use_upload_r, stim_file_r],
                outputs=[dual_pdf_status, dual_pdf_file],
            )

        # ============ 5b. ROI 裁剪 ============
        with gr.Tab("ROI 裁剪", elem_classes="big-tab"):
            gr.Markdown("### ROI 裁剪（MediaPipe 眼睛区域）")
            gr.Markdown("> 对整脸视频先裁眼睛 ROI，输出裁剪视频 + ROI 坐标 CSV + 画框预览。检测失败时沿用上一帧 ROI。")
            with gr.Tabs():
                with gr.Tab("图片 ROI"):
                    with gr.Row():
                        with gr.Column():
                            roi_img_in = gr.Image(type="pil", label="输入图片")
                            roi_img_which = gr.Radio(choices=["左眼", "右眼", "双眼"], value="双眼", label="裁剪哪只眼")
                            roi_img_padding = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="ROI 外扩比例")
                            roi_img_btn = gr.Button("检测 ROI", variant="primary")
                        with gr.Column():
                            roi_img_debug = gr.Image(label="原图 + ROI 框")
                            roi_img_left = gr.Image(label="左眼裁剪")
                            roi_img_right = gr.Image(label="右眼裁剪")
                            roi_img_info = gr.Textbox(label="ROI 坐标", lines=6)

                    def _roi_image(img_pil, which_label, padding):
                        if img_pil is None:
                            raise gr.Error("请先上传图片")
                        which = {"左眼": "left", "右眼": "right", "双眼": "both"}[which_label]
                        frame = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
                        rois = rb.detect_eye_roi_image(frame, padding=float(padding), which=which)
                        debug = rb.draw_roi(frame, rois)
                        left_img = right_img = None
                        lines = []
                        if which in ("left", "both") and len(rois) >= 1:
                            crop, _ = rb.crop_roi(frame, rois[0])
                            left_img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            lines.append(f"左眼 ROI: {rois[0]}")
                        if which in ("right", "both"):
                            idx = 1 if which == "both" else 0
                            if idx < len(rois):
                                crop, _ = rb.crop_roi(frame, rois[idx])
                                right_img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                lines.append(f"右眼 ROI: {rois[idx]}")
                        if not lines:
                            lines.append("未检测到 ROI（可能是半脸/眼部特写，MediaPipe 不适用）")
                        return (cv2.cvtColor(debug, cv2.COLOR_BGR2RGB), left_img, right_img, "\n".join(lines))

                    roi_img_btn.click(_roi_image,
                                      inputs=[roi_img_in, roi_img_which, roi_img_padding],
                                      outputs=[roi_img_debug, roi_img_left, roi_img_right, roi_img_info])

                with gr.Tab("视频 ROI"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            roi_dataset_dd = gr.Dropdown(choices=list(DATASET_DIRS.keys()), value="OpenEDS", label="数据集")
                            roi_refresh = gr.Button("刷新视频列表")
                            roi_video_dd = gr.Dropdown(
                                choices=list_videos("OpenEDS"),
                                value=(list_videos("OpenEDS")[0] if list_videos("OpenEDS") else None),
                                label="选择视频",
                            )
                            roi_preview = gr.Video(label="原视频", interactive=False)
                            roi_which = gr.Radio(choices=["左眼", "右眼", "双眼"], value="双眼", label="裁剪哪只眼")
                            roi_padding = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="ROI 外扩比例")
                            roi_max_frames = gr.Slider(VIDEO_MAX_FRAMES_MIN, VIDEO_MAX_FRAMES_MAX, value=VIDEO_MAX_FRAMES, step=10, label="最多处理帧数")
                            roi_btn = gr.Button("开始 ROI 裁剪", variant="primary")
                        with gr.Column(scale=2):
                            roi_summary = gr.Textbox(label="统计", lines=6, interactive=False)
                            roi_left_video = gr.Video(label="左眼 ROI 视频", format="mp4")
                            roi_right_video = gr.Video(label="右眼 ROI 视频", format="mp4")
                            roi_debug_video = gr.Video(label="画框预览", format="mp4")
                            roi_csv_file = gr.File(label="ROI 坐标 CSV")

                    def _roi_list(ds):
                        vs = list_videos(ds)
                        return gr.update(choices=vs, value=(vs[0] if vs else None))

                    def _roi_load(ds, name):
                        if not ds or not name:
                            return None
                        d = DATASET_DIRS.get(ds)
                        if not d:
                            return None
                        src = os.path.join(d, name)
                        if not os.path.exists(src):
                            return None
                        dst = os.path.join(COPY_DIR, f"roi_{ds}_{name}")
                        if not os.path.exists(dst):
                            shutil.copy(src, dst)
                        return dst

                    def _run_roi_video(ds, video_name, which_label, padding, max_frames, progress=gr.Progress()):
                        if not ds or not video_name:
                            raise gr.Error("请先选择数据集和视频")
                        d = DATASET_DIRS.get(ds)
                        if not d:
                            raise gr.Error(f"未知数据集: {ds}")
                        path = os.path.join(d, video_name)
                        if not os.path.exists(path):
                            raise gr.Error(f"视频不存在: {path}")
                        which = {"左眼": "left", "右眼": "right", "双眼": "both"}[which_label]
                        out_dir = Path(OUTPUT_DIR) / "roi_results"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        stem = Path(video_name).stem

                        rois, stats = rb.detect_eye_roi_video(path, which=which, padding=float(padding),
                                                              fallback="prev", max_frames=int(max_frames),
                                                              progress=progress)
                        csv_path = out_dir / f"{stem}_roi.csv"
                        rb.write_roi_csv(rois, str(csv_path))

                        left_out = right_out = None
                        debug_out = None
                        if which in ("left", "both"):
                            left_out, debug_out, _ = rb.crop_video_with_roi(
                                path, rois, eye="left",
                                out_path=str(out_dir / f"{stem}_roi_left.mp4"),
                                debug_path=str(out_dir / f"{stem}_roi_debug.mp4"),
                                max_frames=int(max_frames), progress=progress)
                        if which in ("right", "both"):
                            right_out, dbg_r, _ = rb.crop_video_with_roi(
                                path, rois, eye="right",
                                out_path=str(out_dir / f"{stem}_roi_right.mp4"),
                                debug_path=None if debug_out else str(out_dir / f"{stem}_roi_debug.mp4"),
                                max_frames=int(max_frames), progress=progress)
                            if debug_out is None:
                                debug_out = dbg_r

                        summary = (f"视频: {video_name}\n"
                                   f"帧数: {stats['frames']}\n"
                                   f"左眼检出: {stats['detect_left']}，沿用: {stats['fallback_left']}\n"
                                   f"右眼检出: {stats['detect_right']}，沿用: {stats['fallback_right']}\n"
                                   f"CSV: {csv_path}")
                        return summary, left_out, right_out, debug_out, str(csv_path)

                    roi_dataset_dd.change(_roi_list, inputs=[roi_dataset_dd], outputs=[roi_video_dd])
                    roi_video_dd.change(_roi_load, inputs=[roi_dataset_dd, roi_video_dd], outputs=[roi_preview])
                    roi_refresh.click(lambda ds: _roi_list(ds), inputs=[roi_dataset_dd], outputs=[roi_video_dd])
                    roi_btn.click(_run_roi_video,
                                  inputs=[roi_dataset_dd, roi_video_dd, roi_which, roi_padding, roi_max_frames],
                                  outputs=[roi_summary, roi_left_video, roi_right_video, roi_debug_video, roi_csv_file])

        # ============ 6. 刺激标定 ============
        # ============ 6. 刺激标定 ============
        # ============ 6. 刺激标定 ============
        # ============ 6. 刺激标定 ============
        # ============ 6. 刺激标定 ============
        with gr.Tab("刺激标定", elem_classes="big-tab"):
            gr.Markdown(
                "### 刺激标定（光刺激时刻检测）\n\n"
                "> 注：一个视频只取**最后一次标记**的结果作为 PLR 刺激时刻。"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    stim_dataset_dd = gr.Dropdown(choices=list(DATASET_DIRS.keys()), value="OpenEDS", label="数据集")
                    stim_refresh = gr.Button("刷新视频列表")
                    stim_video_dd = gr.Dropdown(
                        choices=list_videos("OpenEDS"),
                        value=(list_videos("OpenEDS")[0] if list_videos("OpenEDS") else None),
                        label="选择视频",
                    )
                    stim_method = gr.Radio(
                        choices=["手动标定（点标记）", "亮度突变检测", "光流突变检测"],
                        value="手动标定（点标记）",
                        label="检测方法",
                    )
                    stim_manual_time = gr.Number(value=0, label="手动输入时刻 (s)，可选")
                with gr.Column(scale=2):
                    stim_result = gr.Textbox(label="标定结果", lines=6, interactive=False)

            with gr.Row():
                stim_marks_dd = gr.Dropdown(
                    choices=list_stim_marks(),
                    label="已有标定文件（outputs/stimulus_marks）",
                    scale=3,
                    interactive=True,
                )
                with gr.Column(scale=1, min_width=80):
                    refresh_marks_btn = gr.Button("刷新列表", scale=1)
                    delete_marks_btn = gr.Button("删除选中文件", variant="stop", scale=1)
            stim_marks_info = gr.Textbox(label="标定文件内容", lines=8, interactive=False)

            stim_video = gr.Video(
                label="原视频（播放/暂停后点下方「标记当前时刻」）",
                interactive=False,
                elem_id="stim-video",
            )
            stim_mark_html = gr.HTML(
                value="",
                html_template="""
                <div style="margin-top:8px;">
                  <button id="mark-btn" style="padding:6px 16px; cursor:pointer; background:#2563eb; color:#fff; border:none; border-radius:6px;">📍 标记当前时刻</button>
                  <span id="mark-info" style="margin-left:12px; font-size:14px; color:#666;"></span>
                </div>
                """,
                js_on_load="""
                const btn = element.querySelector('#mark-btn');
                const info = element.querySelector('#mark-info');
                if (btn) {
                  btn.addEventListener('click', () => {
                    const target = document.querySelector('#stim-video video');
                    if (!target) { info.textContent = '找不到视频'; return; }
                    const t = target.currentTime;
                    info.textContent = '已标记: ' + t.toFixed(2) + 's';
                    props.value = String(t);
                  });
                }
                """,
            )

            stim_state = gr.State(0.0)

            def _stim_list(ds):
                vs = list_videos(ds)
                return gr.update(choices=vs, value=(vs[0] if vs else None))

            def _stim_load(ds, name):
                if not ds or not name:
                    return None, ""
                d = DATASET_DIRS.get(ds)
                if not d:
                    return None, ""
                src_path = os.path.join(d, name)
                if not os.path.exists(src_path):
                    return None, ""
                dst = os.path.join(COPY_DIR, f"stim_{ds}_{name}")
                if not os.path.exists(dst):
                    shutil.copy(src_path, dst)
                marks_dir = Path(STIMULUS_DIR)
                mark_file = marks_dir / f"{Path(name).stem}.json"
                if mark_file.exists():
                    try:
                        import json
                        with open(mark_file) as f:
                            data = json.load(f)
                        n = len(data.get("marks", []))
                        cur = data.get("current")
                        info = f"已有标定: {n} 次，最新 {cur:.2f}s（{mark_file.name}）"
                    except Exception:
                        info = ""
                else:
                    info = "该视频暂无标定"
                return dst, info

            def _on_mark(mark_value, cur, ds, name):
                import json
                from datetime import datetime
                try:
                    t = float(mark_value)
                except Exception:
                    t = 0.0
                if t <= 0:
                    return "⚠️ 标记时刻无效", t
                if not name:
                    return f"已标记: {t:.2f}s（未选择视频，未保存）", t

                marks_dir = Path(STIMULUS_DIR)
                marks_dir.mkdir(parents=True, exist_ok=True)
                out = marks_dir / f"{Path(name).stem}.json"
                data = {"video": name, "dataset": ds, "marks": [], "current": None}
                if out.exists():
                    try:
                        with open(out) as f:
                            data = json.load(f)
                    except Exception:
                        applog.log_here()
                if "marks" not in data or not isinstance(data["marks"], list):
                    data["marks"] = []
                now_str = datetime.now().isoformat(timespec="seconds")
                if data["marks"] and abs(float(data["marks"][-1].get("time", -1)) - float(t)) < 1e-6:
                    data["current"] = float(t)
                    data["video"] = name
                    data["dataset"] = ds
                    with open(out, "w") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    return f"✅ 已标记: {t:.2f}s（重复，跳过；累计 {len(data['marks'])} 次）", t
                data["marks"].append({
                    "time": float(t),
                    "source": "video_mark",
                    "timestamp": now_str,
                })
                data["current"] = float(t)
                data["video"] = name
                data["dataset"] = ds
                with open(out, "w") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                return f"✅ 已标记并保存: {t:.2f}s（累计 {len(data['marks'])} 次）", t

            def _delete_stim_mark(name):
                if not name:
                    return "⚠️ 请先选择要删除的标定文件", gr.update()
                import os
                p = Path(STIMULUS_DIR) / name
                if not p.exists():
                    return f"⚠️ 文件不存在: {p}", gr.update()
                try:
                    os.remove(p)
                    return f"✅ 已删除: {p}", gr.update(choices=list_stim_marks(), value=None)
                except Exception as e:
                    return f"❌ 删除失败: {e}", gr.update()

            stim_dataset_dd.change(_stim_list, inputs=[stim_dataset_dd], outputs=[stim_video_dd])
            stim_video_dd.change(_stim_load, inputs=[stim_dataset_dd, stim_video_dd], outputs=[stim_video, stim_result])
            stim_refresh.click(lambda ds: _stim_list(ds), inputs=[stim_dataset_dd], outputs=[stim_video_dd])
            stim_mark_html.change(_on_mark, inputs=[stim_mark_html, stim_state, stim_dataset_dd, stim_video_dd], outputs=[stim_result, stim_state])

            refresh_marks_btn.click(lambda: gr.update(choices=list_stim_marks()), outputs=[stim_marks_dd])
            delete_marks_btn.click(
                _delete_stim_mark,
                inputs=[stim_marks_dd],
                outputs=[stim_result, stim_marks_dd],
            )
            stim_marks_dd.change(load_stim_mark, inputs=[stim_marks_dd], outputs=[stim_marks_info])

if __name__ == "__main__":
    demo.queue(max_size=8, default_concurrency_limit=2)   # 长任务排队，不再把界面卡死
    demo.launch(
        server_name=SERVER_HOST, server_port=SERVER_PORT,
        allowed_paths=allowed_paths(),
        show_error=True,
        css=CSS,
        head=HEAD,
    )
