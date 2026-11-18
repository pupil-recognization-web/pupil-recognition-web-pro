# 瞳孔识别平台（v2）

Gradio 单页 Web 平台：**瞳孔/虹膜分割 → 逐帧数据 → PLR 参数 → 临床判读 → 报告与 3D 注视**，
面向 OpenEDS / LPW 数据集与自采视频。

本目录是基于 `pupil_recognition_web` 的**润色版**：交互与输出口径保持不变，重点在
配置集中化、路径可迁移、缺陷修复与文档补齐。改动清单见 [CHANGELOG.md](CHANGELOG.md)，
数据流与算法口径见 [ARCHITECTURE.md](ARCHITECTURE.md)。

---


## 相关仓库

| 仓库 | 内容 |
|---|---|
[`pupil-recognition-web-pro`](https://github.com/pupil-recognization-web/pupil-recognition-web-pro) | 专业版（训练/推理/去模糊/光流/ROI/3D） |
[`pupil-recognition-web-lite`](https://github.com/pupil-recognization-web/pupil-recognition-web-lite) | 轻量版（三步引导，面向医生） |

**模型权重已随本仓库提交**，`git clone` 后即可直接运行，无需另外下载。
数据集（openEDS / LPW 录像）体积过大不入库，用环境变量指向本地路径：
`PUPIL_OPENEDS_DIR`、`PUPIL_LPW_DIR`。

## 快速开始

```bash
# 1) 依赖（建议直接用现成的 conda 环境）
conda activate yolo            # 本机为 /mnt/backup/miniconda3/envs/yolo
pip install -r requirements.txt

# 2) 启动
python app.py                  # 默认 0.0.0.0:8080，可用 PUPIL_PORT 改

# 3) 不启动界面也能自检（推荐先跑）
python tools/selfcheck.py
```

`tools/selfcheck.py` 会检查依赖、数据集目录、模型权重指纹、以及 9 份历史 CSV 的
「解析 → 形态/运动 → PLR → 判读」整条链路，输出 `FAIL/WARN` 分级结论。

---

## 配置：不再有写死的绝对路径

所有路径与阈值集中在 [config.py](config.py)，可用环境变量整体搬迁（换机器不用改代码）：

| 变量 | 作用 | 默认 |
|---|---|---|
`PUPIL_YOLO_ROOT` | ultralytics / 数据集 / 训练脚本所在工程根 | `/home/qingque/qingque_yolo` |
`PUPIL_VOGEO_ROOT` | VOGeo-Gaze 仓库根 | `$PUPIL_YOLO_ROOT/VOGeo-Gaze` |
`PUPIL_OPENEDS_DIR` | OpenEDS 视频目录 | `$PUPIL_YOLO_ROOT/openEDS/openEDS/videos` |
`PUPIL_LPW_DIR` | LPW 视频目录 | `$PUPIL_YOLO_ROOT/LPW/videos` |
`PUPIL_DEBLUR_DIR` | 去模糊专家权重根 | `$PUPIL_YOLO_ROOT/LPW/models` |
`PUPIL_MODELS_DIR` | 本项目模型目录 | `<项目>/models` |
`PUPIL_OUTPUT_DIR` | 产出目录 | `<项目>/outputs` |
`PUPIL_PORT` / `PUPIL_HOST` | 服务地址 | `8080` / `0.0.0.0` |
`PUPIL_DEVICE` | VOGeo-Gaze 设备 | `cuda` |

阈值同处一地：`CONF_THRESHOLD`、`IOU_THRESHOLD`、`IMG_SIZE`、`VIDEO_MAX_FRAMES`
（界面滑块上下限与后端共用一份，不会再出现“界面能拉到 2000、后端只处理 600”）、
`CORNEA_MM`（像素→毫米的角膜参照）、`PLR_SMOOTH_WINDOW`、
`RELIABLE_DETECT_RATE` 等。

---

## 八个 Tab

| Tab | 做什么 | 关键后端 |
|---|---|---|
瞳孔分割推理 | 图片 / 视频 / 数据集三种来源，PLR 模式出 12 列 CSV，可勾光流补帧与 3D 注视 | `inference.py` |
瞳孔分割训练 | 调 `scripts/yolo_training/*.py`，输出目录带 `_web` 后缀与手动训练隔离 | `train_backend.py` |
瞳孔运动特征提取 | 只跑 DeblurUNet 出清晰视频；「智能」按光流统计量自动选专家 | `motion_backend.py`、`deblur_backend.py` |
运动光流特征提取 | StatsSeqTransformer 训练（断点续传）+ 24 维光流统计与专家推荐 | `train_backend.py`、`flow_stats_backend.py` |
瞳孔分割结果分析 | 6 个子 Tab：形态/运动、14 项 PLR、3D 注视、临床判读、视频与模型概况、逐帧与异常帧 | `analysis_backend.py`、`vogeo_backend.py` |
双眼瞳孔分割结果分析 | 左右眼各一套输入，形态/PLR/判读/gaze 全对比 | `analysis_backend.py` |
ROI 裁剪 | MediaPipe FaceLandmarker 478 点 → 眼睛 ROI 裁剪与坐标 CSV | `roi_backend.py` |
刺激标定 | 播放视频点「标记当前时刻」写 `stimulus_marks/<视频>.json`，只取最后一次 | `app.py` |

---

## 目录结构

```
pupil_recognition_web_v2/
├── app.py                 # 界面与回调（约 1800 行）
├── config.py              # 唯一配置源（路径 / 阈值 / 注册表 / UI 边界）
├── inference.py           # 模型加载、单帧/逐帧推理、CSV、H.264、光流补帧
├── plr_core.py            # PLR 参数唯一实现（网页 / PDF / CLI 共用）
├── analysis_backend.py    # 形态运动、PLR 展示、判读、双眼对比、质量门
├── pdf_export.py          # HTML → wkhtmltopdf 报告（单眼/双眼）
├── deblur_backend.py      # DeblurUNet / V2（CBAM + EPA）
├── motion_backend.py      # 去模糊视频与运动特征
├── train_backend.py       # 两套训练的启停与日志解析
├── flow_stats_backend.py  # 24 维 Farneback 统计、专家推荐
├── flow_fill_backend.py   # 光流补帧唯一实现（其余模块 import 它）
├── roi_backend.py         # MediaPipe ROI 检测与裁剪
├── vogeo_backend.py       # VOGeo-Gaze 调用与 gaze 出图
├── model_info.py          # 权重指纹 / PLR 权重一致性检查
├── fill_clear_masks.py    # 命令行：清晰视频分割 + 补帧
├── fill_from_csv.py       # 命令行：按已有 CSV 补帧
├── plr_params.py          # 命令行：PLR 参数提取
├── standalone_infer.py    # 命令行：单图推理（含 Segment cv4 补丁）
├── batch_deblur.py        # 命令行：整目录批量去模糊
├── plot_compare.py        # 命令行：三模型训练曲线对比
├── tools/selfcheck.py     # 离线自检
└── models/{baseline,mac,strip}/{model,plr_model}.{pt,yaml}
```

产物统一落在 `outputs/`：`segmentation_results/`（CSV + 视频）、`plots/`、
`pdf_reports/`、`stimulus_marks/`、`deblur_videos/`、`motion/`、`vogeo_results/`（含 `vogeo.log`）、`roi_results/`。

---

## 三个必须知道的坑

1. **PLR 模式的“模型选择”是假的**。三份 `plr_model.pt` 是同一个文件（md5 相同），
   `plr_model.yaml` 也逐字节相同。勾 PLR 后无论选 baseline / mac / strip，加载的都是
   同一份双类权重。本版不再静默：勾 PLR 时界面会直接显示提示（`model_info.plr_mode_note`），
   `tools/selfcheck.py` 也会报出来。要真正区分，需要重新训练各自的 PLR 权重。
2. **检出率决定结论可信度**。低检出率的 CSV 仍能算出 NPi，但没有意义。
   本版在「概览」和双眼摘要里前置可靠性提示（`analysis_backend.reliability_warnings`），
   阈值 `RELIABLE_DETECT_RATE`（默认 30%）。历史数据里
   `2388_roi_right_*` 只有 1.3%，`1_1_*` 只有 69%，都会被标注。
3. **像素→毫米靠 11.5 mm 角膜参照**。真实设备请按 `CORNEA_MM` 标定，
   否则 mm 口径的 PLR 参数会整体偏。
4. **3D 注视模型是外部仓库 + 子进程**。它依赖 `$PUPIL_VOGEO_ROOT` 下的
   `VOGeo-Gaze` 仓库（`python -m vogeo_gaze`）。旧版有三个坑，本版都已修：
   子进程写死 `"python"`（PATH 里不是当前环境就静默失败）、写死 `--device cuda`
   （无 GPU 必失败）、失败只印在控制台。现在：用 `sys.executable`、设备默认交给
   VOGeo 自判（`PUPIL_DEVICE` 可强制）、stdout/stderr 落到
   `outputs/vogeo_results/<stem>/vogeo.log`、结果状态直接回填到界面。
   另外「3D 动态注视模型」子 Tab 新增「（重新）生成 3D 模型」按钮，
   可以只对已有 CSV 重建（源视频从 `<stem>.run.json` 反查），不必重跑分割推理。

---

## PLR 分析模式（v2 / legacy）

默认走 **v2**：刺激前基线窗 + 刺激后 0–2 s 收缩窗 + 「有没有引出收缩」判定 + 质量标记
（帧率、潜伏期分辨率、剔除闭眼帧数）。依据见 [CLINICAL.md](CLINICAL.md)。

```bash
PUPIL_PLR_MODE=v2       # 默认
PUPIL_PLR_MODE=legacy   # 旧口径（全序列 argmin），仅用于复现历史数字
```

为什么默认换了：旧口径在现有 8 份有效 CSV 里有 4 份的"最小点"落在刺激**之前**，
导致 MCV、潜伏期恒为 0，NPi 与判读随之失真；v2 会把这类情况标成
"未引出（需复核）"，而不是给一个看起来正常的数。

阈值都在 [config.py](config.py)：`PLR_BASE_WINDOW`、`PLR_RESPONSE_WINDOW`、
`PLR_SMOOTH_SEC`、`PLR_MIN_RESPONSE_PCT`、`PLR_LATENCY_DROP`、`PLR_MCV_REF_MM_S`。

## 已知限制

- 补帧用 CSV 的椭圆重建 mask，而不是原始 YOLO mask：精度上限受 `fitEllipse` 限制。
- 补帧阶段为控内存改为「中间视频 + 流式读回」，与旧版相比多一代压缩（视频平均绝对差
  约 2%，只影响可视化，不影响 CSV）。
- `analysis_backend` 的历史函数（`analyze_csv` 等）仍会写 `outputs/plots`，
  批量跑时注意并发写同一文件名。
- `wkhtmltopdf` 是系统依赖，不在 `requirements.txt` 里。

---

## 免责声明

自动化分析工具，仅供研究与决策支持，不构成临床诊断。

## 运维与自检（改完代码先跑这个）

```bash
./tools/healthcheck.sh      # 端口 / 内存 / 进程 RSS / 日志大小 / 最近 Traceback
./tools/smoke_test.py       # 冒烟测试（含 tools/baseline.json 数值回归）
./tools/check.sh            # 上面两个一起跑；加 --stop 会先停服务腾内存
./start.sh / ./stop.sh      # 起停
```

- **数值回归**：`tools/baseline.json` 记录了每份 CSV 的临床数值。
  如果改动后 smoke test 报"与基线不一致"，说明临床口径被改了——
  确认是有意为之再执行 `python tools/make_baseline.py --update` 更新基线。
- **日志**：`app.log` 超过 5 MB 自动滚动为 `app.log.1`；以前被静默吞掉的异常现在都会写进去。
- **开机自启**：项目根目录有 `install_service.sh`（两个服务一起装，需 sudo）。

## GPU 加速（可选）

这台机器有 RTX 5060（8 GB），两个服务默认自动用 GPU：

```bash
./start.sh                         # 自动：有 CUDA 就用
PUPIL_YOLO_DEVICE=cpu ./start.sh   # 强制 CPU（GPU 被别的任务占满时）
```

实测：分割推理 66 → 14 ms/帧，端到端 293 → 97 ms/帧，数值与 CPU 一致（直径差 <0.2 px）。
去模糊也会自动走 GPU（95 ms/帧）。

显存只剩 2 G 左右时（被别的任务占用），建议先用 `PUPIL_YOLO_DEVICE=cpu` 跑推理，
不要在此时启动训练。
