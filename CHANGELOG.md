# 变更记录

## v2（2026-11-17）润色版

原则：**交互与数值口径不变，先修缺陷与结构**。每条都标注了验证方式。
回归基线：9 份历史 CSV 的 PLR 14 项参数 + 临床判读，与旧版**逐字节一致**。

### 修复

1. **PLR 权重名不副实（静默失效）**
   三份 `plr_model.pt` md5 相同（`64453279…`），三份 `plr_model.yaml` 也逐字节相同 ——
   选 mac/strip 再勾 PLR 实际仍加载同一份双类权重。
   处理：新增 `model_info.py` 算指纹，勾 PLR 时界面直接提示，`selfcheck` 也会报；
   行为不擅自改动（换模型需要重训权重）。
   *验证：`model_info.plr_is_shared() == True`，界面提示可见。*

2. **刺激标定文件前缀误匹配**
   `_load_stimulus_time` 用 `csv_stem.startswith(video_stem)`，`1_1` 会吃到 `1_10` 的标定。
   改为「完全相同 / stem+分隔符 开头」，多候选取最长匹配，精确匹配优先。
   *验证：S_1 仍准确命中 `stimulus_marks/S_1.json`（1.49 s），S_0/S_6/S_7 仍走自动检测。*

3. **低检出率数据仍出结论**
   `2388_roi_right_*` 检出率 1.3%、`1_1_*` 69%，界面上照旧给 NPi。
   新增 `analysis_backend.reliability_warnings`，单眼概览与双眼摘要前置提示，
   阈值集中在 `config.RELIABLE_DETECT_RATE`。
   *验证：selfcheck 对这 9 份 CSV 输出的 WARN 与预期一致。*

4. **补帧内存无上限（可 OOM）**
   旧版同时把整段 `frames` 与 `overlays` 留在内存（600 帧 720p ≈ 3 GB）。
   改为「overlay 直接落盘 → 补帧阶段流式读回」，峰值只与单段漏检长度有关
   （`FILL_CHUNK=64`）。
   *验证：30 帧真实视频，`flow_filled=1`、输出 30 帧、CSV 与旧版 diff 为空；
   已知代价为多一代压缩（逐帧平均绝对差 ≈ 4.8/255）。*

5. **补帧实现三份拷贝**
   `inference.py`、`fill_from_csv.py`、`fill_clear_masks.py` 各一份 warp/融合代码。
   合并到 `flow_fill_backend.py`，其余 import；`fill_masks` 新增可选的 `grays` 参数，
   避免重复 `cvtColor`。

6. **PLR 计算两份实现**
   `analysis_backend.analyze_plr` 与 `plr_params.py` 公式各自维护。
   抽出 `plr_core.py`，两个入口只负责排版；补充 `stimulus_source` 以保留
   “手动 / 标定文件 / 自动”标注。
   *验证：9 份 CSV 的参数表、判读表、来源标注与旧版逐字节相同。*

7. **`VIDEO_MAX_FRAMES=600` 与界面 2000 不一致**
   滑块上下限与默认值改为读 `config.VIDEO_MAX_FRAMES_MIN/MAX/VIDEO_MAX_FRAMES`。

8. **硬编码绝对路径**
   `app.py`、`train_backend.py`、`deblur_backend.py`、`motion_backend.py`、
   `roi_backend.py`、`pdf_export.py`、`vogeo_backend.py`、`standalone_infer.py`、
   `batch_deblur.py`、`plot_compare.py` 全部改走 `config.py`，支持环境变量整体搬迁。

9. **死代码**
   删除 `vogeo_backend.gaze_html_player` / `plot_gaze_window`（从未挂到界面），
   以及 `app._load_vogeo` 里算完就丢的 `gaze_html_data`；
   `_load_vogeo` 早退分支原先只返回 3 个值（与 4 个输出不匹配），已修。

10. **三处重复的刺激解析**
    `_run_all_analysis`、`_export_single_pdf`、双眼分析与双眼 PDF 四段 if/elif
    合并为 `app._resolve_stim_time`。

### 新增

- `tools/selfcheck.py`：离线自检（依赖 / 目录 / 权重指纹 / 9 份 CSV 全链路），
  `FAIL` 决定退出码，`WARN` 只提示。
- `model_info.py`：MD5 指纹、PLR 权重一致性、界面提示文案。
- `plr_core.py`：PLR 唯一实现 + 参数表/字典生成。
- `ARCHITECTURE.md`：数据流、CSV 列定义、PLR 公式、判读阈值、产物命名、扩展点。
- 训练 Tab 的 YAML/数据集下拉改为读 `config.TRAIN_MODEL_YAMLS` / `DATASET_YAMLS`。

### v2.1（同日）3D 注视模型不生成 / 不显示

11. **子进程解释器写死 `"python"`**
    `vogeo_backend.run_vogeo` 用 `["python", "-m", "vogeo_gaze", ...]`。若服务的
    PATH 里 `python` 不是运行本应用的那套环境（如 conda base 而非 `envs/yolo`），
    子进程 import 失败、returncode≠0，而 `app.run_video_common` 丢弃了返回值、只
    `print` 到控制台 —— 用户只看到“overlay 不存在”。改为 `sys.executable`，
    并把状态回填到统计框（新增一次 yield，不改变既有输出）。
12. **设备写死 `--device cuda`**
    无 GPU / 驱动未加载时必失败。改为默认不传 `--device`，交给 VOGeo 自判；
    `PUPIL_DEVICE` 可强制。实测本机 CPU：8 帧 10 秒、151 帧约 15 秒，完全可用。
13. **产物定位靠猜**
    旧版解析 stdout 的 `[done]` 并在 4 个候选路径里瞎找。现在直接
    `--out-dir <临时目录>`，再按目录约定收编，`gaze.csv / overlay.mp4 / run_info.json`
    齐全才算成功。
14. **可诊断**
    全量 stdout/stderr 写入 `outputs/vogeo_results/<stem>/vogeo.log`；
    `status_for()` 在“目录为空”时把日志尾部直接显示到界面。
15. **不能单独重建 3D**
    新增 `<csv>.run.json` 侧车（记录源视频、模型、帧率、是否 PLR/补帧）与
    「（重新）生成 3D 模型」按钮；源视频按 `run.json → 数据集 → cache → assets`
    反查，按“视频名前缀 + 分隔符”最长匹配，避免 `S_1` 抢 `S_10`。
16. **历史结果迁移**
    把原项目的 10 份逐帧 CSV、4 个刺激标定、9 套 `vogeo_results`（89 MB）复制到
    新目录，打开即可分析/播放，无需重跑。

### v2.2（同日）按共识重写 PLR 与判读

依据《瞳孔数值及临床问题》PPT 里的三份共识（穿透性颅脑损伤 2024、成人 TBI 院前与急诊
2025、眼肌型重症肌无力 2025），逐条落在 [CLINICAL.md](CLINICAL.md)。

17. **PLR 改为刺激窗口径（v2，默认）**
    基线 = 刺激前 [-0.6, 0) s 中位数；最小点只在刺激后 [0, 2] s 内找；
    收缩 <5% 判"未引出"。旧口径下 8 份有效 CSV 有 4 份最小点落在刺激前
    （MCV/潜伏期恒 0），1 份落在刺激后 27 s，全部会给出伪阳性参数。
18. **平滑改为时间基（0.2 s）**，10 fps 与 30 fps 可比；旧版固定 5 帧，
    同一条曲线换窗口 MCV 从 13.7 → 3.9。
19. **MCV 用 3 点最小二乘斜率**，抗单帧噪声；**潜伏期需连续 2 帧确认**，
    避免单帧抖动给出 0.01 s。
20. **NPi 重标定**：有 mm 标定时按 2 mm/s 锚点归一收缩速度（原公式用
    "基线相对速度/3"，与 mm/s 量纲不匹配）；像素数据明确标注"仅供趋势"。
21. **眨眼剔除**：读同名 `vogeo_results/<stem>/gaze.csv`，按 `open_eye_score<0.5`
    剔除闭眼帧，并在表里显示剔除了几帧。
22. **判读文案对齐共识**：散大分档改为 5–6 / 6–9 / >9 mm；6–9 mm 明写
    "晚期脑疝/脑干功能衰竭，脑死亡需整套评估"；双眼差 >1 mm 标注
    "A 级证据，提示 ICP 升高/脑疝，需紧急降颅压"；未引出收缩时不再写
    "对光反射消失"，而是"未引出（需复核刺激/光源/录像）"。
23. **参数表新增 4 行**：响应判定、帧率、潜伏期分辨率、数据质量（含剔除帧数）。
24. **CLI/自检**：`plr_params.py --mode v2|legacy`；`tools/selfcheck.py`
    输出 INFO 级质量提示，不再混进 WARN。

### 已知取舍

- 补帧视频比旧版多一代压缩（见上第 4 条），只影响可视化，CSV 完全一致。
- `analysis_backend` 仍保留历史函数与 `outputs/plots` 直写行为（低风险优先）。
- `model_info` 只报告不纠正 PLR 权重共享，需重训才能真正区分变体。
