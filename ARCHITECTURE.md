# 架构与口径

本文记录「数据怎么流、参数怎么算、文件叫什么名字」。改代码前先看这里，
避免再出现“同一个公式两份实现”“界面能选但不生效”这类问题。

---

## 1. 分层

```
app.py                     界面 / 回调 / 组件联动（不写算法）
  ├── inference.py         分割推理、CSV、视频编码、补帧调度
  ├── analysis_backend.py  统计与判读（展示层）
  │     └── plr_core.py    PLR 计算（唯一实现）
  ├── pdf_export.py        报告排版（复用 analysis_backend 的结论）
  ├── deblur_backend.py    去模糊模型
  ├── motion_backend.py    去模糊视频 / 运动特征
  ├── flow_stats_backend.py 光流统计与专家推荐
  ├── flow_fill_backend.py 光流补帧（唯一实现）
  ├── roi_backend.py       MediaPipe ROI
  ├── train_backend.py     两套训练的子进程管理
  ├── vogeo_backend.py     VOGeo-Gaze 调用
  └── model_info.py        权重指纹与一致性检查
        └── config.py      路径 / 阈值 / 注册表（唯一来源）
```

约束：

- **路径**：除 `config.py` 外任何文件不得出现绝对路径。
- **公式**：PLR 只在 `plr_core.py`；补帧只在 `flow_fill_backend.py`。
  `plr_params.py`（CLI）与 `analysis_backend.analyze_plr`（网页/PDF）都调 `plr_core`。
- **参数**：阈值只改 `config.py`。

---

## 2. 一次视频推理的数据流

```
视频 ──逐帧──► YOLO(seg) ──► mask ──► fitEllipse ──► (major, minor, cx, cy)
                    │                                        │
                    ├─ overlay(轮廓+椭圆) ──► H.264 ──► outputs/segmentation_results/<stem>_<model>{_plr}{_flow}_seg.mp4
                    └─ 逐帧行 ──────────────► 同名 .csv
```

CSV 列（`_seg` 8 列；`_plr_seg` 12 列）：

| 列 | 含义 |
|---|---|
`frame` / `time_sec` | 帧号与时间（`frame / fps`） |
`detected` | 瞳孔是否检出（0/1） |
`major` / `minor` | 拟合椭圆长轴/短轴（px） |
`cx` / `cy` | 椭圆中心（px） |
`conf` | 该帧最高检测置信度 |
`iris_detected` | 虹膜是否检出（仅 PLR 模式） |
`iris_major` / `iris_minor` | 虹膜椭圆长短轴（px） |
`pupil_mm` | `major / iris_major × CORNEA_MM` |

命名规则：`<视频名>_<模型名>[_plr][_flow]_seg.{mp4,csv}`。
后续所有分析（PLR、判读、VOGeo 结果目录）都以**这个 CSV 的 stem** 为 key。

### 光流补帧（`flow_fill=True`）

1. pass1：逐帧推理，overlay 直接写入中间视频（不驻留内存），得到 CSV；
2. pass2：**流式**重读源视频与中间视频，对每段连续 `detected==0` 的帧：
   - 取前一个/后一个检出帧的 mask；
   - `Farneback(prev → cur)`、`Farneback(next → cur)` 分别 warp 到当前帧；
   - 按 `1/帧距` 加权融合，阈值 127.5 二值化；
   - 画到原始帧上写回。
3. 单段最多缓存 `FILL_CHUNK=64` 帧，超长漏检段分段处理（前向参考改用上一段末尾 mask）。

对应旧版：旧版把整段 `frames` 与 `overlays` 都留在内存（600 帧 720p ≈ 3 GB），
内存峰值随视频长度线性增长且必然 OOM。

---

## 3. PLR 计算口径（`plr_core.py`）

```
d(t) = median_filter_5(pupil_mm | major)           # 只用 detected==1
D0   = median(d[t < t_stim])                       # 不足 3 点则取前 1/10
Dmin = min(d),  t_min = argmin
t_stim 优先级：手动 > stimulus_marks 标定文件 > 自动检测（连续 5 帧下降且降幅 >1%）
收缩时间   = t_min - t[t_stim]
收缩幅度   = D0 - Dmin
收缩百分比 = 幅度 / D0 × 100
潜伏期     = 刺激后首次 (d[i+1] < 0.995·d[i]) 的时间差，无则退化为收缩时间
MCV        = -min(diff(d)/diff(t)) over [t_stim, t_min]
恢复时间   = 从 t_min 回到 D0 - 0.5(D0 - Dmin) 的时间
NPi 近似   = 1.5·s_lat + 2.0·s_mcv + 1.5·s_amp   (0–5)
             s_lat = clip((0.5 - latency)/0.5), s_mcv = clip(mcv_norm/3), s_amp = clip(幅度比/0.4)
```

**这不是 NPi-200**，是三个可计算量的加权近似，只能看趋势。

---

## 4. 判读规则（`analysis_backend.clinical_interpretation`）

| 判据 | 阈值 | 结论 |
|---|---|---|
基线直径 | < 2 mm | 针尖样瞳孔 |
基线直径 | > 5 mm | 散大（≥6 mm 显著） |
收缩百分比 | < 5% | 对光反射消失 |
收缩百分比 | 5–15% | 反射减弱 |
MCV | < 0.5 / < 1.0 | 极低 / 偏低 |
NPi 近似 | < 3 | 异常 |
长短轴比 | < 0.8 | 形态欠规则 |
gaze theta 标准差 | > 2° | 视线漂移 → PLR 质控不通过 |
单眼 1–2 mm | — | 小瞳（Horner 鉴别） |
散大 + 反射消失 | — | 散大固定（脑疝 / 动眼神经） |
双眼基线差 | > 1 mm | 病理性不等大 |
双眼 gaze 均值差 | > 5° | 眼外肌/复视/滑车神经（示意，需同人同步） |

单眼 13 类病症对照、双眼 12 类对照分别由 `ppt_reference_single` / `ppt_reference_dual`
生成；每行都给出“判读 / 依据（本视频）/ 判读条件”三列，不出现空结论。

---

## 5. 可靠性门（本版新增）

`analysis_backend.reliability_warnings(csv_name)` 返回提示列表，界面把它前置到「概览」，
双眼分析分别加「左眼/右眼」前缀：

- 检出率 < `RELIABLE_DETECT_RATE`（30%）→ 结论不可靠；
- 检出率 < `QUALITY_DETECT_RATE_FAIR`（70%）→ 曲线可能断点；
- 非 PLR CSV（无 `pupil_mm`）→ mm 参考范围不适用；
- 无刺激标定 → 潜伏期/收缩时间可能失真。

---

## 6. 双眼分析

左/右各跑一遍单眼链路，然后：

- `compare_eyes_plots`：直径、速度、轨迹（各自中心化）、长短轴分布、左右差值；
- `compare_eyes_plr_plots`：原始 PLR 叠加 + 归一化（%基线）叠加，各带左右刺激竖线；
- `_dual_interp`（app）/ `pdf_export._dual_interp`：逐项差值判读 + 双眼病症对照；
- `compare_eyes_gaze_plots`：theta_h/theta_v 叠加 + 2D 轨迹叠加（仅当两眼的 `gaze.csv` 都在）。

---

## 7. 产出结构

```
outputs/
├── segmentation_results/<stem>.csv / <stem>.mp4
├── plots/<stem>_{timeseries,dist,trajectory,plr_analysis,morph_*}.png
│         dual_<L>_vs_<R>_{diameter,speed,trajectory,dist,diff,plr_overlay,plr_norm}.png
├── pdf_reports/<stem>_report.pdf / dual_<L>_vs_<R>_report.pdf
├── stimulus_marks/<视频 stem>.json        {video, dataset, marks[], current}
├── deblur_videos/<stem>_<expert>_deblur.mp4
├── motion/<stem>_motion.csv
├── vogeo_results/<stem>/{gaze.csv, overlay.mp4, run_info.json, *_gaze_*.png}
└── roi_results/<stem>_roi.{csv,mp4} / <stem>_roi_{left,right,debug}.mp4
```

`stimulus_marks` 的匹配规则：CSV stem 与 `video` 字段 stem **完全相同**，或以
「stem + `_`/`-`/`.`」开头；多个候选取最长匹配，精确匹配额外优先。
（旧版用裸 `startswith`，`1_1` 会误吃 `1_10` 的标定。）

---

## 8. 扩展点

- **加数据集**：`config.DATASET_DIRS` 加一项（或用 `PUPIL_OPENEDS_DIR` 之类环境变量）。
- **加去模糊专家**：`$PUPIL_YOLO_ROOT/LPW/models/<name>/best.pth`，下拉自动出现；
  `deblur_backend.load_deblur` 会按 state_dict 里有无 `epa_bottleneck.` / `cbam1.` 自动判定结构。
- **加判读项**：`analysis_backend.clinical_interpretation` 加一段 `rows.append([...])`，
  再到 `ppt_reference_single` 补一行病症映射；PDF 会自动带上（中英对照需补 `_EN_NAMES`）。
- **加报告章节**：`pdf_export._section_*` 加函数，随后 `_build_html` 里插一行。
