# LGADNet CEMP 候选星发现全流程文档

---

## 📁 目录结构

```
pipeline3/
├── README.md                                    # 本文档
├── scripts/
│   ├── 01_train/                                # 阶段1: 训练数据准备与模型训练 (2脚本)
│   ├── 02_predict/                              # 阶段2: DR12全量预测与CEMP提取 (5脚本)
│   ├── 03_gaia_crossmatch/                      # 阶段3: Gaia交叉匹配与BJ距离处理 (3脚本)
│   ├── 04_validate/                             # 阶段4: 候选体验证与筛选 (4脚本)
│   └── 05_analysis/                             # 阶段5: 分析与可视化 (9脚本)
```

---

## 🔄 流程总览

```
┌─────────── 阶段1: 训练 ───────────┐
│ 01_prepare_training_dataset.py      │
│    ↓                                │
│ 02_train_lgadnet.py                 │
└────────────────────────────────────┘
              ↓ (模型 .pth)
┌─────────── 阶段2: 预测 ───────────┐
│ 01_predict_lgadnet_dr12.py          │
│    ↓                                │
│ 02_add_desig_from_fits.py           │
│    ↓                                │
│ 03_dedup_cemp_by_desig.py           │
│    ↓                                │
│ 04_cemp_spatial_prefilter.py (|b|>30°)│
│    ↓                                │
│ 05_prepare_vizier_upload.py         │
└────────────────────────────────────┘
              ↓ (VizieR 上传文件)
       [手动] VizieR Gaia DR3 交叉匹配
              ↓ (VizieR 匹配结果)
┌─────────── 阶段3: Gaia交叉匹配 ────┐
│ 01_dedup_vizier_double.py           │
│    ↓                                │
│ 02_query_bailer_jones_distance.py   │
│    ↓                                │
│ 03_check_bj_uncertainty.py          │
└────────────────────────────────────┘
              ↓
┌─────────── 阶段4: 验证 ───────────┐
│ 01_calculate_egp.py                 │
│    ↓                                │
│ 02_plot_bj_robust_gaia_cmd.py (CMD)  │
│    ↓                                │
│ 03_apply_apogee_constraint.py       │
│    ↓                                │
│ 04_export_final_catalog.py          │
└────────────────────────────────────┘
              ↓ (最终 CEMP 星表)
┌─────────── 阶段5: 分析 ───────────┐
│ 01~05: 模型评估 + 外部验证          │
│ 06~09: 科学可视化与论文图           │
└────────────────────────────────────┘
```

---

## 📊 阶段1: 训练数据准备与模型训练

### 1.1 准备训练数据集

| 项目 | 内容 |
|------|------|
| **脚本** | `01_train/01_prepare_training_dataset.py` |
| **功能** | 从 LAMOST DR12 光谱和交叉匹配标签一步生成 LGADNet v6 训练数据集，包括光谱预处理、CEMP分类、分层采样和 train/val 划分 |
| **输入** | `mapping_result_snr5.csv`（交叉匹配标签）、`lamost_dr12_cleaned_snr5.csv`（DR12元数据）、`dr12_fits/split_all/`（FITS光谱） |
| **输出** | `selected_all.csv`、`new_dataset_train_y.csv`、`new_dataset_val_y.csv`、`new_dataset_train_x.npy`、`new_dataset_val_x.npy` |

**关键参数**：
```python
--cemp-target 3020       # CEMP 类别目标采样数
--mp-target 6872         # MP-no-CEMP 类别目标采样数
--other-target 6800      # Other 类别目标采样数
--val-ratio 0.2          # 验证集比例
--seed 42                # 随机种子
```

**CEMP分类阈值**（Aoki 2007）：
- `logL <= 2.3` 时 `[C/Fe] >= 0.7`
- `logL > 2.3` 时 `[C/Fe] >= 3.0 - logL`
- 贫金属阈值：`[Fe/H] < -1.0`

**光谱预处理链**：截取(3900-8800Å) → 1Å重采样 → Savitzky-Golay去噪(window=15,poly=3) → 5阶多项式连续谱归一化 → 3σ裁剪

**使用方法**：
```bash
python 01_prepare_training_dataset.py
```

---

### 1.2 训练 LGADNet 模型

| 项目 | 内容 |
|------|------|
| **脚本** | `01_train/02_train_lgadnet.py` |
| **功能** | 定义 LGADNet 回归模型(CNN+Transformer)，加载训练数据，执行四参数(LOGG, TEFF, C_FE, FE_H)回归训练 |
| **输入** | `new_dataset_train_x.npy`、`new_dataset_train_y.csv`、`new_dataset_val_x.npy`、`new_dataset_val_y.csv` |
| **输出** | `{work_dir}/{experiment_name}_*_best_model.pth`（最佳模型checkpoint） |

**模型架构**：6层 ResNet1D + IDConv1dFull 动态卷积下采样 → Transformer Encoder(nhead=8, num_layers=2) → MLP回归头

**关键参数**：
```python
hidden_channels = 64
dropout = 0.2
batch_size = 32
max_epochs = 500
lr = 0.0001
weight_decay = 1e-5
seed = 2024
patience = 20
min_delta = 0.0001
```

**使用方法**：
```bash
python 02_train_lgadnet.py
```

---

## 📊 阶段2: DR12全量预测与CEMP提取

### 2.1 LGADNet 全量预测

| 项目 | 内容 |
|------|------|
| **脚本** | `02_predict/01_predict_lgadnet_dr12.py` |
| **功能** | 对 LAMOST DR12 全量分片光谱执行 LGADNet 四参数回归预测和CEMP分类，支持断点续跑和分片合并 |
| **输入** | 训练好的模型 `.pth`、`dr12_fits/split_all/`（FITS光谱） |
| **输出** | `predict_by_shard/dr12_predict_{shard}.csv`、`cemp_by_shard/dr12_cemp_{shard}.csv`、`dr12_predict_all.csv`、`dr12_cemp_all.csv` |

**关键参数**：
```python
--predict-batch-size 32   # 预测批大小（可设3072加速）
--shard / --all-shards    # 指定分片或全量
--resume                  # 断点续跑
--merge                   # 合并分片结果
--label-mean / --label-std  # 可覆盖checkpoint中的标签统计量
```

**使用方法**：
```bash
python 01_predict_lgadnet_dr12.py --all-shards --predict-batch-size 3072
python 01_predict_lgadnet_dr12.py --merge  # 合并分片结果
```

---

### 2.2 补充 DESIG/OBJNAME

| 项目 | 内容 |
|------|------|
| **脚本** | `02_predict/02_add_desig_from_fits.py` |
| **功能** | 从 FITS header 提取 DESIG/OBJNAME 并补充到 CEMP 候选 CSV |
| **输入** | `dr12_cemp_all.csv`（含 source_path 列） |
| **输出** | `dr12_cemp_all_with_desig.csv`（新增 DESIG, OBJNAME 列） |

**使用方法**：
```bash
python 02_add_desig_from_fits.py
```

---

### 2.3 按 DESIG 去重

| 项目 | 内容 |
|------|------|
| **脚本** | `02_predict/03_dedup_cemp_by_desig.py` |
| **功能** | 按 DESIG 对 CEMP 候选去重，保留每个目标的代表观测 |
| **输入** | `dr12_cemp_all_with_desig.csv` |
| **输出** | `dr12_cemp_unique_by_desig.csv` |

**去重优先级**：`cemp_margin(降序) > snrg(降序) > lmjd(降序) > obsid(降序)`

**使用方法**：
```bash
python 03_dedup_cemp_by_desig.py
```

---

### 2.4 空间预筛选

| 项目 | 内容 |
|------|------|
| **脚本** | `02_predict/04_cemp_spatial_prefilter.py` |
| **功能** | 计算银道坐标(b, l)，空间预筛选 |b| > 30° 的 CEMP 候选 |
| **输入** | `dr12_cemp_unique_by_desig.csv` |
| **输出** | `cemp_unique_b_greater_30.csv` |

**关键参数**：
```python
--b-threshold 30.0  # 银纬阈值
```

**使用方法**：
```bash
python 04_cemp_spatial_prefilter.py
```

---

### 2.5 准备 VizieR 上传文件

| 项目 | 内容 |
|------|------|
| **脚本** | `02_predict/05_prepare_vizier_upload.py` |
| **功能** | 准备 VizieR 交叉匹配上传文件，提取 candidate_id, ra, dec 三列 |
| **输入** | `cemp_unique_b_greater_30.csv` |
| **输出** | `vizier_upload_with_id.csv` |

**使用方法**：
```bash
python 05_prepare_vizier_upload.py
```

> ⚠️ **手动步骤**：上传 `vizier_upload_with_id.csv` 到 [VizieR SED](http://vizier.cds.unistra.fr/vizier/sed/) 执行 Gaia DR3 交叉匹配（默认参数），下载匹配结果（如 `1781361837819A.csv`）后进入阶段3。

---

## 📊 阶段3: Gaia交叉匹配与BJ距离处理

### 3.1 VizieR 交叉匹配双重去重

| 项目 | 内容 |
|------|------|
| **脚本** | `03_gaia_crossmatch/01_dedup_vizier_double.py` |
| **功能** | 对 VizieR 交叉匹配结果进行两级去重，建立一对一对应关系 |
| **输入** | VizieR 原始匹配结果（如 `1781361837819A.csv`） |
| **输出** | `vizier_dedup_double.csv` |

**去重逻辑**：
1. 第一级（按 `candidate_id`）：每个 LAMOST 候选保留角距离最小的 Gaia 匹配
2. 第二级（按 `Source`）：每个 Gaia 源保留角距离最小的 LAMOST 匹配

**使用方法**：
```bash
python 01_dedup_vizier_double.py
```

---

### 3.2 查询 Bailer-Jones 距离

| 项目 | 内容 |
|------|------|
| **脚本** | `03_gaia_crossmatch/02_query_bailer_jones_distance.py` |
| **功能** | 从 VizieR TAP 服务查询 Gaia DR3 Bailer-Jones 距离估计 |
| **输入** | `vizier_dedup_double.csv`（Gaia Source ID 列表） |
| **输出** | `bailer_jones_distance_for_candidates.csv` |

**查询参数**：
```python
BATCH_SIZE = 1000        # 每批查询 Source 数量
SLEEP_SEC = 1.0          # 批次间休眠秒数
# 数据源: Gaia EDR3距离星表 (I/352/gedr3dis)
```

**输出字段**：`Source, r_med_geo, r_lo_geo, r_hi_geo, r_med_photogeo, r_lo_photogeo, r_hi_photogeo, bj_flag`

**使用方法**：
```bash
python 02_query_bailer_jones_distance.py
```

---

### 3.3 BJ 距离不确定性分析

| 项目 | 内容 |
|------|------|
| **脚本** | `03_gaia_crossmatch/03_check_bj_uncertainty.py` |
| **功能** | 评估 BJ 距离不确定性，进行稳健的空间/运动学选择 |
| **输入** | `cemp_with_bailer_jones_distance_compare.csv` |
| **输出** | `bailer_jones_distance_uncertainty_flags.csv`、`bailer_jones_distance_uncertainty_summary.txt` |

**关键参数**：
```python
Z_SUN_KPC = 0.0208        # 太阳距银盘高度 (Bennett & Bovy 2019)
ABS_Z_THRESHOLD = 3.0     # 银盘高度阈值 (kpc)
VTAN_THRESHOLD = 180.0    # 切向速度阈值 (km/s)
```

**稳健选择逻辑**：即使使用距离下界（`r_lo_geo`）仍满足 |Z| > 3 kpc 且 Vtan > 180 km/s → 标记为 `bj_lo_both_pass`

**使用方法**：
```bash
python 03_check_bj_uncertainty.py
```

---

## 📊 阶段4: 候选体验证与筛选

### 4.1 EGP 指数计算

| 项目 | 内容 |
|------|------|
| **脚本** | `04_validate/01_calculate_egp.py` |
| **功能** | 从 LAMOST FITS 光谱计算 EGP（Enhanced G-Band Parameter）指数，评估碳增丰程度 |
| **输入** | `bailer_jones_distance_uncertainty_flags.csv`、`cemp_unique_b_greater_30.csv`（光谱路径） |
| **输出** | `bailer_jones_distance_uncertainty_flags_with_egp.csv` |

**EGP 定义**：
```
EGP = -2.5 * log10(Flux[4200-4400Å] / Flux[4425-4520Å])
```
- 分子：4200-4400Å（G波段，碳敏感）
- 分母：4425-4520Å（参考波段）

**光谱预处理**：红移改正 → 1Å重采样 → Savitzky-Golay去噪(window=15,poly=3) → 5阶多项式连续谱归一化 → 3σ裁剪

**使用方法**：
```bash
python 01_calculate_egp.py
```

---

### 4.2 CMD 质量检查

| 项目 | 内容 |
|------|------|
| **脚本** | `04_validate/02_plot_bj_robust_gaia_cmd.py` |
| **功能** | 在 Gaia CMD 上对 CEMP 候选体进行质量检查和筛选 |
| **输入** | `cemp_bj_robust_candidates_6928.csv`（BJ稳健筛选后） |
| **输出** | `cemp_bj_robust_cmd_flags.csv`、`cemp_bj_robust_cmd_summary.txt`、`cemp_bj_robust_gaia_cmd.pdf` |

**CMD 参数计算**：
```
BP-RP_0 = (BPmag - RPmag) - E(BP-RP)    # 红化改正后颜色
M_G = Gmag - 5*log10(distance) - AG      # 绝对星等
```

**CMD 区域分类**：
```python
COLOR_CUT = 0.4    # BP-RP 颜色阈值
MG_CUT = 7.0       # 绝对星等阈值

cmd_retained  ← BP-RP_0 >= 0.4 且 M_G_bj_lo <= 7.0  (主序区域)
cmd_hot_blue  ← BP-RP_0 < 0.4                          (热/蓝区域)
cmd_faint     ← M_G_bj_lo > 7.0                         (暗端区域)
```

**使用方法**：
```bash
python 02_plot_bj_robust_gaia_cmd.py
```

---

### 4.3 APOGEE 金属丰度约束

| 项目 | 内容 |
|------|------|
| **脚本** | `04_validate/03_apply_apogee_constraint.py` |
| **功能** | 使用 APOGEE 高分辨率光谱数据剔除假阳性候选体 |
| **输入** | `cemp_bj_robust_cmd_flags.csv`、`apogee_dr17.csv` |
| **输出** | `cemp_bj_robust_after_apogee.csv`、`cemp_bj_robust_apogee_matches.csv`、`cemp_bj_robust_apogee_summary.txt` |

**关键参数**：
```python
MATCH_RADIUS_ARCSEC = 3.0        # 匹配半径
APOGEE_FEH_THRESHOLD = -1.0      # 贫金属星阈值
```

**处理逻辑**：坐标匹配(3角秒) → APOGEE [Fe/H] ≥ -1.0 则剔除 → 保留 [Fe/H] < -1.0 或无APOGEE数据

**使用方法**：
```bash
python 03_apply_apogee_constraint.py
```

---

### 4.4 导出最终星表

| 项目 | 内容 |
|------|------|
| **脚本** | `04_validate/04_export_final_catalog.py` |
| **功能** | 整理输出最终 CEMP 候选体目录 |
| **输入** | `cemp_bj_robust_after_apogee.csv`、`cemp_unique_b_greater_30.csv` |
| **输出** | `cemp_bj_robust_final_catalog.csv`（全量）、`cemp_bj_robust_cmdretained_final_catalog.csv`（CMD-retained子集） |

**输出列**：ObsID, RA, DEC, Teff/Logg/FeH/CFe(LGADNet), Dist_BJ, Abs_Z, Vtan, CMD_Flag, RUWE, EGP, APOGEE信息等

**使用方法**：
```bash
python 04_export_final_catalog.py
```

---

## 📊 阶段5: 分析与可视化

### 模型评估 (01-02)

| # | 脚本 | 功能 | 依赖 |
|---|------|------|------|
| 01 | `01_evaluate_regression.py` | 验证集四参数回归评估 (MAE/RMSE/R²) + 散点图 | 阶段1模型 |
| 02 | `02_evaluate_multiclass.py` | 验证集三分类评估 (Accuracy/Precision/Recall/F1/MCC/AUC) | 阶段1模型 |

### EGP与统计 (03-05)

| # | 脚本 | 功能 | 依赖 |
|---|------|------|------|
| 03 | `03_calculate_egp_for_datasets.py` | 为训练/验证/选择/最终数据集计算EGP | 阶段1+4 |
| 04 | `04_analyze_cemp_fraction.py` | 分[Fe/H]区间统计CEMP占MP星比例 | 阶段2预测结果 |
| 05 | `05_galah_lamost_validation.py` | GALAH-LAMOST交叉匹配外部验证 | 阶段2预测结果 |

### 科学可视化 (06-09)

| # | 脚本 | 功能 | 输出图 | 依赖 |
|---|------|------|--------|------|
| 06 | `06_plot_cemp_density.py` | 最终CEMP候选体[Fe/H]-[C/Fe]和Teff-logL KDE密度图 | `cemp_parameter_density_*.pdf` | 阶段4 |
| 07 | `07_plot_egp_comparison.py` | 训练集vs最终候选体EGP分布对比 | `egp_train_vs_cemp_*.pdf` | 阶段4 |
| 09 | `09_plot_ch_gband_response.py` | 三窗口下低/高[C/Fe]组CH G-band吸收深度对比 | `ch_gband_cfe_*.pdf` | 脚本03 |
| 10 | `10_plot_representative_spectra.py` | 三类(CEMP/MP-no-CEMP/Other)代表性光谱对比 | `representative_spectra_v2.pdf` | 阶段1数据 |

---

## 📈 关键参数汇总

### CEMP 判定阈值

| 参数 | 值 | 说明 |
|------|-----|------|
| CEMP [C/Fe] 阈值 | ≥ 0.7 (logL≤2.3) 或 ≥ 3.0-logL (logL>2.3) | Aoki 2007 |
| 贫金属 [Fe/H] 阈值 | < -1.0 | 标准定义 |

### 空间与运动学筛选

| 参数 | 值 | 说明 |
|------|-----|------|
| \|b\| 预筛选阈值 | 30° | 远离银盘 |
| \|Z\| 阈值 | 3 kpc | 银盘高度阈值 |
| Vtan 阈值 | 180 km/s | 晕星族切向速度 |
| Z_sun | 0.0208 kpc | 太阳距银盘高度 |

### BJ 距离与 CMD 筛选

| 参数 | 值 | 说明 |
|------|-----|------|
| BJ 距离稳健策略 | 使用 r_lo_geo（距离下界） | 保守筛选 |
| BP-RP 颜色阈值 | 0.4 | CMD热/蓝端截断 |
| M_G 绝对星等阈值 | 7.0 | CMD暗端截断 |

### APOGEE 验证

| 参数 | 值 | 说明 |
|------|-----|------|
| 匹配半径 | 3.0 角秒 | 坐标匹配 |
| [Fe/H] 剔除阈值 | ≥ -1.0 | APOGEE确认非贫金属则剔除 |

### EGP 指数

| 参数 | 值 | 说明 |
|------|-----|------|
| G波段(分子) | 4200-4400 Å | CH G-band碳敏感区 |
| 参考波段(分母) | 4425-4520 Å | 连续谱参考区 |

