#!/usr/bin/env python3
"""
绘制代表性光谱对比图 - 基于原始代码改进版

改动说明:
1. 数据集: regression_snr5_selected_v6_cemp3020_20260611
2. 样本选择: 所有类别统一按 C_FE 中位数选择（替代随机选择）
3. 样式: 完全保持原始代码风格
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================= 0. 设置随机种子 (保证可复现) =================
# 注意：现在使用中位数选择，不再需要随机种子
# np.random.seed(42)  # 已废弃

# ================= 配置路径 =================
x_path = "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/regression_snr5_selected_v6_cemp3020_20260611/new_dataset_train_x.npy"
y_path = "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/regression_snr5_selected_v6_cemp3020_20260611/new_dataset_train_y.csv"
save_path = "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/figures/representative_spectra_v2.pdf"

# ================= 1. 读取数据 =================
# 读取标签
df_labels = pd.read_csv(y_path)

# 读取光谱数据 (使用 mmap_mode='r' 以节省内存)
data_spectra = np.load(x_path, mmap_mode='r')

# 检查长度以生成波长轴
n_samples, n_points = data_spectra.shape
print(f"数据加载完成: {n_samples} 条光谱, 每条长度 {n_points}")

# 生成波长轴 (3900 - 8800)
wavelength = np.linspace(3900, 8800, n_points)

# ================= 2. 选择代表性样本（中位数策略） =================
classes = ['cemp', 'mp-no-cemp', 'other']

samples = {}
for cls in classes:
    # 找到该类别的所有索引
    df_class = df_labels[df_labels['class'] == cls]

    if len(df_class) > 0:
        # 改进：选择 C_FE 最接近中位数的样本（替代随机选择）
        median_cfe = df_class['C_FE'].median()
        closest_idx = (df_class['C_FE'] - median_cfe).abs().idxmin()

        samples[cls] = data_spectra[closest_idx]

        # 输出选择的样本信息
        sample_info = df_class.loc[closest_idx]
        print(f"类别 {cls}: 选中样本索引 {closest_idx}")
        print(f"  ObsID={int(sample_info['obsid'])}, [C/Fe]={sample_info['C_FE']:.2f}, [Fe/H]={sample_info['FE_H']:.2f}, Teff={sample_info['TEFF']:.0f}K")
    else:
        print(f"警告: 未找到类别 {cls}")

# ================= 3. 定义重要的谱线 (单位: Angstrom) =================
# 谱线标记：红色标记碳线，蓝色标记铁线
lines_info = {
    # --- 碳相关 (Carbon / CH / C2) ---
    4300: ('CH G-band', 'red'),
    4737: ('C₂ Swan', 'red'),
    5165: ('C₂ Swan', 'red'),
    5635: ('C₂ Swan', 'red'),

    # --- 铁相关 (Iron / Fe) ---
    4383: ('Fe I', 'blue'),
    5270: ('Fe I', 'blue'),
    5335: ('Fe I', 'blue'),
}

# ================= 4. 绘图 =================
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, dpi=100)
plt.subplots_adjust(hspace=0.1)

for i, cls in enumerate(classes):
    ax = axes[i]
    if cls not in samples:
        continue

    flux = samples[cls]

    # 简单的归一化 (除以中位数)
    norm_flux = flux / (np.median(flux) + 1e-6)

    # 画光谱线（科研论文标准深蓝色）
    ax.plot(wavelength, norm_flux, color='#2166AC', linewidth=0.8, label=f'Class: {cls}')

    # 标注重要的线
    for wl_val, (name, color) in lines_info.items():
        ax.axvline(x=wl_val, color=color, linestyle='--', alpha=0.5, linewidth=1)

        # 仅在第一幅图上标文字
        if i == 0:
            ax.text(wl_val, ax.get_ylim()[1]*0.95, name, color=color,
                    rotation=90, fontsize=8, ha='right', va='top')

    # G-band 区域标注（使用浅蓝色半透明）
    if i == 0:
        ax.axvspan(4280, 4320, color='#85C1E9', alpha=0.3, label='Carbon Features')
    else:
        ax.axvspan(4280, 4320, color='#85C1E9', alpha=0.3)

    # 图例与标签
    ax.legend(loc='upper right', frameon=True)
    ax.set_ylabel('Normalized Flux')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title(f"Sample Spectrum: {cls}", fontsize=12)

axes[-1].set_xlabel(r'Wavelength ($\AA$)')
axes[-1].set_xlim(3900, 6000) # 限制显示范围

# 删除大标题，保持简洁
# plt.suptitle('Comparison of Spectra Classes with Key C & Fe Lines', fontsize=16)
plt.tight_layout()

# 保存图片
plt.savefig(save_path)
print(f"\n图片已保存至: {save_path}")
plt.show()
