#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GALAH-LAMOST DR12 交叉匹配与参数对比分析（精简版）

功能：
1. 坐标交叉匹配（1角秒半径）
2. 计算误差统计（MAE, RMSE, R², Bias, Std）
3. 输出匹配数据表
4. 参数对比散点图（1x4 布局，密度加权颜色）

输出目录：galah_validation/
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from astropy.coordinates import SkyCoord
import astropy.units as u
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def log(message: str) -> None:
    """打印带时间戳的日志。"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def crossmatch_coordinates(galah_df: pd.DataFrame, lamost_df: pd.DataFrame, radius_arcsec: float = 1.0) -> pd.DataFrame:
    """
    坐标交叉匹配。

    Args:
        galah_df: GALAH 数据（真值）
        lamost_df: LAMOST 预测数据
        radius_arcsec: 匹配半径（角秒）

    Returns:
        匹配结果 DataFrame
    """
    log(f"开始坐标匹配，半径={radius_arcsec}角秒")

    # 构建 SkyCoord 对象
    galah_coords = SkyCoord(ra=galah_df['RA'].values * u.deg, dec=galah_df['DEC'].values * u.deg, frame='icrs')
    lamost_coords = SkyCoord(ra=lamost_df['ra'].values * u.deg, dec=lamost_df['dec'].values * u.deg, frame='icrs')

    # 交叉匹配
    log("执行交叉匹配（可能需要几分钟）...")
    idx_lamost, idx_galah, sep, _ = galah_coords.search_around_sky(lamost_coords, radius_arcsec * u.arcsec)

    log(f"匹配结果：{len(idx_galah)} 对")

    if len(idx_galah) == 0:
        log("⚠️ 警告：未找到任何匹配源！尝试增大匹配半径")
        return pd.DataFrame()

    # 过滤有效索引
    valid_mask = (idx_galah < len(galah_df)) & (idx_lamost < len(lamost_df))
    idx_galah = idx_galah[valid_mask]
    idx_lamost = idx_lamost[valid_mask]
    sep = sep[valid_mask]

    if len(idx_galah) == 0:
        log("⚠️ 警告：过滤后无有效匹配源！")
        return pd.DataFrame()

    log(f"有效匹配源：{len(idx_galah)} 对")

    # 构建匹配表
    matched_galah = galah_df.iloc[idx_galah].copy().reset_index(drop=True)
    matched_lamost = lamost_df.iloc[idx_lamost].copy().reset_index(drop=True)

    # 合并数据
    matched_df = pd.DataFrame({
        # GALAH 数据（真值）
        'galah_ra': matched_galah['RA'],
        'galah_dec': matched_galah['DEC'],
        'galah_snr': matched_galah['SNR'],
        'galah_logg': matched_galah['LOGG'],
        'galah_teff': matched_galah['TEFF'],
        'galah_c_fe': matched_galah['C_FE'],
        'galah_fe_h': matched_galah['FE_H'],

        # LAMOST 数据（预测）
        'lamost_ra': matched_lamost['ra'],
        'lamost_dec': matched_lamost['dec'],
        'lamost_obsid': matched_lamost['obsid'],
        'lamost_logg': matched_lamost['LOGG'],
        'lamost_teff': matched_lamost['TEFF'],
        'lamost_c_fe': matched_lamost['C_FE'],
        'lamost_fe_h': matched_lamost['FE_H'],
        'lamost_class': matched_lamost['class'],
        'lamost_is_cemp': matched_lamost['is_cemp'],

        # 匹配信息
        'separation_arcsec': sep.arcsec,
    })

    # 如果一个 GALAH 源匹配多个 LAMOST 观测，只保留最近的
    matched_df = matched_df.sort_values(['galah_ra', 'galah_dec', 'separation_arcsec'])
    matched_df = matched_df.drop_duplicates(subset=['galah_ra', 'galah_dec'], keep='first')

    log(f"去重后匹配源数量：{len(matched_df)}")

    return matched_df.reset_index(drop=True)


def calculate_galah_cemp(matched_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 GALAH 数据的 CEMP 分类（使用光度自适应阈值）。

    Args:
        matched_df: 匹配数据

    Returns:
        添加 CEMP 分类后的数据
    """
    log("计算 GALAH CEMP 分类")

    # 计算光度
    logg = matched_df['galah_logg']
    teff = matched_df['galah_teff']
    log_l = np.log10(0.8) - (logg - 4.44) + 4.0 * np.log10(teff / 5780.0)

    # CEMP 阈值
    threshold = np.where(log_l <= 2.3, 0.7, 3.0 - log_l)

    # 分类
    is_mp = matched_df['galah_fe_h'] < -1.0
    is_cemp = is_mp & (matched_df['galah_c_fe'] >= threshold)

    matched_df['galah_logL'] = log_l
    matched_df['galah_cemp_threshold'] = threshold
    matched_df['galah_is_mp'] = is_mp
    matched_df['galah_is_cemp'] = is_cemp
    matched_df['galah_class'] = 'other'
    matched_df.loc[is_mp & ~is_cemp, 'galah_class'] = 'mp-no-cemp'
    matched_df.loc[is_cemp, 'galah_class'] = 'cemp'

    log(f"GALAH CEMP 分类结果:\n{matched_df['galah_class'].value_counts()}")

    return matched_df


def plot_comparison(matched_df: pd.DataFrame, output_dir: Path) -> None:
    """
    绘制参数对比散点图（1x4 布局，密度加权颜色）。

    Args:
        matched_df: 匹配数据
        output_dir: 输出目录
    """
    log("绘制参数对比散点图")

    # 计算 [C/H]
    matched_df['galah_c_h'] = matched_df['galah_c_fe'] + matched_df['galah_fe_h']
    matched_df['lamost_c_h'] = matched_df['lamost_c_fe'] + matched_df['lamost_fe_h']

    # 设置绘图样式
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['figure.dpi'] = 150

    # 参数列表
    params = ['logg', 'teff', 'c_h', 'fe_h']
    param_labels = {
        'logg': r'$\log g$',
        'teff': r'$T_{\mathrm{eff}}$ (K)',
        'c_h': r'[C/H]',
        'fe_h': r'[Fe/H]',
    }

    # 创建 1x4 子图
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for idx, param in enumerate(params):
        ax = axes[idx]

        galah_col = f'galah_{param}'
        lamost_col = f'lamost_{param}'

        # 提取有效数据
        valid = matched_df[[galah_col, lamost_col]].dropna()

        if len(valid) > 0:
            true_vals = valid[galah_col].values
            pred_vals = valid[lamost_col].values

            # 计算误差
            errors = pred_vals - true_vals
            bias = np.mean(errors)
            std = np.std(errors)

            # 计算密度（使用 2D 直方图快速估计）
            nbins = 100
            H, xedges, yedges = np.histogram2d(true_vals, pred_vals, bins=nbins)

            # 计算每个点所在的网格索引
            xidx = np.clip(np.digitize(true_vals, xedges) - 1, 0, nbins-1)
            yidx = np.clip(np.digitize(pred_vals, yedges) - 1, 0, nbins-1)

            # 获取每个点的密度
            density = H[xidx, yidx]

            # 归一化密度
            density_norm = (density - density.min()) / (density.max() - density.min())

            # 根据密度排序（密度低的先绘制，密度高的后绘制）
            sorted_idx = np.argsort(density_norm)
            x_sorted = true_vals[sorted_idx]
            y_sorted = pred_vals[sorted_idx]
            density_sorted = density_norm[sorted_idx]

            # 绘制散点图，根据密度调整颜色
            ax.scatter(x_sorted, y_sorted,
                      c=density_sorted,
                      cmap='coolwarm',  # 蓝色到红色
                      s=3,
                      alpha=0.6,
                      edgecolors='none',
                      rasterized=True)

            # 1:1 参考线（黑色虚线）
            xmin, xmax = min(true_vals.min(), pred_vals.min()), max(true_vals.max(), pred_vals.max())
            ax.plot([xmin, xmax], [xmin, xmax], 'k--', linewidth=2, alpha=0.8, label='1:1 line')

            # 3-sigma 线（绿色虚线）
            y_upper = lambda x: x + bias + 3 * std
            y_lower = lambda x: x + bias - 3 * std

            x_line = np.linspace(xmin, xmax, 100)
            ax.plot(x_line, y_upper(x_line), 'g--', linewidth=1.5, alpha=0.7, label=f'+3σ')
            ax.plot(x_line, y_lower(x_line), 'g--', linewidth=1.5, alpha=0.7, label=f'-3σ')

            # 统计信息
            r = pearsonr(true_vals, pred_vals)[0]
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors**2))

            # 在图中添加统计信息
            stats_text = f'MAE = {mae:.3f}\n'
            stats_text += f'RMSE = {rmse:.3f}\n'
            stats_text += f'Bias = {bias:.3f}\n'
            stats_text += f'Std = {std:.3f}\n'
            stats_text += f'R² = {r**2:.3f}'

            ax.text(0.05, 0.95, stats_text,
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   family='monospace')

            ax.set_xlabel(f"GALAH {param_labels[param]} (True)")
            ax.set_ylabel(f"LAMOST {param_labels[param]} (Predicted)")
            ax.legend(fontsize=9, loc='lower right')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'comparison_scatter.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / 'comparison_scatter.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)

    log("✓ 参数对比散点图已生成")


def main() -> None:
    """主函数。"""
    # 路径配置
    galah_path = Path('/home/DM13/workspace/sky/data/galah_selected_data.csv')
    lamost_path = Path('/home/DM13/workspace/sky/data/new_dataset3/lgadnet/cemp_pipeline_lgadnet_v6_cemp3020_bs3072_labelscale_fixed/dr12_predict_all.csv')
    output_dir = Path('/home/DM13/workspace/sky/data/new_dataset3/lgadnet/galah_validation')

    output_dir.mkdir(parents=True, exist_ok=True)

    log("="*80)
    log("GALAH-LAMOST DR12 交叉匹配验证分析（精简版）")
    log("="*80)

    # 读取数据（只读取必要的列以节省内存）
    log("读取 GALAH 数据...")
    galah_df = pd.read_csv(galah_path)
    log(f"GALAH: {len(galah_df)} 行")

    log("读取 LAMOST 预测数据（可能需要几分钟）...")
    lamost_cols = ['obsid', 'ra', 'dec', 'LOGG', 'TEFF', 'C_FE', 'FE_H', 'class', 'is_cemp']
    lamost_df = pd.read_csv(lamost_path, usecols=lamost_cols)
    log(f"LAMOST: {len(lamost_df)} 行")

    # 交叉匹配
    matched_df = crossmatch_coordinates(galah_df, lamost_df, radius_arcsec=1.0)

    if len(matched_df) == 0:
        log("❌ 匹配失败，程序终止")
        return

    # 计算 GALAH CEMP 分类
    matched_df = calculate_galah_cemp(matched_df)

    # 绘制参数对比散点图
    plot_comparison(matched_df, output_dir)

    # 保存匹配数据
    log("保存匹配数据...")
    matched_df.to_csv(output_dir / 'matched_data.csv', index=False)

    log("="*80)
    log("✅ 验证分析完成！")
    log(f"输出目录: {output_dir}")
    log(f"输出文件:")
    log(f"  - matched_data.csv ({len(matched_df)} 行)")
    log(f"  - comparison_scatter.png/pdf (参数对比散点图)")
    log("="*80)


if __name__ == '__main__':
    main()