#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 LGADNet checkpoint 在验证集上做四参数回归评估并生成散点图。"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT_DIR = Path("/home/DM13/workspace/sky")
LGADNET_DIR = ROOT_DIR / "data/new_dataset3/lgadnet"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(LGADNET_DIR) not in sys.path:
    sys.path.insert(0, str(LGADNET_DIR))

from pipelines.train.train_lgadnet import LGADNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CustomDataset(Dataset):
    def __init__(self, feature_path: Path, label_path: Path, normalize: bool = True):
        self.features = np.load(feature_path)
        label_df = pd.read_csv(label_path)
        self.labels = label_df[["LOGG", "TEFF", "C_FE", "FE_H"]].values.astype(np.float32)

        # 过滤 NaN 样本（在加载时过滤）
        valid_mask = np.all(np.isfinite(self.features), axis=1) & np.all(np.isfinite(self.labels), axis=1)
        self.features = self.features[valid_mask]
        self.labels = self.labels[valid_mask]
        self.valid_indices = np.where(valid_mask)[0]
        print(f"[INFO] 过滤 NaN 样本: 原始 {len(valid_mask)} -> 保留 {len(self.valid_indices)} (丢弃 {len(valid_mask) - len(self.valid_indices)})")

        if normalize:
            self.label_mean = np.mean(self.labels, axis=0)
            self.label_std = np.std(self.labels, axis=0)
            self.label_std[self.label_std == 0] = 1.0
            self.labels = (self.labels - self.label_mean) / self.label_std
        else:
            self.label_mean = None
            self.label_std = None

    def __getitem__(self, index: int):
        feature = torch.tensor(self.features[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return feature, label

    def __len__(self) -> int:
        return len(self.features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--train-feature-path", type=Path, required=True)
    parser.add_argument("--train-label-path", type=Path, required=True)
    parser.add_argument("--test-feature-path", type=Path, required=True)
    parser.add_argument("--test-label-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--output-pdf", type=Path, required=True, help="输出PDF文件路径")
    return parser.parse_args()


def plot_scatter(true_values: np.ndarray, pred_values: np.ndarray, output_pdf: Path) -> None:
    """生成四参数对比散点图（密度加权颜色）。"""
    param_names = ["$\log g$", "$T_{\\mathrm{eff}}$", "[C/Fe]", "[Fe/H]"]
    units = ["dex", "K", "dex", "dex"]

    # 设置绘图样式
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['figure.dpi'] = 150

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for idx, (ax, param_name, unit) in enumerate(zip(axes, param_names, units)):
        true = true_values[:, idx]
        pred = pred_values[:, idx]

        # 计算误差统计
        mae = mean_absolute_error(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        r2 = r2_score(true, pred)
        residuals = pred - true
        bias = np.mean(residuals)
        std_res = np.std(residuals)
        sigma_3 = std_res * 3

        # 计算密度（使用 2D 直方图快速估计）
        nbins = 100
        H, xedges, yedges = np.histogram2d(true, pred, bins=nbins)

        # 计算每个点所在的网格索引
        xidx = np.clip(np.digitize(true, xedges) - 1, 0, nbins-1)
        yidx = np.clip(np.digitize(pred, yedges) - 1, 0, nbins-1)

        # 获取每个点的密度
        density = H[xidx, yidx]

        # 归一化密度
        density_norm = (density - density.min()) / (density.max() - density.min())

        # 根据密度排序（密度低的先绘制,密度高的后绘制）
        sorted_idx = np.argsort(density_norm)
        x_sorted = true[sorted_idx]
        y_sorted = pred[sorted_idx]
        density_sorted = density_norm[sorted_idx]

        # 绘制散点图,根据密度调整颜色
        ax.scatter(x_sorted, y_sorted,
                  c=density_sorted,
                  cmap='coolwarm',  # 蓝色到红色
                  s=10,
                  alpha=0.6,
                  edgecolors='none',
                  rasterized=True)

        # 1:1 参考线（黑色虚线）
        min_val = min(true.min(), pred.min())
        max_val = max(true.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.8, label='1:1 line')

        # 3-sigma 线（绿色虚线）
        ax.plot([min_val, max_val], [min_val + sigma_3, max_val + sigma_3],
               'g--', linewidth=1.5, alpha=0.7, label=f'+3σ')
        ax.plot([min_val, max_val], [min_val - sigma_3, max_val - sigma_3],
               'g--', linewidth=1.5, alpha=0.7, label=f'-3σ')

        # 统计信息文本框
        textstr = f'MAE = {mae:.3f}\n'
        textstr += f'RMSE = {rmse:.3f}\n'
        textstr += f'Bias = {bias:.3f}\n'
        textstr += f'Std = {std_res:.3f}\n'
        textstr += f'R² = {r2:.3f}'

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props, family='monospace')

        ax.set_xlabel(f"{param_name} ({unit})", fontsize=11)
        ax.set_ylabel(f"{param_name} ({unit})", fontsize=11)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    # 同时保存 PNG 格式
    png_path = output_pdf.with_suffix('.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ PDF 已保存: {output_pdf}")
    print(f"✓ PNG 已保存: {png_path}")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    # 加载模型
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    cfg = checkpoint.get("config", {})

    model = LGADNet(
        num_label=cfg.get("num_label", 4),
        len_spectrum=cfg.get("len_spectrum", 4901),
        hidden_channels=cfg.get("hidden_channels", 64),
        dropout=cfg.get("dropout", 0.2),
        nhead=cfg.get("nhead", 8),
        num_layers=cfg.get("num_layers", 2),
        dim_feedforward=cfg.get("dim_feedforward", 512),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    # 加载数据
    train_dataset = CustomDataset(args.train_feature_path, args.train_label_path, normalize=True)
    test_dataset = CustomDataset(args.test_feature_path, args.test_label_path, normalize=False)
    test_dataset.labels = (test_dataset.labels - train_dataset.label_mean) / train_dataset.label_std

    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 预测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            outputs = model(features)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 反归一化
    all_preds_original = all_preds * train_dataset.label_std + train_dataset.label_mean
    all_labels_original = all_labels * train_dataset.label_std + train_dataset.label_mean

    # 打印指标
    mae = np.mean(np.abs(all_preds_original - all_labels_original), axis=0)
    rmse = np.sqrt(np.mean((all_preds_original - all_labels_original) ** 2, axis=0))
    print(f"\n样本数: {len(all_labels)}")
    print(f"MAE (LOGG, TEFF, C_FE, FE_H): {mae}")
    print(f"RMSE (LOGG, TEFF, C_FE, FE_H): {rmse}")

    # 生成可视化
    plot_scatter(all_labels_original, all_preds_original, args.output_pdf)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())