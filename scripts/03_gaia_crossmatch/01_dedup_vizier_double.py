#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双重去重脚本：对VizieR交叉匹配结果进行两级去重

去重策略：
1. 第一级：按 candidate_id 去重（每个LAMOST候选只保留最近的Gaia匹配）
2. 第二级：按 Source 去重（每个Gaia源只保留最近的一个候选）

输入：VizieR返回的匹配结果CSV
输出：双重去重后的CSV文件
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(
    "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/"
    "spatial_filtering_unique/1781361837819A.csv"
)
DEFAULT_OUTPUT = Path(
    "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/"
    "spatial_filtering_unique/vizier_dedup_double.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    # 检查输入文件
    if not input_path.is_file():
        raise FileNotFoundError(f"输入 CSV 不存在: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"输出文件已存在，请换路径或加 --overwrite: {output_path}")

    # 加载数据
    print("=" * 80)
    print("VizieR 匹配结果双重去重")
    print("=" * 80)
    print(f"\n输入文件: {input_path}")

    df = pd.read_csv(input_path)
    print(f"原始记录数: {len(df):,}")

    # 检查必要列
    required_cols = ['candidate_id', 'angDist', 'Source']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"输入 CSV 缺少必要列: {missing_cols}")

    # 转换 angDist 为数值类型
    df['angDist'] = pd.to_numeric(df['angDist'], errors='coerce')
    df = df.dropna(subset=['angDist'])

    # ============================================================
    # 第一级去重：按 candidate_id（每个LAMOST候选只保留最近的Gaia匹配）
    # ============================================================
    print("\n" + "-" * 80)
    print("第一级去重：按 candidate_id（保留 angDist 最小）")
    print("-" * 80)

    df_step1 = df.sort_values(['candidate_id', 'angDist']).drop_duplicates(
        'candidate_id', keep='first'
    )
    removed_step1 = len(df) - len(df_step1)
    print(f"  原始记录数: {len(df):,}")
    print(f"  去重后记录数: {len(df_step1):,}")
    print(f"  移除记录数: {removed_step1:,}")
    print(f"  唯一 candidate_id: {df_step1['candidate_id'].nunique():,}")

    # ============================================================
    # 第二级去重：按 Source（每个Gaia源只保留最近的一个候选）
    # ============================================================
    print("\n" + "-" * 80)
    print("第二级去重：按 Source（保留 angDist 最小）")
    print("-" * 80)

    # 检查 Source 重复情况
    source_counts = df_step1['Source'].value_counts()
    multi_source = source_counts[source_counts > 1]
    print(f"  有重复的 Source 数量: {len(multi_source):,}")
    print(f"  涉及记录数: {multi_source.sum():,}")

    # 执行第二级去重
    df_step2 = df_step1.sort_values(['Source', 'angDist']).drop_duplicates(
        'Source', keep='first'
    )
    removed_step2 = len(df_step1) - len(df_step2)
    print(f"  第一级去重后: {len(df_step1):,}")
    print(f"  第二级去重后: {len(df_step2):,}")
    print(f"  移除记录数: {removed_step2:,}")
    print(f"  唯一 Source: {df_step2['Source'].nunique():,}")

    # ============================================================
    # 统计总结
    # ============================================================
    print("\n" + "=" * 80)
    print("去重统计总结")
    print("=" * 80)
    print(f"  原始记录数: {len(df):,}")
    print(f"  第一级去重移除: {removed_step1:,} (candidate_id 重复)")
    print(f"  第二级去重移除: {removed_step2:,} (Source 重复)")
    print(f"  最终记录数: {len(df_step2):,}")
    print(f"  总移除记录数: {len(df) - len(df_step2):,}")
    print(f"  保留比例: {len(df_step2)/len(df)*100:.2f}%")

    # 验证唯一性
    print("\n" + "-" * 80)
    print("唯一性验证")
    print("-" * 80)
    print(f"  ✓ candidate_id 唯一: {df_step2['candidate_id'].nunique() == len(df_step2)}")
    print(f"  ✓ Source 唯一: {df_step2['Source'].nunique() == len(df_step2)}")

    # ============================================================
    # 保存结果
    # ============================================================
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_step2.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("输出结果")
    print("=" * 80)
    print(f"  输出文件: {output_path}")
    print(f"  记录数: {len(df_step2):,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
