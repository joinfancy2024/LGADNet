#!/usr/bin/env python3
"""
分析原始预测数据：分[Fe/H]区间统计CEMP占比
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# 定义路径
input_file = Path("/home/DM13/workspace/sky/data/new_dataset3/lgadnet/cemp_pipeline_lgadnet_v6_cemp3020_bs3072_labelscale_fixed/dr12_predict_all.csv")
FINAL_OUTPUT_DIR = Path("/home/DM13/workspace/sky/data/new_dataset3/lgadnet/final_validated")
FINAL_OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("分析原始预测数据：CEMP占比统计")
print("=" * 80)

# 分批读取数据（避免内存问题）
print("\n步骤1: 分批读取数据并统计...")
print("-" * 80)

# 定义区间（bin宽度0.4 dex）
bins = [
    (-4.5, -4.1, -4.3),
    (-4.1, -3.7, -3.9),
    (-3.7, -3.3, -3.5),
    (-3.3, -2.9, -3.1),
    (-2.9, -2.5, -2.7),
    (-2.5, -2.1, -2.3),
    (-2.1, -1.7, -1.9),
    (-1.7, -1.3, -1.5),
    (-1.3, -1.0, -1.15),
]

# 初始化统计字典
stats = {}
for low, high, center in bins:
    stats[center] = {
        'cemp': 0,
        'mp_no_cemp': 0,
        'mp_total': 0,
        'other': 0
    }

# 分批读取（每次读取100万行）
chunk_size = 1000000
total_rows = 0

for chunk in pd.read_csv(input_file, chunksize=chunk_size, usecols=['FE_H', 'class']):
    total_rows += len(chunk)
    print(f"  已处理: {total_rows:,}行")

    # 统计每个区间
    for low, high, center in bins:
        mask = (chunk['FE_H'] >= low) & (chunk['FE_H'] < high)

        # 统计各类别
        cemp_count = (mask & (chunk['class'] == 'cemp')).sum()
        mp_no_cemp_count = (mask & (chunk['class'] == 'mp-no-cemp')).sum()
        other_count = (mask & (chunk['class'] == 'other')).sum()

        stats[center]['cemp'] += cemp_count
        stats[center]['mp_no_cemp'] += mp_no_cemp_count
        stats[center]['other'] += other_count

print(f"\n总处理行数: {total_rows:,}")

# 计算MP总数和CEMP占比
print("\n步骤2: 计算CEMP占比...")
print("-" * 80)

results = []
for center in sorted(stats.keys(), reverse=True):
    s = stats[center]

    mp_total = s['cemp'] + s['mp_no_cemp']
    s['mp_total'] = mp_total

    if mp_total > 0:
        cemp_fraction = s['cemp'] / mp_total * 100
    else:
        cemp_fraction = 0

    results.append({
        'feh_center': center,
        'cemp_count': s['cemp'],
        'mp_no_cemp_count': s['mp_no_cemp'],
        'mp_total': mp_total,
        'other_count': s['other'],
        'cemp_fraction': cemp_fraction
    })

# 显示结果
print("\n统计结果:")
print("=" * 80)
print(f"[Fe/H]区间 | CEMP数量 | MP-no-CEMP | MP总数 | CEMP占比(%)")
print("-" * 80)

for r in results:
    print(f"{r['feh_center']:7.2f}   | {r['cemp_count']:8,} | {r['mp_no_cemp_count']:9,} | {r['mp_total']:7,} | {r['cemp_fraction']:6.2f}%")

# 分析封顶值建议
fractions = [r['cemp_fraction'] for r in results if r['mp_total'] > 0]

print("\n封顶值建议:")
print("-" * 80)
print(f"CEMP占比范围: {min(fractions):.2f}% ~ {max(fractions):.2f}%")
print(f"CEMP占比中位数: {np.median(fractions):.2f}%")

# 建议封顶值
max_fraction = max(fractions)
if max_fraction > 60:
    cap_value = 60
elif max_fraction > 50:
    cap_value = 50
else:
    cap_value = max_fraction

print(f"\n建议封顶值: {cap_value}%")
print(f"理由: 最大占比为{max_fraction:.2f}%，设置封顶值{cap_value}%可以让图表更美观")

# 保存统计结果
print("\n步骤3: 保存统计结果...")
print("-" * 80)

output_file = FINAL_OUTPUT_DIR / "cemp_fraction_statistics.csv"
df_stats = pd.DataFrame(results)
df_stats.to_csv(output_file, index=False)
print(f"✓ 保存统计结果: {output_file.name}")

print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)