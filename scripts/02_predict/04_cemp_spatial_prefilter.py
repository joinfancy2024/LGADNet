#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CEMP候选空间预筛选：计算银纬银经，筛选 |b| > 30° 的候选。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u


ROOT_DIR = Path("/home/DM13/workspace/sky")
LGADNET_ROOT = ROOT_DIR / "data/new_dataset3/lgadnet"

DEFAULT_INPUT = (
    LGADNET_ROOT / "cemp_pipeline_lgadnet_v6_cemp3020_bs3072_labelscale_fixed/dr12_cemp_unique_by_desig.csv"
)
DEFAULT_OUTPUT_DIR = LGADNET_ROOT / "spatial_filtering_unique"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--b-threshold", type=float, default=30.0, help="银纬阈值（默认30°）")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 读取输入
    print(f"读取: {args.input}")
    df = pd.read_csv(args.input)
    print(f"输入样本数: {len(df):,}")

    # 计算银道坐标
    print("计算银纬和银经...")
    coords = SkyCoord(ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg, frame="icrs")
    gal = coords.galactic
    df["b"] = gal.b.deg
    df["l"] = gal.l.deg

    print(f"银纬范围: {df['b'].min():.1f}° ~ {df['b'].max():.1f}°")

    # 空间筛选
    filter_mask = np.abs(df["b"]) > args.b_threshold
    filtered = df[filter_mask].copy()
    print(f"筛选 |b| > {args.b_threshold}°: {len(filtered):,} 颗 ({len(filtered) / len(df) * 100:.1f}%)")

    # 输出
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / "cemp_unique_b_greater_30.csv"

    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"输出文件已存在；如需覆盖请加 --overwrite: {output_file}")

    filtered.to_csv(output_file, index=False)
    print(f"输出: {output_file}")
    print(f"完成: {len(filtered):,} 颗 CEMP 候选")


if __name__ == "__main__":
    main()
