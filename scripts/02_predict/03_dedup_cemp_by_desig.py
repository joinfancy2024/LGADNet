#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 DESIG 对 CEMP 候选去重，保留每个目标的代表观测。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(
    "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/"
    "cemp_pipeline_lgadnet_v6_cemp3020_bs3072_labelscale_fixed/dr12_cemp_all_with_desig.csv"
)
DEFAULT_OUTPUT = Path(
    "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/"
    "cemp_pipeline_lgadnet_v6_cemp3020_bs3072_labelscale_fixed/dr12_cemp_unique_by_desig.csv"
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

    if not input_path.is_file():
        raise FileNotFoundError(f"输入 CSV 不存在: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"输出文件已存在，请换路径或加 --overwrite: {output_path}")

    df = pd.read_csv(input_path)
    if "DESIG" not in df.columns:
        raise ValueError(f"输入 CSV 缺少 DESIG 列: {input_path}")

    # === 构建排序列 ===
    work = df.copy()
    work["DESIG"] = work["DESIG"].astype("string").fillna("").str.strip()

    # CEMP边际值：C_FE超出阈值的幅度（越大越优先）
    work["cemp_margin"] = pd.to_numeric(work["C_FE"], errors="coerce") - pd.to_numeric(
        work["cemp_cfe_threshold"], errors="coerce"
    )

    # 辅助排序列
    work["snr_sort"] = (
        pd.to_numeric(work["snrg"], errors="coerce")
        if "snrg" in work.columns
        else 0.0
    )
    work["lmjd_sort"] = pd.to_numeric(work.get("lmjd"), errors="coerce")
    work["obsid_sort"] = pd.to_numeric(work.get("obsid"), errors="coerce")

    # === 排序并去重 ===
    # 优先级：cemp_margin(降序) > snr(降序) > lmjd(降序) > obsid(降序)
    work = work.sort_values(
        by=["DESIG", "cemp_margin", "snr_sort", "lmjd_sort", "obsid_sort"],
        ascending=[True, False, False, False, False],
        na_position="last",
    )
    dedup = work.drop_duplicates(subset="DESIG", keep="first").copy()

    # 清理辅助列
    dedup = dedup.drop(columns=["snr_sort", "lmjd_sort", "obsid_sort"])

    # === 确定输出列顺序 ===
    output_columns = list(df.columns)
    if "cemp_margin" not in df.columns and "cemp_cfe_threshold" in df.columns:
        insert_idx = df.columns.get_loc("cemp_cfe_threshold") + 1
        output_columns.insert(insert_idx, "cemp_margin")

    # === 输出 ===
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dedup.to_csv(output_path, index=False, columns=output_columns)

    print(f"input_rows={len(df)}")
    print(f"unique_desig_rows={len(dedup)}")
    print(f"removed_rows={len(df) - len(dedup)}")
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
