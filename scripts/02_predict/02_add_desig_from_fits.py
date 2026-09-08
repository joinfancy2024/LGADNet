#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 FITS header 提取 DESIG/OBJNAME 并补充到现有结果 CSV。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from astropy.io import fits


DEFAULT_INPUT = Path(
    "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/"
    "cemp_pipeline_lgadnet_v6_cemp3020_bs3072_labelscale_fixed/dr12_cemp_all.csv"
)
DEFAULT_OUTPUT = Path(
    "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/"
    "cemp_pipeline_lgadnet_v6_cemp3020_bs3072_labelscale_fixed/dr12_cemp_all_with_desig.csv"
)

INSERT_AFTER = "source_path"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_header_value(path: str, key: str) -> str:
    try:
        with fits.open(path, memmap=False) as hdul:
            value = hdul[0].header.get(key)
    except Exception:
        return ""

    if value is None:
        return ""
    return str(value).strip()


def build_output_columns(columns: list[str]) -> list[str]:
    if "DESIG" in columns and "OBJNAME" in columns:
        return columns

    output_columns: list[str] = []
    inserted = False
    for column in columns:
        output_columns.append(column)
        if column == INSERT_AFTER:
            if "DESIG" not in columns:
                output_columns.append("DESIG")
            if "OBJNAME" not in columns:
                output_columns.append("OBJNAME")
            inserted = True

    if not inserted:
        if "DESIG" not in columns:
            output_columns.append("DESIG")
        if "OBJNAME" not in columns:
            output_columns.append("OBJNAME")
    return output_columns


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"输入 CSV 不存在: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"输出文件已存在，请换路径或加 --overwrite: {output_path}")

    df = pd.read_csv(input_path)
    if "source_path" not in df.columns:
        raise ValueError(f"输入 CSV 缺少 source_path 列: {input_path}")

    df["DESIG"] = df["source_path"].astype(str).map(lambda path: read_header_value(path, "DESIG"))
    df["OBJNAME"] = df["source_path"].astype(str).map(lambda path: read_header_value(path, "OBJNAME"))

    output_columns = build_output_columns(list(df.columns))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, columns=output_columns)

    desig_nonempty = int(df["DESIG"].astype(str).str.len().gt(0).sum())
    objname_nonempty = int(df["OBJNAME"].astype(str).str.len().gt(0).sum())
    print(f"rows={len(df)}")
    print(f"desig_nonempty={desig_nonempty}")
    print(f"objname_nonempty={objname_nonempty}")
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
