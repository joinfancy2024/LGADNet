#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Append DESIG/OBJNAME from FITS headers to the CEMP candidates and
deduplicate them by DESIG, keeping the representative observation per target."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from astropy.io import fits


LGADNET_ROOT = Path("/path/to/your/lgadnet_data")   # <-- EDIT THIS root

DEFAULT_INPUT = LGADNET_ROOT / "predictions" / "dr12_cemp_all.csv"
DEFAULT_OUTPUT = LGADNET_ROOT / "predictions" / "dr12_cemp_unique_by_desig.csv"

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
    # Always move DESIG/OBJNAME to right after source_path (drop any
    # pre-existing occurrence so the position is authoritative).
    cols = [c for c in columns if c not in ("DESIG", "OBJNAME")]
    output_columns: list[str] = []
    inserted = False
    for column in cols:
        output_columns.append(column)
        if column == INSERT_AFTER:
            output_columns.extend(["DESIG", "OBJNAME"])
            inserted = True
    if not inserted:
        output_columns.extend(["DESIG", "OBJNAME"])
    return output_columns


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"input CSV does not exist: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"output file already exists; pick another path or add --overwrite: {output_path}"
        )

    df = pd.read_csv(input_path)
    if "source_path" not in df.columns:
        raise ValueError(f"input CSV is missing the source_path column: {input_path}")

    # Extract DESIG / OBJNAME from the FITS headers.
    df["DESIG"] = df["source_path"].astype(str).map(lambda p: read_header_value(p, "DESIG"))
    df["OBJNAME"] = df["source_path"].astype(str).map(lambda p: read_header_value(p, "OBJNAME"))

    # Build the sort key and deduplicate by DESIG (keep the representative observation).
    work = df.copy()
    work["DESIG"] = work["DESIG"].astype("string").fillna("").str.strip()
    work["cemp_margin"] = pd.to_numeric(work["C_FE"], errors="coerce") - pd.to_numeric(
        work["cemp_cfe_threshold"], errors="coerce"
    )
    work["snr_sort"] = (
        pd.to_numeric(work["snrg"], errors="coerce") if "snrg" in work.columns else 0.0
    )
    work["lmjd_sort"] = pd.to_numeric(work.get("lmjd"), errors="coerce")
    work["obsid_sort"] = pd.to_numeric(work.get("obsid"), errors="coerce")

    work = work.sort_values(
        by=["DESIG", "cemp_margin", "snr_sort", "lmjd_sort", "obsid_sort"],
        ascending=[True, False, False, False, False],
        na_position="last",
    )
    dedup = work.drop_duplicates(subset="DESIG", keep="first").copy()
    dedup = dedup.drop(columns=["snr_sort", "lmjd_sort", "obsid_sort"])

    output_columns = build_output_columns(list(df.columns))
    if "cemp_margin" not in df.columns and "cemp_cfe_threshold" in df.columns:
        idx = df.columns.get_loc("cemp_cfe_threshold") + 1
        output_columns.insert(idx, "cemp_margin")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dedup.to_csv(output_path, index=False, columns=output_columns)

    print(f"input_rows={len(df)}")
    print(f"unique_desig_rows={len(dedup)}")
    print(f"removed_rows={len(df) - len(dedup)}")
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())