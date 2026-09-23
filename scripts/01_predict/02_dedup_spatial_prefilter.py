#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deduplicate CEMP candidates by DESIG and apply the |b| > 30 deg spatial pref filter.

Merged from the former step-02 (dedup by DESIG + append DESIG/OBJNAME from FITS
headers) and step-03 (compute Galactic latitude/longitude and keep |b| > 30 deg).
Order is irrelevant because the Galactic latitude b is an intrinsic property of a
target: all observations sharing a DESIG have the same coordinates and therefore
the same b, so dedup-by-DESIG and the b cut commute.

Input:   cemp_all.csv   (all CEMP predictions from step 01)
Output:  cemp_unique_b_greater_30.csv  (dedup by DESIG, |b| > 30 deg kept)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u

LGADNET_ROOT = Path(os.environ.get("LGADNET_ROOT", "/path/to/lgadnet_data"))
INTER = LGADNET_ROOT / "crossmatch"
PRED = LGADNET_ROOT / "predictions"

DEFAULT_INPUT = PRED / "cemp_all.csv"
DEFAULT_OUTPUT = INTER / "cemp_unique_b_greater_30.csv"

INSERT_AFTER = "source_path"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--b-threshold", type=float, default=30.0,
                        help="Galactic latitude threshold in degrees (default 30)")
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
    if "ra" not in df.columns or "dec" not in df.columns:
        raise ValueError(f"input CSV is missing ra/dec columns: {input_path}")

    # --- Galactic coordinates ---
    coords = SkyCoord(ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg,
                      frame="icrs")
    gal = coords.galactic
    df["b"] = gal.b.deg
    df["l"] = gal.l.deg

    # --- Extract DESIG / OBJNAME from the FITS headers ---
    df["DESIG"] = df["source_path"].astype(str).map(lambda p: read_header_value(p, "DESIG"))
    df["OBJNAME"] = df["source_path"].astype(str).map(lambda p: read_header_value(p, "OBJNAME"))

    # --- Deduplicate by DESIG (keep the representative observation) ---
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

    # --- Spatial filter: |b| > threshold ---
    filter_mask = np.abs(dedup["b"]) > args.b_threshold
    filtered = dedup[filter_mask].copy()

    # --- Output columns ---
    output_columns = build_output_columns(list(df.columns))
    if "cemp_margin" not in df.columns and "cemp_cfe_threshold" in df.columns:
        idx = df.columns.get_loc("cemp_cfe_threshold") + 1
        output_columns.insert(idx, "cemp_margin")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, index=False, columns=output_columns)

    print(f"input_rows={len(df)}")
    print(f"unique_desig_rows={len(dedup)}")
    print(f"kept_after_b_cut={len(filtered)}")
    print(f"removed_dedup={len(df) - len(dedup)}")
    print(f"removed_b_cut={len(dedup) - len(filtered)}")
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())