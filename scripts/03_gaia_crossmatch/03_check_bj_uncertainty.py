#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check Bailer-Jones distance uncertainty and robust spatial/kinematic selection.

Input:
    cemp_with_bailer_jones_distance_compare.csv

Outputs:
    bailer_jones_distance_uncertainty_flags.csv
    bailer_jones_distance_uncertainty_summary.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/"
    "spatial_filtering_unique/cemp_with_bailer_jones_distance_compare.csv"
)
OUTPUT_CSV = Path(
    "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/"
    "spatial_filtering_unique/bailer_jones_distance_uncertainty_flags.csv"
)
OUTPUT_REPORT = Path(
    "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/"
    "spatial_filtering_unique/bailer_jones_distance_uncertainty_summary.txt"
)

Z_SUN_KPC = 0.0208
ABS_Z_THRESHOLD = 3.0
VTAN_THRESHOLD = 180.0


def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main() -> int:
    df = pd.read_csv(INPUT)

    # Keep Source safe as string
    if "Source" in df.columns:
        df["Source"] = df["Source"].astype(str).str.strip()

    df = to_numeric(
        df,
        [
            "r_med_geo",
            "r_lo_geo",
            "r_hi_geo",
            "r_med_photogeo",
            "r_lo_photogeo",
            "r_hi_photogeo",
            "gaia_b",
            "pmRA",
            "pmDE",
            "Plx",
            "e_Plx",
            "d_plx_kpc",
            "d_bj_geo_kpc",
            "abs_Z_plx_kpc",
            "Vtan_plx_kms",
            "abs_Z_bj_geo_kpc",
            "Vtan_bj_geo_kms",
        ],
    )

    # Basic BJ availability
    df["has_bj_geo"] = df["r_med_geo"].notna()
    df["has_bj_geo_interval"] = (
        df["r_med_geo"].notna()
        & df["r_lo_geo"].notna()
        & df["r_hi_geo"].notna()
        & (df["r_med_geo"] > 0)
    )

    # Relative half-width of BJ distance interval
    df["bj_geo_rel_half_width"] = np.where(
        df["has_bj_geo_interval"],
        (df["r_hi_geo"] - df["r_lo_geo"]) / (2.0 * df["r_med_geo"]),
        np.nan,
    )

    df["bj_geo_rel_width_lt_0p2"] = df["bj_geo_rel_half_width"] < 0.2
    df["bj_geo_rel_width_lt_0p3"] = df["bj_geo_rel_half_width"] < 0.3
    df["bj_geo_rel_width_lt_0p5"] = df["bj_geo_rel_half_width"] < 0.5

    # Distances from BJ lower/upper interval, pc -> kpc
    df["d_bj_lo_kpc"] = df["r_lo_geo"] / 1000.0
    df["d_bj_hi_kpc"] = df["r_hi_geo"] / 1000.0

    b_rad = np.deg2rad(df["gaia_b"])
    df["mu_masyr"] = np.sqrt(df["pmRA"] ** 2 + df["pmDE"] ** 2)

    # Lower and upper bound induced Z/Vtan
    df["Z_bj_lo_kpc"] = df["d_bj_lo_kpc"] * np.sin(b_rad) + Z_SUN_KPC
    df["abs_Z_bj_lo_kpc"] = np.abs(df["Z_bj_lo_kpc"])
    df["Vtan_bj_lo_kms"] = 4.74047 * df["mu_masyr"] * df["d_bj_lo_kpc"]

    df["Z_bj_hi_kpc"] = df["d_bj_hi_kpc"] * np.sin(b_rad) + Z_SUN_KPC
    df["abs_Z_bj_hi_kpc"] = np.abs(df["Z_bj_hi_kpc"])
    df["Vtan_bj_hi_kms"] = 4.74047 * df["mu_masyr"] * df["d_bj_hi_kpc"]

    # Median-distance BJ selection
    df["bj_med_absZ_pass"] = df["has_bj_geo"] & (
        df["abs_Z_bj_geo_kpc"] > ABS_Z_THRESHOLD
    )
    df["bj_med_vtan_pass"] = df["has_bj_geo"] & (
        df["Vtan_bj_geo_kms"] > VTAN_THRESHOLD
    )
    df["bj_med_both_pass"] = df["bj_med_absZ_pass"] & df["bj_med_vtan_pass"]

    # Robust selection: even using lower distance bound still passes both cuts
    df["bj_lo_absZ_pass"] = df["has_bj_geo_interval"] & (
        df["abs_Z_bj_lo_kpc"] > ABS_Z_THRESHOLD
    )
    df["bj_lo_vtan_pass"] = df["has_bj_geo_interval"] & (
        df["Vtan_bj_lo_kms"] > VTAN_THRESHOLD
    )
    df["bj_lo_both_pass"] = df["bj_lo_absZ_pass"] & df["bj_lo_vtan_pass"]

    # Borderline: median passes, lower bound fails
    df["bj_borderline_absZ"] = df["bj_med_absZ_pass"] & ~df["bj_lo_absZ_pass"]
    df["bj_borderline_vtan"] = df["bj_med_vtan_pass"] & ~df["bj_lo_vtan_pass"]
    df["bj_borderline_both"] = df["bj_med_both_pass"] & ~df["bj_lo_both_pass"]

    # Conservative uncertainty-qualified selections
    df["bj_med_both_rel02"] = df["bj_med_both_pass"] & df["bj_geo_rel_width_lt_0p2"]
    df["bj_med_both_rel03"] = df["bj_med_both_pass"] & df["bj_geo_rel_width_lt_0p3"]
    df["bj_med_both_rel05"] = df["bj_med_both_pass"] & df["bj_geo_rel_width_lt_0p5"]

    # Compare to current plx selection if present
    if "plx_both_pass" in df.columns:
        # csv may read bools as bool or str; normalize
        if df["plx_both_pass"].dtype == object:
            df["plx_both_pass_bool"] = df["plx_both_pass"].astype(str).str.lower().eq("true")
        else:
            df["plx_both_pass_bool"] = df["plx_both_pass"].fillna(False).astype(bool)
    else:
        df["plx_both_pass_bool"] = False

    common_plx_bj_lo = df["plx_both_pass_bool"] & df["bj_lo_both_pass"]
    plx_only_vs_bj_lo = df["plx_both_pass_bool"] & ~df["bj_lo_both_pass"]
    bj_lo_only_vs_plx = ~df["plx_both_pass_bool"] & df["bj_lo_both_pass"]

    # Report
    lines = []
    add = lines.append
    add("=" * 80)
    add("Bailer-Jones Distance Uncertainty and Robust Selection Summary")
    add("=" * 80)
    add(f"Total candidates: {len(df):,}")
    add(f"Has BJ r_med_geo: {df['has_bj_geo'].sum():,}")
    add(f"Has BJ r_lo/r_med/r_hi interval: {df['has_bj_geo_interval'].sum():,}")
    add("")
    add("BJ geometric distance relative half-width:")
    add(f"  < 20%: {df['bj_geo_rel_width_lt_0p2'].sum():,}")
    add(f"  < 30%: {df['bj_geo_rel_width_lt_0p3'].sum():,}")
    add(f"  < 50%: {df['bj_geo_rel_width_lt_0p5'].sum():,}")
    valid_width = df["bj_geo_rel_half_width"].dropna()
    if len(valid_width) > 0:
        add(f"  median: {valid_width.median():.3f}")
        add(f"  16%-84%: {valid_width.quantile(0.16):.3f} - {valid_width.quantile(0.84):.3f}")
        add(f"  5%-95%: {valid_width.quantile(0.05):.3f} - {valid_width.quantile(0.95):.3f}")
    add("")
    add("BJ median-distance selection:")
    add(f"  |Z| > 3 kpc: {df['bj_med_absZ_pass'].sum():,}")
    add(f"  Vtan > 180 km/s: {df['bj_med_vtan_pass'].sum():,}")
    add(f"  both: {df['bj_med_both_pass'].sum():,}")
    add("")
    add("BJ lower-bound robust selection:")
    add(f"  |Z_lo| > 3 kpc: {df['bj_lo_absZ_pass'].sum():,}")
    add(f"  Vtan_lo > 180 km/s: {df['bj_lo_vtan_pass'].sum():,}")
    add(f"  both: {df['bj_lo_both_pass'].sum():,}")
    add("")
    add("BJ median passes but lower bound fails:")
    add(f"  |Z| borderline: {df['bj_borderline_absZ'].sum():,}")
    add(f"  Vtan borderline: {df['bj_borderline_vtan'].sum():,}")
    add(f"  both borderline: {df['bj_borderline_both'].sum():,}")
    add("")
    add("BJ median both-pass with relative distance half-width cuts:")
    add(f"  rel half-width < 20%: {df['bj_med_both_rel02'].sum():,}")
    add(f"  rel half-width < 30%: {df['bj_med_both_rel03'].sum():,}")
    add(f"  rel half-width < 50%: {df['bj_med_both_rel05'].sum():,}")
    add("")
    add("Comparison: current plx both-pass vs BJ lower-bound robust both-pass:")
    add(f"  current plx both-pass: {df['plx_both_pass_bool'].sum():,}")
    add(f"  BJ lower-bound robust both-pass: {df['bj_lo_both_pass'].sum():,}")
    add(f"  common: {common_plx_bj_lo.sum():,}")
    add(f"  current only: {plx_only_vs_bj_lo.sum():,}")
    add(f"  BJ robust only: {bj_lo_only_vs_plx.sum():,}")

    report = "\n".join(lines)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_REPORT.write_text(report, encoding="utf-8")

    print(report)
    print("=" * 80)
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Output report: {OUTPUT_REPORT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
