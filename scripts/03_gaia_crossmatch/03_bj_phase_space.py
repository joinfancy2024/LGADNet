#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge the Gaia cross-match with Bailer-Jones distances and apply the phase-space
(kinematic) selection.

Combines the double-deduplicated Gaia cross-match (`vizier_dedup_double.csv`) with the
Bailer-Jones geometric distance estimates (`bailer_jones_distance_for_candidates.csv`)
and the original CEMP candidate parameters, computes Galactic latitude and the vertical
height / tangential velocity from both the parallax and the conservative (lower-bound)
Bailer-Jones distance, and keeps candidates satisfying:

    |Z_BJ,lo| > 3 kpc   and   Vtan_BJ,lo > 180 km/s

using the lower confidence bound of the Bailer-Jones distance so the selection is robust.

Outputs (both in `spatial_filtering_unique`):
    cemp_with_bailer_jones_distance_compare.csv  -- full merged table with flags
    cemp_bj_robust_candidates_6928.csv           -- the phase-space-selected subset
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u


LGADNET_ROOT = Path("/path/to/your/lgadnet_data")   # <-- EDIT THIS root
DATA = LGADNET_ROOT

Z_SUN_KPC = 0.0208        # solar height above the Galactic plane (Bennett & Bovy 2019)
ABS_Z_THRESHOLD = 3.0     # vertical-height threshold (kpc)
VTAN_THRESHOLD = 180.0    # tangential-velocity threshold (km/s)
K_VTAN = 4.74047          # 1 mas/yr * 1 kpc -> km/s

DEFAULT_GAIA_CSV = DATA / "spatial_filtering_unique" / "vizier_dedup_double.csv"
DEFAULT_BJ_CSV = DATA / "spatial_filtering_unique" / "bailer_jones_distance_for_candidates.csv"
DEFAULT_CEMP_CSV = DATA / "spatial_filtering_unique" / "cemp_unique_b_greater_30.csv"
DEFAULT_OUTPUT_DIR = DATA / "spatial_filtering_unique"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaia-csv", type=Path, default=DEFAULT_GAIA_CSV,
                        help="double-deduplicated Gaia cross-match CSV")
    parser.add_argument("--bj-csv", type=Path, default=DEFAULT_BJ_CSV,
                        help="Bailer-Jones distance CSV")
    parser.add_argument("--cemp-csv", type=Path, default=DEFAULT_CEMP_CSV,
                        help="original CEMP candidate CSV (source_path, z, parameters)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def pick_first(df: pd.DataFrame, names: list[str]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def main() -> int:
    args = parse_args()
    gaia_path = args.gaia_csv.resolve()
    bj_path = args.bj_csv.resolve()
    cemp_path = args.cemp_csv.resolve()

    for p in (gaia_path, bj_path, cemp_path):
        if not p.is_file():
            raise FileNotFoundError(f"input CSV does not exist: {p}")

    # Load the Gaia cross-match candidates.
    df = pd.read_csv(gaia_path)
    required = ["candidate_id", "Source"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Gaia cross-match CSV is missing columns: {missing}")
    if "angDist" in df.columns:
        df["angDist"] = pd.to_numeric(df["angDist"], errors="coerce")

    # Load and attach the Bailer-Jones distances (match on Source).
    bj = pd.read_csv(bj_path)
    source_col = "Source"
    cols = ["r_med_geo", "r_lo_geo", "r_hi_geo",
            "r_med_photogeo", "r_lo_photogeo", "r_hi_photogeo", "bj_flag"]
    present_bj = [c for c in cols if c in bj.columns]
    merge_keys = [source_col]
    df = df.merge(bj[merge_keys + present_bj], on=source_col, how="left")

    # Attach the original CEMP candidate parameters (source_path, z, LAMOST params).
    cemp = pd.read_csv(cemp_path)
    cemp_cols = [c for c in ["source_path", "z", "LOGG", "TEFF", "C_FE", "FE_H",
                             "lamost_class", "lamost_subclass"] if c in cemp.columns]
    df = df.merge(cemp[["candidate_id"] + cemp_cols], on="candidate_id", how="left")
    if "candidate_id" in df.columns:
        df["candidate_id"] = pd.to_numeric(df["candidate_id"], errors="coerce").astype("Int64")

    # Galactic latitude from the Gaia coordinates.
    ra_col = pick_first(df, ["RAJ2000", "ra", "RA"])
    dec_col = pick_first(df, ["DEJ2000", "dec", "DEC"])
    has_radec = ra_col is not None and dec_col is not None
    if has_radec:
        coords = SkyCoord(ra=df[ra_col].values * u.deg,
                          dec=df[dec_col].values * u.deg, frame="icrs")
        gal = coords.galactic
        df["gaia_b"] = gal.b.deg
        df["gaia_l"] = gal.l.deg
    else:
        raise ValueError("Gaia cross-match CSV has no RA/Dec column for Galactic latitude")

    # Numeric coercion.
    pmra_col = pick_first(df, ["pmRA", "pmra"])
    pmde_col = pick_first(df, ["pmDE", "pmde"])
    num_cols = ["Plx", "e_Plx", "pmRA", "pmDE", "ruwe", "RUWE",
                "r_med_geo", "r_lo_geo", "r_hi_geo",
                "r_med_photogeo", "r_lo_photogeo", "r_hi_photogeo"]
    to_numeric(df, num_cols)

    # Distance, vertical height and tangential velocity.
    b_rad = np.deg2rad(df["gaia_b"])
    df["mu_masyr"] = np.sqrt(df["pmRA"] ** 2 + df["pmDE"] ** 2)

    # Parallax-based distance.
    df["d_plx_kpc"] = np.where(df["Plx"] > 0, 1.0 / df["Plx"], np.nan)
    df["abs_Z_plx_kpc"] = np.abs(df["d_plx_kpc"] * np.sin(b_rad) + Z_SUN_KPC)
    df["Vtan_plx_kms"] = K_VTAN * df["mu_masyr"] * df["d_plx_kpc"]

    # Bailer-Jones median-distance height / velocity.
    df["d_bj_geo_kpc"] = df["r_med_geo"] / 1000.0
    df["abs_Z_bj_geo_kpc"] = np.abs(df["d_bj_geo_kpc"] * np.sin(b_rad) + Z_SUN_KPC)
    df["Vtan_bj_geo_kms"] = K_VTAN * df["mu_masyr"] * df["d_bj_geo_kpc"]

    # Bailer-Jones lower-confidence-bound (robust) height / velocity.
    df["d_bj_lo_kpc"] = df["r_lo_geo"] / 1000.0
    df["abs_Z_bj_lo_kpc"] = np.abs(df["d_bj_lo_kpc"] * np.sin(b_rad) + Z_SUN_KPC)
    df["Vtan_bj_lo_kms"] = K_VTAN * df["mu_masyr"] * df["d_bj_lo_kpc"]

    # Relative half-width of the BJ distance interval (catalog column).
    df["bj_geo_rel_half_width"] = np.where(
        df["has_bj_geo_interval"],
        (df["r_hi_geo"] - df["r_lo_geo"]) / (2.0 * df["r_med_geo"]),
        np.nan,
    )

    # Availability flags and phase-space selections.
    df["has_bj_geo"] = df["r_med_geo"].notna()
    df["has_bj_geo_interval"] = (
        df["r_med_geo"].notna() & df["r_lo_geo"].notna()
        & df["r_hi_geo"].notna() & (df["r_med_geo"] > 0)
    )

    df["bj_med_absZ_pass"] = df["has_bj_geo"] & (df["abs_Z_bj_geo_kpc"] > ABS_Z_THRESHOLD)
    df["bj_med_vtan_pass"] = df["has_bj_geo"] & (df["Vtan_bj_geo_kms"] > VTAN_THRESHOLD)
    df["bj_med_both_pass"] = df["bj_med_absZ_pass"] & df["bj_med_vtan_pass"]

    df["bj_lo_absZ_pass"] = df["has_bj_geo_interval"] & (df["abs_Z_bj_lo_kpc"] > ABS_Z_THRESHOLD)
    df["bj_lo_vtan_pass"] = df["has_bj_geo_interval"] & (df["Vtan_bj_lo_kms"] > VTAN_THRESHOLD)
    df["bj_lo_both_pass"] = df["bj_lo_absZ_pass"] & df["bj_lo_vtan_pass"]

    # Write the full merged working table.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    compare_out = args.output_dir / "cemp_with_bailer_jones_distance_compare.csv"
    df.to_csv(compare_out, index=False)

    # Phase-space-selected subset (paper selection-chain step 4).
    robust = df[df["bj_lo_both_pass"] == True].copy()  # noqa: E712
    robust_out = args.output_dir / "cemp_bj_robust_candidates_6928.csv"
    robust.to_csv(robust_out, index=False)

    print(f"total_candidates={len(df):,}")
    print(f"has_bj_distance={df['has_bj_geo'].sum():,}")
    print(f"has_bj_interval={df['has_bj_geo_interval'].sum():,}")
    print(f"phase_space_selected={len(robust):,}")
    print(f"written_compare={compare_out}")
    print(f"written_subset={robust_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())