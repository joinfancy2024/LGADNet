#!/usr/bin/env python3
"""Step 1: Phase-space quantities (Z, Vtan) for all candidates.

Reads the upstream Gaia cross-match + Bailer-Jones geometric median
distances (r_med_geo), computes vertical height Z = d*sin(b) + Z_sun
and tangential velocity Vtan = 4.74047*mu*d, then applies the phase-space
selection |Z| > 3 kpc AND Vtan > 180 km/s.

Input:  vizier_dedup_double.csv + bailer_jones_distance_for_candidates.csv
Output: phase.csv (all candidates with derived quantities)
        candidates.csv (phase-space-selected candidates, feeds step 02)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

LGADNET_ROOT = Path(os.environ.get("LGADNET_ROOT", "/path/to/lgadnet_data"))
UPSTREAM = LGADNET_ROOT / "crossmatch"
DEFAULT_GAIA_CSV = UPSTREAM / "vizier_dedup_double.csv"
DEFAULT_BJ_CSV = UPSTREAM / "bailer_jones_distance_for_candidates.csv"
ABS_Z_THRESHOLD = 3.0
VTAN_THRESHOLD = 180.0
K_VTAN = 4.74047


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zsun", type=float, default=0.0208,
                   help="solar height (kpc)")
    p.add_argument("--gaia-csv", type=Path, default=DEFAULT_GAIA_CSV)
    p.add_argument("--bj-csv", type=Path, default=DEFAULT_BJ_CSV)
    p.add_argument("--outdir", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    zsun = args.zsun
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.gaia_csv)

    # 1) Galactic coordinates (needed for Z = d*sin(b) + Zsun)
    coords = SkyCoord(ra=df["RAdeg"].values * u.deg, dec=df["DEdeg"].values * u.deg,
                      frame="icrs")
    gal = coords.galactic
    df["gaia_b"] = gal.b.deg
    df["gaia_l"] = gal.l.deg
    b_rad = np.deg2rad(df["gaia_b"])

    # 2) Bailer-Jones median geometric distance (r_med_geo, pc -> kpc)
    bj = pd.read_csv(args.bj_csv)
    bj_cols = ["r_med_geo"]
    df = df.merge(bj[["Source"] + bj_cols], on="Source", how="left",
                  indicator="bj_merge")

    # 3) Distance in kpc + has-distance flag
    df["d_bj_geo_kpc"] = df["r_med_geo"] / 1000.0
    df["has_bj_distance"] = df["r_med_geo"].notna()

    # 4) Proper motion amplitude (mas/yr)
    df["mu_masyr"] = np.sqrt(df["pmRA"] ** 2 + df["pmDE"] ** 2)

    # 5) Derived phase-space quantities (BJ geom. median distance)
    df["Z_BJ_med_kpc"] = df["d_bj_geo_kpc"] * np.sin(b_rad) + zsun
    df["abs_Z_BJ_med_kpc"] = np.abs(df["Z_BJ_med_kpc"])
    df["Vtan_BJ_med_kms"] = K_VTAN * df["mu_masyr"] * df["d_bj_geo_kpc"]

    # 6) Selection flags (BJ geom. median distance, no parallax quality gate)
    df["phase_absZ_pass"] = df["has_bj_distance"] & (df["abs_Z_BJ_med_kpc"] > ABS_Z_THRESHOLD)
    df["phase_vtan_pass"] = df["has_bj_distance"] & (df["Vtan_BJ_med_kms"] > VTAN_THRESHOLD)
    df["phase_both_pass"] = df["phase_absZ_pass"] & df["phase_vtan_pass"]

    # 7) Save
    out = args.outdir / "phase.csv"
    df.to_csv(out, index=False)

    # 8) Selection candidate file
    n_pass = int(df["phase_both_pass"].sum())
    robust = df[df["phase_both_pass"]].copy()
    robust_out = args.outdir / "candidates.csv"
    robust.to_csv(robust_out, index=False)

    print(f"zsun            = {zsun}")
    print(f"rows            = {len(df)}")
    print(f"has_bj_distance = {int(df['has_bj_distance'].sum())}")
    print(f"|Z| > 3         = {int(df['phase_absZ_pass'].sum())}")
    print(f"Vtan > 180      = {int(df['phase_vtan_pass'].sum())}")
    print(f"both pass       = {int(df['phase_both_pass'].sum())}")
    print(f"written         = {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())