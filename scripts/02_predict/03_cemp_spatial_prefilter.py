#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CEMP spatial prefilter: compute Galactic latitude/longitude and keep candidates with |b| > 30 deg."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u


LGADNET_ROOT = Path("/path/to/your/lgadnet_data")   # <-- EDIT THIS root

DEFAULT_INPUT = (
    LGADNET_ROOT / "predictions" / "dr12_cemp_unique_by_desig.csv"
)
DEFAULT_OUTPUT_DIR = LGADNET_ROOT / "spatial_filtering_unique"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--b-threshold", type=float, default=30.0, help="Galactic latitude threshold in degrees (default 30)")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # read the input
    print(f"Reading: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Input samples: {len(df):,}")

    # compute Galactic coordinates
    print("Computing Galactic latitude and longitude...")
    coords = SkyCoord(ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg, frame="icrs")
    gal = coords.galactic
    df["b"] = gal.b.deg
    df["l"] = gal.l.deg

    print(f"Galactic latitude range: {df['b'].min():.1f} deg ~ {df['b'].max():.1f} deg")

    # spatial filter
    filter_mask = np.abs(df["b"]) > args.b_threshold
    filtered = df[filter_mask].copy()
    print(f"|b| > {args.b_threshold} deg: {len(filtered):,} kept ({len(filtered) / len(df) * 100:.1f}%)")

    # output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / "cemp_unique_b_greater_30.csv"

    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"output file already exists; use --overwrite to replace: {output_file}")

    filtered.to_csv(output_file, index=False)
    print(f"Output: {output_file}")
    print(f"Done: {len(filtered):,} CEMP candidates")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
