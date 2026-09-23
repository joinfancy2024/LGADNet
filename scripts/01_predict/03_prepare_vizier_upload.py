#!/usr/bin/env python3
"""Prepare the upload file for VizieR cross-matching.

Reads the dedup'd, spatially-filtered CEMP candidates (cemp_unique_b_greater_30.csv)
and writes the minimal cross-match upload table (candidate_id, ra, dec) as
vizier_upload_with_id.csv.

USAGE NOTE (online cross-match):
    1. Run this script to produce vizier_upload_with_id.csv.
    2. Upload that file to the VizieR cross-match service (e.g. against Gaia DR3).
    3. Download the returned match table and save it as vizier_crossmatch_result.csv
       inside the same directory. The next step (04_dedup_vizier_double.py) reads it.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

LGADNET_ROOT = Path(os.environ.get("LGADNET_ROOT", "/path/to/lgadnet_data"))
INTER = LGADNET_ROOT / "crossmatch"

DEFAULT_INPUT = INTER / "cemp_unique_b_greater_30.csv"
DEFAULT_OUTPUT = INTER / "vizier_upload_with_id.csv"


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
        raise FileNotFoundError(f"input CSV does not exist: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"output file already exists; pick another path or add --overwrite: {output_path}"
        )

    df = pd.read_csv(input_path)
    if 'candidate_id' not in df.columns:
        df['candidate_id'] = range(len(df))
    if 'ra' not in df.columns or 'dec' not in df.columns:
        raise ValueError(f"input CSV is missing ra/dec columns: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[['candidate_id', 'ra', 'dec']].to_csv(output_path, index=False)

    print(f"input_rows={len(df)}")
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())