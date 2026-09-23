#!/usr/bin/env python3
"""Double-deduplicate the VizieR cross-match result to a 1:1 LAMOST-Gaia mapping.

Level 1: keep the nearest Gaia source (smallest angDist) per LAMOST candidate_id.
Level 2: keep the nearest candidate per Gaia Source.
The result is unique on both candidate_id and Source.

Input:  vizier_crossmatch_result.csv  (VizieR match table, see step 03)
Output: vizier_dedup_double.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

LGADNET_ROOT = Path(os.environ.get("LGADNET_ROOT", "/path/to/lgadnet_data"))
INTER = LGADNET_ROOT / "crossmatch"

DEFAULT_INPUT = INTER / "vizier_crossmatch_result.csv"
DEFAULT_OUTPUT = INTER / "vizier_dedup_double.csv"


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
    required_cols = ["candidate_id", "angDist", "Source"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"input CSV is missing required columns: {missing}")

    df["angDist"] = pd.to_numeric(df["angDist"], errors="coerce")
    df = df.dropna(subset=["angDist"])

    # Level 1: one nearest Gaia source per LAMOST candidate.
    df = df.sort_values(["candidate_id", "angDist"]).drop_duplicates(
        "candidate_id", keep="first"
    )
    # Level 2: one nearest candidate per Gaia source.
    df = df.sort_values(["Source", "angDist"]).drop_duplicates(
        "Source", keep="first"
    )

    if not (df["candidate_id"].nunique() == len(df) and df["Source"].nunique() == len(df)):
        raise RuntimeError("after dedup, candidate_id or Source is not unique")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"rows_after_dedup={len(df)}")
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())