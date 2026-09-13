#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Double-deduplication script: two-level deduplication of the VizieR cross-match results.

Deduplication strategy:
1. Level 1: deduplicate by candidate_id (keep only the nearest Gaia match for each LAMOST candidate)
2. Level 2: deduplicate by Source (keep only the nearest candidate for each Gaia source)

Input: VizieR returned cross-match result CSV
Output: CSV file after double deduplication
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LGADNET_ROOT = Path("/path/to/your/lgadnet_data")   # <-- EDIT THIS root
DATA = LGADNET_ROOT

# Input is the cross-match result CSV returned by VizieR after uploading
# vizier_upload_with_id.csv. Save your downloaded result under this name, or
# override with --input.
DEFAULT_INPUT = DATA / "spatial_filtering_unique" / "vizier_crossmatch_result.csv"
DEFAULT_OUTPUT = DATA / "spatial_filtering_unique" / "vizier_dedup_double.csv"


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

    # Check the input file
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists; use a different path or add --overwrite: {output_path}")

    # Load data
    print("=" * 80)
    print("VizieR cross-match double deduplication")
    print("=" * 80)
    print(f"\nInput file: {input_path}")

    df = pd.read_csv(input_path)
    print(f"Number of raw records: {len(df):,}")

    # Check required columns
    required_cols = ['candidate_id', 'angDist', 'Source']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input CSV is missing required columns: {missing_cols}")

    # Convert angDist to numeric type
    df['angDist'] = pd.to_numeric(df['angDist'], errors='coerce')
    df = df.dropna(subset=['angDist'])

    # ============================================================
    # Level 1 deduplication: by candidate_id (keep the nearest Gaia match for each LAMOST candidate)
    # ============================================================
    print("\n" + "-" * 80)
    print("Level 1 deduplication: by candidate_id (keep the smallest angDist)")
    print("-" * 80)

    df_step1 = df.sort_values(['candidate_id', 'angDist']).drop_duplicates(
        'candidate_id', keep='first'
    )
    removed_step1 = len(df) - len(df_step1)
    print(f"  Raw records: {len(df):,}")
    print(f"  Records after deduplication: {len(df_step1):,}")
    print(f"  Removed records: {removed_step1:,}")
    print(f"  Unique candidate_id: {df_step1['candidate_id'].nunique():,}")

    # ============================================================
    # Level 2 deduplication: by Source (keep only the nearest candidate for each Gaia source)
    # ============================================================
    print("\n" + "-" * 80)
    print("Level 2 deduplication: by Source (keep the smallest angDist)")
    print("-" * 80)

    # Check Source duplicate counts
    source_counts = df_step1['Source'].value_counts()
    multi_source = source_counts[source_counts > 1]
    print(f"  Count of duplicate Sources: {len(multi_source):,}")
    print(f"  Records involved: {multi_source.sum():,}")

    # Perform level 2 deduplication
    df_step2 = df_step1.sort_values(['Source', 'angDist']).drop_duplicates(
        'Source', keep='first'
    )
    removed_step2 = len(df_step1) - len(df_step2)
    print(f"  After level 1 deduplication: {len(df_step1):,}")
    print(f"  After level 2 deduplication: {len(df_step2):,}")
    print(f"  Removed records: {removed_step2:,}")
    print(f"  Unique Source: {df_step2['Source'].nunique():,}")

    # ============================================================
    # Summary statistics
    # ============================================================
    print("\n" + "=" * 80)
    print("Deduplication summary")
    print("=" * 80)
    print(f"  Raw records: {len(df):,}")
    print(f"  Removed by level 1: {removed_step1:,} (duplicate candidate_id)")
    print(f"  Removed by level 2: {removed_step2:,} (duplicate Source)")
    print(f"  Final records: {len(df_step2):,}")
    print(f"  Total removed records: {len(df) - len(df_step2):,}")
    print(f"  Retention ratio: {len(df_step2)/len(df)*100:.2f}%")

    # Verify uniqueness
    print("\n" + "-" * 80)
    print("Uniqueness verification")
    print("-" * 80)
    print(f"  candidate_id unique: {df_step2['candidate_id'].nunique() == len(df_step2)}")
    print(f"  Source unique: {df_step2['Source'].nunique() == len(df_step2)}")

    # ============================================================
    # Save results
    # ============================================================
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_step2.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("Output")
    print("=" * 80)
    print(f"  Output file: {output_path}")
    print(f"  Records: {len(df_step2):,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())