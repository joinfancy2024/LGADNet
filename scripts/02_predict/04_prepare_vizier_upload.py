#!/usr/bin/env python3
"""Prepare the upload file for VizieR cross-matching."""
import pandas as pd
from pathlib import Path

LGADNET_ROOT = Path("/path/to/your/lgadnet_data")   # <-- EDIT THIS root

# configuration
INPUT_FILE = LGADNET_ROOT / "spatial_filtering_unique" / "cemp_unique_b_greater_30.csv"
OUTPUT_DIR = LGADNET_ROOT / "spatial_filtering_unique"


def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # read the data
    print(f"Reading: {INPUT_FILE.name}")
    df = pd.read_csv(INPUT_FILE)
    print(f"Candidate count: {len(df):,}")

    # ensure an ID column exists
    if 'candidate_id' not in df.columns:
        df['candidate_id'] = range(len(df))

    # save the upload file
    output_file = OUTPUT_DIR / "vizier_upload_with_id.csv"
    df[['candidate_id', 'ra', 'dec']].to_csv(output_file, index=False)
    print(f"Saved: {output_file.name}")
    print("Done!")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
