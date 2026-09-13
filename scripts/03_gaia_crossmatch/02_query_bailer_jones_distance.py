#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query Bailer-Jones distance estimates.

Query Gaia DR3 Bailer-Jones distance estimates from the VizieR TAP service,
including geometric and photogeometric distances.

Input: vizier_dedup_double.csv (the double-deduplicated cross-match result)
Output: bailer_jones_distance_for_candidates.csv
"""

from __future__ import annotations

import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


LGADNET_ROOT = Path("/path/to/your/lgadnet_data")   # <-- EDIT THIS root
DATA = LGADNET_ROOT

INPUT = DATA / "spatial_filtering_unique" / "vizier_dedup_double.csv"
OUTPUT = DATA / "spatial_filtering_unique" / "bailer_jones_distance_for_candidates.csv"

TAP_URL = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
BATCH_SIZE = 1000
SLEEP_SEC = 1.0


def query_batch(source_ids: list[int]) -> pd.DataFrame:
    """
    Query Bailer-Jones distances for a batch of Source IDs.

    Parameters
    ----------
    source_ids : list[int]
        List of Gaia DR3 Source IDs

    Returns
    -------
    pd.DataFrame
        Query result containing distance estimates and confidence intervals
    """
    source_list = ",".join(str(int(x)) for x in source_ids)

    query = f"""
    SELECT
        Source,
        rgeo AS r_med_geo,
        "b_rgeo" AS r_lo_geo,
        "B_rgeo" AS r_hi_geo,
        rpgeo AS r_med_photogeo,
        "b_rpgeo" AS r_lo_photogeo,
        "B_rpgeo" AS r_hi_photogeo,
        Flag AS bj_flag
    FROM "I/352/gedr3dis"
    WHERE Source IN ({source_list})
    """

    response = requests.post(
        TAP_URL,
        data={
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "csv",
            "QUERY": query,
        },
        timeout=120,
    )
    response.raise_for_status()

    text = response.text.strip()
    if not text or text.startswith("<?xml"):
        return pd.DataFrame()

    return pd.read_csv(StringIO(text))


def main() -> int:
    print("=" * 80)
    print("Query Bailer-Jones distance estimates")
    print("=" * 80)

    # Read the input file
    df = pd.read_csv(INPUT, usecols=["Source"])
    sources = (
        pd.to_numeric(df["Source"], errors="coerce")
        .dropna()
        .astype("int64")
        .drop_duplicates()
        .tolist()
    )

    print(f"\nInput file: {INPUT}")
    print(f"Number of unique Sources: {len(sources):,}")

    # Query in batches
    all_parts = []
    total = len(sources)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\nBatch size: {BATCH_SIZE}")
    print(f"Total batches: {total_batches}")
    print(f"Estimated time: ~{total_batches * SLEEP_SEC / 60:.1f} minutes (excluding network time)")
    print("-" * 80)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = sources[start:end]
        batch_num = start // BATCH_SIZE + 1

        print(f"Batch {batch_num}/{total_batches}: querying {start:,} - {end:,} / {total:,}")

        try:
            part = query_batch(batch)
            if len(part) > 0:
                all_parts.append(part)
            print(f"  Matched: {len(part):,}")
        except Exception as exc:
            print(f"  Error: {exc}")

        time.sleep(SLEEP_SEC)

    # Merge results
    print("\n" + "-" * 80)
    print("Merging results")

    if all_parts:
        out = pd.concat(all_parts, ignore_index=True)
        out = out.drop_duplicates("Source", keep="first")
    else:
        out = pd.DataFrame(
            columns=[
                "Source",
                "r_med_geo",
                "r_lo_geo",
                "r_hi_geo",
                "r_med_photogeo",
                "r_lo_photogeo",
                "r_hi_photogeo",
                "bj_flag",
            ]
        )

    # Save output
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)

    print("=" * 80)
    print("Query complete")
    print("=" * 80)
    print(f"Output file: {OUTPUT}")
    print(f"Matched rows: {len(out):,}")
    print(f"Match rate: {len(out) / len(sources) * 100:.2f}%")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())