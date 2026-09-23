#!/usr/bin/env python3
"""Query Bailer-Jones distance estimates from Gaia DR3 via the VizieR TAP service.

Batch-queries the Gaia DR3 Bailer-Jones geometric / photogeometric distances
(table I/352/gedr3dis) for every Source, and writes them back for the screening
downstream. Batch size and sleep guard the TAP service against rate limits.

Input:  vizier_dedup_double.csv   (double-deduplicated Gaia cross-match, see step 04)
Output: bailer_jones_distance_for_candidates.csv
"""

from __future__ import annotations

import argparse
import time
from io import StringIO
import os
from pathlib import Path

import pandas as pd
import requests

LGADNET_ROOT = Path(os.environ.get("LGADNET_ROOT", "/path/to/lgadnet_data"))
INTER = LGADNET_ROOT / "crossmatch"

DEFAULT_INPUT = INTER / "vizier_dedup_double.csv"
DEFAULT_OUTPUT = INTER / "bailer_jones_distance_for_candidates.csv"

TAP_URL = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
BATCH_SIZE = 1000
SLEEP_SEC = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def query_batch(source_ids: list[int]) -> pd.DataFrame:
    """Query BJ distances for a batch of Source IDs; empty if no result."""
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
        data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query},
        timeout=120,
    )
    response.raise_for_status()

    text = response.text.strip()
    if not text or text.startswith("<?xml"):
        return pd.DataFrame()
    return pd.read_csv(StringIO(text))


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

    df = pd.read_csv(input_path, usecols=["Source"])
    sources = (
        pd.to_numeric(df["Source"], errors="coerce")
        .dropna()
        .astype("int64")
        .drop_duplicates()
        .tolist()
    )
    total = len(sources)
    print(f"unique_sources={total}")

    all_parts = []
    for start in range(0, total, args.batch_size):
        end = min(start + args.batch_size, total)
        batch = sources[start:end]
        try:
            part = query_batch(batch)
            if len(part) > 0:
                all_parts.append(part)
            print(f"batch[{start//args.batch_size+1}]: matched={len(part)}")
        except Exception as exc:  # noqa: BLE001
            print(f"batch[{start//args.batch_size+1}]: error={exc}")
        time.sleep(SLEEP_SEC)

    if all_parts:
        out = pd.concat(all_parts, ignore_index=True)
        out = out.drop_duplicates("Source", keep="first")
    else:
        out = pd.DataFrame(
            columns=[
                "Source", "r_med_geo", "r_lo_geo", "r_hi_geo",
                "r_med_photogeo", "r_lo_photogeo", "r_hi_photogeo", "bj_flag",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"matched={len(out)}/{total} ({len(out) / total * 100:.2f}%)")
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())