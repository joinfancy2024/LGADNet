#!/usr/bin/env python3
"""Step 4: Build and export the final CEMP candidate sample for the paper.

Reads the CMD-flagged CSV from step 03 (cmd_flags.csv), merges the four
model-estimated stellar parameters (TEFF/LOGG/FE_H/C_FE) from the upstream
candidate table, keeps only cmd_retained==True rows, and writes the final
catalog used for the paper plots plus a screening summary.

Input:  cmd_flags.csv + upstream cemp_unique_b_greater_30.csv (four params)
Output: final.csv (CMD-retained sample) + cemp_final_summary.txt.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Upstream table handling the four model-estimated parameter columns
# (TEFF/LOGG/FE_H/C_FE), keyed by candidate_id.
LGADNET_ROOT = Path(os.environ.get("LGADNET_ROOT", "/path/to/lgadnet_data"))
DEFAULT_UPSTREAM = LGADNET_ROOT / "crossmatch/cemp_unique_b_greater_30.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-csv", type=Path, required=True,
                   help="CMD-flagged CSV (output of 04_cmd_screening)")
    p.add_argument("--upstream-csv", type=Path, default=DEFAULT_UPSTREAM)
    p.add_argument("--outdir", type=Path, required=True)
    return p.parse_args()


def bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].fillna(False).astype(bool)


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.in_csv)
    up = pd.read_csv(args.upstream_csv,
                     usecols=["candidate_id", "TEFF", "LOGG", "FE_H", "C_FE"])

    df["candidate_id"] = pd.to_numeric(df["candidate_id"], errors="coerce").astype("Int64")
    up["candidate_id"] = pd.to_numeric(up["candidate_id"], errors="coerce").astype("Int64")
    up = up.drop_duplicates(subset="candidate_id", keep="first")

    merged = df.merge(up, on="candidate_id", how="left")

    # Screening summary (counts only; selection was decided in step 04)
    cmd_ret = bool_series(merged, "cmd_retained")
    cmd_hot = bool_series(merged, "cmd_hot_blue")
    cmd_has = bool_series(merged, "cmd_has_data")
    lines = [
        "=" * 80,
        "Final CEMP Candidate Sample — Screening Summary",
        "=" * 80,
        f"Input rows:                {len(merged):,}",
        f"  Complete CMD data:       {cmd_has.sum():,}",
        f"  CMD retained:            {cmd_ret.sum():,}",
        f"  CMD hot/blue:            {cmd_hot.sum():,}",
        f"  CMD missing data:        {(~cmd_has).sum():,}",
    ]
    report = "\n".join(lines)

    # Final sample = CMD retained only
    final = merged[cmd_ret].copy()

    # Columns the downstream plot scripts actually consume.
    keep = [
        "candidate_id", "RAdeg", "DEdeg",
        "Gmag", "BPmag", "RPmag",
        "TEFF", "LOGG", "FE_H", "C_FE",
        "bp_rp_0", "M_G_bj_med",
        "EGP", "EGP_status",
    ]
    present = [c for c in keep if c in final.columns]
    out = final[present].copy()

    out_csv = args.outdir / "final.csv"
    out_report = args.outdir / "cemp_final_summary.txt"
    out.to_csv(out_csv, index=False)
    out_report.write_text(report + "\n", encoding="utf-8")

    print(report)
    print(f"Final sample: {len(out):,} rows")
    print(f"Catalog:   {out_csv}")
    print(f"Report:    {out_report}")
    print(f"Columns:   {present}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())