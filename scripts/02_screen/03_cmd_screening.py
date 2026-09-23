#!/usr/bin/env python3
"""Step 3: CMD screening — compute dereddened colours, absolute magnitudes,
apply CMD selection criteria (BP-RP_0 >= 0.4, M_G_BJ_med <= 7.0).

Input:  egp.csv (from step 02, with d_bj_geo_kpc).
Output: cmd_flags.csv (all candidates with CMD flags) + cemp_cmd_summary.txt.
Uses the BJ geometric median distance (M_G_BJ_med).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


COLOR_CUT = 0.4
MG_CUT = 7.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-csv", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    return p.parse_args()


def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def summarize_numeric(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return "NA"
    return (f"n={len(x):,}, median={x.median():.3f}, "
            f"16-84%={x.quantile(0.16):.3f}-{x.quantile(0.84):.3f}")


def count_pct(mask: pd.Series, denom: int) -> str:
    n = int(mask.fillna(False).sum())
    return f"{n:,} ({100.0 * n / denom:.2f}%)" if denom else "0"


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.in_csv, engine="python")

    numeric_cols = ["Gmag", "BPmag", "RPmag", "AG", "E(BP-RP)",
                    "d_bj_geo_kpc",
                    "TEFF", "LOGG", "FE_H", "C_FE"]
    df = to_numeric(df, numeric_cols)

    required = ["Gmag", "BPmag", "RPmag", "d_bj_geo_kpc"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    ebprp_col = "E(BP-RP)" if "E(BP-RP)" in df.columns else None
    ag_col = "AG" if "AG" in df.columns else None

    color_obs = df["BPmag"] - df["RPmag"]
    df["bp_rp_0"] = color_obs - df[ebprp_col] if ebprp_col else color_obs

    d_med_pc = df["d_bj_geo_kpc"] * 1000.0
    d_med_pc = d_med_pc.where(d_med_pc > 0)

    df["M_G_bj_med"] = df["Gmag"] - 5.0 * np.log10(d_med_pc / 10.0)
    if ag_col:
        ag_ok = df[ag_col].notna()
        df.loc[ag_ok, "M_G_bj_med"] -= df.loc[ag_ok, ag_col]

    df["cmd_has_data"] = df["bp_rp_0"].notna() & df["M_G_bj_med"].notna()
    df["cmd_hot_blue"] = df["cmd_has_data"] & (df["bp_rp_0"] < COLOR_CUT)
    df["cmd_faint"] = df["cmd_has_data"] & (df["M_G_bj_med"] > MG_CUT)
    df["cmd_retained"] = df["cmd_has_data"] & (df["bp_rp_0"] >= COLOR_CUT) & (df["M_G_bj_med"] <= MG_CUT)

    n = len(df)
    lines = [
        "=" * 80,
        "Gaia CMD Quality Check",
        "=" * 80,
        f"Input: {n:,}",
        f"Complete CMD data: {count_pct(df['cmd_has_data'], n)}",
        f"Hot/blue (BP-RP_0 < {COLOR_CUT}): {count_pct(df['cmd_hot_blue'], n)}",
        f"Faint (M_G_bj_med > {MG_CUT}): {count_pct(df['cmd_faint'], n)}",
        f"CMD retained (BP-RP_0>={COLOR_CUT} and M_G_bj_med<={MG_CUT}): {count_pct(df['cmd_retained'], n)}",
        "",
        f"BP-RP_0: {summarize_numeric(df['bp_rp_0'])}",
        f"M_G_bj_med: {summarize_numeric(df['M_G_bj_med'])}",
    ]
    if "LOGG" in df.columns:
        lines.append(f"LOGG: {summarize_numeric(df['LOGG'])}")
    if "TEFF" in df.columns:
        lines.append(f"TEFF: {summarize_numeric(df['TEFF'])}")
    report = "\n".join(lines)

    out_csv = args.outdir / "cmd_flags.csv"
    out_report = args.outdir / "cemp_cmd_summary.txt"

    df.to_csv(out_csv, index=False)
    out_report.write_text(report + "\n", encoding="utf-8")

    print(report)
    print(f"CSV: {out_csv}")
    print(f"Report: {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())