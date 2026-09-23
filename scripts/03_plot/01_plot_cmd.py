#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild: Gaia CMD quality check figure for ALL phase-space candidates.

Draws the (BP-RP)_0 vs M_G(med) CMD for all 15,742 phase-space candidates
(not just the CMD-retained ones), coloring by the model-estimated log g and
T_eff, with the selection boundaries overlaid. This shows the screening
itself. Output goes to a NEW directory; original files untouched.

Input:  --in-csv  candidate table with bp_rp_0 / M_G_bj_med / cmd_retained
Output: cemp_all_candidates_med_gaia_cmd.{pdf,png}
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

LGADNET_ROOT = Path(os.environ.get("LGADNET_ROOT", "/path/to/lgadnet_data"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR_CUT = 0.4
MG_CUT = 7.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-csv", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Full phase-space candidate table with CMD flags (15742 rows)
    df = pd.read_csv(args.in_csv, engine="python")

    # Merge the four model-estimated parameters from the upstream table
    if "LOGG" not in df.columns or "TEFF" not in df.columns:
        upstream = pd.read_csv(
            LGADNET_ROOT / "crossmatch/cemp_unique_b_greater_30.csv",
            usecols=["candidate_id", "TEFF", "LOGG", "FE_H", "C_FE"],
            engine="python",
        )
        df["candidate_id"] = pd.to_numeric(df["candidate_id"], errors="coerce").astype("Int64")
        upstream["candidate_id"] = pd.to_numeric(upstream["candidate_id"], errors="coerce").astype("Int64")
        upstream = upstream.drop_duplicates(subset="candidate_id", keep="first")
        df = df.merge(upstream, on="candidate_id", how="left")

    for c in ["bp_rp_0", "M_G_bj_med", "LOGG", "TEFF", "cmd_retained"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    has = df.dropna(subset=["bp_rp_0", "M_G_bj_med"]).copy()
    ret = has[has["cmd_retained"].fillna(False) == 1]

    plt.rcParams.update({
        "font.family": "serif", "font.size": 20,
        "axes.linewidth": 1.2,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)
    panels = [
        (axes[0], "LOGG", r"$\log g$ (dex)", r"Colored by $\log g$", "(a)"),
        (axes[1], "TEFF", r"$T_{\rm eff}$ (K)", r"Colored by $T_{\rm eff}$", "(b)"),
    ]
    for ax, color_col, clab, title, lbl in panels:
        # All phase-space candidates first
        cv = pd.to_numeric(has[color_col], errors="coerce")
        sc = ax.scatter(
            has["bp_rp_0"], has["M_G_bj_med"],
            c=cv, s=8, cmap="coolwarm", alpha=0.9,
            edgecolor="none", rasterized=True,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(clab, fontsize=24)
        cbar.ax.tick_params(labelsize=22)
        ax.axvline(COLOR_CUT, color="darkred", lw=1.5, ls="--")
        ax.axhline(MG_CUT, color="darkblue", lw=1.5, ls="--")
        ax.set_xlabel(r"$(BP-RP)_0$", fontsize=24)
        ax.set_title(title, fontsize=24)
        ax.tick_params(axis="both", labelsize=22)
        ax.text(0.03, 0.97, lbl, transform=ax.transAxes, fontsize=24,
                fontweight="bold", va="top")
    axes[0].set_ylabel(r"$M_G$ (mag)", fontsize=24)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(alpha=0.2, lw=0.5)
    fig.tight_layout()

    out_pdf = args.outdir / "cemp_all_candidates_med_gaia_cmd.pdf"
    out_png = args.outdir / "cemp_all_candidates_med_gaia_cmd.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"All phase-space candidates (rows): {len(df):,}")
    print(f"  With complete CMD data (plotted): {len(has):,}")
    print(f"  CMD-retained: {len(ret):,}")
    print(f"  Removed by CMD: {len(has) - len(ret):,}")
    print(f"PDF: {out_pdf}")
    print(f"PNG: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())