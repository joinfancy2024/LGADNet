#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaia CMD quality check for BJ robust CEMP candidates.

Input:
    cemp_bj_robust_candidates_6928.csv

Outputs:
    cemp_bj_robust_cmd_flags.csv
    cemp_bj_robust_gaia_cmd.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LGADNET_ROOT = Path("/path/to/your/lgadnet_data")   # <-- EDIT THIS root
DATA = LGADNET_ROOT

INPUT = DATA / "spatial_filtering_unique" / "cemp_bj_robust_candidates_6928.csv"

OUTPUT_CSV = DATA / "spatial_filtering_unique" / "cemp_bj_robust_cmd_flags.csv"
OUTPUT_FIG = DATA / "spatial_filtering_unique" / "cemp_bj_robust_gaia_cmd.pdf"

COLOR_CUT = 0.4
MG_CUT = 7.0


def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def summarize_numeric(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return "NA"
    return (
        f"n={len(x):,}, "
        f"median={x.median():.3f}, "
        f"16-84%={x.quantile(0.16):.3f}-{x.quantile(0.84):.3f}, "
        f"5-95%={x.quantile(0.05):.3f}-{x.quantile(0.95):.3f}"
    )


def count_pct(mask: pd.Series, denom: int) -> str:
    n = int(mask.fillna(False).sum())
    pct = 100.0 * n / denom if denom else 0.0
    return f"{n:,} ({pct:.2f}%)"



def main() -> int:
    df = pd.read_csv(INPUT, engine="python")

    if "Source" in df.columns:
        df["Source"] = df["Source"].astype(str).str.strip()

    numeric_cols = [
        "Gmag", "BPmag", "RPmag", "AG", "E(BP-RP)",
        "d_bj_geo_kpc", "d_bj_lo_kpc",
        "TEFF", "LOGG", "FE_H", "C_FE",
    ]
    df = to_numeric(df, numeric_cols)

    required = ["Gmag", "BPmag", "RPmag", "d_bj_geo_kpc", "d_bj_lo_kpc"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    ebprp_col = "E(BP-RP)" if "E(BP-RP)" in df.columns else None
    ag_col = "AG" if "AG" in df.columns else None

    color_obs = df["BPmag"] - df["RPmag"]
    if ebprp_col:
        df["bp_rp_0"] = color_obs - df[ebprp_col]
    else:
        df["bp_rp_0"] = color_obs

    d_med_pc = df["d_bj_geo_kpc"] * 1000.0
    d_lo_pc = df["d_bj_lo_kpc"] * 1000.0
    d_med_pc = d_med_pc.where(d_med_pc > 0)
    d_lo_pc = d_lo_pc.where(d_lo_pc > 0)

    df["M_G_bj_med"] = df["Gmag"] - 5.0 * np.log10(d_med_pc / 10.0)
    df["M_G_bj_lo"] = df["Gmag"] - 5.0 * np.log10(d_lo_pc / 10.0)

    if ag_col:
        df["M_G_bj_med"] = df["M_G_bj_med"] - df[ag_col]
        df["M_G_bj_lo"] = df["M_G_bj_lo"] - df[ag_col]

    df["cmd_has_data"] = (
        df["bp_rp_0"].notna()
        & df["M_G_bj_med"].notna()
        & df["M_G_bj_lo"].notna()
    )
    df["cmd_hot_blue"] = df["cmd_has_data"] & (df["bp_rp_0"] < COLOR_CUT)
    df["cmd_faint"] = df["cmd_has_data"] & (df["M_G_bj_lo"] > MG_CUT)
    df["cmd_retained"] = (
        df["cmd_has_data"]
        & (df["bp_rp_0"] >= COLOR_CUT)
        & (df["M_G_bj_lo"] <= MG_CUT)
    )

    logg_col = "LOGG" if "LOGG" in df.columns else None
    teff_col = "TEFF" if "TEFF" in df.columns else None

    n = len(df)
    lines = []
    add = lines.append

    add("=" * 80)
    add("Gaia CMD Quality Check for BJ Robust CEMP Candidates")
    add("=" * 80)
    add(f"Input rows: {n:,}")
    add(f"Complete CMD data: {count_pct(df['cmd_has_data'], n)}")
    add(f"Missing CMD data: {count_pct(~df['cmd_has_data'], n)}")
    add("")
    add(f"CMD hot/blue region (BP-RP_0 < {COLOR_CUT}): {count_pct(df['cmd_hot_blue'], n)}")
    add(f"CMD faint region (M_G_bj_lo > {MG_CUT}): {count_pct(df['cmd_faint'], n)}")
    add(
        f"CMD retained region (BP-RP_0 >= {COLOR_CUT} and M_G_bj_lo <= {MG_CUT}): "
        f"{count_pct(df['cmd_retained'], n)}"
    )
    add("")
    add(f"BP-RP_0: {summarize_numeric(df['bp_rp_0'])}")
    add(f"M_G_bj_med: {summarize_numeric(df['M_G_bj_med'])}")
    add(f"M_G_bj_lo: {summarize_numeric(df['M_G_bj_lo'])}")
    if logg_col:
        add(f"{logg_col}: {summarize_numeric(df[logg_col])}")
    if teff_col:
        add(f"{teff_col}: {summarize_numeric(df[teff_col])}")

    report = "\n".join(lines)

    # Set global plot style (consistent with plot_cemp_parameter_density.py)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    plot_df = df[df["cmd_has_data"]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)

    panels = [
        (axes[0], logg_col, r"$\log g$ (dex)", r"Colored by $\log g$", "(a)"),
        (axes[1], teff_col, r"$T_{\rm eff}$ (K)", r"Colored by $T_{\rm eff}$", "(b)"),
    ]

    for ax, color_col, cbar_label, title, label in panels:
        if color_col and color_col in plot_df.columns:
            color_values = pd.to_numeric(plot_df[color_col], errors="coerce")
            sc = ax.scatter(
                plot_df["bp_rp_0"],
                plot_df["M_G_bj_lo"],
                c=color_values,
                s=8,
                cmap="coolwarm",
                alpha=0.9,
                edgecolor="none",
                rasterized=True,
            )
            cbar = fig.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label(cbar_label, fontsize=12)
            cbar.ax.tick_params(labelsize=11)
        else:
            ax.scatter(
                plot_df["bp_rp_0"],
                plot_df["M_G_bj_lo"],
                s=8,
                c="gray",
                alpha=0.9,
                edgecolor="none",
                rasterized=True,
            )

        ax.axvline(COLOR_CUT, color="darkred", lw=1.5, ls="--")
        ax.axhline(MG_CUT, color="darkblue", lw=1.5, ls="--")
        ax.set_xlabel(r"$(BP-RP)_0$", fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis="both", labelsize=11)

        # Add panel label
        ax.text(
            0.03,
            0.97,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            verticalalignment="top",
        )

    axes[0].set_ylabel(r"$M_G$ (BJ lower distance)", fontsize=12)
    axes[0].invert_yaxis()

    for ax in axes:
        ax.grid(alpha=0.2, lw=0.5)

    fig.tight_layout()

    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(report)
    print("=" * 80)
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Output figure: {OUTPUT_FIG}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
