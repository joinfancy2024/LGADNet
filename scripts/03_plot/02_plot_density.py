#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild: parameter-space density figure for the final med sample.

Reads the final med CMD-retained catalog (already carrying the four
model-estimated parameters) and draws the [Fe/H]-[C/Fe] abundance plane
and the T_eff-log(L/L_sun) evolutionary-position panel.
Output goes to a NEW directory; original files untouched.

Input:  --in-csv  final catalog with TEFF / LOGG / FE_H / C_FE
Output: cemp_final_med_parameter_density.{pdf,png} + stats json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

DEFAULT_MASS_SUN = 0.8  # representative old metal-poor star mass (M_sun)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-csv", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    return p.parse_args()


def kde_ordered(x, y):
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = np.argsort(z)
    return x[idx], y[idx], z[idx]


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.in_csv, engine="python")
    for c in ["TEFF", "LOGG", "FE_H", "C_FE"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    work = df.dropna(subset=["TEFF", "LOGG", "FE_H", "C_FE"]).copy()
    teff = work["TEFF"].to_numpy(float)
    logg = work["LOGG"].to_numpy(float)
    feh = work["FE_H"].to_numpy(float)
    cfe = work["C_FE"].to_numpy(float)

    # log(L/L_sun) from mass, logg and Teff
    logll = (
        np.log10(DEFAULT_MASS_SUN) - (logg - 4.44) + 4.0 * np.log10(teff / 5780.0)
    )

    plt.rcParams.update({
        "font.family": "serif", "font.size": 20,
        "axes.linewidth": 1.2,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    x1, y1, z1 = kde_ordered(feh, cfe)
    sc1 = ax1.scatter(x1, y1, c=z1, s=8, cmap="coolwarm", alpha=0.9,
                      edgecolor="none", rasterized=True)
    ax1.axhline(0.7, color="darkred", ls="--", lw=1.5, label="CEMP threshold")
    ax1.axvline(-1.0, color="darkred", ls="--", lw=1.5)
    ax1.axvline(-3.0, color="#b33b3b", ls=":", lw=1.1)
    ax1.set_xlabel("[Fe/H] (dex)", fontsize=24)
    ax1.set_ylabel("[C/Fe] (dex)", fontsize=24)
    ax1.legend(loc="lower right", fontsize=18, framealpha=0.9)
    ax1.set_xlim(-4.0, -1.0)
    ax1.set_ylim(-0.5, 3.5)
    ax1.tick_params(axis="both", labelsize=22)
    ax1.text(0.97, 0.97, f"N = {len(x1):,}", transform=ax1.transAxes, fontsize=22,
             horizontalalignment="right", verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax1.text(0.03, 0.88, "(a)", transform=ax1.transAxes, fontsize=24, fontweight="bold", va="top")

    x2, y2, z2 = kde_ordered(teff, logll)
    sc2 = ax2.scatter(x2, y2, c=z2, s=8, cmap="coolwarm", alpha=0.9,
                      edgecolor="none", rasterized=True)
    coeffs = np.polyfit(teff, logll, deg=2)
    poly = np.poly1d(coeffs)
    tf = np.linspace(teff.min(), teff.max(), 50)
    ax2.plot(tf, poly(tf), color="darkblue", ls="--", lw=2.5, alpha=0.8,
             label="Evolutionary trend")
    ax2.set_xlabel(r"$T_{\rm eff}$ (K)", fontsize=24)
    ax2.set_ylabel(r"$\log(L/L_\odot)$", fontsize=24)
    ax2.legend(loc="lower right", fontsize=18, framealpha=0.9)
    ax2.set_xlim(7000, 3500)
    ax2.set_ylim(-2, 4)
    ax2.tick_params(axis="both", labelsize=22)
    ax2.text(0.97, 0.97, f"N = {len(x2):,}", transform=ax2.transAxes, fontsize=22,
             horizontalalignment="right", verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax2.text(0.03, 0.88, "(b)", transform=ax2.transAxes, fontsize=24, fontweight="bold", va="top")

    cbar = fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Density", fontsize=24)
    cbar.ax.tick_params(labelsize=22)

    fig.tight_layout(pad=1.5, w_pad=0.8)

    out_pdf = args.outdir / "cemp_final_med_parameter_density.pdf"
    out_png = args.outdir / "cemp_final_med_parameter_density.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    stats = {
        "n_input": int(len(df)),
        "n_valid_parameter_rows": int(len(work)),
        "Teff_median": float(work["TEFF"].median()),
        "Logg_median": float(work["LOGG"].median()),
        "FeH_median": float(work["FE_H"].median()),
        "CFe_median": float(work["C_FE"].median()),
        "logLLodot_median": float(np.median(logll)),
        "n_feh_lt_minus3": int((feh < -3.0).sum()),
        "n_cfe_gt_2": int((cfe > 2.0).sum()),
        "n_feh_lt_minus25": int((feh < -2.5).sum()),
        "n_cfe_gt_1": int((cfe > 1.0).sum()),
    }
    json.dump(stats, open(args.outdir / "cemp_final_med_parameter_density_stats.json", "w"),
              ensure_ascii=False, indent=2)

    print("=" * 70)
    print("Final med parameter-space plot")
    print("=" * 70)
    print(f"Input rows: {len(df):,}")
    print(f"Valid parameter rows: {len(work):,}")
    print(f"[Fe/H] median: {stats['FeH_median']:.3f}")
    print(f"[C/Fe] median: {stats['CFe_median']:.3f}")
    print(f"[Fe/H] < -3.0: {stats['n_feh_lt_minus3']:,}")
    print(f"[C/Fe] > 2.0: {stats['n_cfe_gt_2']:,}")
    print(f"PDF: {out_pdf}")
    print(f"PNG: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())