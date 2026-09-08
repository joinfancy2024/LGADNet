#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot parameter-space distribution for the final CMD-retained CEMP candidates.

Input:
  cemp_bj_robust_cmdretained_final_catalog.csv

Outputs:
  cemp_parameter_density_bj_cmdretained_final.pdf
  cemp_parameter_density_bj_cmdretained_final.png
  cemp_parameter_density_bj_cmdretained_final_stats.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


INPUT_CSV = Path(
    "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/"
    "spatial_filtering_unique/cemp_bj_robust_cmdretained_final_catalog.csv"
)
OUTPUT_DIR = Path(
    "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/"
    "spatial_filtering_unique"
)

OUT_PDF = OUTPUT_DIR / "cemp_parameter_density_bj_cmdretained_final.pdf"
OUT_PNG = OUTPUT_DIR / "cemp_parameter_density_bj_cmdretained_final.png"
OUT_JSON = OUTPUT_DIR / "cemp_parameter_density_bj_cmdretained_final_stats.json"

COLS = {
    "teff": "Teff_LGADNet",
    "logg": "Logg_LGADNet",
    "feh": "FeH_LGADNet",
    "cfe": "CFe_LGADNet",
    "z": "Abs_Z_BJ_lo_kpc",
    "vtan": "Vtan_BJ_lo_kms",
}


def as_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def summarize(values: pd.Series) -> dict:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if len(x) == 0:
        return {"n": 0}
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)),
        "min": float(x.min()),
        "p05": float(x.quantile(0.05)),
        "p16": float(x.quantile(0.16)),
        "median": float(x.median()),
        "p84": float(x.quantile(0.84)),
        "p95": float(x.quantile(0.95)),
        "max": float(x.max()),
    }


def kde_ordered_points(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    z = kde(xy)
    idx = z.argsort()
    return x[idx], y[idx], z[idx]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV, engine="python")
    required = list(COLS.values())
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = as_numeric(df, required)
    work = df.dropna(subset=[COLS["teff"], COLS["logg"], COLS["feh"], COLS["cfe"]]).copy()
    if work.empty:
        raise ValueError("No valid rows for plotting after dropping NaNs in Teff/Logg/FeH/CFe")

    teff = work[COLS["teff"]].to_numpy(dtype=float)
    logg = work[COLS["logg"]].to_numpy(dtype=float)
    feh = work[COLS["feh"]].to_numpy(dtype=float)
    cfe = work[COLS["cfe"]].to_numpy(dtype=float)

    logll = np.log10(0.8) - (logg - 4.44) + 4.0 * np.log10(teff / 5780.0)

    stats = {
        "input_csv": str(INPUT_CSV),
        "n_input": int(len(df)),
        "n_valid_parameter_rows": int(len(work)),
        "columns": COLS,
        "Teff_LGADNet": summarize(work[COLS["teff"]]),
        "Logg_LGADNet": summarize(work[COLS["logg"]]),
        "FeH_LGADNet": summarize(work[COLS["feh"]]),
        "CFe_LGADNet": summarize(work[COLS["cfe"]]),
        "Abs_Z_BJ_lo_kpc": summarize(df[COLS["z"]]),
        "Vtan_BJ_lo_kms": summarize(df[COLS["vtan"]]),
        "logLLodot": summarize(pd.Series(logll)),
        "n_feh_lt_minus3": int((feh < -3.0).sum()),
        "n_cfe_gt_2": int((cfe > 2.0).sum()),
        "n_feh_lt_minus25": int((feh < -2.5).sum()),
        "n_cfe_gt_1": int((cfe > 1.0).sum()),
    }

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x1, y1, z1 = kde_ordered_points(feh, cfe)
    sc1 = ax1.scatter(
        x1,
        y1,
        c=z1,
        s=8,
        cmap="coolwarm",
        alpha=0.9,
        edgecolor="none",
        rasterized=True,
    )
    ax1.axhline(0.7, color="darkred", linestyle="--", linewidth=1.5, label="CEMP threshold")
    ax1.axvline(-1.0, color="darkred", linestyle="--", linewidth=1.5)
    ax1.axvline(-3.0, color="#b33b3b", linestyle=":", linewidth=1.1)
    ax1.set_xlabel("[Fe/H] (dex)", fontsize=12)
    ax1.set_ylabel("[C/Fe] (dex)", fontsize=12)
    ax1.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax1.grid(False)
    ax1.set_xlim(-4.0, -1.0)
    ax1.set_ylim(-0.5, 3.5)
    ax1.tick_params(axis="both", labelsize=11)
    ax1.text(
        0.03,
        0.97,
        f"N = {len(x1):,}",
        transform=ax1.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax1.text(
        0.03,
        0.88,
        "(a)",
        transform=ax1.transAxes,
        fontsize=12,
        fontweight="bold",
        verticalalignment="top",
    )

    x2, y2, z2 = kde_ordered_points(teff, logll)
    sc2 = ax2.scatter(
        x2,
        y2,
        c=z2,
        s=8,
        cmap="coolwarm",
        alpha=0.9,
        edgecolor="none",
        rasterized=True,
    )
    coefficients = np.polyfit(teff, logll, deg=2)
    poly_fit = np.poly1d(coefficients)
    teff_fit = np.linspace(teff.min(), teff.max(), 50)
    logl_fit = poly_fit(teff_fit)
    ax2.plot(
        teff_fit,
        logl_fit,
        color="darkblue",
        linestyle="--",
        linewidth=2.5,
        alpha=0.8,
        label="Evolutionary trend",
    )
    ax2.set_xlabel("T_eff (K)", fontsize=12)
    ax2.set_ylabel(r"log(L/L$_\odot$)", fontsize=12)
    ax2.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax2.grid(False)
    ax2.set_xlim(7000, 3500)
    ax2.set_ylim(-2, 4)
    ax2.tick_params(axis="both", labelsize=11)
    ax2.text(
        0.03,
        0.97,
        f"N = {len(x2):,}",
        transform=ax2.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax2.text(
        0.03,
        0.88,
        "(b)",
        transform=ax2.transAxes,
        fontsize=12,
        fontweight="bold",
        verticalalignment="top",
    )

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sc2, cax=cbar_ax)
    cbar.set_label("Density", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight", format="pdf")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Final CEMP parameter-space plot")
    print("=" * 80)
    print(f"Input rows: {len(df):,}")
    print(f"Valid parameter rows: {len(work):,}")
    print(f"[Fe/H] median: {stats['FeH_LGADNet']['median']:.3f}")
    print(f"[C/Fe] median: {stats['CFe_LGADNet']['median']:.3f}")
    print(f"[Fe/H] < -3.0: {stats['n_feh_lt_minus3']:,}")
    print(f"[C/Fe] > 2.0: {stats['n_cfe_gt_2']:,}")
    print(f"PDF: {OUT_PDF}")
    print(f"PNG: {OUT_PNG}")
    print(f"Stats: {OUT_JSON}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
