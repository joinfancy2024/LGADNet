#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild: EGP distribution comparison (training CEMP vs CMD-retained candidates).

Reads the training CEMP EGP table and the CMD-retained CEMP candidate table
(having an EGP column) and draws the EGP density comparison.
Output goes to a NEW directory; original files untouched.

Input:  --train-csv  training CEMP EGP table
        --final-csv  CMD-retained CEMP candidate table with an EGP column
Output: egp_train_vs_cemp_final_med.{pdf,png} + stats json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

LGADNET_ROOT = Path(os.environ.get("LGADNET_ROOT", "/path/to/lgadnet_data"))
DEFAULT_TRAIN_CSV = LGADNET_ROOT / "training/train_labels_with_egp.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--final-csv", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--bins", type=int, default=45)
    return p.parse_args()


def load_egp(path: Path) -> np.ndarray:
    table = pd.read_csv(path, engine="python")
    if "EGP_status" in table.columns:
        table = table[table["EGP_status"].eq("success")]
    egp = pd.to_numeric(table["EGP"], errors="coerce")
    egp = egp.replace([np.inf, -np.inf], np.nan).dropna()
    return egp.to_numpy(float)


def summarize(v: np.ndarray) -> dict:
    s = pd.Series(v)
    return {"n": int(s.size), "p01": float(s.quantile(0.01)),
            "p50": float(s.quantile(0.50)), "p99": float(s.quantile(0.99))}


def kde_line(v, grid):
    if v.size < 2 or np.allclose(v, v[0]):
        return np.zeros_like(grid)
    return gaussian_kde(v)(grid)


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    train = load_egp(args.train_csv)
    final = load_egp(args.final_csv)
    ts, fs = summarize(train), summarize(final)

    q_low = min(ts["p01"], fs["p01"])
    q_high = max(ts["p99"], fs["p99"])
    span = max(q_high - q_low, 1e-6)
    xlim = (q_low - 0.08 * span, q_high + 0.18 * span)
    grid = np.linspace(xlim[0], xlim[1], 700)

    plt.rcParams.update({"font.family": "serif", "font.size": 20,
                         "axes.linewidth": 1.2,
                         "pdf.fonttype": 42, "ps.fonttype": 42})

    blue, red = "#2454A6", "#B5392D"
    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    ax.hist(train, bins=args.bins, range=xlim, density=True,
            color=blue, alpha=0.08, edgecolor="none")
    ax.hist(final, bins=args.bins, range=xlim, density=True,
            color=red, alpha=0.08, edgecolor="none")
    ax.plot(grid, kde_line(train, grid), color=blue, lw=3.0, label="Training CEMP")
    ax.plot(grid, kde_line(final, grid), color=red, lw=3.0,
            label="CMD-retained CEMP candidates")
    ax.axvline(ts["p50"], color=blue, ls="--", lw=2.0, alpha=0.9)
    ax.axvline(fs["p50"], color=red, ls="--", lw=2.0, alpha=0.9)
    ax.set_xlabel("EGP", fontsize=24)
    ax.set_ylabel("Density", fontsize=24)
    ax.set_xlim(*xlim)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.25, lw=0.8)
    ax.tick_params(axis="both", labelsize=22)
    ax.legend(loc="upper right", fontsize=22, frameon=True, framealpha=0.95)
    fig.tight_layout()

    out_pdf = args.outdir / "egp_train_vs_cemp_final_med.pdf"
    out_png = args.outdir / "egp_train_vs_cemp_final_med.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    json.dump({"train": ts, "final": fs, "xlim": list(xlim)},
              open(args.outdir / "egp_train_vs_cemp_final_med_stats.json", "w"),
              ensure_ascii=False, indent=2)

    print(f"Training CEMP EGP: n={ts['n']:,}, median={ts['p50']:.3f}")
    print(f"CMD-retained CEMP candidates EGP: n={fs['n']:,}, median={fs['p50']:.3f}")
    print(f"PDF: {out_pdf}")
    print(f"PNG: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())