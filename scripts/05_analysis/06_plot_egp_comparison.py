#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild the EGP distribution plot for training and validated CEMP samples."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


LGADNET_ROOT = Path("/path/to/your/lgadnet_data")   # <-- EDIT THIS root
DEFAULT_TRAIN_CSV = (
    LGADNET_ROOT
    / "lgadnet_dataset/train_labels_with_egp.csv"
)
DEFAULT_FINAL_CSV = LGADNET_ROOT / "final_validated" / "cemp_final_cmdretained.csv"
DEFAULT_OUTPUT_DIR = LGADNET_ROOT / "spatial_filtering_unique"
DEFAULT_PREFIX = "egp_train_vs_cemp_bj_cmdretained_final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--final-csv", type=Path, default=DEFAULT_FINAL_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--bins", type=int, default=45)
    return parser.parse_args()


def load_egp(path: Path) -> np.ndarray:
    table = pd.read_csv(path, engine="python")
    if "EGP_status" in table.columns:
        table = table[table["EGP_status"].eq("success")].copy()
    egp = pd.to_numeric(table["EGP"], errors="coerce")
    egp = egp.replace([np.inf, -np.inf], np.nan).dropna()
    return egp.to_numpy(dtype=float)


def summarize(values: np.ndarray) -> dict[str, float | int]:
    series = pd.Series(values)
    return {
        "n": int(series.size),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)),
        "min": float(series.min()),
        "max": float(series.max()),
        "p01": float(series.quantile(0.01)),
        "p05": float(series.quantile(0.05)),
        "p50": float(series.quantile(0.50)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
    }


def get_plot_xlim(train_stats: dict[str, float | int], final_stats: dict[str, float | int]) -> tuple[float, float]:
    q_low = min(float(train_stats["p01"]), float(final_stats["p01"]))
    q_high = max(float(train_stats["p99"]), float(final_stats["p99"]))
    span = max(q_high - q_low, 1e-6)
    return q_low - 0.08 * span, q_high + 0.18 * span


def kde_line(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    if values.size < 2 or np.allclose(values, values[0]):
        return np.zeros_like(x_grid)
    return gaussian_kde(values)(x_grid)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_distribution(
    train_values: np.ndarray,
    final_values: np.ndarray,
    train_stats: dict[str, float | int],
    final_stats: dict[str, float | int],
    output_png: Path,
    output_pdf: Path,
    bins: int,
) -> None:
    setup_style()
    blue = "#2454A6"
    red = "#B5392D"
    xlim = get_plot_xlim(train_stats, final_stats)
    x_grid = np.linspace(xlim[0], xlim[1], 700)

    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    ax.hist(
        train_values,
        bins=bins,
        range=xlim,
        density=True,
        color=blue,
        alpha=0.08,
        edgecolor="none",
    )
    ax.hist(
        final_values,
        bins=bins,
        range=xlim,
        density=True,
        color=red,
        alpha=0.08,
        edgecolor="none",
    )

    ax.plot(
        x_grid,
        kde_line(train_values, x_grid),
        color=blue,
        linewidth=3.0,
        label="Train",
    )
    ax.plot(
        x_grid,
        kde_line(final_values, x_grid),
        color=red,
        linewidth=3.0,
        label="Final candidates",
    )

    ax.axvline(float(train_stats["p50"]), color=blue, linestyle="--", linewidth=2.0, alpha=0.9, label="Median line")
    ax.axvline(float(final_stats["p50"]), color=red, linestyle="--", linewidth=2.0, alpha=0.9)

    ax.set_xlabel("EGP", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.set_xlim(*xlim)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(loc="upper right", fontsize=14, frameon=True, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_values = load_egp(args.train_csv)
    final_values = load_egp(args.final_csv)
    train_stats = summarize(train_values)
    final_stats = summarize(final_values)

    output_png = args.output_dir / f"{args.prefix}.png"
    output_pdf = args.output_dir / f"{args.prefix}.pdf"
    output_stats = args.output_dir / f"{args.prefix}_stats.json"

    plot_distribution(
        train_values=train_values,
        final_values=final_values,
        train_stats=train_stats,
        final_stats=final_stats,
        output_png=output_png,
        output_pdf=output_pdf,
        bins=args.bins,
    )

    payload = {
        "created_at": datetime.now().isoformat(),
        "train_csv": str(args.train_csv),
        "final_csv": str(args.final_csv),
        "output_png": str(output_png),
        "output_pdf": str(output_pdf),
        "train": train_stats,
        "final_validated_cemp": final_stats,
        "plot_xlim": list(get_plot_xlim(train_stats, final_stats)),
        "bins": args.bins,
    }
    output_stats.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"saved: {output_png}")
    print(f"saved: {output_pdf}")
    print(f"saved: {output_stats}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
