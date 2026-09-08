#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot CH G-band absorption depth for low/high C_FE groups across atmospheric windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from numpy.polynomial import Polynomial
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


ROOT_DIR = Path("/home/DM13/workspace/sky")
DEFAULT_INPUT_CSV = (
    ROOT_DIR
    / "data/new_dataset3/lgadnet/regression_snr5_selected_v6_cemp3020_20260611/selected_all_with_egp.csv"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data/new_dataset3/lgadnet/final_validated"
DEFAULT_PREFIX = "ch_gband_cfe_response"

WAVELENGTH_SCOPE = (3900, 8800)
TARGET_WAVELENGTHS = np.arange(WAVELENGTH_SCOPE[0], WAVELENGTH_SCOPE[1] + 1, 1, dtype=float)
PLOT_RANGE = (4140, 4560)
EGP_NUMERATOR_RANGE = (4200, 4400)
EGP_DENOMINATOR_RANGE = (4425, 4520)

DEFAULT_WINDOWS = [
    {
        "label": "Cooler / lower gravity",
        "center": {"TEFF": 5000.0, "LOGG": 3.0, "FE_H": -2.0},
        "half_width": {"TEFF": 450.0, "LOGG": 0.7, "FE_H": 0.45},
    },
    {
        "label": "Warm turnoff",
        "center": {"TEFF": 5900.0, "LOGG": 3.4, "FE_H": -2.1},
        "half_width": {"TEFF": 400.0, "LOGG": 0.6, "FE_H": 0.40},
    },
    {
        "label": "Hotter / higher gravity",
        "center": {"TEFF": 6500.0, "LOGG": 3.8, "FE_H": -2.2},
        "half_width": {"TEFF": 450.0, "LOGG": 0.6, "FE_H": 0.45},
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", type=str, default=DEFAULT_PREFIX)
    parser.add_argument(
        "--max-per-group",
        type=int,
        default=0,
        help="Maximum spectra per low/high group. Use 0 to use all available spectra.",
    )
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--with-title", action="store_true", help="Add panel titles and a figure title.")
    return parser.parse_args()


def load_input(path: Path) -> pd.DataFrame:
    columns = ["TEFF", "LOGG", "FE_H", "C_FE", "EGP", "fits_source_path", "z", "snrg"]
    df = pd.read_csv(path, usecols=columns)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=columns)
    for column in ["TEFF", "LOGG", "FE_H", "C_FE", "EGP", "z", "snrg"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["TEFF", "LOGG", "FE_H", "C_FE", "EGP", "z"]).copy()


def preprocess_spectrum(path: Path, z: float) -> np.ndarray | None:
    try:
        with fits.open(path, memmap=False) as hdul:
            data = hdul[1].data
            wavelength = np.asarray(data["WAVELENGTH"], dtype=float)
            flux = np.asarray(data["FLUX"], dtype=float)
    except Exception:
        return None

    finite_mask = np.isfinite(wavelength) & np.isfinite(flux)
    if finite_mask.sum() < 2 or not np.isfinite(z) or z <= -1:
        return None

    wavelength = wavelength[finite_mask] / (1.0 + z)
    flux = flux[finite_mask]
    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    flux = flux[order]
    wavelength, unique_index = np.unique(wavelength, return_index=True)
    flux = flux[unique_index]

    if wavelength.size < 4:
        return None
    if wavelength.min() > WAVELENGTH_SCOPE[0] or wavelength.max() < WAVELENGTH_SCOPE[1]:
        return None

    scope_mask = (wavelength >= WAVELENGTH_SCOPE[0]) & (wavelength <= WAVELENGTH_SCOPE[1])
    wavelength = wavelength[scope_mask]
    flux = flux[scope_mask]
    if wavelength.size < 4:
        return None

    try:
        interpolator = interp1d(
            wavelength,
            flux,
            kind="cubic" if wavelength.size >= 4 else "linear",
            bounds_error=False,
            fill_value=(flux[0], flux[-1]),
        )
        flux_interpolated = interpolator(TARGET_WAVELENGTHS)
        flux_denoised = savgol_filter(flux_interpolated, window_length=15, polyorder=3, mode="nearest")
        continuum = Polynomial.fit(TARGET_WAVELENGTHS, flux_denoised, 5)(TARGET_WAVELENGTHS)
    except Exception:
        return None

    if not np.all(np.isfinite(continuum)) or np.any(np.abs(continuum) < 1e-12):
        return None

    normalized_flux = flux_interpolated / continuum
    if not np.all(np.isfinite(normalized_flux)):
        return None

    mean_flux = float(np.mean(normalized_flux))
    std_flux = float(np.std(normalized_flux))
    if not np.isfinite(mean_flux) or not np.isfinite(std_flux) or std_flux <= 0:
        return None

    normalized_flux[normalized_flux > mean_flux + 3.0 * std_flux] = mean_flux
    normalized_flux[normalized_flux < mean_flux - 3.0 * std_flux] = mean_flux
    return normalized_flux


def select_window(df: pd.DataFrame, window: dict) -> pd.DataFrame:
    center = window["center"]
    half_width = window["half_width"]
    return df[
        df["TEFF"].between(center["TEFF"] - half_width["TEFF"], center["TEFF"] + half_width["TEFF"])
        & df["LOGG"].between(center["LOGG"] - half_width["LOGG"], center["LOGG"] + half_width["LOGG"])
        & df["FE_H"].between(center["FE_H"] - half_width["FE_H"], center["FE_H"] + half_width["FE_H"])
    ].copy()


def maybe_limit_group(group: pd.DataFrame, max_per_group: int, seed: int) -> pd.DataFrame:
    if max_per_group <= 0 or len(group) <= max_per_group:
        return group
    return group.sample(n=max_per_group, random_state=seed)


def stack_group(group: pd.DataFrame, plot_mask: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    fluxes: list[np.ndarray] = []
    used_rows: list[pd.Series] = []
    for _, row in group.iterrows():
        normalized = preprocess_spectrum(Path(row["fits_source_path"]), float(row["z"]))
        if normalized is None:
            continue
        fluxes.append(normalized[plot_mask])
        used_rows.append(row)

    if not fluxes:
        raise RuntimeError("No usable spectra found for a C_FE group.")

    stacked = np.vstack(fluxes)
    return np.nanmedian(stacked, axis=0), pd.DataFrame(used_rows)


def summarize_group(group: pd.DataFrame) -> dict:
    return {
        "used_n": int(len(group)),
        "C_FE_median": float(group["C_FE"].median()),
        "C_FE_range": [float(group["C_FE"].min()), float(group["C_FE"].max())],
        "EGP_median": float(group["EGP"].median()),
        "TEFF_median": float(group["TEFF"].median()),
        "LOGG_median": float(group["LOGG"].median()),
        "FE_H_median": float(group["FE_H"].median()),
        "snrg_median": float(group["snrg"].median()),
    }


def build_profiles(df: pd.DataFrame, max_per_group: int, seed: int) -> tuple[list[dict], list[dict]]:
    plot_mask = (TARGET_WAVELENGTHS >= PLOT_RANGE[0]) & (TARGET_WAVELENGTHS <= PLOT_RANGE[1])
    panels: list[dict] = []
    profiles: list[dict] = []

    for window in DEFAULT_WINDOWS:
        subset = select_window(df, window)
        q20, q80 = subset["C_FE"].quantile([0.2, 0.8])
        low = maybe_limit_group(subset[subset["C_FE"] <= q20].copy(), max_per_group, seed)
        high = maybe_limit_group(subset[subset["C_FE"] >= q80].copy(), max_per_group, seed)

        low_profile, low_used = stack_group(low, plot_mask)
        high_profile, high_used = stack_group(high, plot_mask)

        center = window["center"]
        half_width = window["half_width"]
        panels.append(
            {
                "label": window["label"],
                "parameter_window": {
                    "TEFF": [center["TEFF"] - half_width["TEFF"], center["TEFF"] + half_width["TEFF"]],
                    "LOGG": [center["LOGG"] - half_width["LOGG"], center["LOGG"] + half_width["LOGG"]],
                    "FE_H": [center["FE_H"] - half_width["FE_H"], center["FE_H"] + half_width["FE_H"]],
                },
                "window_n": int(len(subset)),
                "cfe_quantiles": {"q20": float(q20), "q80": float(q80)},
                "groups": {
                    "Low C_FE": summarize_group(low_used),
                    "High C_FE": summarize_group(high_used),
                },
            }
        )
        profiles.append({"Low C_FE": low_profile, "High C_FE": high_profile})

    return panels, profiles


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_profiles(panels: list[dict], profiles: list[dict], output_png: Path, output_pdf: Path, with_title: bool) -> None:
    setup_style()
    plot_mask = (TARGET_WAVELENGTHS >= PLOT_RANGE[0]) & (TARGET_WAVELENGTHS <= PLOT_RANGE[1])
    plot_wave = TARGET_WAVELENGTHS[plot_mask]
    colors = {"Low C_FE": "#2454A6", "High C_FE": "#B5392D"}

    fig, axes = plt.subplots(1, 3, figsize=(17.6, 5.2), constrained_layout=True, sharey=True)
    if with_title:
        fig.suptitle("CH G-band absorption depth for low and high carbon abundance groups", fontsize=14)

    for ax, panel, profile in zip(axes, panels, profiles):
        for name in ["Low C_FE", "High C_FE"]:
            stats = panel["groups"][name]
            absorption = 1.0 - profile[name]
            ax.plot(
                plot_wave,
                absorption,
                color=colors[name],
                linewidth=2.7,
                label=f"{name} (N={stats['used_n']}, median [C/Fe]={stats['C_FE_median']:.2f})",
            )

        ax.axvspan(*EGP_NUMERATOR_RANGE, color="#F6AD55", alpha=0.18)
        ax.axvspan(*EGP_DENOMINATOR_RANGE, color="#68D391", alpha=0.16)
        ax.set_xlim(*PLOT_RANGE)
        ax.set_xlabel("Rest wavelength (Å)", fontsize=12)
        if with_title:
            ax.set_title(panel["label"], fontsize=12)

        window = panel["parameter_window"]
        text = (
            f"TEFF: {window['TEFF'][0]:.0f}-{window['TEFF'][1]:.0f} K\n"
            f"LOGG: {window['LOGG'][0]:.1f}-{window['LOGG'][1]:.1f}\n"
            f"[Fe/H]: {window['FE_H'][0]:.2f} to {window['FE_H'][1]:.2f}\n"
            f"Parent N={panel['window_n']}"
        )
        ax.text(
            0.03,
            0.97,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9.1,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.88, edgecolor="#cccccc"),
        )
        ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.94)

    axes[0].set_ylabel("Absorption depth: 1 - normalized flux", fontsize=12)
    fig.savefig(output_png, dpi=260, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_input(args.input_csv)
    panels, profiles = build_profiles(df, args.max_per_group, args.seed)

    output_png = args.output_dir / f"{args.prefix}.png"
    output_pdf = args.output_dir / f"{args.prefix}.pdf"
    plot_profiles(panels, profiles, output_png, output_pdf, args.with_title)

    print(f"saved: {output_png}")
    print(f"saved: {output_pdf}")


if __name__ == "__main__":
    main()
