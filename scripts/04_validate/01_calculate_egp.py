#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate EGP for spatial filtering candidates.

Input: bailer_jones_distance_uncertainty_flags.csv
Output: bailer_jones_distance_uncertainty_flags_with_egp.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from tqdm import tqdm


WAVELENGTH_SCOPE = (3900, 8800)
TARGET_WAVELENGTHS = np.arange(WAVELENGTH_SCOPE[0], WAVELENGTH_SCOPE[1] + 1, 1)
EGP_NUMERATOR_RANGE = (4200, 4400)
EGP_DENOMINATOR_RANGE = (4425, 4520)


INPUT = Path(
    "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/"
    "spatial_filtering_unique/bailer_jones_distance_uncertainty_flags.csv"
)
UPSTREAM = Path(
    "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/"
    "spatial_filtering_unique/cemp_unique_b_greater_30.csv"
)
OUTPUT = Path(
    "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/"
    "spatial_filtering_unique/bailer_jones_distance_uncertainty_flags_with_egp.csv"
)


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def get_egp(
    wavelength: np.ndarray,
    flux: np.ndarray,
    numerator_range: tuple[float, float] = EGP_NUMERATOR_RANGE,
    denominator_range: tuple[float, float] = EGP_DENOMINATOR_RANGE,
) -> float:
    """Compute the EGP index on the provided wavelength grid."""
    interpolator = interp1d(wavelength, flux, bounds_error=False, fill_value="extrapolate")

    numerator_range = (
        max(numerator_range[0], float(wavelength.min())),
        min(numerator_range[1], float(wavelength.max())),
    )
    denominator_range = (
        max(denominator_range[0], float(wavelength.min())),
        min(denominator_range[1], float(wavelength.max())),
    )

    if numerator_range[1] <= numerator_range[0] or denominator_range[1] <= denominator_range[0]:
        return float("nan")

    wave_num = np.arange(numerator_range[0], numerator_range[1], 1)
    wave_den = np.arange(denominator_range[0], denominator_range[1], 1)
    if wave_num.size < 2 or wave_den.size < 2:
        return float("nan")

    num = np.trapezoid(interpolator(wave_num), wave_num)
    den = np.trapezoid(interpolator(wave_den), wave_den)

    if not np.isfinite(num) or not np.isfinite(den) or num <= 0 or den <= 0:
        return float("nan")

    return float(-2.5 * np.log10(num / den))


def preprocess_spectrum_for_egp(
    spectrum_path: Path,
    z: float,
) -> tuple[float | None, str]:
    """Read a FITS spectrum and compute EGP with the same preprocessing chain."""
    if not spectrum_path.is_file():
        return None, "fits_not_found"
    if not np.isfinite(z) or z <= -1.0:
        return None, "invalid_z"

    try:
        with fits.open(spectrum_path, memmap=False) as hdul:
            data = hdul[1].data
            wavelength = np.asarray(data["WAVELENGTH"], dtype=np.float64)
            flux = np.asarray(data["FLUX"], dtype=np.float64)
    except Exception:
        return None, "fits_read_error"

    finite_mask = np.isfinite(wavelength) & np.isfinite(flux)
    if finite_mask.sum() < 2:
        return None, "too_few_finite_points"

    wavelength_rest = wavelength[finite_mask] / (1.0 + z)
    flux = flux[finite_mask]

    order = np.argsort(wavelength_rest)
    wavelength_rest = wavelength_rest[order]
    flux = flux[order]

    wavelength_rest, unique_index = np.unique(wavelength_rest, return_index=True)
    flux = flux[unique_index]

    target_min, target_max = WAVELENGTH_SCOPE
    if wavelength_rest.size < 2:
        return None, "too_few_unique_points"
    if float(np.min(wavelength_rest)) > target_min or float(np.max(wavelength_rest)) < target_max:
        return None, "wave_range_not_cover"

    mask = (wavelength_rest >= target_min) & (wavelength_rest <= target_max)
    wavelength_cut = wavelength_rest[mask]
    flux_cut = flux[mask]
    if wavelength_cut.size < 2:
        return None, "too_few_points_in_scope"

    try:
        interpolator = interp1d(
            wavelength_cut,
            flux_cut,
            kind="cubic" if wavelength_cut.size >= 4 else "linear",
            bounds_error=False,
            fill_value=(flux_cut[0], flux_cut[-1]),
        )
        flux_interpolated = interpolator(TARGET_WAVELENGTHS)
    except Exception:
        return None, "interp_error"

    if not np.all(np.isfinite(flux_interpolated)):
        return None, "interp_non_finite"

    try:
        flux_denoised = savgol_filter(
            x=flux_interpolated,
            window_length=15,
            polyorder=3,
            mode="nearest",
        )
        continuum_fit = Polynomial.fit(TARGET_WAVELENGTHS, flux_denoised, 5)
        continuum = continuum_fit(TARGET_WAVELENGTHS)
    except Exception:
        return None, "continuum_fit_error"

    if not np.all(np.isfinite(continuum)):
        return None, "continuum_non_finite"
    if np.any(np.abs(continuum) < 1e-12):
        return None, "continuum_zero"

    normalized_flux = flux_interpolated / continuum
    if not np.all(np.isfinite(normalized_flux)):
        return None, "normalized_non_finite"

    mean_flux = float(np.mean(normalized_flux))
    std_flux = float(np.std(normalized_flux))
    if not np.isfinite(mean_flux) or not np.isfinite(std_flux) or std_flux <= 0:
        return None, "invalid_stats"

    threshold_upper = mean_flux + 3.0 * std_flux
    threshold_lower = mean_flux - 3.0 * std_flux
    normalized_flux[normalized_flux > threshold_upper] = mean_flux
    normalized_flux[normalized_flux < threshold_lower] = mean_flux

    egp = get_egp(TARGET_WAVELENGTHS, normalized_flux)
    if not np.isfinite(egp):
        return None, "egp_non_finite"

    return egp, "success"


def main() -> int:
    log(f"Reading input: {INPUT}")
    df = pd.read_csv(INPUT, engine="python")

    log(f"Reading upstream for FITS paths: {UPSTREAM}")
    upstream = pd.read_csv(UPSTREAM, usecols=["candidate_id", "source_path", "z"], engine="python")

    # Ensure candidate_id is unique in upstream (defensive)
    if upstream["candidate_id"].nunique() != len(upstream):
        log(f"Warning: UPSTREAM has duplicate candidate_id, keeping first occurrence")
        upstream = upstream.drop_duplicates(subset="candidate_id", keep="first")

    # Ensure candidate_id is consistent type
    df["candidate_id"] = pd.to_numeric(df["candidate_id"], errors="coerce").astype("Int64")
    upstream["candidate_id"] = pd.to_numeric(upstream["candidate_id"], errors="coerce").astype("Int64")

    log(f"Merging FITS paths and redshifts...")
    merged = df.merge(upstream, on="candidate_id", how="left")

    missing = int(merged["source_path"].isna().sum())
    if missing > 0:
        log(f"Warning: {missing} candidates missing source_path")

    merged["EGP"] = np.nan
    merged["EGP_status"] = "pending"

    error_counts: dict[str, int] = {}

    log(f"Computing EGP for {len(merged):,} spectra...")
    for idx, row in tqdm(merged.iterrows(), total=len(merged)):
        if pd.isna(row["source_path"]) or pd.isna(row["z"]):
            merged.at[idx, "EGP_status"] = "missing_path_or_z"
            continue

        path = Path(str(row["source_path"]))
        egp, status = preprocess_spectrum_for_egp(path, float(row["z"]))

        if egp is None:
            error_counts[status] = error_counts.get(status, 0) + 1
            merged.at[idx, "EGP_status"] = status
        else:
            merged.at[idx, "EGP"] = egp
            merged.at[idx, "EGP_status"] = status

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT, index=False)

    # Statistics
    egp_values = merged["EGP"].dropna()
    log(f"Completed: success={len(egp_values):,}/{len(merged):,}")
    if len(egp_values) > 0:
        log(f"EGP statistics: mean={egp_values.mean():.3f}, std={egp_values.std():.3f}, median={egp_values.median():.3f}")
    log(f"Error counts: {error_counts}")
    log(f"Output: {OUTPUT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
