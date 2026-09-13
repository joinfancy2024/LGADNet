#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calculate EGP for the training datasets (selected-samples, train, val).

The final CEMP catalog already carries an EGP column (computed in 04_validate by
03_calculate_egp and exported by 04_export_final_catalog), so it is not recomputed
here. This script only adds EGP to the three LGADNet training related samples used
in the EGP comparison figures.
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

LGADNET_ROOT = Path("/path/to/your/lgadnet_data")   # <-- EDIT THIS root

# Defaults below: dataset files produced by 01_prepare_training_dataset.py (train/val/selected).
DEFAULT_SELECTED_INPUT = LGADNET_ROOT / "lgadnet_dataset" / "selected_samples.csv"
DEFAULT_SELECTED_OUTPUT = LGADNET_ROOT / "lgadnet_dataset" / "selected_samples_with_egp.csv"
DEFAULT_TRAIN_INPUT = LGADNET_ROOT / "lgadnet_dataset" / "train_labels.csv"
DEFAULT_TRAIN_OUTPUT = LGADNET_ROOT / "lgadnet_dataset" / "train_labels_with_egp.csv"
DEFAULT_VAL_INPUT = LGADNET_ROOT / "lgadnet_dataset" / "val_labels.csv"
DEFAULT_VAL_OUTPUT = LGADNET_ROOT / "lgadnet_dataset" / "val_labels_with_egp.csv"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-input", type=Path, default=DEFAULT_SELECTED_INPUT)
    parser.add_argument("--selected-output", type=Path, default=DEFAULT_SELECTED_OUTPUT)
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRAIN_INPUT)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--val-input", type=Path, default=DEFAULT_VAL_INPUT)
    parser.add_argument("--val-output", type=Path, default=DEFAULT_VAL_OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--skip-selected", action="store_true", help="Skip the selected-samples dataset.")
    parser.add_argument("--skip-train", action="store_true", help="Skip train dataset.")
    parser.add_argument("--skip-val", action="store_true", help="Skip val dataset.")
    return parser.parse_args()


def require_overwrite(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists; use --overwrite to replace: {path}")


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


def process_direct_dataset(
    input_csv: Path,
    output_csv: Path,
    limit: int | None,
    overwrite: bool,
) -> dict:
    """Process a dataset directly (selected-samples, train, val)."""
    require_overwrite(output_csv, overwrite)

    df = pd.read_csv(input_csv)
    path_column = None
    for candidate in ("source_path", "fits_source_path"):
        if candidate in df.columns:
            path_column = candidate
            break
    if path_column is None:
        raise ValueError(f"{input_csv} is missing the source_path / fits_source_path column")
    if "z" not in df.columns:
        raise ValueError(f"{input_csv} is missing the z column")

    work = df.copy()
    if limit is not None:
        work = work.head(limit).copy()

    work["EGP"] = np.nan
    work["EGP_status"] = "pending"

    error_counts: dict[str, int] = {}
    for idx, row in tqdm(work.iterrows(), total=len(work), desc=input_csv.name):
        path = Path(str(row[path_column]))
        egp, status = preprocess_spectrum_for_egp(path, float(row["z"]))
        if egp is None:
            error_counts[status] = error_counts.get(status, 0) + 1
            work.at[idx, "EGP_status"] = status
            continue

        work.at[idx, "EGP"] = egp
        work.at[idx, "EGP_status"] = status

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    work.to_csv(output_csv, index=False)

    egp_values = work["EGP"].dropna()
    stats = {
        "total": int(len(work)),
        "success": int(work["EGP"].notna().sum()),
        "failed": int(work["EGP"].isna().sum()),
        "error_counts": error_counts,
        "egp_mean": float(egp_values.mean()) if len(egp_values) > 0 else None,
        "egp_std": float(egp_values.std()) if len(egp_values) > 0 else None,
        "egp_median": float(egp_values.median()) if len(egp_values) > 0 else None,
    }
    return stats


def main() -> int:
    args = parse_args()

    if not args.skip_selected:
        log(f"Processing selected-samples file: {args.selected_input}")
        stats = process_direct_dataset(
            input_csv=args.selected_input,
            output_csv=args.selected_output,
            limit=args.limit,
            overwrite=args.overwrite,
        )
        log(f"Done: success={stats['success']}/{stats['total']}, failed={stats['failed']}")
        log(f"EGP: mean={stats['egp_mean']:.3f}, std={stats['egp_std']:.3f}, median={stats['egp_median']:.3f}" if stats['egp_mean'] else "EGP: no valid data")
        log(f"Output: {args.selected_output}")

    if not args.skip_train:
        log(f"Processing train: {args.train_input}")
        stats = process_direct_dataset(
            input_csv=args.train_input,
            output_csv=args.train_output,
            limit=args.limit,
            overwrite=args.overwrite,
        )
        log(f"Done: success={stats['success']}/{stats['total']}, failed={stats['failed']}")
        log(f"EGP: mean={stats['egp_mean']:.3f}, std={stats['egp_std']:.3f}, median={stats['egp_median']:.3f}" if stats['egp_mean'] else "EGP: no valid data")
        log(f"Output: {args.train_output}")

    if not args.skip_val:
        log(f"Processing val: {args.val_input}")
        stats = process_direct_dataset(
            input_csv=args.val_input,
            output_csv=args.val_output,
            limit=args.limit,
            overwrite=args.overwrite,
        )
        log(f"Done: success={stats['success']}/{stats['total']}, failed={stats['failed']}")
        log(f"EGP: mean={stats['egp_mean']:.3f}, std={stats['egp_std']:.3f}, median={stats['egp_median']:.3f}" if stats['egp_mean'] else "EGP: no valid data")
        log(f"Output: {args.val_output}")

    log("All done")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
