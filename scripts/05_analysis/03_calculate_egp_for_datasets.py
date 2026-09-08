#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calculate EGP for LGADNet datasets."""

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

ROOT_DIR = Path("/home/DM13/workspace/sky")
LGADNET_ROOT = ROOT_DIR / "data/new_dataset3/lgadnet"

DEFAULT_SELECTED_INPUT = (
    LGADNET_ROOT / "regression_snr5_selected_v6_cemp3020_20260611/selected_all.csv"
)
DEFAULT_SELECTED_OUTPUT = (
    LGADNET_ROOT / "regression_snr5_selected_v6_cemp3020_20260611/selected_all_with_egp.csv"
)
DEFAULT_TRAIN_INPUT = (
    LGADNET_ROOT / "regression_snr5_selected_v6_cemp3020_20260611/new_dataset_train_y.csv"
)
DEFAULT_TRAIN_OUTPUT = (
    LGADNET_ROOT / "regression_snr5_selected_v6_cemp3020_20260611/new_dataset_train_y_with_egp.csv"
)
DEFAULT_VAL_INPUT = (
    LGADNET_ROOT / "regression_snr5_selected_v6_cemp3020_20260611/new_dataset_val_y.csv"
)
DEFAULT_VAL_OUTPUT = (
    LGADNET_ROOT / "regression_snr5_selected_v6_cemp3020_20260611/new_dataset_val_y_with_egp.csv"
)
DEFAULT_FINAL_INPUT = (
    LGADNET_ROOT / "final_validated/cemp_final_validated.csv"
)
DEFAULT_UPSTREAM_INPUT = (
    LGADNET_ROOT / "spatial_filtering_unique/cemp_unique_b_greater_30.csv"
)
DEFAULT_FINAL_OUTPUT = (
    LGADNET_ROOT / "final_validated/cemp_final_validated_with_egp.csv"
)


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
    parser.add_argument("--final-input", type=Path, default=DEFAULT_FINAL_INPUT)
    parser.add_argument("--upstream-input", type=Path, default=DEFAULT_UPSTREAM_INPUT)
    parser.add_argument("--final-output", type=Path, default=DEFAULT_FINAL_OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--skip-selected", action="store_true", help="Skip selected_all dataset.")
    parser.add_argument("--skip-train", action="store_true", help="Skip train dataset.")
    parser.add_argument("--skip-val", action="store_true", help="Skip val dataset.")
    parser.add_argument("--skip-final", action="store_true", help="Skip final_validated dataset.")
    return parser.parse_args()


def require_overwrite(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在；如需覆盖请加 --overwrite: {path}")


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
    """Process a dataset directly (selected_all, train, val)."""
    require_overwrite(output_csv, overwrite)

    df = pd.read_csv(input_csv)
    path_column = None
    for candidate in ("source_path", "fits_source_path"):
        if candidate in df.columns:
            path_column = candidate
            break
    if path_column is None:
        raise ValueError(f"{input_csv} 缺少 source_path / fits_source_path 列")
    if "z" not in df.columns:
        raise ValueError(f"{input_csv} 缺少 z 列")

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


def process_final_validated_dataset(
    final_csv: Path,
    upstream_csv: Path,
    output_csv: Path,
    limit: int | None,
    overwrite: bool,
) -> dict:
    """Process final_validated dataset (needs upstream to recover FITS paths)."""
    require_overwrite(output_csv, overwrite)

    final_df = pd.read_csv(final_csv)
    if "candidate_id" not in final_df.columns:
        raise ValueError(f"{final_csv} 缺少 candidate_id 列")
    if "z" not in final_df.columns:
        raise ValueError(f"{final_csv} 缺少 z 列")

    upstream = pd.read_csv(upstream_csv, usecols=["candidate_id", "obsid", "filename", "source_path"])

    final_df["candidate_id"] = pd.to_numeric(final_df["candidate_id"], errors="coerce")
    upstream["candidate_id"] = pd.to_numeric(upstream["candidate_id"], errors="coerce")
    if final_df["candidate_id"].isna().any():
        raise ValueError(f"{final_csv} 中存在无法转换为数值的 candidate_id")
    if upstream["candidate_id"].isna().any():
        raise ValueError(f"{upstream_csv} 中存在无法转换为数值的 candidate_id")

    final_df["candidate_id"] = final_df["candidate_id"].astype("int64")
    upstream["candidate_id"] = upstream["candidate_id"].astype("int64")

    merged = final_df.merge(upstream, on="candidate_id", how="left", validate="one_to_one")
    missing = int(merged["source_path"].isna().sum())
    if missing:
        raise ValueError(f"回溯 FITS 路径失败: 缺失 {missing} 条 source_path")

    if limit is not None:
        merged = merged.head(limit).copy()

    merged["EGP"] = np.nan
    merged["EGP_status"] = "pending"

    error_counts: dict[str, int] = {}
    for idx, row in tqdm(merged.iterrows(), total=len(merged), desc=final_csv.name):
        path = Path(str(row["source_path"]))
        egp, status = preprocess_spectrum_for_egp(path, float(row["z"]))
        if egp is None:
            error_counts[status] = error_counts.get(status, 0) + 1
            merged.at[idx, "EGP_status"] = status
            continue

        merged.at[idx, "EGP"] = egp
        merged.at[idx, "EGP_status"] = status

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    egp_values = merged["EGP"].dropna()
    stats = {
        "total": int(len(merged)),
        "success": int(merged["EGP"].notna().sum()),
        "failed": int(merged["EGP"].isna().sum()),
        "error_counts": error_counts,
        "egp_mean": float(egp_values.mean()) if len(egp_values) > 0 else None,
        "egp_std": float(egp_values.std()) if len(egp_values) > 0 else None,
        "egp_median": float(egp_values.median()) if len(egp_values) > 0 else None,
    }
    return stats


def main() -> None:
    args = parse_args()

    if not args.skip_selected:
        log(f"处理 selected_all: {args.selected_input}")
        stats = process_direct_dataset(
            input_csv=args.selected_input,
            output_csv=args.selected_output,
            limit=args.limit,
            overwrite=args.overwrite,
        )
        log(f"完成: success={stats['success']}/{stats['total']}, failed={stats['failed']}")
        log(f"EGP: mean={stats['egp_mean']:.3f}, std={stats['egp_std']:.3f}, median={stats['egp_median']:.3f}" if stats['egp_mean'] else "EGP: 无有效数据")
        log(f"输出: {args.selected_output}")

    if not args.skip_train:
        log(f"处理 train: {args.train_input}")
        stats = process_direct_dataset(
            input_csv=args.train_input,
            output_csv=args.train_output,
            limit=args.limit,
            overwrite=args.overwrite,
        )
        log(f"完成: success={stats['success']}/{stats['total']}, failed={stats['failed']}")
        log(f"EGP: mean={stats['egp_mean']:.3f}, std={stats['egp_std']:.3f}, median={stats['egp_median']:.3f}" if stats['egp_mean'] else "EGP: 无有效数据")
        log(f"输出: {args.train_output}")

    if not args.skip_val:
        log(f"处理 val: {args.val_input}")
        stats = process_direct_dataset(
            input_csv=args.val_input,
            output_csv=args.val_output,
            limit=args.limit,
            overwrite=args.overwrite,
        )
        log(f"完成: success={stats['success']}/{stats['total']}, failed={stats['failed']}")
        log(f"EGP: mean={stats['egp_mean']:.3f}, std={stats['egp_std']:.3f}, median={stats['egp_median']:.3f}" if stats['egp_mean'] else "EGP: 无有效数据")
        log(f"输出: {args.val_output}")

    if not args.skip_final:
        log(f"处理 final_validated: {args.final_input}")
        stats = process_final_validated_dataset(
            final_csv=args.final_input,
            upstream_csv=args.upstream_input,
            output_csv=args.final_output,
            limit=args.limit,
            overwrite=args.overwrite,
        )
        log(f"完成: success={stats['success']}/{stats['total']}, failed={stats['failed']}")
        log(f"EGP: mean={stats['egp_mean']:.3f}, std={stats['egp_std']:.3f}, median={stats['egp_median']:.3f}" if stats['egp_mean'] else "EGP: 无有效数据")
        log(f"输出: {args.final_output}")

    log("全部完成")


if __name__ == "__main__":
    main()
