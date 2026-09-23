#!/usr/bin/env python3
"""Step 2: Compute EGP for the phase-space-selected candidates.

Reads the candidates CSV (from step 01) and merges upstream FITS paths +
redshifts.  For each spectrum: de-redshift, interpolate to a common
wavelength grid (3900-8800 A at 1 A/pixel), normalise via a smooth
continuum (Savgol + 5th-degree polynomial), 3-sigma clip, and compute
EGP = -2.5*log10(F_{4200-4400} / F_{4425-4520}).

Input:  candidates.csv + upstream cemp_unique_b_greater_30.csv (FITS paths, z)
Output: egp.csv
"""

from __future__ import annotations

import argparse
import os
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

LGADNET_ROOT = Path(os.environ.get("LGADNET_ROOT", "/path/to/lgadnet_data"))
DEFAULT_UPSTREAM = LGADNET_ROOT / "crossmatch/cemp_unique_b_greater_30.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-csv", type=Path, required=True)
    p.add_argument("--upstream-csv", type=Path, default=DEFAULT_UPSTREAM)
    p.add_argument("--outdir", type=Path, required=True)
    return p.parse_args()


def get_egp(flux) -> float:
    """EGP on the already-sampled TARGET_WAVELENGTHS grid (3900..8800, step 1).

    flux is the normalised spectrum on that fixed grid; numerator/denominator
    bands are fixed integer slices, so no re-interpolation is needed.
    """
    w0 = TARGET_WAVELENGTHS[0]  # 3900
    i_num = slice(EGP_NUMERATOR_RANGE[0] - w0, EGP_NUMERATOR_RANGE[1] - w0)
    i_den = slice(EGP_DENOMINATOR_RANGE[0] - w0, EGP_DENOMINATOR_RANGE[1] - w0)
    num = np.trapezoid(flux[i_num])
    den = np.trapezoid(flux[i_den])
    if not np.isfinite(num) or not np.isfinite(den) or num <= 0 or den <= 0:
        return float("nan")
    return float(-2.5 * np.log10(num / den))


def preprocess_and_egp(spectrum_path: Path, z: float) -> tuple[float | None, str]:
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

    finite = np.isfinite(wavelength) & np.isfinite(flux)
    if finite.sum() < 2:
        return None, "too_few_finite_points"
    wavelength_rest = wavelength[finite] / (1.0 + z)
    flux = flux[finite]
    order = np.argsort(wavelength_rest)
    wavelength_rest = wavelength_rest[order]
    flux = flux[order]
    wavelength_rest, uidx = np.unique(wavelength_rest, return_index=True)
    flux = flux[uidx]

    tmin, tmax = WAVELENGTH_SCOPE
    if wavelength_rest.size < 2:
        return None, "too_few_unique_points"
    if float(np.min(wavelength_rest)) > tmin or float(np.max(wavelength_rest)) < tmax:
        return None, "wave_range_not_cover"

    m = (wavelength_rest >= tmin) & (wavelength_rest <= tmax)
    wc = wavelength_rest[m]
    fc = flux[m]
    if wc.size < 2:
        return None, "too_few_points_in_scope"
    try:
        fi = interp1d(wc, fc, kind="cubic" if wc.size >= 4 else "linear",
                      bounds_error=False, fill_value=(fc[0], fc[-1]))
        fint = fi(TARGET_WAVELENGTHS)
    except Exception:
        return None, "interp_error"
    if not np.all(np.isfinite(fint)):
        return None, "interp_non_finite"
    try:
        fd = savgol_filter(fint, window_length=15, polyorder=3, mode="nearest")
        cont = Polynomial.fit(TARGET_WAVELENGTHS, fd, 5)(TARGET_WAVELENGTHS)
    except Exception:
        return None, "continuum_fit_error"
    if not np.all(np.isfinite(cont)):
        return None, "continuum_non_finite"
    if np.any(np.abs(cont) < 1e-12):
        return None, "continuum_zero"
    nf = fint / cont
    if not np.all(np.isfinite(nf)):
        return None, "normalized_non_finite"
    mn = float(np.mean(nf))
    sd = float(np.std(nf))
    if not np.isfinite(mn) or not np.isfinite(sd) or sd <= 0:
        return None, "invalid_stats"
    nf[nf > mn + 3 * sd] = mn
    nf[nf < mn - 3 * sd] = mn
    egp = get_egp(nf)
    if not np.isfinite(egp):
        return None, "egp_non_finite"
    return egp, "success"


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"[03] Reading candidates: {args.in_csv}")
    df = pd.read_csv(args.in_csv)

    print(f"[03] Reading upstream (FITS paths): {args.upstream_csv}")
    upstream = pd.read_csv(args.upstream_csv,
                           usecols=["candidate_id", "source_path", "z"])
    if upstream["candidate_id"].nunique() != len(upstream):
        upstream = upstream.drop_duplicates(subset="candidate_id", keep="first")

    df["candidate_id"] = pd.to_numeric(df["candidate_id"], errors="coerce").astype("Int64")
    upstream["candidate_id"] = pd.to_numeric(upstream["candidate_id"], errors="coerce").astype("Int64")

    merged = df.merge(upstream, on="candidate_id", how="left")
    missing = int(merged["source_path"].isna().sum())
    if missing > 0:
        print(f"[03] Warning: {missing} candidates missing source_path")

    merged["EGP"] = np.nan
    merged["EGP_status"] = ""
    err_counts: dict[str, int] = {}

    print(f"[03] Computing EGP for {len(merged):,} spectra...")
    for idx, row in tqdm(merged.iterrows(), total=len(merged)):
        if pd.isna(row["source_path"]) or pd.isna(row["z"]):
            merged.at[idx, "EGP_status"] = "missing_path_or_z"
            continue
        path = Path(str(row["source_path"]))
        egp, status = preprocess_and_egp(path, row["z"])
        if egp is None:
            err_counts[status] = err_counts.get(status, 0) + 1
            merged.at[idx, "EGP_status"] = status
        else:
            merged.at[idx, "EGP"] = egp
            merged.at[idx, "EGP_status"] = status

    out = args.outdir / "egp.csv"
    merged.to_csv(out, index=False)

    egp_vals = merged["EGP"].dropna()
    print(f"[03] EGP success: {len(egp_vals):,}/{len(merged):,}")
    if len(egp_vals) > 0:
        print(f"[03] EGP stats: median={egp_vals.median():.3f}, "
              f"mean={egp_vals.mean():.3f}, std={egp_vals.std():.3f}")
    print(f"[03] Error counts: {err_counts}")
    print(f"[03] Output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())