#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a CDS/submission-ready CEMP candidate catalog from the final CMD-retained catalog.

Reads the final CMD-retained catalog (produced by 04_validate/04_export_final_catalog) and
emits a release CSV with publication column names and uniform per-star uncertainties
(the validation-set RMSE). Reading is read-only; nothing in the source tree is modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LGADNET_ROOT = Path("/path/to/your/lgadnet_data")   # <-- EDIT THIS root

DEFAULT_INPUT = LGADNET_ROOT / "final_validated" / "cemp_final_cmdretained.csv"
DEFAULT_RELEASE_DIR = LGADNET_ROOT / "release"
DEFAULT_OUTPUT = DEFAULT_RELEASE_DIR / "cemp_dr12_catalog.csv"
DEFAULT_README = DEFAULT_RELEASE_DIR / "cemp_catalog_README.txt"

# Validation-set RMSE on the physical scale (reported in the paper); used as a uniform
# per-star uncertainty until an ensemble / SNR-resolved sigma becomes available.
SIGMA = {
    "e_Teff": 152.42,   # K
    "e_Logg": 0.375,    # dex
    "e_Fe_H": 0.247,    # dex
    "e_C_Fe": 0.270,    # dex
}

# Input (final catalog) column -> release column.
COL_MAP = {
    "ObsID":            "ObsID",
    "RA":               "RA",
    "DEC":              "DEC",
    "Gaia_Source":      "Gaia_Source",
    "Teff_LGADNet":     "Teff",
    "Logg_LGADNet":     "Logg",
    "FeH_LGADNet":      "Fe_H",
    "CFe_LGADNet":      "C_Fe",
    "Dist_BJ_lo_kpc":   "Dist_BJ_lo_kpc",
    "Abs_Z_BJ_lo_kpc":  "Abs_Z_BJ_lo_kpc",
    "Vtan_BJ_lo_kms":   "Vtan_BJ_lo_kms",
    "BP_RP_0":          "BP_RP_0",
    "M_G_BJ_lo":        "M_G_BJ_lo",
    "RUWE":             "RUWE",
    "EGP":              "EGP",
    "APOGEE_FeH":       "APOGEE_FeH",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="final CMD-retained catalog CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="release catalog CSV to write")
    parser.add_argument("--readme", type=Path, default=DEFAULT_README,
                        help="README text to write")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"final catalog does not exist: {args.input}")

    df = pd.read_csv(args.input)
    missing = [k for k in COL_MAP if k not in df.columns]
    if missing:
        raise KeyError(f"final catalog is missing columns: {missing}")

    out = pd.DataFrame()
    for src_col, dst_col in COL_MAP.items():
        out[dst_col] = df[src_col]

    # Append uncertainty columns (uniform per-star = validation-set RMSE).
    for col, val in SIGMA.items():
        out[col] = val

    # Sort by [Fe/H] ascending, [C/Fe] descending.
    out = out.sort_values(["Fe_H", "C_Fe"], ascending=[True, False]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    med = {
        "Fe_H_median": out["Fe_H"].median(),
        "C_Fe_median": out["C_Fe"].median(),
        "Z_median":    out["Abs_Z_BJ_lo_kpc"].median(),
        "Vtan_median": out["Vtan_BJ_lo_kms"].median(),
    }
    report = "\n".join([
        "=" * 78,
        "LAMOST DR12 CEMP candidate catalog (CDS-ready)",
        "=" * 78,
        f"Rows            : {len(out):,}",
        f"Columns         : {len(out.columns)}",
        f"[Fe/H] median   : {med['Fe_H_median']:.2f} dex",
        f"[C/Fe] median   : {med['C_Fe_median']:.2f} dex",
        f"|Z_BJ,lo| median: {med['Z_median']:.2f} kpc",
        f"Vtan_BJ,lo med  : {med['Vtan_median']:.0f} km/s",
        f"Output          : {args.output}",
        "NOTE: sigma columns are the validation-set RMSE used as a uniform per-star",
        "uncertainty (override if ensemble/SNR-resolved sigma becomes available).",
        "NOTE: reading is read-only; nothing in the source tree is modified.",
    ])
    print(report)
    args.readme.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())