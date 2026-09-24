#!/usr/bin/env python3
"""Build a full metal-poor catalog with screening-step flags.

Reads all predictions, retains rows classified as metal-poor, derives Galactic
coordinates, and reads DESIG and OBJNAME from the source FITS headers. The
candidate bridge maps observation IDs to candidate IDs used by the crossmatch
and screening tables. Screening outcomes and available phase-space measurements
are added to the parent sample; rows without measurements remain empty.

Flag values are 1 = passed, 0 = failed, and 2 = not applicable to non-CEMP
stars. CEMP stars carry 0 or 1 in every CEMP-specific screening flag, including
when they were rejected at an earlier step.

Input:  predictions_all.csv, source FITS files, candidate bridge, Gaia
        crossmatch, phase-space table, and CMD screening table under LGADNET_ROOT
Output: mp_catalog.csv under --outdir
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u

LGADNET_ROOT = Path(os.environ.get("LGADNET_ROOT", "/path/to/lgadnet_data"))

DEFAULT_PREDICTIONS = LGADNET_ROOT / "predictions/predictions_all.csv"
DEFAULT_BRIDGE = LGADNET_ROOT / "crossmatch/cemp_unique_b_greater_30.csv"
DEFAULT_GAIA = LGADNET_ROOT / "crossmatch/vizier_dedup_double.csv"
DEFAULT_PHASE = LGADNET_ROOT / "screening/phase.csv"
DEFAULT_CMD = LGADNET_ROOT / "screening/cmd_flags.csv"
DEFAULT_OUTDIR = LGADNET_ROOT / "catalog"

B_THRESHOLD = 30.0
VMP_FEH = -2.0

PASS, FAIL, NOT_APPLICABLE = 1, 0, 2
FLAG_COLUMNS = ("flag_lat", "flag_gaia_match", "flag_phase_space", "flag_cmd")
DERIVED_COLUMNS = ("abs_Z_BJ_med_kpc", "Vtan_BJ_med_kms")
PARENT_DROP_COLUMNS = ("is_mp", "is_cemp", "class")
WRITE_BUFFER_BYTES = 8 << 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--gaia", type=Path, default=DEFAULT_GAIA)
    parser.add_argument("--phase", type=Path, default=DEFAULT_PHASE)
    parser.add_argument("--cmd", type=Path, default=DEFAULT_CMD)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def bool_series(series: pd.Series) -> pd.Series:
    """Parse boolean values stored as booleans or common CSV strings."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})


def read_fits_identifiers(source_path: object) -> tuple[str, str]:
    """Read target identifiers from a source FITS primary header."""
    if pd.isna(source_path):
        return "", ""
    try:
        with fits.open(str(source_path), memmap=False) as hdul:
            header = hdul[0].header
            desig = header.get("DESIG")
            objname = header.get("OBJNAME")
    except Exception:
        return "", ""
    return ("" if desig is None else str(desig).strip(),
            "" if objname is None else str(objname).strip())


def prepare_parent(predictions: pd.DataFrame) -> pd.DataFrame:
    """Select valid metal-poor predictions and add coordinates and identifiers."""
    parent = predictions.loc[bool_series(predictions["is_mp"])].copy()
    identifiers = [read_fits_identifiers(path) for path in parent["source_path"]]
    parent["DESIG"] = [item[0] for item in identifiers]
    parent["OBJNAME"] = [item[1] for item in identifiers]
    coords = SkyCoord(
        ra=pd.to_numeric(parent["ra"], errors="coerce").to_numpy() * u.deg,
        dec=pd.to_numeric(parent["dec"], errors="coerce").to_numpy() * u.deg,
        frame="icrs",
    )
    parent["b"] = coords.galactic.b.deg
    parent["l"] = coords.galactic.l.deg
    return parent


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions, dtype={"obsid": str, "source_path": str})
    parent = prepare_parent(predictions)

    bridge = pd.read_csv(
        args.bridge,
        usecols=["obsid", "candidate_id"],
        dtype={"obsid": str, "candidate_id": str},
    )
    candidate_id = parent["obsid"].map(dict(zip(bridge["obsid"], bridge["candidate_id"])))

    gaia_ids = set(
        pd.read_csv(args.gaia, usecols=["candidate_id"], dtype={"candidate_id": str})[
            "candidate_id"
        ]
    )

    phase = pd.read_csv(
        args.phase,
        dtype={"candidate_id": str},
        usecols=[
            "candidate_id",
            "phase_both_pass",
            "abs_Z_BJ_med_kpc",
            "Vtan_BJ_med_kms",
        ],
    )
    phase_pass = dict(zip(phase["candidate_id"], bool_series(phase["phase_both_pass"])))

    cmd = pd.read_csv(
        args.cmd,
        usecols=["candidate_id", "cmd_retained"],
        dtype={"candidate_id": str},
    )
    cmd_pass = dict(zip(cmd["candidate_id"], bool_series(cmd["cmd_retained"])))

    is_cemp = bool_series(parent["is_cemp"]).to_numpy()
    parent["flag_vmp"] = (pd.to_numeric(parent["FE_H"], errors="coerce") < VMP_FEH).astype(int)
    parent["flag_cemp"] = is_cemp.astype(int)

    latitude = pd.to_numeric(parent["b"], errors="coerce").abs().to_numpy()
    flag_lat = np.where(is_cemp, np.where(latitude > B_THRESHOLD, PASS, FAIL), NOT_APPLICABLE)
    parent["flag_lat"] = flag_lat

    reaches_downstream = (flag_lat == PASS) & candidate_id.notna().to_numpy()

    def downstream_flag(passed: pd.Series) -> np.ndarray:
        """Return 1 for passed, 0 for failed CEMP stars, and 2 for non-CEMP stars."""
        flag = np.full(len(parent), NOT_APPLICABLE, dtype=int)
        flag[is_cemp] = FAIL
        flag[reaches_downstream] = np.where(passed.to_numpy()[reaches_downstream], PASS, FAIL)
        return flag

    parent["flag_gaia_match"] = downstream_flag(candidate_id.isin(gaia_ids))
    parent["flag_phase_space"] = downstream_flag(
        candidate_id.map(phase_pass).fillna(False).astype(bool)
    )
    parent["flag_cmd"] = downstream_flag(
        candidate_id.map(cmd_pass).fillna(False).astype(bool)
    )

    parent["abs_Z_BJ_med_kpc"] = candidate_id.map(
        dict(zip(phase["candidate_id"], phase["abs_Z_BJ_med_kpc"]))
    )
    parent["Vtan_BJ_med_kms"] = candidate_id.map(
        dict(zip(phase["candidate_id"], phase["Vtan_BJ_med_kms"]))
    )

    derived = list(FLAG_COLUMNS) + list(DERIVED_COLUMNS)
    catalog = parent.drop(columns=[column for column in PARENT_DROP_COLUMNS if column in parent.columns])
    catalog = catalog[[column for column in catalog.columns if column not in derived] + derived]

    output_csv = args.outdir / "mp_catalog.csv"
    with open(output_csv, "w", newline="", buffering=WRITE_BUFFER_BYTES) as output_file:
        catalog.to_csv(output_file, index=False)

    print(f"Catalog: {output_csv} ({len(catalog):,} rows x {catalog.shape[1]} columns)")
    for column in ("flag_cemp", "flag_vmp", *FLAG_COLUMNS):
        print(f"  {column} = 1: {int((catalog[column] == 1).sum()):,}")
    print(f"  abs_Z_BJ_med_kpc filled: {int(catalog['abs_Z_BJ_med_kpc'].notna().sum()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
