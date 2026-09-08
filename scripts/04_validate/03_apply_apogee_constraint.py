#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply APOGEE external metallicity constraint to BJ robust CEMP candidates.

Input:
    cemp_bj_robust_cmd_flags.csv
    APOGEE catalog CSV

Outputs:
    cemp_bj_robust_after_apogee.csv
    cemp_bj_robust_apogee_summary.txt
"""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord


INPUT = Path(
    "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/"
    "spatial_filtering_unique/cemp_bj_robust_cmd_flags.csv"
)
APOGEE_FILE = Path("/home/DM13/workspace/sky/data/apogee_dr17.csv")

OUTPUT_CSV = Path(
    "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/"
    "spatial_filtering_unique/cemp_bj_robust_after_apogee.csv"
)
OUTPUT_MATCHED_CSV = Path(
    "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/"
    "spatial_filtering_unique/cemp_bj_robust_apogee_matches.csv"
)
OUTPUT_REPORT = Path(
    "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/"
    "spatial_filtering_unique/cemp_bj_robust_apogee_summary.txt"
)

MATCH_RADIUS_ARCSEC = 3.0
APOGEE_FEH_THRESHOLD = -1.0


def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    if df[col].dtype == bool:
        return df[col].fillna(False)
    return df[col].astype(str).str.lower().isin(["true", "1", "yes"])


def main() -> int:
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)
    if not APOGEE_FILE.is_file():
        raise FileNotFoundError(APOGEE_FILE)

    cand = pd.read_csv(INPUT, engine="python")
    apogee = pd.read_csv(APOGEE_FILE, engine="python")

    if "Source" in cand.columns:
        cand["Source"] = cand["Source"].astype(str).str.strip()

    cand_ra = pick_col(cand, ["ra", "RA", "ra_original", "RAdeg", "RAJ2000"])
    cand_dec = pick_col(cand, ["dec", "DEC", "dec_original", "DEdeg", "DEJ2000"])
    if cand_ra is None or cand_dec is None:
        raise ValueError("Cannot find candidate RA/Dec columns")

    apo_ra = pick_col(apogee, ["RA", "ra", "RAJ2000", "raj2000"])
    apo_dec = pick_col(apogee, ["DEC", "dec", "DEJ2000", "dej2000"])
    apo_feh = pick_col(
        apogee,
        [
            "FE_H",
            "fe_h",
            "FEH",
            "feh",
            "M_H",
            "m_h",
            "PARAM_FE_H",
            "param_fe_h",
        ],
    )

    if apo_ra is None or apo_dec is None:
        raise ValueError("Cannot find APOGEE RA/Dec columns")
    if apo_feh is None:
        raise ValueError("Cannot find APOGEE [Fe/H] column")

    cand = to_numeric(cand, [cand_ra, cand_dec])
    apogee = to_numeric(apogee, [apo_ra, apo_dec, apo_feh])

    cand_valid = cand[cand[cand_ra].notna() & cand[cand_dec].notna()].copy()
    apo_valid = apogee[apogee[apo_ra].notna() & apogee[apo_dec].notna()].copy()

    print(f"Input candidates: {len(cand):,}")
    print(f"Candidates with coordinates: {len(cand_valid):,}")
    print(f"APOGEE rows: {len(apogee):,}")
    print(f"APOGEE rows with coordinates: {len(apo_valid):,}")
    print(f"Using candidate coordinates: {cand_ra}, {cand_dec}")
    print(f"Using APOGEE coordinates: {apo_ra}, {apo_dec}")
    print(f"Using APOGEE metallicity: {apo_feh}")

    cand_coord = SkyCoord(
        ra=cand_valid[cand_ra].values * u.deg,
        dec=cand_valid[cand_dec].values * u.deg,
        frame="icrs",
    )
    apo_coord = SkyCoord(
        ra=apo_valid[apo_ra].values * u.deg,
        dec=apo_valid[apo_dec].values * u.deg,
        frame="icrs",
    )

    idx, sep2d, _ = cand_coord.match_to_catalog_sky(apo_coord)
    matched_mask = sep2d.arcsec <= MATCH_RADIUS_ARCSEC

    matched_candidates = cand_valid.loc[matched_mask].copy()
    matched_apogee = apo_valid.iloc[idx[matched_mask]].copy().reset_index(drop=True)
    matched_candidates = matched_candidates.reset_index(drop=False).rename(columns={"index": "candidate_row_index"})

    matched_apogee = matched_apogee.add_prefix("apogee_")
    matched = pd.concat([matched_candidates, matched_apogee], axis=1)
    matched["apogee_sep_arcsec"] = sep2d.arcsec[matched_mask]

    apogee_feh_col = f"apogee_{apo_feh}"
    matched[apogee_feh_col] = pd.to_numeric(matched[apogee_feh_col], errors="coerce")
    matched["apogee_feh_available"] = matched[apogee_feh_col].notna()
    matched["apogee_non_mp"] = matched["apogee_feh_available"] & (
        matched[apogee_feh_col] >= APOGEE_FEH_THRESHOLD
    )

    remove_indices = set(matched.loc[matched["apogee_non_mp"], "candidate_row_index"].tolist())

    final = cand.copy()
    final["has_apogee_match"] = False
    final["apogee_feh"] = np.nan
    final["apogee_sep_arcsec"] = np.nan
    final["apogee_non_mp"] = False
    final["removed_by_apogee"] = False

    for _, row in matched.iterrows():
        i = int(row["candidate_row_index"])
        final.loc[i, "has_apogee_match"] = True
        final.loc[i, "apogee_feh"] = row[apogee_feh_col]
        final.loc[i, "apogee_sep_arcsec"] = row["apogee_sep_arcsec"]
        final.loc[i, "apogee_non_mp"] = bool(row["apogee_non_mp"])

    if remove_indices:
        final.loc[list(remove_indices), "removed_by_apogee"] = True
    final_kept = final[~final["removed_by_apogee"]].copy()

    cmd_retained = bool_series(final_kept, "cmd_retained")
    cmd_hot_blue = bool_series(final_kept, "cmd_hot_blue")
    cmd_has_data = bool_series(final_kept, "cmd_has_data")
    cmd_missing = ~cmd_has_data

    lines = []
    add = lines.append
    add("=" * 80)
    add("APOGEE External Metallicity Constraint Summary")
    add("=" * 80)
    add(f"Input BJ robust candidates: {len(cand):,}")
    add(f"APOGEE match radius: {MATCH_RADIUS_ARCSEC:.1f} arcsec")
    add(f"APOGEE matched candidates: {len(matched):,}")
    add(f"APOGEE [Fe/H] available: {matched['apogee_feh_available'].sum():,}")
    add(f"APOGEE [Fe/H] >= {APOGEE_FEH_THRESHOLD}: {matched['apogee_non_mp'].sum():,}")
    add(f"Removed by APOGEE metallicity: {len(remove_indices):,}")
    add(f"Final kept candidates: {len(final_kept):,}")
    add("")
    add("CMD flags among final kept candidates:")
    add(f"  CMD retained: {cmd_retained.sum():,}")
    add(f"  CMD hot/blue: {cmd_hot_blue.sum():,}")
    add(f"  CMD missing: {cmd_missing.sum():,}")
    add("")
    if matched["apogee_feh_available"].sum() > 0:
        feh = matched.loc[matched["apogee_feh_available"], apogee_feh_col]
        add("APOGEE [Fe/H] for matched candidates:")
        add(f"  median: {feh.median():.3f}")
        add(f"  16%-84%: {feh.quantile(0.16):.3f} - {feh.quantile(0.84):.3f}")
        add(f"  min/max: {feh.min():.3f} / {feh.max():.3f}")

    report = "\n".join(lines)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    final_kept.to_csv(OUTPUT_CSV, index=False)
    matched.to_csv(OUTPUT_MATCHED_CSV, index=False)
    OUTPUT_REPORT.write_text(report, encoding="utf-8")

    print(report)
    print("=" * 80)
    print(f"Output final CSV: {OUTPUT_CSV}")
    print(f"Output matched CSV: {OUTPUT_MATCHED_CSV}")
    print(f"Output report: {OUTPUT_REPORT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
