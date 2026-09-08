#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export final CEMP candidate catalog and a compact sample table.

Input:
    cemp_bj_robust_after_apogee.csv

Outputs:
    cemp_bj_robust_final_catalog.csv
    cemp_bj_robust_cmdretained_final_catalog.csv  (CMD-retained subset)
    cemp_bj_robust_final_catalog_sample.csv
    cemp_bj_robust_final_catalog_summary.txt
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT = Path(
    "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/"
    "spatial_filtering_unique/cemp_bj_robust_after_apogee.csv"
)
ORIG_FILE = Path(
    "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/"
    "spatial_filtering_unique/cemp_unique_b_greater_30.csv"
)

OUTPUT_FULL = Path(
    "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/"
    "spatial_filtering_unique/cemp_bj_robust_final_catalog.csv"
)
OUTPUT_CMDRETAINED = Path(
    "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/"
    "spatial_filtering_unique/cemp_bj_robust_cmdretained_final_catalog.csv"
)
OUTPUT_SAMPLE = Path(
    "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/"
    "spatial_filtering_unique/cemp_bj_robust_final_catalog_sample.csv"
)
OUTPUT_REPORT = Path(
    "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/"
    "spatial_filtering_unique/cemp_bj_robust_final_catalog_summary.txt"
)

SAMPLE_SIZE = 16


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    if df[col].dtype == bool:
        return df[col].fillna(False)
    return df[col].astype(str).str.lower().isin(["true", "1", "yes"])


def make_cmd_flag(df: pd.DataFrame) -> pd.Series:
    cmd_retained = bool_series(df, "cmd_retained")
    cmd_hot_blue = bool_series(df, "cmd_hot_blue")
    cmd_has_data = bool_series(df, "cmd_has_data")

    flag = pd.Series("cmd_missing", index=df.index, dtype="object")
    flag.loc[cmd_retained] = "cmd_retained"
    flag.loc[cmd_hot_blue] = "cmd_hot_blue"
    flag.loc[~cmd_has_data] = "cmd_missing"
    return flag


def main() -> int:
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)
    if not ORIG_FILE.is_file():
        raise FileNotFoundError(ORIG_FILE)

    df = pd.read_csv(INPUT, engine="python")
    ref = pd.read_csv(ORIG_FILE, engine="python", usecols=["candidate_id", "obsid"])
    ref = ref.drop_duplicates(subset="candidate_id", keep="first")

    if "candidate_id" not in df.columns:
        raise ValueError("Missing required column: candidate_id")

    df = df.merge(ref, on="candidate_id", how="left")

    if "Source" in df.columns:
        df["Source"] = df["Source"].astype(str).str.strip()

    obsid_col = pick_col(df, ["obsid", "ObsID", "OBSID"])
    ra_col = pick_col(df, ["ra", "RA", "ra_original", "RAdeg", "RAJ2000"])
    dec_col = pick_col(df, ["dec", "DEC", "dec_original", "DEdeg", "DEJ2000"])

    required = {
        "obsid": obsid_col,
        "ra": ra_col,
        "dec": dec_col,
    }
    missing = [name for name, col in required.items() if col is None]
    if missing:
        raise ValueError(f"Missing required columns for final catalog: {missing}")

    df = to_numeric(
        df,
        [
            obsid_col,
            ra_col,
            dec_col,
            "TEFF",
            "LOGG",
            "FE_H",
            "C_FE",
            "abs_Z_bj_lo_kpc",
            "Vtan_bj_lo_kms",
            "d_bj_geo_kpc",
            "d_bj_lo_kpc",
            "bj_geo_rel_half_width",
            "bp_rp_0",
            "M_G_bj_med",
            "M_G_bj_lo",
            "EGP",
            "RUWE",
            "epsi",
            "apogee_feh",
            "apogee_sep_arcsec",
        ],
    )

    out = pd.DataFrame()
    out["ObsID"] = df[obsid_col]
    out["RA"] = df[ra_col]
    out["DEC"] = df[dec_col]

    if "Source" in df.columns:
        out["Gaia_Source"] = df["Source"]

    out["Teff_LGADNet"] = df["TEFF"]
    out["Logg_LGADNet"] = df["LOGG"]
    out["FeH_LGADNet"] = df["FE_H"]
    out["CFe_LGADNet"] = df["C_FE"]

    out["Dist_BJ_geo_kpc"] = df["d_bj_geo_kpc"]
    out["Dist_BJ_lo_kpc"] = df["d_bj_lo_kpc"]
    out["Abs_Z_BJ_lo_kpc"] = df["abs_Z_bj_lo_kpc"]
    out["Vtan_BJ_lo_kms"] = df["Vtan_bj_lo_kms"]
    out["BJ_geo_rel_half_width"] = df["bj_geo_rel_half_width"]

    if "bp_rp_0" in df.columns:
        out["BP_RP_0"] = df["bp_rp_0"]
    if "M_G_bj_med" in df.columns:
        out["M_G_BJ_med"] = df["M_G_bj_med"]
    if "M_G_bj_lo" in df.columns:
        out["M_G_BJ_lo"] = df["M_G_bj_lo"]

    out["CMD_Flag"] = make_cmd_flag(df)

    if "RUWE" in df.columns:
        out["RUWE"] = df["RUWE"]
    if "epsi" in df.columns:
        out["Astrometric_Excess_Noise"] = df["epsi"]
    if "VarFlag" in df.columns:
        out["Gaia_VarFlag"] = df["VarFlag"]

    if "EGP" in df.columns:
        out["EGP"] = df["EGP"]
    if "EGP_status" in df.columns:
        out["EGP_status"] = df["EGP_status"]

    if "has_apogee_match" in df.columns:
        out["Has_APOGEE_Match"] = df["has_apogee_match"]
    if "apogee_feh" in df.columns:
        out["APOGEE_FeH"] = df["apogee_feh"]
    if "apogee_sep_arcsec" in df.columns:
        out["APOGEE_Sep_arcsec"] = df["apogee_sep_arcsec"]

    out = out.sort_values(["FeH_LGADNet", "CFe_LGADNet"], ascending=[True, False])

    full = out.copy()

    sample_cols = [
        "ObsID",
        "RA",
        "DEC",
        "Teff_LGADNet",
        "Logg_LGADNet",
        "FeH_LGADNet",
        "CFe_LGADNet",
        "Abs_Z_BJ_lo_kpc",
        "Vtan_BJ_lo_kms",
        "CMD_Flag",
    ]
    sample_cols = [c for c in sample_cols if c in full.columns]

    sample_parts = []
    if "CMD_Flag" in full.columns:
        for flag in ["cmd_retained", "cmd_hot_blue", "cmd_missing"]:
            part = full[full["CMD_Flag"].eq(flag)].head(4)
            if len(part) > 0:
                sample_parts.append(part)
    if sample_parts:
        sample = pd.concat(sample_parts, ignore_index=True)
        if len(sample) < SAMPLE_SIZE:
            used = set(sample["ObsID"].astype(str))
            extra = full[~full["ObsID"].astype(str).isin(used)].head(SAMPLE_SIZE - len(sample))
            sample = pd.concat([sample, extra], ignore_index=True)
        sample = sample.head(SAMPLE_SIZE)
    else:
        sample = full.head(SAMPLE_SIZE)

    sample = sample[sample_cols].copy()

    round_map = {
        "RA": 6,
        "DEC": 6,
        "Teff_LGADNet": 0,
        "Logg_LGADNet": 3,
        "FeH_LGADNet": 3,
        "CFe_LGADNet": 3,
        "Abs_Z_BJ_lo_kpc": 3,
        "Vtan_BJ_lo_kms": 1,
    }
    for col, nd in round_map.items():
        if col in sample.columns:
            sample[col] = pd.to_numeric(sample[col], errors="coerce").round(nd)

    OUTPUT_FULL.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(OUTPUT_FULL, index=False)
    sample.to_csv(OUTPUT_SAMPLE, index=False)

    # Export CMD-retained subset
    cmdretained = full[full["CMD_Flag"] == "cmd_retained"].copy()
    cmdretained.to_csv(OUTPUT_CMDRETAINED, index=False)

    lines = []
    add = lines.append
    add("=" * 80)
    add("Final CEMP Candidate Catalog Export Summary")
    add("=" * 80)
    add(f"Input rows: {len(df):,}")
    add(f"Final catalog rows: {len(full):,}")
    add(f"CMD-retained catalog rows: {len(cmdretained):,}")
    add(f"Sample table rows: {len(sample):,}")
    add("")
    if "CMD_Flag" in full.columns:
        add("CMD flag counts:")
        for k, v in full["CMD_Flag"].value_counts(dropna=False).items():
            add(f"  {k}: {v:,}")
    add("")
    if "EGP_status" in full.columns:
        add("EGP status counts:")
        for k, v in full["EGP_status"].value_counts(dropna=False).items():
            add(f"  {k}: {v:,}")
    add("")
    if "Has_APOGEE_Match" in full.columns:
        has_apo = full["Has_APOGEE_Match"].astype(str).str.lower().isin(["true", "1", "yes"]).sum()
        add(f"APOGEE matched in final catalog: {has_apo:,}")
    add("")
    add(f"Full catalog: {OUTPUT_FULL}")
    add(f"CMD-retained catalog: {OUTPUT_CMDRETAINED}")
    add(f"Sample table: {OUTPUT_SAMPLE}")

    report = "\n".join(lines)
    OUTPUT_REPORT.write_text(report, encoding="utf-8")

    print(report)
    print("=" * 80)
    print(f"Output full catalog: {OUTPUT_FULL}")
    print(f"Output CMD-retained catalog: {OUTPUT_CMDRETAINED}")
    print(f"Output sample table: {OUTPUT_SAMPLE}")
    print(f"Output report: {OUTPUT_REPORT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
