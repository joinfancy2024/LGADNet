#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一步生成 LGADNet v6 训练数据集。

输出文件仅包含训练实际使用的内容：
- selected_all.csv
- new_dataset_train_y.csv
- new_dataset_val_y.csv
- new_dataset_train_x.npy
- new_dataset_val_x.npy
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm


SKY_ROOT = Path("/home/DM13/workspace/sky")
LGADNET_ROOT = SKY_ROOT / "data/new_dataset3/lgadnet"

DEFAULT_MAPPING_CSV = SKY_ROOT / "data/mapping_result_snr5.csv"
DEFAULT_DR12_CSV = SKY_ROOT / "data/lamost_dr12_cleaned_snr5.csv"
DEFAULT_SPLIT_DIR = LGADNET_ROOT / "dr12_fits/split_all"
DEFAULT_OUTPUT_DIR = LGADNET_ROOT / "regression_snr5_selected_v6_cemp3020_20260611"

DR12_COLUMNS = [
    "obsid",
    "dist_arcsec",
    "lmjd",
    "mjd",
    "planid",
    "spid",
    "fiberid",
    "snrg",
    "class",
    "subclass",
    "z",
    "ra",
    "dec",
    "teff",
    "feh",
    "rv",
    "logg",
]

LABEL_COLUMNS = [
    "RA",
    "DEC",
    "LOGG",
    "TEFF",
    "C_FE",
    "FE_H",
    "source",
    "obsid",
    "uid",
    "snrg",
    "subclass",
    "z",
    "fits_filename",
    "fits_source_path",
    "fits_exists",
    "group_id",
    "logLLodot",
    "cemp_cfe_threshold",
    "is_mp",
    "is_cemp",
    "class",
    "cfe_margin",
    "hard_negative",
    "FE_H_bin",
    "TEFF_bin",
    "C_FE_bin",
    "snrg_bin",
    "sample_stratum",
    "split_stratum",
]

FEH_BINS = [-np.inf, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, np.inf]
FEH_LABELS = [
    "<-3.0",
    "-3.0~-2.5",
    "-2.5~-2.0",
    "-2.0~-1.5",
    "-1.5~-1.0",
    "-1.0~-0.5",
    "-0.5~0.0",
    "0.0~0.5",
    ">0.5",
]

TEFF_BINS = [-np.inf, 4500, 5000, 5500, 5800, 6000, 6200, 6500, np.inf]
TEFF_LABELS = [
    "<4500",
    "4500~5000",
    "5000~5500",
    "5500~5800",
    "5800~6000",
    "6000~6200",
    "6200~6500",
    ">6500",
]

CFE_BINS = [-np.inf, -0.5, -0.2, 0.0, 0.2, 0.5, 0.7, 1.0, np.inf]
CFE_LABELS = [
    "<-0.5",
    "-0.5~-0.2",
    "-0.2~0.0",
    "0.0~0.2",
    "0.2~0.5",
    "0.5~0.7",
    "0.7~1.0",
    ">1.0",
]

SNRG_BINS = [0, 10, 20, 50, 100, np.inf]
#悬着的内容中没有小于5的，因此是5-10
SNRG_LABELS = ["5~10", "10~20", "20~50", "50~100", ">100"]

WAVELENGTH_SCOPE = [3900, 8800]
TARGET_WAVELENGTHS = np.arange(WAVELENGTH_SCOPE[0], WAVELENGTH_SCOPE[1] + 1, 1)
LEN_SPECTRUM = 4901


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-csv", type=Path, default=DEFAULT_MAPPING_CSV)
    parser.add_argument("--dr12-csv", type=Path, default=DEFAULT_DR12_CSV)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cemp-target", type=int, default=3020)
    parser.add_argument("--mp-target", type=int, default=6872)
    parser.add_argument("--other-target", type=int, default=6800)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str], source: Path | str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source} 缺少列: {missing}")


def build_fits_filename(df: pd.DataFrame) -> pd.Series:
    valid = (
        df["lmjd"].notna()
        & df["planid"].notna()
        & df["spid"].notna()
        & df["fiberid"].notna()
    )
    filename = pd.Series(pd.NA, index=df.index, dtype="string")
    filename.loc[valid] = (
        "spec-"
        + df.loc[valid, "lmjd"].astype("int64").astype(str)
        + "-"
        + df.loc[valid, "planid"].astype(str)
        + "_sp"
        + df.loc[valid, "spid"].astype("int64").astype(str).str.zfill(2)
        + "-"
        + df.loc[valid, "fiberid"].astype("int64").astype(str).str.zfill(3)
        + ".fits.gz"
    )
    return filename


def load_dr12_metadata(dr12_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(dr12_csv, usecols=lambda column: column in DR12_COLUMNS)
    require_columns(df, ["obsid", "lmjd", "planid", "spid", "fiberid"], dr12_csv)

    df["obsid"] = pd.to_numeric(df["obsid"], errors="coerce")
    df = df.dropna(subset=["obsid"]).copy()
    df["obsid"] = df["obsid"].astype("int64")

    if df["obsid"].duplicated().any():
        sort_cols = ["obsid"]
        ascending = [True]
        if "snrg" in df.columns:
            df["_snrg_sort"] = pd.to_numeric(df["snrg"], errors="coerce").fillna(-np.inf)
            sort_cols.append("_snrg_sort")
            ascending.append(False)
        df = df.sort_values(sort_cols, ascending=ascending)
        df = df.drop_duplicates("obsid", keep="first")
        if "_snrg_sort" in df.columns:
            df = df.drop(columns=["_snrg_sort"])

    df["fits_filename"] = build_fits_filename(df)
    rename_map = {
        column: f"dr12_{column}"
        for column in df.columns
        if column not in {"obsid", "fits_filename"}
    }
    return df.rename(columns=rename_map)


def find_local_fits_paths(split_dir: Path, filenames: set[str]) -> dict[str, str]:
    if not split_dir.is_dir():
        raise FileNotFoundError(f"split_all 目录不存在: {split_dir}")

    remaining = set(filenames)
    found: dict[str, str] = {}

    for shard_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        if not remaining:
            break
        with os.scandir(shard_dir) as entries:
            for entry in entries:
                if entry.name not in remaining or not entry.is_file(follow_symlinks=False):
                    continue
                found[entry.name] = entry.path
                remaining.remove(entry.name)
                if not remaining:
                    break
    return found


def attach_dr12_fits_paths(mapping_csv: Path, dr12_csv: Path, split_dir: Path) -> pd.DataFrame:
    log(f"读取 mapping: {mapping_csv}")
    mapping = pd.read_csv(mapping_csv)
    require_columns(mapping, ["obsid"], mapping_csv)
    mapping["obsid"] = pd.to_numeric(mapping["obsid"], errors="coerce")
    if mapping["obsid"].isna().any():
        raise ValueError("mapping 中存在无法转换为整数的 obsid")
    mapping["obsid"] = mapping["obsid"].astype("int64")

    log(f"读取 DR12 元数据: {dr12_csv}")
    dr12 = load_dr12_metadata(dr12_csv)

    log("按 obsid 合并 DR12 元数据")
    merged = mapping.merge(dr12, on="obsid", how="left", validate="many_to_one")

    filenames = set(merged["fits_filename"].dropna().astype(str))
    log(f"扫描 split_all 查找本地 FITS: {len(filenames)} 个候选文件名")
    path_map = find_local_fits_paths(split_dir, filenames)
    merged["fits_source_path"] = merged["fits_filename"].map(path_map)
    merged["fits_exists"] = merged["fits_source_path"].notna()
    return merged


def classify_cemp(df: pd.DataFrame) -> pd.DataFrame:
    required = ["LOGG", "TEFF", "C_FE", "FE_H", "obsid", "snrg", "z", "fits_source_path", "fits_exists"]
    require_columns(df, required, "mapping+DR12")

    work = df.copy()
    for column in ["LOGG", "TEFF", "C_FE", "FE_H", "snrg", "z"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["obsid"] = pd.to_numeric(work["obsid"], errors="coerce")
    work["fits_exists"] = work["fits_exists"].astype(bool)
    work["fits_source_path"] = work["fits_source_path"].astype("string").fillna("").str.strip()

    valid = work["fits_exists"] & work["fits_source_path"].ne("")
    valid &= work["obsid"].notna()
    valid &= work["TEFF"].gt(0)
    for column in ["LOGG", "TEFF", "C_FE", "FE_H", "snrg", "z"]:
        valid &= np.isfinite(work[column])
    work = work.loc[valid].copy()

    work["obsid"] = work["obsid"].astype("int64")
    work["_snrg_sort"] = work["snrg"].fillna(-np.inf)
    work = work.sort_values(["obsid", "_snrg_sort"], ascending=[True, False])
    work = work.drop_duplicates("obsid", keep="first").drop(columns=["_snrg_sort"])

    if "uid" in work.columns:
        uid = work["uid"].astype("string").fillna("").str.strip()
        work["group_id"] = uid.where(uid.ne(""), work["obsid"].astype(str))
    else:
        work["group_id"] = work["obsid"].astype(str)

    teff = work["TEFF"]
    logg = work["LOGG"]
    cfe = work["C_FE"]
    feh = work["FE_H"]

    log_l = np.log10(0.8) - (logg - 4.44) + 4.0 * np.log10(teff / 5780.0)
    threshold = np.where(log_l <= 2.3, 0.7, 3.0 - log_l)
    is_mp = feh < -1.0
    is_cemp = is_mp & (cfe >= threshold)

    work["logLLodot"] = log_l
    work["cemp_cfe_threshold"] = threshold
    work["is_mp"] = is_mp.astype(bool)
    work["is_cemp"] = is_cemp.astype(bool)
    work["class"] = "other"
    work.loc[is_mp & ~is_cemp, "class"] = "mp-no-cemp"
    work.loc[is_cemp, "class"] = "cemp"
    work["cfe_margin"] = work["C_FE"] - work["cemp_cfe_threshold"]
    work["hard_negative"] = (
        work["class"].eq("mp-no-cemp")
        & work["TEFF"].gt(6000)
        & work["FE_H"].lt(-1.5)
    )

    work["FE_H_bin"] = pd.cut(work["FE_H"], FEH_BINS, labels=FEH_LABELS, include_lowest=True).astype(str)
    work["TEFF_bin"] = pd.cut(work["TEFF"], TEFF_BINS, labels=TEFF_LABELS, include_lowest=True).astype(str)
    work["C_FE_bin"] = pd.cut(work["C_FE"], CFE_BINS, labels=CFE_LABELS, include_lowest=True).astype(str)
    work["snrg_bin"] = pd.cut(work["snrg"], SNRG_BINS, labels=SNRG_LABELS, include_lowest=True).astype(str)
    work["sample_stratum"] = work["FE_H_bin"] + "|" + work["C_FE_bin"]
    work["split_stratum"] = work["class"].astype(str) + "|" + work["sample_stratum"]
    return work.reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    if target <= 0:
        return df.iloc[0:0].copy()
    if len(df) <= target:
        return df.copy()

    counts = df["sample_stratum"].value_counts()
    usable = df["sample_stratum"].map(counts).fillna(0).gt(1)
    usable_df = df.loc[usable].copy()

    if len(usable_df) >= target:
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=target, random_state=seed)
        for selected_idx, _ in splitter.split(usable_df, usable_df["sample_stratum"]):
            return usable_df.iloc[selected_idx].copy()

    selected_parts = [usable_df]
    remaining_target = target - len(usable_df)
    rare_df = df.loc[~usable].copy()
    if remaining_target > 0 and not rare_df.empty:
        selected_parts.append(
            rare_df.sample(n=min(remaining_target, len(rare_df)), random_state=seed)
        )
    return pd.concat(selected_parts, ignore_index=True)


def build_selected(df: pd.DataFrame, cemp_target: int, mp_target: int, other_target: int, seed: int) -> pd.DataFrame:
    selected = pd.concat(
        [
            stratified_sample(df[df["class"].eq("cemp")].copy(), cemp_target, seed),
            stratified_sample(df[df["class"].eq("mp-no-cemp")].copy(), mp_target, seed + 1),
            stratified_sample(df[df["class"].eq("other")].copy(), other_target, seed + 2),
        ],
        ignore_index=True,
    )
    return selected.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def fallback_split_stratum(df: pd.DataFrame) -> pd.Series:
    stratum = df["split_stratum"].astype(str).copy()
    fallback_columns = [
        ["class", "FE_H_bin"],
        ["class", "C_FE_bin"],
        ["class"],
    ]

    for columns in fallback_columns:
        counts = stratum.value_counts()
        rare = stratum.map(counts).fillna(0).lt(2)
        if not rare.any():
            return stratum
        fallback = df.loc[rare, columns].astype(str).agg("|".join, axis=1)
        stratum.loc[rare] = fallback

    counts = stratum.value_counts()
    rare = stratum.map(counts).fillna(0).lt(2)
    if rare.any():
        stratum.loc[rare] = counts.idxmax()
    return stratum


def split_train_val(df: pd.DataFrame, val_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_meta = df.groupby("group_id", sort=False).agg(
        rows=("obsid", "size"),
        split_stratum=("split_stratum", lambda values: values.mode().iat[0] if not values.mode().empty else values.iloc[0]),
        class_name=("class", lambda values: values.mode().iat[0] if not values.mode().empty else values.iloc[0]),
        feh_bin=("FE_H_bin", lambda values: values.mode().iat[0] if not values.mode().empty else values.iloc[0]),
        cfe_bin=("C_FE_bin", lambda values: values.mode().iat[0] if not values.mode().empty else values.iloc[0]),
    ).reset_index()

    split_df = group_meta.rename(
        columns={
            "class_name": "class",
            "feh_bin": "FE_H_bin",
            "cfe_bin": "C_FE_bin",
        }
    )
    split_stratum = fallback_split_stratum(split_df)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    for train_group_idx, val_group_idx in splitter.split(group_meta, split_stratum):
        train_groups = set(group_meta.iloc[train_group_idx]["group_id"].astype(str))
        val_groups = set(group_meta.iloc[val_group_idx]["group_id"].astype(str))
        train_df = df[df["group_id"].astype(str).isin(train_groups)].reset_index(drop=True)
        val_df = df[df["group_id"].astype(str).isin(val_groups)].reset_index(drop=True)
        return train_df, val_df
    raise RuntimeError("无法完成 train/val 划分")


def ordered_columns(df: pd.DataFrame) -> list[str]:
    first = [column for column in LABEL_COLUMNS if column in df.columns]
    rest = [column for column in df.columns if column not in first]
    return first + rest


def process_spectrum_path(spectrum_path: Path, fallback_z: float) -> tuple[np.ndarray | None, str]:
    try:
        with fits.open(spectrum_path, memmap=False) as hdul:
            header_z = hdul[0].header.get("Z")
            try:
                z = float(header_z)
            except (TypeError, ValueError):
                z = float(fallback_z)

            if not np.isfinite(z) or z <= -1.0:
                return None, "invalid_z"

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

    unique_wave, unique_index = np.unique(wavelength_rest, return_index=True)
    wavelength_rest = unique_wave
    flux = flux[unique_index]

    target_min, target_max = WAVELENGTH_SCOPE
    if wavelength_rest.size < 2:
        return None, "too_few_unique_points"
    if np.min(wavelength_rest) > target_min or np.max(wavelength_rest) < target_max:
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
        continuous_spectrum_fit = Polynomial.fit(TARGET_WAVELENGTHS, flux_denoised, 5)
        continuous_spectrum = continuous_spectrum_fit(TARGET_WAVELENGTHS)
    except Exception:
        return None, "continuum_fit_error"

    if not np.all(np.isfinite(continuous_spectrum)):
        return None, "continuum_non_finite"
    if np.any(np.abs(continuous_spectrum) < 1e-12):
        return None, "continuum_zero"

    normalized_flux = flux_interpolated / continuous_spectrum
    if not np.all(np.isfinite(normalized_flux)):
        return None, "normalized_non_finite"

    mean_flux = float(np.mean(normalized_flux))
    std_flux = float(np.std(normalized_flux))
    if not np.isfinite(mean_flux) or not np.isfinite(std_flux) or std_flux <= 0:
        return None, "std_zero_or_non_finite"

    normalized_flux[normalized_flux > mean_flux + 3.0 * std_flux] = mean_flux
    normalized_flux[normalized_flux < mean_flux - 3.0 * std_flux] = mean_flux
    normalized_flux = (normalized_flux - mean_flux) / std_flux

    if normalized_flux.shape[0] != LEN_SPECTRUM:
        return None, "invalid_flux_length"
    if not np.all(np.isfinite(normalized_flux)):
        return None, "final_non_finite"

    return normalized_flux.astype(np.float32), "success"


def generate_npy(csv_path: Path, output_path: Path, overwrite: bool, desc: str) -> dict[str, object]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在；如需覆盖请加 --overwrite: {output_path}")

    df = pd.read_csv(csv_path)
    require_columns(df, ["fits_source_path", "z"], csv_path)

    features = np.full((len(df), LEN_SPECTRUM), np.nan, dtype=np.float32)
    error_counts: dict[str, int] = {}
    success = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        spectrum_path = Path(str(row["fits_source_path"]))
        if not spectrum_path.is_file():
            error_counts["fits_not_found"] = error_counts.get("fits_not_found", 0) + 1
            continue

        flux, status = process_spectrum_path(spectrum_path, float(row["z"]))
        if flux is None:
            error_counts[status] = error_counts.get(status, 0) + 1
            continue

        features[idx] = flux
        success += 1

    np.save(output_path, features)
    return {
        "total": int(len(df)),
        "success": int(success),
        "failed": int(len(df) - success),
        "error_counts": error_counts,
    }


def check_outputs(output_dir: Path, overwrite: bool) -> None:
    targets = [
        output_dir / "selected_all.csv",
        output_dir / "new_dataset_train_y.csv",
        output_dir / "new_dataset_val_y.csv",
        output_dir / "new_dataset_train_x.npy",
        output_dir / "new_dataset_val_x.npy",
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "输出文件已存在；如需覆盖请加 --overwrite: "
            + ", ".join(str(path) for path in existing)
        )


def main() -> None:
    args = parse_args()
    check_outputs(args.output_dir, args.overwrite)

    merged = attach_dr12_fits_paths(args.mapping_csv, args.dr12_csv, args.split_dir)
    log("清洗、去重并计算 CEMP 类别")
    pool = classify_cemp(merged)
    log(f"可用样本池: {len(pool)}")
    log(f"类别池:\n{pool['class'].value_counts()}")

    log("直接按 v6 规则采样")
    selected = build_selected(
        pool,
        cemp_target=args.cemp_target,
        mp_target=args.mp_target,
        other_target=args.other_target,
        seed=args.seed,
    )
    log(f"选择集类别:\n{selected['class'].value_counts()}")

    log("按 group_id 划分 train/val")
    train_df, val_df = split_train_val(selected, args.val_ratio, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.output_dir / "selected_all.csv"
    train_y_path = args.output_dir / "new_dataset_train_y.csv"
    val_y_path = args.output_dir / "new_dataset_val_y.csv"
    train_x_path = args.output_dir / "new_dataset_train_x.npy"
    val_x_path = args.output_dir / "new_dataset_val_x.npy"

    selected.reindex(columns=ordered_columns(selected)).to_csv(selected_path, index=False)
    train_df.reindex(columns=ordered_columns(train_df)).to_csv(train_y_path, index=False)
    val_df.reindex(columns=ordered_columns(val_df)).to_csv(val_y_path, index=False)
    log(f"写出标签: {selected_path}, {train_y_path}, {val_y_path}")

    log("生成 train npy")
    train_stats = generate_npy(train_y_path, train_x_path, args.overwrite, "train spectra")
    log(f"train npy 完成: {train_stats}")

    log("生成 val npy")
    val_stats = generate_npy(val_y_path, val_x_path, args.overwrite, "val spectra")
    log(f"val npy 完成: {val_stats}")

    train_val_obsid_overlap = len(set(train_df["obsid"]) & set(val_df["obsid"]))
    train_val_group_overlap = len(set(train_df["group_id"].astype(str)) & set(val_df["group_id"].astype(str)))
    log(
        "完成: "
        f"selected={len(selected)}, train={len(train_df)}, val={len(val_df)}, "
        f"train_val_obsid_overlap={train_val_obsid_overlap}, "
        f"train_val_group_overlap={train_val_group_overlap}"
    )


if __name__ == "__main__":
    main()
