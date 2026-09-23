#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LAMOST DR12 sharded-spectra LGADNet prediction + CEMP classification pipeline.

Design goals:
1. Process shards (split_all/all01, all02, ...) individually for resumability.
2. Stream-read metadata CSV and predict in mini-batches per shard to avoid
   loading millions of spectra into memory at once.
3. After each batch, apply the CEMP classification and write full predictions,
   CEMP candidates, failure records, and per-shard summaries.
4. Optionally merge shard results and generate CEMP diagnostics.

Input:  LAMOST DR12 FITS (split_all) + trained LGADNet checkpoint
Output: predictions_all.csv (all predictions)
        cemp_all.csv    (CEMP-classified subset, feeds step 02)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from astropy.io import fits
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from tqdm import tqdm


# =============================================================================
# 默认路径与常量
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
LGADNET_ROOT = Path(os.environ.get("LGADNET_ROOT", "/path/to/lgadnet_data"))
DEFAULT_CSV_PATH = LGADNET_ROOT / "dr12/lamost_dr12_cleaned.csv"
DEFAULT_SPLIT_DIR = LGADNET_ROOT / "dr12/split_all"
DEFAULT_OUTPUT_DIR = LGADNET_ROOT / "predictions"
DEFAULT_MODEL_WEIGHT_PATH = LGADNET_ROOT / "models/lgadnet_best.pth"

WAVELENGTH_SCOPE = [3900, 8800]
TARGET_WAVELENGTHS = np.arange(WAVELENGTH_SCOPE[0], WAVELENGTH_SCOPE[1] + 1, 1)
LEN_SPECTRUM = 4901

NUM_LABEL = 4
HIDDEN_CHANNELS = 64
DROPOUT = 0.2
NHEAD = 8
NUM_LAYERS = 2
DIM_FEEDFORWARD = 512

LABEL_NAMES = ["LOGG", "TEFF", "C_FE", "FE_H"]

FITS_METADATA_KEYS = {
    "obsid": "OBSID",
    "lmjd": "LMJD",
    "mjd": "MJD",
    "planid": "PLANID",
    "spid": "SPID",
    "fiberid": "FIBERID",
    "ra": "RA",
    "dec": "DEC",
    "z": "Z",
    "z_err": "Z_ERR",
    "zflag": "ZFLAG",
    "lamost_class": "CLASS",
    "lamost_subclass": "SUBCLASS",
}

META_COLUMNS = [
    "shard",
    "obsid",
    "filename",
    "source_path",
    "lmjd",
    "mjd",
    "planid",
    "spid",
    "fiberid",
    "ra",
    "dec",
    "z",
    "z_err",
    "zflag",
    "lamost_class",
    "lamost_subclass",
]
RESULT_COLUMNS = META_COLUMNS + [
    "LOGG",
    "TEFF",
    "C_FE",
    "FE_H",
    "logLLodot",
    "cemp_cfe_threshold",
    "is_mp",
    "is_cemp",
    "class",
]
FAILED_COLUMNS = META_COLUMNS + ["reason"]


# =============================================================================
# 模型定义：与训练脚本结构保持一致
# =============================================================================

class IDConv1dFull(nn.Module):
    """非深度卷积的动态卷积。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        num_kernels: int = 4,
        reduction_ratio: int = 4,
        bias: bool = True,
    ):
        super().__init__()
        assert num_kernels > 1, "num_kernels 应 > 1"
        assert kernel_size >= 1, "kernel_size 应 >= 1"
        if padding is None:
            padding = kernel_size // 2

        self.Cin = in_channels
        self.Cout = out_channels
        self.K = kernel_size
        self.S = stride
        self.P = padding
        self.G = num_kernels
        self.has_bias = bias

        self.weight = nn.Parameter(torch.empty(self.G, self.Cout, self.Cin, self.K))
        self.bias = nn.Parameter(torch.empty(self.G, self.Cout)) if bias else None

        red = max(1, in_channels // reduction_ratio)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.gate = nn.Sequential(
            nn.Conv1d(in_channels, red, kernel_size=1, bias=False),
            nn.BatchNorm1d(red),
            nn.GELU(),
            nn.Conv1d(red, self.G, kernel_size=1, bias=True),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        last = [m for m in self.gate.modules() if isinstance(m, nn.Conv1d)][-1]
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, cin, length = x.shape
        assert cin == self.Cin, f"Cin 对不上: 期望 {self.Cin}, 实际 {cin}"

        g_logits = self.gate(self.pool(x)).squeeze(-1)
        alpha = torch.softmax(g_logits, dim=1)

        wmix = torch.einsum("bg,gock->bock", alpha, self.weight)
        wmix = wmix.reshape(batch_size * self.Cout, self.Cin, self.K)

        if self.has_bias:
            bmix = torch.einsum("bg,gc->bc", alpha, self.bias).reshape(-1)
        else:
            bmix = None

        y = F.conv1d(
            x.reshape(1, batch_size * cin, length),
            wmix,
            bias=bmix,
            stride=self.S,
            padding=self.P,
            groups=batch_size,
        )
        return y.view(batch_size, self.Cout, y.shape[-1])


class ResNet1D_Block_ResDynFull(nn.Module):
    """一维残差块。"""

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        num_kernels=4,
        reduction_ratio=4,
        bias=True,
        kernel_size=3,
    ):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=bias,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=bias,
            ),
        )

        if stride != 1 or in_channels != out_channels:
            self.residual_path = IDConv1dFull(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                num_kernels=num_kernels,
                reduction_ratio=reduction_ratio,
                bias=bias,
            )
            self.residual_path2 = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
            )
        else:
            self.residual_path = nn.Identity()
            self.residual_path2 = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.main_path(x) + self.residual_path(x) + self.residual_path2(x)
        return self.relu(out)


class LGADNet(nn.Module):
    """LGADNet 回归模型。"""

    def __init__(
        self,
        num_label=4,
        len_spectrum=4901,
        hidden_channels=64,
        dropout=0.2,
        nhead=8,
        num_layers=2,
        dim_feedforward=512,
    ):
        super().__init__()

        self.len_spectrum = len_spectrum
        self.base_c = hidden_channels
        self.c_list = [
            self.base_c,
            self.base_c,
            self.base_c * 2,
            self.base_c * 2,
            self.base_c * 4,
            self.base_c * 4,
        ]
        self.num_blocks = len(self.c_list)
        self.final_c = self.c_list[-1]

        self.input_proj = nn.Conv1d(1, self.base_c, kernel_size=3, padding=1)

        blocks = []
        in_c = self.base_c
        for out_c in self.c_list:
            blocks.append(
                ResNet1D_Block_ResDynFull(
                    in_channels=in_c,
                    out_channels=out_c,
                    stride=2,
                    num_kernels=4,
                    reduction_ratio=4,
                    bias=True,
                    kernel_size=3,
                )
            )
            in_c = out_c
        self.downsample_blocks = nn.Sequential(*blocks)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.final_c,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, self.final_c))
        self.transformer_gate = nn.Sequential(nn.Linear(self.final_c, self.final_c), nn.Sigmoid())

        self.l_downsampled = self._compute_downsampled_length(len_spectrum, self.num_blocks)
        flattened_dim = self.l_downsampled * self.final_c

        self.regressor = nn.Sequential(
            nn.Linear(flattened_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_label),
        )

    @staticmethod
    def _compute_downsampled_length(input_len, num_stages):
        length = input_len
        for _ in range(num_stages):
            length = (length - 1) // 2 + 1
        return length

    def forward(self, x):
        batch_size, length = x.shape
        x = x.view(batch_size, 1, length)
        x = self.input_proj(x)
        x = self.downsample_blocks(x)

        x = x.transpose(1, 2)
        pos = self.pos_embedding[:, : x.size(1), :]
        x = x + pos
        x = x * self.transformer_gate(x)
        x = self.transformer(x)

        flat = x.reshape(batch_size, -1)
        return self.regressor(flat)


# =============================================================================
# 数据结构与通用工具
# =============================================================================

@dataclass
class ShardStats:
    shard: str
    shard_files: int = 0
    skipped_done: int = 0
    attempted: int = 0
    predicted: int = 0
    cemp: int = 0
    failed: int = 0
    limit_reached: bool = False
    started_at: str = ""
    ended_at: str = ""


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def parse_float_list(value: str) -> np.ndarray:
    try:
        parsed = np.array([float(item.strip()) for item in value.split(",")], dtype=np.float32)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"无法解析浮点数列表: {value}") from exc

    if parsed.shape[0] != NUM_LABEL:
        raise argparse.ArgumentTypeError(f"需要 {NUM_LABEL} 个数值，实际得到 {parsed.shape[0]} 个")
    return parsed


def log(message: str, log_path: Path | None = None) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def safe_value(row: pd.Series, column: str):
    if column not in row:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def append_dataframe(path: Path, df: pd.DataFrame, columns: Sequence[str]) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    expected_columns = list(columns)
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        with path.open("r", encoding="utf-8", newline="") as handle:
            existing_columns = next(csv.reader(handle), [])
        if existing_columns != expected_columns:
            raise RuntimeError(
                f"已有输出文件列结构与当前脚本不一致: {path}\n"
                "请使用 --overwrite 重跑，或指定新的 --output-dir，避免新旧结果混写。"
            )

    df.reindex(columns=expected_columns).to_csv(path, mode="a", header=write_header, index=False)


def remove_output_if_needed(paths: Iterable[Path], overwrite: bool, resume: bool) -> None:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return
    if resume:
        return
    if not overwrite:
        joined = "\n".join(str(p) for p in existing)
        raise RuntimeError(
            "输出文件已存在，为避免重复追加已停止。"
            "如需断点续跑加 --resume；如需重跑覆盖加 --overwrite。\n"
            f"{joined}"
        )
    for path in existing:
        path.unlink()


def read_csv_header(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
    if not header:
        raise RuntimeError(f"CSV 文件为空或无法读取表头: {csv_path}")
    return header


# =============================================================================
# 光谱文件发现
# =============================================================================

def discover_shards(split_dir: Path) -> list[str]:
    if not split_dir.is_dir():
        raise RuntimeError(f"分片目录不存在: {split_dir}")
    shards = [p.name for p in split_dir.iterdir() if p.is_dir() and p.name.startswith("all")]
    return sorted(shards)


def build_shard_file_map(shard_dir: Path) -> dict[str, str]:
    """构建当前分片内 filename -> full path 映射。"""
    if not shard_dir.is_dir():
        raise RuntimeError(f"分片目录不存在: {shard_dir}")

    files: dict[str, str] = {}
    with os.scandir(shard_dir) as entries:
        for entry in entries:
            if not entry.is_file(follow_symlinks=False):
                continue
            if not entry.name.endswith(".fits.gz"):
                continue
            if entry.name in files:
                raise RuntimeError(f"分片内出现重复文件名: {entry.name}")
            files[entry.name] = entry.path
    return files


def load_done_filenames(predict_path: Path, shard_files: set[str]) -> set[str]:
    if not predict_path.exists() or predict_path.stat().st_size == 0:
        return set()

    done: set[str] = set()
    for chunk in pd.read_csv(predict_path, usecols=["filename"], chunksize=200_000):
        done.update(name for name in chunk["filename"].dropna().astype(str) if name in shard_files)
    return done


# =============================================================================
# 光谱预处理与 CEMP 分类
# =============================================================================

def normalize_header_value(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, np.generic):
        return value.item()
    return value


def metadata_from_header(header, spectrum_path: str | Path) -> dict:
    meta = {
        "filename": Path(spectrum_path).name,
        "source_path": str(spectrum_path),
    }
    for output_key, header_key in FITS_METADATA_KEYS.items():
        meta[output_key] = normalize_header_value(header.get(header_key))
    return meta


def process_spectrum_file(spectrum_path: str | Path) -> tuple[np.ndarray | None, dict, str | None]:
    """读取 FITS header 中的 z 和元数据，并预处理单个光谱。"""
    meta = {
        "filename": Path(spectrum_path).name,
        "source_path": str(spectrum_path),
    }

    try:
        with fits.open(spectrum_path, memmap=False) as hdul:
            meta.update(metadata_from_header(hdul[0].header, spectrum_path))

            try:
                z = float(meta.get("z"))
            except (TypeError, ValueError):
                return None, meta, "invalid_z"

            if not np.isfinite(z) or z <= -1.0:
                return None, meta, "invalid_z"

            data = hdul[1].data
            wavelength = np.asarray(data["WAVELENGTH"], dtype=np.float64)
            flux = np.asarray(data["FLUX"], dtype=np.float64)
    except Exception:
        return None, meta, "fits_read_error"

    finite_mask = np.isfinite(wavelength) & np.isfinite(flux)
    if finite_mask.sum() < 2:
        return None, meta, "too_few_finite_points"

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
        return None, meta, "too_few_unique_points"
    if np.min(wavelength_rest) > target_min or np.max(wavelength_rest) < target_max:
        return None, meta, "wave_range_not_cover"

    mask = (wavelength_rest >= target_min) & (wavelength_rest <= target_max)
    wavelength_cut = wavelength_rest[mask]
    flux_cut = flux[mask]
    if wavelength_cut.size < 2:
        return None, meta, "too_few_points_in_scope"

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
        return None, meta, "interp_error"

    if not np.all(np.isfinite(flux_interpolated)):
        return None, meta, "interp_non_finite"

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
        return None, meta, "continuum_fit_error"

    if not np.all(np.isfinite(continuous_spectrum)):
        return None, meta, "continuum_non_finite"
    if np.any(np.abs(continuous_spectrum) < 1e-12):
        return None, meta, "continuum_zero"

    normalized_flux = flux_interpolated / continuous_spectrum
    if not np.all(np.isfinite(normalized_flux)):
        return None, meta, "normalized_non_finite"

    mean_flux = float(np.mean(normalized_flux))
    std_flux = float(np.std(normalized_flux))
    if not np.isfinite(mean_flux) or not np.isfinite(std_flux) or std_flux <= 0:
        return None, meta, "std_zero_or_non_finite"

    threshold_upper = mean_flux + 3.0 * std_flux
    threshold_lower = mean_flux - 3.0 * std_flux
    normalized_flux[normalized_flux > threshold_upper] = mean_flux
    normalized_flux[normalized_flux < threshold_lower] = mean_flux
    normalized_flux = (normalized_flux - mean_flux) / std_flux

    if normalized_flux.shape[0] != LEN_SPECTRUM:
        return None, meta, "invalid_flux_length"
    if not np.all(np.isfinite(normalized_flux)):
        return None, meta, "final_non_finite"

    return normalized_flux.astype(np.float32), meta, None


def classify_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """根据预测参数计算 logL 并分类 CEMP。"""
    df = df.copy()

    teff = pd.to_numeric(df["TEFF"], errors="coerce")
    logg = pd.to_numeric(df["LOGG"], errors="coerce")
    feh = pd.to_numeric(df["FE_H"], errors="coerce")
    cfe = pd.to_numeric(df["C_FE"], errors="coerce")

    valid = (
        np.isfinite(teff)
        & np.isfinite(logg)
        & np.isfinite(feh)
        & np.isfinite(cfe)
        & (teff > 0)
    )

    log_l = pd.Series(np.nan, index=df.index, dtype="float64")
    log_l.loc[valid] = (
        np.log10(0.8)
        - (logg.loc[valid] - 4.44)
        + 4.0 * np.log10(teff.loc[valid] / 5780.0)
    )

    threshold = pd.Series(np.nan, index=df.index, dtype="float64")
    threshold.loc[valid] = np.where(log_l.loc[valid] <= 2.3, 0.7, 3.0 - log_l.loc[valid])

    is_mp = valid & (feh < -1.0)
    is_cemp = is_mp & (cfe >= threshold)

    df["logLLodot"] = log_l
    df["cemp_cfe_threshold"] = threshold
    df["is_mp"] = is_mp.astype(bool)
    df["is_cemp"] = is_cemp.astype(bool)
    df["class"] = "invalid_prediction"
    df.loc[valid, "class"] = "other"
    df.loc[is_mp & ~is_cemp, "class"] = "mp-no-cemp"
    df.loc[is_cemp, "class"] = "cemp"
    return df


# =============================================================================
# 模型加载与预测写出
# =============================================================================

def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def normalize_state_dict(raw_state):
    if isinstance(raw_state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in raw_state and isinstance(raw_state[key], dict):
                raw_state = raw_state[key]
                break

    if not isinstance(raw_state, dict):
        raise RuntimeError("模型权重不是 state_dict 格式")

    if any(str(key).startswith("module.") for key in raw_state.keys()):
        raw_state = {str(key).removeprefix("module."): value for key, value in raw_state.items()}
    return raw_state


def load_checkpoint(model_weight_path: Path, map_location="cpu"):
    return torch.load(model_weight_path, map_location=map_location)


def load_checkpoint_label_stats(checkpoint, model_weight_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"checkpoint 不是 dict 格式: {model_weight_path}")
    if "label_mean" not in checkpoint or "label_std" not in checkpoint:
        raise RuntimeError(f"checkpoint 缺少 label_mean/label_std: {model_weight_path}")

    label_mean = np.asarray(checkpoint["label_mean"], dtype=np.float32)
    label_std = np.asarray(checkpoint["label_std"], dtype=np.float32)
    if label_mean.shape != (NUM_LABEL,) or label_std.shape != (NUM_LABEL,):
        raise RuntimeError(f"checkpoint 标签统计量维度错误: mean={label_mean.shape}, std={label_std.shape}")
    if not np.all(np.isfinite(label_mean)) or not np.all(np.isfinite(label_std)) or np.any(label_std == 0):
        raise RuntimeError("checkpoint 标签统计量包含非法值")
    return label_mean, label_std


def load_model(checkpoint, device: torch.device) -> LGADNet:
    model = LGADNet(
        num_label=NUM_LABEL,
        len_spectrum=LEN_SPECTRUM,
        hidden_channels=HIDDEN_CHANNELS,
        dropout=DROPOUT,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
    )
    model.load_state_dict(normalize_state_dict(checkpoint))
    model.to(device)
    model.eval()
    return model



def flush_predictions(
    model: LGADNet,
    device: torch.device,
    flux_buffer: list[np.ndarray],
    meta_buffer: list[dict],
    predict_path: Path,
    cemp_path: Path,
    label_mean: np.ndarray,
    label_std: np.ndarray,
) -> tuple[int, int]:
    if not flux_buffer:
        return 0, 0

    batch_array = np.stack(flux_buffer).astype(np.float32, copy=False)
    batch_tensor = torch.as_tensor(batch_array, dtype=torch.float32, device=device)

    with torch.no_grad():
        predictions = model(batch_tensor).detach().cpu().numpy()

    predictions = np.asarray(predictions, dtype=np.float32)
    label_mean = np.asarray(label_mean, dtype=np.float32)
    label_std = np.asarray(label_std, dtype=np.float32)
    if predictions.ndim != 2 or predictions.shape[1] != NUM_LABEL:
        raise RuntimeError(f"模型输出维度错误: {predictions.shape}")
    predictions = predictions * label_std.reshape(1, -1) + label_mean.reshape(1, -1)
    result_df = pd.DataFrame(meta_buffer)
    for index, label in enumerate(LABEL_NAMES):
        result_df[label] = predictions[:, index]

    result_df = classify_predictions(result_df)
    cemp_df = result_df[result_df["class"] == "cemp"].copy()

    append_dataframe(predict_path, result_df, RESULT_COLUMNS)
    append_dataframe(cemp_path, cemp_df, RESULT_COLUMNS)

    return len(result_df), len(cemp_df)


def flush_failed(failed_path: Path, failed_buffer: list[dict]) -> int:
    if not failed_buffer:
        return 0
    failed_df = pd.DataFrame(failed_buffer)
    append_dataframe(failed_path, failed_df, FAILED_COLUMNS)
    count = len(failed_buffer)
    failed_buffer.clear()
    return count


# =============================================================================
# 分片处理
# =============================================================================

def process_one_shard(
    args: argparse.Namespace,
    shard: str,
    model: LGADNet,
    device: torch.device,
    log_path: Path,
) -> ShardStats:
    shard_dir = args.split_dir / shard
    predict_path = args.output_dir / "shard_predictions" / f"predict_{shard}.csv"
    cemp_path = args.output_dir / "shard_cemp" / f"cemp_{shard}.csv"
    failed_path = args.output_dir / "shard_failed" / f"failed_{shard}.csv"
    summary_path = args.output_dir / "summary_by_shard" / f"{shard}_summary.txt"

    remove_output_if_needed([predict_path, cemp_path, failed_path, summary_path], args.overwrite, args.resume)

    stats = ShardStats(shard=shard, started_at=now())
    log(f"开始处理分片 {shard}: {shard_dir}", log_path)

    file_paths = build_shard_file_map(shard_dir)
    shard_file_names = set(file_paths.keys())
    stats.shard_files = len(file_paths)
    log(f"{shard}: 分片 FITS 文件数 {stats.shard_files:,}", log_path)

    done_filenames = load_done_filenames(predict_path, shard_file_names) if args.resume else set()
    stats.skipped_done = len(done_filenames)
    if done_filenames:
        log(f"{shard}: resume 跳过已完成 {len(done_filenames):,} 个文件", log_path)

    flux_buffer: list[np.ndarray] = []
    meta_buffer: list[dict] = []
    failed_buffer: list[dict] = []

    total_for_progress = args.limit or max(0, stats.shard_files - len(done_filenames))
    pbar = tqdm(desc=f"{shard} 处理", unit="file", total=total_for_progress)

    try:
        for filename, source_path in sorted(file_paths.items()):
            if filename in done_filenames:
                continue
            if args.limit is not None and stats.attempted >= args.limit:
                stats.limit_reached = True
                break

            stats.attempted += 1
            pbar.update(1)

            flux, meta, reason = process_spectrum_file(source_path)
            meta["shard"] = shard
            meta["filename"] = filename
            meta["source_path"] = source_path

            if flux is None:
                meta["reason"] = reason or "unknown_preprocess_error"
                failed_buffer.append(meta)
                stats.failed += 1
                if len(failed_buffer) >= args.failed_flush_size:
                    flush_failed(failed_path, failed_buffer)
                continue

            flux_buffer.append(flux)
            meta_buffer.append(meta)

            if len(flux_buffer) >= args.predict_batch_size:
                predicted, cemp = flush_predictions(
                    model,
                    device,
                    flux_buffer,
                    meta_buffer,
                    predict_path,
                    cemp_path,
                    args.label_mean,
                    args.label_std,
                )
                stats.predicted += predicted
                stats.cemp += cemp
                flux_buffer.clear()
                meta_buffer.clear()

            if stats.attempted % args.progress_interval == 0:
                log(
                    f"{shard}: attempted={stats.attempted:,}, "
                    f"predicted={stats.predicted:,}, cemp={stats.cemp:,}, failed={stats.failed:,}",
                    log_path,
                )

        if flux_buffer:
            predicted, cemp = flush_predictions(
                model,
                device,
                flux_buffer,
                meta_buffer,
                predict_path,
                cemp_path,
                args.label_mean,
                args.label_std,
            )
            stats.predicted += predicted
            stats.cemp += cemp
            flux_buffer.clear()
            meta_buffer.clear()

        flush_failed(failed_path, failed_buffer)

    finally:
        pbar.close()

    stats.ended_at = now()
    write_shard_summary(summary_path, stats, predict_path, cemp_path, failed_path)
    log(
        f"完成分片 {shard}: attempted={stats.attempted:,}, predicted={stats.predicted:,}, "
        f"cemp={stats.cemp:,}, failed={stats.failed:,}",
        log_path,
    )
    return stats


def write_shard_summary(
    summary_path: Path,
    stats: ShardStats,
    predict_path: Path,
    cemp_path: Path,
    failed_path: Path,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"shard: {stats.shard}",
        f"started_at: {stats.started_at}",
        f"ended_at: {stats.ended_at}",
        f"shard_files: {stats.shard_files:,}",
        f"skipped_done: {stats.skipped_done:,}",
        f"attempted: {stats.attempted:,}",
        f"predicted: {stats.predicted:,}",
        f"cemp: {stats.cemp:,}",
        f"failed: {stats.failed:,}",
        f"limit_reached: {stats.limit_reached}",
        f"predict_csv: {predict_path}",
        f"cemp_csv: {cemp_path}",
        f"failed_csv: {failed_path}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# 合并结果
# =============================================================================

def merge_csv_files(
    input_paths: Sequence[Path],
    output_path: Path,
    overwrite: bool,
    columns: Sequence[str],
    allow_empty: bool = False,
) -> int:
    existing_inputs = [p for p in input_paths if p.exists() and p.stat().st_size > 0]
    if not existing_inputs and not allow_empty:
        raise RuntimeError(f"没有找到可合并的输入文件: {output_path.name}")
    if output_path.exists():
        if not overwrite:
            raise RuntimeError(f"合并输出已存在，避免覆盖: {output_path}")
        output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    header_written = False

    with output_path.open("w", encoding="utf-8", newline="") as out_handle:
        if not existing_inputs and allow_empty:
            out_handle.write(",".join(columns) + "\n")
            return 0

        for input_path in existing_inputs:
            with input_path.open("r", encoding="utf-8", newline="") as in_handle:
                header = in_handle.readline()
                if not header:
                    continue
                if not header_written:
                    out_handle.write(header)
                    header_written = True
                for line in in_handle:
                    out_handle.write(line)
                    total_rows += 1

        if not header_written and allow_empty:
            out_handle.write(",".join(columns) + "\n")
    return total_rows


def predict_files_for_shards(output_dir: Path, shards: Sequence[str] | None = None) -> list[Path]:
    predict_dir = output_dir / "predict_by_shard"
    if shards:
        return [predict_dir / f"predict_{shard}.csv" for shard in shards]
    return sorted(predict_dir.glob("predict_all*.csv"))


def cemp_files_for_shards(output_dir: Path, shards: Sequence[str] | None = None) -> list[Path]:
    cemp_dir = output_dir / "cemp_by_shard"
    if shards:
        return [cemp_dir / f"cemp_{shard}.csv" for shard in shards]
    return sorted(cemp_dir.glob("cemp_all*.csv"))


def merge_outputs(args: argparse.Namespace, shards: Sequence[str] | None, log_path: Path) -> None:
    predict_output = args.output_dir / "predictions_all.csv"
    cemp_output = args.output_dir / "cemp_all.csv"

    predict_rows = merge_csv_files(
        predict_files_for_shards(args.output_dir, shards),
        predict_output,
        args.overwrite,
        RESULT_COLUMNS,
        allow_empty=False,
    )
    cemp_rows = merge_csv_files(
        cemp_files_for_shards(args.output_dir, shards),
        cemp_output,
        args.overwrite,
        RESULT_COLUMNS,
        allow_empty=True,
    )

    log(f"合并预测结果完成: {predict_output} rows={predict_rows:,}", log_path)
    log(f"合并 CEMP 结果完成: {cemp_output} rows={cemp_rows:,}", log_path)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按分片处理 LAMOST DR12 光谱，执行 LGADNet 预测和 CEMP 分类。"
    )
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH, help="兼容旧参数；当前默认直接从 FITS header 读取 z 和元数据，不依赖该 CSV")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-weight", type=Path, default=DEFAULT_MODEL_WEIGHT_PATH)
    parser.add_argument("--shard", nargs="+", dest="shards", help="指定一个或多个分片，如 all01 all02")
    parser.add_argument("--all-shards", action="store_true", help="处理 split-dir 下所有 all* 分片")
    parser.add_argument("--csv-chunksize", type=positive_int, default=50_000, help="兼容旧参数；FITS header 模式不使用")
    parser.add_argument("--predict-batch-size", type=positive_int, default=32)
    parser.add_argument("--failed-flush-size", type=positive_int, default=1_000)
    parser.add_argument("--progress-interval", type=positive_int, default=10_000)
    parser.add_argument("--limit", type=positive_int, help="每个分片最多尝试处理的光谱数，用于小样本测试")
    parser.add_argument("--device", default="auto", help="auto/cuda/cpu 或具体 torch device")
    parser.add_argument(
        "--label-mean",
        type=parse_float_list,
        help="可选覆盖 checkpoint 中的 4 维标签均值，按 LOGG,TEFF,C_FE,FE_H 顺序，逗号分隔",
    )
    parser.add_argument(
        "--label-std",
        type=parse_float_list,
        help="可选覆盖 checkpoint 中的 4 维标签标准差，按 LOGG,TEFF,C_FE,FE_H 顺序，逗号分隔",
    )
    parser.add_argument("--resume", action="store_true", help="断点续跑，跳过已写入 predict CSV 的 filename")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出文件")
    parser.add_argument("--record-unmatched-files", action="store_true", help="兼容旧参数；FITS header 模式不产生 unmatched 文件")
    parser.add_argument("--merge", action="store_true", help="处理完成后合并分片 predict/cemp CSV")
    parser.add_argument("--merge-only", action="store_true", help="只合并已有分片结果")
    return parser.parse_args()


def resolve_selected_shards(args: argparse.Namespace) -> list[str] | None:
    if args.all_shards:
        shards = discover_shards(args.split_dir)
        if not shards:
            raise RuntimeError(f"没有在分片目录中找到 all* 子目录: {args.split_dir}")
        return shards
    if args.shards:
        return args.shards
    if args.merge_only:
        return None
    raise RuntimeError("请指定 --shard all01 或 --all-shards")


def validate_inputs(args: argparse.Namespace) -> None:
    if not args.split_dir.is_dir():
        raise RuntimeError(f"分片目录不存在: {args.split_dir}")
    if not args.model_weight.is_file() and not args.merge_only:
        raise RuntimeError(f"模型权重不存在: {args.model_weight}")


def main() -> int:
    args = parse_args()
    args.csv_path = args.csv_path.resolve()
    args.split_dir = args.split_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.model_weight = args.model_weight.resolve()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "pipeline.log"

    shards = resolve_selected_shards(args)
    validate_inputs(args)

    needs_prediction = not args.merge_only
    checkpoint = None
    if needs_prediction:
        if (args.label_mean is None) != (args.label_std is None):
            raise RuntimeError("--label-mean 和 --label-std 必须同时提供")
        checkpoint = load_checkpoint(args.model_weight)
        if args.label_mean is None:
            args.label_mean, args.label_std = load_checkpoint_label_stats(checkpoint, args.model_weight)

    log("=" * 72, log_path)
    log("LAMOST DR12 LGADNet CEMP pipeline started", log_path)
    log("metadata_source=FITS header", log_path)
    log(f"split_dir={args.split_dir}", log_path)
    log(f"output_dir={args.output_dir}", log_path)
    log(f"shards={shards if shards else '已有结果'}", log_path)
    if args.label_mean is not None:
        log(f"label_mean={args.label_mean.tolist()}", log_path)
        log(f"label_std={args.label_std.tolist()}", log_path)
    else:
        log("label stats: skipped (merge-only)", log_path)

    if args.merge_only:
        merge_outputs(args, shards, log_path)
        return 0

    device = resolve_device(args.device)
    log(f"device={device}", log_path)
    log(f"loading model: {args.model_weight}", log_path)
    model = load_model(checkpoint, device)
    log("model loaded", log_path)

    all_stats: list[ShardStats] = []
    assert shards is not None
    for shard in shards:
        all_stats.append(process_one_shard(args, shard, model, device, log_path))

    write_run_summary(args.output_dir / "run_summary.txt", all_stats)

    if args.merge:
        merge_outputs(args, shards, log_path)
    log("LAMOST DR12 LGADNet CEMP pipeline completed", log_path)
    return 0


def write_run_summary(summary_path: Path, stats_list: Sequence[ShardStats]) -> None:
    total = defaultdict(int)
    for stats in stats_list:
        total["shard_files"] += stats.shard_files
        total["skipped_done"] += stats.skipped_done
        total["attempted"] += stats.attempted
        total["predicted"] += stats.predicted
        total["cemp"] += stats.cemp
        total["failed"] += stats.failed

    lines = [
        "run summary",
        f"generated_at: {now()}",
        "metadata_source: FITS header",
        "",
        "totals:",
    ]
    for key in [
        "shard_files",
        "skipped_done",
        "attempted",
        "predicted",
        "cemp",
        "failed",
    ]:
        lines.append(f"  {key}: {total[key]:,}")

    lines.extend(["", "by_shard:"])
    for stats in stats_list:
        lines.append(
            f"  {stats.shard}: attempted={stats.attempted:,}, "
            f"predicted={stats.predicted:,}, cemp={stats.cemp:,}, "
            f"failed={stats.failed:,}, limit_reached={stats.limit_reached}"
        )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
