#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LGADNet training script (integrated implementation).
Contains the complete implementation of the model, dataset, and training loop.
"""

import os
import sys
import json
import random
import logging
import datetime
import argparse
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

# =============================================================================
# CONFIGURATION
# =============================================================================
# The dataset is produced by 01_prepare_training_dataset.py; LGADNET_ROOT and
# DEFAULT_OUTPUT_DIR are shared with that script. Most values below are
# overridable with a matching command-line argument.
import importlib
_prepare = importlib.import_module("01_prepare_training_dataset")  # name starts with a digit -> needs importlib
LGADNET_ROOT = _prepare.LGADNET_ROOT
DEFAULT_OUTPUT_DIR = _prepare.DEFAULT_OUTPUT_DIR
CONFIG = {
    # Experiment name
    "experiment_name": "lgadnet_regression",

    # Data paths (dataset produced by 01_prepare_training_dataset.py)
    "train_feature_path": str(DEFAULT_OUTPUT_DIR / "train_features.npy"),
    "train_label_path": str(DEFAULT_OUTPUT_DIR / "train_labels.csv"),
    "test_feature_path": str(DEFAULT_OUTPUT_DIR / "val_features.npy"),
    "test_label_path": str(DEFAULT_OUTPUT_DIR / "val_labels.csv"),

    # Model parameters
    "num_label": 4,
    "len_spectrum": 4901,
    "hidden_channels": 64,
    "dropout": 0.2,
    "nhead": 8,
    "num_layers": 2,
    "dim_feedforward": 512,

    # Training parameters
    "batch_size": 32,
    "num_workers": 4,
    "max_epochs": 500,
    "lr": 0.0001,
    "weight_decay": 1e-5,

    # Other settings
    "seed": 2024,
    "log_to_console": False,
    "use_early_stopping": False,
    "early_stopping_patience": 20,
    "early_stopping_min_delta": 0.0001,
    "work_dir": str(LGADNET_ROOT / "outputs" / "lgadnet_regression"),

    # GPU settings
    # "cuda_device": "0",
}


# =============================================================================
# Model definition
# =============================================================================

class IDConv1dFull(nn.Module):
    """
    Dynamic convolution (no grouped/depthwise restriction):
      - Candidate kernels: [G, Cout, Cin, K]
      - Per-sample mixing weights alpha in R^G yield Wmix in [B, Cout, Cin, K]
      - Applied as a single grouped convolution with groups=B over the batch
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int | None = None,
                 num_kernels: int = 4,
                 reduction_ratio: int = 4,
                 bias: bool = True):
        super().__init__()
        assert num_kernels > 1, "num_kernels must be > 1"
        assert kernel_size >= 1, "kernel_size must be >= 1"
        if padding is None:
            padding = kernel_size // 2

        self.Cin = in_channels
        self.Cout = out_channels
        self.K = kernel_size
        self.S = stride
        self.P = padding
        self.G = num_kernels
        self.has_bias = bias

        # candidate kernels / bias
        self.weight = nn.Parameter(torch.empty(self.G, self.Cout, self.Cin, self.K))
        self.bias = nn.Parameter(torch.empty(self.G, self.Cout)) if bias else None

        # gating: per-sample G-dim logits from the GAP-pooled features
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
        B, Cin, L = x.shape
        assert Cin == self.Cin, f"Cin mismatch: expected {self.Cin}, got {Cin}"

        # per-sample G-dim mixing weights
        g_logits = self.gate(self.pool(x)).squeeze(-1)
        alpha = torch.softmax(g_logits, dim=1)

        # mix to obtain the full per-sample kernel
        Wmix = torch.einsum('bg,gock->bock', alpha, self.weight)
        Wmix = Wmix.reshape(B * self.Cout, self.Cin, self.K)

        if self.has_bias:
            bmix = torch.einsum('bg,gc->bc', alpha, self.bias).reshape(-1)
        else:
            bmix = None

        # a single grouped convolution with groups=B
        y = F.conv1d(
            x.reshape(1, B * Cin, L),
            Wmix,
            bias=bmix,
            stride=self.S,
            padding=self.P,
            groups=B
        )
        return y.view(B, self.Cout, y.shape[-1])


class ResNet1D_Block_ResDynFull(nn.Module):
    """
    Main branch: standard convolution.
    Residual branch: the 1x1 convolution is replaced by IDConv1dFull.
    """
    def __init__(self, in_channels, out_channels, stride=1,
                 num_kernels=4, reduction_ratio=4, bias=True, kernel_size=3):
        super().__init__()
        # main branch
        self.main_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=kernel_size//2, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                      stride=1, padding=kernel_size//2, bias=bias),
        )

        # residual branch
        if stride != 1 or in_channels != out_channels:
            self.residual_path = IDConv1dFull(
                in_channels, out_channels,
                kernel_size=1, stride=stride, padding=0,
                num_kernels=num_kernels, reduction_ratio=reduction_ratio, bias=bias
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
    """
    LGADNet model.
    Six ResNet1D_Block_ResDynFull stages downsample while growing channels:
    64, 64, 128, 128, 256, 256. A Transformer Encoder replaces the LSTM.
    """
    def __init__(self,
                 num_label=4,
                 len_spectrum=4901,
                 hidden_channels=64,
                 dropout=0.2,
                 nhead=8,
                 num_layers=2,
                 dim_feedforward=512):
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

        # input projection
        self.input_proj = nn.Conv1d(1, self.base_c, kernel_size=3, padding=1)

        # CNN Backbone
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
                    kernel_size=3
                )
            )
            in_c = out_c
        self.downsample_blocks = nn.Sequential(*blocks)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.final_c,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, self.final_c))

        # gating
        self.transformer_gate = nn.Sequential(
            nn.Linear(self.final_c, self.final_c),
            nn.Sigmoid()
        )

        # compute the downsampled length
        self.l_downsampled = self._compute_downsampled_length(len_spectrum, num_stages=self.num_blocks)
        flattened_dim = self.l_downsampled * self.final_c

        # regressor head
        self.regressor = nn.Sequential(
            nn.Linear(flattened_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_label)
        )

    @staticmethod
    def _compute_downsampled_length(input_len, num_stages):
        L = input_len
        for _ in range(num_stages):
            L = (L - 1) // 2 + 1
        return L

    def forward(self, x):
        B, L = x.shape
        x = x.view(B, 1, L)
        x = self.input_proj(x)
        x = self.downsample_blocks(x)

        # Transformer input
        x = x.transpose(1, 2)
        pos = self.pos_embedding[:, :x.size(1), :]
        x = x + pos

        # gating
        x = x * self.transformer_gate(x)

        # Transformer encoding
        x = self.transformer(x)

        # flatten + regress
        flat = x.reshape(B, -1)
        out = self.regressor(flat)
        return out


# =============================================================================
# Dataset definition
# =============================================================================

class CustomDataset(Dataset):
    """Custom dataset."""
    def __init__(self, feature_path, label_path, normalize=True):
        self.features = np.load(feature_path)
        label_df = pd.read_csv(label_path)
        self.labels = label_df[['LOGG', 'TEFF', 'C_FE', 'FE_H']].values.astype(np.float32)

        # drop samples containing NaN
        valid_mask = np.all(np.isfinite(self.features), axis=1) & np.all(np.isfinite(self.labels), axis=1)
        self.features = self.features[valid_mask]
        self.labels = self.labels[valid_mask]
        self.valid_indices = np.where(valid_mask)[0]
        print(f"[INFO] Dropped NaN samples: {len(valid_mask)} -> kept {len(self.valid_indices)} (removed {len(valid_mask) - len(self.valid_indices)})")

        if normalize:
            self.label_mean = np.mean(self.labels, axis=0)
            self.label_std = np.std(self.labels, axis=0)
            self.label_std[self.label_std == 0] = 1.0
            self.labels = (self.labels - self.label_mean) / self.label_std
        else:
            self.label_mean = None
            self.label_std = None

    def __getitem__(self, index):
        feature = torch.tensor(self.features[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return feature, label

    def __len__(self):
        return len(self.features)


# =============================================================================
# Loss function definition
# =============================================================================

class MSELoss(nn.Module):
    """MSE loss function."""
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, mean, std):
        total_loss = F.mse_loss(y_pred, y_true)
        class_loss = torch.tensor(0.0, device=total_loss.device, dtype=total_loss.dtype)
        reg_loss = torch.tensor(0.0, device=total_loss.device, dtype=total_loss.dtype)

        return {
            'total_loss': total_loss,
            'reg_loss': reg_loss,
            'cls_loss': class_loss,
        }


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:
    """Early stopping."""
    def __init__(self,
                 patience: int = 20,
                 min_delta: float = 0.0001,
                 save_best_model: bool = True,
                 model_save_path: Optional[str] = None,
                 verbose: bool = True,
                 logger: Optional[logging.Logger] = None):
        self.patience = patience
        self.min_delta = min_delta
        self.save_best_model = save_best_model
        self.model_save_path = model_save_path
        self.verbose = verbose
        self.logger = logger

        self.best_loss = float('inf')
        self.counter = 0
        self.best_epoch = 0
        self.best_model_state = None
        self.early_stop = False
        self.total_epochs = 0
        self.improvement_epochs = []

        self._log(f"EarlyStopping initialized: patience={patience}, min_delta={min_delta}")

    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> Dict[str, Any]:
        self.total_epochs = epoch
        is_best = False
        message = ""

        if val_loss < self.best_loss - self.min_delta:
            old_best = self.best_loss
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            is_best = True
            self.improvement_epochs.append(epoch)

            if self.save_best_model:
                self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if self.model_save_path:
                    self._save_model_to_file(model, val_loss, epoch)

            message = f"Validation loss improved from {old_best:.6f} to {val_loss:.6f}"
            self._log(message)
        else:
            self.counter += 1
            message = f"No improvement for {self.counter}/{self.patience} epochs"
            self._log(message)

        if self.counter >= self.patience:
            self.early_stop = True
            stop_message = f"Early stopping triggered! Best loss: {self.best_loss:.6f} at epoch {self.best_epoch}"
            self._log(stop_message)
            message = stop_message

        return {
            'should_stop': self.early_stop,
            'is_best': is_best,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'counter': self.counter,
            'patience': self.patience,
            'message': message,
            'total_improvements': len(self.improvement_epochs)
        }

    def load_best_model(self, model: nn.Module) -> bool:
        if self.best_model_state is not None:
            try:
                device = next(model.parameters()).device
                state_dict = {k: v.to(device) for k, v in self.best_model_state.items()}
                model.load_state_dict(state_dict)
                self._log(f"Successfully loaded best model state (loss: {self.best_loss:.6f})")
                return True
            except Exception as e:
                self._log(f"Failed to load best model state: {e}")
                return False
        return False

    def _save_model_to_file(self, model: nn.Module, val_loss: float, epoch: int):
        try:
            os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
            save_dict = {
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'epoch': epoch,
            }
            torch.save(save_dict, self.model_save_path)
            self._log(f"Model saved to {self.model_save_path}")
        except Exception as e:
            self._log(f"Failed to save model: {e}")

    def _log(self, message: str):
        if self.verbose:
            if self.logger:
                self.logger.info(f"[EarlyStopping] {message}")
            else:
                print(f"[EarlyStopping] {message}")

    def get_summary(self) -> Dict[str, Any]:
        return {
            'early_stopped': self.early_stop,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'total_epochs': self.total_epochs,
            'total_improvements': len(self.improvement_epochs),
            'improvement_epochs': self.improvement_epochs.copy(),
        }


# =============================================================================
# Helper functions
# =============================================================================

def setup_logger(work_dir, experiment_name, log_to_console=False):
    """Set up logging."""
    log_dir = os.path.join(work_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{experiment_name}_{timestamp}.log')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def setup_tensorboard(work_dir, experiment_name):
    """Set up TensorBoard."""
    tensorboard_dir = os.path.join(work_dir, 'tensorboard', experiment_name)
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    return writer


def set_seed(seed):
    """Set the random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def classify_star(row):
    """Classify a star by [Fe/H] < -1 and the criterion of Aoki et al. (2007)."""
    if row["Fe_H"] < -1.0:
        if row["logLLodot"] <= 2.3 and row["C_Fe"] >= 0.7:
            return "cemp"
        if row["logLLodot"] > 2.3 and row["C_Fe"] >= (3.0 - row["logLLodot"]):
            return "cemp"
        return "mp_no_cemp"
    return "other"


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Train the LGADNet 4-parameter regression model")
    parser.add_argument("--train-feature-path")
    parser.add_argument("--train-label-path")
    parser.add_argument("--test-feature-path")
    parser.add_argument("--test-label-path")
    parser.add_argument("--work-dir")
    parser.add_argument("--experiment-name")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--log-to-console", action="store_true")
    return parser.parse_args()


def apply_cli_overrides(cfg, args):
    override_map = {
        "train_feature_path": args.train_feature_path,
        "train_label_path": args.train_label_path,
        "test_feature_path": args.test_feature_path,
        "test_label_path": args.test_label_path,
        "work_dir": args.work_dir,
        "experiment_name": args.experiment_name,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
    }
    cfg = cfg.copy()
    for key, value in override_map.items():
        if value is not None:
            cfg[key] = value
    if args.log_to_console:
        cfg["log_to_console"] = True
    return cfg


# =============================================================================
# Main training function
# =============================================================================

def main(cfg) -> int:
    # GPU setup
    # os.environ["CUDA_VISIBLE_DEVICES"] = cfg["cuda_device"]

    # create the working directory
    os.makedirs(cfg["work_dir"], exist_ok=True)

    # initialise logging and TensorBoard
    logger = setup_logger(cfg["work_dir"], cfg["experiment_name"], cfg["log_to_console"])
    writer = setup_tensorboard(cfg["work_dir"], cfg["experiment_name"])

    # log the configuration
    logger.info("Configuration: " + "=" * 50)
    logger.info(json.dumps(cfg, indent=4, ensure_ascii=False))
    logger.info("=" * 50)

    # set the random seed
    set_seed(cfg["seed"])
    logger.info(f"Random seed: {cfg['seed']}")

    # create a deterministic random generator
    generator = torch.Generator()
    generator.manual_seed(cfg["seed"])

    # build the model
    model = LGADNet(
        num_label=cfg["num_label"],
        len_spectrum=cfg["len_spectrum"],
        hidden_channels=cfg["hidden_channels"],
        dropout=cfg["dropout"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"]
    )
    logger.info(f"Model architecture: {model}")

    # load the datasets
    train_dataset = CustomDataset(
        feature_path=cfg["train_feature_path"],
        label_path=cfg["train_label_path"],
        normalize=True
    )
    test_dataset = CustomDataset(
        feature_path=cfg["test_feature_path"],
        label_path=cfg["test_label_path"],
        normalize=False
    )

    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Test set size: {len(test_dataset)}")
    logger.info(f"Training label mean: {train_dataset.label_mean}")
    logger.info(f"Training label std: {train_dataset.label_std}")

    # standardise the test set using training-set statistics
    test_dataset.labels = (test_dataset.labels - train_dataset.label_mean) / train_dataset.label_std

    # create the DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        generator=generator
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True
    )

    logger.info(f"Training loader batches: {len(train_loader)}")
    logger.info(f"Test loader batches: {len(test_loader)}")

    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU model: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")

    # build the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )

    # loss function
    criterion = MSELoss()

    # early stopping
    early_stopping = None
    if cfg["use_early_stopping"]:
        early_stopping = EarlyStopping(
            patience=cfg["early_stopping_patience"],
            min_delta=cfg["early_stopping_min_delta"],
            save_best_model=True,
            model_save_path=os.path.join(cfg["work_dir"], f"{cfg['experiment_name']}_early_stop_best.pth"),
            verbose=True,
            logger=logger
        )
        logger.info(f"Early stopping enabled, patience={cfg['early_stopping_patience']}")

    # training loop
    max_epochs = cfg["max_epochs"]
    best_loss = float('inf')
    model_count = 0

    mean = torch.tensor(train_dataset.label_mean, device=device, dtype=torch.float32)
    std = torch.tensor(train_dataset.label_std, device=device, dtype=torch.float32)

    for epoch in range(max_epochs):
        # training phase
        model.train()
        total_loss = 0
        total_reg_loss = 0
        total_cls_loss = 0

        print(f"Epoch {epoch+1}/{max_epochs} - Training...")

        for batch_idx, batch in enumerate(train_loader):
            features, labels = batch
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(features)
            loss_dict = criterion(outputs, labels, mean, std)

            loss = loss_dict['total_loss']
            reg_loss = loss_dict['reg_loss']
            cls_loss = loss_dict['cls_loss']

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_reg_loss += reg_loss.item()
            total_cls_loss += cls_loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_reg_loss = total_reg_loss / len(train_loader)
        avg_train_cls_loss = total_cls_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{max_epochs} - Training completed.")
        print(f"  Average loss: {avg_train_loss:.4f}")

        logger.info(f"Epoch {epoch+1}/{max_epochs}, average training loss: {avg_train_loss:.4f}")
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)

        # validation phase
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []

        print(f"Epoch {epoch+1}/{max_epochs} - Validating...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                features, labels = batch
                features, labels = features.to(device), labels.to(device)

                outputs = model(features)
                loss_dict = criterion(outputs, labels, mean, std)

                total_val_loss += loss_dict['total_loss'].item()
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(test_loader)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # denormalise
        all_preds_original = all_preds * train_dataset.label_std + train_dataset.label_mean
        all_labels_original = all_labels * train_dataset.label_std + train_dataset.label_mean

        # compute MAE
        mae_original = np.mean(np.abs(all_preds_original - all_labels_original), axis=0)
        mae_normalized = np.mean(np.abs(all_preds - all_labels), axis=0)

        print(f"Epoch {epoch+1}/{max_epochs} - Validation completed.")
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        print(f"  Original MAE: {mae_original}")

        logger.info(f"Epoch {epoch+1}/{max_epochs}, validation loss: {avg_val_loss:.4f}")
        logger.info(f"Epoch {epoch+1}/{max_epochs}, raw-scale validation MAE: {mae_original}")

        # TensorBoard logging
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        for i, col in enumerate(['LOGG', 'TEFF', 'C_FE', 'FE_H']):
            writer.add_scalar(f'MAE_original/{col}', mae_original[i], epoch)
        writer.add_scalar('MAE_original/average', np.mean(mae_original), epoch)

        # save the best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_path = os.path.join(cfg["work_dir"], f"{cfg['experiment_name']}_{model_count}_best_model.pth")
            model_count += 1

            # compute classification metrics
            pred_df = pd.DataFrame(all_preds_original, columns=['logg', 'Teff', 'C_Fe', 'Fe_H'])
            true_df = pd.DataFrame(all_labels_original, columns=['logg', 'Teff', 'C_Fe', 'Fe_H'])
            pred_df["logLLodot"] = np.log10(0.8) - (pred_df["logg"] - 4.44) + 4.0 * np.log10(pred_df["Teff"] / 5780.0)
            true_df["logLLodot"] = np.log10(0.8) - (true_df["logg"] - 4.44) + 4.0 * np.log10(true_df["Teff"] / 5780.0)

            pred_df['class'] = pred_df.apply(classify_star, axis=1)
            true_df['class'] = true_df.apply(classify_star, axis=1)

            true_labels = (true_df['class'] == 'cemp').astype(int)
            pred_labels = (pred_df['class'] == 'cemp').astype(int)

            accuracy = accuracy_score(true_labels, pred_labels)
            recall = recall_score(true_labels, pred_labels, zero_division=0)
            precision = precision_score(true_labels, pred_labels, zero_division=0)
            f1 = f1_score(true_labels, pred_labels, zero_division=0)

            logger.info(f"Metrics (positive class 'cemp'):")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  F1 score: {f1:.4f}")

            torch.save({
                "model_state_dict": model.state_dict(),
                "label_mean": train_dataset.label_mean,
                "label_std": train_dataset.label_std,
                "label_names": ["LOGG", "TEFF", "C_FE", "FE_H"],
                "config": cfg,
                "val_loss": float(best_loss),
                "epoch": epoch + 1,
            }, best_model_path)
            print(f"  New best model saved! Validation loss: {best_loss:.4f}")
            logger.info(f"Saved the best model, validation loss: {best_loss:.4f}")

        # early stopping check
        if early_stopping is not None:
            early_stop_result = early_stopping(avg_val_loss, model, epoch+1)
            logger.info(early_stop_result['message'])

            if early_stop_result['should_stop']:
                print(f"Early stopping triggered at epoch {epoch+1}!")
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                if early_stopping.load_best_model(model):
                    logger.info("Restored the best model weights")
                break

        print("-" * 60)

    # end of training
    print(f"Training completed! Best validation loss: {best_loss:.4f}")
    logger.info(f"Training finished, best validation loss: {best_loss:.4f}")
    logger.info(f"Best model saved at: {best_model_path}")

    writer.close()
    print("done")

    return 0

if __name__ == "__main__":
    raise SystemExit(main(apply_cli_overrides(CONFIG, parse_cli_args())))
