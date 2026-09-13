#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Three-class evaluation on a given dataset using a specified LGADNet checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

# The LGADNet model class lives in the prediction script. Expose
# scripts/02_predict on sys.path and load it (numeric filename -> importlib).
import importlib

_PREDICT_DIR = Path(__file__).resolve().parents[1] / "02_predict"
if str(_PREDICT_DIR) not in sys.path:
    sys.path.insert(0, str(_PREDICT_DIR))
_predict = importlib.import_module("01_predict_lgadnet_dr12")  # name starts with a digit
LGADNet = _predict.LGADNet


CLASS_ORDER = ["other", "mp_no_cemp", "cemp"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_ORDER)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def classify_star(row: pd.Series) -> str:
    if row["Fe_H"] < -1.0:
        if row["logLLodot"] <= 2.3 and row["C_Fe"] >= 0.7:
            return "cemp"
        if row["logLLodot"] > 2.3 and row["C_Fe"] >= (3.0 - row["logLLodot"]):
            return "cemp"
        return "mp_no_cemp"
    return "other"


class CustomDataset(Dataset):
    def __init__(self, feature_path: Path, label_path: Path, normalize: bool = True):
        self.features = np.load(feature_path)
        label_df = pd.read_csv(label_path)
        self.labels = label_df[["LOGG", "TEFF", "C_FE", "FE_H"]].values.astype(np.float32)

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

    def __getitem__(self, index: int):
        feature = torch.tensor(self.features[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return feature, label

    def __len__(self) -> int:
        return len(self.features)


def compute_gmean_from_confusion(cm: np.ndarray) -> float:
    recalls = []
    for index in range(cm.shape[0]):
        denom = cm[index, :].sum()
        recalls.append((cm[index, index] / denom) if denom > 0 else 0.0)
    return float(np.prod(recalls) ** (1.0 / len(recalls)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--train-feature-path", type=Path, required=True)
    parser.add_argument("--train-label-path", type=Path, required=True)
    parser.add_argument("--test-feature-path", type=Path, required=True)
    parser.add_argument("--test-label-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    cfg = checkpoint.get("config", {})

    model = LGADNet(
        num_label=cfg.get("num_label", 4),
        len_spectrum=cfg.get("len_spectrum", 4901),
        hidden_channels=cfg.get("hidden_channels", 64),
        dropout=cfg.get("dropout", 0.2),
        nhead=cfg.get("nhead", 8),
        num_layers=cfg.get("num_layers", 2),
        dim_feedforward=cfg.get("dim_feedforward", 512),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    train_dataset = CustomDataset(args.train_feature_path, args.train_label_path, normalize=True)
    test_dataset = CustomDataset(args.test_feature_path, args.test_label_path, normalize=False)
    test_dataset.labels = (test_dataset.labels - train_dataset.label_mean) / train_dataset.label_std

    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            outputs = model(features)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds_original = all_preds * train_dataset.label_std + train_dataset.label_mean
    all_labels_original = all_labels * train_dataset.label_std + train_dataset.label_mean

    mae_original = np.mean(np.abs(all_preds_original - all_labels_original), axis=0)

    pred_df = pd.DataFrame(all_preds_original, columns=["logg", "Teff", "C_Fe", "Fe_H"])
    true_df = pd.DataFrame(all_labels_original, columns=["logg", "Teff", "C_Fe", "Fe_H"])
    pred_df["logLLodot"] = np.log10(0.8) - (pred_df["logg"] - 4.44) + 4.0 * np.log10(pred_df["Teff"] / 5780.0)
    true_df["logLLodot"] = np.log10(0.8) - (true_df["logg"] - 4.44) + 4.0 * np.log10(true_df["Teff"] / 5780.0)
    pred_df["class"] = pred_df.apply(classify_star, axis=1)
    true_df["class"] = true_df.apply(classify_star, axis=1)

    true_labels = true_df["class"].map(CLASS_TO_INDEX).to_numpy()
    pred_labels = pred_df["class"].map(CLASS_TO_INDEX).to_numpy()

    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1, 2])
    accuracy = accuracy_score(true_labels, pred_labels)
    precision_macro = precision_score(true_labels, pred_labels, average="macro", zero_division=0)
    precision_weighted = precision_score(true_labels, pred_labels, average="weighted", zero_division=0)
    recall_macro = recall_score(true_labels, pred_labels, average="macro", zero_division=0)
    recall_weighted = recall_score(true_labels, pred_labels, average="weighted", zero_division=0)
    f1_macro = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    f1_weighted = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(true_labels, pred_labels)
    gmean = compute_gmean_from_confusion(cm)

    true_onehot = np.eye(len(CLASS_ORDER))[true_labels]
    pred_onehot = np.eye(len(CLASS_ORDER))[pred_labels]
    auc_ovr_macro = roc_auc_score(true_onehot, pred_onehot, multi_class="ovr", average="macro")
    auc_ovr_weighted = roc_auc_score(true_onehot, pred_onehot, multi_class="ovr", average="weighted")

    per_class = {}
    for class_name, class_index in CLASS_TO_INDEX.items():
        y_true = (true_labels == class_index).astype(int)
        y_pred = (pred_labels == class_index).astype(int)
        per_class[class_name] = {
            "support": int(y_true.sum()),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

    metrics = {
        "checkpoint_path": str(args.checkpoint_path.resolve()),
        "mae_original": [float(x) for x in mae_original],
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "precision_weighted": float(precision_weighted),
        "recall_macro": float(recall_macro),
        "recall_weighted": float(recall_weighted),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "gmean_macro_recall": float(gmean),
        "mcc": float(mcc),
        "auc_ovr_macro": float(auc_ovr_macro),
        "auc_ovr_weighted": float(auc_ovr_weighted),
        "class_order": CLASS_ORDER,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "samples": int(len(true_labels)),
    }

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output_json:
        args.output_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
