"""
Project 2 - Patch classifier starter
====================================

Trains a simple CNN on manually labeled ROI patches. This script expects a CSV
with ROI coordinates and labels generated from export_roi_tasks.py and filled
in by students.

Expected labels:
  negative
  positive

Dependencies:
  numpy
  pandas
  tifffile
  torch

Usage:
  python project_2/train_patch_classifier.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from core.model_utils import build_small_cnn


def import_or_explain():
    try:
        import numpy as np
        import pandas as pd
        import tifffile
        import torch
        import torch.nn.functional as F
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency for Project 2 classifier. Install: "
            "numpy pandas tifffile torch"
        ) from exc
    return np, pd, tifffile, torch, F, nn, Dataset, DataLoader


PROJECT_DIR = Path(__file__).resolve().parent.parent
ANNOTATIONS_CSV = PROJECT_DIR / "annotations" / "roi_tasks.csv"
MODEL_DIR = PROJECT_DIR / "model"
DEFAULT_PATCH_SIZE = 64


class RoiDataset:
    def __init__(self, frame, root_dir, tifffile, np, torch, F, patch_size):
        self.frame = frame.reset_index(drop=True)
        self.root_dir = root_dir
        self.tifffile = tifffile
        self.np = np
        self.torch = torch
        self.F = F
        self.patch_size = patch_size
        self.label_map = {"negative": 0, "positive": 1}

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = self.tifffile.imread(self.root_dir / row["relative_path"]).astype("float32")

        x = int(row["roi_x"])
        y = int(row["roi_y"])
        w = int(row["roi_width"])
        h = int(row["roi_height"])

        patch = image[y : y + h, x : x + w]
        if patch.size == 0:
            raise ValueError(f"Empty ROI for row {index}")

        patch = patch - patch.min()
        if patch.max() > 0:
            patch = patch / patch.max()

        tensor = self.torch.tensor(
            self.np.expand_dims(patch, axis=0), dtype=self.torch.float32
        ).unsqueeze(0)
        tensor = self.F.interpolate(
            tensor,
            size=(self.patch_size, self.patch_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        label = self.label_map[row["label"]]
        return tensor, label


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np, pd, tifffile, torch, F, nn, Dataset, DataLoader = import_or_explain()

    if not ANNOTATIONS_CSV.exists():
        raise SystemExit(
            "ROI labels not found. Run `python project_2/start_annotation.py` first."
        )

    frame = pd.read_csv(ANNOTATIONS_CSV)
    frame = frame[frame["label"].isin(["negative", "positive"])].copy()
    required = ["roi_x", "roi_y", "roi_width", "roi_height"]
    frame = frame.dropna(subset=required)

    if frame.empty:
        raise SystemExit(
            "No labeled ROI rows found. Fill in roi_tasks.csv with coordinates and labels."
        )

    class TorchRoiDataset(Dataset):
        def __init__(self, base_dataset):
            self.base_dataset = base_dataset

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, index):
            tensor, label = self.base_dataset[index]
            return tensor, torch.tensor(label, dtype=torch.long)

    train_frame = frame[frame["split"] == "train"].copy()
    val_frame = frame[frame["split"] == "val"].copy()

    if train_frame.empty or val_frame.empty:
        raise SystemExit("Need labeled ROI rows in both train and val splits.")

    print("=" * 60)
    print("Project 2 patch classifier")
    print(f"Labeled ROIs : total={len(frame)} train={len(train_frame)} val={len(val_frame)}")
    print(f"Patch size   : {args.patch_size}x{args.patch_size}")
    print("=" * 60)

    train_dataset = TorchRoiDataset(
        RoiDataset(train_frame, PROJECT_DIR, tifffile, np, torch, F, args.patch_size)
    )
    val_dataset = TorchRoiDataset(
        RoiDataset(val_frame, PROJECT_DIR, tifffile, np, torch, F, args.patch_size)
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_small_cnn(nn, args.patch_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def run_epoch(loader, training):
        model.train(training)
        total = 0
        correct = 0
        loss_sum = 0.0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total += labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            loss_sum += loss.item() * labels.size(0)
        return loss_sum / total, correct / total

    best_val_acc = 0.0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, 11):
        train_loss, train_acc = run_epoch(train_loader, training=True)
        val_loss, val_acc = run_epoch(val_loader, training=False)
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "patch_size": args.patch_size,
                    "labels": {"negative": 0, "positive": 1},
                },
                MODEL_DIR / "patch_classifier.pt",
            )

    with (MODEL_DIR / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "index"])
        writer.writerow(["negative", 0])
        writer.writerow(["positive", 1])

    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {MODEL_DIR / 'patch_classifier.pt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
