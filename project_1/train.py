"""
Project 1 - Training Script
=============================
Fine-tunes a pretrained MobileNetV2 on the reorganized dataset.

Expected data structure:
  project_1/data/train/<person_name>/
  project_1/data/train/Unknown/
  project_1/data/val/<person_name>/
  project_1/data/val/Unknown/

Output:
  project_1/model/best_model.pth
  project_1/model/class_names.txt

Usage:
  python project_1/train.py
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR   = PROJECT_DIR / "model"
LEGACY_DATA_ROOT = PROJECT_DIR / "project_1" / "data"
EPOCHS      = 15
BATCH_SIZE  = 4
LR          = 1e-3
IMG_SIZE    = 128
NUM_WORKERS = 2
# ──────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_jpgs(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*.jpg") if path.is_file())


def resolve_data_root() -> Path:
    candidates = [PROJECT_DIR / "data", LEGACY_DATA_ROOT]
    ranked = sorted(candidates, key=count_jpgs, reverse=True)
    best = ranked[0]
    if count_jpgs(best) == 0:
        return PROJECT_DIR / "data"
    return best


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_model(num_classes):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct      += (outputs.argmax(1) == labels).sum().item()
        total        += labels.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            correct      += (outputs.argmax(1) == labels).sum().item()
            total        += labels.size(0)
    return running_loss / total, correct / total


def main():
    print("=" * 55)
    print("  Project 1 — Training (MobileNetV2)")
    print("=" * 55)
    print(f"  Device : {DEVICE}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    data_root = resolve_data_root()
    print(f"  Data   : {data_root}")

    train_tf, val_tf = get_transforms()

    train_dataset = datasets.ImageFolder(
        str(data_root / "train"), transform=train_tf)
    val_dataset   = datasets.ImageFolder(
        str(data_root / "val"),   transform=val_tf)

    class_names = train_dataset.classes
    print(f"  Classes : {class_names}")
    print(f"  Train   : {len(train_dataset)} images")
    print(f"  Val     : {len(val_dataset)} images\n")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)

    model     = build_model(num_classes=len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer)
        val_loss, val_acc     = evaluate(model, val_loader, criterion)
        scheduler.step()

        print(f"  Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       MODEL_DIR / "best_model.pth")
            print(f"    -> New best model saved (val_acc={val_acc:.3f})")

    # Save class names so the GUI can load them
    with open(MODEL_DIR / "class_names.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))

    print("\n" + "=" * 55)
    print(f"  Training complete! Best val accuracy: {best_val_acc:.3f}")
    print(f"  Model saved to: {MODEL_DIR}/best_model.pth")
    print("=" * 55)


if __name__ == "__main__":
    main()
