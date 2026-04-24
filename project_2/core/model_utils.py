"""
Project 2 - Model helpers
=========================

Shared helpers for training and inference with the Project 2 patch classifier.
"""

from __future__ import annotations

import math


def build_small_cnn(nn, patch_size: int):
    feature_h = max(1, patch_size // 4)
    feature_w = max(1, patch_size // 4)

    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(16 * feature_h * feature_w, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return SmallCNN()


def infer_patch_size_from_state_dict(state_dict: dict) -> int:
    first_linear = state_dict.get("classifier.1.weight")
    if first_linear is None:
        return 64

    in_features = int(first_linear.shape[1])
    feature_area = in_features / 16.0
    feature_side = int(round(math.sqrt(feature_area)))
    if feature_side <= 0:
        return 64
    return max(4, feature_side * 4)


def load_patch_classifier(model_path, torch, nn, device):
    payload = torch.load(model_path, map_location=device)

    if isinstance(payload, dict) and "model_state" in payload:
        patch_size = int(payload.get("patch_size", 64))
        state_dict = payload["model_state"]
    else:
        state_dict = payload
        patch_size = infer_patch_size_from_state_dict(state_dict)

    model = build_small_cnn(nn, patch_size).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, patch_size
