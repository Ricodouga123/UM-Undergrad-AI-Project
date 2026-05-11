"""
Project 2 - ROI analysis helpers
================================

Shared display and ROI-analysis helpers for the annotation and inference GUIs.
"""

from __future__ import annotations

from io import BytesIO


def normalize_for_display(image, np, contrast: float = 1.0):
    image = image.astype("float32")
    if image.ndim > 2:
        image = image.squeeze()

    min_val = float(image.min())
    max_val = float(image.max())
    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)
    else:
        image = image * 0.0

    image = ((image - 0.5) * float(contrast)) + 0.5
    return (image.clip(0.0, 1.0) * 255.0).astype("uint8")


def centered_box(image_shape, center_x: int, center_y: int, box_size: int):
    height, width = image_shape[:2]
    half = max(1, int(box_size) // 2)
    x0 = max(0, int(center_x) - half)
    y0 = max(0, int(center_y) - half)
    x1 = min(width, x0 + int(box_size))
    y1 = min(height, y0 + int(box_size))
    if x1 - x0 < int(box_size):
        x0 = max(0, x1 - int(box_size))
    if y1 - y0 < int(box_size):
        y0 = max(0, y1 - int(box_size))
    return x0, y0, x1, y1


def crop_box(image, box):
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1]


def compute_roi_metrics(roi, np):
    roi = roi.astype("float32")
    center_y = roi.shape[0] // 2
    center_x = roi.shape[1] // 2
    return {
        "shape": f"{roi.shape[1]}x{roi.shape[0]}",
        "mean": float(np.mean(roi)),
        "median": float(np.median(roi)),
        "std": float(np.std(roi)),
        "min": float(np.min(roi)),
        "max": float(np.max(roi)),
        "sum": float(np.sum(roi)),
        "p05": float(np.percentile(roi, 5)),
        "p95": float(np.percentile(roi, 95)),
        "center_pixel": float(roi[center_y, center_x]),
    }


def render_roi_analysis_image(roi, np):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    roi = roi.astype("float32")
    display_roi = roi - roi.min()
    if display_roi.max() > 0:
        display_roi = display_roi / display_roi.max()

    center_y = roi.shape[0] // 2
    center_x = roi.shape[1] // 2

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=120)
    axes[0].imshow(display_roi, cmap="gray")
    axes[0].set_title("ROI")
    axes[0].axis("off")

    axes[1].hist(roi.flatten(), bins=32, color="#1f77b4", alpha=0.9)
    axes[1].set_title("Intensity Histogram")
    axes[1].set_xlabel("Intensity")
    axes[1].set_ylabel("Count")

    axes[2].plot(roi[center_y, :], label="Center row", color="#d62728")
    axes[2].plot(roi[:, center_x], label="Center col", color="#2ca02c")
    axes[2].set_title("Center Profiles")
    axes[2].legend(fontsize=7)

    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer)
