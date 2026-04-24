"""
Project 2 - Classical blob-counting baseline
============================================

This baseline uses thresholding plus connected-component filtering to count
candidate spots. It is intended as a teaching baseline before students move
to supervised learning.

Dependencies:
  numpy
  pandas
  tifffile
  scikit-image

Usage:
  python project_2/baseline_count.py --split val
"""

from __future__ import annotations

import argparse
from pathlib import Path


def import_or_explain():
    try:
        import numpy as np
        import pandas as pd
        import tifffile
        from skimage import exposure, filters, measure, morphology
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency for Project 2 baseline. Install: "
            "numpy pandas tifffile scikit-image"
        ) from exc
    return np, pd, tifffile, exposure, filters, measure, morphology


PROJECT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_CSV = PROJECT_DIR / "data" / "manifest.csv"


def count_spots(
    image,
    np,
    exposure,
    filters,
    measure,
    morphology,
    min_area: int,
    bright_quantile: float,
):
    image = image.astype("float32")
    image = exposure.rescale_intensity(image, out_range=(0.0, 1.0))
    threshold = filters.threshold_otsu(image)
    binary = image > max(threshold, float(np.quantile(image, bright_quantile)))
    binary = morphology.remove_small_objects(binary, min_size=min_area)
    labeled = measure.label(binary)
    regions = measure.regionprops(labeled, intensity_image=image)
    return {
        "count": len(regions),
        "mean_intensity": float(image.mean()),
        "threshold": float(threshold),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--min-area", type=int, default=6)
    parser.add_argument("--bright-quantile", type=float, default=0.92)
    args = parser.parse_args()

    np, pd, tifffile, exposure, filters, measure, morphology = import_or_explain()

    if not MANIFEST_CSV.exists():
        raise SystemExit(
            "Manifest not found. Run `python project_2/build_manifest.py` first."
        )

    manifest = pd.read_csv(MANIFEST_CSV)
    subset = manifest[manifest["split"] == args.split].copy()
    if subset.empty:
        raise SystemExit(f"No rows found for split={args.split}")

    results = []
    for row in subset.itertuples(index=False):
        image_path = PROJECT_DIR / row.relative_path
        image = tifffile.imread(image_path)
        stats = count_spots(
            image=image,
            np=np,
            exposure=exposure,
            filters=filters,
            measure=measure,
            morphology=morphology,
            min_area=args.min_area,
            bright_quantile=args.bright_quantile,
        )
        stats.update(
            {
                "image_id": row.image_id,
                "split": row.split,
                "biomarker": row.biomarker,
                "channel": row.channel,
                "relative_path": row.relative_path,
            }
        )
        results.append(stats)

    results_df = pd.DataFrame(results)
    summary = (
        results_df.groupby(["biomarker", "channel"])["count"]
        .agg(["mean", "median", "min", "max"])
        .reset_index()
    )

    out_path = PROJECT_DIR / "data" / f"baseline_counts_{args.split}.csv"
    results_df.to_csv(out_path, index=False)

    print("=" * 60)
    print(f"Baseline count results for split={args.split}")
    print(summary.to_string(index=False))
    print(f"Saved per-image results to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
