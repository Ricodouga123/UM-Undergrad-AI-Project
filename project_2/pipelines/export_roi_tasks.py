"""
Project 2 - ROI task exporter
=============================

Creates a starter annotation sheet so students can label candidate vesicle
regions of interest. This is intentionally simple and spreadsheet-friendly.

Usage:
  python project_2/pipelines/export_roi_tasks.py
"""

from __future__ import annotations

import csv
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
ANNOTATIONS_DIR = PROJECT_DIR / "annotations"
MANIFEST_CSV = DATA_DIR / "manifest.csv"
ROI_TASKS_CSV = ANNOTATIONS_DIR / "roi_tasks.csv"
FIELDNAMES = [
    "task_id",
    "image_id",
    "split",
    "biomarker",
    "channel",
    "relative_path",
    "roi_x",
    "roi_y",
    "roi_width",
    "roi_height",
    "label",
    "notes",
]


def load_manifest_rows():
    if not MANIFEST_CSV.exists():
        raise SystemExit(
            "Manifest not found. Run `python project_2/build_manifest.py` first."
        )

    with MANIFEST_CSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_existing_rows():
    if not ROI_TASKS_CSV.exists():
        return []
    with ROI_TASKS_CSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def starter_row(manifest_row):
    return {
        "task_id": f"{manifest_row['image_id']}_000",
        "image_id": manifest_row["image_id"],
        "split": manifest_row["split"],
        "biomarker": manifest_row["biomarker"],
        "channel": manifest_row["channel"],
        "relative_path": manifest_row["relative_path"],
        "roi_x": "",
        "roi_y": "",
        "roi_width": "",
        "roi_height": "",
        "label": "",
        "notes": "",
    }


def sync_rows(manifest_rows, existing_rows):
    if not existing_rows:
        return [starter_row(row) for row in manifest_rows], len(manifest_rows), 0

    manifest_by_image = {row["image_id"]: row for row in manifest_rows}
    synced_rows = []
    updated_rows = 0

    for row in existing_rows:
        image_id = row.get("image_id", "")
        manifest_row = manifest_by_image.get(image_id)
        if manifest_row is None:
            continue

        merged = {field: row.get(field, "") for field in FIELDNAMES}
        for key in ["split", "biomarker", "channel", "relative_path"]:
            if merged.get(key, "") != manifest_row[key]:
                merged[key] = manifest_row[key]
                updated_rows += 1
        synced_rows.append(merged)

    existing_ids = {row["image_id"] for row in synced_rows}
    added_rows = 0
    for manifest_row in manifest_rows:
        if manifest_row["image_id"] not in existing_ids:
            synced_rows.append(starter_row(manifest_row))
            added_rows += 1

    return synced_rows, added_rows, updated_rows


def write_rows(rows):
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    with ROI_TASKS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def main() -> None:
    rows = load_manifest_rows()
    existing_rows = load_existing_rows()
    synced_rows, added_rows, updated_rows = sync_rows(rows, existing_rows)
    write_rows(synced_rows)

    print("=" * 60)
    print("ROI annotation sheet ready")
    print(f"Rows written : {len(synced_rows)}")
    print(f"Images found : {len(rows)}")
    print(f"New starter rows added : {added_rows}")
    print(f"Existing rows updated  : {updated_rows}")
    print(f"File : {ROI_TASKS_CSV}")
    print("Suggested labels: negative, positive, uncertain")
    print("=" * 60)


if __name__ == "__main__":
    main()
