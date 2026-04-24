"""
Project 2 - Dataset manifest builder
====================================

Scans the raw TIFF files, extracts metadata from filenames, and creates a
deterministic train/val/test split that keeps related images together.

Usage:
  python project_2/pipelines/build_manifest.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_DIR / "images"
DATA_DIR = PROJECT_DIR / "data"
MANIFEST_CSV = DATA_DIR / "manifest.csv"
MANIFEST_JSON = DATA_DIR / "manifest.json"

SPLIT_BUCKETS: Tuple[Tuple[str, float], ...] = (
    ("train", 0.70),
    ("val", 0.15),
    ("test", 0.15),
)


@dataclass
class ManifestRow:
    image_id: str
    biomarker: str
    channel: str
    cell_line: str
    concentration: str
    sample_group: str
    pair_key: str
    split: str
    relative_path: str
    filename: str


def infer_channel(filename: str) -> str:
    upper = filename.upper()
    if "_DF_" in upper or upper.startswith("DF"):
        return "df"
    if "_RAMAN_" in upper or upper.startswith("R"):
        return "raman"
    return "unknown"


def infer_cell_line(filename: str) -> str:
    upper = filename.upper()
    if "SKBR3" in upper:
        return "SKBR3"
    if "MM231" in upper:
        return "MM231"
    return "unknown"


def infer_concentration(filename: str) -> str:
    match = re.search(r"(\d+(?:\.\d+)?)NM", filename.upper())
    if match:
        return f"{match.group(1)}nM"
    return ""


def normalize_group_name(stem: str) -> str:
    group = stem
    group = re.sub(r"\s*\(\d+\)$", "", group)
    group = re.sub(r"_DF_", "_CHANNEL_", group, flags=re.IGNORECASE)
    group = re.sub(r"_RAMAN_", "_CHANNEL_", group, flags=re.IGNORECASE)
    group = re.sub(r"^DF", "CHANNEL", group, flags=re.IGNORECASE)
    group = re.sub(r"^R", "CHANNEL", group, flags=re.IGNORECASE)
    return group.upper()


def assign_split(group_key: str) -> str:
    digest = hashlib.md5(group_key.encode("utf-8")).hexdigest()
    bucket_value = int(digest[:8], 16) / 0xFFFFFFFF
    cumulative = 0.0
    for split_name, ratio in SPLIT_BUCKETS:
        cumulative += ratio
        if bucket_value <= cumulative:
            return split_name
    return SPLIT_BUCKETS[-1][0]


def iter_tiffs() -> Iterable[Path]:
    for biomarker_dir in sorted(IMAGES_DIR.iterdir()):
        if biomarker_dir.is_dir():
            yield from sorted(biomarker_dir.glob("*.tif"))


def build_rows() -> List[ManifestRow]:
    rows: List[ManifestRow] = []
    for path in iter_tiffs():
        biomarker = path.parent.name.lower()
        filename = path.name
        stem = path.stem
        channel = infer_channel(filename)
        cell_line = infer_cell_line(filename)
        concentration = infer_concentration(filename)
        sample_group = normalize_group_name(stem)
        pair_key = f"{biomarker}:{sample_group}"
        image_id = hashlib.md5(str(path.relative_to(PROJECT_DIR)).encode("utf-8")).hexdigest()[:12]
        split = assign_split(pair_key)

        rows.append(
            ManifestRow(
                image_id=image_id,
                biomarker=biomarker,
                channel=channel,
                cell_line=cell_line,
                concentration=concentration,
                sample_group=sample_group,
                pair_key=pair_key,
                split=split,
                relative_path=path.relative_to(PROJECT_DIR).as_posix(),
                filename=filename,
            )
        )
    return rows


def write_csv(rows: List[ManifestRow]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def summarize(rows: List[ManifestRow]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for row in rows:
        split_stats = summary.setdefault(row.split, {"total": 0, "df": 0, "raman": 0})
        split_stats["total"] += 1
        if row.channel in ("df", "raman"):
            split_stats[row.channel] += 1
    return summary


def write_json(rows: List[ManifestRow], summary: Dict[str, Dict[str, int]]) -> None:
    payload = {
        "project": "project_2",
        "num_images": len(rows),
        "splits": summary,
        "rows": [asdict(row) for row in rows],
    }
    with MANIFEST_JSON.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    rows = build_rows()
    if not rows:
        raise SystemExit(f"No TIFF files found under {IMAGES_DIR}")

    write_csv(rows)
    summary = summarize(rows)
    write_json(rows, summary)

    print("=" * 60)
    print("Project 2 manifest created")
    print(f"Images found : {len(rows)}")
    for split, stats in summary.items():
        print(
            f"{split:>5} -> total={stats['total']:3d}  "
            f"df={stats['df']:3d}  raman={stats['raman']:3d}"
        )
    print(f"CSV  : {MANIFEST_CSV}")
    print(f"JSON : {MANIFEST_JSON}")
    print("=" * 60)


if __name__ == "__main__":
    main()
