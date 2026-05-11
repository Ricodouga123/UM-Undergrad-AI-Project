"""
Project 2 - ROI annotation GUI
==============================
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import tkinter as tk
from tkinter import messagebox, ttk

from core.image_analysis import (
    centered_box,
    compute_roi_metrics,
    crop_box,
    normalize_for_display,
    render_roi_analysis_image,
)


def import_or_explain():
    try:
        import numpy as np
        import pandas as pd
        import tifffile
        from PIL import Image, ImageTk
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency for Project 2 annotation GUI. Install: "
            "numpy pandas tifffile pillow"
        ) from exc
    return np, pd, tifffile, Image, ImageTk


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MANIFEST_CSV = DATA_DIR / "manifest.csv"
PAIRINGS_CSV = DATA_DIR / "pairings.csv"
ANNOTATIONS_CSV = PROJECT_DIR / "annotations" / "roi_tasks.csv"
CANVAS_MAX_W = 900
CANVAS_MAX_H = 700
DEFAULT_ROI_SIZE = 32


def parse_duplicate_rank(stem: str):
    match = re.search(r"\s*\((\d+)\)$", stem)
    if match:
        return stem[: match.start()].rstrip(), int(match.group(1))
    return stem, 1


def normalize_group_name(stem: str):
    group = stem
    group = re.sub(r"_DF_", "_CHANNEL_", group, flags=re.IGNORECASE)
    group = re.sub(r"_RAMAN_", "_CHANNEL_", group, flags=re.IGNORECASE)
    group = re.sub(r"^DF", "CHANNEL", group, flags=re.IGNORECASE)
    group = re.sub(r"^R", "CHANNEL", group, flags=re.IGNORECASE)
    return group.upper()


class AnnotatorApp:
    def __init__(self, root, full_frame, manifest_frame, csv_path, np, pd, tifffile, Image, ImageTk):
        self.root = root
        self.full_frame = full_frame.copy()
        self.manifest_frame = manifest_frame.copy()
        self.csv_path = csv_path
        self.np = np
        self.pd = pd
        self.tifffile = tifffile
        self.Image = Image
        self.ImageTk = ImageTk
        self.field_order = list(self.full_frame.columns)
        self.manifest_by_id = {
            str(row["image_id"]): row for row in self.manifest_frame.to_dict("records")
        }
        self.pair_overrides = self.load_pair_overrides()
        self.pair_choice_map = {}
        self.analysis_windows = []

        self.all_image_frame = self.build_image_frame(self.manifest_frame, self.full_frame)
        self.image_frame = self.all_image_frame.copy()

        self.current_index = 0
        self.current_scale = 1.0
        self.current_photo = None
        self.current_image_raw = None
        self.display_width = 0
        self.display_height = 0
        self.drag_start = None
        self.pending_box = None
        self.pending_roi = None
        self.active_label = "positive"

        self.root.title("Project 2 - ROI Annotator")
        self.root.configure(bg="#1f2329")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.header_var = tk.StringVar(value="Loading...")
        self.status_var = tk.StringVar(
            value="Choose a label mode, drag to annotate, or right-click to analyze a 32x32 ROI."
        )
        self.label_var = tk.StringVar(value="")
        self.note_var = tk.StringVar(value="")
        self.split_var = tk.StringVar(value="all")
        self.channel_var = tk.StringVar(value="all")
        self.biomarker_var = tk.StringVar(value="all")
        self.unlabeled_only_var = tk.BooleanVar(value=False)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.roi_size_var = tk.IntVar(value=DEFAULT_ROI_SIZE)
        self.current_pair_var = tk.StringVar(value="")
        self.pair_choice_var = tk.StringVar(value="")

        tk.Label(
            root,
            text="Project 2 ROI Annotator",
            font=("Helvetica", 16, "bold"),
            fg="white",
            bg="#1f2329",
        ).pack(pady=(10, 4))

        tk.Label(
            root,
            textvariable=self.header_var,
            font=("Courier", 10),
            fg="#c9d1d9",
            bg="#1f2329",
        ).pack(pady=(0, 8))

        filter_row = tk.Frame(root, bg="#1f2329")
        filter_row.pack(fill="x", padx=12, pady=(0, 6))

        tk.Label(filter_row, text="Split", fg="white", bg="#1f2329").pack(side="left", padx=(0, 4))
        tk.OptionMenu(filter_row, self.split_var, "all", "train", "val", "test").pack(side="left", padx=4)
        tk.Label(filter_row, text="Channel", fg="white", bg="#1f2329").pack(side="left", padx=(10, 4))
        tk.OptionMenu(filter_row, self.channel_var, "all", "df", "raman").pack(side="left", padx=4)
        tk.Label(filter_row, text="Biomarker", fg="white", bg="#1f2329").pack(side="left", padx=(10, 4))
        tk.OptionMenu(filter_row, self.biomarker_var, "all", "cd44", "epcam", "her2").pack(side="left", padx=4)
        tk.Checkbutton(
            filter_row,
            text="Unlabeled Only",
            variable=self.unlabeled_only_var,
            fg="white",
            bg="#1f2329",
            selectcolor="#1f2329",
            activebackground="#1f2329",
            activeforeground="white",
        ).pack(side="left", padx=(10, 4))
        tk.Button(filter_row, text="Apply Filters", command=self.apply_filters, width=12).pack(side="left", padx=6)
        tk.Button(filter_row, text="Reset Filters", command=self.reset_filters, width=12).pack(side="left", padx=4)
        self.top_positive_button = tk.Button(
            filter_row, text="Positive [1]", command=lambda: self.set_active_label("positive"), width=12, fg="white"
        )
        self.top_positive_button.pack(side="left", padx=(12, 4))
        self.top_negative_button = tk.Button(
            filter_row, text="Negative [2]", command=lambda: self.set_active_label("negative"), width=12, fg="white"
        )
        self.top_negative_button.pack(side="left", padx=4)
        self.top_uncertain_button = tk.Button(
            filter_row, text="Uncertain [3]", command=lambda: self.set_active_label("uncertain"), width=12, fg="white"
        )
        self.top_uncertain_button.pack(side="left", padx=4)
        tk.Button(filter_row, text="Reset Image", command=self.reset_current_image_annotations, width=12).pack(side="left", padx=(12, 4))
        tk.Button(filter_row, text="Reset All", command=self.reset_all_annotations, width=12).pack(side="left", padx=4)

        display_row = tk.Frame(root, bg="#1f2329")
        display_row.pack(fill="x", padx=12, pady=(0, 8))
        tk.Label(display_row, text="Contrast", fg="white", bg="#1f2329").pack(side="left", padx=(0, 6))
        tk.Scale(
            display_row,
            from_=0.5,
            to=3.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.contrast_var,
            command=lambda _value: self.refresh_display(),
            bg="#1f2329",
            fg="white",
            highlightthickness=0,
            length=220,
        ).pack(side="left")
        tk.Label(display_row, text="Default ROI", fg="white", bg="#1f2329").pack(side="left", padx=(18, 6))
        tk.Spinbox(display_row, textvariable=self.roi_size_var, from_=8, to=256, increment=2, width=8).pack(side="left")
        tk.Label(
            display_row,
            text="Right-click analyzes a centered ROI",
            fg="#c9d1d9",
            bg="#1f2329",
            font=("Courier", 9),
        ).pack(side="left", padx=(14, 4))

        pair_row = tk.Frame(root, bg="#1f2329")
        pair_row.pack(fill="x", padx=12, pady=(0, 8))
        tk.Label(pair_row, text="Current Pair", fg="white", bg="#1f2329").pack(side="left", padx=(0, 6))
        tk.Label(pair_row, textvariable=self.current_pair_var, fg="#c9d1d9", bg="#1f2329", width=48, anchor="w").pack(side="left")
        self.pair_combo = ttk.Combobox(pair_row, textvariable=self.pair_choice_var, state="readonly", width=52)
        self.pair_combo.pack(side="left", padx=6)
        tk.Button(pair_row, text="Save Pair", command=self.save_pair_selection, width=10).pack(side="left", padx=4)
        tk.Button(pair_row, text="Go To Pair", command=self.go_to_pair, width=10).pack(side="left", padx=4)

        self.canvas = tk.Canvas(
            root,
            width=CANVAS_MAX_W,
            height=CANVAS_MAX_H,
            bg="black",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(padx=12, pady=8)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.on_analyze_click)

        nav_controls = tk.Frame(root, bg="#1f2329")
        nav_controls.pack(fill="x", padx=12, pady=(4, 6))
        tk.Button(nav_controls, text="Previous", command=self.previous_image, width=12).pack(side="left", padx=4)
        tk.Button(nav_controls, text="Next", command=self.next_image, width=12).pack(side="left", padx=4)
        tk.Button(nav_controls, text="Save CSV", command=self.save_csv, width=12).pack(side="left", padx=4)

        label_controls = tk.Frame(root, bg="#1f2329")
        label_controls.pack(fill="x", padx=12, pady=(0, 6))
        self.positive_button = tk.Button(
            label_controls, text="Positive [1]", command=lambda: self.set_active_label("positive"), width=14, fg="white"
        )
        self.positive_button.pack(side="left", padx=4)
        self.negative_button = tk.Button(
            label_controls, text="Negative [2]", command=lambda: self.set_active_label("negative"), width=14, fg="white"
        )
        self.negative_button.pack(side="left", padx=4)
        self.uncertain_button = tk.Button(
            label_controls, text="Uncertain [3]", command=lambda: self.set_active_label("uncertain"), width=14, fg="white"
        )
        self.uncertain_button.pack(side="left", padx=4)

        annotation_controls = tk.Frame(root, bg="#1f2329")
        annotation_controls.pack(fill="x", padx=12, pady=(0, 6))
        tk.Button(annotation_controls, text="Delete Last ROI", command=self.delete_last_roi, width=14).pack(side="left", padx=4)
        tk.Button(annotation_controls, text="Reset Image", command=self.reset_current_image_annotations, width=14).pack(side="left", padx=4)
        tk.Button(annotation_controls, text="Reset All", command=self.reset_all_annotations, width=14).pack(side="left", padx=4)

        notes_row = tk.Frame(root, bg="#1f2329")
        notes_row.pack(fill="x", padx=12, pady=(0, 8))
        tk.Label(notes_row, text="Notes:", fg="white", bg="#1f2329").pack(side="left", padx=(0, 6))
        self.notes_entry = tk.Entry(notes_row, textvariable=self.note_var, width=60)
        self.notes_entry.pack(side="left", fill="x", expand=True)

        tk.Label(root, textvariable=self.label_var, font=("Courier", 10), fg="#8b949e", bg="#1f2329").pack(pady=(0, 4))
        tk.Label(root, textvariable=self.status_var, font=("Courier", 10), fg="#58a6ff", bg="#1f2329").pack(pady=(0, 10))

        self.root.bind("<Left>", lambda _event: self.previous_image())
        self.root.bind("<Right>", lambda _event: self.next_image())
        self.root.bind("1", lambda _event: self.set_active_label("positive"))
        self.root.bind("2", lambda _event: self.set_active_label("negative"))
        self.root.bind("3", lambda _event: self.set_active_label("uncertain"))
        self.root.bind("s", lambda _event: self.save_csv())
        self.root.bind("<Delete>", lambda _event: self.delete_last_roi())

        self.update_label_buttons()
        self.apply_filters(initial=True)

    @staticmethod
    def build_image_frame(manifest_frame, full_frame):
        if manifest_frame is not None and not manifest_frame.empty:
            cols = [
                "image_id",
                "split",
                "biomarker",
                "channel",
                "cell_line",
                "concentration",
                "sample_group",
                "relative_path",
                "filename",
            ]
            available = [col for col in cols if col in manifest_frame.columns]
            return manifest_frame[available].drop_duplicates().reset_index(drop=True)

        cols = ["image_id", "split", "biomarker", "channel", "relative_path"]
        return full_frame[cols].drop_duplicates().reset_index(drop=True)

    def load_pair_overrides(self):
        if not PAIRINGS_CSV.exists():
            return {}
        with PAIRINGS_CSV.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        overrides = {}
        for row in rows:
            image_id = str(row.get("image_id", "")).strip()
            paired_id = str(row.get("paired_image_id", "")).strip()
            if image_id:
                overrides[image_id] = paired_id
        return overrides

    def save_pair_overrides(self):
        PAIRINGS_CSV.parent.mkdir(parents=True, exist_ok=True)
        with PAIRINGS_CSV.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_id", "paired_image_id"])
            writer.writeheader()
            for image_id in sorted(self.pair_overrides):
                writer.writerow(
                    {"image_id": image_id, "paired_image_id": self.pair_overrides.get(image_id, "")}
                )

    def effective_pair_id(self, image_id: str) -> str:
        return self.pair_overrides.get(str(image_id), "")

    def image_metadata(self, row):
        filename = str(row.get("filename", ""))
        stem = Path(filename).stem if filename else Path(str(row.get("relative_path", ""))).stem
        clean_stem, duplicate_rank = parse_duplicate_rank(stem)
        return {
            "duplicate_rank": duplicate_rank,
            "sample_group": normalize_group_name(clean_stem),
        }

    def filtered_image_frame(self):
        image_frame = self.all_image_frame.copy()
        if self.split_var.get() != "all":
            image_frame = image_frame[image_frame["split"] == self.split_var.get()]
        if self.channel_var.get() != "all":
            image_frame = image_frame[image_frame["channel"] == self.channel_var.get()]
        if self.biomarker_var.get() != "all":
            image_frame = image_frame[image_frame["biomarker"] == self.biomarker_var.get()]
        if self.unlabeled_only_var.get():
            labeled_image_ids = set(
                self.full_frame.loc[
                    self.full_frame["label"].notna()
                    & (self.full_frame["label"].astype(str).str.strip() != ""),
                    "image_id",
                ].astype(str)
            )
            image_frame = image_frame[~image_frame["image_id"].astype(str).isin(labeled_image_ids)]
        return image_frame.reset_index(drop=True)

    def apply_filters(self, initial=False):
        new_frame = self.filtered_image_frame()
        if new_frame.empty:
            self.status_var.set("No images match the current filters.")
            if not initial:
                messagebox.showinfo("No matching images", "No images match the selected filters.")
            return
        self.image_frame = new_frame
        self.current_index = 0
        self.load_current_image()
        self.status_var.set(
            f"Loaded {len(self.image_frame)} images with the current filters."
        )

    def reset_filters(self):
        self.split_var.set("all")
        self.channel_var.set("all")
        self.biomarker_var.set("all")
        self.unlabeled_only_var.set(False)
        self.apply_filters()

    def current_image_row(self):
        return self.image_frame.iloc[self.current_index]

    def current_image_id(self):
        return str(self.current_image_row()["image_id"])

    def build_blank_row(self, image_row):
        blank_row = {field: "" for field in self.field_order}
        for key in ["image_id", "split", "biomarker", "channel", "relative_path"]:
            if key in blank_row:
                blank_row[key] = image_row.get(key, "")
        blank_row["task_id"] = f"{blank_row['image_id']}_000"
        return blank_row

    def rebuild_blank_annotations(self):
        rows = [self.build_blank_row(row) for row in self.all_image_frame.to_dict("records")]
        return self.pd.DataFrame(rows, columns=self.field_order)

    def image_annotations(self):
        frame = self.full_frame[self.full_frame["image_id"].astype(str) == self.current_image_id()].copy()
        required = ["roi_x", "roi_y", "roi_width", "roi_height"]
        for field in required:
            frame = frame[frame[field].notna() & (frame[field].astype(str).str.strip() != "")]
        return frame.reset_index(drop=True)

    def update_label_buttons(self):
        button_styles = {
            "positive": ([self.positive_button, self.top_positive_button], "#1f8b4c"),
            "negative": ([self.negative_button, self.top_negative_button], "#9c2f2f"),
            "uncertain": ([self.uncertain_button, self.top_uncertain_button], "#9a6d10"),
        }
        for label, (buttons, active_color) in button_styles.items():
            for button in buttons:
                if label == self.active_label:
                    button.configure(bg=active_color, relief="sunken")
                else:
                    button.configure(bg="#4b5563", relief="raised")

    def set_active_label(self, label):
        self.active_label = label
        self.update_label_buttons()
        self.status_var.set(f"Active label mode: {label}. Drag to annotate or right-click to analyze.")

    def annotation_color(self, label):
        return {
            "positive": "#00d084",
            "negative": "#ff6b6b",
            "uncertain": "#ffd33d",
        }.get(str(label), "#58a6ff")

    def scaled_box(self, x, y, w, h):
        return (
            int(round(x * self.current_scale)),
            int(round(y * self.current_scale)),
            int(round((x + w) * self.current_scale)),
            int(round((y + h) * self.current_scale)),
        )

    def current_pair_display(self):
        pair_id = self.effective_pair_id(self.current_image_id())
        if not pair_id:
            return "(unpaired)"
        pair_row = self.manifest_by_id.get(pair_id)
        if not pair_row:
            return f"{pair_id} (missing)"
        return f"{pair_row.get('channel', '?')} -> {pair_row.get('relative_path', pair_id)}"

    def candidate_rows_for_current(self):
        row = self.current_image_row()
        candidates = self.manifest_frame[
            self.manifest_frame["image_id"].astype(str) != str(row["image_id"])
        ].copy()

        current_meta = self.image_metadata(row)

        def score(candidate):
            value = 0
            candidate_meta = self.image_metadata(candidate)
            if str(candidate.get("channel", "")) != str(row.get("channel", "")):
                value += 20
            if str(candidate.get("biomarker", "")) == str(row.get("biomarker", "")):
                value += 10
            if str(candidate.get("cell_line", "")) == str(row.get("cell_line", "")):
                value += 5
            if str(candidate.get("concentration", "")) == str(row.get("concentration", "")):
                value += 5
            if candidate_meta["sample_group"] == current_meta["sample_group"]:
                value += 100
            if candidate_meta["duplicate_rank"] == current_meta["duplicate_rank"]:
                value += 10
            if str(candidate.get("split", "")) == str(row.get("split", "")):
                value += 3
            return (-value, str(candidate.get("relative_path", "")))

        records = sorted(candidates.to_dict("records"), key=score)
        return records

    def refresh_pair_controls(self):
        self.current_pair_var.set(self.current_pair_display())
        current_pair_id = self.effective_pair_id(self.current_image_id())
        self.pair_choice_map = {"(unpaired)": ""}
        values = ["(unpaired)"]
        selected_value = "(unpaired)"

        for candidate in self.candidate_rows_for_current():
            candidate_id = str(candidate["image_id"])
            label = f"{candidate.get('channel', '?')} | {candidate.get('relative_path', candidate_id)}"
            self.pair_choice_map[label] = candidate_id
            values.append(label)
            if candidate_id == current_pair_id:
                selected_value = label

        self.pair_combo["values"] = values
        self.pair_choice_var.set(selected_value)

    def redraw_canvas(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_photo)
        for row in self.image_annotations().itertuples(index=False):
            x1, y1, x2, y2 = self.scaled_box(
                int(row.roi_x), int(row.roi_y), int(row.roi_width), int(row.roi_height)
            )
            self.canvas.create_rectangle(
                x1, y1, x2, y2, outline=self.annotation_color(getattr(row, "label", "")), width=2
            )
        if self.pending_box:
            self.canvas.create_rectangle(*self.pending_box, outline="#58a6ff", width=2)

    def refresh_display(self):
        if self.current_image_raw is None:
            return
        display = normalize_for_display(self.current_image_raw, self.np, self.contrast_var.get())
        pil_image = self.Image.fromarray(display)
        scale = min(CANVAS_MAX_W / pil_image.width, CANVAS_MAX_H / pil_image.height, 1.0)
        self.current_scale = scale
        self.display_width = max(1, int(round(pil_image.width * scale)))
        self.display_height = max(1, int(round(pil_image.height * scale)))
        pil_image = pil_image.resize((self.display_width, self.display_height))
        self.current_photo = self.ImageTk.PhotoImage(pil_image)
        self.canvas.config(width=self.display_width, height=self.display_height)
        self.redraw_canvas()

    def load_current_image(self):
        row = self.current_image_row()
        image_path = PROJECT_DIR / row["relative_path"]
        self.current_image_raw = self.tifffile.imread(image_path).astype("float32")
        self.refresh_display()

        annotations = self.image_annotations()
        self.header_var.set(
            f"[{self.current_index + 1}/{len(self.image_frame)}] "
            f"{row['biomarker']} | {row['channel']} | {row['split']} | {row['relative_path']}"
        )
        self.label_var.set(
            f"Saved ROIs on this image: {len(annotations)}    "
            f"Active mode: {self.active_label}    Default ROI: {self.roi_size_var.get()}x{self.roi_size_var.get()}"
        )
        self.pending_box = None
        self.pending_roi = None
        self.note_var.set("")
        self.refresh_pair_controls()

    def canvas_to_image_coords(self, x, y):
        return int(round(x / self.current_scale)), int(round(y / self.current_scale))

    def on_press(self, event):
        self.drag_start = (event.x, event.y)
        self.pending_box = None
        self.pending_roi = None
        self.redraw_canvas()

    def on_drag(self, event):
        if not self.drag_start:
            return
        x0, y0 = self.drag_start
        self.pending_box = (
            max(0, min(x0, event.x)),
            max(0, min(y0, event.y)),
            min(self.display_width, max(x0, event.x)),
            min(self.display_height, max(y0, event.y)),
        )
        self.redraw_canvas()

    def on_release(self, event):
        if not self.drag_start:
            return

        x0, y0 = self.drag_start
        x1 = max(0, min(x0, event.x))
        y1 = max(0, min(y0, event.y))
        x2 = min(self.display_width, max(x0, event.x))
        y2 = min(self.display_height, max(y0, event.y))
        self.drag_start = None

        ix1, iy1 = self.canvas_to_image_coords(x1, y1)
        ix2, iy2 = self.canvas_to_image_coords(x2, y2)
        if abs(x2 - x1) < 4 or abs(y2 - y1) < 4:
            box = centered_box(self.current_image_raw.shape, ix1, iy1, int(self.roi_size_var.get()))
        else:
            box = (min(ix1, ix2), min(iy1, iy2), max(ix1, ix2), max(iy1, iy2))

        x0i, y0i, x1i, y1i = box
        self.pending_roi = {
            "roi_x": x0i,
            "roi_y": y0i,
            "roi_width": max(1, x1i - x0i),
            "roi_height": max(1, y1i - y0i),
        }
        self.apply_label(self.active_label)

    def build_annotation_row(self, image_row, label):
        new_row = {field: "" for field in self.field_order}
        for key in ["image_id", "split", "biomarker", "channel", "relative_path"]:
            if key in new_row:
                new_row[key] = image_row.get(key, "")
        new_row["task_id"] = self.next_task_id(str(image_row["image_id"]))
        new_row["roi_x"] = self.pending_roi["roi_x"]
        new_row["roi_y"] = self.pending_roi["roi_y"]
        new_row["roi_width"] = self.pending_roi["roi_width"]
        new_row["roi_height"] = self.pending_roi["roi_height"]
        new_row["label"] = label
        if "notes" in new_row:
            new_row["notes"] = self.note_var.get().strip()
        return new_row

    def next_task_id(self, image_id):
        prefix = f"{image_id}_"
        existing = self.full_frame["task_id"].fillna("").astype(str)
        suffixes = []
        for value in existing:
            if value.startswith(prefix):
                tail = value[len(prefix):]
                if tail.isdigit():
                    suffixes.append(int(tail))
        next_value = max(suffixes, default=-1) + 1
        return f"{image_id}_{next_value:03d}"

    def apply_label(self, label):
        if not self.pending_roi:
            self.status_var.set("Draw a new box to add an ROI.")
            return
        image_row = self.current_image_row().to_dict()
        new_row = self.build_annotation_row(image_row, label)
        self.full_frame = self.pd.concat([self.full_frame, self.pd.DataFrame([new_row])], ignore_index=True)
        self.status_var.set(
            f"Added new {label} ROI. Drag again to add more, or right-click to analyze."
        )
        self.load_current_image()

    def save_pair_selection(self):
        selected = self.pair_choice_var.get().strip()
        if selected not in self.pair_choice_map:
            self.status_var.set("Choose a valid pair from the dropdown first.")
            return

        current_id = self.current_image_id()
        chosen_id = self.pair_choice_map[selected]
        previous_pair = self.pair_overrides.get(current_id, "")
        if previous_pair:
            self.pair_overrides.pop(previous_pair, None)

        if chosen_id:
            previous_other = self.pair_overrides.get(chosen_id, "")
            if previous_other:
                self.pair_overrides.pop(previous_other, None)
            self.pair_overrides[current_id] = chosen_id
            self.pair_overrides[chosen_id] = current_id
        else:
            self.pair_overrides.pop(current_id, None)
        self.save_pair_overrides()
        self.refresh_pair_controls()
        self.status_var.set("Pair override saved.")

    def go_to_pair(self):
        pair_id = self.effective_pair_id(self.current_image_id())
        if not pair_id:
            self.status_var.set("This image is currently unpaired.")
            return
        matches = self.image_frame.index[self.image_frame["image_id"].astype(str) == pair_id].tolist()
        if matches:
            self.current_index = matches[0]
            self.load_current_image()
            return
        all_matches = self.all_image_frame.index[self.all_image_frame["image_id"].astype(str) == pair_id].tolist()
        if all_matches:
            self.image_frame = self.all_image_frame.copy()
            self.current_index = all_matches[0]
            self.load_current_image()
            self.status_var.set("Opened paired image outside the current filter.")
            return
        self.status_var.set("Paired image could not be found in the catalog.")

    def show_roi_analysis(self, center_x: int, center_y: int):
        box = centered_box(self.current_image_raw.shape, center_x, center_y, int(self.roi_size_var.get()))
        roi = crop_box(self.current_image_raw, box)
        metrics = compute_roi_metrics(roi, self.np)
        plot_image = render_roi_analysis_image(roi, self.np)

        window = tk.Toplevel(self.root)
        window.title("ROI Analyzer")
        window.configure(bg="#1f2329")
        self.analysis_windows.append(window)

        x0, y0, x1, y1 = box
        tk.Label(
            window,
            text=(
                f"ROI box: x={x0}, y={y0}, w={x1 - x0}, h={y1 - y0}\n"
                "Note: Hounsfield Units are not relevant here; these are raw microscopy intensities."
            ),
            fg="white",
            bg="#1f2329",
            justify="left",
        ).pack(anchor="w", padx=10, pady=(10, 6))

        metrics_text = "\n".join(
            [
                f"shape={metrics['shape']}",
                f"mean={metrics['mean']:.3f}",
                f"median={metrics['median']:.3f}",
                f"std={metrics['std']:.3f}",
                f"min={metrics['min']:.3f}",
                f"max={metrics['max']:.3f}",
                f"sum={metrics['sum']:.3f}",
                f"p05={metrics['p05']:.3f}",
                f"p95={metrics['p95']:.3f}",
                f"center_pixel={metrics['center_pixel']:.3f}",
            ]
        )
        tk.Label(window, text=metrics_text, fg="#c9d1d9", bg="#1f2329", justify="left", font=("Courier", 10)).pack(
            anchor="w", padx=10, pady=(0, 8)
        )

        plot_photo = self.ImageTk.PhotoImage(plot_image)
        label = tk.Label(window, image=plot_photo, bg="#1f2329")
        label.image = plot_photo
        label.pack(padx=10, pady=(0, 10))

    def on_analyze_click(self, event):
        if self.current_image_raw is None:
            return
        x, y = self.canvas_to_image_coords(event.x, event.y)
        self.show_roi_analysis(x, y)
        self.status_var.set("Opened ROI analyzer for the clicked location.")

    def delete_last_roi(self):
        annotations = self.image_annotations()
        if annotations.empty:
            self.status_var.set("No saved ROI to delete on this image.")
            return
        last_task_id = str(annotations.iloc[-1]["task_id"])
        self.full_frame = self.full_frame[self.full_frame["task_id"] != last_task_id].reset_index(drop=True)
        self.status_var.set(f"Deleted ROI {last_task_id}.")
        self.load_current_image()

    def reset_current_image_annotations(self):
        image_row = self.current_image_row().to_dict()
        image_id = str(image_row["image_id"])
        if not messagebox.askyesno(
            "Reset current image",
            "Remove all saved ROIs for this image and restore its blank starter row?",
        ):
            return
        kept = self.full_frame[self.full_frame["image_id"].astype(str) != image_id].copy()
        restored = self.pd.DataFrame([self.build_blank_row(image_row)], columns=self.field_order)
        self.full_frame = self.pd.concat([kept, restored], ignore_index=True)
        self.pending_box = None
        self.pending_roi = None
        self.note_var.set("")
        self.status_var.set("Cleared all annotations on the current image.")
        self.load_current_image()

    def reset_all_annotations(self):
        if not messagebox.askyesno(
            "Reset all annotations",
            "This will remove all saved ROIs and pair assignments for every image. Continue?",
        ):
            return
        self.full_frame = self.rebuild_blank_annotations()
        self.pair_overrides = {}
        self.pending_box = None
        self.pending_roi = None
        self.note_var.set("")
        self.save_csv()
        self.apply_filters(initial=True)
        self.status_var.set("Reset all annotations and pair assignments.")

    def save_csv(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.full_frame.to_csv(self.csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
        self.save_pair_overrides()
        self.status_var.set(f"Saved annotations to {self.csv_path}")

    def previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()

    def next_image(self):
        if self.current_index < len(self.image_frame) - 1:
            self.current_index += 1
            self.load_current_image()

    def on_close(self):
        if messagebox.askyesno("Exit annotator", "Save annotations before closing?"):
            self.save_csv()
        self.root.destroy()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(ANNOTATIONS_CSV))
    return parser.parse_args()


def load_manifest_frame(pd):
    if MANIFEST_CSV.exists():
        return pd.read_csv(MANIFEST_CSV)
    return pd.DataFrame()


def main():
    args = parse_args()
    np, pd, tifffile, Image, ImageTk = import_or_explain()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(
            f"Annotation CSV not found: {csv_path}\n"
            "Run `python project_2/start_annotation.py` first."
        )

    full_frame = pd.read_csv(csv_path)
    manifest_frame = load_manifest_frame(pd)

    root = tk.Tk()
    AnnotatorApp(root, full_frame, manifest_frame, csv_path, np, pd, tifffile, Image, ImageTk)
    root.mainloop()


if __name__ == "__main__":
    main()
