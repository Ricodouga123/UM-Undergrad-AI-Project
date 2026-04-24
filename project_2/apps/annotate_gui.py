"""
Project 2 - ROI annotation GUI
==============================

Lightweight labeling tool for project_2/annotations/roi_tasks.csv.

This GUI is image-centric: each image can have many saved ROIs, and each ROI
is stored as its own row in the CSV.

Usage:
  python project_2/annotate_gui.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import tkinter as tk
from tkinter import messagebox


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
ANNOTATIONS_CSV = PROJECT_DIR / "annotations" / "roi_tasks.csv"
CANVAS_MAX_W = 900
CANVAS_MAX_H = 700


class AnnotatorApp:
    def __init__(self, root, full_frame, csv_path, np, pd, tifffile, Image, ImageTk):
        self.root = root
        self.full_frame = full_frame.copy()
        self.csv_path = csv_path
        self.np = np
        self.pd = pd
        self.tifffile = tifffile
        self.Image = Image
        self.ImageTk = ImageTk

        self.all_image_frame = self.build_image_frame(self.full_frame)
        self.image_frame = self.all_image_frame.copy()

        self.current_index = 0
        self.current_scale = 1.0
        self.current_photo = None
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
        self.status_var = tk.StringVar(value="Choose a label mode, then drag boxes to add ROIs.")
        self.label_var = tk.StringVar(value="")
        self.note_var = tk.StringVar(value="")
        self.split_var = tk.StringVar(value="all")
        self.channel_var = tk.StringVar(value="all")
        self.biomarker_var = tk.StringVar(value="all")
        self.unlabeled_only_var = tk.BooleanVar(value=False)

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
        filter_row.pack(fill="x", padx=12, pady=(0, 8))

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
        tk.Button(filter_row, text="Apply Filters", command=self.apply_filters, width=12).pack(
            side="left", padx=6
        )
        tk.Button(filter_row, text="Reset", command=self.reset_filters, width=8).pack(side="left", padx=4)

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

        controls = tk.Frame(root, bg="#1f2329")
        controls.pack(fill="x", padx=12, pady=(4, 8))

        tk.Button(controls, text="Previous", command=self.previous_image, width=12).pack(
            side="left", padx=4
        )
        tk.Button(controls, text="Next", command=self.next_image, width=12).pack(
            side="left", padx=4
        )
        tk.Button(controls, text="Clear Draft", command=self.clear_draft, width=12).pack(
            side="left", padx=4
        )
        tk.Button(controls, text="Delete Last ROI", command=self.delete_last_roi, width=12).pack(
            side="left", padx=4
        )
        tk.Button(controls, text="Save CSV", command=self.save_csv, width=12).pack(
            side="left", padx=4
        )

        label_controls = tk.Frame(root, bg="#1f2329")
        label_controls.pack(fill="x", padx=12, pady=(0, 6))

        self.positive_button = tk.Button(
            label_controls,
            text="Positive [1]",
            command=lambda: self.set_active_label("positive"),
            width=14,
            fg="white",
        )
        self.positive_button.pack(side="left", padx=4)
        self.negative_button = tk.Button(
            label_controls,
            text="Negative [2]",
            command=lambda: self.set_active_label("negative"),
            width=14,
            fg="white",
        )
        self.negative_button.pack(side="left", padx=4)
        self.uncertain_button = tk.Button(
            label_controls,
            text="Uncertain [3]",
            command=lambda: self.set_active_label("uncertain"),
            width=14,
            fg="white",
        )
        self.uncertain_button.pack(side="left", padx=4)

        notes_row = tk.Frame(root, bg="#1f2329")
        notes_row.pack(fill="x", padx=12, pady=(0, 8))

        tk.Label(notes_row, text="Notes:", fg="white", bg="#1f2329").pack(side="left", padx=(0, 6))
        self.notes_entry = tk.Entry(notes_row, textvariable=self.note_var, width=60)
        self.notes_entry.pack(side="left", fill="x", expand=True)

        tk.Label(
            root,
            textvariable=self.label_var,
            font=("Courier", 10),
            fg="#8b949e",
            bg="#1f2329",
        ).pack(pady=(0, 4))

        tk.Label(
            root,
            textvariable=self.status_var,
            font=("Courier", 10),
            fg="#58a6ff",
            bg="#1f2329",
        ).pack(pady=(0, 10))

        self.root.bind("<Left>", lambda event: self.previous_image())
        self.root.bind("<Right>", lambda event: self.next_image())
        self.root.bind("1", lambda event: self.set_active_label("positive"))
        self.root.bind("2", lambda event: self.set_active_label("negative"))
        self.root.bind("3", lambda event: self.set_active_label("uncertain"))
        self.root.bind("s", lambda event: self.save_csv())
        self.root.bind("c", lambda event: self.clear_draft())
        self.root.bind("<Delete>", lambda event: self.delete_last_roi())

        self.update_label_buttons()
        self.apply_filters(initial=True)

    @staticmethod
    def build_image_frame(full_frame):
        cols = ["image_id", "split", "biomarker", "channel", "relative_path"]
        return full_frame[cols].drop_duplicates().reset_index(drop=True)

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

    def image_annotations(self):
        frame = self.full_frame[self.full_frame["image_id"] == self.current_image_id()].copy()
        required = ["roi_x", "roi_y", "roi_width", "roi_height"]
        for field in required:
            frame = frame[frame[field].notna() & (frame[field].astype(str).str.strip() != "")]
        return frame.reset_index(drop=True)

    def normalize_for_display(self, image):
        image = image.astype("float32")
        if image.ndim > 2:
            image = image.squeeze()
        min_val = float(image.min())
        max_val = float(image.max())
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        else:
            image = image * 0.0
        return (image * 255.0).clip(0, 255).astype("uint8")

    def scaled_box(self, x, y, w, h):
        return (
            int(round(x * self.current_scale)),
            int(round(y * self.current_scale)),
            int(round((x + w) * self.current_scale)),
            int(round((y + h) * self.current_scale)),
        )

    def annotation_color(self, label):
        return {
            "positive": "#00d084",
            "negative": "#ff6b6b",
            "uncertain": "#ffd33d",
        }.get(str(label), "#58a6ff")

    def update_label_buttons(self):
        button_styles = {
            "positive": (self.positive_button, "#1f8b4c"),
            "negative": (self.negative_button, "#9c2f2f"),
            "uncertain": (self.uncertain_button, "#9a6d10"),
        }
        for label, (button, active_color) in button_styles.items():
            if label == self.active_label:
                button.configure(bg=active_color, relief="sunken")
            else:
                button.configure(bg="#4b5563", relief="raised")

    def set_active_label(self, label):
        self.active_label = label
        self.update_label_buttons()
        self.status_var.set(f"Active label mode: {label}. Drag boxes to add ROIs.")

    def redraw_canvas(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_photo)

        for row in self.image_annotations().itertuples(index=False):
            x1, y1, x2, y2 = self.scaled_box(
                int(row.roi_x), int(row.roi_y), int(row.roi_width), int(row.roi_height)
            )
            color = self.annotation_color(getattr(row, "label", ""))
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)

        if self.pending_box:
            self.canvas.create_rectangle(*self.pending_box, outline="#58a6ff", width=2)

    def load_current_image(self):
        row = self.current_image_row()
        image_path = PROJECT_DIR / row["relative_path"]
        image = self.tifffile.imread(image_path)
        display = self.normalize_for_display(image)
        pil_image = self.Image.fromarray(display)

        scale = min(CANVAS_MAX_W / pil_image.width, CANVAS_MAX_H / pil_image.height, 1.0)
        self.current_scale = scale
        self.display_width = max(1, int(round(pil_image.width * scale)))
        self.display_height = max(1, int(round(pil_image.height * scale)))
        pil_image = pil_image.resize((self.display_width, self.display_height))

        self.current_photo = self.ImageTk.PhotoImage(pil_image)
        self.canvas.config(width=self.display_width, height=self.display_height)

        annotations = self.image_annotations()
        self.header_var.set(
            f"[{self.current_index + 1}/{len(self.image_frame)}] "
            f"{row['biomarker']} | {row['channel']} | {row['split']} | {row['relative_path']}"
        )
        self.label_var.set(
            f"Saved ROIs on this image: {len(annotations)}    "
            f"Active mode: {self.active_label}    "
            "Colors: green=positive, red=negative, yellow=uncertain"
        )
        self.pending_box = None
        self.pending_roi = None
        self.note_var.set("")
        self.redraw_canvas()

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

        if abs(x2 - x1) < 4 or abs(y2 - y1) < 4:
            self.pending_box = None
            self.pending_roi = None
            self.status_var.set("Box too small. Drag a slightly larger ROI.")
            self.redraw_canvas()
            return

        ix1, iy1 = self.canvas_to_image_coords(x1, y1)
        ix2, iy2 = self.canvas_to_image_coords(x2, y2)
        self.pending_roi = {
            "roi_x": min(ix1, ix2),
            "roi_y": min(iy1, iy2),
            "roi_width": max(1, abs(ix2 - ix1)),
            "roi_height": max(1, abs(iy2 - iy1)),
        }
        self.apply_label(self.active_label)

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

        row = self.current_image_row()
        new_row = {
            "task_id": self.next_task_id(row["image_id"]),
            "image_id": row["image_id"],
            "split": row["split"],
            "biomarker": row["biomarker"],
            "channel": row["channel"],
            "relative_path": row["relative_path"],
            "roi_x": self.pending_roi["roi_x"],
            "roi_y": self.pending_roi["roi_y"],
            "roi_width": self.pending_roi["roi_width"],
            "roi_height": self.pending_roi["roi_height"],
            "label": label,
            "notes": self.note_var.get().strip(),
        }

        self.full_frame = self.pd.concat(
            [self.full_frame, self.pd.DataFrame([new_row])], ignore_index=True
        )
        self.status_var.set(f"Added new {label} ROI. Drag again to add more, or press Save CSV.")
        self.load_current_image()

    def clear_draft(self):
        self.pending_box = None
        self.pending_roi = None
        self.status_var.set("Cleared the draft ROI. Saved annotations were not changed.")
        self.redraw_canvas()

    def delete_last_roi(self):
        annotations = self.image_annotations()
        if annotations.empty:
            self.status_var.set("No saved ROI to delete on this image.")
            return

        last_task_id = str(annotations.iloc[-1]["task_id"])
        self.full_frame = self.full_frame[self.full_frame["task_id"] != last_task_id].reset_index(drop=True)
        self.status_var.set(f"Deleted ROI {last_task_id}. Press Save CSV to persist to disk.")
        self.load_current_image()

    def save_csv(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.full_frame.to_csv(self.csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
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


def main():
    args = parse_args()
    np, pd, tifffile, Image, ImageTk = import_or_explain()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(
            f"Annotation CSV not found: {csv_path}\n"
            "Run `python project_2/export_roi_tasks.py` first."
        )

    full_frame = pd.read_csv(csv_path)

    root = tk.Tk()
    AnnotatorApp(root, full_frame, csv_path, np, pd, tifffile, Image, ImageTk)
    root.mainloop()


if __name__ == "__main__":
    main()
