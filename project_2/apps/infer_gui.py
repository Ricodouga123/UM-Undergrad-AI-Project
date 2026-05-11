"""
Project 2 - Inference GUI
=========================
"""

from __future__ import annotations

import csv
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

from core.image_analysis import (
    centered_box,
    compute_roi_metrics,
    crop_box,
    normalize_for_display,
    render_roi_analysis_image,
)
from core.model_utils import load_patch_classifier


def import_or_explain():
    try:
        import numpy as np
        import tifffile
        import torch
        from PIL import Image, ImageDraw, ImageTk
        from skimage.feature import peak_local_max
        from skimage.filters import gaussian
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency for Project 2 inference GUI. Install: "
            "numpy tifffile pillow torch scikit-image"
        ) from exc
    return np, tifffile, torch, Image, ImageDraw, ImageTk, peak_local_max, gaussian


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "model" / "patch_classifier.pt"
CANVAS_MAX_W = 1000
CANVAS_MAX_H = 720
DEFAULT_ROI_SIZE = 32


class InferenceApp:
    def __init__(
        self, root, np, tifffile, torch, Image, ImageDraw, ImageTk, peak_local_max, gaussian
    ):
        self.root = root
        self.np = np
        self.tifffile = tifffile
        self.torch = torch
        self.Image = Image
        self.ImageDraw = ImageDraw
        self.ImageTk = ImageTk
        self.peak_local_max = peak_local_max
        self.gaussian = gaussian

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.patch_size = None
        self.current_image_raw = None
        self.current_results = []
        self.current_path = None
        self.current_photo = None
        self.scale = 1.0
        self.analysis_windows = []

        self.model_path_var = tk.StringVar(value=str(MODEL_PATH))
        self.image_path_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Load a model and choose an image.")
        self.summary_var = tk.StringVar(value="")
        self.threshold_var = tk.DoubleVar(value=0.70)
        self.min_distance_var = tk.IntVar(value=18)
        self.max_candidates_var = tk.IntVar(value=200)
        self.candidate_quantile_var = tk.DoubleVar(value=0.985)
        self.box_size_var = tk.IntVar(value=DEFAULT_ROI_SIZE)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.analysis_roi_size_var = tk.IntVar(value=DEFAULT_ROI_SIZE)
        self.show_all_var = tk.BooleanVar(value=False)

        root.title("Project 2 - Inference Viewer")
        root.configure(bg="#1f2329")

        tk.Label(
            root,
            text="Project 2 Inference Viewer",
            font=("Helvetica", 16, "bold"),
            fg="white",
            bg="#1f2329",
        ).pack(pady=(10, 6))

        top = tk.Frame(root, bg="#1f2329")
        top.pack(fill="x", padx=12, pady=(0, 8))

        tk.Label(top, text="Model", fg="white", bg="#1f2329").grid(row=0, column=0, sticky="w")
        tk.Entry(top, textvariable=self.model_path_var, width=80).grid(row=0, column=1, padx=6, sticky="ew")
        tk.Button(top, text="Browse", command=self.choose_model, width=10).grid(row=0, column=2, padx=4)
        tk.Button(top, text="Load Model", command=self.load_model, width=10).grid(row=0, column=3, padx=4)

        tk.Label(top, text="Image", fg="white", bg="#1f2329").grid(row=1, column=0, sticky="w")
        tk.Entry(top, textvariable=self.image_path_var, width=80).grid(row=1, column=1, padx=6, sticky="ew")
        tk.Button(top, text="Browse", command=self.choose_image, width=10).grid(row=1, column=2, padx=4)
        tk.Button(top, text="Run", command=self.run_inference, width=10).grid(row=1, column=3, padx=4)
        top.columnconfigure(1, weight=1)

        controls = tk.Frame(root, bg="#1f2329")
        controls.pack(fill="x", padx=12, pady=(0, 8))

        self.add_spinbox(controls, "Positive Threshold", self.threshold_var, 0, 0, 0.05, 0.0, 1.0)
        self.add_spinbox(controls, "Min Distance", self.min_distance_var, 0, 2, 1, 1, 200)
        self.add_spinbox(controls, "Max Candidates", self.max_candidates_var, 0, 4, 10, 10, 1000)
        self.add_spinbox(controls, "Candidate Quantile", self.candidate_quantile_var, 1, 0, 0.001, 0.5, 0.999)
        self.add_spinbox(controls, "Proposal Box", self.box_size_var, 1, 2, 2, 8, 256)
        self.add_spinbox(controls, "Analyzer ROI", self.analysis_roi_size_var, 1, 4, 2, 8, 256)
        tk.Checkbutton(
            controls,
            text="Show negatives too",
            variable=self.show_all_var,
            fg="white",
            bg="#1f2329",
            selectcolor="#1f2329",
            activebackground="#1f2329",
            activeforeground="white",
            command=self.refresh_display,
        ).grid(row=1, column=6, padx=10, sticky="w")

        contrast_row = tk.Frame(root, bg="#1f2329")
        contrast_row.pack(fill="x", padx=12, pady=(0, 8))
        tk.Label(contrast_row, text="Contrast", fg="white", bg="#1f2329").pack(side="left", padx=(0, 6))
        tk.Scale(
            contrast_row,
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
        tk.Label(
            contrast_row,
            text="Right-click analyzes a centered ROI",
            fg="#c9d1d9",
            bg="#1f2329",
            font=("Courier", 9),
        ).pack(side="left", padx=(14, 4))

        self.canvas = tk.Canvas(
            root,
            width=CANVAS_MAX_W,
            height=CANVAS_MAX_H,
            bg="black",
            highlightthickness=0,
        )
        self.canvas.pack(padx=12, pady=8)
        self.canvas.bind("<Button-3>", self.on_analyze_click)

        tk.Label(root, textvariable=self.summary_var, font=("Courier", 10), fg="#c9d1d9", bg="#1f2329").pack()
        tk.Label(root, textvariable=self.status_var, font=("Courier", 10), fg="#58a6ff", bg="#1f2329").pack(pady=(2, 10))

        if MODEL_PATH.exists():
            self.load_model()

    def add_spinbox(self, parent, label, variable, row, column, increment, from_, to):
        tk.Label(parent, text=label, fg="white", bg="#1f2329").grid(row=row, column=column, sticky="w", padx=4)
        tk.Spinbox(
            parent,
            textvariable=variable,
            increment=increment,
            from_=from_,
            to=to,
            width=10,
        ).grid(row=row, column=column + 1, sticky="w", padx=(0, 12))

    def choose_model(self):
        path = filedialog.askopenfilename(
            title="Choose model",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self.model_path_var.set(path)

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Choose image",
            filetypes=[("TIFF images", "*.tif *.tiff"), ("All files", "*.*")],
            initialdir=str(PROJECT_DIR / "images"),
        )
        if path:
            self.image_path_var.set(path)
            self.load_image(Path(path))

    def load_model(self):
        path = Path(self.model_path_var.get())
        if not path.exists():
            messagebox.showerror("Missing model", f"Model file not found:\n{path}")
            return
        import torch.nn as nn

        self.model, self.patch_size = load_patch_classifier(path, self.torch, nn, self.device)
        self.summary_var.set(
            f"Loaded model: {path.name}    patch_size={self.patch_size}    device={self.device}"
        )
        self.status_var.set("Model ready.")

    def load_image(self, path: Path):
        if not path.exists():
            messagebox.showerror("Missing image", f"Image file not found:\n{path}")
            return
        self.current_path = path
        self.current_image_raw = self.tifffile.imread(path).astype("float32")
        self.current_results = []
        self.refresh_display()
        self.status_var.set("Image loaded. Click Run to infer spots.")

    def crop_patch(self, image, center_x, center_y, box_size):
        half = int(box_size) // 2
        x0 = max(0, int(center_x) - half)
        y0 = max(0, int(center_y) - half)
        x1 = min(image.shape[1], x0 + int(box_size))
        y1 = min(image.shape[0], y0 + int(box_size))
        if x1 - x0 < int(box_size):
            x0 = max(0, x1 - int(box_size))
        if y1 - y0 < int(box_size):
            y0 = max(0, y1 - int(box_size))
        patch = image[y0:y1, x0:x1]
        return patch, (x0, y0, x1, y1)

    def patch_to_tensor(self, patch):
        patch = patch.astype("float32")
        patch = patch - patch.min()
        if patch.max() > 0:
            patch = patch / patch.max()
        tensor = self.torch.tensor(
            self.np.expand_dims(patch, axis=0), dtype=self.torch.float32, device=self.device
        ).unsqueeze(0)
        tensor = self.torch.nn.functional.interpolate(
            tensor,
            size=(self.patch_size, self.patch_size),
            mode="bilinear",
            align_corners=False,
        )
        return tensor

    def propose_candidates(self, image):
        smoothed = self.gaussian(image, sigma=1.0, preserve_range=True)
        cutoff = float(self.np.quantile(smoothed, self.candidate_quantile_var.get()))
        peaks = self.peak_local_max(
            smoothed,
            min_distance=self.min_distance_var.get(),
            threshold_abs=cutoff,
            num_peaks=self.max_candidates_var.get(),
        )
        return peaks

    def classify_candidates(self, image, peaks):
        results = []
        for y, x in peaks:
            patch, box = self.crop_patch(image, int(x), int(y), int(self.box_size_var.get()))
            if patch.size == 0 or patch.shape[0] < 4 or patch.shape[1] < 4:
                continue
            tensor = self.patch_to_tensor(patch)
            with self.torch.no_grad():
                probs = self.torch.softmax(self.model(tensor), dim=1)[0]
            negative_prob = float(probs[0].item())
            positive_prob = float(probs[1].item())
            label = "positive" if positive_prob >= self.threshold_var.get() else "negative"
            results.append(
                {
                    "center_x": int(x),
                    "center_y": int(y),
                    "box": box,
                    "positive_prob": positive_prob,
                    "negative_prob": negative_prob,
                    "label": label,
                }
            )
        return results

    def refresh_display(self):
        if self.current_image_raw is None:
            return
        display = normalize_for_display(self.current_image_raw, self.np, self.contrast_var.get())
        pil_image = self.Image.fromarray(display).convert("RGB")
        draw = self.ImageDraw.Draw(pil_image)

        shown = []
        for item in self.current_results:
            if item["label"] != "positive" and not self.show_all_var.get():
                continue
            x0, y0, x1, y1 = item["box"]
            color = "#00d084" if item["label"] == "positive" else "#ff6b6b"
            draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
            draw.text((x0 + 2, max(0, y0 - 12)), f"{item['positive_prob']:.2f}", fill=color)
            shown.append(item)

        scale = min(CANVAS_MAX_W / pil_image.width, CANVAS_MAX_H / pil_image.height, 1.0)
        self.scale = scale
        resized = pil_image.resize(
            (max(1, int(round(pil_image.width * scale))), max(1, int(round(pil_image.height * scale))))
        )

        self.current_photo = self.ImageTk.PhotoImage(resized)
        self.canvas.config(width=resized.width, height=resized.height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_photo)

        positives = sum(1 for item in self.current_results if item["label"] == "positive")
        negatives = sum(1 for item in self.current_results if item["label"] == "negative")
        self.summary_var.set(
            f"Candidates={len(self.current_results)}    positives={positives}    negatives={negatives}    shown={len(shown)}"
        )

    def run_inference(self):
        if self.model is None:
            self.load_model()
            if self.model is None:
                return

        image_path = self.image_path_var.get().strip()
        if image_path:
            path = Path(image_path)
            if self.current_path != path:
                self.load_image(path)

        if self.current_image_raw is None:
            messagebox.showinfo("Choose image", "Pick an image first.")
            return

        peaks = self.propose_candidates(self.current_image_raw)
        self.current_results = self.classify_candidates(self.current_image_raw, peaks)
        self.refresh_display()

        csv_path = PROJECT_DIR / "model" / "last_inference.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["center_x", "center_y", "positive_prob", "negative_prob", "label", "box"],
            )
            writer.writeheader()
            for item in self.current_results:
                writer.writerow(
                    {
                        "center_x": item["center_x"],
                        "center_y": item["center_y"],
                        "positive_prob": f"{item['positive_prob']:.6f}",
                        "negative_prob": f"{item['negative_prob']:.6f}",
                        "label": item["label"],
                        "box": item["box"],
                    }
                )

        self.status_var.set(
            f"Finished inference on {self.current_path.name}. Results saved to {csv_path.name}."
        )

    def canvas_to_image_coords(self, x, y):
        return int(round(x / self.scale)), int(round(y / self.scale))

    def show_roi_analysis(self, center_x: int, center_y: int):
        box = centered_box(self.current_image_raw.shape, center_x, center_y, int(self.analysis_roi_size_var.get()))
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


def main():
    np, tifffile, torch, Image, ImageDraw, ImageTk, peak_local_max, gaussian = import_or_explain()
    root = tk.Tk()
    InferenceApp(root, np, tifffile, torch, Image, ImageDraw, ImageTk, peak_local_max, gaussian)
    root.mainloop()


if __name__ == "__main__":
    main()
