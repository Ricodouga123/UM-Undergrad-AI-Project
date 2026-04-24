"""
Project 2 - Inference GUI
=========================

Interactive viewer for running the trained patch classifier on a selected
image and drawing predicted spots.

Usage:
  python project_2/infer_gui.py
"""

from __future__ import annotations

import csv
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

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
LABELS_PATH = PROJECT_DIR / "model" / "labels.csv"
CANVAS_MAX_W = 1000
CANVAS_MAX_H = 720


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
        self.current_image = None
        self.current_path = None
        self.current_photo = None
        self.scale = 1.0

        self.model_path_var = tk.StringVar(value=str(MODEL_PATH))
        self.image_path_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Load a model and choose an image.")
        self.summary_var = tk.StringVar(value="")
        self.threshold_var = tk.DoubleVar(value=0.70)
        self.min_distance_var = tk.IntVar(value=18)
        self.max_candidates_var = tk.IntVar(value=200)
        self.candidate_quantile_var = tk.DoubleVar(value=0.985)
        self.box_size_var = tk.IntVar(value=64)
        self.show_all_var = tk.BooleanVar(value=False)

        root.title("Project 2 - Inference Viewer")
        root.configure(bg="#1f2329")

        tk.Label(
            root, text="Project 2 Inference Viewer", font=("Helvetica", 16, "bold"),
            fg="white", bg="#1f2329"
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
        self.add_spinbox(controls, "Proposal Box", self.box_size_var, 1, 2, 2, 16, 256)
        tk.Checkbutton(
            controls,
            text="Show negatives too",
            variable=self.show_all_var,
            fg="white",
            bg="#1f2329",
            selectcolor="#1f2329",
            activebackground="#1f2329",
            activeforeground="white",
        ).grid(row=1, column=4, padx=10, sticky="w")

        self.canvas = tk.Canvas(
            root, width=CANVAS_MAX_W, height=CANVAS_MAX_H, bg="black", highlightthickness=0
        )
        self.canvas.pack(padx=12, pady=8)

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
        self.summary_var.set(f"Loaded model: {path.name}    patch_size={self.patch_size}    device={self.device}")
        self.status_var.set("Model ready.")

    def load_image(self, path: Path):
        if not path.exists():
            messagebox.showerror("Missing image", f"Image file not found:\n{path}")
            return
        self.current_path = path
        self.current_image = self.tifffile.imread(path).astype("float32")
        self.display_image(self.current_image, [])
        self.status_var.set("Image loaded. Click Run to infer spots.")

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

    def crop_patch(self, image, center_x, center_y, box_size):
        half = box_size // 2
        x0 = max(0, center_x - half)
        y0 = max(0, center_y - half)
        x1 = min(image.shape[1], center_x + half)
        y1 = min(image.shape[0], center_y + half)
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

    def display_image(self, image, results):
        display = self.normalize_for_display(image)
        pil_image = self.Image.fromarray(display).convert("RGB")
        draw = self.ImageDraw.Draw(pil_image)

        shown = []
        for item in results:
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

        positives = sum(1 for item in results if item["label"] == "positive")
        negatives = sum(1 for item in results if item["label"] == "negative")
        self.summary_var.set(
            f"Candidates={len(results)}    positives={positives}    negatives={negatives}    shown={len(shown)}"
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

        if self.current_image is None:
            messagebox.showinfo("Choose image", "Pick an image first.")
            return

        peaks = self.propose_candidates(self.current_image)
        results = self.classify_candidates(self.current_image, peaks)
        self.display_image(self.current_image, results)

        csv_path = PROJECT_DIR / "model" / "last_inference.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["center_x", "center_y", "positive_prob", "negative_prob", "label", "box"],
            )
            writer.writeheader()
            for item in results:
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


def main():
    np, tifffile, torch, Image, ImageDraw, ImageTk, peak_local_max, gaussian = import_or_explain()
    root = tk.Tk()
    InferenceApp(root, np, tifffile, torch, Image, ImageDraw, ImageTk, peak_local_max, gaussian)
    root.mainloop()


if __name__ == "__main__":
    main()
