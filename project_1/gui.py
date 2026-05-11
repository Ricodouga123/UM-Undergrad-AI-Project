"""
Project 1 - Live Inference GUI
================================
Loads the trained MobileNetV2 and runs live face detection +
identity classification on the webcam feed.

- Green box = recognized person, shows their name
- Red box   = Unknown face

Usage:
  python project_1/gui.py
"""

import os
import sys
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import tkinter as tk
from PIL import Image, ImageTk
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_DIR          = Path(__file__).resolve().parent
LEGACY_PROJECT_DIR   = PROJECT_DIR / "project_1"
CAMERA_INDEX        = 0
IMG_SIZE            = 128
HAAR_CASCADE        = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CONTEXT_SCALE  = 2.5

# If the model's top confidence is below this value the face is treated as
# Unknown regardless of what class index won.  Tune between 0.55 – 0.80.
CONFIDENCE_THRESHOLD = 0.50
# ──────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_model_paths():
    candidate_dirs = [PROJECT_DIR / "model", LEGACY_PROJECT_DIR / "model"]
    best_pair = None
    best_mtime = -1.0
    for model_dir in candidate_dirs:
        model_path = model_dir / "best_model.pth"
        classes_path = model_dir / "class_names.txt"
        if not model_path.exists() or not classes_path.exists():
            continue
        mtime = model_path.stat().st_mtime
        if mtime > best_mtime:
            best_pair = (model_path, classes_path)
            best_mtime = mtime
    if best_pair is not None:
        return best_pair
    return (PROJECT_DIR / "model" / "best_model.pth", PROJECT_DIR / "model" / "class_names.txt")


def load_class_names(path):
    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f.readlines() if line.strip()]
    print(f"  Loaded {len(names)} classes: {names}")
    # Sanity-check: warn if Unknown is missing
    if "Unknown" not in names:
        print("  WARNING: 'Unknown' class not found in class_names.txt")
    return names


def load_model(model_path, num_classes):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def expand_face_crop(frame_bgr, x, y, w, h, scale=FACE_CONTEXT_SCALE):
    height, width = frame_bgr.shape[:2]
    cx = x + (w / 2.0)
    cy = y + (h / 2.0)
    crop_w = w * scale
    crop_h = h * scale
    x0 = max(0, int(round(cx - (crop_w / 2.0))))
    y0 = max(0, int(round(cy - (crop_h / 2.0))))
    x1 = min(width, int(round(cx + (crop_w / 2.0))))
    y1 = min(height, int(round(cy + (crop_h / 2.0))))
    return frame_bgr[y0:y1, x0:x1], (x0, y0, x1, y1)


def predict(model, transform, face_bgr, class_names):
    """
    Returns (label, confidence).

    Falls back to 'Unknown' when the winning probability is below
    CONFIDENCE_THRESHOLD so borderline faces are not misidentified.
    """
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    tensor   = transform(Image.fromarray(face_rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    conf, idx = probs.max(0)
    label     = class_names[idx.item()]
    conf_val  = conf.item()

    # Below threshold -> treat as Unknown regardless of predicted class
    if conf_val < CONFIDENCE_THRESHOLD:
        label = "Unknown"

    return label, conf_val


class LiveGUI:
    def __init__(self, root, cap, model, class_names, transform):
        self.root        = root
        self.cap         = cap
        self.model       = model
        self.class_names = class_names
        self.transform   = transform
        self.detector    = cv2.CascadeClassifier(HAAR_CASCADE)
        self.running     = True

        root.title("Project 1 — Face Identity")
        root.configure(bg="#1e1e1e")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        tk.Label(root, text="Face Identity — Live Feed",
                 font=("Helvetica", 16, "bold"),
                 fg="#ffffff", bg="#1e1e1e").pack(pady=(12, 2))

        tk.Label(root, text=f"Device: {DEVICE}",
                 font=("Courier", 10), fg="#00cc88", bg="#1e1e1e").pack()

        self.canvas = tk.Canvas(root, width=640, height=480,
                                bg="#000000", highlightthickness=0)
        self.canvas.pack(padx=16, pady=10)

        self.status_var = tk.StringVar(value="Starting...")
        tk.Label(root, textvariable=self.status_var,
                 font=("Courier", 11), fg="#aaaaaa", bg="#1e1e1e").pack(pady=(0, 6))

        tk.Button(root, text="Quit", command=self.on_close,
                  font=("Helvetica", 11), bg="#cc3333", fg="white",
                  activebackground="#aa1111", relief="flat",
                  padx=20, pady=6).pack(pady=(0, 14))

        self.update_frame()

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_var.set("ERROR: Could not read frame.")
            self.root.after(100, self.update_frame)
            return

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        status_parts = []

        for (x, y, w, h) in faces:
            face_crop, context_box = expand_face_crop(frame, x, y, w, h)
            if face_crop.size == 0:
                continue

            label, conf = predict(self.model, self.transform,
                                  face_crop, self.class_names)

            is_known = label != "Unknown"
            color    = (0, 220, 0) if is_known else (0, 0, 220)
            display  = label.replace("_", " ")

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            x0, y0, x1, y1 = context_box
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 1)

            text = f"{display}  {conf*100:.1f}%"
            tpos = (x, max(y - 10, 20))
            cv2.putText(frame, text, (tpos[0]+1, tpos[1]+1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(frame, text, tpos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            status_parts.append(display)

        self.status_var.set(
            f"Detected: {', '.join(status_parts)}" if status_parts else "No face detected"
        )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (640, 480))
        img       = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.image = img

        self.root.after(30, self.update_frame)

    def on_close(self):
        self.running = False
        self.cap.release()
        self.root.destroy()


def main():
    model_path, classes_path = resolve_model_paths()
    for path in [model_path, classes_path]:
        if not path.exists():
            print(f"ERROR: Missing file: {path}")
            print("Run train.py first.")
            sys.exit(1)

    print(f"Loading model from {model_path} ...")
    class_names = load_class_names(classes_path)
    model       = load_model(model_path, num_classes=len(class_names))
    transform   = get_transform()
    print(f"  Device  : {DEVICE}")
    print(f"  Threshold: {CONFIDENCE_THRESHOLD}")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera at index {CAMERA_INDEX}.")
        sys.exit(1)

    root = tk.Tk()
    LiveGUI(root, cap, model, class_names, transform)
    root.mainloop()


if __name__ == "__main__":
    main()
