"""
Project 1 - Data Collection + Migration (Combined)
====================================================
Step 1 — Captures labeled webcam data for each subject:
  project_1/data/<name>/present/train/
  project_1/data/<name>/present/val/
  project_1/data/<name>/absent/train/
  project_1/data/<name>/absent/val/

Step 2 — Immediately migrates into the training structure:
  project_1/data/train/<name>/
  project_1/data/train/Unknown/
  project_1/data/val/<name>/
  project_1/data/val/Unknown/

  The Unknown class is balanced to match the size of the named class
  to prevent the model from defaulting to "Unknown" at inference time.

  The absent phase prompts you to show OTHER faces or leave the frame
  completely so the Unknown class is meaningful during training.

Usage:
  python project_1/collect_and_migrate.py
"""

import cv2
import os
import time
import shutil
import random

# ── Configuration ─────────────────────────────────────────────────────────────
CAMERA_INDEX   = 0
SAMPLE_RATE    = 5    # Frames per second to save
RECORD_SECONDS = 15   # Total recording time per phase
TRAIN_SECONDS  = 10   # First N seconds -> train/
VAL_SECONDS    = 5    # Remaining seconds -> val/
COUNTDOWN      = 5    # Countdown before each phase
DATA_ROOT      = os.path.join("project_1", "data")
# ──────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════
#  PART 1 — DATA COLLECTION
# ═══════════════════════════════════════════════════════════════

def make_dirs(name):
    """Create raw collection folders for one subject."""
    paths = {}
    for label in ["present", "absent"]:
        for split in ["train", "val"]:
            path = os.path.join(DATA_ROOT, name, label, split)
            os.makedirs(path, exist_ok=True)
            paths[(label, split)] = path
    return paths


def countdown(cap, seconds, message):
    start = time.time()
    while True:
        remaining = seconds - int(time.time() - start)
        if remaining <= 0:
            break
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, message, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, f"Starting in {remaining}s...", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow("Data Collection", frame)
        cv2.waitKey(1)


def record_phase(cap, label, paths):
    """Record one phase and save frames split into train/ and val/."""
    frame_interval = 1.0 / SAMPLE_RATE
    start_time     = time.time()
    last_saved     = start_time - frame_interval
    frame_count    = 0

    print(f"\n  Recording '{label}' phase...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("  WARNING: Failed to grab frame.")
            break

        elapsed = time.time() - start_time
        if elapsed >= RECORD_SECONDS:
            break

        split     = "train" if elapsed < TRAIN_SECONDS else "val"
        save_path = paths[(label, split)]

        if time.time() - last_saved >= frame_interval:
            filename = os.path.join(save_path, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            last_saved = time.time()
            frame_count += 1

        remaining = RECORD_SECONDS - elapsed
        cv2.putText(frame, f"Phase : {label.upper()}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        cv2.putText(frame, f"Split : {split}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Time  : {remaining:.1f}s remaining", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Saved : {frame_count} frames", (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        cv2.imshow("Data Collection", frame)
        cv2.waitKey(1)

    print(f"  Saved {frame_count} frames for '{label}' phase.")
    return frame_count


def collect_subject(cap, name):
    """Run present + absent recording for one subject."""
    paths = make_dirs(name)
    print(f"\nData will be saved to: {os.path.join(DATA_ROOT, name)}/")

    # Present phase
    countdown(cap, COUNTDOWN, "Get in front of the camera!")
    record_phase(cap, "present", paths)

    # Absent phase — prompt for a different face or empty background
    countdown(
        cap, COUNTDOWN,
        "Step away / show a DIFFERENT face (for Unknown class)"
    )
    record_phase(cap, "absent", paths)

    # Summary
    print(f"\n  Collection summary for '{name}':")
    for label in ["present", "absent"]:
        for split in ["train", "val"]:
            p     = paths[(label, split)]
            count = len([f for f in os.listdir(p) if f.endswith(".jpg")])
            print(f"    {label:8s} / {split:5s}  ->  {count:3d} frames")

    return paths


# ═══════════════════════════════════════════════════════════════
#  PART 2 — MIGRATION
# ═══════════════════════════════════════════════════════════════

def migrate(subjects):
    """
    Reorganize raw subject folders into train/<class>/ and val/<class>/.

    Key fix: Unknown is randomly DOWN-SAMPLED to match the size of the
    largest named class so training stays balanced.
    """
    print("\n" + "=" * 50)
    print("  Migrating to training structure...")
    print("=" * 50)

    for split in ["train", "val"]:
        unknown_dir = os.path.join(DATA_ROOT, split, "Unknown")
        os.makedirs(unknown_dir, exist_ok=True)

        # ── Collect all absent frames across subjects first ──────────────
        all_absent = []   # list of (src_path, dest_filename)
        present_counts = []

        for subject in subjects:
            # Present frames -> named class folder
            name_dir = os.path.join(DATA_ROOT, split, subject)
            os.makedirs(name_dir, exist_ok=True)

            src_present = os.path.join(DATA_ROOT, subject, "present", split)
            present_copied = 0
            if os.path.isdir(src_present):
                files = [f for f in os.listdir(src_present) if f.endswith(".jpg")]
                for f in files:
                    shutil.copy2(
                        os.path.join(src_present, f),
                        os.path.join(name_dir, f"{subject}_{f}")
                    )
                present_copied = len(files)
            present_counts.append(present_copied)
            print(f"  {split}/{subject}  <- {present_copied} present frames")

            # Gather absent frames (don't copy yet)
            src_absent = os.path.join(DATA_ROOT, subject, "absent", split)
            if os.path.isdir(src_absent):
                files = [f for f in os.listdir(src_absent) if f.endswith(".jpg")]
                for f in files:
                    all_absent.append((
                        os.path.join(src_absent, f),
                        f"{subject}_absent_{f}"
                    ))

        # ── Balance: cap Unknown to match the largest named class ─────────
        max_present = max(present_counts) if present_counts else len(all_absent)
        if len(all_absent) > max_present:
            print(f"  Balancing Unknown: {len(all_absent)} -> {max_present} frames")
            all_absent = random.sample(all_absent, max_present)

        for src, dst_name in all_absent:
            shutil.copy2(src, os.path.join(unknown_dir, dst_name))
        print(f"  {split}/Unknown   <- {len(all_absent)} absent frames (balanced)")

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  Migration complete — final class counts:")
    print("=" * 50)
    for split in ["train", "val"]:
        split_path = os.path.join(DATA_ROOT, split)
        for cls in sorted(os.listdir(split_path)):
            cls_path = os.path.join(split_path, cls)
            if not os.path.isdir(cls_path):
                continue
            count = len([f for f in os.listdir(cls_path) if f.endswith(".jpg")])
            print(f"  {split}/{cls}  ->  {count} images")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 50)
    print("  Project 1 — Collect & Migrate")
    print("=" * 50)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera at index {CAMERA_INDEX}.")
        return

    collected_subjects = []

    try:
        while True:
            name = input("\nEnter subject name (no spaces) or press Enter to finish: ").strip()
            if not name:
                break
            collect_subject(cap, name)
            collected_subjects.append(name)

            another = input("Add another subject? [y/N]: ").strip().lower()
            if another != "y":
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not collected_subjects:
        print("No subjects collected. Exiting.")
        return

    # Also pick up any previously collected subjects not listed above
    existing = [
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d))
        and d not in ("train", "val")
        and d not in collected_subjects
    ]
    if existing:
        print(f"\nFound existing subject folders: {existing}")
        include = input("Include them in migration too? [Y/n]: ").strip().lower()
        if include != "n":
            collected_subjects.extend(existing)

    migrate(collected_subjects)
    print("\nNow run: python project_1/train.py")


if __name__ == "__main__":
    main()
