# Project 2 - Dual-Imaging Vesicle Counting

This project turns the paper's image-analysis idea into an undergraduate-friendly AI workflow.

## What the spots mean

There are two image channels in the dataset:

- `Raman` images are the "find all vesicles" channel.
  In the paper this role is played by the fluorescence EV mask: it tells us where all EVs are located.
- `DF` images are the "find marker-positive vesicles" channel.
  Bright dark-field spots correspond to EVs with a bound gold nanoparticle (AuNP), which means the target surface marker was detected.

In practical terms:

- dimmer spots = vesicles present, but not strongly marker-positive
- bright spots with a halo/ring in dark-field = strong scatterers, usually AuNP-positive vesicles

That "ring" is not a separate biological object. It is an optical imaging artifact caused by the microscope point-spread function around a bright scattering particle.

## What we want students to learn

This project is intentionally staged so students can grow from basic image processing to AI:

1. `Dataset engineering`
   Organize raw TIFFs, parse metadata, and create reproducible train/validation/test splits.
2. `Classical computer vision baseline`
   Use thresholding and blob detection to count likely vesicles and bright AuNP-positive spots.
3. `ROI labeling`
   Create a small manually labeled dataset of vesicle-centered patches and mark them as `negative`, `positive`, or `uncertain`.
4. `Supervised learning`
   Train a lightweight patch classifier on the ROI labels.
5. `Advanced extension`
   Move from patch classification to spot detection or segmentation.

Reinforcement learning is not a natural fit for this problem, so the recommended advanced path is `segmentation/detection`, not RL.

## Current dataset assumptions

The raw images currently live under:

- [project_2/images/cd44](/C:/Users/ricod/Projects/UM-Undergrad-AI-Project/project_2/images/cd44)
- [project_2/images/epcam](/C:/Users/ricod/Projects/UM-Undergrad-AI-Project/project_2/images/epcam)
- [project_2/images/her2](/C:/Users/ricod/Projects/UM-Undergrad-AI-Project/project_2/images/her2)

Each image belongs to:

- a `biomarker`: `cd44`, `epcam`, or `her2`
- a `channel`: `df` or `raman`
- often a `cell line`: `SKBR3`, `MM231`, or `unknown`

The filename parser in `build_manifest.py` extracts this metadata automatically.

## Layout

The project is organized like this:

- [project_2/apps](/C:/Users/ricod/Projects/UM-Undergrad-AI-Project/project_2/apps) for interactive GUIs
- [project_2/pipelines](/C:/Users/ricod/Projects/UM-Undergrad-AI-Project/project_2/pipelines) for dataset prep, training, and baseline scripts
- [project_2/core](/C:/Users/ricod/Projects/UM-Undergrad-AI-Project/project_2/core) for shared helpers
- top-level scripts in [project_2](/C:/Users/ricod/Projects/UM-Undergrad-AI-Project/project_2) as simple launchers for common tasks

## Recommended workflow

1. Fastest path: prepare everything and open the annotation GUI:

```powershell
python project_2/start_annotation.py
```

This will:

- rebuild the manifest
- sync `roi_tasks.csv` without wiping your existing annotations
- launch the annotation GUI

2. If you want to run the steps separately, build a manifest and split the data:

```powershell
python project_2/pipelines/build_manifest.py
```

This creates:

- [project_2/data/manifest.csv](/C:/Users/ricod/Projects/UM-Undergrad-AI-Project/project_2/data/manifest.csv)
- [project_2/data/manifest.json](/C:/Users/ricod/Projects/UM-Undergrad-AI-Project/project_2/data/manifest.json)

The split is grouped so near-duplicate images stay in the same split. The `test` split is the verification set and must remain untouched during training.

3. Generate or sync the ROI labeling sheet:

```powershell
python project_2/pipelines/export_roi_tasks.py
```

4. Launch the annotation GUI:

```powershell
python project_2/annotate_gui.py
```

By default it opens the CSV at [project_2/annotations/roi_tasks.csv](/C:/Users/ricod/Projects/UM-Undergrad-AI-Project/project_2/annotations/roi_tasks.csv), shows one image at a time, and lets you:

- drag a rectangle over a spot
- click `Positive`, `Negative`, or `Uncertain`
- save the ROI directly back into the CSV
- move with `Next` / `Previous`
- switch `split`, `channel`, `biomarker`, and `unlabeled only` filters inside the GUI

5. Run the classical baseline:

```powershell
python project_2/baseline_count.py --split val
```

6. After manual ROI labels exist, train the patch classifier:

```powershell
python project_2/train_patch_classifier.py
```

7. Open the inference GUI to test the trained model on an image:

```powershell
python project_2/infer_gui.py
```

The inference GUI lets you:

- load a trained model
- browse to a TIFF image
- run candidate spot detection plus patch classification
- overlay predicted positives on the image
- adjust threshold and proposal settings interactively

## Dependencies

```powershell
pip install numpy pandas tifffile scikit-image matplotlib torch torchvision
```

For Jetson Orin Nano, use the NVIDIA-compatible PyTorch build rather than the default PyPI wheel.

## Suggested evaluation targets

Students should report:

- total vesicle count from the `raman` channel
- AuNP-positive vesicle count from the `df` channel
- positive fraction: `positive_count / total_count`
- split-aware metrics on the held-out `test` set

## Files

- [project_2/start_annotation.py](/C:/Users/ricod/Projects/UM-Undergrad-AI-Project/project_2/start_annotation.py): one-command setup plus GUI launch
- [project_2/annotate_gui.py](/C:/Users/ricod/Projects/UM-Undergrad-AI-Project/project_2/annotate_gui.py): top-level launcher for the annotation GUI
- [project_2/train_patch_classifier.py](/C:/Users/ricod/Projects/UM-Undergrad-AI-Project/project_2/train_patch_classifier.py): top-level launcher for model training
- [project_2/infer_gui.py](/C:/Users/ricod/Projects/UM-Undergrad-AI-Project/project_2/infer_gui.py): top-level launcher for the inference viewer
- [project_2/baseline_count.py](/C:/Users/ricod/Projects/UM-Undergrad-AI-Project/project_2/baseline_count.py): top-level launcher for the classical baseline
- [project_2/pipelines/build_manifest.py](/C:/Users/ricod/Projects/UM-Undergrad-AI-Project/project_2/pipelines/build_manifest.py): parse filenames and create train/val/test metadata
- [project_2/pipelines/export_roi_tasks.py](/C:/Users/ricod/Projects/UM-Undergrad-AI-Project/project_2/pipelines/export_roi_tasks.py): create or sync the annotation sheet
- [project_2/pipelines/train_patch_classifier.py](/C:/Users/ricod/Projects/UM-Undergrad-AI-Project/project_2/pipelines/train_patch_classifier.py): training implementation
- [project_2/apps/annotate_gui.py](/C:/Users/ricod/Projects/UM-Undergrad-AI-Project/project_2/apps/annotate_gui.py): click-and-drag ROI labeling tool
- [project_2/apps/infer_gui.py](/C:/Users/ricod/Projects/UM-Undergrad-AI-Project/project_2/apps/infer_gui.py): interactive model viewer for image-level predictions
- [project_2/core/model_utils.py](/C:/Users/ricod/Projects/UM-Undergrad-AI-Project/project_2/core/model_utils.py): shared classifier/model helpers
