# UM-Undergrad-AI-Project
Training ground for undergrads

## Project 1

Use the camera input to collect training data. Then use the model to predict the person's identity based on their facial image.

Then apply the model to real time camera feed, so that it can highlight face


## Project 2

Microscopy training ground based on dual-imaging extracellular vesicle data.

Students work with paired images:
- `Raman` / fluorescence-style images to find all vesicles.
- `DF` dark-field images to find brighter AuNP-positive vesicles.

The current workflow for `project_2` is:
- build a dataset manifest and deterministic train/val/test split
- start with classical blob counting as a baseline
- add ROI labeling for supervised learning
- train a simple patch classifier or detector on labeled spots

See [project_2/README.md](/C:/Users/ricod/Projects/UM-Undergrad-AI-Project/project_2/README.md) for details.
