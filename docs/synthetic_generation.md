# Synthetic Glyph Generation Pipeline

This document describes the procedural rendering engine used to generate the synthetic glyph training set.

## 1. Engine Components (`synthetic_generator/`)

The generator converts character vectors and TTF fonts into realistic scans:
* **Renderer (`renderer.py`)**: Renders characters on virtual canvases using custom PIL draw calls, handling scaling, positions, and aspect ratios.
* **Texture Blending (`texture_blending.py`)**: Blends glyph channels with background paper textures using blending modes (e.g. multiply, burn) to simulate real-world ink bleeding.
* **Augmentations (`augmentations.py`)**: Applies geometric and perspective warps, local elastic deformations, salt-and-pepper scan noise, and gaussian blurs.
* **Pipeline (`pipeline.py`)**: Combines components into a single multi-threaded synthesis pipeline, generating large volumes of characters under balanced class conditions.

---

## 2. Generator Execution (`scripts/generate_sample.py`)

A script to run the synthetic rendering pipeline.
* Configures output resolutions, fonts, background directories, ink textures, and augmentations.
* Exports generated PNGs along with a structured CSV label map `labels.csv` to the target directory.
