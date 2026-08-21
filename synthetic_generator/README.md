# Procedural Domain Simulation Module

This module implements a completely independent, modular **Procedural Domain Simulation** pipeline. Its primary objective is to synthetically generate large, realistic OCR datasets by extracting handwriting stroke templates from the training dataset, rendering them on textured backgrounds, and applying configurable document degradation, lighting variation, and sensor artifacts.

The goal is to simulate realistic acquisition domains to evaluate whether synthetic domain diversity improves OCR model robustness on out-of-distribution (OOD) camera-captured test datasets.

---

## Directory Structure

```
synthetic_generator/
 ├── renderer.py               # Handles transparent canvas rendering, template loading, and perturbations
 ├── texture_blending.py       # Implements texture-aware displacement and multi-mode blending
 ├── augmentations.py          # Wraps Augraphy degradation presets and custom OCR augmentations
 ├── metadata.py               # Records metadata and performs dataset integrity verification
 ├── pipeline.py               # Orchestrates stage-by-stage image generation
 ├── generator.py              # Main CLI dataset generator & hybrid merger execution entrypoint
 ├── utils.py                  # Helper functions (seeding, font downloading, texture generators)
 └── README.md                 # This documentation file
```

---

## Rendering Pipeline Workflow

The default procedural generation pipeline progresses through the following modular stages:

```
[ Template Extraction ] ──> Extracts character strokes from real images (range 250-1060)
         │
         ▼
[ Font Perturbation ] ──> Applies baseline shifts, rotation, shear, scale, and translation
         │
         ▼
[ Texture Blending  ] ──> Displaces boundaries using texture gradients and blends texture features
         │
         ▼
[ Doc Degradation   ] ──> Applies Augraphy effects (ink bleed, letterpress, paper wrinkles, shadows)
         │
         ▼
[ OCR Augmentations ] ──> Applies imaging distortions (motion blur, JPEG compression, noise)
         │
         ▼
[   Save Image      ] ──> Resizes and saves final RGB image and records detailed metadata
```

---

## Configuration

Unlike the training pipeline, the generator is **not** driven by `src/configs/config.py`. It is configured directly via the arguments to `run_synthetic_generation()` in [generator.py](generator.py), most conveniently by editing the call at the bottom of that file (`if __name__ == "__main__":`).

### Key Parameters

| Parameter | Description |
| :--- | :--- |
| `num_per_class` | Number of synthetic images to generate per class (the shipped default at the bottom of `generator.py` is `5`, for debug/preview runs — set this to e.g. `200` for a full ~9,000-image corpus across the 45 classes). |
| `output_size` | Dimensions of generated images as `(width, height)`, default `(128, 128)`. |
| `blend_mode` | Texture/glyph blend mode passed to `texture_blending.blend_texture_and_glyph` (default `"carve"`, simulating stone-engraving-style displacement). |
| `debug` | If `True`, saves intermediate per-stage images for the first 5 samples (see below). |

Background textures are auto-loaded (no config flag needed) by `synthetic_generator/utils.load_textures()`, which reads every image it finds in `assets/target_domains/temple_wall/` and `assets/textures/`, in that order.

To build the hybrid dataset (real + synthetic, `real_`/`synth_` filename prefixes to avoid collisions), call `create_hybrid_dataset()` in `generator.py` — it will run the synthetic generator first if `generated_dataset/` doesn't already exist.

---

## Textures and Target Domains

* **Where to place textures**: 
  * Generic textures should be placed in `assets/textures/`.
  * Target domain textures should be placed in `assets/target_domains/{domain_name}/` (e.g. `assets/target_domains/temple_300yr/`).
* **Texture Dimensions**:
  * Textures can be of any size (e.g., $512 \times 512$ or $1024 \times 1024$). The generator automatically crops and resizes the texture to match the target canvas dimensions dynamically at runtime.

---

## Usage Instructions

All commands must be executed from the project root directory.

### 1. Generating a Synthetic Dataset
To run the generator and compile a synthetic dataset:
```bash
python synthetic_generator/generator.py
```
This script will:
* Load the real dataset and filter template images to the range **250 to 1060** matching each class.
* If a class has no template images within that index range, it falls back to the full class subset.
* Crop and extract the dark character strokes into a transparent layer.
* Loop over all classes, render character samples, and save final images to `generated_dataset/`.
* Record detailed per-image metadata to `generated_dataset/metadata/image_metadata.json` (including the source template filename).
* Perform strict post-generation verification checks to ensure dataset integrity.

### 2. Hybrid Dataset Blending
To combine the real dataset and the procedurally simulated synthetic dataset into `generated_dataset_hybrid/`:
```bash
python -c "from synthetic_generator.generator import create_hybrid_dataset; create_hybrid_dataset()"
```
This merges the real images and synthetic images (`real_`/`synth_` filename prefixes to prevent collisions) into a single folder along with a unified `labels.csv`. If `generated_dataset/` doesn't exist yet, it generates the synthetic corpus first (at the `num_per_class` currently set in `generator.py`).

---

## Debugging the Pipeline

Pass `debug=True` to `run_synthetic_generation()` to save intermediate per-stage images for the first 5 samples to `generated_dataset/debug/sample_XXX/`:
* `01_render.png`: Initial text draw on a transparent canvas.
* `02_transformed.png`: Same render (affine perturbation is applied before this stage, not visualized separately).
* `03_texture.png`: Chosen background texture file, resized to the output canvas.
* `04_blended.png`: Texture-blended glyph (after `texture_blending.blend_texture_and_glyph`).
* `05_degraded.png` / `06_augmented.png` / `07_final.png`: Final output after document/sensor degradation (`augmentations.apply_document_degradations`) — these three currently save the same final image.

`scripts/create_pipeline_figure.py` reads a saved debug run from `outputs/sample_tuning/` and composes a labeled 5-stage figure for the paper; see `outputs/sample_tuning/` in the repo root for an example set of stage outputs.
