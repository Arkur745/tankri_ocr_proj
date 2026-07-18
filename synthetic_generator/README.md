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

All parameters are configured in [src/config.py](file:///d:/Dev/tankri_ocr_proj/src/config.py) under the **PROCEDURAL DATASET GENERATION CONFIG** section.

### Key Configuration Variables

| Variable | Description |
| :--- | :--- |
| `ENABLE_PROCEDURAL_GENERATION` | Master flag to enable procedural pipeline integration. |
| `TRAINING_DATASET` | Chooses active dataset: `"real"`, `"synthetic"`, or `"hybrid"`. |
| `IMAGES_PER_CLASS` | Number of synthetic images to generate per class. |
| `OUTPUT_IMAGE_SIZE` | Dimensions of generated images (default `128`). |
| `RENDER_MODE` | Rendering preset: `"paper"`, `"stone"`, `"wall"`, `"manuscript"`, or `"custom"`. |
| `TEXTURE_MODE` | Texture collection: `"generic"` (all textures) or `"target"` (specific domain folder). |
| `TARGET_DOMAIN` | Subfolder in `assets/target_domains/` to use if `TEXTURE_MODE = "target"`. |
| `DEBUG_PIPELINE` | Save intermediate stage images for the first 5 samples. |

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
To automatically combine the real dataset and the procedurally simulated synthetic dataset:
1. Open [src/config.py](file:///d:/Dev/tankri_ocr_proj/src/config.py).
2. Set `TRAINING_DATASET = "hybrid"`.
3. Run the generator:
   ```bash
   python synthetic_generator/generator.py
   ```
The script will generate the synthetic dataset first, and then merge the real images and synthetic images (using a `synth_` prefix to prevent collision) into a single folder `generated_dataset_hybrid/` along with a unified `labels.csv`.

---

## Debugging the Pipeline

If you want to visualize how each augmentation stage alters the transparent character layer, set `DEBUG_PIPELINE = True` in [src/config.py](file:///d:/Dev/tankri_ocr_proj/src/config.py).

The generator will save intermediate step files for the first 5 samples in `generated_dataset/debug/sample_XXX/`:
* `01_render.png`: Initial text draw on a transparent canvas.
* `02_transformed.png`: Image after affine shear, scale, rotation, and translation.
* `03_texture.png`: Chosen background texture file.
* `04_blended.png`: Displaced boundaries and texture-blended glyph.
* `05_degraded.png`: Image after Augraphy document degradation.
* `06_augmented.png`: Image after OpenCV imaging augmentations.
* `07_final.png`: Resized and post-processed final output.
