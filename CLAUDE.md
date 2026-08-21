# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Academic-grade OCR pipeline for the historical **Tankri (Takri) script**. Trains a baseline ResNet18 on 1,205 real handwritten character images, then performs Progressive Domain Adaptation (transfer learning) on a synthetic (9,000 images) or hybrid (1,205 real + 2,000 synthetic) dataset to improve generalization on out-of-distribution photographs (e.g. wall inscriptions). Includes a procedural synthetic glyph generator and Streamlit apps for recognition and annotation.

## Environment Setup

```bash
conda create -n aimlenv python=3.10
conda activate aimlenv
pip install -r requirements.txt

# Pull DVC-tracked datasets and model checkpoints
dvc pull
```

There is no linter or automated test suite configured (`requirements.txt` has no pytest/flake8 entries; the only `*test*.py` file is a manual timing script, not a pytest suite). Validate changes by running the relevant script directly and inspecting output/artifacts.

## Common Commands

```bash
# Verify preprocessing/logit consistency end-to-end
python scripts/verify_pipeline.py

# Evaluate baseline vs domain-adapted model on the official test set
python scripts/evaluate_baseline.py
python scripts/evaluate_baseline.py --use_adapted

# Launch interactive apps
streamlit run app.py          # OCR recognition
streamlit run annotator.py    # dataset label annotation

# Generate synthetic training data
python scripts/generate_sample.py
python scripts/create_hybrid_dataset.py

# View experiment tracking dashboard
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Training itself is driven from `notebooks/experiments.ipynb`, not a CLI script: set config in `src/configs/config.py`, then run the notebook cells (config import → dataset load → stratified split → dataloaders → model init → `train()` → `log_evaluation_artifacts()`).

See [examples/run_recipes.ps1](examples/run_recipes.ps1) / [examples/run_recipes.sh](examples/run_recipes.sh) for copy-pasteable command sequences, and `docs/` for narrative documentation of each subsystem (architecture, dataset, training, evaluation, synthetic_generation, domain_adaptation, experiments).

### Statistical rigor / multi-seed ablation workflow

A newer, in-progress workflow (see `IMPLEMENTATION_STATUS.md` and `PRE_TRAINING_CHECKLIST.md` at repo root) adds a clean 70/15/15 stratified train/val/test split and multi-seed ablation reporting for the paper submission:

```bash
# Create/refresh the stratified split (writes artifacts/data_splits/*.json)
python src/utils/data_split.py

# Run the 6-config x 5-seed ablation grid (30 runs, ~90 min on GPU)
python src/training/multi_seed_ablation.py

# Aggregate MLflow results into mean +/- std tables (val and test splits reported separately)
python src/utils/results_aggregator.py

# Evaluate baseline vs adapted model on the fixed 31-image OOD wall-inscription set
# (Wilson score CIs + Expected Calibration Error + reliability diagrams)
python scripts/evaluate_ood_wall_inscriptions.py
```

Important: `models/best_domain_adapted_model.pth` was trained under the old 80/20 split. Before trusting OOD evaluation results, confirm it has been retrained on the new 70/15/15 `train_split.json` to avoid val/test leakage (tracked as an open action item in `PRE_TRAINING_CHECKLIST.md`).

## Architecture

```
Data (dataset/ | generated_dataset/ | generated_dataset_hybrid/)
  -> Preprocessing & Augmentations (src/dataset/)
  -> ResNet18 / SimpleCNN backbone (src/models/)
  -> Training engine + MLflow logging (src/training/)
  -> Evaluation suite (src/evaluation/)
  -> Inference engine (src/inference/) -> Streamlit apps (app.py, annotator.py)
```

**Single source of truth for configuration** is `src/configs/config.py` — no hardcoded hyperparameters or paths elsewhere in training/evaluation code. Critically, this module has side effects at import time: when `ENABLE_DOMAIN_ADAPTATION = True`, it overrides `EPOCHS`, `LEARNING_RATE`, `BATCH_SIZE`, `DATASET_DIR`, `IMAGES_DIR`, and `LABELS_FILE` based on `DOMAIN_ADAPTATION_DATASET` (`"synthetic"` -> `generated_dataset/`, `"hybrid"` -> `generated_dataset_hybrid/`, else `dataset/`). Always check this flag before assuming which dataset/hyperparameters are active.

**Models** (`src/models/`):
- `resnet.py` — wraps `torchvision.models.resnet18`; supports selective layer freezing (`configure_trainable_layers`, controlled by `FREEZE_BACKBONE`/`TRAIN_LAYER3`/`TRAIN_LAYER4`/`TRAIN_CLASSIFIER`) and an optional MLP domain-adaptation head (`ADD_ADAPTATION_HEAD`) vs. the default single linear `fc` layer.
- `simple_cnn.py` — lightweight 3-conv-layer baseline.

**Domain adaptation flow** (see `docs/domain_adaptation.md`): baseline trained on real images with early layers (`conv1`, `bn1`, `layer1`, `layer2`) later frozen; `layer3`/`layer4` and classifier are fine-tuned at a much lower LR (`1e-5`–`5e-5`) on synthetic/hybrid data for 15–25 epochs, loading `best_model.pth` as the starting checkpoint and saving to `best_domain_adapted_model.pth`/`last_domain_adapted_model.pth` (baseline checkpoints are `best_model.pth`/`last_model.pth`).

**Training** (`src/training/train.py`): standard train/val loop, Adam + `CosineAnnealingLR`, `nn.CrossEntropyLoss` with optional label smoothing, early-checkpoint auto-save. All runs and artifacts (label mappings, config snapshot, augmentation config, weights) log to MLflow backed by local SQLite `mlflow.db`.

**Preprocessing** (`src/dataset/preprocessing.py`): grayscale -> inverse-threshold binarize -> `cv2.findNonZero` bounding box -> tight crop with 20px white border. This exact pipeline must match between training and `src/inference/inference.py` (`preprocess_image_for_model`) — the two are meant to stay in lockstep, which is what `scripts/verify_pipeline.py` checks.

**Synthetic generator** (`synthetic_generator/`): `renderer.py` (PIL glyph rendering from TTF fonts) -> `texture_blending.py` (paper/ink blend modes) -> `augmentations.py` (perspective warps, elastic deformation, scan noise) -> `pipeline.py` (multi-threaded orchestration producing balanced-class image sets + `labels.csv`).

**Path resolution**: `src/inference/inference.py` resolves checkpoints/label-mapping paths relative to both CWD and project root with fallback candidate lists (`resolve_checkpoint_path`, `load_label_mappings`) since the Streamlit apps and scripts may be invoked from different working directories.

## Data & Model Versioning

Real dataset (`dataset/`) and model checkpoints (`models/`) are DVC-tracked (`.dvc` files at repo root), not committed directly to git — use `dvc pull`/`dvc push`. `dataset/images/` and `dataset/processed/` are also gitignored directly. Synthetic datasets (`generated_dataset/`, `generated_dataset_hybrid/`) are generated locally via the scripts above rather than versioned.

Label mappings (`label_to_idx.json`/`idx_to_label.json`) are duplicated across `artifacts/`, `models/`, and sometimes `notebooks/` — these must stay consistent with whichever checkpoint is loaded; `src/utils/label_mapping.py` is the canonical loader.
