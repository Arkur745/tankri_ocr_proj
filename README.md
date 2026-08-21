# Tankri OCR: Progressive Domain Adaptation & Synthetic Generation

An academic-grade, end-to-end Handwritten Optical Character Recognition (OCR) pipeline for the historical **Tankri script**. This repository includes a custom procedural dataset generator, a progressive domain adaptation framework using PyTorch, and a Streamlit-based interactive recognition tool.

---

## 1. Project Overview & Motivation

The Tankri script (or Takri) is an endangered historical script of northern India. Digitizing historical documents written in Tankri is challenging due to the lack of annotated data.

To solve this, this project:
1. Trains a baseline model on the **1,205 real handwritten images**.
2. Performs **Transfer Learning / Progressive Domain Adaptation (PDA)** starting from the baseline weights and training on the **9,000 synthetic images** (or a balanced **hybrid dataset** of 1,205 real + 2,000 synthetic images) to adapt representations and boost generalization.

```
  [1,205 Real Handwritten Images]
                 │ (Baseline Training)
                 ▼
         [best_model.pth]
                 │
                 ▼
     [Transfer Learning / PDA]  <── (Fine-Tuning on Synthetic / Hybrid Set)
                 │
                 ▼
 [Progressive Domain Adaptation] <── (Freeze conv1-layer2, tune layer4+fc)
                 │
                 ▼
   [best_domain_adapted_model.pth] (19.35% Top-1 on OOD wall inscriptions, up from 6.45%)
```

---

## 2. Key Research Contributions

1. **Procedural Synthetic Generator**: Multi-threaded font rendering pipeline simulating physical degradation, perspective distortions, motion blur, and realistic ink bleeding.
2. **Hybrid Dataset Balancer**: Programmatic combination of real scans with balanced class-specific synthetic samples to prevent representation collapse.
3. **Progressive Unfreezing Strategy**: Freezing early feature extractors while tuning deep semantic representation layers (`layer4` and class classification heads) to adapt to handwriting strokes.
4. **Interactive Streamlit Interface**: Clean app allowing direct image upload, binarization, bounding box cropping, and comparison of baseline vs domain-adapted models.

---

## 3. Results Summary

All numbers below use the 5-seed, leakage-free evaluation methodology in [docs/statistical_rigor.md](docs/statistical_rigor.md) — mean ± std across seeds for in-domain accuracy, Wilson 95% confidence intervals and raw counts for the small out-of-distribution (OOD) test set, not single-run figures.

### In-Domain Validation Accuracy

Best configuration (E05, OCR augmentation + progressive unfreezing combined), 5-seed mean ± std on the held-out validation split (182 images, disjoint from both training and the final test split):

| Metric | Value |
| :--- | :---: |
| **Validation Accuracy** | **89.23% ± 0.95%** |

See [docs/statistical_rigor.md](docs/statistical_rigor.md) for the full E00–E05 ablation breakdown (baseline starts at 83.59% ± 0.97%).

### Out-of-Distribution Domain Shift: Wall Inscriptions

In-domain accuracy above is measured on images from the same acquisition domain as training (scanned/photographed handwritten characters). The real test of domain adaptation is a genuine distribution shift: 31 held-out photographs of temple wall inscriptions — different substrate, lighting, and degradation than any training image. This is a small, hard OOD test set; confidence intervals are wide and reported explicitly rather than glossed over.

| Metric | Baseline (before adaptation) | Domain-Adapted (after adaptation) | Improvement |
| :--- | :---: | :---: | :---: |
| **Top-1 Accuracy** | 6.45% (2/31) [1.79%, 20.72%] | **19.35%** (6/31) [9.19%, 36.28%] | +12.90pp (3.0x) |
| **Top-3 Accuracy** | 19.35% (6/31) [9.19%, 36.28%] | **35.48%** (11/31) [21.12%, 53.05%] | +16.13pp (1.8x) |
| **Top-5 Accuracy** | 19.35% (6/31) [9.19%, 36.28%] | **48.39%** (15/31) [31.97%, 65.16%] | +29.03pp (2.5x) |
| **Mean Confidence** | 59.90% | 48.39% | ECE 0.53 → 0.33 (better calibrated) |

Bracketed ranges are Wilson 95% confidence intervals; see [docs/statistical_rigor.md](docs/statistical_rigor.md) for methodology and [reports/ood_evaluation/](reports/ood_evaluation/) for the full evaluation output. *Note: the CIs and ECE values here are computed by the same evaluation code the paper describes in §III-C, but are more granular than the paper's printed Table VI, which reports point estimates only — this README isn't claiming to reproduce that table verbatim, just providing the fuller detail the underlying methodology already supports.*

**Honest framing:** 19.35% Top-1 on a genuine cross-domain shift is a real, statistically-supported improvement over the 6.45% baseline — but it is a promising first step, not a finished OCR tool for wall inscriptions. The wide confidence intervals above reflect the small OOD sample size (31 images) and should be read as such.

---

## 4. Repository Structure

```
tankri_ocr_proj/
├── archive/                   # Archived obsolete files (configs, notebooks, scripts)
├── artifacts/                 # Shared metadata (label mapping JSONs)
├── configs/                   # Configuration source of truth
├── data/                      # Dataset specifications and DVC guides
├── dataset/                   # DVC-tracked real handwritten dataset (images, test, labels)
├── docs/                      # Pipeline documentation
├── examples/                  # Runnable PowerShell & Shell command recipes
├── models/                    # DVC-tracked model checkpoints
├── notebooks/                 # Research training experiments notebooks
├── reports/                   # Performance reports and matrix plots
├── scripts/                   # CLI verification, execution, and evaluation scripts
├── src/                       # Production OCR codebase
│   ├── configs/               # Global configuration
│   ├── dataset/               # Data loaders, preprocessors, and augmentations
│   ├── inference/             # Standalone inference engine
│   ├── models/                # ResNet18 and SimpleCNN architectures
│   ├── training/              # Training engine
│   ├── evaluation/            # Validation logging and evaluation suite
│   └── utils/                 # Label mapping, MLflow helpers, and general utilities
├── synthetic_generator/       # Procedural glyph generation engine
├── annotator.py               # Dataset annotation tool (Streamlit)
└── app.py                     # Main OCR Recognition Application (Streamlit)
```

---

## 5. Installation

Create and activate a conda environment, then install dependencies:
```bash
conda create -n aimlenv python=3.10
conda activate aimlenv
pip install -r requirements.txt
```

---

## 6. Dataset & Model Availability

**Neither the raw dataset (`dataset/`) nor the trained model checkpoints (`models/*.pth`) are currently publicly released.** The dataset is private pending confirmation of manuscript image rights; the checkpoints are being treated as a separate decision, not yet released either. See [data/README.md](data/README.md) for the full explanation. `dataset.dvc`/`models.dvc` record what's version-tracked locally via DVC, but no public remote is configured, so `dvc pull` will not work for external clones.

What **is** fully public and reproducible: the code, the (untrained) model architecture definitions, the training/evaluation/statistical-rigor methodology (see [docs/](docs/)), and the result artifacts already committed to this repository (`reports/`, `outputs/sample_tuning/`). Given access to the dataset — e.g. as a collaborator — every step below runs as documented.

---

## 7. Execution Guide

These scripts require local dataset/checkpoint access (see Section 6) — they are documented here for methodological transparency and for anyone with dataset access, not as a zero-setup demo:
```bash
# 1. Verify that pipeline data preprocessing and logits are consistent
python scripts/verify_pipeline.py

# 2. Evaluate the baseline model on the test set
python scripts/evaluate_baseline.py

# 3. Evaluate the domain-adapted model on the test set
python scripts/evaluate_baseline.py --use_adapted

# 4. Start the interactive Streamlit recognition application
streamlit run app.py

# 5. Start the dataset label annotator
streamlit run annotator.py
```

Refer to the [examples/](examples/) directory for specific shell (`.sh`) and PowerShell (`.ps1`) recipes.

---

## 8. Documentation

Comprehensive pipeline guides are located in the `docs/` folder:
* [Architecture Documentation](docs/architecture.md)
* [Dataset & Preprocessing Documentation](docs/dataset.md)
* [Training pipeline Documentation](docs/training.md)
* [Evaluation suite Documentation](docs/evaluation.md)
* [Synthetic Generation Documentation](docs/synthetic_generation.md)
* [Progressive Domain Adaptation Documentation](docs/domain_adaptation.md)
* [Notebook & MLflow Tracking Documentation](docs/experiments.md)
* [Statistical Rigor (splits, multi-seed ablations, OOD calibration)](docs/statistical_rigor.md)

---

## 9. Citation

If you build upon this work, please cite:
```bibtex
@software{tankri_ocr_2026,
  author = {Tankri OCR Contributors},
  title = {Tankri Handwritten OCR: Progressive Domain Adaptation and Synthetic Generation Pipeline},
  url = {https://github.com/Arkur745/tankri_ocr_proj},
  version = {1.0.0},
  year = {2026}
}
```

---

## 10. License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
