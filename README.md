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
   [best_domain_adapted_model.pth] (22.58% accuracy)
```

---

## 2. Key Research Contributions

1. **Procedural Synthetic Generator**: Multi-threaded font rendering pipeline simulating physical degradation, perspective distortions, motion blur, and realistic ink bleeding.
2. **Hybrid Dataset Balancer**: Programmatic combination of real scans with balanced class-specific synthetic samples to prevent representation collapse.
3. **Progressive Unfreezing Strategy**: Freezing early feature extractors while tuning deep semantic representation layers (`layer4` and class classification heads) to adapt to handwriting strokes.
4. **Interactive Streamlit Interface**: Clean app allowing direct image upload, binarization, bounding box cropping, and comparison of baseline vs domain-adapted models.

---

## 3. Results Summary

Evaluation conducted on the official unseen handwritten test set (31 samples, 45 classes):

| Metric | Baseline (Trained on 1,205 Real Images) | Domain-Adapted (PDA Transfer Learned on Synthetic/Hybrid) | Improvement |
| :--- | :---: | :---: | :---: |
| **Top-1 Accuracy** | 6.45% | **22.58%** | **+16.13% (3.5x)** |
| **Top-3 Accuracy** | 19.35% | **54.84%** | **+35.49% (2.8x)** |
| **Top-5 Accuracy** | 19.35% | **64.52%** | **+45.17% (3.3x)** |
| **Mean Confidence** | 59.90% | 44.00% | (More Calibrated) |

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

## 6. Dataset Preparation & DVC Usage

Datasets and checkpoints are managed by DVC. To fetch the data:
```bash
# Pull images and model weights
dvc pull
```
For detailed dataset specifications and updates, refer to [data/README.md](file:///d:/Dev/tankri_ocr_proj/data/README.md).

---

## 7. Execution Guide

Run scripts using the python runner:
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

Refer to the [examples/](file:///d:/Dev/tankri_ocr_proj/examples/) directory for specific shell (`.sh`) and PowerShell (`.ps1`) recipes.

---

## 8. Documentation

Comprehensive pipeline guides are located in the `docs/` folder:
* [Architecture Documentation](file:///d:/Dev/tankri_ocr_proj/docs/architecture.md)
* [Dataset & Preprocessing Documentation](file:///d:/Dev/tankri_ocr_proj/docs/dataset.md)
* [Training pipeline Documentation](file:///d:/Dev/tankri_ocr_proj/docs/training.md)
* [Evaluation suite Documentation](file:///d:/Dev/tankri_ocr_proj/docs/evaluation.md)
* [Synthetic Generation Documentation](file:///d:/Dev/tankri_ocr_proj/docs/synthetic_generation.md)
* [Progressive Domain Adaptation Documentation](file:///d:/Dev/tankri_ocr_proj/docs/domain_adaptation.md)
* [Notebook & MLflow Tracking Documentation](file:///d:/Dev/tankri_ocr_proj/docs/experiments.md)

---

## 9. Citation

If you build upon this work, please cite:
```bibtex
@software{tankri_ocr_2026,
  author = {Tankri OCR Contributors},
  title = {Tankri Handwritten OCR: Progressive Domain Adaptation and Synthetic Generation Pipeline},
  url = {https://github.com/yourusername/tankri-handwritten-ocr},
  version = {1.0.0},
  year = {2026}
}
```

---

## 10. License

This project is licensed under the MIT License. See [LICENSE](file:///d:/Dev/tankri_ocr_proj/LICENSE) for details.
