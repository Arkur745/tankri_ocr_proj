# Evaluation Pipeline

This document details the metrics, visualization tools, and standalone evaluation scripts.

## 1. Metrics & Logged Artifacts (`src/evaluation/evaluate.py`)

At the end of training, `log_evaluation_artifacts()` is executed to run predictions on the validation/test set and save performance reports:
* **Classification Report (`classification_report.csv`)**: Evaluates precision, recall, and f1-score globally.
* **Per-Class Metrics (`per_class_metrics.csv`)**: Lists precision, recall, and f1-score individually for all 35 characters.
* **Confusion Matrix Matrix Heatmap (`confusion_matrix.png`)**: Generates a custom seaborn heatmap highlighting misclassifications and character substitutions.
* **Training Curves (`training_curves.png`)**: Compiles loss and accuracy curves for training and validation sets over all epochs.
* **JSON Mappings (`label_to_idx.json`, `idx_to_label.json`)**: Persists dictionary files alongside the metrics for deployment consistency.

---

## 2. Standalone Evaluation Script (`scripts/evaluate_baseline.py`)

A command-line script to evaluate model performance on the test set:
* **Arguments**:
  * `--use_adapted`: Flag to load `best_domain_adapted_model.pth` instead of the baseline `best_model.pth`.
* **Execution flow**:
  1. Resolves and loads the correct model checkpoint from the candidates directory.
  2. Dynamically configures the model architecture wrapper (supporting adaptation MLP heads or default fc classifiers).
  3. Loads the official test dataset (`dataset/test/` with `labels/test_labels.csv`).
  4. Generates a terminal classification report and displays overall accuracy.

---

## 3. Statistical Rigor: Multi-Seed Ablations, Confidence Intervals & Calibration

For paper-grade reporting (multi-seed ablation mean ± std, Wilson confidence intervals and Expected Calibration Error on the small OOD wall-inscription test set, and the leakage-free 70/15/15 split these depend on), see [statistical_rigor.md](statistical_rigor.md).
