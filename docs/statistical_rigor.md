# Statistical Rigor: Splits, Multi-Seed Ablations, and OOD Metrics

This document describes the statistical methodology added on top of the core training/evaluation pipeline (see [training.md](training.md) and [evaluation.md](evaluation.md)) to support rigorous, reviewer-defensible reporting: a leakage-free three-way split, multi-seed ablation reporting, and calibration/confidence-interval reporting for the small out-of-distribution (OOD) test set.

## 1. Clean 70/15/15 Train/Val/Test Split (`src/utils/data_split.py`)

Earlier experiments used an 80/20 train/val split and reported ablation results on the same validation set used for model selection — meaning the "headline" number and the number used to pick the best configuration were not independent.

`create_stratified_split()` produces a stratified, class-balanced 70/15/15 split of the 1,205-image real dataset:
* **Train (863 images, 71.6%)** — model fitting.
* **Val (182 images, 15.1%)** — model/hyperparameter selection across ablation configs.
* **Test (160 images, 13.3%)** — held out for the final reported headline numbers only.

Splits are saved as JSON (`artifacts/data_splits/{train,val,test}_split.json`) for reproducibility, and re-loadable via `load_split()`. `print_split_distribution()` flags any class with fewer than 3 samples in val or test (expected for a handful of the rarer classes given the dataset's long tail — currently 7 of 45 classes).

`artifacts/data_splits_v1_benchmark/` (via `scripts/create_v1_benchmark_split.py`) holds an equivalent split restricted to the original ≤1060-index "V1" benchmark images, used to keep the E00–E05 ablation study on the same benchmark the original single-run baseline numbers were computed on, separate from the expanded 1,205-image set used for domain adaptation.

## 2. Multi-Seed Ablation Reporting (`src/training/multi_seed_ablation.py`, `src/utils/results_aggregator.py`)

Single-run ablation numbers overstate confidence — a single seed can't distinguish a real effect from training-run noise. `multi_seed_ablation.py` reruns each of 6 ablation configurations (E00 baseline through E05 combined) across 5 seeds (`42, 123, 2024, 7, 99`), logging every run to MLflow with `ablation_id` and `seed` tags, and records both validation accuracy (model selection) and test accuracy (headline reporting) per run.

`results_aggregator.py` then queries MLflow, groups by ablation ID, and reports **mean ± std** across seeds instead of a single value, with the `seed=42` run highlighted separately as a sanity check against any earlier single-run results.

## 3. OOD Evaluation: Wilson Confidence Intervals & ECE (`src/evaluation/ood_metrics.py`, `scripts/evaluate_ood_wall_inscriptions.py`)

The out-of-distribution wall-inscription test set is small (31 images), so raw percentages alone are misleading — e.g. the project's actual after-adaptation Top-1 result, `19.35% (6/31)`, has a 95% CI of `[9.19%, 36.28%]` — a wide enough range that a bare percentage overstates precision. `ood_metrics.py` reports:

* **Wilson score 95% confidence intervals** (`wilson_score_ci`) rather than a normal approximation, since Wilson CIs stay well-behaved at small `n` and at the extremes (0 or all successes). Metrics are reported as e.g. `19.35% (6/31) [9.19%, 36.28%]`.
* **Expected Calibration Error (ECE)** (`compute_ece`), a 10-bin histogram calibration metric, replacing the earlier qualitative-only confidence discussion. Bins with fewer than 3 samples are explicitly flagged as low-confidence rather than silently included.
* **Reliability diagrams** (`plot_reliability_diagram`) — calibration curves plotting model confidence against empirical accuracy per bin, with low-sample bins marked.

`scripts/evaluate_ood_wall_inscriptions.py` runs the full before/after comparison (baseline vs. domain-adapted model) on a verified-identical 31-image order, and writes:
* `reports/ood_evaluation/table_vi_ood_comparison.txt` — the paper's Table VI.
* `reports/ood_evaluation/ood_evaluation_results.json` — full metrics detail.
* `reports/ood_evaluation/reliability_diagram_{before,after}_adaptation.png`.
* `reports/ood_evaluation/wall_inscription_image_list.json` — the exact verified image order, for auditability.

## 4. Data-Leakage Note

The domain-adapted checkpoint must be trained (or retrained, via `scripts/retrain_domain_adapted_model.py`) on the **new** 70/15/15 `train_split.json` before its OOD numbers are reported — a checkpoint trained under the old 80/20 split may have seen images that now fall in the new held-out test split, which would invalidate the Fix-4 leakage guarantee above for that specific checkpoint.

## 5. Running the Pipeline

```bash
# 1. (Re)create the stratified split
python src/utils/data_split.py

# 2. Run the 6-config x 5-seed ablation grid (30 runs)
python src/training/multi_seed_ablation.py

# 3. Aggregate results into mean ± std tables
python src/utils/results_aggregator.py

# 4. Evaluate baseline vs. adapted model on the OOD wall-inscription set
python scripts/evaluate_ood_wall_inscriptions.py
```
