# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Leakage-free 70/15/15 stratified train/val/test split (`src/utils/data_split.py`, `artifacts/data_splits/`), replacing the earlier 80/20 split where the model-selection and headline-reporting sets overlapped.
- Multi-seed ablation runner (`src/training/multi_seed_ablation.py`) and results aggregator (`src/utils/results_aggregator.py`) reporting mean ± std across 5 seeds for the E00–E05 ablation configurations.
- OOD evaluation module (`src/evaluation/ood_metrics.py`, `scripts/evaluate_ood_wall_inscriptions.py`) with Wilson score 95% confidence intervals, raw counts, Expected Calibration Error, and reliability diagrams for the wall-inscription test set.
- `docs/statistical_rigor.md` documenting the above methodology.

### Fixed
- `.gitignore` had an unanchored `models/` rule that was silently excluding `src/models/` (the ResNet18/SimpleCNN architecture source) from version control.
- `requirements.txt` declared `pytorch` (an unrelated placeholder package) instead of `torch`, and was missing `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `tqdm`, and `dvc`.
- Removed a stray tracked duplicate of `mlflow.db` under `notebooks/`.
- Scrubbed local machine paths/usernames from committed notebook outputs.
- Corrected stale documentation in `synthetic_generator/README.md` referencing a `src/config.py`-driven configuration interface that no longer exists (the generator is now configured via function arguments, not config flags).
- Fixed `README.md`'s Results Summary, which reported OOD wall-inscription Top-1/3/5 accuracy of 22.58%/54.84%/64.52% from a since-superseded checkpoint (`notebooks/models/best_domain_adapted_model.pth`, trained under the old 80/20 split, flagged for data-leakage risk in the split-cleanup work above). The verified numbers from the retrained, leakage-free checkpoint are 19.35%/35.48%/48.39%; the in-domain validation baseline (89.23% ± 0.95%, 5-seed mean) was also added, having previously been omitted entirely. Cross-checked directly against the manuscript source (`latex/main.tex`), not just prompted values.
- Consolidated the baseline checkpoint: `models/best_model.pth` (canonical location) previously held a different, unverified checkpoint (best guess: an earlier training run from around 2026-07-06/07, based on git history and MLflow run timing — best validation accuracy 89.21% on the old 80/20 split of the 1,205-image set, epoch 27/50) than `notebooks/models/best_model.pth`, which several scripts had already identified and hardcoded as "the confirmed-correct baseline checkpoint" for reproducing the paper's OOD numbers. `models/best_model.pth` is now that verified-correct checkpoint (the old file was backed up outside the repo before being overwritten, not discarded). `scripts/retrain_domain_adapted_model.py` and `scripts/evaluate_ood_wall_inscriptions.py` no longer need their local path overrides as a result; `scripts/evaluate_baseline.py`, `scripts/verify_pipeline.py`, `scripts/check_label_mapping.py`, and `app.py` no longer prefer `notebooks/models/` over `models/` when searching for a checkpoint.
- `scripts/evaluate_baseline.py`: removed a silent override where evaluating "the baseline" (no `--use_adapted` flag) would actually evaluate the domain-adapted model whenever `config.ENABLE_DOMAIN_ADAPTATION` was `True`; output filenames/titles (`reports/baseline_*` vs `reports/domain_adapted_*`) now reflect which model was actually evaluated instead of always saying "baseline"; checkpoint paths written to output JSON are now relative instead of embedding the local machine path.

## [1.0.0] - 2026-07-18

### Added
- Created hybrid dataset (1,205 real + 2,000 synthetic images) under `generated_dataset_hybrid/`.
- Progressive domain adaptation pipeline with unfreezing (`conv1`, `bn1`, `layer1`, `layer2` frozen; `layer3`, `layer4` and classification head fine-tuned).
- Streamlit application (`app.py`) for live prediction with model selection (Baseline vs adapted).
- Streamlit data annotator tool (`annotator.py`).
- Standalone CLI validation script (`scripts/verify_pipeline.py`).
- Standalone CLI baseline evaluation script (`scripts/evaluate_baseline.py`).
- Complete research and architecture documentation under `docs/`.

### Changed
- Reorganized codebase into clean package layout under `src/`.
- Standardized config module as single source of truth (`src/configs/config.py`).
- Integrated dynamic learning rate and epochs overrides when domain adaptation mode is activated.
- Restructured and cleaned commented-out code in all transforms and training utilities.
