# Changelog

All notable changes to this project will be documented in this file.

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
