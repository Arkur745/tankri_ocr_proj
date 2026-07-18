# ==============================================================================
# Tankri Handwritten OCR PowerShell Command Recipes
# Guide demonstrating how to run each component of the pipeline in PowerShell.
# ==============================================================================

Write-Host "=== Tankri OCR Workflow Recipes ===" -ForegroundColor Green

# 1. Dataset Initialization & DVC Pull
# Fetch image files and checkpoints from DVC remote caches:
# dvc pull

# 2. Synthetic Dataset Generation
Write-Host "`n[1] Generating synthetic characters..." -ForegroundColor Yellow
# python scripts/generate_sample.py

# 3. Hybrid Dataset Construction
Write-Host "`n[2] Building hybrid dataset..." -ForegroundColor Yellow
# python scripts/create_hybrid_dataset.py

# 4. Running Baseline Model Training
# Configure training parameters (e.g. ENABLE_DOMAIN_ADAPTATION=False) in src/configs/config.py,
# then run the cells in notebooks/experiments.ipynb.

# 5. Domain Adaptation Training
# Set ENABLE_DOMAIN_ADAPTATION=True in src/configs/config.py,
# then run the cells in notebooks/experiments.ipynb.

# 6. Model Evaluation
Write-Host "`n[3] Evaluating baseline model on test set..." -ForegroundColor Yellow
python scripts/evaluate_baseline.py

Write-Host "`n[4] Evaluating domain-adapted model on test set..." -ForegroundColor Yellow
python scripts/evaluate_baseline.py --use_adapted

# 7. Pipeline Verification
Write-Host "`n[5] Verifying pipeline preprocessing & logits consistency..." -ForegroundColor Yellow
python scripts/verify_pipeline.py

# 8. Start Streamlit Interactive Applications
# Launch the main OCR Recognition application:
# streamlit run app.py
# Launch the dataset label annotator utility:
# streamlit run annotator.py
