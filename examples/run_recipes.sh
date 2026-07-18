#!/bin/bash

# ==============================================================================
# Tankri Handwritten OCR Command Recipes
# Shell guide demonstrating how to run each component of the pipeline.
# ==============================================================================

# Exit on error
set -e

echo "=== Tankri OCR Workflow Recipes ==="

# ------------------------------------------------------------------------------
# 1. Dataset Initialization & DVC Pull
# ------------------------------------------------------------------------------
# Fetch image files and checkpoints from DVC remote caches:
# dvc pull

# ------------------------------------------------------------------------------
# 2. Synthetic Dataset Generation
# ------------------------------------------------------------------------------
# Generate procedural characters with augmentations:
echo "Recipe: Generating synthetic characters..."
# python scripts/generate_sample.py

# ------------------------------------------------------------------------------
# 3. Hybrid Dataset Construction
# ------------------------------------------------------------------------------
# Construct the hybrid dataset (1205 real + 2000 balanced synthetic images):
echo "Recipe: Building hybrid dataset..."
# python scripts/create_hybrid_dataset.py

# ------------------------------------------------------------------------------
# 4. Running Baseline Model Training (via Jupyter Notebook)
# ------------------------------------------------------------------------------
# Configure training parameters (e.g. ENABLE_DOMAIN_ADAPTATION=False) in src/configs/config.py,
# then run the cells in notebooks/experiments.ipynb.

# ------------------------------------------------------------------------------
# 5. Domain Adaptation Training
# ------------------------------------------------------------------------------
# Set ENABLE_DOMAIN_ADAPTATION=True in src/configs/config.py,
# then run the cells in notebooks/experiments.ipynb to fine-tune the ResNet18 model.

# ------------------------------------------------------------------------------
# 6. Model Evaluation
# ------------------------------------------------------------------------------
# Evaluate the baseline model:
echo "Recipe: Evaluating baseline model on test set..."
python scripts/evaluate_baseline.py

# Evaluate the domain-adapted model:
echo "Recipe: Evaluating domain-adapted model on test set..."
python scripts/evaluate_baseline.py --use_adapted

# ------------------------------------------------------------------------------
# 7. Pipeline Verification
# ------------------------------------------------------------------------------
# Verify that preprocessing and inference predictions match perfectly:
echo "Recipe: Verifying pipeline preprocessing & logits consistency..."
python scripts/verify_pipeline.py

# ------------------------------------------------------------------------------
# 8. Start Streamlit Interactive Applications
# ------------------------------------------------------------------------------
# Launch the main OCR Recognition application:
echo "Recipe: Launching main Streamlit OCR recognition app..."
# streamlit run app.py

# Launch the dataset label annotator utility:
echo "Recipe: Launching Streamlit annotator app..."
# streamlit run annotator.py
