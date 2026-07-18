# Experimental Pipeline

This document describes the experimental workflow, notebooks, and MLflow tracking integration.

## 1. Notebook Workflow (`notebooks/experiments.ipynb`)

All primary research experiments are recorded in `experiments.ipynb`. The notebook cells are designed to run in a step-by-step manner:
1. **Config Import**: Loads global variables from `src.configs.config`.
2. **Dataset Load**: Imports the target labels dataframe and maps classes to integer indices.
3. **Data Splitting**: Performs stratified splits to create balanced training and validation sub-samples.
4. **Data Loaders**: Initializes PyTorch dataloaders for ResNet18 and SimpleCNN using configured augmentations.
5. **Model Initialization**: Dynamically initializes SimpleCNN or ResNet18Model.
6. **Training Loop**: Calls the `train()` function under the active MLflow run context.
7. **Evaluation**: Calls `log_evaluation_artifacts()` to compute performance metrics and save visual heatmaps.

---

## 2. Tracking Dashboard (MLflow)

All runs are recorded to the local database file `mlflow.db`.
To view the experiment dashboards:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
This launches a browser interface at `http://localhost:5000` to:
* Compare loss curves and validation accuracies across baseline and domain adapted runs.
* Verify hyperparameter values (learning rates, batch sizes, unfreezing options).
* Download saved artifact files (confusion matrix PNGs, classification reports, configuration snapshots, models).
