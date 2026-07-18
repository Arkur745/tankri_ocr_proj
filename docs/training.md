# Training Pipeline

This document describes the training execution, optimization, and checkpoint logging systems.

## 1. Core Training Interface (`src/training/train.py`)

The `train()` function executes the standard epoch-based training loops:
* **Train Loop**: Computes losses, gradients, executes optimizer steps, and logs metrics.
* **Validation Loop**: Runs model in evaluation mode (`model.eval()`) under `torch.no_grad()`.
* **Cosine Learning Rate Decay**: Schedulers step at the end of each epoch to scale learning rates.
* **Early Checkpoint Auto-Save**: Keeps track of validation accuracies and saves local files `best_model.pth` and `last_model.pth`.

---

## 2. Optimization Details

* **Optimizer**: Adam optimizer with custom default learning rates (baseline `1e-4`, domain adaptation `1e-5` / `5e-5`).
* **Loss Function**: `nn.CrossEntropyLoss` with label smoothing enabled via config.
* **Learning Rate Scheduler**: `CosineAnnealingLR` decay decaying the rate over the maximum epoch count.

---

## 3. Dynamic Domain Adaptation Mode

When `ENABLE_DOMAIN_ADAPTATION = True`, the training module automatically triggers overrides:
1. **Checkpoint Loading**: Loads baseline weights from `best_model.pth` into the ResNet18 model.
2. **Selective Parameter Freezing**: Freezes the early convolution layers via `configure_trainable_layers` based on configuration options (`FREEZE_BACKBONE`, `TRAIN_LAYER3`, `TRAIN_LAYER4`).
3. **Optimizer & Scheduler Re-binding**: Dynamically creates a fresh Adam optimizer passing *only* parameters with `requires_grad = True` and binds the custom domain adaptation learning rate. A fresh scheduler is also initialized.
4. **Checkpoint Filenames**: Saves checkpoints as `best_domain_adapted_model.pth` and `last_domain_adapted_model.pth`.

---

## 4. Experiment Logging (MLflow)

All training metadata is logged automatically to the SQLite tracking server `mlflow.db`:
* Parameters: Optimizer type, learning rates, epochs, image sizes, frozen/trainable layer details, loss function configurations, and seeds.
* Metrics: Step-wise training loss, training accuracy, validation loss, and validation accuracy.
* Artifacts: `label_to_idx.json`, `idx_to_label.json`, `config_snapshot.json`, `augmentation_config.txt`, and PyTorch weight binaries.
