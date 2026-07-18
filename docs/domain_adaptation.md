# Progressive Domain Adaptation

This document describes the Progressive Domain Adaptation / Transfer Learning method designed to expand the baseline model's representations using synthetic and hybrid datasets.

## 1. Low-Resource & Domain Gap Problem

Handwritten Tankri characters feature unique stroke variances, pressure gradients, and ink bleed patterns.
* **Overfitting**: Training a deep network directly on a small real handwritten dataset (1,205 images) leads to rapid overfitting and poor generalization.
* **Transfer Learning**: To solve this, we pre-train the baseline model on the real dataset, then perform transfer learning / Progressive Domain Adaptation on the **9,000 synthetic images** or the balanced **hybrid dataset** (1,205 real + 2,000 synthetic images) to expand its feature space and improve accuracy.

---

## 2. Progressive Unfreezing Strategy

To combat catastrophic forgetting while adapting representation layers, a progressive unfreezing policy is implemented:
* **Feature Extractors**: Early layers (`conv1`, `bn1`, `layer1`, `layer2`) capture low-level strokes. These are kept **frozen** (`FREEZE_BACKBONE = True`) to preserve general feature extraction weights.
* **Semantic Layers**: Deep semantic layer blocks (`layer3`, `layer4`) capture character glyph structures. These are selectively **fine-tuned** (`TRAIN_LAYER4 = True`) using small learning rates.
* **Classifier**: The final fully-connected linear layer is fine-tuned (`TRAIN_CLASSIFIER = True`) to match target domains.

---

## 3. Classifier Fine-Tuning Strategies

* **Preserving Pre-trained Classifier (`ADD_ADAPTATION_HEAD = False`)**:
  * Keeps the baseline classification weights intact and performs minor adjustments. This is the optimal configuration for fine-tuning on a hybrid dataset to avoid representation collapse.
* **MLP Adaptation Head (`ADD_ADAPTATION_HEAD = True`)**:
  * Replaces the final classification layer with a multi-layer perceptron (e.g. 512 $\rightarrow$ 256 $\rightarrow$ 35). This is suitable when mapping to an entirely different distribution, but resets baseline classifier weights.

---

## 4. Fine-Tuning on Hybrid Dataset

To prevent representation collapse, the model is fine-tuned on the **Hybrid Dataset** (1,205 real images + 2,000 balanced synthetic images).
* Learning Rate: Set to a small value (e.g., `1e-5` to `5e-5`) to avoid catastrophic weight displacement.
* Training duration: Limited to `15` to `25` epochs with early stopping.
