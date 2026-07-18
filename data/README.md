# Dataset Specifications & DVC Data Management

This directory documentation details how data is managed and version-tracked in this repository.

## 1. DVC Configuration

The actual image directories are managed by **DVC (Data Version Control)** to prevent committing large image files to the Git history.
* Tracked directories:
  * `dataset/` (contains raw training, test, and annotation files) $\rightarrow$ Tracked by `dataset.dvc`
  * `models/` (contains trained PyTorch checkpoint weights) $\rightarrow$ Tracked by `models.dvc`

### Fetching Data
To retrieve image directories and checkpoints from remote storage (e.g. Google Drive, S3, or shared directory):
```bash
dvc pull
```

### Staging Local Changes
If you expand the dataset or train new baseline checkpoints, update DVC and commit trackers to git:
```bash
dvc add dataset
dvc add models
git add dataset.dvc models.dvc
git commit -m "Update DVC tracked datasets and model checkpoints"
dvc push
```

---

## 2. Directory Configurations

* **Real dataset (`dataset/`)**:
  * `images/`: Grayscale character scans.
  * `test/`: Unseen test scans for evaluation.
  * `labels/labels.csv`: Main label dictionary mappings.
* **Synthetic dataset (`generated_dataset/`)**:
  * Procedurally generated images matching target categories.
* **Hybrid dataset (`generated_dataset_hybrid/`)**:
  * Merged training dataset combining 1,205 real images and 2,000 balanced synthetic samples.
