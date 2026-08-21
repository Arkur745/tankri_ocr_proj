# Dataset Specifications & DVC Data Management

This directory documentation details how data is managed and version-tracked in this repository.

## 1. DVC Configuration

The actual image directories are managed locally by **DVC (Data Version Control)** to keep large image/checkpoint binaries out of the Git history.
* Tracked directories:
  * `dataset/` (contains raw training, test, and annotation files) $\rightarrow$ Tracked by `dataset.dvc`
  * `models/` (contains trained PyTorch checkpoint weights) $\rightarrow$ Tracked by `models.dvc`

**No DVC remote is configured, and none is public.** `dataset.dvc`/`models.dvc` are committed as a record of what's version-tracked and their content hashes — not as a working `dvc pull` path. There is currently no way to fetch `dataset/` or `models/` content from this repository; see [Section 3](#3-licensing--provenance-of-manuscript-derived-images) for why. `dvc add`/`dvc push` remain useful for the maintainer's own local versioning, but do not imply public availability.

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

---

## 3. Licensing & Provenance of Manuscript-Derived Images

**The `dataset/` images are currently private and not publicly released**, pending confirmation of the rights described below. There is no announced release date.

The `dataset/` images are tightly cropped single-character glyphs extracted from photographs of handwritten and inscribed Tankri-script source material (including temple wall inscriptions). These are historical and, in some cases, cultural-heritage materials, and the copyright/ownership status of the underlying source manuscripts and inscriptions has not yet been fully confirmed — the right to redistribute derived crops may depend on factors (site/institutional ownership, local heritage law, photographer permissions) that are still being verified. The dataset is kept private specifically until that is resolved; this page will be updated if and when that changes.

**The trained model checkpoints (`models/*.pth`) are also currently private.** This is being treated as a separate, independent decision from the dataset's release status, not an automatic consequence of it — redistributing trained weights derived from copyrighted source material raises different questions than redistributing the source material itself. No decision has been made to release the checkpoints; it may be revisited on its own timeline regardless of what happens with the raw dataset.

Everything else in this repository — code, model architecture (the untrained network definitions), training/evaluation methodology, and the result artifacts in `reports/` and `outputs/sample_tuning/` — is public and does not depend on the dataset or checkpoints being released.

If you are the rights-holder for any of the source material, or have a research collaboration inquiry, please open an issue on this repository or contact the maintainer.
