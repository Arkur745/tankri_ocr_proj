# Dataset & Preprocessing

This document describes the dataset structures, preprocessing pipelines, and augmentations used in the Tankri handwritten character recognition experiments.

## 1. Dataset Directory Structure

All image datasets are managed locally and versions are tracked via DVC.
* **Real Dataset (`dataset/`)**:
  * `dataset/images/`: 1,205 handwritten characters (35 glyph classes).
  * `dataset/test/`: Official evaluation/test set containing handwritten characters.
  * `dataset/labels/labels.csv`: Main label annotations for training images.
  * `dataset/labels/test_labels.csv`: Label annotations for test images.
* **Synthetic Dataset (`generated_dataset/`)**:
  * 9,000 synthetically rendered glyphs using modern augmentations and font renderers.
* **Hybrid Dataset (`generated_dataset_hybrid/`)**:
  * 1,205 real images combined with 2,000 balanced synthetic images to prevent over-representation.

---

## 2. Preprocessing & Cropping (`src/dataset/preprocessing.py`)

To improve classifier focus, characters are tightly cropped:
1. **Binarization**: Grayscale images are binarized using inverse thresholding.
2. **Bounding Box**: Bounding boxes of non-zero pixels are found using OpenCV `cv2.findNonZero`.
3. **Padding**: Bounding boxes are cropped tightly, and a uniform 20-pixel white border is added to prevent glyph boundaries from clashing with edge pooling.

---

## 3. Augmentations (`src/dataset/augmentation.py`)

A set of custom torchvision-compatible transforms are used:
* **Motion Blur**: Approximates camera/motion shake using `ImageFilter.BoxBlur` with a randomized radius.
* **JPEG Compression**: Simulates low-quality scanning artifacts by compressing image arrays into memory buffers using standard quality profiles.
* **Gaussian Noise**: Adds zero-mean Gaussian noise to raw tensors for sensor simulation.
* **Geometric Transforms**: Uses RandomAffine (rotations, translations, scaling, shears) and RandomPerspective.
