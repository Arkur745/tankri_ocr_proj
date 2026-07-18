from pathlib import Path

# ==========================================================
# Path Configuration
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_FILE = DATASET_DIR / "labels" / "labels.csv"
MODELS_DIR = PROJECT_ROOT / "models"

# ==========================================================
# Active Experiment Configuration (Single Source of Truth)
# ==========================================================
EXPERIMENT_NAME = "Synthetic_Data_Generation_v1  c"
EXPERIMENT_DESCRIPTION = """
Evaluates the impact of camera-specific domain simulation on Tankri OCR generalization. Building upon the OCR augmentation baseline, this experiment introduces realistic image acquisition artifacts, including motion blur, JPEG compression, and sensor noise, to better simulate real-world photographs captured using mobile devices. The objective is to improve robustness to domain shift while preserving in-distribution validation performance.
"""
DATASET_VERSION = "v1.0"

# Model Architecture
MODEL_NAME = "ResNet18"  # "ResNet18" or "SimpleCNN"
PRETRAINED = True

# Training Hyperparameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50

# Regularization & Data Augmentation
LABEL_SMOOTHING = 0.0
DROPOUT = 0.0
MIXUP_ALPHA = 0.0  # MixUp is deactivated if <= 0.0

# Layer Freezing/Unfreezing
UNFREEZE_LAYER3 = True
UNFREEZE_LAYER4 = True

# Reproducibility
RANDOM_SEED = 42

# Image Renaming Target
# Set to "train" to rename images in dataset/images
# Set to "test" to rename images in dataset/test
RENAME_TARGET = "test"

# ==========================================================
# Progressive Synthetic Domain Adaptation Configuration
# ==========================================================
ENABLE_DOMAIN_ADAPTATION = True
DOMAIN_ADAPTATION_CHECKPOINT = MODELS_DIR / "best_model.pth"
DOMAIN_ADAPTATION_DATASET = "hybrid"
DOMAIN_ADAPTATION_EPOCHS = 25
DOMAIN_ADAPTATION_LR = 1e-5
DOMAIN_ADAPTATION_BATCH_SIZE = 32
FREEZE_BACKBONE = True
TRAIN_LAYER3 = False
TRAIN_LAYER4 = True
TRAIN_CLASSIFIER = True
ADD_ADAPTATION_HEAD = False
ADAPTATION_HEAD_DIM = 256

if ENABLE_DOMAIN_ADAPTATION:
    EPOCHS = DOMAIN_ADAPTATION_EPOCHS
    LEARNING_RATE = DOMAIN_ADAPTATION_LR
    BATCH_SIZE = DOMAIN_ADAPTATION_BATCH_SIZE
    
    if DOMAIN_ADAPTATION_DATASET == "synthetic":
        DATASET_DIR = PROJECT_ROOT / "generated_dataset"
    elif DOMAIN_ADAPTATION_DATASET == "hybrid":
        DATASET_DIR = PROJECT_ROOT / "generated_dataset_hybrid"
    else:
        DATASET_DIR = PROJECT_ROOT / "dataset"
        
    IMAGES_DIR = DATASET_DIR / "images"
    LABELS_FILE = DATASET_DIR / "labels" / "labels.csv"