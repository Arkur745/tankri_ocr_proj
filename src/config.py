from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset
DATASET_DIR = PROJECT_ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_FILE = DATASET_DIR / "labels" / "labels.csv"

# Models
MODELS_DIR = PROJECT_ROOT / "models"

# Training
IMAGE_SIZE = 256
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 20

# Random Seed
RANDOM_SEED = 42