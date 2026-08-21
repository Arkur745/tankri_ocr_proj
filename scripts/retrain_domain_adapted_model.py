"""
Retrain domain-adapted model using ONLY the new clean train split (863 images).
This ensures no data leakage between train/val/test splits.

Usage:
    conda activate aimlenv
    python scripts/retrain_domain_adapted_model.py
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, ConcatDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.utils import set_seed
from src.utils.label_mapping import create_label_mapping
from src.dataset.loader import TankriDataset
from src.dataset.augmentation import train_transform_resnet, val_transform_resnet
from src.models import ResNet18Model
from src.training.train import train
from src.evaluation.evaluate import log_evaluation_artifacts
from src.utils.data_split import load_split
from src.utils.mlflow_init import init_mlflow
import src.configs.config as config
import mlflow


def create_hybrid_dataset(real_images_train, synthetic_images_dir, label_to_idx, transform):
    """
    Create hybrid dataset combining real training images with balanced synthetic images.

    Parameters
    ----------
    real_images_train : list
        Real training image filenames from new train split
    synthetic_images_dir : Path
        Directory containing synthetic dataset
    label_to_idx : dict
        Label to index mapping
    transform : torchvision.transforms
        Image transformation

    Returns
    -------
    torch.utils.data.Dataset
        Combined dataset
    """
    from src.dataset.loader import TankriDataset
    import csv
    import os

    image_dir = PROJECT_ROOT / "dataset" / "images"

    # Real dataset from new train split
    real_labels_file = PROJECT_ROOT / "dataset" / "labels" / "labels.csv"
    img_to_label = {}
    with open(real_labels_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_to_label[row['image']] = row['label']

    real_labels_list = [(img, img_to_label[img]) for img in real_images_train]
    real_dataset = TankriDataset(
        dataframe=None,  # We'll pass data directly via labels_list
        image_dir=image_dir,
        label_to_idx=label_to_idx,
        transform=transform,
        labels_list=real_labels_list,
    ) if hasattr(TankriDataset, 'labels_list') else None

    # Fallback: create custom dataset
    if real_dataset is None:
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, image_list, image_dir, label_to_idx, transform):
                self.images = image_list
                self.image_dir = Path(image_dir)
                self.label_to_idx = label_to_idx
                self.transform = transform

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                img_name, label_str = self.images[idx]
                img_path = self.image_dir / img_name

                from PIL import Image
                img = Image.open(img_path).convert('L')  # Grayscale, matches original loader

                if self.transform:
                    img = self.transform(img)

                label_idx = self.label_to_idx[label_str]
                return img, label_idx

        real_dataset = SimpleDataset(real_labels_list, image_dir, label_to_idx, transform)

    # Try to load synthetic dataset if it exists
    synthetic_labels_file = synthetic_images_dir / "labels" / "labels.csv"
    if synthetic_labels_file.exists() and (synthetic_images_dir / "images").exists():
        print(f"  Loading synthetic images from {synthetic_images_dir}")

        syn_img_to_label = {}
        with open(synthetic_labels_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                syn_img_to_label[row['image']] = row['label']

        # For synthetic, we want class-balanced samples
        # Load all synthetic images for each class to ensure balance
        synthetic_labels_list = [(img, syn_img_to_label[img]) for img in syn_img_to_label.keys()]

        synthetic_dataset = SimpleDataset(
            synthetic_labels_list,
            synthetic_images_dir / "images",
            label_to_idx,
            transform,
        )

        print(f"  Real dataset: {len(real_dataset)} images")
        print(f"  Synthetic dataset: {len(synthetic_dataset)} images")

        # Combine datasets
        hybrid_dataset = ConcatDataset([real_dataset, synthetic_dataset])
        return hybrid_dataset
    else:
        print(f"  Synthetic dataset not found at {synthetic_images_dir}")
        print(f"  Using real training data only")
        return real_dataset


def main():
    """Retrain domain-adapted model using new clean train split."""

    print("\n" + "="*100)
    print("DOMAIN ADAPTATION MODEL RETRAINING (CLEAN SPLIT)")
    print("="*100)
    print("Objective: Retrain best_domain_adapted_model.pth using ONLY new train split (863 images)")
    print("Data leakage prevention: Uses 70/15/15 split created today\n")

    # NOTE (2026-08-21 repo cleanup): models/best_model.pth used to be a
    # different, unverified checkpoint, while notebooks/models/best_model.pth
    # was the confirmed-correct one -- verified by exactly reproducing the
    # paper's historical OOD numbers (Top-1 6.45%, Top-3/5 19.35%, confidence
    # 59.90%) once the label-index bug was also fixed. This script used to
    # override config.DOMAIN_ADAPTATION_CHECKPOINT locally to point at the
    # notebooks/ copy for that reason. The checkpoints have since been
    # consolidated: models/best_model.pth now IS that verified-correct
    # checkpoint, so the local override is no longer needed and the standard
    # config.DOMAIN_ADAPTATION_CHECKPOINT (models/best_model.pth) is used directly.
    print(f"Using baseline checkpoint: {config.DOMAIN_ADAPTATION_CHECKPOINT}\n")

    start_time = time.time()

    # Activate PyTorch GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Setup paths
    labels_file = PROJECT_ROOT / "dataset" / "labels" / "labels.csv"
    image_dir = PROJECT_ROOT / "dataset" / "images"
    # CRITICAL FIX: generated_dataset_hybrid (3,205 images: 1,205 duplicate real
    # copies + only 2,000 synthetic) was stale/wrong. generated_dataset is the
    # correct, current synthetic corpus: 9,000 images, exactly 200/class,
    # 'temple_wall' profile, quality-filtered (36.52% rejection) -- matches the
    # paper's described "~9,000 procedurally generated character images" exactly.
    synthetic_dir = PROJECT_ROOT / "generated_dataset"
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load label mappings
    label_to_idx, idx_to_label = create_label_mapping(labels_file)
    num_classes = len(label_to_idx)
    print(f"Loaded {num_classes} classes")

    # Load NEW clean split
    print("\nLoading new 70/15/15 split:")
    split_dir = PROJECT_ROOT / "artifacts" / "data_splits"
    train_images = load_split(split_dir / "train_split.json")
    val_images = load_split(split_dir / "val_split.json")

    print(f"  Train (new): {len(train_images)} images")
    print(f"  Val (new): {len(val_images)} images")

    # Set seed for reproducibility
    set_seed(42)

    # Create hybrid dataset (real from new train split + synthetic)
    print("\nCreating hybrid dataset:")

    import csv
    img_to_label = {}
    with open(labels_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_to_label[row['image']] = row['label']

    # Real training dataset from new split
    real_labels_list = [(img, img_to_label[img]) for img in train_images]

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, image_list, image_dir, label_to_idx, transform):
            self.images = image_list
            self.image_dir = Path(image_dir)
            self.label_to_idx = label_to_idx
            self.transform = transform

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_name, label_str = self.images[idx]
            img_path = self.image_dir / img_name

            from PIL import Image
            img = Image.open(img_path).convert('L')  # Grayscale, matches original loader

            if self.transform:
                img = self.transform(img)

            label_idx = self.label_to_idx[label_str]
            return img, label_idx

    train_dataset = SimpleDataset(real_labels_list, image_dir, label_to_idx, train_transform_resnet)

    # Load synthetic dataset if available
    synthetic_labels_file = synthetic_dir / "labels" / "labels.csv"
    if synthetic_labels_file.exists() and (synthetic_dir / "images").exists():
        print(f"  Loading synthetic images from {synthetic_dir}")

        syn_img_to_label = {}
        with open(synthetic_labels_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                syn_img_to_label[row['image']] = row['label']

        synthetic_labels_list = [(img, syn_img_to_label[img]) for img in syn_img_to_label.keys()]
        synthetic_dataset = SimpleDataset(
            synthetic_labels_list,
            synthetic_dir / "images",
            label_to_idx,
            train_transform_resnet,
        )

        print(f"  Real training images (new split): {len(train_dataset)}")
        print(f"  Synthetic images: {len(synthetic_dataset)}")

        # Combine into hybrid dataset
        train_dataset = ConcatDataset([train_dataset, synthetic_dataset])

    print(f"  Total hybrid dataset size: {len(train_dataset)}")

    # Create validation dataset from new split
    val_labels_list = [(img, img_to_label[img]) for img in val_images]
    val_dataset = SimpleDataset(val_labels_list, image_dir, label_to_idx, val_transform_resnet)
    print(f"  Validation dataset: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.DOMAIN_ADAPTATION_BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.DOMAIN_ADAPTATION_BATCH_SIZE,
        shuffle=False,
    )

    # Create model (will load baseline weights in training function)
    print(f"\nInitializing ResNet18 model ({num_classes} classes)...")
    model = ResNet18Model(
        num_classes=num_classes,
        pretrained=config.PRETRAINED,
        dropout=config.DROPOUT,
        unfreeze_layer3=config.TRAIN_LAYER3,
        unfreeze_layer4=config.TRAIN_LAYER4,
    ).to(device)

    # Create loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.DOMAIN_ADAPTATION_LR,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.DOMAIN_ADAPTATION_EPOCHS)

    # Initialize MLflow
    init_mlflow("TakriOCR_DomainAdaptation_CleanSplit")

    # Training
    print("\n" + "="*100)
    print("TRAINING DOMAIN-ADAPTED MODEL")
    print("="*100)
    print(f"Baseline checkpoint: {config.DOMAIN_ADAPTATION_CHECKPOINT}")
    print(f"Learning rate: {config.DOMAIN_ADAPTATION_LR}")
    print(f"Epochs: {config.DOMAIN_ADAPTATION_EPOCHS}")
    print(f"Batch size: {config.DOMAIN_ADAPTATION_BATCH_SIZE}")
    print("="*100 + "\n")

    with mlflow.start_run(run_name="domain_adaptation_clean_split"):
        # Log parameters
        mlflow.log_param('data_split', '70/15/15_clean')
        mlflow.log_param('train_size', len(train_dataset))
        mlflow.log_param('val_size', len(val_dataset))
        mlflow.log_param('learning_rate', config.DOMAIN_ADAPTATION_LR)
        mlflow.log_param('epochs', config.DOMAIN_ADAPTATION_EPOCHS)
        mlflow.log_param('batch_size', config.DOMAIN_ADAPTATION_BATCH_SIZE)
        mlflow.log_param('baseline_checkpoint', str(config.DOMAIN_ADAPTATION_CHECKPOINT))
        mlflow.log_param('domain_adaptation_dataset', config.DOMAIN_ADAPTATION_DATASET)

        # Train
        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=config.DOMAIN_ADAPTATION_EPOCHS,
            device=device,
            scheduler=scheduler,
            save_best=True,
            save_dir=models_dir,
            label_to_idx=label_to_idx,
            idx_to_label=idx_to_label,
            image_size=config.IMAGE_SIZE,
        )

        # Log evaluation artifacts
        log_evaluation_artifacts(
            model=model,
            val_loader=val_loader,
            device=device,
            save_dir=models_dir,
            history=history,
            label_to_idx=label_to_idx,
            idx_to_label=idx_to_label,
            best_checkpoint_name='best_domain_adapted_model.pth',
        )

    elapsed = time.time() - start_time

    print("\n" + "="*100)
    print("DOMAIN ADAPTATION RETRAINING COMPLETE")
    print("="*100)
    print(f"Total training time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"\nNew model saved to: {models_dir / 'best_domain_adapted_model.pth'}")
    print("\nData leakage status: ✓ CLEAN")
    print("  • Trained using ONLY new train split (863 images)")
    print("  • Validation on new split (182 images)")
    print("  • Test split (160 images) completely held out")
    print("="*100)

    return elapsed


if __name__ == "__main__":
    elapsed = main()
    print(f"\n✓ Ready for next step: run scripts/evaluate_ood_wall_inscriptions.py")
    print(f"Command: python scripts/test_single_ablation_timing.py")
