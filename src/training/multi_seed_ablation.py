"""
Multi-seed ablation study runner for TakriOCR.
Runs E00-E05 configurations with 5 different random seeds.
"""
import sys
import json
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import mlflow

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.utils import set_seed
from src.utils.label_mapping import create_label_mapping, save_label_mapping
from src.dataset.loader import TankriDataset
from src.dataset.augmentation import train_transform_resnet, val_transform_resnet, baseline_transform_resnet
from src.models import ResNet18Model
from src.training.train import train
from src.evaluation.evaluate import log_evaluation_artifacts
from src.utils.data_split import load_split
import src.configs.config as config


SEEDS = [42, 123, 2024, 7, 99]

ABLATION_CONFIGS = {
    'E00': {
        'name': 'Baseline',
        'description': 'Standard ResNet18 baseline',
        'dropout': 0.0,
        'label_smoothing': 0.0,
        'unfreeze_layer3': False,
        'unfreeze_layer4': True,
        'aug_mode': 'none',
    },
    'E01': {
        'name': 'OCR Augmentation',
        'description': 'Apply AugOCR (affine, noise, blur, JPEG)',
        'dropout': 0.0,
        'label_smoothing': 0.0,
        'unfreeze_layer3': False,
        'unfreeze_layer4': True,
        'aug_mode': 'ocr',
    },
    'E02': {
        'name': 'Label Smoothing',
        'description': 'Apply label smoothing (epsilon=0.1)',
        'dropout': 0.0,
        'label_smoothing': 0.1,
        'unfreeze_layer3': False,
        'unfreeze_layer4': True,
        'aug_mode': 'none',
    },
    'E03': {
        'name': 'Classifier Dropout',
        'description': 'Apply dropout before classifier (p=0.2)',
        'dropout': 0.2,
        'label_smoothing': 0.0,
        'unfreeze_layer3': False,
        'unfreeze_layer4': True,
        'aug_mode': 'none',
    },
    'E04': {
        'name': 'Progressive Unfreezing',
        'description': 'Unfreeze layer3 + layer4 (no augmentation)',
        'dropout': 0.0,
        'label_smoothing': 0.0,
        'unfreeze_layer3': True,
        'unfreeze_layer4': True,
        'aug_mode': 'none',
    },
    'E05': {
        'name': 'Combined Model',
        'description': 'AugOCR + Progressive Unfreezing (optimal)',
        'dropout': 0.0,
        'label_smoothing': 0.0,
        'unfreeze_layer3': True,
        'unfreeze_layer4': True,
        'aug_mode': 'ocr',
    },
}


def get_augmentation_transform(aug_mode):
    """Get the appropriate augmentation transform based on mode.

    'none' means no OCR-specific augmentation (baseline geometric
    RandomAffine only), NOT literally zero augmentation.
    """
    if aug_mode == 'ocr':
        return train_transform_resnet
    else:
        return baseline_transform_resnet


def run_ablation_with_seed(
    exp_id,
    ablation_config,
    seed,
    device,
    split_dict,
    labels_file,
    image_dir,
    models_dir,
):
    """
    Run a single ablation configuration with a specific seed.

    Parameters
    ----------
    exp_id : str
        Experiment ID (E00-E05)
    ablation_config : dict
        Configuration dictionary for this ablation
    seed : int
        Random seed
    device : torch.device
        Compute device
    split_dict : dict
        Dictionary with 'train', 'val', 'test' image lists
    labels_file : Path
        Path to labels CSV
    image_dir : Path
        Path to image directory
    models_dir : Path
        Path to models directory

    Returns
    -------
    dict
        Results dictionary with metrics
    """
    print(f"\n{'='*80}")
    print(f"Running {exp_id} ({ablation_config['name']}) with seed={seed}")
    print(f"{'='*80}")

    set_seed(seed)

    # Load label mappings
    label_to_idx, idx_to_label = create_label_mapping(labels_file)
    num_classes = len(label_to_idx)

    # Create datasets using the split
    # For training, use the 'train' split with augmentation
    import csv
    img_to_label = {}
    with open(labels_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_to_label[row['image']] = row['label']

    # Filter images for train/val using the split
    train_labels_list = [(img, img_to_label[img]) for img in split_dict['train']]
    val_labels_list = [(img, img_to_label[img]) for img in split_dict['val']]

    # Get augmentation transform
    # 'none' means no OCR-specific augmentation (baseline geometric RandomAffine
    # only), NOT literally zero augmentation -- confirmed against original design
    # intent and historical MLflow runs.
    train_transform = get_augmentation_transform(ablation_config['aug_mode'])

    # Create datasets with subset of data
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
            # CRITICAL: Match original TankriDataset behavior
            # Load as grayscale ('L' mode), NOT RGB
            img = Image.open(img_path).convert('L')

            if self.transform:
                img = self.transform(img)

            label_idx = self.label_to_idx[label_str]
            return img, label_idx

    train_dataset = SimpleDataset(train_labels_list, image_dir, label_to_idx, train_transform)
    val_dataset = SimpleDataset(val_labels_list, image_dir, label_to_idx, val_transform_resnet)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    # Create model
    model = ResNet18Model(
        num_classes=num_classes,
        pretrained=config.PRETRAINED,
        dropout=ablation_config['dropout'],
        unfreeze_layer3=ablation_config['unfreeze_layer3'],
        unfreeze_layer4=ablation_config['unfreeze_layer4'],
    ).to(device)

    # Create criterion with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=ablation_config['label_smoothing'])

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
    )

    # Create scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    # Create MLflow run
    run_name = f"{exp_id}_seed{seed}"
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param('exp_id', exp_id)
        mlflow.log_param('exp_name', ablation_config['name'])
        mlflow.log_param('random_seed', seed)
        mlflow.log_param('dropout', ablation_config['dropout'])
        mlflow.log_param('label_smoothing', ablation_config['label_smoothing'])
        mlflow.log_param('unfreeze_layer3', ablation_config['unfreeze_layer3'])
        mlflow.log_param('unfreeze_layer4', ablation_config['unfreeze_layer4'])
        mlflow.log_param('aug_mode', ablation_config['aug_mode'])
        mlflow.log_param('learning_rate', config.LEARNING_RATE)
        mlflow.log_param('batch_size', config.BATCH_SIZE)
        mlflow.log_param('epochs', config.EPOCHS)
        mlflow.log_param('image_size', config.IMAGE_SIZE)
        mlflow.log_param('optimizer', 'Adam')
        mlflow.log_param('scheduler', 'CosineAnnealingLR')
        mlflow.log_param('model_architecture', 'ResNet18')
        mlflow.log_param('pretrained', config.PRETRAINED)
        mlflow.log_param('num_classes', num_classes)
        mlflow.log_param('train_size', len(train_dataset))
        mlflow.log_param('val_size', len(val_dataset))

        # Set tags
        mlflow.set_tag('ablation_id', exp_id)
        mlflow.set_tag('seed', seed)

        # Train model
        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=config.EPOCHS,
            device=device,
            scheduler=scheduler,
            save_best=True,
            save_dir=models_dir / f"{exp_id}_seed{seed}",
            label_to_idx=label_to_idx,
            idx_to_label=idx_to_label,
        )

        # Log evaluation artifacts
        save_dir = models_dir / f"{exp_id}_seed{seed}"
        log_evaluation_artifacts(
            model=model,
            val_loader=val_loader,
            device=device,
            save_dir=save_dir,
            history=history,
            label_to_idx=label_to_idx,
            idx_to_label=idx_to_label,
            best_checkpoint_name='best_model.pth',
        )

        # Extract validation accuracy from history
        final_val_acc = history['val_accuracy'][-1]
        best_val_acc = max(history['val_accuracy'])

        # Also compute test accuracy for final reporting
        from src.evaluation.evaluate import evaluate
        save_dir_path = models_dir / f"{exp_id}_seed{seed}"
        best_model_path = save_dir_path / 'best_model.pth'

        test_images_list = split_dict['test']
        test_labels_list = [(img, img_to_label[img]) for img in test_images_list]
        test_dataset = SimpleDataset(test_labels_list, image_dir, label_to_idx, val_transform_resnet)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        # Load best checkpoint and evaluate on test set
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        else:
            test_acc = 0.0
            test_loss = 0.0

        # Log metrics to MLflow
        mlflow.log_metric('final_val_accuracy', final_val_acc)
        mlflow.log_metric('best_val_accuracy', best_val_acc)
        mlflow.log_metric('test_accuracy', test_acc)
        mlflow.log_metric('test_loss', test_loss)

        print(f"\n✓ Completed {exp_id} with seed {seed}")
        print(f"  Val Acc ({len(val_dataset)} images): {best_val_acc:.4f} [model selection]")
        print(f"  Test Acc ({len(test_dataset)} images): {test_acc:.4f} [held-out evaluation]")

        return {
            'exp_id': exp_id,
            'seed': seed,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
        }


def main():
    """Run multi-seed ablation study."""
    # ========================================================================
    # CRITICAL: DISABLE DOMAIN ADAPTATION FOR ABLATION STUDY
    # ========================================================================
    # The config has ENABLE_DOMAIN_ADAPTATION=True globally, which overrides
    # learning rate and epochs to domain adaptation values (1e-5 LR, 25 epochs).
    # For the ablation study, we need the BASELINE hyperparameters:
    # Learning rate 1e-4, 50 epochs (from lines 28-29 of config.py)
    # This override is SCOPED LOCALLY to this ablation script only.
    import src.configs.config as config_module
    original_enable_da = config_module.ENABLE_DOMAIN_ADAPTATION
    config_module.ENABLE_DOMAIN_ADAPTATION = False
    # Recalculate the hyperparameters without domain adaptation override
    config_module.EPOCHS = 50
    config_module.LEARNING_RATE = 1e-4
    config_module.BATCH_SIZE = 32
    print("\n" + "="*80)
    print("ABLATION STUDY INITIALIZATION")
    print("="*80)
    print("⚠️  DOMAIN ADAPTATION TEMPORARILY DISABLED FOR ABLATION")
    print(f"  Original ENABLE_DOMAIN_ADAPTATION: {original_enable_da}")
    print(f"  Ablation LR: {config_module.LEARNING_RATE} (overrides DA's 1e-5)")
    print(f"  Ablation Epochs: {config_module.EPOCHS} (overrides DA's 25)")
    print(f"  Will be restored at end of run")
    print("="*80 + "\n")

    # Setup paths
    # CRITICAL: Ablation study (E00-E05) stays on the ORIGINAL V1 benchmark
    # dataset (~1,060 images), per confirmed design intent. The expanded
    # 1,205-image dataset (V1 + 145 real-world additions) is reserved for
    # domain adaptation / OOD evaluation only, NOT this ablation study.
    labels_file = PROJECT_ROOT / "dataset" / "labels" / "labels_v1_benchmark.csv"
    image_dir = PROJECT_ROOT / "dataset" / "images"
    models_dir = PROJECT_ROOT / "models" / "ablation_study"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load split (clean stratified 70/15/15 split of the V1 benchmark dataset)
    split_dir = PROJECT_ROOT / "artifacts" / "data_splits_v1_benchmark"
    split_dict = {}
    for split_name in ['train', 'val', 'test']:
        split_file = split_dir / f"{split_name}_split.json"
        split_dict[split_name] = load_split(split_file)

    print(f"\nLoaded data splits:")
    print(f"  Train: {len(split_dict['train'])} images")
    print(f"  Val: {len(split_dict['val'])} images")
    print(f"  Test: {len(split_dict['test'])} images")

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Initialize MLflow
    from src.utils.mlflow_init import init_mlflow
    init_mlflow("TakriOCR_MultiSeed_Ablation")

    # Run all configurations with all seeds
    all_results = []

    exp_ids = list(ABLATION_CONFIGS.keys())
    print(f"\nRunning {len(exp_ids)} ablation configurations × {len(SEEDS)} seeds = {len(exp_ids) * len(SEEDS)} total runs")
    print(f"Experiment IDs: {', '.join(exp_ids)}")
    print(f"Seeds: {', '.join(map(str, SEEDS))}")

    for exp_id in exp_ids:
        ablation_config = ABLATION_CONFIGS[exp_id]
        for seed in SEEDS:
            try:
                result = run_ablation_with_seed(
                    exp_id=exp_id,
                    ablation_config=ablation_config,
                    seed=seed,
                    device=device,
                    split_dict=split_dict,
                    labels_file=labels_file,
                    image_dir=image_dir,
                    models_dir=models_dir,
                )
                all_results.append(result)
            except Exception as e:
                print(f"\n❌ Error in {exp_id} with seed {seed}: {e}")
                import traceback
                traceback.print_exc()

    # Save results summary
    results_file = models_dir / "ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved results to {results_file}")

    print("\n" + "="*80)
    print("MULTI-SEED ABLATION STUDY COMPLETE")
    print("="*80)

    # RESTORE DOMAIN ADAPTATION FLAG (for subsequent phases)
    config_module.ENABLE_DOMAIN_ADAPTATION = original_enable_da
    config_module.EPOCHS = config.DOMAIN_ADAPTATION_EPOCHS if original_enable_da else 50
    config_module.LEARNING_RATE = config.DOMAIN_ADAPTATION_LR if original_enable_da else 1e-4
    print(f"\n✓ Restored ENABLE_DOMAIN_ADAPTATION to {original_enable_da}")
    print(f"  (Ready for Phase 3: OOD evaluation)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
