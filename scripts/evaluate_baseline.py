import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import warnings

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding='utf-8')

import src.configs.config as config
from src.utils.label_mapping import load_label_mapping
from src.dataset.loader import TankriDataset
from src.dataset.augmentation import val_transform_resnet

def run_evaluation(use_adapted=False):
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # If use_adapted is False, fallback to checking configs
    if not use_adapted:
        use_adapted = getattr(config, "ENABLE_DOMAIN_ADAPTATION", False)
        
    ckpt_name = "best_domain_adapted_model.pth" if use_adapted else "best_model.pth"

    print("==========================================================")
    print(f"Evaluating {('Adapted' if use_adapted else 'Baseline')} Model on Test Set")
    print("==========================================================")

    # 1. Load label mapping
    artifacts_dir = PROJECT_ROOT / "artifacts"
    label_to_idx, idx_to_label = load_label_mapping(artifacts_dir)
    num_classes = len(label_to_idx)
    target_names = [idx_to_label[i] for i in range(num_classes)]
    print(f"✔ Loaded label mappings. Number of classes: {num_classes}")

    # 2. Find and load the model checkpoint
    checkpoint_candidates = [
        PROJECT_ROOT / "notebooks" / "models" / ckpt_name,
        PROJECT_ROOT / "models" / ckpt_name,
        PROJECT_ROOT / "notebooks" / ckpt_name,
        PROJECT_ROOT / ckpt_name,
    ]
    checkpoint_path = None
    for cand in checkpoint_candidates:
        if cand.exists():
            checkpoint_path = cand
            break

    if checkpoint_path is None:
        print(f"❌ Error: No trained model checkpoint ({ckpt_name}) found. Please run training first.")
        sys.exit(1)

    print(f"✔ Found model checkpoint at: {checkpoint_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✔ Using device: {device}")

    # Load checkpoint state dict to inspect structure
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    
    # Detect adaptation head keys
    has_adaptation_head = any("fc.0" in k for k in state_dict.keys())
    if has_adaptation_head:
        print("✔ Checkpoint contains adaptation MLP head structure.")

    # 3. Instantiate model dynamically based on configuration
    if config.MODEL_NAME == "ResNet18":
        from src.models import ResNet18Model
        from unittest.mock import patch
        with patch.object(config, "ADD_ADAPTATION_HEAD", has_adaptation_head), \
             patch.object(config, "ENABLE_DOMAIN_ADAPTATION", has_adaptation_head or use_adapted):
            model = ResNet18Model(
                num_classes=num_classes,
                pretrained=config.PRETRAINED,
                dropout=config.DROPOUT,
                unfreeze_layer3=config.UNFREEZE_LAYER3,
                unfreeze_layer4=config.UNFREEZE_LAYER4,
            )
    elif config.MODEL_NAME == "SimpleCNN":
        from src.models import SimpleCNN
        model = SimpleCNN(
            num_classes=num_classes,
            dropout=config.DROPOUT,
        )
    else:
        print(f"❌ Error: Unknown model name '{config.MODEL_NAME}' in config.py.")
        sys.exit(1)

    # Load model weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print("✔ Model instantiated and weights loaded successfully.")

    # 4. Load the official test dataset
    test_labels_file = PROJECT_ROOT / "dataset" / "labels" / "test_labels.csv"
    test_images_dir = PROJECT_ROOT / "dataset" / "test"
    
    if not test_labels_file.exists():
        print(f"❌ Error: Test labels file not found at {test_labels_file}.")
        sys.exit(1)
        
    test_df = pd.read_csv(test_labels_file)
    
    val_dataset = TankriDataset(
        dataframe=test_df,
        image_dir=test_images_dir,
        label_to_idx=label_to_idx,
        transform=val_transform_resnet,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )
    print(f"✔ Loaded test dataset. Test samples: {len(val_dataset)}")

    # 5. Run Evaluation Loop
    correct_1 = 0
    correct_3 = 0
    correct_5 = 0
    total = 0
    confidences = []
    
    y_true = []
    y_pred = []

    print("\nRunning evaluation on test set...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            # Confidence of prediction
            max_probs, preds = torch.max(probs, dim=1)
            confidences.extend(max_probs.cpu().tolist())
            
            # Top-k predictions
            maxk = min(5, num_classes)
            _, topk_preds = outputs.topk(maxk, dim=1, largest=True, sorted=True)
            
            for i in range(labels.size(0)):
                target = labels[i].item()
                pred_list = topk_preds[i].cpu().tolist()
                
                if target == pred_list[0]:
                    correct_1 += 1
                if target in pred_list[:min(3, len(pred_list))]:
                    correct_3 += 1
                if target in pred_list[:min(5, len(pred_list))]:
                    correct_5 += 1
                    
                y_true.append(target)
                y_pred.append(pred_list[0])
                
            total += labels.size(0)

    # Calculate metrics
    top1_acc = correct_1 / total
    top3_acc = correct_3 / total
    top5_acc = correct_5 / total
    mean_confidence = np.mean(confidences)

    print("\n==========================================================")
    print("Baseline Metrics Summary:")
    print("==========================================================")
    print(f"  Top-1 Accuracy : {top1_acc * 100:.2f}% ({correct_1}/{total})")
    print(f"  Top-3 Accuracy : {top3_acc * 100:.2f}% ({correct_3}/{total})")
    print(f"  Top-5 Accuracy : {top5_acc * 100:.2f}% ({correct_5}/{total})")
    print(f"  Mean Confidence: {mean_confidence * 100:.2f}%")
    print("==========================================================")

    # 6. Confusion Matrix computation & plotting
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Ensure reports directory exists
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Plot and save confusion matrix
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=target_names, 
        yticklabels=target_names,
        annot_kws={"size": 7}
    )
    plt.title(f"Baseline Confusion Matrix ({config.MODEL_NAME})", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    cm_path = reports_dir / "baseline_confusion_matrix.png"
    
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"✔ Confusion matrix saved to: {cm_path}")

    # 7. Save metrics report JSON
    metrics_report = {
        "model_name": config.MODEL_NAME,
        "checkpoint_path": str(checkpoint_path),
        "total_samples": total,
        "top1_accuracy": top1_acc,
        "top3_accuracy": top3_acc,
        "top5_accuracy": top5_acc,
        "mean_confidence": mean_confidence,
    }
    
    report_path = reports_dir / "baseline_metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, ensure_ascii=False, indent=4)
    print(f"✔ Metrics report saved to: {report_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate baseline or domain adapted Tankri OCR model.")
    parser.add_argument("--use_adapted", action="store_true", help="Evaluate the domain-adapted model checkpoint.")
    args = parser.parse_args()
    
    run_evaluation(use_adapted=args.use_adapted)
