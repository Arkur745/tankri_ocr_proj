"""
Comprehensive OOD evaluation on wall inscription test set.
Compares pre-adaptation vs post-adaptation models.
Includes Wilson CIs, ECE, and reliability diagrams.
"""
import sys
import json
from pathlib import Path

import torch
import torchvision.transforms as T

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.label_mapping import load_label_mapping
from src.evaluation.ood_metrics import (
    load_ood_test_set,
    evaluate_ood_with_logits,
    compute_ood_metrics,
    compute_ece,
    plot_reliability_diagram,
    print_ood_comparison_table,
)
from src.models import ResNet18Model


def load_checkpoint(checkpoint_path, model, device):
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model


def main():
    """
    Evaluate both baseline and domain-adapted models on OOD wall inscription set.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup paths
    dataset_dir = PROJECT_ROOT / "dataset"
    test_labels_file = dataset_dir / "labels" / "test_labels.csv"
    test_images_dir = dataset_dir / "test"
    artifacts_dir = PROJECT_ROOT / "artifacts"
    results_dir = PROJECT_ROOT / "reports" / "ood_evaluation"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load label mapping
    label_to_idx, idx_to_label = load_label_mapping(artifacts_dir)
    num_classes = len(label_to_idx)
    print(f"Loaded {num_classes} class labels")

    # Load OOD test set with verified image ordering
    print(f"\nLoading wall inscription test set from {test_labels_file}")
    image_paths, true_labels = load_ood_test_set(
        test_labels_file,
        test_images_dir,
        label_to_idx,
    )
    print(f"✓ Loaded {len(image_paths)} test images (wall inscriptions)")
    print(f"  Image order verified for before/after consistency")

    # Save image list for verification
    image_list_file = results_dir / "wall_inscription_image_list.json"
    with open(image_list_file, 'w') as f:
        json.dump({
            'images': [str(p.name) for p in image_paths],
            'n_images': len(image_paths),
        }, f, indent=2)
    print(f"✓ Saved image list to {image_list_file} for verification")

    # Image transform: reuse val_transform_resnet for consistency with
    # training/validation preprocessing (Resize -> Grayscale(3ch) -> ToTensor -> Normalize).
    # A locally-redefined transform here previously omitted the Grayscale step.
    from src.dataset.augmentation import val_transform_resnet
    transform = val_transform_resnet

    # ========================================================================
    # EVALUATE BASELINE MODEL (BEFORE ADAPTATION)
    # ========================================================================
    print("\n" + "="*80)
    print("EVALUATING BASELINE MODEL (BEFORE DOMAIN ADAPTATION)")
    print("="*80)

    # NOTE (2026-08-21 repo cleanup): models/best_model.pth used to be a
    # different, unverified checkpoint, while notebooks/models/best_model.pth
    # was the confirmed-correct one -- verified by exactly reproducing the
    # paper's historical OOD numbers (Top-1 6.45%, Top-3/5 19.35%, confidence
    # 59.90%) once the label-index bug was also fixed. The checkpoints have
    # since been consolidated: models/best_model.pth now IS that
    # verified-correct checkpoint (see CHANGELOG.md), so the standard path is
    # used directly.
    baseline_checkpoint = PROJECT_ROOT / "models" / "best_model.pth"
    if not baseline_checkpoint.exists():
        print(f"❌ Error: Baseline checkpoint not found at {baseline_checkpoint}")
        return

    model_baseline = ResNet18Model(
        num_classes=num_classes,
        pretrained=True,
        dropout=0.0,
    ).to(device)
    model_baseline = load_checkpoint(baseline_checkpoint, model_baseline, device)
    print(f"✓ Loaded baseline model from {baseline_checkpoint}")

    # Evaluate
    results_before = evaluate_ood_with_logits(
        model_baseline,
        device,
        image_paths,
        true_labels,
        transform,
        return_all_logits=True,
    )
    metrics_before = compute_ood_metrics(results_before)

    print("\nBaseline Model Results:")
    print(f"  Top-1 Accuracy: {metrics_before['top1']['accuracy']:.2%} ({metrics_before['top1']['correct']}/{metrics_before['n_samples']})")
    print(f"    95% CI: [{metrics_before['top1']['lower_ci']:.2%}, {metrics_before['top1']['upper_ci']:.2%}]")
    print(f"  Top-3 Accuracy: {metrics_before['top3']['accuracy']:.2%} ({metrics_before['top3']['correct']}/{metrics_before['n_samples']})")
    print(f"    95% CI: [{metrics_before['top3']['lower_ci']:.2%}, {metrics_before['top3']['upper_ci']:.2%}]")
    print(f"  Top-5 Accuracy: {metrics_before['top5']['accuracy']:.2%} ({metrics_before['top5']['correct']}/{metrics_before['n_samples']})")
    print(f"    95% CI: [{metrics_before['top5']['lower_ci']:.2%}, {metrics_before['top5']['upper_ci']:.2%}]")
    print(f"  Mean Confidence: {metrics_before['mean_confidence']:.2%}")

    # Compute ECE for baseline
    print("\nComputing ECE for baseline model...")
    ece_before = compute_ece(
        true_labels,
        results_before['confidences'],
        logits=results_before['logits'],
        n_bins=10,
    )
    print(f"  Expected Calibration Error: {ece_before['ece']:.4f}")
    print("  Per-bin statistics:")
    for bin_stat in ece_before['bin_stats']:
        if bin_stat['n_samples'] > 0:
            print(f"    Bin [{bin_stat['bin_range'][0]:.2f}, {bin_stat['bin_range'][1]:.2f}]: "
                  f"n={bin_stat['n_samples']:2d}, acc={bin_stat['accuracy']:.2%}, conf={bin_stat['confidence']:.2%}, "
                  f"cal_err={bin_stat['calibration_error']:.4f} {bin_stat['quality']}")

    # Plot reliability diagram
    reliability_plot_before = results_dir / "reliability_diagram_before_adaptation.png"
    plot_reliability_diagram(true_labels, results_before['confidences'], ece_before, reliability_plot_before)

    # ========================================================================
    # EVALUATE ADAPTED MODEL (AFTER ADAPTATION)
    # ========================================================================
    print("\n" + "="*80)
    print("EVALUATING DOMAIN-ADAPTED MODEL (AFTER ADAPTATION)")
    print("="*80)

    adapted_checkpoint = PROJECT_ROOT / "models" / "best_domain_adapted_model.pth"
    if not adapted_checkpoint.exists():
        print(f"⚠️  Warning: Domain-adapted checkpoint not found at {adapted_checkpoint}")
        print("    Skipping post-adaptation evaluation.")
        return

    model_adapted = ResNet18Model(
        num_classes=num_classes,
        pretrained=True,
        dropout=0.0,
    ).to(device)
    model_adapted = load_checkpoint(adapted_checkpoint, model_adapted, device)
    print(f"✓ Loaded adapted model from {adapted_checkpoint}")

    # Evaluate (using SAME image order and paths)
    results_after = evaluate_ood_with_logits(
        model_adapted,
        device,
        image_paths,  # SAME image list
        true_labels,  # SAME true labels
        transform,
        return_all_logits=True,
    )
    metrics_after = compute_ood_metrics(results_after)

    print("\nAdapted Model Results:")
    print(f"  Top-1 Accuracy: {metrics_after['top1']['accuracy']:.2%} ({metrics_after['top1']['correct']}/{metrics_after['n_samples']})")
    print(f"    95% CI: [{metrics_after['top1']['lower_ci']:.2%}, {metrics_after['top1']['upper_ci']:.2%}]")
    print(f"  Top-3 Accuracy: {metrics_after['top3']['accuracy']:.2%} ({metrics_after['top3']['correct']}/{metrics_after['n_samples']})")
    print(f"    95% CI: [{metrics_after['top3']['lower_ci']:.2%}, {metrics_after['top3']['upper_ci']:.2%}]")
    print(f"  Top-5 Accuracy: {metrics_after['top5']['accuracy']:.2%} ({metrics_after['top5']['correct']}/{metrics_after['n_samples']})")
    print(f"    95% CI: [{metrics_after['top5']['lower_ci']:.2%}, {metrics_after['top5']['upper_ci']:.2%}]")
    print(f"  Mean Confidence: {metrics_after['mean_confidence']:.2%}")

    # Compute ECE for adapted
    print("\nComputing ECE for adapted model...")
    ece_after = compute_ece(
        true_labels,
        results_after['confidences'],
        logits=results_after['logits'],
        n_bins=10,
    )
    print(f"  Expected Calibration Error: {ece_after['ece']:.4f}")
    print("  Per-bin statistics:")
    for bin_stat in ece_after['bin_stats']:
        if bin_stat['n_samples'] > 0:
            print(f"    Bin [{bin_stat['bin_range'][0]:.2f}, {bin_stat['bin_range'][1]:.2f}]: "
                  f"n={bin_stat['n_samples']:2d}, acc={bin_stat['accuracy']:.2%}, conf={bin_stat['confidence']:.2%}, "
                  f"cal_err={bin_stat['calibration_error']:.4f} {bin_stat['quality']}")

    # Plot reliability diagram
    reliability_plot_after = results_dir / "reliability_diagram_after_adaptation.png"
    plot_reliability_diagram(true_labels, results_after['confidences'], ece_after, reliability_plot_after)

    # ========================================================================
    # PRINT COMPARISON TABLE (TABLE VI)
    # ========================================================================
    print("\n" + "="*80)
    print("BEFORE/AFTER COMPARISON (TABLE VI)")
    print("="*80)

    table_file = results_dir / "table_vi_ood_comparison.txt"
    print_ood_comparison_table(metrics_before, metrics_after, table_file)

    # ========================================================================
    # SAVE DETAILED RESULTS
    # ========================================================================
    detailed_results = {
        'wall_inscription_test_set': {
            'n_images': len(image_paths),
            'image_list': [str(p.name) for p in image_paths],
        },
        'baseline_model': {
            'checkpoint': str(baseline_checkpoint.relative_to(PROJECT_ROOT)),
            'metrics': metrics_before,
            'ece': ece_before['ece'],
            'ece_bin_stats': ece_before['bin_stats'],
        },
        'adapted_model': {
            'checkpoint': str(adapted_checkpoint.relative_to(PROJECT_ROOT)),
            'metrics': metrics_after,
            'ece': ece_after['ece'],
            'ece_bin_stats': ece_after['bin_stats'],
        },
    }

    results_json = results_dir / "ood_evaluation_results.json"
    with open(results_json, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"\n✓ Saved detailed results to {results_json}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("OOD EVALUATION SUMMARY")
    print("="*80)
    print(f"✓ Evaluated {len(image_paths)} wall inscription test images")
    print(f"✓ Used identical image set for before/after comparison")
    print(f"\nBaseline Model:")
    print(f"  Top-1: {metrics_before['top1']['accuracy']:.2%} ± Wilson CI [{metrics_before['top1']['lower_ci']:.2%}, {metrics_before['top1']['upper_ci']:.2%}]")
    print(f"  ECE: {ece_before['ece']:.4f}")
    print(f"\nAdapted Model:")
    print(f"  Top-1: {metrics_after['top1']['accuracy']:.2%} ± Wilson CI [{metrics_after['top1']['lower_ci']:.2%}, {metrics_after['top1']['upper_ci']:.2%}]")
    print(f"  ECE: {ece_after['ece']:.4f}")
    print(f"\nImprovement:")
    top1_improvement = metrics_after['top1']['accuracy'] - metrics_before['top1']['accuracy']
    ece_improvement = ece_before['ece'] - ece_after['ece']
    print(f"  Top-1: +{top1_improvement:.2%} absolute")
    if metrics_before['top1']['accuracy'] > 0:
        print(f"         {metrics_after['top1']['accuracy']/metrics_before['top1']['accuracy']:.1f}x relative improvement")
    print(f"  ECE: {ece_improvement:+.4f}")
    print(f"\nOutput files:")
    print(f"  {table_file}")
    print(f"  {results_json}")
    print(f"  {reliability_plot_before}")
    print(f"  {reliability_plot_after}")
    print("="*80)


if __name__ == "__main__":
    main()
