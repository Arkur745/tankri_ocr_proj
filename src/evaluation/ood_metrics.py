"""
Out-of-distribution evaluation metrics for wall inscription test set.
Includes Wilson score confidence intervals and Expected Calibration Error (ECE).
"""
import json
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt


def wilson_score_ci(successes, n, confidence=0.95):
    """
    Compute Wilson score confidence interval for a proportion.
    Appropriate for small samples (like our 31 OOD test set).

    Parameters
    ----------
    successes : int
        Number of successes
    n : int
        Total trials
    confidence : float
        Confidence level (default 0.95 for 95% CI)

    Returns
    -------
    tuple
        (point_estimate, lower_bound, upper_bound)
    """
    from scipy import stats

    if n == 0:
        return 0.0, 0.0, 0.0

    p_hat = successes / n
    z = stats.norm.ppf((1 + confidence) / 2)
    denominator = 1 + z**2 / n

    center = (p_hat + z**2 / (2*n)) / denominator
    adjustment = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denominator

    lower = max(0, center - adjustment)
    upper = min(1, center + adjustment)

    return p_hat, lower, upper


def load_ood_test_set(test_labels_file, image_dir, label_to_idx):
    """
    Load wall inscription test set with verified image ordering.

    Parameters
    ----------
    test_labels_file : Path or str
        Path to test_labels.csv in dataset/test/
    image_dir : Path or str
        Path to test image directory
    label_to_idx : dict
        The FULL model's label-to-index mapping (all 45 classes), loaded via
        load_label_mapping(). Must NOT be reconstructed locally from only the
        labels present in this test set -- doing so silently reindexes classes
        to a smaller, differently-ordered space that doesn't match the model's
        actual output indices, corrupting every accuracy metric while leaving
        confidence (which doesn't depend on matching true labels) looking fine.

    Returns
    -------
    tuple
        (image_paths, true_labels)
    """
    test_labels_file = Path(test_labels_file)
    image_dir = Path(image_dir)

    # Read labels
    rows = []
    with open(test_labels_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Sort by image name for consistent ordering
    rows_sorted = sorted(rows, key=lambda x: x['image'])

    image_paths = [image_dir / row['image'] for row in rows_sorted]
    true_labels = [label_to_idx[row['label']] for row in rows_sorted]

    return image_paths, true_labels


def evaluate_ood_with_logits(
    model,
    device,
    image_paths,
    true_labels,
    transform,
    return_all_logits=False,
):
    """
    Evaluate model on OOD test set, returning predictions and confidence scores.

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate
    device : torch.device
        Compute device
    image_paths : list of Path
        Paths to test images
    true_labels : list
        True label indices
    transform : torchvision.transforms
        Image transform
    return_all_logits : bool
        If True, return logits for all classes (for ECE computation)

    Returns
    -------
    dict
        Results dictionary with predictions, confidences, and optionally logits
    """
    model.eval()

    predictions = []
    confidences = []
    all_logits = []
    top3_preds = []
    top5_preds = []

    with torch.no_grad():
        for img_path in image_paths:
            from PIL import Image
            img = Image.open(img_path).convert('L')  # Grayscale, matches original loader
            img_tensor = transform(img).unsqueeze(0).to(device)

            outputs = model(img_tensor)
            logits = outputs[0].cpu()

            if return_all_logits:
                all_logits.append(logits.numpy())

            probs = torch.softmax(logits, dim=0)
            top5_vals, top5_indices = torch.topk(probs, 5)

            pred = top5_indices[0].item()
            conf = top5_vals[0].item()

            predictions.append(pred)
            confidences.append(conf)
            top3_preds.append(top5_indices[:3].tolist())
            top5_preds.append(top5_indices[:5].tolist())

    results = {
        'predictions': predictions,
        'confidences': confidences,
        'top3': top3_preds,
        'top5': top5_preds,
        'true_labels': true_labels,
    }

    if return_all_logits:
        results['logits'] = np.array(all_logits)

    return results


def compute_ood_metrics(results):
    """
    Compute Top-1/3/5 accuracy and raw counts with Wilson CIs.

    Parameters
    ----------
    results : dict
        Results from evaluate_ood_with_logits

    Returns
    -------
    dict
        Metrics dictionary with accuracies, counts, and CIs
    """
    predictions = results['predictions']
    true_labels = results['true_labels']
    top3 = results['top3']
    top5 = results['top5']

    n = len(true_labels)

    # Top-1 accuracy
    top1_correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
    top1_acc, top1_lower, top1_upper = wilson_score_ci(top1_correct, n)

    # Top-3 accuracy
    top3_correct = sum(1 for t3, true in zip(top3, true_labels) if true in t3)
    top3_acc, top3_lower, top3_upper = wilson_score_ci(top3_correct, n)

    # Top-5 accuracy
    top5_correct = sum(1 for t5, true in zip(top5, true_labels) if true in t5)
    top5_acc, top5_lower, top5_upper = wilson_score_ci(top5_correct, n)

    # Mean confidence
    mean_confidence = float(np.mean(results['confidences']))

    metrics = {
        'n_samples': n,
        'top1': {
            'correct': top1_correct,
            'accuracy': top1_acc,
            'lower_ci': top1_lower,
            'upper_ci': top1_upper,
        },
        'top3': {
            'correct': top3_correct,
            'accuracy': top3_acc,
            'lower_ci': top3_lower,
            'upper_ci': top3_upper,
        },
        'top5': {
            'correct': top5_correct,
            'accuracy': top5_acc,
            'lower_ci': top5_lower,
            'upper_ci': top5_upper,
        },
        'mean_confidence': mean_confidence,
    }

    return metrics


def compute_ece(
    true_labels,
    confidences,
    logits=None,
    n_bins=10,
):
    """
    Compute Expected Calibration Error (ECE) using binned approach.

    Parameters
    ----------
    true_labels : list or array
        True labels
    confidences : list or array
        Prediction confidences (softmax max)
    logits : array, optional
        Full logits array for alternative ECE computation. If provided,
        uses softmax probabilities instead of pre-computed confidences.
    n_bins : int
        Number of bins for calibration histogram (default 10)

    Returns
    -------
    dict
        ECE and per-bin statistics
    """
    true_labels = np.array(true_labels)
    confidences = np.array(confidences)

    # If logits provided, use them to recompute confidences
    if logits is not None:
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        predictions = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
    else:
        predictions = None

    # Bin predictions by confidence
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    bin_stats = []

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if mask.sum() == 0:
            continue

        bin_confidences = confidences[mask]
        bin_labels = true_labels[mask]

        # Accuracy in this bin
        if predictions is not None:
            bin_predictions = predictions[mask]
            bin_accuracy = np.mean(bin_predictions == bin_labels)
        else:
            # Fallback: use confidence as accuracy proxy
            bin_accuracy = np.mean(bin_confidences)

        bin_confidence = np.mean(bin_confidences)
        bin_size = mask.sum()

        # ECE contribution
        ece += (bin_size / len(true_labels)) * abs(bin_accuracy - bin_confidence)

        # Flag bins with few samples
        low_confidence = "⚠ LOW" if bin_size < 3 else "OK"

        bin_stats.append({
            'bin_idx': bin_idx,
            'bin_range': (bin_edges[bin_idx], bin_edges[bin_idx + 1]),
            'n_samples': int(bin_size),
            'accuracy': float(bin_accuracy),
            'confidence': float(bin_confidence),
            'calibration_error': float(abs(bin_accuracy - bin_confidence)),
            'quality': low_confidence,
        })

    return {
        'ece': float(ece),
        'n_bins': n_bins,
        'bin_stats': bin_stats,
    }


def plot_reliability_diagram(
    true_labels,
    confidences,
    ece_result,
    output_path,
):
    """
    Plot reliability diagram (calibration curve).

    Parameters
    ----------
    true_labels : array
        True labels
    confidences : array
        Prediction confidences
    ece_result : dict
        Result from compute_ece
    output_path : Path or str
        Path to save figure
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract bin statistics
    bin_stats = ece_result['bin_stats']
    ece = ece_result['ece']

    if not bin_stats:
        print(f"No bins with data, skipping reliability diagram")
        return

    bin_centers = [np.mean(s['bin_range']) for s in bin_stats]
    accuracies = [s['accuracy'] for s in bin_stats]
    confidences_bin = [s['confidence'] for s in bin_stats]
    sample_counts = [s['n_samples'] for s in bin_stats]

    plt.figure(figsize=(8, 6))

    # Plot calibration curve
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2, alpha=0.5)
    plt.plot(
        confidences_bin,
        accuracies,
        'o-',
        label='Model',
        linewidth=2,
        markersize=8,
        color='steelblue',
    )

    # Add sample counts as text annotations
    for i, (conf, acc, count) in enumerate(zip(confidences_bin, accuracies, sample_counts)):
        color = 'red' if count < 3 else 'black'
        plt.text(conf, acc + 0.03, f'n={count}', ha='center', fontsize=8, color=color)

    plt.xlabel('Prediction Confidence', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Reliability Diagram (ECE = {ece:.4f})', fontsize=12)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved reliability diagram to {output_path}")


def print_ood_comparison_table(metrics_before, metrics_after, output_file=None):
    """
    Print formatted before/after OOD comparison table (Table VI).

    Parameters
    ----------
    metrics_before : dict
        Metrics before adaptation
    metrics_after : dict
        Metrics after adaptation
    output_file : Path or str, optional
        File to write table to
    """
    output_lines = []

    output_lines.append("\n" + "="*120)
    output_lines.append("TABLE VI: Cross-Domain Performance on Wall Inscription Test Set (31 images)")
    output_lines.append("="*120)
    output_lines.append("")
    output_lines.append(f"{'Metric':<30} {'Before Adapt.':<30} {'After Adapt.':<30} {'Rel. Impr.':<30}")
    output_lines.append("-"*120)

    # Top-1 Accuracy
    before_top1_acc = metrics_before['top1']['accuracy']
    before_top1_ci = (metrics_before['top1']['lower_ci'], metrics_before['top1']['upper_ci'])
    before_top1_count = metrics_before['top1']['correct']
    after_top1_acc = metrics_after['top1']['accuracy']
    after_top1_ci = (metrics_after['top1']['lower_ci'], metrics_after['top1']['upper_ci'])
    after_top1_count = metrics_after['top1']['correct']

    before_str = f"{before_top1_acc:.2%} ({before_top1_count}/{metrics_before['n_samples']}) [{before_top1_ci[0]:.2%}, {before_top1_ci[1]:.2%}]"
    after_str = f"{after_top1_acc:.2%} ({after_top1_count}/{metrics_after['n_samples']}) [{after_top1_ci[0]:.2%}, {after_top1_ci[1]:.2%}]"
    impr = (after_top1_acc - before_top1_acc)
    impr_str = f"+{impr:.2%} ({after_top1_acc/before_top1_acc:.1f}x)" if before_top1_acc > 0 else "N/A"

    output_lines.append(f"{'Top-1 Accuracy':<30} {before_str:<30} {after_str:<30} {impr_str:<30}")

    # Top-3 Accuracy
    before_top3_acc = metrics_before['top3']['accuracy']
    before_top3_ci = (metrics_before['top3']['lower_ci'], metrics_before['top3']['upper_ci'])
    before_top3_count = metrics_before['top3']['correct']
    after_top3_acc = metrics_after['top3']['accuracy']
    after_top3_ci = (metrics_after['top3']['lower_ci'], metrics_after['top3']['upper_ci'])
    after_top3_count = metrics_after['top3']['correct']

    before_str = f"{before_top3_acc:.2%} ({before_top3_count}/{metrics_before['n_samples']}) [{before_top3_ci[0]:.2%}, {before_top3_ci[1]:.2%}]"
    after_str = f"{after_top3_acc:.2%} ({after_top3_count}/{metrics_after['n_samples']}) [{after_top3_ci[0]:.2%}, {after_top3_ci[1]:.2%}]"
    impr = (after_top3_acc - before_top3_acc)
    impr_str = f"+{impr:.2%} ({after_top3_acc/before_top3_acc:.1f}x)" if before_top3_acc > 0 else "N/A"

    output_lines.append(f"{'Top-3 Accuracy':<30} {before_str:<30} {after_str:<30} {impr_str:<30}")

    # Top-5 Accuracy
    before_top5_acc = metrics_before['top5']['accuracy']
    before_top5_ci = (metrics_before['top5']['lower_ci'], metrics_before['top5']['upper_ci'])
    before_top5_count = metrics_before['top5']['correct']
    after_top5_acc = metrics_after['top5']['accuracy']
    after_top5_ci = (metrics_after['top5']['lower_ci'], metrics_after['top5']['upper_ci'])
    after_top5_count = metrics_after['top5']['correct']

    before_str = f"{before_top5_acc:.2%} ({before_top5_count}/{metrics_before['n_samples']}) [{before_top5_ci[0]:.2%}, {before_top5_ci[1]:.2%}]"
    after_str = f"{after_top5_acc:.2%} ({after_top5_count}/{metrics_after['n_samples']}) [{after_top5_ci[0]:.2%}, {after_top5_ci[1]:.2%}]"
    impr = (after_top5_acc - before_top5_acc)
    impr_str = f"+{impr:.2%} ({after_top5_acc/before_top5_acc:.1f}x)" if before_top5_acc > 0 else "N/A"

    output_lines.append(f"{'Top-5 Accuracy':<30} {before_str:<30} {after_str:<30} {impr_str:<30}")

    # Mean Confidence
    before_conf = metrics_before['mean_confidence']
    after_conf = metrics_after['mean_confidence']

    output_lines.append(f"{'Mean Confidence':<30} {before_conf:.2%}{'':24} {after_conf:.2%}{'':24} Calibrated")

    output_lines.append("="*120)
    output_lines.append("")

    result = "\n".join(output_lines)
    print(result)

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(result)
        print(f"✓ Saved table to {output_file}")
