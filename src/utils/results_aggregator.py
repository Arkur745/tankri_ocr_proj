"""
Aggregate multi-seed ablation results from MLflow.
Computes mean ± std and generates paper-ready tables.
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import mlflow


def query_mlflow_runs(experiment_name, exp_ids=None, seeds=None):
    """
    Query MLflow runs for multi-seed ablation study.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment
    exp_ids : list of str, optional
        Filter to specific experiment IDs (e.g., ['E00', 'E01'])
    seeds : list of int, optional
        Filter to specific seeds

    Returns
    -------
    dict
        Runs organized by (exp_id, seed)
    """
    client = mlflow.tracking.MlflowClient()

    # Get experiment ID
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found in MLflow")
        return {}

    # Get all runs for this experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1000,
    )

    # Filter and organize runs
    filtered_runs = {}
    for run in runs:
        params = run.data.params
        metrics = run.data.metrics

        # Extract exp_id and seed
        run_exp_id = params.get('exp_id')
        run_seed = params.get('random_seed')

        if run_exp_id is None or run_seed is None:
            continue

        # Apply filters
        if exp_ids is not None and run_exp_id not in exp_ids:
            continue
        if seeds is not None and int(run_seed) not in seeds:
            continue

        key = (run_exp_id, int(run_seed))
        filtered_runs[key] = {
            'run_id': run.info.run_id,
            'params': params,
            'metrics': metrics,
        }

    return filtered_runs


def aggregate_results(runs_dict):
    """
    Aggregate results by experiment ID, computing mean and std across seeds.
    Reports both validation split (model selection) and test split (final reporting).

    Parameters
    ----------
    runs_dict : dict
        Runs organized by (exp_id, seed) tuples

    Returns
    -------
    dict
        Aggregated statistics by exp_id (val and test splits reported separately)
    """
    # Group by exp_id
    by_exp_id = defaultdict(list)
    for (exp_id, seed), run_data in runs_dict.items():
        metrics = run_data['metrics']
        # Extract both validation (model selection) and test (final) accuracies
        val_acc = metrics.get('best_val_accuracy', 0.0)  # 182 images for model selection
        test_acc = metrics.get('test_accuracy', 0.0)      # 160 images for headline result
        by_exp_id[exp_id].append({
            'seed': seed,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
        })

    # Compute stats
    aggregated = {}
    for exp_id in sorted(by_exp_id.keys()):
        results = by_exp_id[exp_id]
        val_accuracies = [r['val_accuracy'] for r in results]
        test_accuracies = [r['test_accuracy'] for r in results]
        seeds = [r['seed'] for r in results]

        aggregated[exp_id] = {
            'n_seeds': len(val_accuracies),
            'seeds': sorted(seeds),
            # Validation split (182 images) - for ablation comparison
            'val': {
                'accuracies': val_accuracies,
                'mean': float(np.mean(val_accuracies)),
                'std': float(np.std(val_accuracies)),
                'min': float(np.min(val_accuracies)),
                'max': float(np.max(val_accuracies)),
                'split_size': 182,
                'purpose': 'Model/config selection across E00-E05',
            },
            # Test split (160 images) - for final headline results
            'test': {
                'accuracies': test_accuracies,
                'mean': float(np.mean(test_accuracies)),
                'std': float(np.std(test_accuracies)),
                'min': float(np.min(test_accuracies)),
                'max': float(np.max(test_accuracies)),
                'split_size': 160,
                'purpose': 'Final held-out test set (headline result)',
            },
        }

    return aggregated


def print_ablation_table(aggregated, include_seed_42=True):
    """
    Print formatted ablation results table with mean ± std.
    Shows both validation split (model selection) and test split (headline results).

    Parameters
    ----------
    aggregated : dict
        Results from aggregate_results
    include_seed_42 : bool
        If True, show seed=42 results as sanity check against original single-run
    """
    # Configuration names for reference
    config_names = {
        'E00': 'Baseline (ResNet18)',
        'E01': 'OCR Augmentation (AugOCR)',
        'E02': 'Label Smoothing (ε=0.1)',
        'E03': 'Classifier Dropout (p=0.2)',
        'E04': 'Progressive Unfreezing (L3+L4)',
        'E05': 'Combined Model (AugOCR + L3+L4)',
    }

    print("\n" + "="*150)
    print("TABLE V: SYSTEMATIC ABLATION RESULTS (MULTI-SEED)")
    print("="*150)
    print("\nVALIDATION SPLIT (182 images - for model/config selection across E00-E05)")
    print("-"*150)
    print(f"{'Run':<8} {'Model / Modification':<40} {'Mean Acc':<15} {'Std Dev':<12} {'Min':<10} {'Max':<10} {'Δ (%)':<10}")
    print("-"*150)

    baseline_val_mean = None

    for exp_id in sorted(aggregated.keys()):
        stats = aggregated[exp_id]
        val_stats = stats['val']
        mean_acc = val_stats['mean']
        std_acc = val_stats['std']
        min_acc = val_stats['min']
        max_acc = val_stats['max']

        if exp_id == 'E00':
            baseline_val_mean = mean_acc
            delta = "—"
        else:
            if baseline_val_mean is not None:
                delta = f"+{(mean_acc - baseline_val_mean)*100:.2f}%"
            else:
                delta = "N/A"

        config_name = config_names.get(exp_id, "Unknown")

        print(f"{exp_id:<8} {config_name:<40} {mean_acc:.4f}±{std_acc:.4f}   {std_acc:.4f}      {min_acc:.4f}    {max_acc:.4f}    {delta:<10}")

        # Show all seed accuracies with seed=42 marked as original
        acc_by_seed = {seed: acc for seed, acc in zip(stats['seeds'], val_stats['accuracies'])}
        seed_str_parts = []
        for seed in sorted(acc_by_seed.keys()):
            acc = acc_by_seed[seed]
            marker = " [original single-run]" if seed == 42 else ""
            seed_str_parts.append(f"seed={seed}: {acc:.4f}{marker}")

        for seed_str in seed_str_parts:
            print(f"         {seed_str}")

    print("\n" + "="*150)
    print("TEST SPLIT (160 images - held-out final results for headline reporting)")
    print("-"*150)
    print(f"{'Run':<8} {'Model / Modification':<40} {'Mean Acc':<15} {'Std Dev':<12} {'Min':<10} {'Max':<10}")
    print("-"*150)

    for exp_id in sorted(aggregated.keys()):
        stats = aggregated[exp_id]
        test_stats = stats['test']
        mean_acc = test_stats['mean']
        std_acc = test_stats['std']
        min_acc = test_stats['min']
        max_acc = test_stats['max']

        config_name = config_names.get(exp_id, "Unknown")

        print(f"{exp_id:<8} {config_name:<40} {mean_acc:.4f}±{std_acc:.4f}   {std_acc:.4f}      {min_acc:.4f}    {max_acc:.4f}")

    print("="*150)
    print("")

    # Print summary
    print("INTERPRETATION GUIDE:")
    print("  • Val accuracies (182 images): Use for ablation comparison to select best config (E05)")
    print("  • Test accuracies (160 images): Use for headline results in abstract/results section")
    print("  • Seed=42 marked as 'original single-run' for comparison with originally reported 90.09%")
    print("")


def save_aggregated_results(aggregated, output_path):
    """
    Save aggregated results to JSON file.

    Parameters
    ----------
    aggregated : dict
        Results from aggregate_results
    output_path : Path or str
        Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=2)

    print(f"✓ Saved aggregated results to {output_path}")


def compare_with_original_single_run(aggregated, original_accuracies):
    """
    Compare multi-seed means with originally reported single-run accuracies.

    Parameters
    ----------
    aggregated : dict
        Results from aggregate_results
    original_accuracies : dict
        Original single-run accuracies (exp_id -> accuracy)
    """
    print("\nCOMPARISON WITH ORIGINAL SINGLE-RUN RESULTS")
    print("="*130)
    print(f"{'Run':<8} {'Original Single-Run':<25} {'Multi-Seed Mean':<25} {'Difference':<25}")
    print("-"*130)

    for exp_id in sorted(aggregated.keys()):
        if exp_id not in original_accuracies:
            continue

        original = original_accuracies[exp_id]
        multi_seed_mean = aggregated[exp_id]['mean']
        diff = multi_seed_mean - original

        print(f"{exp_id:<8} {original:.4f}{'':20} {multi_seed_mean:.4f}±{aggregated[exp_id]['std']:.4f}     {diff:+.4f}")

    print("="*130)
    print("\nNote: The seed=42 run should match the original single-run result closely.")
    print("")


if __name__ == "__main__":
    # Original single-run results from the paper (baseline for comparison)
    ORIGINAL_RESULTS = {
        'E00': 0.8160,
        'E01': 0.8868,
        'E02': 0.8302,
        'E03': 0.8302,
        'E04': 0.8868,
        'E05': 0.9009,
    }

    # Query MLflow
    experiment_name = "TakriOCR_MultiSeed_Ablation"
    runs = query_mlflow_runs(experiment_name)

    if not runs:
        print(f"No runs found for experiment '{experiment_name}'")
        print("Make sure the multi-seed ablation has been run and MLflow experiment is created.")
    else:
        print(f"Found {len(runs)} runs in MLflow")

        # Aggregate results
        aggregated = aggregate_results(runs)

        # Print results
        print_ablation_table(aggregated)

        # Compare with original
        compare_with_original_single_run(aggregated, ORIGINAL_RESULTS)

        # Save results
        output_path = Path("artifacts/ablation_results_aggregated.json")
        save_aggregated_results(aggregated, output_path)
