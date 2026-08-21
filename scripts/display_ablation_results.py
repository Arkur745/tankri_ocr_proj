"""Display multi-seed ablation results from local JSON file."""
import json
from pathlib import Path
from collections import defaultdict
import statistics

# Read ablation results
results_file = Path('models/ablation_study/ablation_results.json')
with open(results_file) as f:
    results = json.load(f)

# Group by experiment ID
by_exp = defaultdict(list)
for result in results:
    exp_id = result['exp_id']
    seed = result['seed']
    val_acc = result['best_val_acc']
    test_acc = result['test_acc']
    by_exp[exp_id].append({'seed': seed, 'val_acc': val_acc, 'test_acc': test_acc})

# Config names
config_names = {
    'E00': 'Baseline (ResNet18)',
    'E01': 'OCR Augmentation (AugOCR)',
    'E02': 'Label Smoothing (ε=0.1)',
    'E03': 'Classifier Dropout (p=0.2)',
    'E04': 'Progressive Unfreezing (L3+L4)',
    'E05': 'Combined Model (AugOCR + L3+L4)',
}

print()
print('╔' + '='*118 + '╗')
print('║' + ' '*38 + 'TABLE V: SYSTEMATIC ABLATION RESULTS (MULTI-SEED)' + ' '*31 + '║')
print('╚' + '='*118 + '╝')
print()
print('VALIDATION SPLIT (182 images - for model/config selection across E00-E05)')
print('─'*130)
print(f'{"Run":<8} {"Model / Modification":<45} {"Mean Acc":<15} {"Std Dev":<12} {"Min":<10} {"Max":<10} {"Δ (%)":<10}')
print('─'*130)

baseline_mean = None
for exp_id in sorted(by_exp.keys()):
    runs = by_exp[exp_id]
    val_accs = [r['val_acc'] for r in runs]

    mean_acc = statistics.mean(val_accs)
    std_acc = statistics.stdev(val_accs) if len(val_accs) > 1 else 0.0
    min_acc = min(val_accs)
    max_acc = max(val_accs)

    if exp_id == 'E00':
        baseline_mean = mean_acc
        delta = '—'
    else:
        delta = f'+{(mean_acc - baseline_mean)*100:.2f}%'

    config_name = config_names.get(exp_id, 'Unknown')
    print(f'{exp_id:<8} {config_name:<45} {mean_acc:.4f}±{std_acc:.4f}   {std_acc:.4f}      {min_acc:.4f}    {max_acc:.4f}    {delta:<10}')

    # Show seed breakdown
    for run in sorted(runs, key=lambda x: x['seed']):
        marker = ' [original]' if run['seed'] == 42 else ''
        print(f'         seed={run["seed"]}: {run["val_acc"]:.4f}{marker}')

print()
print('═'*130)
print('TEST SPLIT (160 images - held-out final results)')
print('─'*130)
print(f'{"Run":<8} {"Model / Modification":<45} {"Mean Acc":<15} {"Std Dev":<12} {"Min":<10} {"Max":<10}')
print('─'*130)

for exp_id in sorted(by_exp.keys()):
    runs = by_exp[exp_id]
    test_accs = [r['test_acc'] for r in runs]

    mean_acc = statistics.mean(test_accs)
    std_acc = statistics.stdev(test_accs) if len(test_accs) > 1 else 0.0
    min_acc = min(test_accs)
    max_acc = max(test_accs)

    config_name = config_names.get(exp_id, 'Unknown')
    print(f'{exp_id:<8} {config_name:<45} {mean_acc:.4f}±{std_acc:.4f}   {std_acc:.4f}      {min_acc:.4f}    {max_acc:.4f}')

print('═'*130)
print()
