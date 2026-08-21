"""
Aggregate full Table V statistics (val accuracy, val Macro F1, val Weighted F1,
test accuracy) across all 5 seeds per config, for the paper's final Table V.
"""
import json
import csv
import statistics
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ablation_dir = PROJECT_ROOT / "models" / "ablation_study"

with open(ablation_dir / "ablation_results.json") as f:
    results = json.load(f)

by_exp = defaultdict(list)
for r in results:
    by_exp[r['exp_id']].append(r)

config_names = {
    'E00': 'Baseline (ResNet18)',
    'E01': 'OCR Aug.',
    'E02': 'Label Smooth',
    'E03': 'Classifier Dropout',
    'E04': 'Layer Unfreeze (L3+L4)',
    'E05': 'Combined Model',
}

print(f"{'Run':<6}{'Val Acc':<16}{'Val MacroF1':<16}{'Val WeightedF1':<16}{'Test Acc':<16}")
print("-" * 80)

baseline_val_mean = None
for exp_id in sorted(by_exp.keys()):
    runs = by_exp[exp_id]
    val_accs, test_accs, macro_f1s, weighted_f1s = [], [], [], []

    for r in runs:
        seed = r['seed']
        val_accs.append(r['best_val_acc'])
        test_accs.append(r['test_acc'])

        report_path = ablation_dir / f"{exp_id}_seed{seed}" / "classification_report.csv"
        with open(report_path, encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        for row in rows:
            if row[0] == 'macro avg':
                macro_f1s.append(float(row[3]))
            if row[0] == 'weighted avg':
                weighted_f1s.append(float(row[3]))

    val_mean, val_std = statistics.mean(val_accs), statistics.stdev(val_accs)
    test_mean, test_std = statistics.mean(test_accs), statistics.stdev(test_accs)
    macro_mean, macro_std = statistics.mean(macro_f1s), statistics.stdev(macro_f1s)
    weighted_mean, weighted_std = statistics.mean(weighted_f1s), statistics.stdev(weighted_f1s)

    if exp_id == 'E00':
        baseline_val_mean = val_mean
    delta = (val_mean - baseline_val_mean) * 100

    print(f"{exp_id:<6}{val_mean*100:.2f}±{val_std*100:.2f}    {macro_mean:.3f}±{macro_std:.3f}    {weighted_mean:.3f}±{weighted_std:.3f}    {test_mean*100:.2f}±{test_std*100:.2f}")
    print(f"  LaTeX: {exp_id} & {config_names[exp_id]} & {val_mean*100:.2f}$\\pm${val_std*100:.2f} & {macro_mean:.3f}$\\pm${macro_std:.3f} & {weighted_mean:.3f}$\\pm${weighted_std:.3f} & {'---' if exp_id=='E00' else f'+{delta:.2f}'} \\\\")
