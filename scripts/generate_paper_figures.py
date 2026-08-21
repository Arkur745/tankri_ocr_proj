import os
import matplotlib.pyplot as plt
import numpy as np

# Ensure figures directory exists
figures_dir = os.path.join(os.path.dirname(__file__), "..", "latex", "figures")
os.makedirs(figures_dir, exist_ok=True)

# Set style for publication quality
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.family': 'sans-serif',
    'figure.titlesize': 14,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.autolayout': True,
    'savefig.dpi': 300
})

# ==========================================
# Figure 6: Ablation Study Bar Chart (5-seed mean +/- std)
# ==========================================
fig, ax = plt.subplots(figsize=(8, 4.5))
runs = ['Baseline\n(E00)', 'OCR Aug\n(E01)', 'Label Smooth\n(E02)', 'Dropout\n(E03)', 'Layer3\n(E04)', 'Final Combined\n(E05)']
accuracies = [83.59, 86.41, 83.08, 83.21, 88.72, 89.23]
stds = [0.97, 1.15, 1.61, 1.39, 1.73, 0.95]
colors = ['#7f8c8d', '#2980b9', '#95a5a6', '#95a5a6', '#2980b9', '#27ae60']

bars = ax.bar(runs, accuracies, yerr=stds, capsize=4, color=colors, width=0.55, edgecolor='black', linewidth=0.8,
              error_kw={'elinewidth': 1.2, 'ecolor': '#333333'})
ax.set_ylabel('Validation Accuracy (%)', fontweight='bold')
ax.set_title('Figure 4: Systematic Ablation Study Performance Comparison\n(5-seed mean $\\pm$ std)', fontweight='bold', pad=12)
ax.set_ylim(75, 95)
ax.axhline(83.59, color='#e74c3c', linestyle='--', alpha=0.7, label='Baseline (83.59%)')

# Annotate values
for bar, std in zip(bars, stds):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + std + 0.4, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9.5)

ax.legend(loc='upper left')
plt.savefig(os.path.join(figures_dir, 'fig6_ablation_study.png'), dpi=300)
plt.close()

# ==========================================
# Figure 7: Domain Adaptation Improvement (Clean without arrows)
# ==========================================
fig, ax = plt.subplots(figsize=(7.5, 4.5))
metrics = ['Top-1 Accuracy', 'Top-3 Accuracy', 'Top-5 Accuracy']
before = [6.45, 19.35, 19.35]
after = [19.35, 35.48, 48.39]

x = np.arange(len(metrics))
width = 0.32

rects1 = ax.bar(x - width/2, before, width, label='Baseline Model (Before Adaptation)', color='#e74c3c', edgecolor='black', linewidth=0.8)
rects2 = ax.bar(x + width/2, after, width, label='Domain-Adapted Model (After Adaptation)', color='#2ec4b6', edgecolor='black', linewidth=0.8)

ax.set_ylabel('Accuracy (%) on Wall Inscriptions', fontweight='bold')
ax.set_title('Figure 5: Out-of-Distribution Wall Inscription Domain Adaptation\n(n=31, Wilson 95% CI reported in text)', fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontweight='bold')
ax.set_ylim(0, 60)

for bar in rects1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f'{yval:.2f}%', ha='center', va='bottom', fontsize=9)

for bar in rects2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9.5)

ax.legend(loc='upper left')
plt.savefig(os.path.join(figures_dir, 'fig7_domain_adaptation.png'), dpi=300)
plt.close()

print("Figures 6 and 7 generated cleanly in:", figures_dir)
