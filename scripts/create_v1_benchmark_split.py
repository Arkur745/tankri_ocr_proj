"""
Create a clean stratified 70/15/15 train/val/test split of the ORIGINAL
V1 benchmark dataset (images with numeric ID <= 1060), per GPT's clarification
that the E00-E05 ablation study must stay on the original benchmark dataset,
not the expanded 1,205-image set (which is reserved for domain adaptation).

This applies Fix 4 (clean 3-way split, no leakage) correctly scoped to the
same 1,060-image benchmark that produced the original E01-E05 MLflow numbers,
instead of the previous (incorrect) application to the full 1,205-image set.
"""
import csv
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_split import create_stratified_split, print_split_distribution


def main():
    labels_file = PROJECT_ROOT / "dataset" / "labels" / "labels.csv"
    v1_labels_file = PROJECT_ROOT / "dataset" / "labels" / "labels_v1_benchmark.csv"
    output_dir = PROJECT_ROOT / "artifacts" / "data_splits_v1_benchmark"

    # Filter to V1 images only (numeric id <= 1060)
    with open(labels_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    v1_rows = []
    for row in rows:
        m = re.match(r'^(\d+)\.png$', row['image'])
        if m and int(m.group(1)) <= 1060:
            v1_rows.append(row)

    print(f"Filtered {len(v1_rows)} V1 images (out of {len(rows)} total in labels.csv)")

    with open(v1_labels_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'label'])
        writer.writeheader()
        writer.writerows(v1_rows)
    print(f"Saved V1-only labels to {v1_labels_file}")

    print(f"\nCreating stratified 70/15/15 split of V1 benchmark dataset...")
    split_dict = create_stratified_split(
        labels_file=v1_labels_file,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        output_dir=output_dir,
    )

    print("\nVerifying split distribution...")
    is_good = print_split_distribution(v1_labels_file, split_dict)

    if not is_good:
        print("\n⚠️  Split validation warnings above (small classes) - review before proceeding.")
    else:
        print("\n✓ V1 benchmark split looks good! Ready for corrected ablation study.")

    print(f"\nSplit sizes: train={len(split_dict['train'])}, val={len(split_dict['val'])}, test={len(split_dict['test'])}")


if __name__ == "__main__":
    main()
