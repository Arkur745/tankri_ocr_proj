"""
Create and manage stratified train/val/test splits for the Takri OCR dataset.
Ensures reproducibility and prevents data leakage between evaluation splits.
"""
import json
import csv
from pathlib import Path
import random


def read_csv_as_list_of_dicts(csv_path):
    """Read CSV into list of dicts."""
    result = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            result.append(row)
    return result


def stratified_split_manual(
    data_list,
    test_size,
    random_state=42,
    stratify_by_key='label',
):
    """
    Manually implement stratified train/test split.

    Parameters
    ----------
    data_list : list of dict
        List of data records
    test_size : float
        Fraction for test set
    random_state : int
        Random seed for reproducibility
    stratify_by_key : str
        Key to stratify by

    Returns
    -------
    tuple of list
        (train_list, test_list)
    """
    random.seed(random_state)
    train_list = []
    test_list = []

    # Group by stratification key
    groups = {}
    for item in data_list:
        label = item[stratify_by_key]
        if label not in groups:
            groups[label] = []
        groups[label].append(item)

    # For each group, randomly split
    for label, items in groups.items():
        shuffled = items.copy()
        random.shuffle(shuffled)

        n_test = max(1, int(len(shuffled) * test_size))
        test_list.extend(shuffled[:n_test])
        train_list.extend(shuffled[n_test:])

    return train_list, test_list


def create_stratified_split(
    labels_file,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
    output_dir=None,
):
    """
    Create a stratified train/val/test split from the label CSV.

    Parameters
    ----------
    labels_file : Path or str
        Path to labels.csv containing 'image' and 'label' columns
    train_ratio : float
        Fraction for training (default 0.70)
    val_ratio : float
        Fraction for validation (default 0.15)
    test_ratio : float
        Fraction for test (default 0.15)
    random_seed : int
        Random seed for reproducibility
    output_dir : Path or str, optional
        Directory to save split JSON files. If None, returns dict only.

    Returns
    -------
    dict
        Dictionary with keys 'train', 'val', 'test' containing image filenames
    """
    labels_file = Path(labels_file)
    data_list = read_csv_as_list_of_dicts(str(labels_file))

    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    # First split: train vs (val+test)
    val_test_ratio = val_ratio + test_ratio
    train_list, val_test_list = stratified_split_manual(
        data_list,
        test_size=val_test_ratio,
        random_state=random_seed,
        stratify_by_key='label',
    )

    # Second split: val vs test (from the val_test pool)
    val_frac_of_remaining = val_ratio / val_test_ratio
    val_list, test_list = stratified_split_manual(
        val_test_list,
        test_size=1 - val_frac_of_remaining,
        random_state=random_seed,
        stratify_by_key='label',
    )

    split_dict = {
        'train': sorted([item['image'] for item in train_list]),
        'val': sorted([item['image'] for item in val_list]),
        'test': sorted([item['image'] for item in test_list]),
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for split_name, images in split_dict.items():
            split_path = output_dir / f'{split_name}_split.json'
            with open(split_path, 'w') as f:
                json.dump({'images': images}, f, indent=2)
            print(f"Saved {split_name} split to {split_path}")

    return split_dict


def print_split_distribution(labels_file, split_dict):
    """
    Print per-class distribution across train/val/test splits.
    Flags classes with fewer than 3 samples in val or test.

    Parameters
    ----------
    labels_file : Path or str
        Path to labels.csv
    split_dict : dict
        Dictionary with 'train', 'val', 'test' keys containing image lists
    """
    labels_file = Path(labels_file)
    data_list = read_csv_as_list_of_dicts(str(labels_file))

    # Create mapping from image to label
    img_to_label = {item['image']: item['label'] for item in data_list}

    # Get unique labels in sorted order
    unique_labels = sorted(set(img_to_label.values()))
    num_classes = len(unique_labels)

    print("\n" + "="*100)
    print(f"STRATIFIED SPLIT DISTRIBUTION (70/15/15)")
    print("="*100)
    print(f"\n{'Label':<15} {'Char':<8} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10} {'Status':<20}")
    print("-"*100)

    warnings = []

    for label in unique_labels:
        train_count = sum(1 for img in split_dict['train'] if img_to_label[img] == label)
        val_count = sum(1 for img in split_dict['val'] if img_to_label[img] == label)
        test_count = sum(1 for img in split_dict['test'] if img_to_label[img] == label)
        total_count = train_count + val_count + test_count

        status = "✓ OK"
        if val_count < 3:
            status = "⚠ Val < 3"
            warnings.append(f"  {label} (val_count={val_count})")
        if test_count < 3:
            status = "⚠ Test < 3"
            warnings.append(f"  {label} (test_count={test_count})")

        # Format as label_index for readability
        label_idx = list(unique_labels).index(label)
        print(f"{label_idx:<15} {label:<8} {train_count:<10} {val_count:<10} {test_count:<10} {total_count:<10} {status:<20}")

    # Summary statistics
    print("-"*100)
    train_total = len(split_dict['train'])
    val_total = len(split_dict['val'])
    test_total = len(split_dict['test'])
    total = train_total + val_total + test_total

    print(f"{'TOTAL':<15} {'':8} {train_total:<10} {val_total:<10} {test_total:<10} {total:<10} {'':<20}")
    print(f"{'Ratio':<15} {'':8} {train_total/total:.1%}{'':6} {val_total/total:.1%}{'':6} {test_total/total:.1%}{'':6}")
    print("="*100)

    if warnings:
        print("\n⚠️  WARNINGS: Classes with fewer than 3 samples in val or test split:")
        for warning in warnings:
            print(warning)
        print("\nConsider merging small classes or collecting more samples for these classes.")
    else:
        print("\n✓ All classes have at least 3 samples in val and test splits.")

    print()
    return len(warnings) == 0


def load_split(split_path):
    """
    Load a previously saved split JSON file.

    Parameters
    ----------
    split_path : Path or str
        Path to a split JSON file (e.g., 'train_split.json')

    Returns
    -------
    list
        List of image filenames in the split
    """
    split_path = Path(split_path)
    with open(split_path, 'r') as f:
        data = json.load(f)
    return data['images']


if __name__ == "__main__":
    # Example usage
    from src.configs.config import PROJECT_ROOT, LABELS_FILE

    labels_file = PROJECT_ROOT / "dataset" / "labels" / "labels.csv"
    output_dir = PROJECT_ROOT / "artifacts" / "data_splits"

    print(f"Creating stratified split from {labels_file}")
    split_dict = create_stratified_split(
        labels_file=labels_file,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        output_dir=output_dir,
    )

    print("\nVerifying split distribution...")
    is_good = print_split_distribution(labels_file, split_dict)

    if not is_good:
        print("\n⚠️  Split validation failed. Review warnings above.")
    else:
        print("\n✓ Split looks good! Ready for training.")
