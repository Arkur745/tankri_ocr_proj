import json
import pandas as pd
from pathlib import Path

def create_label_mapping(csv_path, min_samples=2):
    """
    Generate label-to-index and index-to-label mappings from the CSV dataset.
    Only classes with at least `min_samples` are kept, sorted alphabetically.
    
    Parameters
    ----------
    csv_path : str or Path
        Path to labels.csv file.
    min_samples : int
        Minimum number of samples required for a class to be kept.
        
    Returns
    -------
    label_to_idx : dict
    idx_to_label : dict
    """
    df = pd.read_csv(csv_path)
    counts = df["label"].value_counts()
    valid_classes = counts[counts >= min_samples].index
    df_filtered = df[df["label"].isin(valid_classes)].copy()
    
    unique_labels = sorted(df_filtered["label"].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label

def save_label_mapping(label_to_idx, idx_to_label, output_dir):
    """
    Saves label_to_idx.json and idx_to_label.json to the output directory.
    
    Parameters
    ----------
    label_to_idx : dict
    idx_to_label : dict
    output_dir : str or Path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save label_to_idx
    with open(output_dir / "label_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f, ensure_ascii=False, indent=4)
        
    # Convert keys of idx_to_label to strings for JSON compatibility
    idx_to_label_str = {str(k): v for k, v in idx_to_label.items()}
    with open(output_dir / "idx_to_label.json", "w", encoding="utf-8") as f:
        json.dump(idx_to_label_str, f, ensure_ascii=False, indent=4)

def load_label_mapping(path):
    """
    Loads label_to_idx mapping and returns both label_to_idx and idx_to_label.
    `path` can be the output directory or the direct path to `label_to_idx.json`.
    
    Parameters
    ----------
    path : str or Path
        Path to the mapping JSON file or the directory containing it.
        
    Returns
    -------
    label_to_idx : dict
    idx_to_label : dict
    """
    path = Path(path)
    if path.is_dir():
        mapping_path = path / "label_to_idx.json"
    else:
        mapping_path = path
        
    if not mapping_path.exists():
        raise FileNotFoundError(f"Label mapping file not found at: {mapping_path}")
        
    with open(mapping_path, "r", encoding="utf-8") as f:
        label_to_idx = json.load(f)
        
    idx_to_label = {int(idx): label for label, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label
