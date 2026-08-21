import sys
import pandas as pd
import torch
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding='utf-8')

from src.utils.label_mapping import load_label_mapping
from src.configs.config import LABELS_FILE, MODELS_DIR

def run_mapping_checks():
    print("Running mapping consistency checks...")
    
    # 1. Load mappings from artifacts
    artifacts_dir = PROJECT_ROOT / "artifacts"
    try:
        label_to_idx, idx_to_label = load_label_mapping(artifacts_dir)
        print(f"✔ Successfully loaded mappings from {artifacts_dir}. Size: {len(label_to_idx)}")
    except Exception as e:
        print(f"❌ Failed to load mappings from artifacts: {e}")
        sys.exit(1)
        
    # 2. Check consistency between label_to_idx and idx_to_label
    for label, idx in label_to_idx.items():
        if idx_to_label.get(idx) != label:
            print(f"❌ Mismatch: label_to_idx maps {repr(label)} -> {idx}, but idx_to_label maps {idx} -> {repr(idx_to_label.get(idx))}")
            sys.exit(1)
    print("✔ label_to_idx and idx_to_label are internally consistent.")
    
    # 3. Check consistency with dataset labels (min_samples=2 filter)
    df = pd.read_csv(PROJECT_ROOT / LABELS_FILE)
    counts = df["label"].value_counts()
    valid_classes = set(counts[counts >= 2].index)
    mapping_classes = set(label_to_idx.keys())
    
    if valid_classes != mapping_classes:
        print("❌ Mismatch between filtered dataset classes and mapping classes!")
        print(f"Classes in dataset but not in mapping: {valid_classes - mapping_classes}")
        print(f"Classes in mapping but not in dataset: {mapping_classes - valid_classes}")
        sys.exit(1)
    print("✔ Mappings match the filtered dataset classes perfectly.")
    
    # 4. Check checkpoint output dimension
    # models/ is the single canonical, verified-correct checkpoint location
    # (consolidated 2026-08-21 -- see CHANGELOG.md). Other candidates are
    # fallbacks for partial/local-only checkouts only.
    checkpoint_candidates = [
        PROJECT_ROOT / "models" / "best_model.pth",
        PROJECT_ROOT / "notebooks" / "models" / "best_model.pth",
        PROJECT_ROOT / "notebooks" / "best_model.pth",
        PROJECT_ROOT / "best_model.pth",
    ]
    checkpoint_path = None
    for cand in checkpoint_candidates:
        if cand.exists():
            checkpoint_path = cand
            break
            
    if checkpoint_path is None:
        print("⚠ Checkpoint best_model.pth not found. Skipping model check.")
        print("✔ Mappings check PASSED.")
        sys.exit(0)
        
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = checkpoint.get("model_state_dict", checkpoint)
    
    # Extract fc.weight shape
    fc_weight = model_state.get("model.fc.weight", model_state.get("fc.weight", None))
    if fc_weight is None:
        print("❌ Could not find classifier weights ('model.fc.weight' or 'fc.weight') in checkpoint.")
        sys.exit(1)
        
    num_checkpoint_classes = fc_weight.shape[0]
    num_mapping_classes = len(label_to_idx)
    
    if num_checkpoint_classes != num_mapping_classes:
        print(f"❌ Class count mismatch! Checkpoint output dimension: {num_checkpoint_classes}, Mapping classes: {num_mapping_classes}")
        sys.exit(1)
        
    print(f"✔ Checkpoint output dimension ({num_checkpoint_classes}) matches mapping class count ({num_mapping_classes}).")
    print("\n🎉 ALL CHECKS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_mapping_checks()
