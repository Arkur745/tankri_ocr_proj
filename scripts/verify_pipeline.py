import sys
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding='utf-8')

from src.models import ResNet18Model
from src.dataset.augmentation import val_transform_resnet
from src.dataset.loader import TankriDataset
from src.inference.inference import preprocess_image_for_model, predict_from_pil_image, resolve_checkpoint_path
from src.utils.label_mapping import load_label_mapping

def run_verification():
    print("Running end-to-end pipeline verification...")
    
    # 1. Load label mapping
    artifacts_dir = PROJECT_ROOT / "artifacts"
    label_to_idx, idx_to_label = load_label_mapping(artifacts_dir)
    num_classes = len(label_to_idx)
    
    # 2. Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_candidates = [
        PROJECT_ROOT / "notebooks" / "models" / "best_model.pth",
        PROJECT_ROOT / "models" / "best_model.pth",
        PROJECT_ROOT / "notebooks" / "best_model.pth",
        PROJECT_ROOT / "best_model.pth",
    ]
    checkpoint_path = None
    for cand in checkpoint_candidates:
        if cand.exists():
            checkpoint_path = cand
            break
            
    if checkpoint_path is None:
        print("❌ Model checkpoint best_model.pth not found. Cannot verify pipeline prediction stages.")
        sys.exit(1)
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ResNet18Model(num_classes=num_classes, pretrained=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"✔ Model loaded successfully from {checkpoint_path}")
    
    # 3. Take a sample image from the dataset
    labels_csv_path = PROJECT_ROOT / "dataset" / "labels" / "labels.csv"
    df = pd.read_csv(labels_csv_path)
    counts = df["label"].value_counts()
    valid_classes = counts[counts >= 2].index
    df_filtered = df[df["label"].isin(valid_classes)].copy()
    
    sample_row = df_filtered.iloc[0]
    image_name = sample_row["image"]
    true_label = sample_row["label"]
    image_path = PROJECT_ROOT / "dataset" / "images" / image_name
    print(f"✔ Using sample image: {image_name} (True label: {true_label})")
    
    # 4. Pipeline A: Dataset pipeline
    dataset = TankriDataset(
        dataframe=df_filtered,
        image_dir=PROJECT_ROOT / "dataset" / "images",
        label_to_idx=label_to_idx,
        transform=val_transform_resnet,
    )
    dataset_tensor, dataset_label_idx = dataset[0]
    dataset_tensor = dataset_tensor.unsqueeze(0).to(device) # Add batch dim
    
    # 5. Pipeline B: Inference preprocessing
    pil_image = Image.open(image_path).convert("RGB")
    inference_tensor = preprocess_image_for_model(pil_image, val_transform_resnet).to(device)
    
    # Compare Tensors
    tensors_equal = torch.allclose(dataset_tensor, inference_tensor, atol=1e-7)
    max_diff = torch.max(torch.abs(dataset_tensor - inference_tensor)).item()
    
    print(f"  Dataset tensor shape: {dataset_tensor.shape}")
    print(f"  Inference tensor shape: {inference_tensor.shape}")
    print(f"  Tensors identical? {tensors_equal} (Max diff: {max_diff})")
    
    if not tensors_equal:
        print("❌ Verification FAILED: Preprocessing tensors do not match!")
        sys.exit(1)
        
    # 6. Run predictions and extract outputs
    with torch.no_grad():
        dataset_logits = model(dataset_tensor)
        dataset_softmax = torch.softmax(dataset_logits, dim=1)
        dataset_pred_idx = dataset_softmax.argmax(dim=1).item()
        dataset_decoded = idx_to_label[dataset_pred_idx]
        
        # Inference path prediction
        inf_result = predict_from_pil_image(
            image=pil_image,
            model=model,
            transform=val_transform_resnet,
            idx_to_label=idx_to_label,
            device=device,
        )
        inference_logits = inf_result["logits"].unsqueeze(0).to(device)
        inference_softmax = inf_result["softmax_vector"].unsqueeze(0).to(device)
        inference_pred_idx = inf_result["predicted_index"]
        inference_decoded = inf_result["predicted_label"]
        
    # Compare predictions
    logits_equal = torch.allclose(dataset_logits, inference_logits, atol=1e-5)
    softmax_equal = torch.allclose(dataset_softmax, inference_softmax, atol=1e-5)
    pred_idx_equal = (dataset_pred_idx == inference_pred_idx)
    decoded_equal = (dataset_decoded == inference_decoded)
    
    print(f"  Logits match? {logits_equal}")
    print(f"  Softmax match? {softmax_equal}")
    print(f"  Predicted Index match? {pred_idx_equal} ({dataset_pred_idx} vs {inference_pred_idx})")
    print(f"  Decoded Label match? {decoded_equal} ({repr(dataset_decoded)} vs {repr(inference_decoded)})")
    
    if not (logits_equal and softmax_equal and pred_idx_equal and decoded_equal):
        print("❌ Verification FAILED: Prediction stages do not match!")
        sys.exit(1)
        
    print("\n" + "="*30)
    print(" PASS ")
    print("="*30)
    print("Pipeline verification completed successfully!")

if __name__ == "__main__":
    run_verification()
