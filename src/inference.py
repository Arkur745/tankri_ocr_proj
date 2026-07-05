import json
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import torch
from PIL import Image

from src.preprocessing import crop_character


def resolve_existing_path(path: Union[str, Path], fallback_candidates: Optional[Iterable[Union[str, Path]]] = None) -> Path:
    """Resolve an input path from the current working directory or the project root."""

    path = Path(path)
    if path.is_absolute():
        return path if path.exists() else path

    search_roots = [Path.cwd(), Path(__file__).resolve().parents[1]]
    for root in search_roots:
        candidate = root / path
        if candidate.exists():
            return candidate

    if fallback_candidates:
        for candidate in fallback_candidates:
            resolved_candidate = Path(candidate)
            if resolved_candidate.exists():
                return resolved_candidate
            for root in search_roots:
                alt_candidate = root / resolved_candidate
                if alt_candidate.exists():
                    return alt_candidate

    return path


def resolve_checkpoint_path(checkpoint_path: Union[str, Path]) -> Path:
    """Resolve the checkpoint path using the project layout and common fallbacks."""

    return resolve_existing_path(
        checkpoint_path,
        fallback_candidates=[
            "models/best_model.pth",
            "notebooks/best_model.pth",
            "notebooks/models/best_model.pth",
            "best_model.pth",
        ],
    )


def load_model(model, checkpoint_path, device):
    """Load a trained model from checkpoint."""

    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


def load_model_from_checkpoint(model, checkpoint_path, device):
    """Compatibility wrapper for the Streamlit app."""

    return load_model(model, checkpoint_path, device)


def load_label_mappings(mapping_path):
    """Load label-to-index and index-to-label mappings from disk."""

    mapping_path = resolve_existing_path(
        mapping_path,
        fallback_candidates=["notebooks/label_to_idx.json",
                             "artifacts/label_to_idx.json"],
    )
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Label mapping file not found: {mapping_path}")

    with mapping_path.open("r", encoding="utf-8") as handle:
        label_to_idx = json.load(handle)

    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label


def preprocess_image_for_model(image, transform):
    """Apply the same preprocessing pathway used for validation inference."""

    grayscale_image = image.convert("L")
    image_array = np.array(grayscale_image)

    cropped_image = crop_character(image_array)
    pil_image = Image.fromarray(cropped_image)

    tensor = transform(pil_image)
    return tensor.unsqueeze(0)


def predict_from_pil_image(image, model, transform, idx_to_label, device):
    """Run inference for a PIL image and return logits, softmax probabilities, and top-k results."""

    processed_tensor = preprocess_image_for_model(image, transform)
    processed_tensor = processed_tensor.to(device)

    with torch.inference_mode():
        logits = model(processed_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_index = probabilities.argmax().item()
        predicted_label = idx_to_label[predicted_index]

    top_probabilities, top_indices = probabilities.topk(
        min(5, len(idx_to_label)))
    top_probabilities = top_probabilities.cpu().tolist()
    top_indices = top_indices.cpu().tolist()

    return {
        "predicted_label": predicted_label,
        "predicted_index": predicted_index,
        "logits": logits[0].cpu(),
        "softmax_vector": probabilities.cpu(),
        "top_probabilities": top_probabilities,
        "top_indices": top_indices,
        "processed_tensor": processed_tensor.cpu(),
        "confidence": float(probabilities[predicted_index].cpu().item()),
    }


def predict_image(image_path, model, transform, idx_to_label, device):
    """Predict a single image from a file path."""

    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    result = predict_from_pil_image(
        image, model, transform, idx_to_label, device)
    return result["predicted_label"]


def predict_folder(folder_path, model, transform, idx_to_label, device):
    """Predict all images inside a folder."""

    folder_path = Path(folder_path)
    predictions = {}

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    for image_path in sorted(folder_path.iterdir()):
        if image_path.suffix.lower() not in image_extensions:
            continue

        label = predict_image(
            image_path=image_path,
            model=model,
            transform=transform,
            idx_to_label=idx_to_label,
            device=device,
        )

        predictions[image_path.name] = label

    return predictions
