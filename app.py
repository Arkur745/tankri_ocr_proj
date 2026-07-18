import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.configs.config import IMAGES_DIR, MODELS_DIR, PROJECT_ROOT
from src.inference.inference import (
    load_label_mappings,
    load_model_from_checkpoint,
    predict_from_pil_image,
    resolve_checkpoint_path,
)
from src.models import ResNet18Model
from src.dataset.augmentation import val_transform_resnet

st.set_page_config(page_title="Tankri OCR", page_icon="𑚔", layout="wide")


@st.cache_data(show_spinner=False)
def get_label_mappings(mapping_path: str):
    """Load label mappings once and reuse them across the app."""

    return load_label_mappings(Path(mapping_path))


@st.cache_resource(show_spinner=False)
def get_model_and_device():
    """Load the trained model and keep it cached for fast inference."""

    label_to_idx, idx_to_label = get_label_mappings(
        "artifacts/label_to_idx.json")
    num_classes = len(label_to_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18Model(num_classes=num_classes, pretrained=True)

    checkpoint_candidates = [
        PROJECT_ROOT / "notebooks" / "models" / "best_model.pth",
        MODELS_DIR / "best_model.pth",
        PROJECT_ROOT / "notebooks" / "best_model.pth",
        PROJECT_ROOT / "best_model.pth",
    ]

    checkpoint_path = None
    for candidate in checkpoint_candidates:
        resolved_candidate = resolve_checkpoint_path(candidate)
        if resolved_candidate.exists():
            checkpoint_path = resolved_candidate
            break

    if checkpoint_path is None:
        raise FileNotFoundError("Model checkpoint not found. Checked: " +
                                ", ".join(str(p) for p in checkpoint_candidates))

    model = load_model_from_checkpoint(model, checkpoint_path, device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    best_val_accuracy = checkpoint.get(
        "best_val_accuracy", checkpoint.get("val_accuracy", None))
    return model, device, label_to_idx, idx_to_label, best_val_accuracy


@st.cache_data(show_spinner=False)
def list_example_images(image_dir: str, limit: int = 8):
    """Return a small set of example images from the dataset."""

    image_paths = sorted(Path(image_dir).glob("*"))
    image_paths = [p for p in image_paths if p.suffix.lower() in {
        ".png", ".jpg", ".jpeg"}]
    return image_paths[:limit]


@st.cache_data(show_spinner=False)
def load_dataset_stats(labels_path: str):
    """Read dataset statistics for the sidebar summary."""

    df = pd.read_csv(labels_path)
    return {
        "train_images": len(df),
        "num_classes": df["label"].nunique(),
    }


@st.cache_data(show_spinner=False)
def load_validation_accuracy(checkpoint_path: str):
    """Read the best validation accuracy stored in the checkpoint metadata."""

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        best_val_accuracy = checkpoint.get(
            "best_val_accuracy", checkpoint.get("val_accuracy", None))
        if best_val_accuracy is not None:
            return round(float(best_val_accuracy * 100), 2)
    except Exception:
        pass

    return "N/A"


def render_header():
    st.title("Tankri OCR")
    st.caption("Handwritten Tankri Character Recognition")
    st.markdown("Powered by ResNet18 Transfer Learning")


def render_sidebar(model, device, label_to_idx, idx_to_label, best_val_accuracy):
    with st.sidebar:
        st.header("Model Information")
        st.write(f"Model: ResNet18")

        stats = load_dataset_stats("dataset/labels/labels.csv")
        st.write(f"Training Images: {stats['train_images']}")
        st.write(f"Number of Classes: {stats['num_classes']}")
        st.write(
            f"Best Validation Accuracy: {best_val_accuracy:.2f}%" if best_val_accuracy is not None else "Best Validation Accuracy: N/A")
        st.write(f"Device: {device}")

        st.divider()
        st.header("Example Images")
        examples = list_example_images("dataset/images", limit=8)

        for example_path in examples:
            if st.button(f"Use {example_path.name}", key=f"example-{example_path.name}"):
                st.session_state["example_image_path"] = str(example_path)


def run_prediction(image: Image.Image, model, device, idx_to_label, label_to_idx):
    start_time = time.perf_counter()
    result = predict_from_pil_image(
        image=image,
        model=model,
        transform=val_transform_resnet,
        idx_to_label=idx_to_label,
        device=device,
    )
    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    result["prediction_time_ms"] = elapsed_ms
    result["processed_shape"] = tuple(result["processed_tensor"].shape)
    result["confidence_pct"] = round(result["confidence"] * 100, 2)
    return result


def main():
    render_header()

    model, device, label_to_idx, idx_to_label, best_val_accuracy = get_model_and_device()
    render_sidebar(model, device, label_to_idx,
                   idx_to_label, best_val_accuracy)

    st.markdown("### Upload an image")

    uploaded_file = st.file_uploader(
        "Choose a PNG, JPG, or JPEG file",
        type=["png", "jpg", "jpeg"],
    )

    camera_image = st.camera_input("Or take a picture directly")

    image_input = uploaded_file or camera_image

    if image_input is not None:
        image = Image.open(image_input).convert("RGB")

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.image(image, caption="Uploaded Image", width=280)
        with col_right:
            st.image(
                image,
                caption="Model Input Preview",
                width=280,
                clamp=True,
            )

        if st.button("Run Inference", type="primary") or "last_image_bytes" not in st.session_state:
            st.session_state["last_image_bytes"] = image_input.getvalue()

        if st.session_state.get("last_image_bytes") == image_input.getvalue():
            with st.spinner("Analyzing character..."):
                result = run_prediction(
                    image, model, device, idx_to_label, label_to_idx)

            processed_image = result["processed_tensor"].squeeze(0).cpu()
            if processed_image.ndim == 3:
                processed_image = processed_image[0]
            processed_image = processed_image.numpy()
            processed_image = (processed_image - processed_image.min()) / \
                (processed_image.max() - processed_image.min() + 1e-8)
            processed_image = (processed_image * 255).astype("uint8")
            processed_pil = Image.fromarray(processed_image)

            st.markdown("### Model Input Preview")
            st.image(
                processed_pil, caption="Padded / processed image passed to the model", width=280)

            st.markdown("### Prediction")
            st.markdown(
                f"<div style='font-size: 72px; font-weight: 700;'>{result['predicted_label']}</div>",
                unsafe_allow_html=True,
            )
            st.write("Unicode")
            st.write(result["predicted_label"])
            st.write(f"Confidence: {result['confidence_pct']}%")

            st.progress(result["confidence"])

            if result["confidence"] < 0.70:
                st.warning(
                    "⚠ Low confidence prediction. This image may belong to another Tankri glyph variant.")

            st.divider()
            st.subheader("Top 5 Predictions")
            top5_df = pd.DataFrame(
                {
                    "Character": [idx_to_label[idx] for idx in result["top_indices"]],
                    "Confidence %": [round(prob * 100, 2) for prob in result["top_probabilities"]],
                }
            )
            st.dataframe(top5_df, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Size",
                          f"{image.size[0]} × {image.size[1]}")
            with col2:
                st.metric(
                    "Processed Size", f"{result['processed_shape'][2]} × {result['processed_shape'][3]}")
            with col3:
                st.metric("Prediction Time (ms)",
                          f"{result['prediction_time_ms']}")

            st.divider()
            st.subheader("Model Explainability")
            st.info(
                "GradCAM is not implemented yet for this architecture. A future update can add class activation visualizations.")

            with st.expander("Debug"):
                st.write("Raw logits")
                st.code(json.dumps(
                    result["logits"].tolist(), ensure_ascii=False))
                st.write("Softmax vector")
                st.code(json.dumps(
                    result["softmax_vector"].tolist(), ensure_ascii=False))
                st.write("Predicted class index")
                st.write(result["predicted_index"])

    elif st.session_state.get("example_image_path"):
        example_path = Path(st.session_state["example_image_path"])
        if example_path.exists():
            example_image = Image.open(example_path).convert("RGB")
            st.image(example_image, caption=example_path.name,
                     use_container_width=True)
            with st.spinner("Analyzing example image..."):
                result = run_prediction(
                    example_image, model, device, idx_to_label, label_to_idx)

            st.markdown("### Prediction")
            st.markdown(
                f"<div style='font-size: 72px; font-weight: 700;'>{result['predicted_label']}</div>",
                unsafe_allow_html=True,
            )
            st.write(f"Confidence: {result['confidence_pct']}%")
            st.progress(result["confidence"])


if __name__ == "__main__":
    main()
