import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(
    page_title="Tankri Dataset Annotator",
    layout="centered"
)

st.title("Tankri Dataset Annotator")

# -----------------------------
# Dataset Selector
# -----------------------------
dataset_mode = st.radio(
    "Select Dataset to Annotate",
    options=["Training Dataset", "Test Dataset"],
    horizontal=True,
    key="dataset_mode"
)

if dataset_mode == "Training Dataset":
    IMAGES_DIR = Path("dataset/images")
    CSV_PATH = Path("dataset/labels/labels.csv")
else:
    IMAGES_DIR = Path("dataset/test")
    CSV_PATH = Path("dataset/labels/test_labels.csv")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    Path("dataset/labels").mkdir(parents=True, exist_ok=True)

# -----------------------------
# Create CSV if missing
# -----------------------------
if not CSV_PATH.exists():
    pd.DataFrame(
        columns=["image", "label"]
    ).to_csv(
        CSV_PATH,
        index=False
    )

# -----------------------------
# Load existing labels
# -----------------------------
df = pd.read_csv(CSV_PATH)

# -----------------------------
# Get all images in numeric order (fallback to string sorting for non-numeric)
# -----------------------------
def get_sort_key(filename):
    stem = Path(filename).stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)

all_images = sorted(
    [f.name for f in IMAGES_DIR.glob("*.png")],
    key=get_sort_key
)

if len(all_images) == 0:
    st.warning(f"No PNG images found in folder: {IMAGES_DIR.as_posix()}")
    st.stop()

# -----------------------------
# Reset Session State on mode change
# -----------------------------
if "prev_dataset_mode" not in st.session_state or st.session_state.prev_dataset_mode != dataset_mode:
    st.session_state.prev_dataset_mode = dataset_mode
    if "current_index" in st.session_state:
        del st.session_state["current_index"]

# -----------------------------
# Initialize Current Index
# -----------------------------
annotated = set(df["image"])

if "current_index" not in st.session_state:
    first_unannotated_idx = 0
    for idx, img in enumerate(all_images):
        if img not in annotated:
            first_unannotated_idx = idx
            break
    st.session_state.current_index = first_unannotated_idx

# Keep index within bounds
st.session_state.current_index = max(0, min(st.session_state.current_index, len(all_images) - 1))

# -----------------------------
# Progress Bar
# -----------------------------
total = len(all_images)
done = len(annotated)
st.progress(done / total if total > 0 else 0)
st.write(f"Overall Progress: {done} / {total} images annotated.")

# -----------------------------
# Navigation & Jump Controls
# -----------------------------
st.subheader("Navigation")

# Dropdown to jump directly
def on_jump_change():
    st.session_state.current_index = all_images.index(st.session_state.jump_select)

st.selectbox(
    "Jump to Image",
    options=all_images,
    index=st.session_state.current_index,
    key="jump_select",
    on_change=on_jump_change
)

# Prev/Next Buttons
col_nav1, col_nav2 = st.columns(2)
with col_nav1:
    if st.button("⬅ Previous Image", disabled=(st.session_state.current_index == 0), use_container_width=True):
        st.session_state.current_index -= 1
        st.rerun()
with col_nav2:
    if st.button("Next Image ➡", disabled=(st.session_state.current_index == len(all_images) - 1), use_container_width=True):
        st.session_state.current_index += 1
        st.rerun()

# -----------------------------
# Annotation Form
# -----------------------------
current_image = all_images[st.session_state.current_index]

st.divider()
st.subheader(f"Image {st.session_state.current_index + 1} of {total}: {current_image}")

st.image(
    str(IMAGES_DIR / current_image),
    width=300
)

# Check for existing label
existing_row = df[df["image"] == current_image]
if not existing_row.empty:
    existing_label = existing_row.iloc[0]["label"]
    st.success(f"Annotated! Current label: **{existing_label}**")
else:
    existing_label = ""
    st.warning("Status: Unannotated")

# Input field prefilled with existing label if present
label = st.text_input(
    "Label",
    value=existing_label,
    key=f"label_input_{current_image}"
)

col_act1, col_act2 = st.columns(2)

with col_act1:
    if st.button("Save Annotation", type="primary", use_container_width=True):
        if label.strip() == "":
            st.error("Please enter a label before saving.")
        else:
            if not existing_row.empty:
                # Update existing row
                df.loc[df["image"] == current_image, "label"] = label.strip()
            else:
                # Append new row
                new_row = pd.DataFrame([{"image": current_image, "label": label.strip()}])
                df = pd.concat([df, new_row], ignore_index=True)
            
            df.to_csv(CSV_PATH, index=False)
            st.toast(f"Saved {current_image} -> {label.strip()}!")
            
            # Auto-advance to next image
            if st.session_state.current_index < len(all_images) - 1:
                st.session_state.current_index += 1
            st.rerun()

with col_act2:
    if st.button("Skip / Clear", use_container_width=True):
        if st.session_state.current_index < len(all_images) - 1:
            st.session_state.current_index += 1
        st.rerun()