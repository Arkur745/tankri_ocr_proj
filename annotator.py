import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(
    page_title="Tankri Dataset Annotator",
    layout="centered"
)

# -----------------------------
# Paths
# -----------------------------
IMAGES_DIR = Path("dataset/images")
LABELS_DIR = Path("dataset/labels")
LABELS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = LABELS_DIR / "labels.csv"

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
# Get all images in numeric order
# -----------------------------
all_images = sorted(
    [f.name for f in IMAGES_DIR.glob("*.png")],
    key=lambda x: int(Path(x).stem)
)

annotated = set(df["image"])

remaining = [
    img
    for img in all_images
    if img not in annotated
]

remaining = sorted(
    remaining,
    key=lambda x: int(Path(x).stem)
)

# -----------------------------
# UI
# -----------------------------
st.title("Tankri Dataset Annotator")

total = len(all_images)
done = len(annotated)

st.progress(done / total if total > 0 else 0)

st.write(f"Progress: {done} / {total}")

# -----------------------------
# Finished
# -----------------------------
if len(remaining) == 0:
    st.success("🎉 All images annotated!")
    st.stop()

current_image = remaining[0]

st.subheader(f"Current Image: {current_image}")

st.image(
    str(IMAGES_DIR / current_image),
    width=300
)

# -----------------------------
# Label Input
# -----------------------------
label = st.text_input(
    "Label",
    key=current_image
)

col1, col2 = st.columns(2)

# -----------------------------
# Save
# -----------------------------
with col1:
    if st.button("Save & Next"):

        if label.strip() == "":
            st.warning("Please enter a label.")
        else:

            new_row = pd.DataFrame([
                {
                    "image": current_image,
                    "label": label.strip()
                }
            ])

            df = pd.concat(
                [df, new_row],
                ignore_index=True
            )

            df.to_csv(
                CSV_PATH,
                index=False
            )

            st.rerun()

# -----------------------------
# Skip
# -----------------------------
with col2:
    if st.button("Skip"):
        st.rerun()