import sys
from pathlib import Path
import re

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RENAME_TARGET, IMAGES_DIR, DATASET_DIR

if RENAME_TARGET == "test":
    images_dir = DATASET_DIR / "test"
    print("Renaming target: TEST dataset")
else:
    images_dir = IMAGES_DIR
    print("Renaming target: TRAINING dataset")

print(f"Target directory: {images_dir}")

files = list(images_dir.glob("*.png"))

# ----------------------------
# Find already numbered images
# ----------------------------

numbered_files = []

for file in files:
    if file.stem.isdigit():
        numbered_files.append(int(file.stem))

start_number = max(numbered_files, default=0)

print(f"Existing images: {start_number}")

# ----------------------------
# Find new screenshots
# ----------------------------

new_files = [
    f for f in files
    if f.stem.startswith("Screenshot")
]


def extract_time(file):
    match = re.search(
        r"Screenshot (\d{4}-\d{2}-\d{2}) (\d{6})",
        file.stem
    )

    if not match:
        raise ValueError(f"Unexpected filename: {file.name}")

    return match.group(1), match.group(2)


# Sort chronologically
new_files.sort(key=extract_time)

# ----------------------------
# Temporary rename
# ----------------------------

for i, file in enumerate(new_files, start=1):
    file.rename(images_dir / f"tmp_{i:04d}.png")

# ----------------------------
# Final rename
# ----------------------------

for i in range(1, len(new_files) + 1):

    new_name = start_number + i

    (images_dir / f"tmp_{i:04d}.png").rename(
        images_dir / f"{new_name}.png"
    )

print(f"Renamed {len(new_files)} new images.")
print(f"Last image number: {start_number + len(new_files)}")
