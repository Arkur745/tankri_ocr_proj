from pathlib import Path
import re

images_dir = Path("dataset/images")

files = list(images_dir.glob("*.png"))

def extract_time(file):
    match = re.search(
        r"Screenshot (\d{4}-\d{2}-\d{2}) (\d{6})",
        file.stem
    )

    if not match:
        raise ValueError(f"Unexpected filename: {file.name}")

    return match.group(1), match.group(2)

# Oldest first
files.sort(key=extract_time)

# Temporary rename
for i, file in enumerate(files, start=1):
    file.rename(images_dir / f"tmp_{i:04d}.png")

# Final rename
for i in range(1, len(files) + 1):
    (images_dir / f"tmp_{i:04d}.png").rename(
        images_dir / f"{i}.png"
    )

print(f"Renamed {len(files)} files.")