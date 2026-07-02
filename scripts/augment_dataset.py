import random
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

aug_df = pd.read_csv("dataset/augmented/labels.csv")

sample = aug_df.sample(16, random_state=42)

fig, axes = plt.subplots(4, 4, figsize=(10, 10))

for ax, (_, row) in zip(axes.flatten(), sample.iterrows()):

    img = Image.open(f"dataset/augmented/images/{row['image']}")

    ax.imshow(img, cmap="gray")
    ax.set_title(row["label"], fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.show()
