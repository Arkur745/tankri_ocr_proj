import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt

latex_dir = os.path.join(os.path.dirname(__file__), "..", "latex")
figures_dir = os.path.join(latex_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

# Copy image 1.png to figures directory
src_img1 = os.path.join(latex_dir, "image 1.png")
dst_img1 = os.path.join(figures_dir, "fig1_temple_wall.png")
if os.path.exists(src_img1):
    shutil.copy(src_img1, dst_img1)
    print("Copied image 1.png to fig1_temple_wall.png")

# Prepare 3x3 Grid Image
row1_files = [os.path.join(latex_dir, f"char{i}.png") for i in [1, 2, 3]]
row2_files = [os.path.join(latex_dir, f"book{i}.png") for i in [1, 2, 3]]
row3_files = [os.path.join(latex_dir, f"wall{i}.png") for i in [1, 2, 3]]

grid_files = [row1_files, row2_files, row3_files]
row_titles = ["Handwritten Characters (Raw)", "Manuscript Book Scans", "Historical Wall Inscriptions"]

fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.5))
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 9})

for r in range(3):
    for c in range(3):
        ax = axes[r, c]
        img_path = grid_files[r][c]
        if os.path.exists(img_path):
            img = Image.open(img_path)
            ax.imshow(img, cmap='gray' if img.mode != 'RGB' else None)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Row title on middle column top
        if r == 0 and c == 1:
            ax.set_title("Row 1: Real Handwritten Glyphs", fontweight='bold', fontsize=10, pad=6)
        elif r == 1 and c == 1:
            ax.set_title("Row 2: Printed Manuscript Book Scans", fontweight='bold', fontsize=10, pad=6)
        elif r == 2 and c == 1:
            ax.set_title("Row 3: Stone Wall Inscriptions", fontweight='bold', fontsize=10, pad=6)

plt.tight_layout()
dst_grid = os.path.join(figures_dir, "fig2_dataset_grid.png")
plt.savefig(dst_grid, dpi=300)
plt.close()
print("Generated 3x3 grid figure successfully:", dst_grid)
