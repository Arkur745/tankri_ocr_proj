import os
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

outputs_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "sample_tuning")
latex_dir = os.path.join(os.path.dirname(__file__), "..", "latex")
figures_dir = os.path.join(latex_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

steps = [
    ("01_render.png", "Stage 1: Stroke Template"),
    ("02_transformed.png", "Stage 2: Glyph Perturbation"),
    ("03_blended.png", "Stage 3: Substrate Blending"),
    ("04_surface_degraded.png", "Stage 4: Surface Degradation"),
    ("06_final.png", "Stage 5: Sensor Simulation")
]

fig, axes = plt.subplots(1, 5, figsize=(10, 2.5))
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 8.5})

for i, (fn, title) in enumerate(steps):
    ax = axes[i]
    img_path = os.path.join(outputs_dir, fn)
    if os.path.exists(img_path):
        img = Image.open(img_path)
        
        # Convert to numpy array to handle transparency and stroke contrast
        img_arr = np.array(img)
        
        # Handle Stage 1 and Stage 2 (white strokes on transparent/black background)
        if i in [0, 1]:
            # If RGBA, extract alpha or convert to grayscale
            if len(img_arr.shape) == 3 and img_arr.shape[2] == 4:
                # Alpha channel indicates stroke
                alpha = img_arr[:, :, 3]
                # Create white background (255) with black strokes (0)
                processed = np.ones_like(alpha) * 255
                processed[alpha > 30] = 0
            else:
                # Grayscale / RGB: if mean is dark, invert to black stroke on white
                gray = np.array(img.convert('L'))
                if np.mean(gray) < 128:
                    processed = 255 - gray
                else:
                    processed = gray
            ax.imshow(processed, cmap='gray')
        else:
            ax.imshow(img, cmap='gray' if img.mode != 'RGB' else None)

    ax.set_title(title, fontweight='bold', fontsize=8.5, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add subtle box border
    for spine in ax.spines.values():
        spine.set_edgecolor('#34495e')
        spine.set_linewidth(1)

plt.suptitle("Figure 3: Procedural Synthetic Domain Simulation Pipeline Transformation Stages", fontweight='bold', fontsize=10.5, y=1.03)
plt.tight_layout()

dst_path = os.path.join(figures_dir, "fig3_synthetic_pipeline_flow.png")
plt.savefig(dst_path, dpi=300, bbox_inches='tight')
plt.close()
print("Generated synthetic pipeline flow figure successfully with high-contrast Stage 1 & 2:", dst_path)
