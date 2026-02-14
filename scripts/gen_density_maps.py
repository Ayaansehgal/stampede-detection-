import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from predict import CrowdCounter

counter = CrowdCounter(os.path.join(PROJECT_ROOT, 'weights', 'model_best.pth.tar'))
imgs_dir = os.path.join(PROJECT_ROOT, 'data', 'ShanghaiTech', 'part_A', 'test_data', 'images')
out_dir = os.path.join(PROJECT_ROOT, 'outputs', 'density_maps')
os.makedirs(out_dir, exist_ok=True)

for i in range(1, 11):
    img_path = os.path.join(imgs_dir, f'IMG_{i}.jpg')
    if not os.path.exists(img_path):
        print(f'IMG_{i}.jpg not found, skipping.')
        continue

    result = counter.predict(img_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(Image.open(img_path))
    axes[0].set_title(f'IMG_{i}.jpg')
    axes[0].axis('off')

    im = axes[1].imshow(result['density_map'], cmap='jet')
    axes[1].set_title(f"Count: {result['count']} | Density: {result['density_avg']} ppl/1000px\u00b2")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    out_path = os.path.join(out_dir, f'density_IMG_{i}.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"IMG_{i}.jpg -> Count: {result['count']}, Density: {result['density_avg']} | Saved: {out_path}")

print("\nDone!")
