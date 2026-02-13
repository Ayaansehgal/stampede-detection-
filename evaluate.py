"""
Accuracy Evaluation Script for CSRNet on UCF-QNRF Dataset
==========================================================
Evaluates the pretrained model against ground truth annotations.

Metrics computed:
    - MAE  (Mean Absolute Error)
    - MSE  (Mean Squared Error)
    - RMSE (Root Mean Squared Error)

Usage:
    python evaluate.py --model model_best.pth.tar --data data/UCF-QNRF_ECCV18 --split Test
    python evaluate.py --model model_best.pth.tar --data data/UCF-QNRF_ECCV18 --split Test --max-images 10
"""

import argparse
import os
import glob
import time

import torch
import torchvision.transforms as transforms
import numpy as np
import scipy.io
from PIL import Image

from model import CSRNet


def load_ground_truth(ann_path):
    """
    Load ground truth count from a UCF-QNRF .mat annotation file.
    The .mat file contains 'annPoints' with shape (N, 2) — one row per head.
    Returns the number of annotated heads.
    """
    mat = scipy.io.loadmat(ann_path)
    ann_points = mat['annPoints']
    return ann_points.shape[0]


def evaluate(model_path, data_dir, split='Test', device=None, max_images=None):
    """
    Evaluate CSRNet on the UCF-QNRF dataset.

    Args:
        model_path: path to model checkpoint
        data_dir:   path to UCF-QNRF_ECCV18 directory
        split:      'Test' or 'Train'
        device:     'cuda' or 'cpu'
        max_images: limit number of images (for quick testing)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Load model
    print(f"Loading model from {model_path}...")
    model = CSRNet(load_weights=True)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Find all images
    split_dir = os.path.join(data_dir, split)
    image_paths = sorted(glob.glob(os.path.join(split_dir, 'img_*.jpg')))

    if max_images is not None:
        image_paths = image_paths[:max_images]

    total = len(image_paths)
    print(f"Evaluating on {total} images from {split_dir}...\n")

    # Metrics
    abs_errors = []
    sq_errors = []
    results = []

    print(f"{'#':>4}  {'Image':<20}  {'GT':>6}  {'Pred':>6}  {'Error':>8}  {'Time':>6}")
    print("-" * 70)

    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)

        # Ground truth
        ann_name = img_name.replace('.jpg', '_ann.mat')
        ann_path = os.path.join(split_dir, ann_name)

        if not os.path.exists(ann_path):
            print(f"  WARNING: Annotation not found for {img_name}, skipping.")
            continue

        gt_count = load_ground_truth(ann_path)

        # Predict
        t0 = time.time()
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            density_map = model(img_tensor)
        pred_count = int(round(density_map.sum().item()))
        elapsed = time.time() - t0

        # Error
        error = pred_count - gt_count
        abs_error = abs(error)
        abs_errors.append(abs_error)
        sq_errors.append(error ** 2)
        results.append({
            'image': img_name,
            'gt': gt_count,
            'pred': pred_count,
            'error': error,
        })

        print(f"{idx+1:>4}  {img_name:<20}  {gt_count:>6}  {pred_count:>6}  {error:>+8}  {elapsed:>5.1f}s")

    # Summary
    mae = np.mean(abs_errors)
    mse = np.mean(sq_errors)
    rmse = np.sqrt(mse)

    print(f"\n{'='*70}")
    print(f"  EVALUATION RESULTS ({split} set, {len(abs_errors)} images)")
    print(f"{'='*70}")
    print(f"  MAE  (Mean Absolute Error):  {mae:.2f}")
    print(f"  MSE  (Mean Squared Error):   {mse:.2f}")
    print(f"  RMSE (Root Mean Sq. Error):  {rmse:.2f}")
    print(f"{'='*70}")

    # Best and worst predictions
    if results:
        sorted_by_error = sorted(results, key=lambda x: abs(x['error']))
        print(f"\n  Top 5 BEST predictions (lowest error):")
        for r in sorted_by_error[:5]:
            print(f"    {r['image']}: GT={r['gt']}, Pred={r['pred']}, Error={r['error']:+d}")

        print(f"\n  Top 5 WORST predictions (highest error):")
        for r in sorted_by_error[-5:]:
            print(f"    {r['image']}: GT={r['gt']}, Pred={r['pred']}, Error={r['error']:+d}")

    print()
    return mae, mse, rmse


def main():
    parser = argparse.ArgumentParser(description='Evaluate CSRNet on UCF-QNRF dataset')
    parser.add_argument('--model', '-m', required=True, help='Path to model checkpoint')
    parser.add_argument('--data', '-d', default='data/UCF-QNRF_ECCV18', help='Path to UCF-QNRF directory')
    parser.add_argument('--split', '-s', default='Test', choices=['Test', 'Train'], help='Dataset split')
    parser.add_argument('--device', default=None, help='Device: cuda or cpu')
    parser.add_argument('--max-images', type=int, default=None, help='Max images to evaluate (for quick test)')
    args = parser.parse_args()

    evaluate(args.model, args.data, args.split, args.device, args.max_images)


if __name__ == '__main__':
    main()
