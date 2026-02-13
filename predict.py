"""
Crowd Counting Inference Script
================================
Given an image, outputs the estimated number of people.

Usage:
    python predict.py --image path/to/crowd.jpg --model model_best.pth.tar
    python predict.py --image path/to/crowd.jpg --model model_best.pth.tar --save-density

Programmatic usage:
    from predict import CrowdCounter
    counter = CrowdCounter("model_best.pth.tar")
    count = counter.count("crowd.jpg")
"""

import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from model import CSRNet


class CrowdCounter:
    """
    Simple interface to count people in an image using a trained CSRNet model.
    """

    def __init__(self, model_path, device=None):
        """
        Args:
            model_path: path to the .pth.tar checkpoint file
            device: 'cuda' or 'cpu' (auto-detects if None)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # Load model architecture (load_weights=True skips VGG download)
        self.model = CSRNet(load_weights=True)

        # Load trained checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        # Standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def count(self, image_input):
        """
        Count people in an image.

        Args:
            image_input: file path (str), PIL Image, or numpy array

        Returns:
            int: estimated number of people
        """
        result = self._predict(image_input)
        return result['count']

    def predict(self, image_input):
        """
        Get the count, density info, and density map.

        Args:
            image_input: file path (str), PIL Image, or numpy array

        Returns:
            dict with keys:
                'count'       - int, estimated number of people
                'density_avg' - float, average density (people per 1000px²)
                'density_peak'- float, peak density value in the density map
                'density_map' - numpy array, the raw density map
        """
        return self._predict(image_input)

    def _predict(self, image_input):
        # Convert to PIL Image
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        else:
            raise ValueError(f"Unsupported input type: {type(image_input)}")

        img_w, img_h = image.size

        # Preprocess and run inference
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            density_map = self.model(img_tensor)

        count = int(round(density_map.sum().item()))
        density_np = density_map.squeeze().cpu().numpy()

        # Density metrics
        total_pixels = img_w * img_h
        density_avg = (count / total_pixels) * 1000  # people per 1000 px²
        density_peak = float(density_np.max())

        return {
            'count': count,
            'density_avg': round(density_avg, 4),
            'density_peak': round(density_peak, 4),
            'density_map': density_np,
        }


def main():
    parser = argparse.ArgumentParser(description='Count people in a crowd image')
    parser.add_argument('--image', '-i', required=True, help='Path to input image')
    parser.add_argument('--model', '-m', required=True, help='Path to model checkpoint (.pth.tar)')
    parser.add_argument('--device', '-d', default=None, help='Device: cuda or cpu (auto-detect)')
    parser.add_argument('--save-density', action='store_true', help='Save density map as PNG')
    args = parser.parse_args()

    # Run inference
    counter = CrowdCounter(args.model, device=args.device)
    result = counter.predict(args.image)
    count = result['count']
    density_map = result['density_map']

    print(f"\n{'='*50}")
    print(f"  Image:    {args.image}")
    print(f"  People count:     {count}")
    print(f"  Avg density:      {result['density_avg']} people/1000px²")
    print(f"  Peak density:     {result['density_peak']}")
    print(f"{'='*50}\n")

    # Optionally save density map visualization
    if args.save_density:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Original image
            img = Image.open(args.image).convert('RGB')
            axes[0].imshow(img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # Density map
            im = axes[1].imshow(density_map, cmap='jet')
            axes[1].set_title(f'Density Map (Count: {count})')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

            out_path = args.image.rsplit('.', 1)[0] + '_density.png'
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Density map saved to: {out_path}\n")
        except ImportError:
            print("  Install matplotlib to save density maps: pip install matplotlib\n")


if __name__ == '__main__':
    main()
