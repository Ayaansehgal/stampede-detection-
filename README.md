# 🚨 Stampede Detection — Crowd Density Estimation

A deep learning-based crowd counting and density estimation system using **CSRNet** (Dilated Convolutional Neural Networks). Given an image of a crowd, the model outputs:

- **People count** — estimated number of people in the image
- **Density metrics** — average and peak crowd density values
- **Density map** — visual heatmap showing crowd distribution

## 📐 Model Architecture

**CSRNet** (CVPR 2018) — a two-part convolutional neural network:

| Component | Details |
|-----------|---------|
| **Frontend** | VGG-16 (first 10 conv layers, pretrained on ImageNet) |
| **Backend** | 6 dilated convolution layers (dilation rate = 2) |
| **Output** | Single-channel density map → sum = people count |

> The model generates a spatial density map where each pixel represents a local crowd density value. Summing all pixel values gives the total estimated count.

## 📁 Project Structure

```
Yantra/
├── model.py                    # CSRNet architecture (VGG-16 frontend + dilated conv backend)
├── predict.py                  # Inference: image → people count + density metrics
├── evaluate.py                 # Accuracy evaluation (ShanghaiTech & UCF-QNRF)
├── README.md
├── .gitignore
│
├── weights/                    # Model weights (not in repo — download separately)
│   └── model_best.pth.tar      # Pretrained ShanghaiTech-A checkpoint (~130MB)
│
├── scripts/                    # Utility scripts
│   └── gen_density_maps.py     # Batch density map generation
│
├── outputs/                    # Generated results (not in repo)
│   └── density_maps/           # Density map visualizations
│       ├── density_IMG_1.png
│       └── ...
│
└── data/                       # Datasets (not in repo — download separately)
    ├── ShanghaiTech/
    │   ├── part_A/
    │   │   ├── test_data/
    │   │   └── train_data/
    │   └── part_B/
    └── UCF-QNRF_ECCV18/
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision numpy scipy Pillow matplotlib h5py
```

### 2. Download Pretrained Weights

Download the ShanghaiTech Part A weights (~130MB) from [Google Drive](https://drive.google.com/open?id=1Z-atzS5Y2pOd-nEWqZRVBDMYJDreGWHH) and place in `weights/model_best.pth.tar`.

### 3. Count People in an Image

```bash
python predict.py --image path/to/crowd.jpg
```

**Output:**
```
==================================================
  Image:    path/to/crowd.jpg
  People count:     1388
  Avg density:      0.1842 people/1000px²
  Peak density:     0.2671
==================================================
```

### 4. Save Density Map Visualization

```bash
python predict.py --image path/to/crowd.jpg --save-density
```

This saves a side-by-side comparison of the original image and density heatmap.

## 📊 Evaluate Accuracy

### On ShanghaiTech Dataset

```bash
# Full test set (182 images)
python evaluate.py --dataset shanghai --data data/ShanghaiTech/part_A/test_data

# Quick test (first N images)
python evaluate.py --dataset shanghai --data data/ShanghaiTech/part_A/test_data --max-images 10
```

### On UCF-QNRF Dataset

```bash
python evaluate.py --dataset ucf-qnrf --data data/UCF-QNRF_ECCV18 --split Test
```

### Benchmark Results

| Dataset | MAE | Published MAE |
|---------|:---:|:------------:|
| ShanghaiTech Part A | **69.80** | 66.4 |
| ShanghaiTech Part B | — | 10.6 |

## 🐍 Python API

```python
from predict import CrowdCounter

# Initialize
counter = CrowdCounter("weights/model_best.pth.tar")

# Get just the count
count = counter.count("crowd.jpg")
print(f"People: {count}")

# Get full results (count + density + density map)
result = counter.predict("crowd.jpg")
print(f"People: {result['count']}")
print(f"Avg density: {result['density_avg']} people/1000px²")
print(f"Peak density: {result['density_peak']}")
```

## 📚 Datasets

| Dataset | Images | Annotations | Download |
|---------|:------:|:-----------:|:--------:|
| ShanghaiTech | Part A: 482, Part B: 716 | Head point coordinates | [Google Drive](https://drive.google.com/open?id=16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI) |
| UCF-QNRF | 1,535 | Head point coordinates | [Link](https://www.crcv.ucf.edu/data/ucf-qnrf/) |

## 📝 References

```
@inproceedings{li2018csrnet,
  title={CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes},
  author={Li, Yuhong and Zhang, Xiaofan and Chen, Deming},
  booktitle={CVPR},
  pages={1091--1100},
  year={2018}
}
```

- [CSRNet Paper (arXiv)](https://arxiv.org/abs/1802.10062)
- [CSRNet PyTorch Implementation](https://github.com/leeyeehoo/CSRNet-pytorch)