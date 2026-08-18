# TMamba

Official PyTorch implementation of **“RGB-Thermal Semantic Segmentation via Wide-Angle Spatial Scanning and Modality-Interwoven Scanning”**, accepted by PRCV 2026.

> The paper has been accepted but is not publicly available yet. The paper link will be updated after publication.

| Resource | Link |
|---|---|
| Paper | Coming soon <!-- PAPER_URL --> |
| MFNet dataset archive | [Google Drive](https://drive.google.com/file/d/1d6HZINFPQklknuI_uD3cY-Cz_QeXM-pr/view?usp=drive_link) |
| PST900 dataset archive | [Google Drive](https://drive.google.com/file/d/1r5DfFCXmAJbCNht8pO0WiXTOPXYOkz6q/view?usp=drive_link) |
| VMamba-Tiny ImageNet pretrained weights | [Google Drive](https://drive.google.com/file/d/1jzCQGGctBrqcbZOyRtuZ2CLEvjBgayFI/view?usp=drive_link) |
| TMamba checkpoint on MFNet | [Google Drive](https://drive.google.com/file/d/1lfUIvsibxlPeXkLm5Pr1xvP-FexsXBsI/view?usp=drive_link) |

## Overview

TMamba is a pure VMamba-based model for RGB-T semantic segmentation, designed to preserve small objects and fine-grained details during multi-modal feature learning. Its main components are:

- **Three-Stream Adapter:** inserts one shared and two modality-specific adapter branches into a frozen VMamba backbone to improve modality alignment while preserving complementary details.
- **Cross-Modal Guidance Module (CGM):** mutually refines RGB and thermal features using **Wide-Angle Spatial Scanning (WAS)** for joint global-local context learning.
- **Inter-Modal Fusion Module (IFM):** uses **Modality-Interwoven Scanning (MIS)** to promote deep cross-modal interaction and produce a unified fused representation.

### Architecture

[View the TMamba architecture in PDF format](Figs/Models.pdf)

## Efficiency Analysis

Comparison of different backbones and fine-tuning strategies on MFNet:

| Backbone | FT | Adapter | Tuned Params (M) | GFLOPs | mIoU (%) |
|---|:---:|:---:|---:|---:|---:|
| Baseline |  |  | 29.8 | 49.7 | 59.9 |
| VMamba-T | ✓ |  | 52.0 | 49.7 | 60.6 |
| ResNet-50 |  | ✓ | 56.0 | 116.9 | 57.6 |
| Swin-T |  | ✓ | 33.1 | 130.2 | 60.2 |
| MiT-B2 |  | ✓ | 16.4 | 105.5 | 61.3 |
| VMamba-T |  | ✓ | 37.7 | 71.3 | **63.3** |

## Installation

The released code was tested with Python 3.10, PyTorch 2.8.0, CUDA 12.8, and an NVIDIA RTX 5090 GPU.

```bash
conda create -n rgbmamba python=3.10
conda activate rgbmamba
pip install -r requirements.txt
```

The selective-scan CUDA extensions must be compiled for the local PyTorch and CUDA versions.

## Data Preparation

Download the dataset archives from the links above and extract them as follows:

```text
Datasets/
├── dataset/                       # MFNet
│   ├── seperated_images/
│   ├── labels/
│   ├── edge/
│   ├── bound/
│   ├── binary_labels/
│   ├── trainval.txt
│   ├── test.txt
│   ├── test_day.txt
│   └── test_night.txt
└── PST900_RGBT_Dataset/           # PST900
    ├── rgb/
    ├── thermal/
    ├── labels/
    ├── train.txt
    └── test.txt
```

MFNet contains 1,569 aligned RGB-T image pairs at a resolution of 480 × 640. The command-line dataset identifier `IRS` in this repository refers to MFNet.

## Pretrained Models and Checkpoints

Place the downloaded VMamba-Tiny ImageNet pretrained weights at:

```text
pretrained/vssmtiny_dp01_ckpt_epoch_292.pth
```

Place the downloaded TMamba checkpoint at the repository root:

```text
Net_best.pth
```

## Training and Evaluation

Train TMamba on MFNet using the settings reported in the paper:

```bash
python train.py \
    --dataset IRS \
    --exp_name tmamba_mfnet \
    --batch_size 4 \
    --lr 3e-4 \
    --weight_decay 1e-4 \
    --epochs 200
```

Evaluate the released checkpoint:

```bash
python test.py \
    --dataset IRS \
    --exp_name tmamba_mfnet \
    --checkpoint_name "$PWD/Net_best.pth"
```

The released MFNet checkpoint achieves **63.3% mIoU** on the test set.
