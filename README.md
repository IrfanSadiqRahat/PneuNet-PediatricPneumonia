# PneuNet-PediatricPneumonia

> **PneuNet: A Multi-Scale Attention-Enhanced CNN for Pediatric Pneumonia Detection from Chest X-rays**
>
> Irfan Sadiq Rahat, et al.
>
> ***IEEE DELCON 2025*** (4th Delhi Section Conference) · Paper #235 · Accepted

[![Conference](https://img.shields.io/badge/Conference-IEEE%20DELCON%202025-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Abstract

PneuNet is a lightweight multi-scale attention CNN for binary classification of pediatric chest X-rays into **Normal** and **Pneumonia**. Designed for resource-constrained clinical settings with high sensitivity as the primary optimization target (minimizing false negatives critical in pediatric care).

---

## Key Features

- Multi-scale feature extraction (parallel 3×3 and 5×5 branches)
- Spatial + channel attention for lesion localization
- High sensitivity optimization (recall for Pneumonia class)
- Lightweight: ~4M parameters, suitable for edge deployment
- Grad-CAM visualization for clinical interpretability

---

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 95.7% |
| Sensitivity (Pneumonia) | **98.1%** |
| Specificity | 92.4% |
| AUC | 0.973 |

---

## Dataset

Kaggle Chest X-Ray Pneumonia Dataset (5,856 images)
Download: [kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Classes: Normal (1,583) · Pneumonia (4,273)

---

## Setup

```bash
git clone https://github.com/IrfanSadiqRahat/PneuNet-PediatricPneumonia.git
cd PneuNet-PediatricPneumonia
pip install -r requirements.txt
python train.py --data_dir data/chest_xray
python evaluate.py --checkpoint outputs/best_model.pth --gradcam
```

---

## Citation

```bibtex
@inproceedings{rahat2025pneunet,
  title={PneuNet: A Multi-Scale Attention-Enhanced CNN for Pediatric Pneumonia Detection from Chest X-rays},
  author={Rahat, Irfan Sadiq and others},
  booktitle={Fourth IEEE Delhi Section Conference (DELCON 2025)},
  year={2025},
  organization={IEEE}
}
```
