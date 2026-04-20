"""
PneuNet: Multi-Scale Attention CNN for Pediatric Pneumonia Detection.
Published: IEEE DELCON 2025, Paper #235.
Author: Irfan Sadiq Rahat
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """Spatial attention — highlights where the lesions are."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mx  = x.max(1, keepdim=True)[0]
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class PneuNet(nn.Module):
    """
    Lightweight multi-scale attention CNN for pneumonia binary classification.
    Input:  (B, 3, 224, 224)
    Output: (B, 2) logits [Normal, Pneumonia]
    """
    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        # Parallel multi-scale extraction
        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.branch5 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2, bias=False), nn.BatchNorm2d(32), nn.ReLU())

        # Deep feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.spatial_attn = SpatialAttention()

        # Classifier with high recall optimization
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.branch3(x), self.branch5(x)], dim=1)
        x = self.features(x)
        x = self.spatial_attn(x)
        return self.classifier(x)
