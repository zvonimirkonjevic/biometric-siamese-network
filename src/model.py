"""
This module defines Siamese network architecture for one-shot learning of fingerprint detection.
Network architecture is inspired by https://perso.esiee.fr/~chierchg/deep-learning/tutorials/metric/metric-1.html blog.
"""

import torch
from torch import nn
import torch.nn.functional as F


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.features(x)
    

class SiameseNetwork(nn.Module):
    def __init__(self, emb_dim=128, head_dropout=0.3, img_size=96):
        super().__init__()
        self.backbone = Backbone()

        # Compute the output feature dimension of the backbone to define the head architecture
        with torch.no_grad():
            feat_dim = self.backbone(torch.zeros(1, 1, img_size, img_size)).flatten(1).shape[1]
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(512, emb_dim)
        )

    def forward(self, x):
        return F.normalize(self.head(self.backbone(x)), p=2, dim=1)