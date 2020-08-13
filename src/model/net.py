import os

import torch
import torch.nn as nn
from torchvision import models

alexnet_model = models.alexnet(pretrained=True)


class DeepHash(nn.Module):
    def __init__(self, hash_bits: int):
        """
        Args:
            bits (int): lenght of encoded binary bits    
        """
        super(DeepHash, self).__init__()
        self.hash_bits = hash_bits
        self.features = nn.Sequential(*list(alexnet_model.features.children()))
        self.remain = nn.Sequential(*list(alexnet_model.classifier.children())[:-1])
        self.Linear1 = nn.Linear(4096, self.hash_bits)
        self.sigmoid = nn.Sigmoid()
        self.Linear2 = nn.Linear(self.hash_bits, 10)

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.remain(x)
        x = self.Linear1(x)
        features = self.sigmoid(x)
        result = self.Linear2(features)
        return features, result
