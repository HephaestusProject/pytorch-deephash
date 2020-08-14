import os

import omegaconf

import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary as torch_summary

alexnet_model = models.alexnet(pretrained=True)


class DeepHash(nn.Module):
    def __init__(self, config: omegaconf.DictConfig):
        """
        Args:
            config (omegaconf.DictConfig): model configuration 
        """
        super(DeepHash, self).__init__()
        self.hash_bits = config.model.params.hash_bits

        self.width = config.model.params.width
        self.height = config.model.params.height
        self.channels = config.model.params.channels

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

    def summary(self):
        device = str(self.parameters().__next__().device)
        torch_summary(self, input_size=(self.channels, self.height, self.width), device=device)
