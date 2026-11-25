import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, variant='baseline'):
        super().__init__()
        if variant not in ('baseline', 'frequency'):
            raise ValueError('Unknown CNN variant')
        frequency_bins = 1 if variant == 'baseline' else 8
        layers = []
        channels = 1
        for width in (16, 32, 64):
            layers.extend([nn.Conv2d(channels, width, 3, padding=1),
                           nn.ReLU(), nn.MaxPool2d(2)])
            channels = width
        self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((frequency_bins, 1)))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(64 * frequency_bins, 2))

    def forward(self, x):
        return self.head(self.encoder(x))
