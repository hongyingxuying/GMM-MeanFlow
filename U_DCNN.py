import torch
import torch.nn as nn
import random
import numpy as np
import math

class Reshape(torch.nn.Module):
    def __init__(self, batch, channel, size_x, size_y=None):
        super().__init__()
        self.batch = batch
        self.channel = channel
        self.size_x = size_x
        self.size_y = size_y

    def forward(self, x):
        if self.size_y != None:
            return x.view(self.batch, self.channel, self.size_x, self.size_y)
        else:
            return x.view(self.batch, self.channel, self.size_x)

class Classifier(nn.Module):
    def __init__(self, numOfClasses):
        super().__init__()
        self.flatten = nn.Flatten()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
        )
        # Make head input size independent of input length (1024/4096/etc)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, numOfClasses),
        )

    def forward(self, x):
        # x: (B, L) or (B, 1, L) or (B, 1, H, W)
        if x.dim() == 3:
            # (B, 1, L) -> (B, L)
            x = x.squeeze(1)

        if x.dim() == 2:
            length = int(x.shape[-1])
            side = int(math.isqrt(length))
            if side * side != length:
                raise ValueError(f"Classifier expects square length (e.g. 1024/4096), got L={length}")
            x = x.view(x.shape[0], 1, side, side)
        elif x.dim() != 4:
            raise ValueError(f"Unexpected input shape for Classifier: {tuple(x.shape)}")

        x = self.features(x)
        x = self.pool(x)
        return self.head(x)