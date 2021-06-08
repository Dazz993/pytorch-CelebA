'''
Alexnet implementation with PyTorch
Refer to
(1) https://kr.nvidia.com/content/tesla/pdf/machine-learning/imagenet-classification-with-deep-convolutional-nn.pdf
(2) https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),      # (b, 3, 224, 224) -> (b, 96, 55, 55)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                      # (b, 96, 55, 55) -> (b, 96, 27, 27)
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),     # (b, 96, 27, 27) -> (b, 256, 27, 27)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                      # (b, 256, 27, 27) -> (b, 256, 13, 13)
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),    # (b, 256, 13, 13) -> (b, 384, 13, 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),    # (b, 384, 13, 13) -> (b, 384, 13, 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),    # (b, 384, 13, 13) -> (b, 256, 13, 13)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)                       # (b, 256, 13, 13) -> (b, 256, 6, 6)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))                     # (b, 256, 6, 6) -> (b, 256, 6, 6)
        self.classifier = nn.Sequential(
            Rearrange('b c h w -> b (c h w)'),
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = self.classifier(out)

        return out

def alexnet(num_classes=40):
    return AlexNet(num_classes=num_classes)