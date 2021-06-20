'''
GoogLeNet implementation with PyTorch
Refer to
(1) "Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>
'''

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


def ConvBnReLU(inplanes, outplanes, kernel_size, stride=1, padding=0):
    block = [
        nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(outplanes),
        nn.ReLU(inplace=True)
    ]
    return block


class Inception(nn.Module):
    '''
    The implementation of Inception v1 block, refer to <http://arxiv.org/abs/1409.4842>
    '''
    def __init__(self, inplanes, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_planes):
        super(Inception, self).__init__()
        self.branch1 = nn.Sequential(*ConvBnReLU(inplanes, ch1x1, kernel_size=1, stride=1, padding=0))
        self.branch2 = nn.Sequential(
            *ConvBnReLU(inplanes, ch3x3red, kernel_size=1, stride=1, padding=0),
            *ConvBnReLU(ch3x3red, ch3x3, kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            *ConvBnReLU(inplanes, ch5x5red, kernel_size=1, stride=1, padding=0),
            *ConvBnReLU(ch5x5red, ch5x5, kernel_size=3, stride=1, padding=1),
            *ConvBnReLU(ch5x5, ch5x5, kernel_size=3, stride=1, padding=1)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            *ConvBnReLU(inplanes, pool_planes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat([out1, out2, out3, out4], dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=40):
        super(GoogLeNet, self).__init__()
        self.preact = nn.Sequential(
            *ConvBnReLU(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            *ConvBnReLU(64, 64, kernel_size=1),
            *ConvBnReLU(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )

        self._3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self._3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self._4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self._4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self._4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self._4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self._4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self._5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self._5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            Rearrange('b c h w -> b (c h w)'),
            nn.Dropout(p=0.4),
            nn.Linear(1024, num_classes)
        )


    def forward(self, x):
        out = self.preact(x)

        out = self._3a(out)
        out = self._3b(out)
        out = self.maxpool3(out)

        out = self._4a(out)
        out = self._4b(out)
        out = self._4c(out)
        out = self._4d(out)
        out = self._4e(out)
        out = self.maxpool4(out)

        out = self._5a(out)
        out = self._5b(out)

        out = self.avgpool(out)
        out = self.classifier(out)

        return out

def googlenet(num_classes=40):
    return GoogLeNet(num_classes=num_classes)