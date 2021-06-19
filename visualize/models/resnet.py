'''
ResNet implementation with PyTorch
Refer to
(1) https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html
(2) https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        shortcut_layers = []
        if stride != 1 or inplanes != planes * self.expansion:
            shortcut_layers.append(nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False))
            shortcut_layers.append(nn.BatchNorm2d(planes * self.expansion))
        self.shortcut = nn.Sequential(*shortcut_layers)

    def forward(self, x):
        out = F.relu((self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        shortcut_layers = []
        if stride != 1 or inplanes != planes * self.expansion:
            shortcut_layers.append(nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False))
            shortcut_layers.append(nn.BatchNorm2d(planes * self.expansion))
        self.shortcut = nn.Sequential(*shortcut_layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, planes=64, num_blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(block, planes=128, num_blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes=256, num_blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes=512, num_blocks=layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out).reshape(x.shape[0], -1)
        out = self.linear(out)

        return out

def resnet18(num_classes=40):
    return ResNet(BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)

def resnet34(num_classes=40):
    return ResNet(BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes)

def resnet50(num_classes=40):
    return ResNet(Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes)

def resnet101(num_classes=40):
    return ResNet(Bottleneck, layers=[3, 4, 23, 3], num_classes=num_classes)

def resnet152(num_classes=40):
    return ResNet(Bottleneck, layers=[3, 8, 36, 3], num_classes=num_classes)