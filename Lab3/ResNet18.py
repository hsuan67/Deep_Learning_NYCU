import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
        
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias=False),
                nn.BatchNorm2d(out_channels, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.in_channels != self.out_channels:
            out += self.downsample(x)
        else:
            out += x

        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = (7, 7), stride = (2, 2), padding = (3, 3), bias = False)
        self.bn1 = nn.BatchNorm2d(64, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1, dilation = 1, ceil_mode = False)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, 2),
            BasicBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, 2),
            BasicBlock(256, 256)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, 2),
            BasicBlock(512, 512)
        )
        self.flatten = nn.Flatten()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size = (1, 1))
        self.fc = nn.Linear(in_features = 512, out_features = 2, bias = True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out