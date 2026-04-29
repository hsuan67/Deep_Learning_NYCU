import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride = 1):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size = 1, stride = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(mid_channels, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(mid_channels, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size = 1, stride = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(out_channels, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
        self.relu = nn.ReLU(inplace = True)
        
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
            )
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.in_channels != self.out_channels:
            out += self.downsample(x)
        else:
            out += x

        out = self.relu(out)
        return out

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = (7, 7), stride = (2, 2), padding = (3, 3), bias = False)
        self.bn1 = nn.BatchNorm2d(64, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1, dilation = 1, ceil_mode = False)
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, 256),
            Bottleneck(256, 64, 256),
            Bottleneck(256, 64, 256)
        )
        self.layer2 = nn.Sequential(
            Bottleneck(256, 128, 512, 2),
            Bottleneck(512, 128, 512),
            Bottleneck(512, 128, 512),
            Bottleneck(512, 128, 512)
        )
        self.layer3 = nn.Sequential(
            Bottleneck(512, 256, 1024, 2),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024)
        )
        self.layer4 = nn.Sequential(
            Bottleneck(1024, 512, 2048, 2),
            Bottleneck(2048, 512, 2048),
            Bottleneck(2048, 512, 2048)
        )
        self.flatten = nn.Flatten()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size = (1, 1))
        self.fc = nn.Linear(in_features = 2048, out_features = 2, bias = True)

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
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out