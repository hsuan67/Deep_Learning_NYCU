import Lab2_312554029_code
from Lab2_312554029_code import cfg
import torch
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, act = cfg.act):
        super().__init__()
        
        self.firstconv = nn.Sequential( # 進行一些低級特徵的提取
            nn.Conv2d(1, 16, kernel_size = (1,51), stride = (1,1), padding = (0,25), bias = False), # torch.nn.Conv2d(in_channels, out_channels,...)
            nn.BatchNorm2d(16, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True)   # 加速訓練和減少梯度消失問題
        )
        self.depthwiseConv = nn.Sequential(
            # 對 16 個 channels 分組(一個 channel 一組)，每個 channel 分別獨立跟 2*1 的 kernel 做卷積，所以最後輸出有 32 個 channels
            nn.Conv2d(16, 32, kernel_size = (2,1), stride=(1,1), groups = 16, bias = False),
            nn.BatchNorm2d(32, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True),
            self.activation(act),
            nn.AvgPool2d(kernel_size = (1,4), stride = (1,4), padding = 0),
            nn.Dropout(p = 0.25)
        )
        self.separableConv = nn.Sequential(
            # 前一層得到的 32 個 channels 分別跟 1*15 的 kernel 做卷積，所以最後輸出仍為 32 個 channels
            nn.Conv2d(32, 32, kernel_size = (1,15), stride = (1,1), padding = (0,7), bias = False),
            nn.BatchNorm2d(32, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = True),
            self.activation(act),
            nn.AvgPool2d(kernel_size = (1,8), stride = (1,8), padding = 0),
            nn.Dropout(p = 0.25)
        )
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = 736, out_features = 2, bias = True)
        )
    
    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.classify(x)
        return x

    def activation(self, act):
        if act == "ReLU":
            return nn.ReLU()
        elif act == "Leaky ReLU":
            return nn.LeakyReLU()
        elif act == "ELU":
            return nn.ELU()

model = EEGNet(act = "ReLU").to('cuda')
model.load_state_dict(torch.load('/home/pp037/DL/Lab2/model.pkl'))

Lab2_312554029_code.test(model, Lab2_312554029_code.test_loader)