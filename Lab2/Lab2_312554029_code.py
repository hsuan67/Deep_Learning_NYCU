import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import random
import os
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import matplotlib.pyplot as plt

def read_bci_data():
    S4b_train = np.load('S4b_train.npz')
    X11b_train = np.load('X11b_train.npz')
    S4b_test = np.load('S4b_test.npz')
    X11b_test = np.load('X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    # 將 label 從 1-based 編碼轉換為 0-based 編碼
    train_label = train_label - 1
    test_label = test_label - 1

    # 擴充維度: (batch_size, 1, num_channels, height, width)，從兩維變三維
    # 對調後面兩維的數，因為進來是直的，把它轉為橫的
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))
   
    mask = np.where(np.isnan(train_data))   # 用平均值補缺失值
    train_data[mask] = np.nanmean(train_data)
    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    return train_data, train_label, test_data, test_label

train_data = {} # shape = (1080,1,2,750) 為 (樣本數,channel,width,height)，將每一個樣本視為 2*750 大小的圖
test_data = {}
train_data['signal'], train_data['label'], test_data['signal'], test_data['label'] = read_bci_data()

class cfg:
    seed = 1002
    batch_size = 512
    act = "ELU"
    epochs = 150 # 150/300/600/1000
    T0 = 20
    lr = 1.02e-2
    min_lr = 8e-8
def setSeed(seed = cfg.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
setSeed()

class DataSet:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['signal'])

    def __getitem__(self, index):
        signal = self.data['signal'][index] # 第 index 個 data 的 signal
        label = self.data['label'][index]
        return signal, label

train_dataset = DataSet(train_data)
test_dataset = DataSet(test_data)

# dataloader 一次抓 batch_size 的資料量，本身具有可迭代的特性(不用自己切 batch)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = cfg.batch_size, pin_memory = True, drop_last = False, num_workers = 4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = cfg.batch_size, pin_memory = True, drop_last = False, num_workers = 4)

# ----------------------------------------------------

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

# ----------------------------------------------------

class DeepConvNet(nn.Module):
    def __init__(self, act = cfg.act):
        super().__init__()
        
        self.Conv1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size = (1,5), stride = (1,1), padding = (0,0), bias = True),
            nn.Conv2d(25, 25, kernel_size = (2,1), stride = (1,1), padding = (0,0), bias = True),
            nn.BatchNorm2d(25), 
            self.activation(act),
            nn.MaxPool2d(kernel_size = (1,2)), 
            nn.Dropout(p = 0.1)
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size = (1,5), stride = (1,1), padding = (0,0), bias = True),
            nn.BatchNorm2d(50), 
            self.activation(act),
            nn.MaxPool2d(kernel_size = (1,2)), 
            nn.Dropout(p = 0.1)
        )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size = (1,5), stride = (1,1), padding = (0,0), bias = True),
            nn.BatchNorm2d(100), 
            self.activation(act),
            nn.MaxPool2d(kernel_size = (1,2)), 
            nn.Dropout(p = 0.1)
        )
        self.Conv4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size = (1,5), stride = (1,1), padding = (0,0), bias = True),
            nn.BatchNorm2d(200), 
            self.activation(act),
            nn.MaxPool2d(kernel_size = (1, 2)), 
            nn.Dropout(p = 0.1)
        )
        self.Classify = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features = 8600, out_features = 2, bias = True)
        )

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Classify(x)
        return x

    def activation(self, act):
        if act == "ReLU":
            return nn.ReLU()
        elif act == "Leaky ReLU":
            return nn.LeakyReLU()
        elif act == "ELU":
            return nn.ELU()

# ----------------------------------------------------

loss_fn = torch.nn.CrossEntropyLoss()

def train(model, dataLoader, optimizer):
    model.train()   # 才會跑 BatchNorm2d 和 Dropout
    train_loss = 0
    train_accuracy = 0
    total = 0

    for i, (signal, label) in enumerate(dataLoader):
        signal = signal.to('cuda')  # 因為 model 是用 GPU 跑
        signal = signal.type(torch.cuda.FloatTensor) # 型態轉換至根 model 相同
        label = label.to('cuda')
        label = label.to(torch.int64)

        predict = model(signal)
        loss = loss_fn(predict, label)
        predict = predict.cpu().detach().argmax(dim = 1)    # 只需要值，把其他像 gradient 不需要的資訊移除

        optimizer.zero_grad()   # 先清空之前 gradient 的值，之後 backward 才不會受影響
        loss.backward()
        optimizer.step()    # 更新

        train_loss += loss.item()
        train_accuracy = accuracy_score(predict, label.cpu())
        total += train_accuracy

    train_accuracy = total / len(dataLoader)
    print('loss = {:.5f}, accuracy = {:.5f}'.format(loss, train_accuracy))
    return train_accuracy

def test(model, dataLoader, best_accuracy):
    model.eval()    # 才不會跑 BatchNorm2d 和 Dropout
    total_accuracy = 0
    for i, (signal, label) in enumerate(dataLoader):
        signal = signal.to('cuda')
        signal = signal.type(torch.cuda.FloatTensor)
        label = label.to('cuda')
        label = label.to(torch.int64)

        predict = model(signal)
        predict = predict.cpu().detach().argmax(dim = 1)

        test_accuracy = accuracy_score(predict, label.cpu())
        total_accuracy += test_accuracy
    test_accuracy = total_accuracy/len(dataLoader)
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
    print("Test accuracy = {:.5f}".format(test_accuracy))
    print("Best accuracy = {:.5f}".format(best_accuracy))
    return best_accuracy, test_accuracy

# ----------------------------------------------------

accuracy_curve = {}
accuracy_curve['ReLU_train'] = []
accuracy_curve['ReLU_test'] = []
accuracy_curve['Leaky_ReLU_train'] = []
accuracy_curve['Leaky_ReLU_test'] = []
accuracy_curve['ELU_train'] = []
accuracy_curve['ELU_test'] = []

setSeed()
print("EEGNet with ReLU:")
best_accuracy = -1
model_ReLU = EEGNet(act = "ReLU").to('cuda')  # Use GPU for training
optimizer_ReLU = torch.optim.AdamW(model_ReLU.parameters(), lr = cfg.lr)
scheduler_ReLU = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_ReLU, T_0 = cfg.T0, T_mult = 1, eta_min = cfg.min_lr, last_epoch = -1)

epochs = cfg.epochs
for epoch in range(epochs):
    print("\nEpoch {}".format(epoch))
    accuracy_curve['ReLU_train'].append(train(model_ReLU, train_loader, optimizer_ReLU))
    best_accuracy, test_accuracy = test(model_ReLU, test_loader, best_accuracy)
    accuracy_curve['ReLU_test'].append(test_accuracy)
    scheduler_ReLU.step()

# ----------------------------------------------------

setSeed()
print("EEGNet with Leaky ReLU:")
best_accuracy = -1
model_Leaky_ReLU = EEGNet(act = "Leaky ReLU").to('cuda')
optimizer_Leaky_ReLU = torch.optim.AdamW(model_Leaky_ReLU.parameters(), lr = cfg.lr)
scheduler_Leaky_ReLU = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_Leaky_ReLU, T_0 = cfg.T0, T_mult = 1, eta_min = cfg.min_lr, last_epoch = -1)

epochs = cfg.epochs
for epoch in range(epochs):
    print("\nEpoch {}".format(epoch))
    accuracy_curve['Leaky_ReLU_train'].append(train(model_Leaky_ReLU, train_loader, optimizer_Leaky_ReLU))
    best_accuracy, test_accuracy = test(model_Leaky_ReLU, test_loader, best_accuracy)
    accuracy_curve['Leaky_ReLU_test'].append(test_accuracy)
    scheduler_Leaky_ReLU.step()

# ----------------------------------------------------

setSeed()
print("EEGNet with ELU:")
best_accuracy = -1
model_ELU = EEGNet(act = "ELU").to('cuda')
optimizer_ELU = torch.optim.AdamW(model_ELU.parameters(), lr = cfg.lr)
scheduler_ELU = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_ELU, T_0 = cfg.T0, T_mult = 1, eta_min = cfg.min_lr, last_epoch = -1)

epochs = cfg.epochs
for epoch in range(epochs):
    print("\nEpoch {}".format(epoch))
    accuracy_curve['ELU_train'].append(train(model_ELU, train_loader, optimizer_ELU))
    best_accuracy, test_accuracy = test(model_ELU, test_loader, best_accuracy)
    accuracy_curve['ELU_test'].append(test_accuracy)
    scheduler_ELU.step()

plt.figure()
plt.title("Activation function comparision(EEGNet)")
for i in accuracy_curve:
    plt.plot(accuracy_curve[i], label = i)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy(%)")
plt.savefig("EEG_Accuracy_curve")

# ----------------------------------------------------

accuracy_curve = {}
accuracy_curve['ReLU_train'] = []
accuracy_curve['ReLU_test'] = []
accuracy_curve['Leaky_ReLU_train'] = []
accuracy_curve['Leaky_ReLU_test'] = []
accuracy_curve['ELU_train'] = []
accuracy_curve['ELU_test'] = []

setSeed()
print("DeepConvNet with ReLU:")
best_accuracy = -1
model_ReLU = DeepConvNet(act = "ReLU").to('cuda')
optimizer_ReLU = torch.optim.AdamW(model_ReLU.parameters(), lr = cfg.lr)
scheduler_ReLU = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_ReLU, T_0 = cfg.T0, T_mult = 1, eta_min = cfg.min_lr, last_epoch = -1)

epochs = cfg.epochs
for epoch in range(epochs):
    print("\nEpoch {}".format(epoch))
    accuracy_curve['ReLU_train'].append(train(model_ReLU, train_loader, optimizer_ReLU))
    best_accuracy, test_accuracy = test(model_ReLU, test_loader, best_accuracy)
    accuracy_curve['ReLU_test'].append(test_accuracy)
    scheduler_ReLU.step()
# ----------------------------------------------------
setSeed()
print("DeepConvNet with Leaky ReLU:")
best_accuracy = -1
model_Leaky_ReLU = DeepConvNet(act = "Leaky ReLU").to('cuda')
optimizer_Leaky_ReLU = torch.optim.AdamW(model_Leaky_ReLU.parameters(), lr = cfg.lr)
scheduler_Leaky_ReLU = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_Leaky_ReLU, T_0 = cfg.T0, T_mult = 1, eta_min = cfg.min_lr, last_epoch = -1)

epochs = cfg.epochs
for epoch in range(epochs):
    print("\nEpoch {}".format(epoch))
    accuracy_curve['Leaky_ReLU_train'].append(train(model_Leaky_ReLU, train_loader, optimizer_Leaky_ReLU))
    best_accuracy, test_accuracy = test(model_Leaky_ReLU, test_loader, best_accuracy)
    accuracy_curve['Leaky_ReLU_test'].append(test_accuracy)
    scheduler_Leaky_ReLU.step()

# ----------------------------------------------------
setSeed()
print("DeepConvNet with ELU:")
best_accuracy = -1
model_ELU = DeepConvNet(act = "ELU").to('cuda')
optimizer_ELU = torch.optim.AdamW(model_ELU.parameters(), lr = cfg.lr)
scheduler_ELU = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_ELU, T_0 = cfg.T0, T_mult = 1, eta_min = cfg.min_lr, last_epoch = -1)

epochs = cfg.epochs
for epoch in range(epochs):
    print("\nEpoch {}".format(epoch))
    accuracy_curve['ELU_train'].append(train(model_ELU, train_loader, optimizer_ELU))
    best_accuracy, test_accuracy = test(model_ELU, test_loader, best_accuracy)
    accuracy_curve['ELU_test'].append(test_accuracy)
    scheduler_ELU.step()

plt.figure()
plt.title("Activation function comparision(DeepConvNet)")
for i in accuracy_curve:
    plt.plot(accuracy_curve[i], label = i)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy(%)")
plt.savefig("DeepConv_Accuracy_curve")