import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import random
import os
import dataloader
import ResNet18
import ResNet50
import ResNet152

class cfg:
    seed = 1002
    batch_size = 32
    epochs = 50
    T0 = 10
    lr = 1e-3
    min_lr = 7e-5

def setSeed(seed = cfg.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
setSeed()

train_transform = A.Compose([
    A.CenterCrop(350, 350, p = 1.0),
    A.HorizontalFlip(p = 0.5),
    A.VerticalFlip(p = 0.5),
    A.ShiftScaleRotate(p = 0.5),
    A.CoarseDropout(p = 0.5),
    A.Rotate(p = 0.5),
    A.Transpose(p = 0.5),
    A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], max_pixel_value = 255, p = 1.0),
    ToTensorV2(p = 1.0)
])

valid_transform = A.Compose([
    A.CenterCrop(350, 350, p = 1.0),
    A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], max_pixel_value = 255, p = 1.0),
    ToTensorV2(p = 1.0)
])

train_dataset = dataloader.LeukemiaLoader(dataloader.train_df, mode = "train", transform = train_transform)
valid_dataset = dataloader.LeukemiaLoader(dataloader.valid_df, mode = "valid", transform = valid_transform)

train_dataloader = DataLoader(train_dataset, batch_size = cfg.batch_size, num_workers = 4, drop_last = False)
valid_dataloader = DataLoader(valid_dataset, batch_size = cfg.batch_size, num_workers = 4, drop_last = False)

def train(model, optimizer, dataLoader = train_dataloader):
    model.train()
    train_loss = 0
    total_accuracy = 0
    for i, (image, label) in tqdm(enumerate(dataLoader)):
        image = image.to('cuda')
        label = torch.tensor(label.to('cuda'), dtype = torch.long)

        predict = model(image)
        loss = loss_fn(predict, label)
        predict = predict.cpu().detach().argmax(dim = 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        accuracy = accuracy_score(predict, label.cpu())
        total_accuracy += accuracy
    train_accuracy = total_accuracy / len(dataLoader)
    return train_accuracy
        
def evaluate(model, best_accuracy, dataLoader = valid_dataloader):
    #model.eval()
    total_accuracy = 0

    for i, (image, label) in enumerate(dataLoader):
        image = image.to('cuda')
        label = torch.tensor(label.to('cuda'), dtype = torch.long)

        predict = model(image)
        predict = predict.cpu().detach().argmax(dim = 1)

        accuracy = accuracy_score(predict, label.cpu())
        total_accuracy += accuracy

    valid_accuracy = total_accuracy / len(dataLoader)
    if valid_accuracy > best_accuracy:
        best_accuracy = valid_accuracy
        torch.save(model.state_dict(),'model.pkl')
    
    return valid_accuracy, best_accuracy

# --------------------------------------------------------------------------------
accuracy_curve = {}
accuracy_curve['ResNet18_train'] = []
accuracy_curve['ResNet18_valid'] = []
accuracy_curve['ResNet50_train'] = []
accuracy_curve['ResNet50_valid'] = []
accuracy_curve['ResNet152_train'] = []
accuracy_curve['ResNet152_valid'] = []

# --------------------------------------------------------------------------------

loss_fn = torch.nn.CrossEntropyLoss()

# ResNet18 training
best_accuracy = 0
epochs = cfg.epochs
model = ResNet18.ResNet18().to('cuda')
optimizer = torch.optim.AdamW(model.parameters(), lr = cfg.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = cfg.T0, T_mult = 1, eta_min = cfg.min_lr, last_epoch = -1)
print("ResNet18")
for epoch in range(epochs):
    print("\nEpoch {}".format(epoch))
    train_accuracy = train(model, optimizer)
    accuracy_curve['ResNet18_train'].append(train_accuracy)
    
    valid_accuracy, best_accuracy = evaluate(model, best_accuracy)
    accuracy_curve['ResNet18_valid'].append(valid_accuracy)
    
    scheduler.step()
    print("\nTrain accuracy: {}    Valid accuracy: {}    Best accuracy: {}".format(train_accuracy, valid_accuracy, best_accuracy))

# ResNet50 training
best_accuracy = 0
epochs = cfg.epochs
model = ResNet50.ResNet50().to('cuda')
optimizer = torch.optim.AdamW(model.parameters(), lr = cfg.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = cfg.T0, T_mult = 1, eta_min = cfg.min_lr, last_epoch = -1)
print("ResNet50")
for epoch in range(epochs):
    print("\nEpoch {}".format(epoch))
    train_accuracy = train(model, optimizer)
    accuracy_curve['ResNet50_train'].append(train_accuracy)
    
    valid_accuracy, best_accuracy = evaluate(model, best_accuracy)
    accuracy_curve['ResNet50_valid'].append(valid_accuracy)
    
    scheduler.step()
    print("\nTrain accuracy: {}    Valid accuracy: {}    Best accuracy: {}".format(train_accuracy, valid_accuracy, best_accuracy))

# ResNet152 training
best_accuracy = 0
epochs = cfg.epochs
model = ResNet152.ResNet152().to('cuda')
optimizer = torch.optim.AdamW(model.parameters(), lr = cfg.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = cfg.T0, T_mult = 1, eta_min = cfg.min_lr, last_epoch = -1)
print("ResNet152")
for epoch in range(epochs):
    print("\nEpoch {}".format(epoch))
    train_accuracy = train(model, optimizer)
    accuracy_curve['ResNet152_train'].append(train_accuracy)
    
    valid_accuracy, best_accuracy = evaluate(model, best_accuracy)
    accuracy_curve['ResNet152_valid'].append(valid_accuracy)
    
    scheduler.step()
    print("\nTrain accuracy: {}    Valid accuracy: {}    Best accuracy: {}".format(train_accuracy, valid_accuracy, best_accuracy))

# --------------------------------------------------------------------------------

def plot_accuracy_curve(accuracy):
    plt.figure()
    plt.title("Accuracy Curve")
    for i in accuracy:
        plt.plot(accuracy[i], label = i)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")
    plt.legend()
    plt.savefig("Accuracy curve.png")

plot_accuracy_curve(accuracy_curve)