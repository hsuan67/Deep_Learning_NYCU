import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import ResNet18
import ResNet50
import ResNet152
import dataloader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
import os
import ttach as tta

class cfg:
    seed = 1002
    batch_size = 1
    epochs = 20
    T0 = 10
    lr = 1e-3
    min_lr = 8e-8

def setSeed(seed = cfg.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
setSeed()

test_transform = A.Compose([
    A.CenterCrop(350, 350, p = 1.0),
    A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], max_pixel_value = 255, p = 1.0),
    ToTensorV2(p = 1.0)
])

test_df = pd.read_csv("/home/pp037/DL/Lab3/resnet_152_test.csv")
test_dataset = dataloader.LeukemiaLoader(test_df, mode = "test", transform = test_transform)
test_dataloader = DataLoader(test_dataset, batch_size = cfg.batch_size, num_workers = 4, drop_last = False)

def test(model, dataLoader):
    model.eval()
    predict_list = []
    
    for i, (image) in enumerate(dataLoader):
        image = image.to('cuda')
    
        predict = model(image)
        predict = predict.cpu().detach().argmax(dim = 1)

        predict_list.append(predict)
    predict_list = np.concatenate(predict_list, axis=0)
    return predict_list

model = ResNet152.ResNet152().to('cuda')
model.load_state_dict(torch.load('/home/pp037/DL/Lab3/model_152.pkl'))
model = tta.ClassificationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')
predict_list = test(model, test_dataloader)

def save_result(csv_path, predict_list):
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame()
    new_df['ID'] = df['Path']
    new_df['label'] = predict_list
    new_df.to_csv("./312554029_resnet152.csv", index=False)
save_result('/home/pp037/DL/Lab3/resnet_152_test.csv', predict_list)

def evaluate(model, dataLoader):
    #model.eval()
    total_accuracy = 0
    predict_list = []
    for i, (image, label) in enumerate(dataLoader):
        image = image.to('cuda')
        label = torch.tensor(label.to('cuda'), dtype = torch.long)

        predict = model(image)
        predict = predict.cpu().detach().argmax(dim = 1)

        predict_list.append(predict)
    predict_list = np.concatenate(predict_list,axis=0)
    return predict_list

def plot_confusion_matrix(model_name, label, predict, normalize=True):
    cm = confusion_matrix(label, predict)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis = 1)[:,np.newaxis]
    else:
        cm = cm.astype("float") / cm.sum()

    plt.figure(figsize = (15, 15))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(("Confusion matrix of " + model_name))
    plt.colorbar()

    plt.xticks(np.arange(2), rotation = 45)
    plt.yticks(np.arange(2))
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(x = j, y = i, s = ("%.2f"%cm[i][j]), va = 'center', ha = 'center', fontsize = 16)
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(model_name + "_Confusion_Matrix.png")

valid_dataset = dataloader.LeukemiaLoader(dataloader.valid_df, mode = "valid", transform = test_transform)
valid_dataloader = DataLoader(valid_dataset, batch_size = cfg.batch_size, num_workers = 4, drop_last = False)

predict_list = evaluate(model, valid_dataloader)
plot_confusion_matrix("ResNet152", dataloader.valid_df['label'].to_list(), predict_list)