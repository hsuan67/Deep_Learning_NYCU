import pandas as pd
import os
import cv2
from torch.utils import data
from torch.utils.data import DataLoader, Dataset

class LeukemiaLoader(Dataset):
    def __init__(self, df, mode, transform = None):
        self.df = df
        self.mode = mode
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = self.df['Path'][index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image = image)['image']

        if self.mode != "test":
            label = self.df['label'][index]
            return image, label

        return image

train_df = pd.read_csv("/home/pp037/DL/Lab3/train.csv")
valid_df = pd.read_csv("/home/pp037/DL/Lab3/valid.csv")