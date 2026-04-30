import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ICLEVR_dataset(Dataset):
    def __init__(self, root, mode = 'train'):
        self.root = '/home/pp037/DL/Lab6'
        self.mode = mode
        self.images, self.labels = self.get_data()

    def transforms(self):        
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        label = self.labels[index]
        image = self.transforms()(image)
        
        return image, label

    def get_data(self):
        label_dict = json.load(open("/home/pp037/DL/Lab6/objects.json"))
        data_dict = json.load(open(os.path.join(self.root, self.mode + ".json")))

        images = list(data_dict.keys())
        labels = list(data_dict.values())

        image_list, label_list = [], []
        for i in range(len(labels)):
            image_list.append("/home/pp037/DL/Lab6/iclevr/iclevr/" + images[i])

            onehot_label = np.zeros(24, dtype = np.float32)
            for j in range(len(labels[i])):
                onehot_label[label_dict[labels[i][j]]] = 1 
            label_list.append(onehot_label)

        return image_list, label_list