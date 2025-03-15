import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

def process_image_data(data_dir):
    X = []
    y = []
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            batch = []
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                if os.path.isfile(image_path):
                    img = cv2.imread(image_path)
                    img = cv2.resize(img, (512, 512))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    batch.append(img)
            X.append(np.asarray(batch))
        else:
            img = cv2.imread(label_path)
            img = cv2.resize(img, (512, 512))
            y.append(img)
    return np.asarray(X), np.asarray(y)

class PSDataset(Dataset):
    def __init__(self, data_dir, device):
        self.X, self.y = process_image_data(data_dir)
        self.X = self.X.transpose(0, 2, 3, 1).reshape(-1, 20)
        self.y = self.y.reshape(-1, 3)
        self.X = torch.tensor(self.X, dtype=torch.float32).to(device)
        self.y = torch.tensor(self.y, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

