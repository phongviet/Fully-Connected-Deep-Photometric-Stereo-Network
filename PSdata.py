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
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    batch.append(img)
            batch = np.array(batch)  # Convert batch to a NumPy array
            X.append(batch.transpose(1, 2, 0).reshape(-1, 20))
        else:
            img = cv2.imread(label_path)
            y.append(img.reshape(-1, 3))
    X = np.vstack(X)  # Stack all batches vertically
    y = np.vstack(y)  # Stack all labels vertically
    return X, y

class PSDataset(Dataset):
    def __init__(self, data_dir, device):
        self.X, self.y = process_image_data(data_dir)
        self.X = torch.tensor(self.X, dtype=torch.float32).to(device)
        self.y = torch.tensor(self.y, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

