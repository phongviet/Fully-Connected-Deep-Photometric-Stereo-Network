import os

import cv2
import numpy as np
from torch import nn
import torch
from torchvision.transforms import transforms
from tqdm import tqdm

class Dpsn(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.drop_rate = 0.5
        self.in_size = in_size
        self.shadow_drop_rate = 0.1
        self.shadow = nn.Dropout(self.shadow_drop_rate)
        self.dropout = nn.Dropout(self.drop_rate)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_size, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, 2048)
        self.fc5 = nn.Linear(2048, 2048)
        self.fc6 = nn.Linear(2048, 3)

    def forward(self, x):
        x_norm = torch.norm(x, dim=1, keepdim=True)
        x = x / (x_norm + 1e-8)
        x = self.shadow(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(self.dropout(x)))
        x = self.relu(self.fc3(self.dropout(x)))
        x = self.relu(self.fc4(self.dropout(x)))
        x = self.relu(self.fc5(self.dropout(x)))
        x = self.fc6(self.dropout(x))
        return x

def train(model, train_loader, optimizer, criterion, device, epochs):
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for data, target in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}", leave=False):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss / len(train_loader)}")
    return train_loss / len(train_loader)

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing", leave=False):
            output = model(data)
            test_loss += criterion(output, target).item()
    return test_loss / len(test_loader)


def test_on_image(model, in_path, out_path, device):
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize((512, 512)),
        transforms.Grayscale()
    ])

    images = []
    for img_name in os.listdir(in_path):
        img_path = os.path.join(in_path, img_name)
        img = cv2.imread(img_path)
        img = transform(img).unsqueeze(0)  # Add batch dimension
        images.append(img)

    images = torch.cat(images, dim=0).to(device)  # Concatenate all images into a single tensor
    print(images.shape)
    h, w = images.shape[2], images.shape[3]
    normal_map = np.zeros((h, w, 3))

    with torch.no_grad():
        for i in range(h):
            for j in range(w):
                pixel_values = images[:, :, i, j].view(1, -1)  # Reshape to (1, 20)
                normal_map[i, j] = model(pixel_values).cpu().numpy()

    normal_map = (normal_map - normal_map.min()) / (normal_map.max() - normal_map.min())  # Normalize to [0, 1]
    normal_map = (normal_map * 255).astype(np.uint8)  # Convert to uint8
    # cv2.imwrite(out_path, normal_map)
    return normal_map