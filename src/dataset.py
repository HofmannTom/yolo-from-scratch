import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

GRID_SIZE = 7

class DentalDataset(Dataset):
    def __init__(self, images_path, labels_path):
        self.images_path = images_path
        self.labels_path = labels_path
        self.files = [f for f in os.listdir(images_path) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        img_path = os.path.join(self.images_path, file)
        label_path = os.path.join(self.labels_path, file.replace(".jpg", ".txt"))

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))

        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        target = torch.zeros((GRID_SIZE, GRID_SIZE, 5))

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():

                    cls, x, y, w, h = map(float, line.split())

                    gx = int(x * GRID_SIZE)
                    gy = int(y * GRID_SIZE)

                    if gx >= GRID_SIZE or gy >= GRID_SIZE:
                        continue

                    x_cell = x * GRID_SIZE - gx
                    y_cell = y * GRID_SIZE - gy

                    target[gy, gx] = torch.tensor([x_cell, y_cell, w, h, 1])

        return img, target