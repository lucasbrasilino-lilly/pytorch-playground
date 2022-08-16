import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
import matplotlib.pyplot as plt
import pandas as pd

class ChestXrayDataset(Dataset):
    def __init__(self, img_dir, labels, transform=None, target_transform=None):
        if not isinstance(labels,dict):
            raise ValueError
        #self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        for label, label_id in labels.items():
            image_dir = os.path.join(img_dir,label)
            image_files = [file for file in os.listdir(image_dir) if os.path.isfile(file)]
            self.img_label = pd.DataFrame(image_files)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label