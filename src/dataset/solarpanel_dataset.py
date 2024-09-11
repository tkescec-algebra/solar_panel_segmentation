import os

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class SolarPanelDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            mask_dir (string): Directory with all the masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.image_names = os.listdir(image_dir)
        self.mask_names = os.listdir(mask_dir)

        # Ensure the image and mask files are sorted to align them correctly
        self.image_names.sort()
        self.mask_names.sort()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, self.image_names[idx])
        mask_name = os.path.join(self.mask_dir, self.mask_names[idx])

        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name).convert("L")


        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

