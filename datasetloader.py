from collections import defaultdict
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
import cv2

class CTInpaintingDataset(Dataset):
    def __init__(self, data_dir, augment: bool = False):
        # Define paths to each subfolder
        self.data_dir = data_dir
        self.corrupted_dir = os.path.join(data_dir, 'corrupted')
        self.ct_dir = os.path.join(data_dir, 'ct')
        self.mask_dir = os.path.join(data_dir, 'mask')
        self.tissue_dir = os.path.join(data_dir, 'tissue')
        self.vertebrae_dir = os.path.join(data_dir, 'vertebrae')

        # Get the list of files (all subfolders should contain the same filenames)
        file_names = os.listdir(self.corrupted_dir)
        file_names = ["_".join(x.split('_')[1:]).split(".")[0] for x in file_names]
        self.file_names = sorted(file_names)

        self.augment = augment

        self.scaletransform = A.Compose([
            A.ElasticTransform(alpha=120, sigma=120 * 0.5, alpha_affine=None), #120 * 0.03
            A.ShiftScaleRotate(shift_limit=0.0325, scale_limit=0.05, rotate_limit=3, p=0.5, border_mode=cv2.BORDER_REPLICATE),
        ], additional_targets={
            'corrupt': 'image',  # For corrupted
            'mask': 'image',  # For mask
            'tissue': 'image',   # For tissue
        })

        self.colortransform = A.Compose([
            A.RandomBrightnessContrast(p=0.2),
        ], additional_targets={
            'corrupt': 'image',  # For corrupted
        })
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        # Get the file name (without extension) to match files across subfolders
        file_name = self.file_names[idx]

        # Dynamically load the images and vertebrae file for the given index
        ct = np.array(Image.open(os.path.join(self.ct_dir, "ct" + "_" + file_name + '.png'))).astype(np.float32) / 255.0
        corrupted = np.array(Image.open(os.path.join(self.corrupted_dir, "corrupted" + "_" + file_name + '.png'))).astype(np.float32) / 255.0
        mask = np.array(Image.open(os.path.join(self.mask_dir, "mask" + "_" + file_name + '.png'))).astype(np.float32) / 255.0
        tissue = np.array(Image.open(os.path.join(self.tissue_dir, "tissue" + "_" + file_name + '.png'))).astype(np.float32)
        tissue = np.where(tissue == 100, 1.0, tissue)
        tissue = np.where(tissue == 255, 2.0, tissue)

        vertebrae = int(open(os.path.join(self.vertebrae_dir, "vertebrae" + "_" + file_name + '.txt')).read().strip())

        # Apply augmentation (which operates on numpy arrays)
        if self.augment:
            augmented1 = self.scaletransform(image=ct, corrupt=corrupted, mask=mask, tissue=tissue)
            mask = augmented1['mask']
            tissue = augmented1['tissue']
            augmented2 = self.colortransform(image=augmented1['image'], corrupt=augmented1['corrupt'])

            ct = augmented2['image']
            corrupted = augmented2['corrupt']

        # Convert to tensors
        corrupted = torch.tensor(corrupted, dtype=torch.float32).unsqueeze(0)  # Shape: [1, 256, 256]
        ct = torch.tensor(ct, dtype=torch.float32).unsqueeze(0)  # Shape: [1, 256, 256]
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Shape: [1, 256, 256]
        tissue = torch.tensor(tissue, dtype=torch.float32).unsqueeze(0)  # Shape: [1, 256, 256]

        # Return a dictionary with all data
        return {
            'patient_id': file_name,
            'corrupted': corrupted,
            'ct': ct,
            'mask': mask,
            'tissue': tissue,
            'vertebrae': vertebrae,
        }
