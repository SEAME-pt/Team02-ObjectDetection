import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BDD100KDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, width=256, height=128, is_train=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.width = width
        self.height = height
        self.is_train = is_train
        
        # Find all images in directory
        self.images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                             if f.endswith('.jpg') or f.endswith('.png')])
        self.masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) 
                            if f.endswith('.png')])
        
        # Class mapping for BDD100K
        self.class_map = {
            0: 1,       # road
            13: 2,      # car
            6: 3,       # traffic light
            7: 4,       # traffic sign
            11: 5,      # person
            1: 6,       # sidewalk
            14: 2,      # truck
            15: 2,      # bus
            17: 9,      # motorcycle
            18: 9,      # Bicycle
            12: 9,      # Rider
        }
        
        # Augmentation for training
        if is_train:
            self.transform = A.Compose([
                A.Resize(height=height, width=width),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=height, width=width),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        # Load image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Map BDD100K classes to our classes (background, road, car)
        mapped_mask = np.zeros_like(mask)
        for src_class, target_class in self.class_map.items():
            mapped_mask[mask == src_class] = target_class
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mapped_mask)
        image = transformed['image']
        mask = transformed['mask'].long()
        
        return image, mask