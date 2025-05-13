import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CarlaDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, width=256, height=128, is_train=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.width = width
        self.height = height
        self.is_train = is_train
        
        # Find all images in directory - using your CARLA frame-based naming pattern
        self.images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                             if f.endswith('.png')])
        self.masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) 
                            if f.endswith('.png') and not f.endswith('_viz.png')])
        
        self.class_map = {
            1: 1,
            24: 1,
            14: 2,
            7: 3,
            8: 4,
            12: 5,
            2: 6,
            15: 7,
            16: 8,
            18: 9,
            19: 9,
            13: 9,

        }
        
        # Augmentation for training - same as BDD100K for consistency
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

        mapped_mask = np.zeros_like(mask)
    
        # Now map each Carla class to your model classes
        for carla_class, model_class in self.class_map.items():
            mapped_mask[mask == carla_class] = model_class
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mapped_mask)
        image = transformed['image']
        mask = transformed['mask'].long()
        
        return image, mask
        
    def visualize_sample(self, idx):
        """Visualize a sample for debugging"""
        image, mask = self[idx]
        
        # Convert back to numpy for visualization
        image = image.permute(1, 2, 0).numpy()
        image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
        image = image.astype(np.uint8)
        
        mask = mask.numpy()
        
        # Create a colored mask
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        colors = {
            0: [0, 0, 0],       # Background: black
            1: [0, 255, 0],   # Road: purple-ish
            2: [0, 0, 255],      # Car: dark blue
            3: [0, 255, 255],   # Traffic light: orange
            4: [255, 255, 0],    # Traffic sign: yellow
            5: [255, 0, 0]     # Person: red
        }
        
        for class_id, color in colors.items():
            colored_mask[mask == class_id] = color
            
        # Blend image and mask for visualization
        alpha = 0.4
        blended = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
        
        return image, colored_mask, blended

if __name__ == "__main__":
    carla_img_dir = '/home/luis_t2/CarlaSimulation/dataset/images'
    carla_mask_dir = '/home/luis_t2/CarlaSimulation/dataset/masks'

    dataset = CarlaDataset(carla_img_dir, carla_mask_dir, is_train=True)

    # Visualize a sample for debugging
    image, mask, blended = dataset.visualize_sample(30)
    cv2.imshow('Blended Image', cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)