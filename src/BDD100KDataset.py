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
            0: 1,       # road (maps to your class 1)
            6: 3,       # traffic light (maps to your class 3)
            7: 4,       # traffic sign (maps to your class 4)
            11: 5,      # person (maps to your class 5)
            13: 2,      # car (maps to your class 2)
            19: 6,      # lane marking (maps to class 6)
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
        
        # Map BDD100K classes to our classes
        mapped_mask = np.zeros_like(mask)
        for src_class, target_class in self.class_map.items():
            mapped_mask[mask == src_class] = target_class
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mapped_mask)
        image = transformed['image']
        mask = transformed['mask'].long()
        
        return image, mask
        
    def visualize(self, idx):
        """
        Visualize an image with its segmentation mask including lane markings
        """
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        # Load image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Map BDD100K classes
        mapped_mask = np.zeros_like(mask)
        for src_class, target_class in self.class_map.items():
            mapped_mask[mask == src_class] = target_class
        
        # Create a colored visualization
        vis = image.copy()
        
        # Color mapping for visualization
        color_map = {
            1: [128, 64, 128],    # Road (purple-blue)
            2: [0, 0, 142],       # Car (dark red)
            3: [250, 170, 30],    # Traffic light (orange)
            4: [220, 220, 0],     # Traffic sign (yellow)
            5: [220, 20, 60],     # Person (red)
            6: [255, 255, 0]      # Lane marking (bright yellow)
        }
        
        # Create a blended overlay
        overlay = np.zeros_like(image)
        for class_idx, color in color_map.items():
            overlay[mapped_mask == class_idx] = color
        
        # Blend with original image
        alpha = 0.5
        vis = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        
        # Make lane markings more visible (extra bright)
        vis[mapped_mask == 6] = [255, 255, 0]  # Bright yellow for lanes
        
        return vis
    
if __name__ == "__main__":
    # Example usage
    dataset = BDD100KDataset(
        img_dir="/path/to/bdd100k/images",
        mask_dir="/path/to/bdd100k/labels",
        width=640,
        height=360,
        is_train=False
    )
    
    # Set up visualization loop
    current_idx = 0
    
    print(f"Dataset contains {len(dataset)} images")
    print("Controls:")
    print("  Next image: Right arrow or 'n'")
    print("  Previous image: Left arrow or 'p'")
    print("  Toggle lane visibility: 'l'") 
    print("  Quit: 'q' or ESC")
    
    # Flag to highlight lanes
    highlight_lanes = True
    
    while True:
        # Get visualization for current index
        vis = dataset.visualize(current_idx)
        
        # Extra highlighting for lanes if enabled
        if highlight_lanes:
            # Load and map mask
            mask_path = dataset.masks[current_idx]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mapped_mask = np.zeros_like(mask)
            for src_class, target_class in dataset.class_map.items():
                mapped_mask[mask == src_class] = target_class
            
            # Add extra bright highlights for lanes
            vis[mapped_mask == 6] = [255, 255, 0]
        
        # Convert from RGB to BGR for OpenCV
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        
        # Add text with file information
        img_path = dataset.images[current_idx]
        info_text = f"Image {current_idx+1}/{len(dataset)}: {os.path.basename(img_path)}"
        cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        
        lane_status = "Lanes highlighted" if highlight_lanes else "Normal view"
        cv2.putText(vis, lane_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        
        # Display the image
        cv2.imshow("BDD100K with Lane Markings", vis)
        
        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        
        # Handle key press
        if key == ord('q') or key == 27:  # q or ESC
            break
        elif key == ord('n') or key == 83:  # n or right arrow
            current_idx = min(current_idx + 1, len(dataset) - 1)
        elif key == ord('p') or key == 81:  # p or left arrow
            current_idx = max(current_idx - 1, 0)
        elif key == ord('l'):  # Toggle lane highlighting
            highlight_lanes = not highlight_lanes
    
    cv2.destroyAllWindows()