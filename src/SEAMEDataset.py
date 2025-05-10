import os
import cv2
import torch
import numpy as np
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

class SEAMEDataset(Dataset):
    def __init__(self, img_dir, road_annotation_file, lane_annotation_files=None, 
                 width=256, height=128, is_train=True, lane_thickness=5):
        """
        Dataset for road and lane segmentation
        
        Args:
            img_dir: Directory with source images
            road_annotation_file: Path to JSONL file with road annotations
            lane_annotation_files: List of JSON files with lane annotations (TuSimple format)
            width, height: Target dimensions for resizing
            is_train: Whether to use training augmentations
            lane_thickness: Thickness of lane lines in pixels
        """
        self.img_dir = img_dir
        self.width = width
        self.height = height
        self.is_train = is_train
        self.lane_thickness = lane_thickness
        
        # Load road annotations
        self.road_annotations = []
        print(f"Loading road annotations from {road_annotation_file}")
        with open(road_annotation_file, 'r') as f:
            for line in f:
                if line.strip():
                    self.road_annotations.append(json.loads(line))
        
        print(f"Loaded {len(self.road_annotations)} road annotations")
        
        # Load lane annotations if provided
        self.lane_annotations = {}
        if lane_annotation_files:
            for json_path in lane_annotation_files:
                print(f"Loading lane annotations from {json_path}")
                with open(json_path, 'r') as f:
                    for line in f:
                        sample = json.loads(line)
                        # Use the raw_file as key to match with road annotations
                        self.lane_annotations[sample['raw_file']] = sample
            
            print(f"Loaded {len(self.lane_annotations)} lane annotations")
        
        # Filter to only include annotations with available images
        self.road_annotations = [ann for ann in self.road_annotations if 
                           os.path.exists(os.path.join(img_dir, ann['raw_file']))]
        print(f"Found {len(self.road_annotations)} annotations with matching images")
        
        # Set up transformations
        if is_train:
            self.transform = A.Compose([
                A.Resize(height=height, width=width),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], additional_targets={'mask_lanes': 'mask'})
        else:
            self.transform = A.Compose([
                A.Resize(height=height, width=width),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], additional_targets={'mask_lanes': 'mask'})
    
    def __len__(self):
        return len(self.road_annotations)
    
    def __getitem__(self, idx):
        # Get road annotation
        road_annotation = self.road_annotations[idx]
        img_path = os.path.join(self.img_dir, road_annotation['raw_file'])
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get original image dimensions
        height_org, width_org = image.shape[:2]
        
        # Create road mask from polygons (class 1)
        road_mask = np.zeros((height_org, width_org), dtype=np.uint8)
        
        # Fill each polygon with class 1 (road)
        for polygon in road_annotation['polygons']:
            # Convert polygon to numpy array format for cv2.fillPoly
            points = np.array([polygon], dtype=np.int32)
            cv2.fillPoly(road_mask, points, 1)
        
        # Create lane mask if available (class 2)
        lane_mask = np.zeros((height_org, width_org), dtype=np.uint8)
        
        # Check if we have lane annotations for this image
        filename = road_annotation['raw_file']
        if filename in self.lane_annotations:
            lane_anno = self.lane_annotations[filename]
            
            # Process lane points
            x_lanes = lane_anno['lanes']
            y_samples = lane_anno['h_samples']
            
            # Create points list
            pts = [
                [(x, y) for (x, y) in zip(lane, y_samples) if x >= 0]
                for lane in x_lanes
            ]
            
            # Remove empty lanes
            pts = [l for l in pts if len(l) > 0]
            
            # Draw lanes with class 2
            for lane in pts:
                cv2.polylines(
                    lane_mask,
                    np.int32([lane]),
                    isClosed=False,
                    color=2,
                    thickness=self.lane_thickness
                )
        
        # Apply transforms
        transformed = self.transform(image=image, mask=road_mask, mask_lanes=lane_mask)
        image = transformed['image']
        road_mask = transformed['mask']
        lane_mask = transformed['mask_lanes']
        
        # Combine masks (prioritize lanes over road where they overlap)
        combined_mask = road_mask.copy()
        combined_mask[lane_mask == 2] = 2
        
        return image, combined_mask.long()

    def visualize(self, idx):
        """
        Visualize an image with both road and lane annotations
        """
        road_annotation = self.road_annotations[idx]
        img_path = os.path.join(self.img_dir, road_annotation['raw_file'])
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get original dimensions
        height_org, width_org = image.shape[:2]
        
        # Create road mask
        road_mask = np.zeros((height_org, width_org), dtype=np.uint8)
        for polygon in road_annotation['polygons']:
            points = np.array([polygon], dtype=np.int32)
            cv2.fillPoly(road_mask, points, 1)
        
        # Create lane mask
        lane_mask = np.zeros((height_org, width_org), dtype=np.uint8)
        
        filename = road_annotation['raw_file']
        if filename in self.lane_annotations:
            lane_anno = self.lane_annotations[filename]
            
            # Process lane points
            x_lanes = lane_anno['lanes']
            y_samples = lane_anno['h_samples']
            
            pts = [
                [(x, y) for (x, y) in zip(lane, y_samples) if x >= 0]
                for lane in x_lanes
            ]
            
            pts = [l for l in pts if len(l) > 0]
            
            for lane in pts:
                cv2.polylines(
                    lane_mask,
                    np.int32([lane]),
                    isClosed=False,
                    color=1,
                    thickness=self.lane_thickness
                )
        
        # Create colored overlay
        colored_mask = np.zeros_like(image)
        colored_mask[road_mask == 1] = [0, 100, 0]    # Dark green for road
        colored_mask[lane_mask == 1] = [255, 255, 0]  # Yellow for lanes
        
        # Blend image and mask
        alpha = 0.5
        blended = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
        
        # Return visualization
        return blended

if __name__ == "__main__":
    # Hardcoded paths - change these to match your setup
    img_dir = 'frames'
    road_annotation_file = 'road_annotations.json'
    lane_annotation_files = ['lane_annotations.json']  # Optional
    
    # Load the dataset
    dataset = SEAMEDataset(
        img_dir=img_dir,
        road_annotation_file=road_annotation_file,
        lane_annotation_files=lane_annotation_files,  # Set to None if you don't have lane annotations yet
        width=512,  # Using larger size for visualization
        height=256,
        is_train=False  # No augmentations for visualization
    )
    
    # Check if we have any annotations
    if len(dataset) == 0:
        print("No annotations found. Please check your paths.")
    
    # Variables for navigation
    current_idx = 0
    
    print("\n--- Road and Lane Annotation Viewer ---")
    print(f"Dataset contains {len(dataset)} annotated images")
    print("Controls:")
    print("  Next image: Right arrow or 'n'")
    print("  Previous image: Left arrow or 'p'")
    print("  Quit: 'q' or ESC")
    
    while True:
        # Get visualization for current index
        vis = dataset.visualize(current_idx)
        
        # Convert from RGB to BGR for OpenCV
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        
        # Get annotation info
        road_annotation = dataset.road_annotations[current_idx]
        filename = road_annotation['raw_file']
        num_polygons = len(road_annotation['polygons'])
        num_lanes = 0
        if filename in dataset.lane_annotations:
            num_lanes = len([l for l in dataset.lane_annotations[filename]['lanes'] if any(x >= 0 for x in l)])
        
        # Add text with file information
        info_text = f"Image {current_idx+1}/{len(dataset)}: {filename} ({num_polygons} polygons, {num_lanes} lanes)"
        cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        
        # Display the image
        cv2.imshow("Road and Lane Annotations", vis)
        
        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        
        # Handle key press
        if key == ord('q') or key == 27:  # q or ESC
            break
        elif key == ord('n') or key == 83:  # n or right arrow
            current_idx = min(current_idx + 1, len(dataset) - 1)
        elif key == ord('p') or key == 81:  # p or left arrow
            current_idx = max(current_idx - 1, 0)
    
    cv2.destroyAllWindows()
    print("Visualization complete!")
