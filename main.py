import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from src.CombinedDataset import CombinedLaneDataset
from src.train import train_model
from src.unet import UNet, MobileNetV2UNet
import os
import numpy as np

def main():
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    input_size = (256, 128)

    # Your dataset configs
    bdd100k_config = {
        'img_dir': '/home/luis_t2/SEAME/bdd100k_seg/bdd100k/seg/images/train',
        'mask_dir': '/home/luis_t2/SEAME/bdd100k_seg/bdd100k/seg/labels/train',
        'width': input_size[0],
        'height': input_size[1],
        'is_train': True
    }

    carla_config = {
        'img_dir': '/home/luis_t2/CarlaSimulation/obj_dataset/images',
        'mask_dir': '/home/luis_t2/CarlaSimulation/obj_dataset/masks',
        'width': input_size[0],
        'height': input_size[1],
        'is_train': True
    }
    
    sea_config = {
        'img_dir': '/home/luis_t2/SEAME/Team02-Course/Dataset/SEAME/frames',
        'annotation_file': "/home/luis_t2/SEAME/Team02-Course/Dataset/SEAME/road_annotations.json",
        'width': input_size[0],
        'height': input_size[1],
        'is_train': True
    }
    
    # Create the combined dataset with built-in train/val split
    combined_dataset = CombinedLaneDataset(
        bdd100k_config=bdd100k_config, 
        sea_config=sea_config, 
        carla_config=carla_config, 
        val_split=0.0
    )
    
    # Get train and val datasets
    train_dataset = combined_dataset.get_train_dataset()

    # Create weights array for TRAINING data only
    train_bdd100k_size = train_dataset.bdd100k_train_size
    train_sea_size = train_dataset.sea_train_size
    train_carla_size = train_dataset.carla_train_size
    weights = np.zeros(train_dataset.train_size)

    # Calculate weights for equal contribution (adjust percentages as needed)
    total_samples = train_bdd100k_size + train_sea_size
    bdd100k_weight = 0.4 / (train_bdd100k_size / total_samples) if train_bdd100k_size > 0 else 0
    sea_weight = 0.2 / (train_sea_size / total_samples) if train_sea_size > 0 else 0
    carla_weight = 0.4 / (train_carla_size / total_samples) if train_carla_size > 0 else 0

    # Apply weights to all samples
    for i in range(train_dataset.train_size):
        if i < train_bdd100k_size:
            weights[i] = bdd100k_weight
        else:
            weights[i] = sea_weight

    # Create sampler for TRAINING only
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    print(f"Created weighted sampler: train_bdd100k_size={bdd100k_weight:.4f}, SEA={sea_weight:.4f}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8, 
        sampler=sampler,
        num_workers=os.cpu_count() // 2
    )
    
    # Initialize model
    model = MobileNetV2UNet(output_channels=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1.5e-4)
    
    # Train model
    train_model(model, train_loader, criterion, optimizer, device, epochs=200)

if __name__ == '__main__':
    main()