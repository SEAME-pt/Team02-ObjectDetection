import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from src.train import train_model
from src.BDD100KDataset import BDD100KSegmentationDataset
import os

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

    model = UNet().to(device)
    criterion = nn.CrossEntropyLoss()
    model = UNet().to(device)

    train_dataset = BDD100KSegmentationDataset(
        img_dir='/home/luis_t2/SEAME/bdd100k_seg/bdd100k/seg/images/train',
        mask_dir='/home/luis_t2/SEAME/bdd100k_seg/bdd100k/seg/labels/train',
        width=256,
        height=128,
        is_train=True
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=8, 
        sampler=sampler,
        num_workers=os.cpu_count() // 2
    )

    train_model(model, train_loader, criterion, optimizer, device, 50)

if __name__ == '__main__':
    main()