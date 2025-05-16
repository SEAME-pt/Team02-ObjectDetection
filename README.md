# Semantic Segmentation for Autonomous Driving

A comprehensive implementation of semantic segmentation using UNet and MobileNetV2UNet architectures with PyTorch for autonomous driving applications.

## Overview

This repository contains an efficient implementation for semantic segmentation focused on road scene understanding. The system processes images to identify key driving-related elements like roads, vehicles, and pedestrians through pixel-level classification.

## Features

- **MobileNetV2UNet Architecture**: Efficient segmentation using MobileNetV2 as backbone
- **Multi-Dataset Training**: Support for BDD100K, CARLA, and SEAME datasets
- **Weighted Sampling**: Balances data from different sources for better generalization
- **Advanced Data Augmentation**: Enhances robustness through augmentation techniques
- **ONNX Export**: Easy model conversion for deployment on edge devices
- **TensorRT Support**: Optimized inference for NVIDIA platforms

## Model Architecture

The model architecture consists of:

- **Backbone**: MobileNetV2 pre-trained on ImageNet for efficient feature extraction
- **Decoder**: UNet-style upsampling path with skip connections
- **Feature Fusion**: Combines high-level semantic and low-level spatial features
- **Memory-Efficient Design**: Optimized for deployment on autonomous vehicles

## Directory Structure

```
.
├── assets/                 # Video files for testing
├── Models/                 # Saved model weights
│   └── obj/                # Segmentation models
├── src/                    # Source code
│   ├── augmentation.py     # Data augmentation implementations
│   ├── BDD100KDataset.py   # BDD100K dataset loader
│   ├── CarlaDataset.py     # CARLA dataset loader  
│   ├── CombinedDataset.py  # Combined dataset for multi-dataset training
│   ├── SEAMEDataset.py     # SEAME dataset loader
│   ├── train.py            # Training functions
│   └── unet.py             # UNet and MobileNetV2UNet model implementations
├── convert.py              # Script for converting PyTorch model to ONNX
├── inference.py            # Script for running inference
├── main.py                 # Main training script
└── requirements.txt        # Python dependencies
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/SEAME-pt/Team02-ObjectDetection.git
cd Team02-ObjectDetection
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv .env
source .env/bin/activate  # On Windows: .env\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Training

To train the semantic segmentation model:

```bash
python main.py
```

### Inference

To run semantic segmentation on a video file:

```bash
python inference.py
```

### Converting to ONNX

To convert a trained PyTorch model to ONNX format for deployment:

```bash
python convert.py
```

## Datasets

The system supports multiple datasets:

- **BDD100K**: Berkeley DeepDrive dataset with diverse road scenes
- **CARLA**: Synthetic data from CARLA simulator with perfect semantic labels
- **SEAME**: Custom dataset for Southeast Asian roads

## Segmentation Classes

The model is trained to segment various classes relevant to driving scenarios:
- Background
- Road
- Car
- Traffic Light
- Traffic Sign
- Person
- Sidewalk
- Truck
- Bus
- Motorcycle/Bicycle

## Optimization

For deployment on edge devices, the model can be:
- Quantized to FP16 precision
- Converted to TensorRT for further acceleration