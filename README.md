# Object Detection with YOLO

A comprehensive implementation of a YOLO-based object detection system using PyTorch with MobileNetV2 as the backbone network.

## Overview

This repository contains a customized YOLO model implementation for real-time object detection. The system is designed to process images and video data for autonomous driving.

## Features

- Real-time object detection using a YOLO architecture with MobileNetV2 backbone
- Support for multiple datasets (COCO, CARLA, SEAME)
- Advanced data augmentation for robust training
- Multi-scale detection with feature fusion
- Model export to ONNX format for deployment
- Visualization tools for object detection results

## Model Architecture

The model architecture consists of:

- **Backbone**: MobileNetV2 for efficient feature extraction
- **Detection Head**: Custom YOLO-like detection blocks at multiple scales
- **Feature Fusion**: Using skip connections and upsampling to enhance multi-scale detection
- **Anchor System**: Adaptive anchors that adjust based on input resolution

## Directory Structure

```
.
├── assets/             # Video files for testing
├── Models/             # Saved model weights
│   └── Obj/            # Object detection models
├── src/                # Source code
│   ├── augmentation.py         # Data augmentation implementations
│   ├── COCODataset.py          # COCO dataset loader
│   ├── CombinedDataset.py      # Combined dataset for multi-dataset training
│   ├── ObjectDetection.py      # YOLO model implementation
│   ├── SEAMEDataset.py         # SEAME dataset loader
│   └── train.py                # Training functions
├── convert.py          # Script for converting PyTorch model to ONNX
├── inference.py        # Script for running inference
├── main.py             # Main training script
└── requirements.txt    # Python dependencies
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

To train the object detection model on the COCO dataset:

```bash
python main.py
```

### Inference

To run object detection on a video file:

```bash
python inference.py
```

The script is pre-configured to use the test videos in the assets directory.

### Converting to ONNX

To convert a trained PyTorch model to ONNX format for deployment:

```bash
python convert.py
```

## Datasets

The system supports multiple datasets:

- **COCO**: Common Objects in Context dataset
- **CARLA**: Synthetic data from CARLA simulator
- **SEAME**: Custom dataset for Southeast Asian roads

## Object Detection Details

The object detection system is based on YOLO with:
- Custom anchor generation that adapts to input resolution
- Non-maximum suppression for filtering overlapping detections
- IoU-based matching for ground truth assignment during training
- Multi-scale prediction for detecting objects of different sizes

### Loss Function

The system uses a custom YOLOLoss that includes:
- MSE loss for bounding box coordinates and dimensions
- BCE loss for objectness and class predictions
- Weighted loss components for balanced training

### Supported Object Classes

The model is trained to detect various objects relevant to driving scenarios:
- Person
- Bicycle
- Car
- Motorcycle 
- Bus
- Truck
- Traffic Light
