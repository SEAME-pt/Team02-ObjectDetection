import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from torchvision import transforms
from src.unet import UNet
import time

# Set up device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name()}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Load the trained model
model = UNet().to(device)
model.load_state_dict(torch.load('Models/obj/lane_UNet_1_epoch_2.pth', map_location=device))
model.eval()

# Image preprocessing function
def preprocess_image(image, target_size=(256, 128)):
    # Resize image
    img = cv2.resize(image, target_size)
    
    # 2. Enhance contrast within the ROI
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply same transforms as during training
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalizer
    ])
    
    # Apply transforms
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    return img_tensor, img

def overlay_predictions(image, prediction):
    # Create a color map for classes
    color_map = {
        0: [0, 0, 0],       # Background
        1: [128, 64, 128],  # Road
        2: [0, 0, 142]      # Car
    }
    
    # Convert prediction logits to class indices (argmax)
    _, predicted_class = torch.max(prediction, dim=1)
    predicted_class = predicted_class.squeeze().cpu().numpy()
    
    # Resize mask to match original image size
    predicted_class = cv2.resize(predicted_class.astype(np.uint8), 
                                (image.shape[1], image.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
    
    # Create a colored overlay
    overlay = image.copy()
    
    # Apply colors based on class prediction
    for class_idx, color in color_map.items():
        overlay[predicted_class == class_idx] = color
    
    # Blend with original image
    result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
    return result

def add_legend(frame):
    """Add a legend to the bottom of the frame"""
    h, w = frame.shape[:2]
    legend_h = 30
    legend = np.ones((legend_h, w, 3), dtype=np.uint8) * 255
    
    # Add class colors and labels
    cv2.rectangle(legend, (10, 5), (40, 25), (128, 64, 128), -1)
    cv2.putText(legend, "Road", (45, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.rectangle(legend, (120, 5), (150, 25), (0, 0, 142), -1)
    cv2.putText(legend, "Car", (155, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Combine frame with legend
    result = np.vstack([frame, legend])
    return result

# Open video
cap = cv2.VideoCapture("assets/seame_data.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    time.sleep(0.05)  # Optional: Add a small delay to control frame rate
    
    # Preprocess the image
    img_tensor, original_frame = preprocess_image(frame)
    
    # Run inference
    with torch.no_grad():
        predictions = model(img_tensor)
    
    # Overlay predictions on the original frame
    result_frame = overlay_predictions(frame, predictions)
    result_frame = add_legend(result_frame)
    
    # Display the result
    cv2.imshow("Lane Detection", result_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
