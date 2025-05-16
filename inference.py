import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from torchvision import transforms
from src.unet import UNet, MobileNetV2UNet
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
model = MobileNetV2UNet(output_channels=10).to(device)
model.load_state_dict(torch.load('Models/obj/obj_MOB_1_epoch_172.pth', map_location=device))
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

def overlay_predictions(image, prediction, show_debug=True):
    # Create a color map for classes
    color_map = {
        0: [0, 0, 0],         # Background
        1: [0, 255, 0],    # Road
        2: [255, 0, 0],       # Car
        3: [250, 170, 30],    # Traffic Light
        4: [220, 220, 0],     # Traffic Sign
        5: [220, 20, 60],     # Person
        6: [244, 35, 232],    # Sidewalks
        7: [0, 0, 70],        # Truck
        8: [0, 60, 100],      # Bus
        9: [0, 0, 230],       # Motorcycle
    }
    
    # Convert prediction logits to class indices
    _, predicted_class = torch.max(prediction, dim=1)
    predicted_class = predicted_class.squeeze().cpu().numpy()
    
    # Resize mask to match original image size
    predicted_class = cv2.resize(predicted_class.astype(np.uint8), 
                                (image.shape[1], image.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
    
    # Save original road mask for comparison
    original_road_mask = (predicted_class == 1).astype(np.uint8) * 255
    
    # IMPROVEMENT: Clean up road segmentation with morphological operations
    road_mask = original_road_mask.copy()

    # Define kernel - rectangular shape works well for roads
    kernel_size = 5  # Increase for more noticeable effect
    kernel = cv2.getStructuringElement(
        shape=cv2.MORPH_RECT, 
        ksize=(kernel_size, kernel_size)
    )

    # Apply morphological closing to connect nearby road segments
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)

    # Find connected components
    ccs = cv2.connectedComponentsWithStats(
        road_mask, connectivity=8, ltype=cv2.CV_32S)
    labels = ccs[1]
    stats = ccs[2]

    # Keep only the largest component (main road)
    # Ignore label 0 which is background
    if len(stats) > 1:
        # Find the largest component by area, excluding background (index 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        # Create mask with only the largest road component
        cleaned_mask = np.zeros_like(road_mask)
        cleaned_mask[labels == largest_label] = 255
        road_mask = cleaned_mask

    # Update the predicted class with the cleaned road mask
    predicted_class_cleaned = predicted_class.copy()
    predicted_class_cleaned[road_mask == 255] = 1
    
    # Create colored overlays
    overlay = image.copy()
    
    # Apply colors based on class prediction
    for class_idx, color in color_map.items():
        overlay[predicted_class_cleaned == class_idx] = color
    
    # Create car mask for finding car objects
    car_mask = (predicted_class_cleaned == 2).astype(np.uint8) * 255
    
    # Find contours of cars
    contours, _ = cv2.findContours(car_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Dictionary to store detected objects
    detected_objects = {'cars': 0}
    
    # Draw bounding boxes around cars
    for contour in contours:
        area = cv2.contourArea(contour)
        # Filter out small detections (noise)
        if area > 300:  # Adjust this threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Calculate approximate distance (using bottom of bounding box)
            y_bottom = y + h
            distance_factor = 1.0 - (y_bottom / image.shape[0])
            estimated_distance = int(50 * distance_factor)  # Simple approximation
            
            # Label with estimated distance
            cv2.putText(overlay, f"{estimated_distance}m", (x, y-5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            detected_objects['cars'] += 1
    
    # Blend with original image
    result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
    
    return result, detected_objects

# Open video
cap = cv2.VideoCapture("assets/seame_data_new.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # time.sleep(0.05)  # Optional: Add a small delay to control frame rate
    
    # Preprocess the image
    img_tensor, original_frame = preprocess_image(frame)
    
    # Run inference on both models
    with torch.no_grad():
        road_predictions = model(img_tensor)
    
    # Overlay road & object predictions on the original frame
    result_frame, detected_objects = overlay_predictions(frame, road_predictions)
    
    # Display the result
    cv2.imshow("Road & Lane Detection", result_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()