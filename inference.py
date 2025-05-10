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
model = UNet(num_classes=6).to(device)
model.load_state_dict(torch.load('Models/obj/lane_UNet_4_epoch_200.pth', map_location=device))
model.eval()

src_pts = np.float32([
    [248.0, 81.0],
    [394.0, 81.0],
    [32.0, 456.0],
    [608.0, 456.0],
])

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
        1: [128, 64, 128],    # Road
        2: [0, 0, 142],       # Car
        3: [250, 170, 30],    # Traffic Light
        4: [220, 220, 0],     # Traffic Sign
        5: [220, 20, 60]      # Person
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
    kernel_size = 15  # Increase for more noticeable effect
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
        if area > 500:  # Adjust this threshold as needed
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
    
    # Add debug visualization to show the difference
    if show_debug:
        # Add text explaining the processing
        cv2.putText(result, "Road Segmentation: Cleaned with Morphology & Connected Components", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Create small thumbnails to show before/after
        h, w = result.shape[:2]
        thumbnail_size = (w//4, h//4)
        
        # Create before/after thumbnails
        before_vis = np.zeros((h//4, w//4, 3), dtype=np.uint8)
        before_vis[original_road_mask[:h:4, :w:4] > 0] = [128, 64, 128]  # Road color
        
        after_vis = np.zeros((h//4, w//4, 3), dtype=np.uint8)
        after_vis[road_mask[:h:4, :w:4] > 0] = [128, 64, 128]  # Road color
        
        # Add the thumbnails to the corner
        result[10:10+h//4, w-10-w//4:w-10] = before_vis
        result[10+h//4+5:10+h//4+5+h//4, w-10-w//4:w-10] = after_vis
        
        # Add labels
        cv2.putText(result, "Before", (w-10-w//4, 10), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(result, "After", (w-10-w//4, 10+h//4+5), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return result, detected_objects

def calibrate_bev_transform(image_path=None, video_path=None):
    """
    Interactive tool to calibrate BEV with option for extended road visibility
    """
    # Load image or first frame as before
    if image_path:
        frame = cv2.imread(image_path)
    elif video_path:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if not ret:
            print("Failed to read video")
            return None
        cap.release()
    else:
        print("Must provide either image_path or video_path")
        return None
    
    h, w = frame.shape[:2]
    
    src_pts = np.float32([
        [w * 0.43, h * 0.65],  # Top left
        [w * 0.57, h * 0.65],  # Top right
        [w * 0.05, h * 0.95],  # Bottom left
        [w * 0.95, h * 0.95]   # Bottom right
    ])
    
    # Destination points for non-linear mapping
    bev_width, bev_height = w, h
    margin = int(bev_width * 0.1)
    
    dst_pts = np.float32([
        [bev_width * 0.25, 0],           # Top left
        [bev_width * 0.75, 0],           # Top right
        [margin, bev_height],            # Bottom left
        [bev_width - margin, bev_height] # Bottom right
    ])

    # Point being currently dragged (None if no point is being dragged)
    dragging_point = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal dragging_point, src_pts
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is near any source point
            for i, (px, py) in enumerate(src_pts):
                if abs(x - px) < 10 and abs(y - py) < 10:
                    dragging_point = i
                    break
        
        elif event == cv2.EVENT_MOUSEMOVE:
            # Update point position if dragging
            if dragging_point is not None:
                src_pts[dragging_point] = [x, y]
        
        elif event == cv2.EVENT_LBUTTONUP:
            # Stop dragging
            dragging_point = None
    
    # Create window and set mouse callback
    window_name = "BEV Calibration"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        # Create a copy of the frame to draw on
        display = frame.copy()
        
        # Draw the source points
        for i, (x, y) in enumerate(src_pts):
            cv2.circle(display, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(display, f"{i}", (int(x)+10, int(y)+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw lines connecting the source points
        cv2.line(display, (int(src_pts[0][0]), int(src_pts[0][1])), 
                (int(src_pts[1][0]), int(src_pts[1][1])), (0, 255, 0), 2)
        cv2.line(display, (int(src_pts[1][0]), int(src_pts[1][1])), 
                (int(src_pts[3][0]), int(src_pts[3][1])), (0, 255, 0), 2)
        cv2.line(display, (int(src_pts[3][0]), int(src_pts[3][1])), 
                (int(src_pts[2][0]), int(src_pts[2][1])), (0, 255, 0), 2)
        cv2.line(display, (int(src_pts[2][0]), int(src_pts[2][1])), 
                (int(src_pts[0][0]), int(src_pts[0][1])), (0, 255, 0), 2)
        
        # Apply the BEV transform and show the result
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        bev_image = cv2.warpPerspective(frame, M, (w, h))
        
        # Show both images side by side
        cv2.imshow(window_name, np.hstack([display, bev_image]))
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save the source points
            print("Source points:")
            print("src_pts = np.float32([")
            for x, y in src_pts:
                print(f"    [{x}, {y}],")
            print("])")
            break
    
    cv2.destroyAllWindows()
    return src_pts

def get_bird_eye_view(image, mask=None, src_pts=None, dst_pts=None, extended_view=True):
    """
    Transform image and/or mask to bird's eye view with extended road visibility
    """
    h, w = image.shape[:2] if image is not None else mask.shape[:2]
    
    # Default source points if not provided
    if src_pts is None:
        src_pts = np.float32([
            [w * 0.40, h * 0.45],  # Top left (higher up)
            [w * 0.60, h * 0.45],  # Top right (higher up)
            [w * 0.05, h * 0.95],  # Bottom left
            [w * 0.95, h * 0.95]   # Bottom right
        ])
    
    # Create non-linear transformation for better distance perception
    if dst_pts is None:
        bev_width, bev_height = w, h
        margin = int(bev_width * 0.1)
        
        dst_pts = np.float32([
            [bev_width * 0.25, 0],                # Top left (closer to center)
            [bev_width * 0.75, 0],                # Top right (closer to center) 
            [margin, bev_height],                 # Bottom left
            [bev_width - margin, bev_height]      # Bottom right
        ])

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Apply the transform
    bev_image = None
    if image is not None:
        bev_image = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR)
    
    bev_mask = None
    if mask is not None:
        bev_mask = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_NEAREST)
    
    return bev_image, bev_mask

# Open video
cap = cv2.VideoCapture("assets/seame_data.mp4")

# src_pts = calibrate_bev_transform(video_path="assets/seame_data.mp4")

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
    result_frame, detected_objects = overlay_predictions(frame, predictions)
    
    # Display the result
    cv2.imshow("Lane Detection", result_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
