import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def visualize_detections(enhanced_img, detections):
    """Visualize fish detections on enhanced images.
    
    Args:
        enhanced_img: Tensor of enhanced image in range [-1, 1]
        detections: Tensor of detections from YOLOv5 in format [x1, y1, x2, y2, conf, cls]
    """
    # Convert enhanced image to numpy format
    img = (enhanced_img.squeeze().permute(1, 2, 0).cpu().numpy())
    img = (img + 1) * 127.5  # Scale from [-1,1] to [0,255]
    img = img.astype(np.uint8)
    
    # Convert to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Draw each detection
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
        
        # Convert coordinates to integers
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # Draw rectangle
        color = (0, 255, 0)  # Green
        thickness = 2
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Add label
        label = f"Fish {conf:.2f}"
        cv2.putText(img, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    
    # Convert back to RGB for display
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Display using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
