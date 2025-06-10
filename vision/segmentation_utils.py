import cv2
import numpy as np



def get_green_mask(image, hsv_lower=(35, 40, 40), hsv_upper=(85, 255, 255)):
    """
    Creates a binary mask by thresholding green colors in HSV color space.
    
    Args:
        image: Input BGR image
        hsv_lower: Lower HSV threshold for green color (default is a common green range)
        hsv_upper: Upper HSV threshold for green color
    
    Returns:
        Binary mask where green areas are white (255) and rest is black (0)
    """
    # Convert BGR image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create mask using HSV thresholds
    mask = cv2.inRange(hsv_image, np.array(hsv_lower), np.array(hsv_upper))
    
    # Apply morphological operations to remove noise and fill holes
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

