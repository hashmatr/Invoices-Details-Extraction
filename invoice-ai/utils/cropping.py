import cv2
import numpy as np
from typing import List

class ImageCropper:
    @staticmethod
    def crop_region(image: np.ndarray, bbox: List[float], padding: int = 50) -> np.ndarray:
        """
        Crops an image using a bounding box [x1, y1, x2, y2] with optional padding.
        Ensures the crop stays within image boundaries.
        """
        if not bbox or len(bbox) != 4:
            return image
            
        h, w = image.shape[:2]
        
        x1 = max(0, int(bbox[0]) - padding)
        y1 = max(0, int(bbox[1]) - padding)
        x2 = min(w, int(bbox[2]) + padding)
        y2 = min(h, int(bbox[3]) + padding)
        
        # If the box is invalid, return original
        if x1 >= x2 or y1 >= y2:
            return image
            
        return image[y1:y2, x1:x2]
