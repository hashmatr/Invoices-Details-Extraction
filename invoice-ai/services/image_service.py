import cv2
import numpy as np

class ImageService:
    @staticmethod
    def load_image(file_path: str) -> np.ndarray:
        """
        Loads an image from file using OpenCV.
        Returns the image as a numpy array in BGR format.
        """
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not load image from {file_path}")
        return image

    @staticmethod
    def preprocess_image(image: np.ndarray, options: dict = None) -> np.ndarray:
        """
        Optional preprocessing steps: grayscale, thresholding, etc.
        """
        if not options:
            return image
            
        processed = image.copy()
        
        if options.get("grayscale", False):
            if len(processed.shape) == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                # Keep it 3-channel for PaddleOCR consistency if needed, 
                # or let PaddleOCR handle it internally.
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                
        # Other preprocessing like denoise, adaptive thresholding could go here
        
        return processed
