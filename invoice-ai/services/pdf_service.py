import fitz  # PyMuPDF
import numpy as np
import cv2
from typing import List

class PDFService:
    @staticmethod
    def render_pdf_to_images(file_path: str, dpi: int = 200) -> List[np.ndarray]:
        """
        Loads a PDF and renders each page to a numpy array (image).
        """
        images = []
        doc = fitz.open(file_path)
        zoom = dpi / 72  # 72 is default DPI
        mat = fitz.Matrix(zoom, zoom)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert PyMuPDF pixmap to numpy array (RGB)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            
            # PyMuPDF creates RGB by default if alpha is False. We want BGR for OpenCV standard.
            if pix.n == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif pix.n == 1:
                # If grayscale, convert to BGR for consistency
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            
            images.append(img_array)
            
        doc.close()
        return images
