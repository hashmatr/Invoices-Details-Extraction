from typing import List, Dict, Any
import numpy as np
from paddleocr import PaddleOCR
from .ocr_engine import OCREngine
from config.settings import settings
import time


class PaddleOCRService(OCREngine):
    def __init__(self):
        # Initialize PaddleOCR with MKL-DNN disabled to avoid OneDNN crashes
        # on certain CPUs with PaddlePaddle 3.x
        self.ocr = PaddleOCR(
            lang=settings.OCR_LANG,
            enable_mkldnn=False,
            cpu_threads=8,
        )
        
    def extract_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extracts text from the given image using PaddleOCR.
        Compatible with PaddleOCR v3.x (PaddleX-based) API.
        """
        start_time = time.time()
        
        # PaddleOCR v3 uses .predict() which returns a list of result objects.
        # Each result has keys: rec_texts, rec_scores, rec_polys/rec_boxes
        results = self.ocr.predict(image)
        extracted_data = []
        
        if results:
            for result in results:
                texts = result.get("rec_texts", [])
                scores = result.get("rec_scores", [])
                polys = result.get("rec_polys", [])
                boxes = result.get("rec_boxes", [])
                
                for i, text in enumerate(texts):
                    confidence = float(scores[i]) if i < len(scores) else 0.0
                    
                    # Prefer rec_boxes (already in [x_min, y_min, x_max, y_max])
                    # Fall back to converting rec_polys (4-point polygons)
                    if i < len(boxes) and len(boxes[i]) >= 4:
                        bbox = [
                            float(boxes[i][0]),
                            float(boxes[i][1]),
                            float(boxes[i][2]),
                            float(boxes[i][3]),
                        ]
                    elif i < len(polys):
                        poly = polys[i]
                        xs = [p[0] for p in poly]
                        ys = [p[1] for p in poly]
                        bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
                    else:
                        bbox = [0, 0, 0, 0]
                    
                    extracted_data.append({
                        "text": text,
                        "confidence": confidence,
                        "bbox": bbox,
                    })
                
        # Optional: Log the time it took
        # print(f"OCR took {time.time() - start_time} seconds")
        
        return extracted_data
