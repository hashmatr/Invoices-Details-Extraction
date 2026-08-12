from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
import numpy as np

class OCREngine(ABC):
    @abstractmethod
    def extract_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract text from an image.
        Returns a list of dictionaries containing:
        - text: str
        - confidence: float
        - bbox: List[float] [x1, y1, x2, y2]
        """
        pass
