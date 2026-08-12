from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class LayoutEngine(ABC):
    @abstractmethod
    def extract_structure(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract document layout and tables from an image.
        Returns a dictionary containing layout regions and table structures.
        """
        pass
