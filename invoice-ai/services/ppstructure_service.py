from typing import List, Dict, Any
import numpy as np
from paddleocr import PPStructureV3
from .layout_engine import LayoutEngine
from config.settings import settings
import time


class PPStructureService(LayoutEngine):
    def __init__(self):
        # Initialize PPStructure with MKL-DNN disabled to avoid OneDNN crashes
        self.engine = PPStructureV3(
            lang=settings.OCR_LANG,
            enable_mkldnn=False,
            cpu_threads=8,
        )
        
    def extract_structure(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extracts document layout and tables using PP-StructureV3.
        Compatible with PaddleOCR v3.x (PaddleX-based) API.
        """
        start_time = time.time()
        
        # PPStructureV3.predict() returns a list of result objects
        results = self.engine.predict(image)
        
        structured_data = {
            "tables": [],
            "regions": []
        }
        
        if results:
            for result in results:
                # The v3 result is a dict-like object; extract layout regions
                layout_det = result.get("layout_det_res", {})
                boxes = layout_det.get("boxes", []) if isinstance(layout_det, dict) else []
                
                for box_info in boxes:
                    region_type = box_info.get("label", "unknown") if isinstance(box_info, dict) else "unknown"
                    bbox = box_info.get("coordinate", []) if isinstance(box_info, dict) else []
                    score = box_info.get("score", 0.0) if isinstance(box_info, dict) else 0.0
                    
                    if region_type == "table":
                        structured_data["tables"].append({
                            "bbox": bbox,
                            "html": "",
                            "raw_res": box_info
                        })
                    else:
                        structured_data["regions"].append({
                            "type": region_type,
                            "bbox": bbox,
                            "raw_res": box_info,
                            "score": float(score)
                        })
                
                # Also try to extract table HTML if available at the top level
                table_res = result.get("table_res", [])
                if table_res:
                    for table in table_res if isinstance(table_res, list) else [table_res]:
                        html = table.get("html", "") if isinstance(table, dict) else ""
                        table_bbox = table.get("bbox", []) if isinstance(table, dict) else []
                        structured_data["tables"].append({
                            "bbox": table_bbox,
                            "html": html,
                            "raw_res": table
                        })
                
        # print(f"Structure extraction took {time.time() - start_time} seconds")
        
        return structured_data
