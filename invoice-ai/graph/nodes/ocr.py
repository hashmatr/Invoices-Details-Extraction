from typing import Dict, Any
import time
from graph.state import InvoiceState
from services.paddleocr_service import PaddleOCRService
from models.metrics import ProcessingMetrics

ocr_service = PaddleOCRService()

def ocr_node(state: InvoiceState) -> Dict[str, Any]:
    images = state.get("images", [])
    if not images:
        return {"ocr_data": []}
        
    start_time = time.time()
    
    # We only process the first page for this prototype, or loop through all
    # For invoices, usually it's single page. Let's process page 1.
    image = images[0]
    
    ocr_data = ocr_service.extract_text(image)
    
    end_time = time.time()
    
    # Update metrics
    metrics = state.get("metrics")
    if metrics:
        metrics.ocr_time_ms = (end_time - start_time) * 1000
        
    return {"ocr_data": ocr_data, "metrics": metrics}
