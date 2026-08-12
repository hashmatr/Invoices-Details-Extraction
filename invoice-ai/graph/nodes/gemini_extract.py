from typing import Dict, Any
import time
from graph.state import InvoiceState
from services.gemini_service import GeminiService

gemini_service = GeminiService()

def gemini_extract_node(state: InvoiceState) -> Dict[str, Any]:
    images = state.get("images", [])
    if not images:
        return {"invoice_data": None}
        
    start_time = time.time()
    
    # Process the first page
    image = images[0]
    
    # Extract the full invoice using Gemini 1.5 Flash Vision
    invoice = gemini_service.extract_full_invoice(image)
    
    end_time = time.time()
    
    metrics = state.get("metrics")
    if metrics:
        metrics.extraction_time_ms = (end_time - start_time) * 1000
        
    return {
        "invoice_data": invoice,
        "metrics": metrics
    }
