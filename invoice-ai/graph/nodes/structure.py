from typing import Dict, Any
import time
from graph.state import InvoiceState
from services.ppstructure_service import PPStructureService

structure_service = PPStructureService()

def structure_node(state: InvoiceState) -> Dict[str, Any]:
    images = state.get("images", [])
    if not images:
        return {"layout_data": {}}
        
    start_time = time.time()
    
    image = images[0]
    layout_data = structure_service.extract_structure(image)
    
    end_time = time.time()
    
    metrics = state.get("metrics")
    if metrics:
        metrics.structure_time_ms = (end_time - start_time) * 1000
        
    return {"layout_data": layout_data, "metrics": metrics}
