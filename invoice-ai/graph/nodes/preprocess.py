from typing import Dict, Any
from graph.state import InvoiceState
from services.pdf_service import PDFService
from services.image_service import ImageService
import os

def preprocess_node(state: InvoiceState) -> Dict[str, Any]:
    file_path = state.get("file_path")
    file_type = state.get("file_type")
    
    images = []
    
    if not os.path.exists(file_path):
        return {"metrics": state["metrics"]}
        
    if file_type == "pdf":
        images = PDFService.render_pdf_to_images(file_path)
    elif file_type in ["png", "jpg", "jpeg"]:
        img = ImageService.load_image(file_path)
        img = ImageService.preprocess_image(img)
        images = [img]
        
    return {"images": images}
