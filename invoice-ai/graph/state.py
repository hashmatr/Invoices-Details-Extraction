from typing import TypedDict, List, Dict, Any, Optional
from models.invoice import Invoice
from models.metrics import ProcessingMetrics

class InvoiceState(TypedDict):
    # Inputs
    file_path: str
    file_type: str
    
    # Preprocessing
    images: List[Any]  # list of numpy arrays
    
    # OCR and Layout
    ocr_data: List[Dict[str, Any]]
    layout_data: Dict[str, Any]
    
    # QR Code
    qr_code_present: bool
    qr_data: List[str]
    
    # Extraction & Validation
    invoice_data: Optional[Invoice]
    validation_errors: List[Dict[str, Any]]
    suspicious_fields: List[str]
    
    # Gemini
    gemini_corrections: Dict[str, Any]
    
    # Observability & Routing
    metrics: ProcessingMetrics
