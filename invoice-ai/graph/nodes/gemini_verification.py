from typing import Dict, Any
import time
from graph.state import InvoiceState
from services.gemini_service import GeminiService
from utils.cropping import ImageCropper
from models.invoice import Evidence
import decimal

gemini_service = GeminiService()

def gemini_verification_node(state: InvoiceState) -> Dict[str, Any]:
    metrics = state.get("metrics")
    errors = state.get("validation_errors", [])
    invoice = state.get("invoice_data")
    images = state.get("images", [])
    
    if not images or not invoice or not errors:
        return {}
        
    start_time = time.time()
    image = images[0]
    
    corrections = state.get("gemini_corrections", {})
    new_corrections_made = False
    
    # Process only the first error for targeted routing per iteration
    # to avoid hitting rate limits or blowing up token usage at once
    # Or we can process all errors. Let's process all currently identified errors.
    
    for err in errors:
        field_name = err["field"]
        
        # If we already tried correcting this, skip to avoid infinite loop
        if field_name in corrections:
            continue
            
        error_msg = err["message"]
        
        # Determine bbox and extracted value based on field_name
        bbox = []
        extracted_val = ""
        
        if hasattr(invoice, field_name):
            field_obj = getattr(invoice, field_name)
            if field_obj:
                extracted_val = str(field_obj.value)
                if field_obj.evidence:
                    bbox = field_obj.evidence.bounding_box
        
        # If no bbox, use full image (not ideal, but fallback)
        if bbox:
            crop = ImageCropper.crop_region(image, bbox, padding=100)
        else:
            crop = image
            
        # Call Gemini
        result = gemini_service.verify_field(
            image_crop=crop,
            field_name=field_name,
            extracted_value=extracted_val,
            validation_error=error_msg
        )
        
        metrics.gemini_calls += 1
        
        if result.confidence > 0.0:
            corrections[field_name] = result.corrected_value
            new_corrections_made = True
            
            # Merge correction into invoice (simplified merge for totals)
            if hasattr(invoice, field_name):
                field_obj = getattr(invoice, field_name)
                if field_obj:
                    # Depending on field type, convert
                    try:
                        if "amount" in field_name or "vat" in field_name:
                            field_obj.value = decimal.Decimal(result.corrected_value.replace(',',''))
                        else:
                            field_obj.value = result.corrected_value
                            
                        # Update evidence
                        field_obj.evidence = Evidence(
                            source_text=result.corrected_value,
                            page_number=1,
                            bounding_box=bbox,
                            confidence=result.confidence,
                            extraction_source="gemini"
                        )
                    except:
                        pass
                        
    end_time = time.time()
    metrics.gemini_time_ms += (end_time - start_time) * 1000
    metrics.retry_count += 1
    
    return {
        "invoice_data": invoice,
        "gemini_corrections": corrections,
        "metrics": metrics
    }
