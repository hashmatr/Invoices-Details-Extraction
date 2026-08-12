from pydantic import BaseModel, Field
from typing import Dict, Any, List

class ProcessingMetrics(BaseModel):
    ocr_time_ms: float = 0.0
    structure_time_ms: float = 0.0
    extraction_time_ms: float = 0.0
    validation_time_ms: float = 0.0
    gemini_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    gemini_calls: int = 0
    gemini_input_tokens: int = 0
    gemini_output_tokens: int = 0
    
    retry_count: int = 0
    
    validation_failures: List[Dict[str, Any]] = Field(default_factory=list)
    
    final_status: str = "pending"  # 'success', 'needs_review', 'failed'
