from typing import Literal
from .state import InvoiceState

def should_verify(state: InvoiceState) -> Literal["gemini", "end"]:
    """
    Decides whether to route to Gemini Verification or end.
    """
    metrics = state.get("metrics")
    errors = state.get("validation_errors", [])
    
    # If no errors, end
    if not errors:
        metrics.final_status = "success"
        return "end"
        
    # If max retries reached, end with needs_review
    MAX_RETRIES = 2
    if metrics.retry_count >= MAX_RETRIES:
        metrics.final_status = "needs_review"
        return "end"
        
    # Otherwise, route to Gemini
    return "gemini"
