from typing import Dict, Any
import time
from graph.state import InvoiceState
from validators.financial_validator import FinancialValidator

def extraction_node(state: InvoiceState) -> Dict[str, Any]:
    invoice = state.get("invoice_data")
    metrics = state.get("metrics")
    
    if not invoice:
        return {"validation_errors": []}
        
    start_time = time.time()
    
    # Field-level & Financial Validation
    errors = FinancialValidator.validate_invoice_totals(invoice)
    
    # Also evaluate line items if any
    for i, item in enumerate(invoice.line_items):
        is_valid, err_msg, generated_formula = FinancialValidator.validate_line_item(item)
        item.formula = generated_formula
        if not is_valid:
            errors.append({
                "field": f"line_items[{i}]",
                "type": "line_item_math_mismatch",
                "message": err_msg
            })
            
    end_time = time.time()
    
    if metrics:
        metrics.validation_failures.extend(errors)
        
        # Set final status
        if len(errors) > 0:
            metrics.final_status = "needs_review"
        else:
            metrics.final_status = "success"
            
    return {
        "invoice_data": invoice,
        "validation_errors": errors,
        "metrics": metrics
    }
