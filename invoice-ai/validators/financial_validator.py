from decimal import Decimal, ROUND_HALF_UP
from typing import List, Dict, Any, Tuple
from models.invoice import Invoice, LineItem
from config.settings import settings

class FinancialValidator:
    
    @staticmethod
    def _is_close(val1: Decimal, val2: Decimal, tolerance: float) -> bool:
        if val1 is None or val2 is None:
            return False
        return abs(float(val1) - float(val2)) <= tolerance

    @staticmethod
    def validate_line_item(item: LineItem) -> Tuple[bool, str, str]:
        """
        Validates line item math. Generates the formula deterministically.
        Returns: (is_valid, error_msg, generated_formula)
        """
        qty = item.quantity.value if item.quantity else None
        price = item.unit_price.value if item.unit_price else None
        amt = item.amount.value if item.amount else None
        
        formula = ""
        is_valid = True
        error_msg = ""
        
        if qty is not None and price is not None and amt is not None:
            expected_amt = (qty * price).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            formula = f"{qty} × {price} = {expected_amt}"
            
            if not FinancialValidator._is_close(expected_amt, amt, settings.ROUNDING_TOLERANCE):
                is_valid = False
                error_msg = f"Line item amount mismatch: {formula} (found {amt})"
                
        return is_valid, error_msg, formula

    @staticmethod
    def validate_invoice_totals(invoice: Invoice) -> List[Dict[str, Any]]:
        """
        Validates invoice level math.
        Returns a list of validation errors.
        """
        errors = []
        
        subtotal = invoice.amount_before_vat.value
        vat = invoice.vat_amount.value
        total = invoice.total_amount_after_vat.value
        
        # 1. Check Subtotal + VAT = Total
        if subtotal is not None and vat is not None and total is not None:
            expected_total = subtotal + vat
            if not FinancialValidator._is_close(expected_total, total, settings.ROUNDING_TOLERANCE):
                errors.append({
                    "field": "total_amount_after_vat",
                    "type": "mathematical_mismatch",
                    "message": f"Subtotal ({subtotal}) + VAT ({vat}) = {expected_total}, but found {total}"
                })
                
        # 2. Check Line Items Sum = Subtotal
        if invoice.line_items and subtotal is not None:
            items_sum = sum((item.amount.value for item in invoice.line_items if item.amount and item.amount.value), Decimal(0))
            if not FinancialValidator._is_close(items_sum, subtotal, settings.ROUNDING_TOLERANCE):
                errors.append({
                    "field": "amount_before_vat",
                    "type": "mathematical_mismatch",
                    "message": f"Sum of line items ({items_sum}) does not match amount before VAT ({subtotal})"
                })
                
        return errors
