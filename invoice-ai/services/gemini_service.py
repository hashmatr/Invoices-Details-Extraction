from typing import Dict, Any, Optional, List
import base64
import cv2
import numpy as np
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from config.settings import settings
from models.invoice import Invoice, Party, StringField, DateField, DecimalField, LineItem
from decimal import Decimal
from datetime import datetime

class VerificationResult(BaseModel):
    field: str = Field(description="The name of the field being verified")
    corrected_value: str = Field(description="The corrected value extracted from the image. If empty or missing, return empty string.")
    confidence: float = Field(description="Confidence in the extraction (0.0 to 1.0)")
    reasoning: str = Field(description="Brief explanation of the correction or verification")

# Flat, simple models for LLM Structured Output
class SimpleLineItem(BaseModel):
    description: str = Field(description="Name or description of the product/service")
    quantity: Optional[float] = None
    unit: Optional[str] = None
    unit_price: Optional[float] = None
    vat_rate: Optional[float] = None
    vat_amount: Optional[float] = None
    amount: float = Field(description="Total amount for this line item")

class SimpleInvoice(BaseModel):
    invoice_number: str = Field(description="The invoice number")
    invoice_date: Optional[str] = Field(description="The date on the invoice (YYYY-MM-DD)", default=None)
    supplier_name: str = Field(description="Name of the supplier/seller")
    supplier_vat: Optional[str] = Field(description="VAT/Tax ID of the supplier", default=None)
    customer_name: str = Field(description="Name of the customer/buyer")
    customer_vat: Optional[str] = Field(description="VAT/Tax ID of the customer", default=None)
    amount_before_vat: float = Field(description="Subtotal amount before tax")
    vat_amount: float = Field(description="Total tax/VAT amount")
    total_amount_after_vat: float = Field(description="Total amount including tax")
    line_items: List[SimpleLineItem] = Field(default_factory=list)

class GeminiService:
    def __init__(self):
        # Base LLM
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            api_key=settings.GEMINI_API_KEY,
            temperature=0.0
        )
        # Specific structured output chains
        self.verify_chain = self.llm.with_structured_output(VerificationResult)
        # Use the simple model for extraction
        self.extract_chain = self.llm.with_structured_output(SimpleInvoice)
        
    def _encode_image(self, image: np.ndarray) -> str:
        is_success, buffer = cv2.imencode(".jpg", image)
        if not is_success:
            raise ValueError("Failed to encode image to JPEG")
        return base64.b64encode(buffer).decode("utf-8")

    def _convert_to_app_invoice(self, simple: SimpleInvoice) -> Invoice:
        """Converts the flat SimpleInvoice back into the complex application Invoice model."""
        
        def s_field(val):
            return StringField(value=str(val) if val is not None else None)
            
        def d_field(val):
            return DecimalField(value=Decimal(str(val)) if val is not None else None)
            
        def date_field(val):
            if not val:
                return DateField(value=None)
            try:
                # Try to parse YYYY-MM-DD or YYYYMMDD
                clean_val = val.replace("年", "-").replace("月", "-").replace("日", "").strip()
                parsed = datetime.strptime(clean_val, "%Y-%m-%d").date()
                return DateField(value=parsed)
            except Exception:
                return DateField(value=None)

        supplier = Party(name=s_field(simple.supplier_name), vat=s_field(simple.supplier_vat))
        customer = Party(name=s_field(simple.customer_name), vat=s_field(simple.customer_vat))
        
        line_items = []
        for li in simple.line_items:
            line_items.append(LineItem(
                description=s_field(li.description),
                quantity=d_field(li.quantity),
                unit=s_field(li.unit),
                unit_price=d_field(li.unit_price),
                vat_rate=d_field(li.vat_rate),
                vat_amount=d_field(li.vat_amount),
                amount=d_field(li.amount)
            ))

        return Invoice(
            invoice_number=s_field(simple.invoice_number),
            invoice_date=date_field(simple.invoice_date),
            supplier=supplier,
            customer=customer,
            amount_before_vat=d_field(simple.amount_before_vat),
            vat_amount=d_field(simple.vat_amount),
            total_amount_after_vat=d_field(simple.total_amount_after_vat),
            line_items=line_items
        )

    def extract_full_invoice(self, image: np.ndarray) -> Optional[Invoice]:
        """
        Calls Gemini to extract the complete structured invoice directly from the image.
        """
        base64_image = self._encode_image(image)
        
        prompt = """You are an expert at processing and understanding Chinese VAT invoices (fapiao) and general receipts.
Please extract all available information from the provided invoice image.
Follow these rules strictly:
1. All monetary amounts must be numbers only (e.g. 1000.00). Do not include currency symbols.
2. If a field is not present or cannot be read, leave it empty or null.
3. For line items, extract the description, quantity, unit price, VAT rate, VAT amount, and total amount.
4. Ensure the mathematical totals (amount before VAT + VAT = total amount) are accurate if present.
"""

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        )

        try:
            simple_result = self.extract_chain.invoke([message])
            if not simple_result:
                return None
            return self._convert_to_app_invoice(simple_result)
        except Exception as e:
            import traceback
            print(f"Gemini API failure during full extraction: {e}")
            traceback.print_exc()
            return None

    def verify_field(
        self, 
        image_crop: np.ndarray, 
        field_name: str, 
        extracted_value: str, 
        validation_error: str
    ) -> VerificationResult:
        """
        Calls Gemini to verify and correct a specific field using a cropped image region.
        """
        base64_image = self._encode_image(image_crop)
        
        prompt = f"""You are verifying a Chinese invoice extraction.

Field to verify: {field_name}
Originally extracted value: {extracted_value}
Validation context/error: {validation_error}

Look at the provided image crop which contains this field. 
Extract the correct value for '{field_name}'.
If it is a monetary amount, extract only the number (e.g. 1000.00).
"""

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        )

        try:
            result = self.verify_chain.invoke([message])
            return result
        except Exception as e:
            return VerificationResult(
                field=field_name,
                corrected_value=extracted_value,
                confidence=0.0,
                reasoning=f"Gemini API failure: {str(e)}"
            )
