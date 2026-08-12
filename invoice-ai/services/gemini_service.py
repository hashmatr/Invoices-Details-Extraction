from typing import Dict, Any, Optional
import base64
import cv2
import numpy as np
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from config.settings import settings
from models.invoice import Invoice

class VerificationResult(BaseModel):
    field: str = Field(description="The name of the field being verified")
    corrected_value: str = Field(description="The corrected value extracted from the image. If empty or missing, return empty string.")
    confidence: float = Field(description="Confidence in the extraction (0.0 to 1.0)")
    reasoning: str = Field(description="Brief explanation of the correction or verification")

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
        self.extract_chain = self.llm.with_structured_output(Invoice)
        
    def _encode_image(self, image: np.ndarray) -> str:
        is_success, buffer = cv2.imencode(".jpg", image)
        if not is_success:
            raise ValueError("Failed to encode image to JPEG")
        return base64.b64encode(buffer).decode("utf-8")

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
            result = self.extract_chain.invoke([message])
            return result
        except Exception as e:
            print(f"Gemini API failure during full extraction: {e}")
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
