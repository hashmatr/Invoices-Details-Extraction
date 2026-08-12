import pytest
from decimal import Decimal
from graph.graph import build_graph
from models.metrics import ProcessingMetrics
from models.invoice import Invoice, Party, StringField, DecimalField, Evidence
from services.extraction_service import ExtractionService
from services.gemini_service import GeminiService, VerificationResult
from utils.cropping import ImageCropper
import numpy as np

# --- MOCKS ---

def _mock_extraction_success(*args, **kwargs):
    ev = Evidence(source_text="100.00", page_number=1, confidence=0.99, extraction_source="ocr")
    return Invoice(
        invoice_number=StringField(value="123", evidence=ev),
        supplier=Party(name=StringField(value="Sup", evidence=ev)),
        customer=Party(name=StringField(value="Cus", evidence=ev)),
        amount_before_vat=DecimalField(value=Decimal("100.00"), evidence=ev),
        vat_amount=DecimalField(value=Decimal("13.00"), evidence=ev),
        total_amount_after_vat=DecimalField(value=Decimal("113.00"), evidence=ev),
        line_items=[] 
    )

def _mock_extraction_fail(*args, **kwargs):
    ev = Evidence(source_text="999.00", page_number=1, confidence=0.99, extraction_source="ocr")
    return Invoice(
        invoice_number=StringField(value="123", evidence=ev),
        supplier=Party(name=StringField(value="Sup", evidence=ev)),
        customer=Party(name=StringField(value="Cus", evidence=ev)),
        amount_before_vat=DecimalField(value=Decimal("100.00"), evidence=ev),
        vat_amount=DecimalField(value=Decimal("13.00"), evidence=ev),
        total_amount_after_vat=DecimalField(value=Decimal("999.00"), evidence=ev), # Fails math
        line_items=[]
    )

def _mock_gemini_success(*args, **kwargs):
    return VerificationResult(
        field="total_amount_after_vat",
        corrected_value="113.00",
        confidence=0.95,
        reasoning="Corrected total amount."
    )

def _mock_gemini_fail(*args, **kwargs):
    return VerificationResult(
        field="total_amount_after_vat",
        corrected_value="888.00", # Still fails math
        confidence=0.95,
        reasoning="Wrong correction."
    )

def _mock_gemini_malformed(*args, **kwargs):
    raise ValueError("Malformed output from LLM")

# --- TESTS ---

def test_successful_extraction(monkeypatch):
    """Integration Test: Valid invoice terminates successfully without Gemini routing."""
    monkeypatch.setattr(ExtractionService, "extract_invoice_fields", _mock_extraction_success)
    
    app = build_graph(enable_gemini=True)
    initial_state = {"file_path": "dummy.pdf", "file_type": "pdf", "metrics": ProcessingMetrics()}
    
    result_state = None
    for event in app.stream(initial_state):
        for _, state in event.items(): result_state = state
            
    assert result_state["metrics"].final_status == "success"
    assert result_state["metrics"].gemini_calls == 0

def test_targeted_gemini_correction(monkeypatch):
    """Integration Test: Invalid invoice routes to Gemini, gets corrected, and succeeds."""
    monkeypatch.setattr(ExtractionService, "extract_invoice_fields", _mock_extraction_fail)
    monkeypatch.setattr(GeminiService, "verify_field", _mock_gemini_success)
    # Mock image list so gemini node doesn't skip
    monkeypatch.setattr("graph.nodes.preprocess.preprocess_node", lambda s: {"images": [np.zeros((100,100,3))]})
    
    app = build_graph(enable_gemini=True)
    initial_state = {"file_path": "dummy.pdf", "file_type": "pdf", "metrics": ProcessingMetrics(), "validation_errors": []}
    
    result_state = None
    for event in app.stream(initial_state):
        for _, state in event.items(): result_state = state
            
    assert result_state["metrics"].final_status == "success"
    assert result_state["metrics"].gemini_calls > 0
    assert result_state["invoice_data"].total_amount_after_vat.value == Decimal("113.00")
    assert result_state["invoice_data"].total_amount_after_vat.evidence.extraction_source == "gemini"

def test_retry_exhaustion_needs_review(monkeypatch):
    """Integration Test: Gemini fails to correct the math, retry limit hits, exits to needs_review."""
    monkeypatch.setattr(ExtractionService, "extract_invoice_fields", _mock_extraction_fail)
    monkeypatch.setattr(GeminiService, "verify_field", _mock_gemini_fail)
    monkeypatch.setattr("graph.nodes.preprocess.preprocess_node", lambda s: {"images": [np.zeros((100,100,3))]})
    
    app = build_graph(enable_gemini=True)
    initial_state = {"file_path": "dummy.pdf", "file_type": "pdf", "metrics": ProcessingMetrics()}
    
    result_state = None
    for event in app.stream(initial_state):
        for _, state in event.items(): result_state = state
            
    assert result_state["metrics"].final_status == "needs_review"
    assert result_state["metrics"].gemini_calls > 0
    # Maximum retries is 2 as defined in routing.py
    assert result_state["metrics"].retry_count == 2 

def test_malformed_gemini_output(monkeypatch):
    """Integration Test: Gemini throws an error (malformed JSON fallback)."""
    monkeypatch.setattr(ExtractionService, "extract_invoice_fields", _mock_extraction_fail)
    
    # Actually testing the fallback behavior in gemini_service itself
    service = GeminiService()
    # If the LLM throws, the service intercepts it and returns a 0-confidence fallback
    # To mock this, we mock the underlying LLM call
    monkeypatch.setattr(service.llm, "invoke", _mock_gemini_malformed)
    
    res = service.verify_field(np.zeros((10,10,3)), "test_field", "123", "error")
    assert res.confidence == 0.0
    assert "Gemini API failure" in res.reasoning

def test_missing_evidence_cropping():
    """Integration Test: Image cropper handles missing or invalid bounding boxes safely."""
    img = np.zeros((100, 100, 3))
    # Missing bbox returns original image
    assert ImageCropper.crop_region(img, []).shape == img.shape
    # Invalid bbox (x1 > x2) returns original image
    assert ImageCropper.crop_region(img, [50, 50, 10, 10]).shape == img.shape
    # Out of bounds bbox is clipped
    crop = ImageCropper.crop_region(img, [-100, -100, 200, 200], padding=0)
    assert crop.shape == img.shape
