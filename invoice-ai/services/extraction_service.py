import re
from decimal import Decimal
from typing import List, Dict, Any, Optional
from models.invoice import Invoice, Party, LineItem, StringField, DecimalField, DateField, Evidence
from datetime import datetime

class ExtractionService:
    @staticmethod
    def _find_value_by_keywords(ocr_data: List[Dict[str, Any]], keywords: List[str], max_distance_x: int = 200, max_distance_y: int = 20) -> Optional[Dict[str, Any]]:
        """
        Find a value associated with a keyword by looking for text to the right or below the keyword.
        """
        for item in ocr_data:
            text = item["text"]
            for kw in keywords:
                if kw in text:
                    # If value is in the same text block (e.g., "发票号码: 123456")
                    val_str = text.split(kw)[-1].strip(" :：")
                    if val_str:
                        return {
                            "value": val_str,
                            "evidence": Evidence(
                                source_text=text,
                                page_number=1,
                                bounding_box=item["bbox"],
                                confidence=item["confidence"],
                                extraction_source="regex_inline"
                            )
                        }
                    
                    # Look for adjacent block
                    kw_bbox = item["bbox"]
                    for other_item in ocr_data:
                        if other_item == item: continue
                        other_bbox = other_item["bbox"]
                        
                        # Check if it's roughly on the same line and to the right
                        is_same_line = abs(other_bbox[1] - kw_bbox[1]) < max_distance_y
                        is_to_right = kw_bbox[2] <= other_bbox[0] and (other_bbox[0] - kw_bbox[2]) < max_distance_x
                        
                        if is_same_line and is_to_right:
                            return {
                                "value": other_item["text"].strip(" :："),
                                "evidence": Evidence(
                                    source_text=other_item["text"],
                                    page_number=1,
                                    bounding_box=other_item["bbox"],
                                    confidence=other_item["confidence"],
                                    extraction_source="regex_adjacent"
                                )
                            }
        return None

    @staticmethod
    def _parse_decimal(val_str: str) -> Optional[Decimal]:
        try:
            # Remove currency symbols and commas
            clean_str = re.sub(r'[^\d.-]', '', val_str)
            return Decimal(clean_str)
        except:
            return None
            
    @staticmethod
    def _parse_date(val_str: str) -> Optional[datetime.date]:
        try:
            # Try parsing common Chinese date formats
            clean_str = re.sub(r'[^\d]', '-', val_str).strip('-')
            parts = clean_str.split('-')
            if len(parts) >= 3:
                return datetime(int(parts[0]), int(parts[1]), int(parts[2])).date()
        except:
            pass
        return None

    @classmethod
    def extract_invoice_fields(cls, ocr_data: List[Dict[str, Any]], layout_data: Dict[str, Any]) -> Invoice:
        """
        Extract invoice fields based on Chinese domain knowledge.
        """
        # Invoice Number
        inv_no_res = cls._find_value_by_keywords(ocr_data, ["发票号码", "发票号", "Invoice No"])
        inv_no_val = inv_no_res["value"] if inv_no_res else ""
        inv_no_ev = inv_no_res["evidence"] if inv_no_res else None
        
        # Date
        date_res = cls._find_value_by_keywords(ocr_data, ["开票日期", "日期", "Date"])
        date_val = cls._parse_date(date_res["value"]) if date_res else None
        date_ev = date_res["evidence"] if date_res else None
        
        # Parties
        sup_res = cls._find_value_by_keywords(ocr_data, ["销售方", "销售方名称", "Supplier"])
        cus_res = cls._find_value_by_keywords(ocr_data, ["购买方", "购方", "Customer"])
        
        supplier = Party(
            name=StringField(value=sup_res["value"] if sup_res else "", evidence=sup_res["evidence"] if sup_res else None)
        )
        customer = Party(
            name=StringField(value=cus_res["value"] if cus_res else "", evidence=cus_res["evidence"] if cus_res else None)
        )
        
        # Financials
        amt_res = cls._find_value_by_keywords(ocr_data, ["金额", "不含税金额", "Amount"])
        vat_res = cls._find_value_by_keywords(ocr_data, ["税额", "增值税额", "VAT"])
        total_res = cls._find_value_by_keywords(ocr_data, ["价税合计", "含税合计", "Total"])
        
        # Placeholder for building the Invoice object
        # In a real implementation, we would extract line items from layout_data["tables"]
        
        invoice = Invoice(
            invoice_number=StringField(value=inv_no_val, evidence=inv_no_ev),
            invoice_date=DateField(value=date_val, evidence=date_ev) if date_val else DateField(value=None),
            supplier=supplier,
            customer=customer,
            amount_before_vat=DecimalField(
                value=cls._parse_decimal(amt_res["value"]) if amt_res else Decimal(0),
                evidence=amt_res["evidence"] if amt_res else None
            ),
            vat_amount=DecimalField(
                value=cls._parse_decimal(vat_res["value"]) if vat_res else Decimal(0),
                evidence=vat_res["evidence"] if vat_res else None
            ),
            total_amount_after_vat=DecimalField(
                value=cls._parse_decimal(total_res["value"]) if total_res else Decimal(0),
                evidence=total_res["evidence"] if total_res else None
            )
        )
        
        return invoice
