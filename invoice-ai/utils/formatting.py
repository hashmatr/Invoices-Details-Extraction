import re
from datetime import datetime, date
from decimal import Decimal
from typing import Union, Optional

class Normalizer:
    @staticmethod
    def normalize_chinese_text(text: str) -> str:
        """
        Normalizes Chinese text by removing all whitespace, common punctuation, 
        and converting to uniform case to prevent harmless formatting from failing exact matches.
        """
        if text is None:
            return ""
            
        text = str(text)
        # Remove all whitespace
        text = re.sub(r'\s+', '', text)
        # Remove common punctuation (Chinese and English)
        text = re.sub(r'[，。！？；：“”‘’（）《》〈〉【】、,.!?;:"\'()<>[\]\\]', '', text)
        return text.lower()

    @staticmethod
    def normalize_currency(val: Union[str, Decimal, int, float, None]) -> Optional[Decimal]:
        """
        Removes currency symbols (¥, RMB, 元) and commas, and converts to Decimal.
        """
        if val is None:
            return None
        if isinstance(val, Decimal):
            return val
        if isinstance(val, (int, float)):
            return Decimal(str(val))
            
        clean_str = str(val).replace('¥', '').replace('￥', '').replace('RMB', '').replace('元', '').replace(',', '').strip()
        try:
            return Decimal(clean_str)
        except:
            return None

    @staticmethod
    def normalize_date(val: Union[str, date, datetime, None]) -> Optional[date]:
        """
        Converts common Chinese date formats into a standard datetime.date object.
        E.g., 2026年08月10日 -> 2026-08-10
        """
        if val is None:
            return None
        if isinstance(val, date):
            return val
        if isinstance(val, datetime):
            return val.date()
            
        clean_str = str(val)
        clean_str = clean_str.replace('年', '-').replace('月', '-').replace('日', '').strip()
        clean_str = re.sub(r'[^\d-]', '', clean_str)
        
        parts = clean_str.split('-')
        if len(parts) >= 3:
            try:
                return date(int(parts[0]), int(parts[1]), int(parts[2]))
            except:
                pass
        return None
