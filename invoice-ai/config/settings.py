import os

from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path, override=True)

class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    
    OCR_LANG = os.getenv("OCR_LANG", "ch")
    OCR_USE_GPU = os.getenv("OCR_USE_GPU", "false").lower() == "true"
    
    CONFIDENCE_THRESHOLD_TRUSTED = float(os.getenv("CONFIDENCE_THRESHOLD_TRUSTED", "0.95"))
    CONFIDENCE_THRESHOLD_SUSPICIOUS = float(os.getenv("CONFIDENCE_THRESHOLD_SUSPICIOUS", "0.80"))
    ROUNDING_TOLERANCE = float(os.getenv("ROUNDING_TOLERANCE", "0.02"))

settings = Settings()
