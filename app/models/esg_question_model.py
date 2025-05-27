from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from beanie import Document # <--- Import Document จาก beanie
from pydantic import BaseModel, Field as PydanticField # <--- Import Field จาก Pydantic (และ BaseModel ถ้า GeneratedQuestion อยู่ในไฟล์นี้)

class ESGQuestion(Document):
    # Existing fields
    question_text_en: str
    question_text_th: Optional[str] = None
    category: str 
    theme: str 
    keywords: Optional[str] = None
    is_active: bool = True
    version: int = 1
    # For fields with Pydantic features like default_factory, use PydanticField
    generated_at: datetime = PydanticField(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = PydanticField(default_factory=lambda: datetime.now(timezone.utc))
    
    # --- New fields for Consolidated Theme details ---
    theme_description_en: Optional[str] = None
    theme_description_th: Optional[str] = None
    dimension: Optional[str] = None
    source_document_references: Optional[List[str]] = PydanticField(default_factory=list)
    constituent_chunk_ids: Optional[List[str]] = PydanticField(default_factory=list)
    detailed_source_info_for_subquestions: Optional[str] = None 

    class Settings:
        name = "esg_questions"
        indexes = [
            "theme",
            "is_active",
            [("theme", 1), ("version", -1)], 
        ]
