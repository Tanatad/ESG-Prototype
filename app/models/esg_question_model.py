# app/models/esg_question_model.py
from beanie import Document
from pydantic import BaseModel, Field
from datetime import datetime, timezone # Import timezone
from typing import Optional

class ESGQuestion(Document):
    question_text_en: str
    question_text_th: Optional[str] = None
    category: str  # E, S, G
    theme: str
    keywords: Optional[str] = Field(default=None) # Use Pydantic's Field

    is_active: bool = Field(default=True) # Use Pydantic's Field
    version: int = Field(default=1)     # Use Pydantic's Field
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc)) # Use Pydantic's Field
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))   # Use Pydantic's Field

    class Settings:
        name = "esg_questions_bilingual" # ชื่อ Collection ใน MongoDB

