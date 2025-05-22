# app/models/esg_question_model.py
from beanie import Document
from pydantic import Field
from datetime import datetime
from typing import Optional

class ESGQuestion(Document):
    question_text_en: str
    question_text_th: Optional[str] = None # Optional
    category: str # E, S, G
    theme: str
    # source_keywords: Optional[str] = None # Optional
    # source_context_summary: Optional[str] = None # Optional
    is_active: bool = True # เผื่อต้องการ Deactivate คำถามบางข้อ
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "esg_questions_bilingual" # ชื่อ Collection ใน MongoDB