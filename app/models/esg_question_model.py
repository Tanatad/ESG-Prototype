# file: app/models/esg_question_model.py

from beanie import Document, PydanticObjectId
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict
# --- Sub-Models (These are already Schemas, no changes needed) ---
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, *args, **kwargs):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")

class RelatedSETQuestion(BaseModel):
    set_id: str = Field(..., description="The unique identifier for the SET benchmark question.")
    title_th: str = Field(..., description="The Thai title or question text of the SET benchmark.")
    relevance_score: Optional[float] = Field(None, description="The calculated relevance score.")

class SubQuestionDetail(BaseModel):
    sub_question_text_en: str
    sub_question_text_th: Optional[str] = None
    sub_theme_name: str
    category_dimension: str
    keywords: Optional[str] = None
    theme_description_en: Optional[str] = None
    theme_description_th: Optional[str] = None
    constituent_entity_ids: List[str] = Field(default_factory=list)
    source_document_references: List[str] = Field(default_factory=list)
    detailed_source_info: Optional[str] = None

# --- START OF FIX ---

# --- 1. สร้าง Schema (พิมพ์เขียวข้อมูล) ---
# คลาสนี้ใช้สำหรับเก็บข้อมูลและแสดงผล โดยไม่จำเป็นต้อง Initialize Beanie
class ESGQuestionSchema(BaseModel):
    # เพิ่ม field 'id' เพื่อรับค่า '_id' จาก MongoDB ได้อย่างถูกต้อง
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")

    # --- คัดลอก Fields ทั้งหมดจาก ESGQuestion เดิมมาไว้ที่นี่ ---
    # Core Content Fields
    theme: str = Field(..., description="The main theme or category name of the question set.", index=True)
    category: str = Field(..., description="The primary ESG dimension (E, S, or G).")
    main_question_text_en: str = Field(..., description="The main, high-level question.")
    
    # Optional Descriptive Fields
    keywords: Optional[str] = None
    theme_description_en: Optional[str] = None
    
    # Translated Fields
    theme_th: Optional[str] = None
    main_question_text_th: Optional[str] = None
    theme_description_th: Optional[str] = None
    
    # Structured Data
    sub_questions_sets: List[SubQuestionDetail] = Field(default_factory=list)
    related_set_questions: List[RelatedSETQuestion] = Field(default_factory=list)

    # Metadata Fields
    is_active: bool = Field(default=True, index=True)
    version: int = Field(default=1, index=True)
    generation_method: Optional[str] = None
    
    # Timestamps
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        # อนุญาตให้ Pydantic map ข้อมูลจาก key '_id' ใน MongoDB มาใส่ใน field 'id' ของโมเดลได้
        populate_by_name = True
        # ป้องกัน Error หากมี field เกินมาในข้อมูลจาก DB
        extra = "ignore" 

# --- 2. สร้าง Document (ตัวคุยกับ DB) ให้สืบทอดจาก Schema ---
# คลาสนี้จะถูกใช้โดยส่วนอื่นๆ ของแอปที่ทำงานกับ Beanie แบบ Async
class ESGQuestion(Document, ESGQuestionSchema):
    # ไม่ต้องประกาศ Fields ซ้ำซ้อนอีกต่อไป เพราะมันสืบทอดมาจาก ESGQuestionSchema แล้ว
    
    # มีแค่ Settings ที่เป็นของ Beanie โดยเฉพาะ
    class Settings:
        name = "esg_questions_final" # The name of the MongoDB collection

    # --- FIX: Add Pydantic V2 model_config ---
    # This tells FastAPI how to handle the model correctly for JSON conversion

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={
            ObjectId: str,
            PydanticObjectId: str  # This handles Beanie's specific type
        }
    )
# --- END OF FIX ---