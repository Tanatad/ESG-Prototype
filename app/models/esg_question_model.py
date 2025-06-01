# ใน esg_question_model.py
from beanie import Document
from pydantic import BaseModel, Field # Field จาก pydantic โดยตรงสำหรับ sub-model
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

class SubQuestionDetail(BaseModel): # Pydantic BaseModel สำหรับรายละเอียดของ Sub-Question
    sub_question_text_en: str
    sub_question_text_th: Optional[str] = None
    sub_theme_name: str # ชื่อของ Consolidated Sub-Theme ที่ Sub-Question นี้เกี่ยวข้อง
    category_dimension: str # E, S, G dimension ของ Sub-Theme นี้
    keywords: Optional[str] = None
    theme_description_en: Optional[str] = None # Description ของ Sub-Theme นี้
    theme_description_th: Optional[str] = None
    constituent_entity_ids: List[str] = Field(default_factory=list)
    source_document_references: List[str] = Field(default_factory=list)
    detailed_source_info: Optional[str] = None # Detailed source info for this set of sub-questions
    # เพิ่ม metadata ที่เฉพาะเจาะจงกับ sub-question นี้ได้ถ้าต้องการ
    # เช่น _first_order_community_id_source

class ESGQuestion(Document): # Document นี้จะแทน Main Category/Main Theme และ Main Question ของมัน
    # --- Main Category / Main Theme Information ---
    theme: str = Field(..., description="The Main ESG Category name.") # ชัดเจนว่าเป็น Main Category
    category: str = Field(..., description="The primary ESG dimension (E, S, G) for this Main Category.")
    keywords: Optional[str] = Field(None, description="Keywords for this Main Category.")
    theme_description_en: Optional[str] = Field(None, description="English description of this Main Category.")
    theme_description_th: Optional[str] = None
    
    # --- Main Question for this Main Category ---
    main_question_text_en: str = Field(..., description="The single, open-ended Main Question for this Main Category.")
    main_question_text_th: Optional[str] = None
    
    # --- Sub-Questions related to this Main Category ---
    # แต่ละ item ใน list นี้คือชุดของ sub-questions ที่มาจาก consolidated sub-theme หนึ่งๆ
    sub_questions_sets: List[SubQuestionDetail] = Field(default_factory=list, description="List of sub-question sets, each corresponding to a consolidated sub-theme under this main category.")

    # --- Context and Source Information for the Main Category overall ---
    # (อาจจะเก็บ constituent entities และ source docs ของ Main Category โดยรวมไว้ที่นี่)
    main_category_constituent_entity_ids: List[str] = Field(default_factory=list)
    main_category_source_document_references: List[str] = Field(default_factory=list)
    
    # --- Versioning and Status for the Main Category/Question document ---
    version: int = 1
    is_active: bool = True
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # --- Provenance and Metadata ---
    generation_method: Optional[str] = Field(None, description="Method used to generate this Main Category and its questions.")
    metadata_extras: Optional[Dict[str, Any]] = Field(None, description="Extra metadata, e.g., _main_category_raw_id.")

    class Settings:
        name = "esg_hierarchical_questions_denormalized" # ใช้ชื่อ collection ใหม่ที่ชัดเจน
        # indexes = [
        #     "theme", # Main Category Name
        #     "is_active",
        # ]