# app/controllers/question_ai_controller.py

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from typing import List, Optional
from app.models.esg_question_model import ESGQuestion
from app.services.question_generation_service import QuestionGenerationService, GeneratedQuestion
from app.services.neo4j_service import Neo4jService # จำเป็นสำหรับการสร้าง QGService

# --- ส่วน Dependency Injection (ปรับตามการตั้งค่าของคุณ) ---
# ตัวอย่างนี้จะสมมติว่าคุณมีวิธี get service instances
# ใน app/main.py หรือ app/dependencies.py คุณควรจะสร้าง instance เหล่านี้
# และ provide มันผ่าน Depends()

# นี่คือ placeholder สำหรับ service instances ที่ควรถูกจัดการใน app/main.py หรือ dependencies.py
# เพื่อให้ตัวอย่างนี้ทำงานได้ ผมจะสร้างมันแบบง่ายๆ ที่นี่ (ไม่แนะนำสำหรับ production)
# ใน Production คุณควรใช้ app_state หรือ FastAPI's Depends ที่ setup ไว้ใน main.py

_temp_neo4j_service: Optional[Neo4jService] = None
_temp_qg_service: Optional[QuestionGenerationService] = None

def get_temp_neo4j_service(): # Placeholder
    global _temp_neo4j_service
    if _temp_neo4j_service is None:
        _temp_neo4j_service = Neo4jService()
    return _temp_neo4j_service

def get_temp_qg_service(neo4j_service: Neo4jService = Depends(get_temp_neo4j_service)) -> QuestionGenerationService: # Placeholder
    global _temp_qg_service
    if _temp_qg_service is None:
        if not neo4j_service.llm_embbedding:
             raise HTTPException(status_code=500, detail="Neo4jService embedding model not initialized. Cannot create QGService.")
        _temp_qg_service = QuestionGenerationService(
            neo4j_service=neo4j_service,
            similarity_embedding_model=neo4j_service.llm_embbedding
        )
    return _temp_qg_service
# --- สิ้นสุดส่วน Dependency Injection ตัวอย่าง ---


router = APIRouter(
    prefix="/question-ai", # กำหนด prefix สำหรับ controller นี้
    tags=["Question AI Management"] # Tag สำหรับ OpenAPI docs
)

@router.get("/active-questions", response_model=List[ESGQuestion]) # หรือ List[GeneratedQuestion] ถ้าต้องการ
async def get_all_active_esg_questions(
    limit: int = Query(default=50, ge=1, le=200),
    skip: int = Query(default=0, ge=0)
    # สามารถเพิ่ม filter อื่นๆ เช่น category, theme ได้ในอนาคต
):
    """
    Retrieve a list of currently active ESG questions (Question AI).
    Sorted by the most recently generated/updated.
    """
    try:
        # ดึงคำถามที่ active และเรียงตาม updated_at หรือ generated_at ล่าสุด
        # การเรียงตาม version ด้วยก็ดีเพื่อให้ได้ version ล่าสุดจริงๆ ของแต่ละ theme
        questions = await ESGQuestion.find(
            ESGQuestion.is_active == True
        ).sort(-ESGQuestion.updated_at, -ESGQuestion.version).skip(skip).limit(limit).to_list()

        if not questions:
            return []
        return questions # Beanie model สามารถถูก serialize โดย FastAPI ได้โดยตรง
    except Exception as e:
        print(f"[CONTROLLER ERROR /active-questions] Error retrieving ESG questions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve ESG questions: {str(e)}")


@router.post("/trigger-generation/full-regeneration", response_model=List[GeneratedQuestion])
async def trigger_full_question_regeneration(
    qg_service: QuestionGenerationService = Depends(get_temp_qg_service) # ใช้ Dependency Injection
):
    """
    Manually triggers a full regeneration of all ESG questions.
    This will deactivate all existing questions and create a new set based on the current graph schema.
    """
    try:
        print("[CONTROLLER INFO /trigger-generation/full-regeneration] Full regeneration requested.")
        # เรียก service โดยไม่ส่ง uploaded_file_content_bytes (หรือจะส่ง force_full_regeneration=True ก็ได้ถ้า service มี parameter นั้น)
        generated_questions = await qg_service.generate_and_store_questions(uploaded_file_content_bytes=None)
        return generated_questions
    except Exception as e:
        print(f"[CONTROLLER ERROR /trigger-generation/full-regeneration] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger full question regeneration: {str(e)}")


@router.post("/trigger-generation/from-pdf", response_model=List[GeneratedQuestion])
async def trigger_question_update_from_pdf(
    file: UploadFile = File(..., description="PDF file to analyze for impacting question themes."),
    qg_service: QuestionGenerationService = Depends(get_temp_qg_service) # ใช้ Dependency Injection
):
    """
    Triggers an update (or regeneration if needed) of ESG questions based on the content
    of the uploaded PDF file. The PDF is analyzed for its impact on existing themes.
    """
    try:
        print(f"[CONTROLLER INFO /trigger-generation/from-pdf] PDF upload received: {file.filename}")
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Uploaded PDF file is empty.")

        # เรียก service โดยส่งเนื้อหาไฟล์ PDF เข้าไป
        generated_questions = await qg_service.generate_and_store_questions(uploaded_file_content_bytes=pdf_bytes)
        return generated_questions
    except HTTPException as http_exc: # Re-raise HTTP exceptions
        raise http_exc
    except Exception as e:
        print(f"[CONTROLLER ERROR /trigger-generation/from-pdf] Error processing PDF for question update: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF for question update: {str(e)}")

# --- อาจจะมี Endpoints อื่นๆ ตาม Use Case ของคุณ ---
# เช่น Endpoint สำหรับ Use Case 1: User input ข้อมูลเพื่อสร้าง Sustain Report
# @router.post("/generate-sustainability-report/")
# async def generate_sustainability_report_for_user(
#     user_data_pdf: UploadFile = File(...), # PDF ข้อมูลของ User
#     # ... other necessary inputs ...
#     qg_service: QuestionGenerationService = Depends(get_temp_qg_service),
#     neo4j_service: Neo4jService = Depends(get_temp_neo4j_service)
# ):
#     # 1. ดึง Question AI ล่าสุด (จาก get_all_active_esg_questions หรือ query โดยตรง)
#     # 2. ประมวลผล user_data_pdf เพื่อดูว่าตอบ Question AI ข้อไหนได้บ้าง
#     # 3. ถ้าไม่ครบ -> สร้าง Suggestion Report (อาจจะใช้ qg_service หรือ neo4j_service ช่วย)
#     # 4. ถ้าครบ -> สร้าง Sustain Report
#     pass

# เช่น Endpoint สำหรับ Use Case 3 (ส่วนที่อัปเดต Knowledge Graph เอง)
# อาจจะอยู่ใน GraphController แต่หลังจากอัปเดต Graph แล้ว อาจจะมีการเรียก
# trigger_full_question_regeneration หรือ trigger_question_update_from_pdf (ถ้า PDF นั้นใช้ update graph และ Qs)