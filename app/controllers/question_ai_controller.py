# app/controllers/question_ai_controller.py

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from typing import List, Optional
from app.models.esg_question_model import ESGQuestion
from app.services.question_generation_service import QuestionGenerationService, GeneratedQuestion
from app.services.neo4j_service import Neo4jService # จำเป็นสำหรับการสร้าง QGService
from app.dependencies import get_neo4j_service, get_question_generation_service
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
    qg_service: QuestionGenerationService = Depends(get_question_generation_service) # Corrected DI
):
    """
    Manually triggers a full regeneration of all ESG questions.
    This will typically re-identify themes from the KG and evolve questions.
    """
    try:
        print("[CONTROLLER INFO /trigger-generation/full-regeneration] Full regeneration requested.")
        # เปลี่ยนชื่อเมธอดที่เรียก
        generated_questions = await qg_service.evolve_and_store_questions(uploaded_file_content_bytes=None)
        return generated_questions
    except Exception as e:
        print(f"[CONTROLLER ERROR /trigger-generation/full-regeneration] Error: {e}")
        traceback.print_exc() # เพิ่ม traceback print
        raise HTTPException(status_code=500, detail=f"Failed to trigger full question regeneration: {str(e)}")


@router.post("/trigger-generation/from-pdf", response_model=List[GeneratedQuestion])
async def trigger_question_update_from_pdf(
    file: UploadFile = File(..., description="PDF file to analyze for impacting question themes."),
    qg_service: QuestionGenerationService = Depends(get_question_generation_service) # Corrected DI
):
    """
    Triggers an update/evolution of ESG questions based on the current KG state,
    considering that a new PDF has been (or is being) processed into the KG.
    Note: This endpoint itself doesn't process the PDF into the KG; that should be done
    via the /graph/uploadfile endpoint first or concurrently. This endpoint signals
    that question evolution should consider a recent KG update potentially driven by a PDF.
    For a more integrated flow, the /graph/uploadfile endpoint already triggers this.
    """
    try:
        print(f"[CONTROLLER INFO /trigger-generation/from-pdf] PDF-driven question evolution requested with file: {file.filename}")
        pdf_bytes = await file.read() # อ่าน bytes ถ้า QG service จะใช้
        await file.close() # อย่าลืม close file
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Uploaded PDF file is empty.")

        # เปลี่ยนชื่อเมธอดที่เรียก
        # การส่ง pdf_bytes ไปที่ evolve_and_store_questions หมายความว่า QG Service จะใช้มัน
        # ใน logic ใหม่ของเรา เราตั้งใจให้ flow ของ graph_controller เรียก neo4j_service.flow ก่อน
        # แล้วค่อย trigger qg_service.evolve_and_store_questions(uploaded_file_content_bytes=None)
        # ดังนั้น endpoint นี้อาจจะไม่จำเป็นแล้ว หรือถ้าจะให้มันทำงานแยกต่างหากจริงๆ
        # QG Service จะต้องมี logic ที่รับ pdf_bytes แล้วไปสั่ง neo4j_service ให้อัปเดต KG ก่อน แล้วค่อย evolve Qs
        # เพื่อความง่าย ผมจะสมมติว่า PDF นี้ "เพิ่งถูกอัปโหลด" และ KG อัปเดตแล้ว
        # ดังนั้นการเรียก evolve_and_store_questions(uploaded_file_content_bytes=pdf_bytes) จะเข้า scope KG_UPDATED_FULL_SCAN_FROM_CONTENT
        generated_questions = await qg_service.evolve_and_store_questions(uploaded_file_content_bytes=pdf_bytes) # ถ้าส่ง bytes จะเข้าเงื่อนไข uploaded_file_content_bytes ใน QGService
        return generated_questions
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"[CONTROLLER ERROR /trigger-generation/from-pdf] Error processing PDF for question update: {e}")
        traceback.print_exc() # เพิ่ม traceback print
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