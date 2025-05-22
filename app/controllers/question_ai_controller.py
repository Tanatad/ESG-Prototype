from fastapi import APIRouter, Depends, HTTPException, Query # เพิ่ม Query เข้ามา
from typing import List
from app.models.esg_question_model import ESGQuestion # Import Beanie model ที่คุณสร้าง
# from app.dependencies import get_question_generation_service # อาจจะไม่จำเป็นถ้าดึงจาก DB โดยตรง

router = APIRouter()

@router.get("/questions", response_model=List[ESGQuestion])
async def get_all_active_esg_questions(
    # สามารถเพิ่ม Parameters สำหรับ Pagination, Filtering ตาม Category, Theme ได้ในอนาคต
    limit: int = Query(default=20, ge=10, le=100) # ge คือ Greater than or equal to, le คือ Less than or equal to
):
    """
    Retrieve a list of active ESG questions, most recently generated.
    """
    try:
        # Query จาก MongoDB โดยใช้ ESGQuestion model
        # ดึงเฉพาะ is_active=True และเรียงตาม generated_at ล่าสุด
        questions = await ESGQuestion.find(
            ESGQuestion.is_active == True
        ).sort(-ESGQuestion.generated_at).limit(limit).to_list()

        if not questions:
            # อาจจะคืนค่า 404 หรือ [] ก็ได้ ขึ้นอยู่กับการออกแบบ
            return [] 
        return questions
    except Exception as e:
        # ควร Log error ด้วย
        print(f"[CONTROLLER ERROR /questions] Error retrieving ESG questions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve ESG questions.")

# อาจจะมี Endpoints อื่นๆ เช่น GET /questions/{category} หรือ GET /questions/themes