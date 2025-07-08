from fastapi import APIRouter, Depends, HTTPException
from app.controllers import chat_controller, graph_controller, question_ai_controller, report_controller # Import ใหม่
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/health", tags=["Health Check"])
async def health_check():
    return JSONResponse(content={"status": "Service is running"}, status_code=200)

router.include_router(chat_controller.router, prefix="/chat", tags=["Chat RAG"])
router.include_router(graph_controller.router, prefix="/graph", tags=["Graph Operations"])
router.include_router(question_ai_controller.router, tags=["Question AI Management"])
router.include_router(report_controller.router, prefix="/report", tags=["Sustainability Report Generation"])
