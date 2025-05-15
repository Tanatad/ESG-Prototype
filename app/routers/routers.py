from fastapi import APIRouter, Depends, HTTPException
from app.controllers import chat_controller, graph_controller
from fastapi.responses import JSONResponse
router = APIRouter()

@router.get("/health", tags=["Health Check"])
async def health_check():
    return JSONResponse(content={"status": "Service is running"}, status_code=200)

router.include_router(chat_controller.router, prefix="/chat")
router.include_router(graph_controller.router, prefix="/graph")