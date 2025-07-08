# app/controllers/report_controller.py

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from typing import List
import traceback

from app.dependencies import get_report_generation_service
from app.services.report_generation_service import ReportGenerationService

router = APIRouter()

@router.post("/generate")
async def generate_report_from_pdfs(
    files: List[UploadFile] = File(...),
    report_service: ReportGenerationService = Depends(get_report_generation_service)
):
    """
    Accepts user-uploaded PDF files and generates a sustainability report
    by answering a standard set of questions based on the document content.
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No PDF files provided.")
            
        file_streams = [file.file for file in files]
        file_names = [file.filename for file in files]
        
        report_data = await report_service.generate_sustainability_report(
            files=file_streams,
            file_names=file_names
        )
        
        return report_data
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")