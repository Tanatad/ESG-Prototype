# app/controllers/report_controller.py
import io
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Body
from typing import List
import traceback
from fastapi.responses import StreamingResponse
from app.dependencies import get_report_generation_service
from app.services.report_generation_service import ReportGenerationService
from weasyprint import HTML, CSS
import markdown2

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
        
        report_output = await report_service.generate_sustainability_report(
            files=file_streams,
            file_names=file_names
        )
        
        return report_output
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")
    
@router.post("/create-pdf")
async def create_pdf_from_markdown(
    markdown_content: str = Body(..., embed=True)
):
    """
    Accepts Markdown text and converts it into a downloadable PDF file.
    """
    try:
        html_content = markdown2.markdown(
            markdown_content, 
            extras=["tables", "fenced-code-blocks", "header-ids", "footnotes"]
        )

        # This is the CSS string you want to apply
        css_style_string = """
            @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;700&display=swap');
            body { font-family: 'Sarabun', sans-serif; line-height: 1.6; }
            h1, h2, h3 { color: #003366; }
            h1 { text-align: center; border-bottom: 2px solid #005A9C; }
            h2 { border-bottom: 1px solid #DDDDDD; padding-bottom: 10px; margin-top: 30px; }
            hr { border: none; height: 1px; background-color: #ecf0f1; margin: 40px 0; }
        """

        # --- FIX: Create a CSS object from the string ---
        css = CSS(string=css_style_string)
        # -------------------------------------------------

        # Generate PDF in memory, passing the CSS object
        pdf_bytes = HTML(string=html_content).write_pdf(stylesheets=[css])

        # Stream the PDF back to the client
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=sustainability_report.pdf"}
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create PDF: {str(e)}")