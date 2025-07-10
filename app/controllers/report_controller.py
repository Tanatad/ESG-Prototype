from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Form, Body
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Dict, Any
import traceback
import io
import json
import tempfile
from datetime import datetime
import pypandoc
# --- Import Services and Dependencies ---
from app.dependencies import get_report_generation_service
from app.services.report_generation_service import ReportGenerationService
import os
# --- Import for PDF Generation and Special Types ---
from weasyprint import HTML, CSS
import markdown2
from beanie import PydanticObjectId

router = APIRouter()

def robust_json_serializer(data: Any) -> Any:
    """
    A robust custom serializer that recursively traverses data structures
    to convert non-serializable types like PydanticObjectId and datetime.
    """
    if isinstance(data, list):
        return [robust_json_serializer(item) for item in data]
    
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            # Beanie's ObjectId is stored in 'id', which Pydantic renames to '_id' on output.
            # We explicitly handle both possibilities.
            if key in ["id", "_id"] and isinstance(value, PydanticObjectId):
                new_dict["_id"] = str(value)
            else:
                new_dict[key] = robust_json_serializer(value)
        return new_dict
        
    if isinstance(data, PydanticObjectId):
        return str(data)
        
    if isinstance(data, datetime):
        return data.isoformat()
        
    return data

@router.post("/generate")
async def generate_report_from_pdfs(
    company_name: str = Form(...), 
    files: List[UploadFile] = File(...),
    report_service: ReportGenerationService = Depends(get_report_generation_service)
) -> JSONResponse:
    """
    Handles file upload, ingestion, and report generation, then returns a
    manually serialized JSON response to ensure compatibility.
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No PDF files provided.")
            
        file_streams = [io.BytesIO(await file.read()) for file in files]
        file_names = [file.filename for file in files]
        
        report_output_object = await report_service.generate_sustainability_report(
            files=file_streams,
            file_names=file_names,
            company_name=company_name 
        )
        
        # Use our robust, custom serializer to guarantee a clean dictionary/list
        json_compatible_data = robust_json_serializer(report_output_object)
        
        return JSONResponse(content=json_compatible_data)
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


@router.post("/create-pdf")
async def create_pdf_from_markdown(
    markdown_content: str = Body(..., embed=True)
):
    """
    Accepts Markdown text and converts it into a downloadable PDF file using Pandoc,
    with the YAML metadata block extension disabled.
    """
    try:
        output_pdf_path = None
        temp_md_path = None
        # Use a temporary file to handle the conversion
        with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w", encoding="utf-8") as temp_md:
            temp_md.write(markdown_content)
            temp_md_path = temp_md.name

        output_pdf_path = temp_md_path.replace(".md", ".pdf")

        # --- THIS IS THE FIX ---
        # Specify the input format as markdown and disable the yaml_metadata_block extension
        pypandoc.convert_file(
            temp_md_path,
            'pdf',
            format='markdown-yaml_metadata_block', # Disable YAML parsing
            outputfile=output_pdf_path,
            extra_args=['--pdf-engine=xelatex', '-V', 'mainfont=TH Sarabun New']
        )
        # ----------------------

        # Read the generated PDF back into bytes
        with open(output_pdf_path, "rb") as f:
            pdf_bytes = f.read()
        
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=sustainability_report.pdf"}
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create PDF with Pandoc: {str(e)}")
    finally:
        # Clean up temporary files
        if temp_md_path and os.path.exists(temp_md_path):
            os.remove(temp_md_path)
        if output_pdf_path and os.path.exists(output_pdf_path):
            os.remove(output_pdf_path)