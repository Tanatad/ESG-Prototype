from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Form, Body, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Dict, Any
import traceback
import io
import json
from datetime import datetime
from beanie import PydanticObjectId
import os
import shutil
import uuid

# --- Import Services and Dependencies ---
from app.dependencies import get_report_generation_service
from app.services.report_generation_service import ReportGenerationService

# --- Import PDF Generation Tools ---
import markdown2
from weasyprint import HTML, CSS

router = APIRouter()

# ---- ส่วนสำหรับจัดการ Background Jobs ----
# สร้าง Dictionary แบบ Global เพื่อเก็บสถานะและผลลัพธ์ของแต่ละงาน
# หมายเหตุ: ในแอปพลิเคชันจริง ควรใช้ระบบที่ทนทานกว่านี้ เช่น Redis หรือฐานข้อมูล
jobs: Dict[str, Dict[str, Any]] = {}

async def process_files_in_background(job_id: str, temp_dir: str, company_name: str, report_service: ReportGenerationService):
    """
    ฟังก์ชันนี้จะทำงานในเบื้องหลังเพื่อประมวลผลไฟล์และสร้างรายงาน
    """
    try:
        # อ่านไฟล์จากโฟลเดอร์ชั่วคราว
        file_streams = []
        file_names = []
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            file_streams.append(open(file_path, 'rb'))
            file_names.append(filename)
        
        # ---- ใส่โค้ดการประมวลผลไฟล์และเรียก LLM ทั้งหมดไว้ตรงนี้ ----
        # สมมติว่า ReportGenerationService มีฟังก์ชันที่ทำงานหนัก
        # ในที่นี้ เราจะจำลองการทำงานที่ใช้เวลานาน
        report_output_object = await report_service.generate_sustainability_report(
            files=file_streams,
            file_names=file_names,
            company_name=company_name
        )
        
        # ผลลัพธ์สมมติเพื่อการทดสอบ
        # import asyncio
        # await asyncio.sleep(15) # จำลองการทำงาน 15 วินาที
        # report_output_object = {
        #     "company_name": company_name,
        #     "report_summary": "นี่คือสรุปรายงานที่สร้างเสร็จสมบูรณ์...",
        #     "generated_questions": "นี่คือชุดคำถามที่สร้างจากการวิเคราะห์ไฟล์ทั้งหมด"
        # }
        # --------------------------------------------------------

        # เมื่อทำงานเสร็จ ให้เก็บผลลัพธ์และเปลี่ยนสถานะ
        jobs[job_id]['status'] = 'complete'
        jobs[job_id]['result'] = report_output_object
        
    except Exception as e:
        # หากเกิดข้อผิดพลาด ให้บันทึก error และเปลี่ยนสถานะ
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['result'] = str(e)
        traceback.print_exc()
        
    finally:
        # ปิดไฟล์และลบโฟลเดอร์ชั่วคราว
        for stream in file_streams:
            stream.close()
        shutil.rmtree(temp_dir)

# ----------------------------------------

def robust_json_serializer(data: Any) -> Any:
    if isinstance(data, list):
        return [robust_json_serializer(item) for item in data]
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
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

def render_report_with_style(markdown_text: str) -> str:
    html_body = markdown2.markdown(
        markdown_text,
        extras=["tables", "fenced-code-blocks", "spoiler", "header-ids"]
    )
    styled_html = f"""
    <html>
    <head>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;700&display=swap');
            @page {{ size: A4; margin: 2cm; }}
            body {{ font-family: 'Sarabun', sans-serif; }}
            .report-container {{ color: #333333; line-height: 1.6; }}
            .report-container h1 {{ color: #1a1a1a; text-align: center; border-bottom: 2px solid #005A9C; padding-bottom: 15px; margin-bottom: 40px; }}
            .report-container h2 {{ color: #005A9C; border-bottom: 1px solid #DDDDDD; padding-bottom: 10px; margin-top: 30px; margin-bottom: 20px; }}
            .report-container h3 {{ color: #333333; margin-top: 20px; margin-bottom: 10px; }}
            .report-container p {{ text-align: justify; }}
            .report-container ul {{ padding-left: 20px; }}
            .report-container table {{ border-collapse: collapse; width: 100%; margin-bottom: 1em; }}
            .report-container th, .report-container td {{ border: 1px solid #dddddd; text-align: left; padding: 8px; }}
            .report-container th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body><div class="report-container">{html_body}</div></body>
    </html>
    """
    return styled_html

@router.post("/generate")
async def start_generation_job(
    background_tasks: BackgroundTasks,
    report_service: ReportGenerationService = Depends(get_report_generation_service),
    company_name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
        
    job_id = str(uuid.uuid4())
    temp_dir = f"temp_{job_id}"
    os.makedirs(temp_dir, exist_ok=True)

    for file in files:
        file_path = os.path.join(temp_dir, file.filename)
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        finally:
            file.file.close()

    jobs[job_id] = {'status': 'processing', 'result': None}
    background_tasks.add_task(process_files_in_background, job_id, temp_dir, company_name, report_service)
    
    return JSONResponse(status_code=202, content={"job_id": job_id, "message": "Job accepted and is now processing."})

@router.get("/generate/status/{job_id}")
async def get_job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job['status'] == 'complete':
        return {"status": job['status'], "result": robust_json_serializer(job['result'])}
        
    return {"status": job['status'], "result": job.get('result')}

@router.post("/create-pdf")
async def create_pdf_from_markdown(markdown_content: str = Body(..., embed=True)):
    try:
        styled_html_content = render_report_with_style(markdown_content)
        pdf_bytes = HTML(string=styled_html_content).write_pdf()
        
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=sustainability_report.pdf"}
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create PDF with WeasyPrint: {str(e)}")