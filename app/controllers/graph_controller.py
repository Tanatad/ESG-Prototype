from fastapi import APIRouter, Depends, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from app.services.neo4j_service import Neo4jService
from app.schemas.graph import GraphRagRequest, GraphRagResponse
from typing import List, Dict, Any
from io import BytesIO
import time
import traceback
import os
import uuid
import shutil
from app.dependencies import get_neo4j_service, get_question_generation_service
from app.services.question_generation_service import QuestionGenerationService
from app.models.esg_question_model import ESGQuestion
import io
router = APIRouter()

jobs: Dict[str, Dict[str, Any]] = {}

async def process_upload_and_evolve_in_background(job_id: str, temp_dir: str, is_baseline: bool, neo4j_service: Neo4jService, qg_service: QuestionGenerationService):
    """ฟังก์ชันใหม่สำหรับทำงานเบื้องหลัง"""
    try:
        file_streams = []
        file_names = []
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            file_streams.append(io.BytesIO(open(file_path, 'rb').read()))
            file_names.append(filename)

        processed_doc_ids = await neo4j_service.flow(files=file_streams, file_names=file_names)
        if not processed_doc_ids:
            raise Exception("Document ingestion failed.")

        comparison_result = await qg_service.evolve_and_store_questions(
            document_ids=processed_doc_ids, 
            is_baseline_upload=is_baseline
        )
        
        jobs[job_id]['status'] = 'complete'
        jobs[job_id]['result'] = comparison_result

    except Exception as e:
        traceback.print_exc()
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['result'] = str(e)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# แก้ไข endpoint เดิม
@router.post("/uploadfile")
async def start_upload_job(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    is_baseline: bool = False,
    neo4j_service: Neo4jService = Depends(get_neo4j_service),
    qg_service: QuestionGenerationService = Depends(get_question_generation_service)
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    job_id = str(uuid.uuid4())
    temp_dir = f"temp_dev_job_{job_id}"
    os.makedirs(temp_dir, exist_ok=True)

    for file in files:
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file.file.close()

    jobs[job_id] = {'status': 'processing', 'result': None}
    background_tasks.add_task(process_upload_and_evolve_in_background, job_id, temp_dir, is_baseline, neo4j_service, qg_service)
    
    return JSONResponse(status_code=202, content={"job_id": job_id, "message": "Question evolution job accepted."})

# เพิ่ม endpoint ใหม่สำหรับเช็คสถานะ
@router.get("/uploadfile/status/{job_id}")
async def get_upload_job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(content=job)
# --- END: ส่วนจัดการ Background Job ---

@router.post("/query", response_model=GraphRagResponse)
async def query(
    request: GraphRagRequest,
    neo4j_service: Neo4jService = Depends(get_neo4j_service)
):
    print(f"[CONTROLLER LOG /query] Received request with query: '{request.query}' and top_k: {request.top_k}")
    start_req_time = time.time()
    try:
        print("[CONTROLLER LOG /query] Calling neo4j_service.get_output...")
        retrieved_data = await neo4j_service.get_output(query=request.query, k=request.top_k)

        documents_content = []
        if hasattr(retrieved_data, 'relate_documents') and retrieved_data.relate_documents:
            documents_content = [doc.page_content for doc in retrieved_data.relate_documents if hasattr(doc, 'page_content')]
        
        if not documents_content:
            return GraphRagResponse(topKDocuments="")

        concatenated_content = "\n\n---\n\n".join(documents_content)

    except Exception as e:
        print(f"[CONTROLLER ERROR /query] Error in /query endpoint: {e}")
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

    req_duration = time.time() - start_req_time
    print(f"[CONTROLLER LOG /query] Request successfully completed in {req_duration:.4f} seconds.")
    return GraphRagResponse(topKDocuments=concatenated_content)