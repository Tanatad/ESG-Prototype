from fastapi import APIRouter, Depends, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from app.services.neo4j_service import Neo4jService
from app.schemas.graph import GraphRagRequest, GraphRagResponse
from typing import List, Tuple
from io import BytesIO
import time
import traceback
from app.dependencies import get_neo4j_service, get_question_generation_service
from app.services.question_generation_service import QuestionGenerationService
from app.models.esg_question_model import ESGQuestion

router = APIRouter()

@router.post("/uploadfile", summary="Upload files to build KG and evolve questions automatically.")
async def uploadfile(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="One or more PDF files to process into the Knowledge Graph."),
    neo4j_service: Neo4jService = Depends(get_neo4j_service),
    qg_service: QuestionGenerationService = Depends(get_question_generation_service)
):
    print(f"[CONTROLLER LOG /uploadfile] Received {len(files)} file(s).")
    start_req_time = time.time()

    if not files:
        return JSONResponse(content={"error": "No files received."}, status_code=400)

    file_data_for_neo4j: List[Tuple[str, BytesIO]] = []
    original_filenames_for_log: List[str] = []

    try:
        for i, file_upload in enumerate(files):
            original_filenames_for_log.append(file_upload.filename)
            content = await file_upload.read()
            processed_filename = f"files_{i}_{file_upload.filename}" 
            file_data_for_neo4j.append((processed_filename, BytesIO(content)))
            await file_upload.close()
    except Exception as e:
        return JSONResponse(content={"error": f"Error during file reading: {str(e)}"}, status_code=500)

    try:
        if file_data_for_neo4j:
            print(f"[CONTROLLER LOG /uploadfile] Calling neo4j_service.flow to ingest data...")
            streams_only = [fs_tuple[1] for fs_tuple in file_data_for_neo4j]
            names_only = [fs_tuple[0] for fs_tuple in file_data_for_neo4j]
            # --- START: ส่วนที่แก้ไข ---
            await neo4j_service.flow(streams_only, names_only) # ไม่ต้องรับค่าใส่ตัวแปร
            # --- END: ส่วนที่แก้ไข ---
            print("[CONTROLLER LOG /uploadfile] KG ingestion completed.")
    except Exception as e:
        return JSONResponse(content={"error": f"Error processing files for Knowledge Graph: {str(e)}"}, status_code=500)

    active_questions_count = await ESGQuestion.find(ESGQuestion.is_active == True).count()
    is_baseline_upload_automated = (active_questions_count == 0)
    
    print(f"[CONTROLLER LOG /uploadfile] Auto-detected baseline mode: {is_baseline_upload_automated} (Active questions in DB: {active_questions_count})")

    background_tasks.add_task(
        qg_service.evolve_and_store_questions,
        is_baseline_upload=is_baseline_upload_automated
    )

    print("[CONTROLLER LOG /uploadfile] Question AI evolution task added to background.")
    
    req_duration = time.time() - start_req_time
    print(f"[CONTROLLER LOG /uploadfile] Request successfully completed in {req_duration:.4f} seconds.")
    return JSONResponse(
        content={
            "status": "Success", 
            "message": f"Successfully processed {len(file_data_for_neo4j)} files. Question AI evolution (Auto-Detected Baseline={is_baseline_upload_automated}) triggered.",
            "original_filenames": original_filenames_for_log,
        }, 
        status_code=202
    )

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